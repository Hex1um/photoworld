#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0/P1 修复的单步回归：紧凑 GLB、断崖边缘取色、地平线对齐、多视角偏差。"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "pipeline"), os.path.join(ROOT, "pipeline", "v3")):
    if p not in sys.path:
        sys.path.insert(0, p)

import glb as glbmod                      # noqa: E402
from v3 import frustum, glbread, pano, terrain  # noqa: E402

FAIL = []
PASS = []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail else ""))


def synthetic_layer(n=64):
    t = np.linspace(0, 1, n, dtype=np.float32)
    pos = np.stack([t, np.sin(t * 4), -t * 3], axis=1).astype(np.float32)
    nrm = np.stack([np.zeros(n, np.float32), np.ones(n, np.float32),
                    np.zeros(n, np.float32)], axis=1).astype(np.float32)
    uv = np.stack([t, 1.0 - t], axis=1).astype(np.float32)
    color = np.stack([t, 1 - t, np.full(n, 0.5, np.float32), np.ones(n, np.float32)],
                     axis=1).astype(np.float32)
    index = np.array([[i, i + 1, i + 2] for i in range(n - 2)], dtype=np.uint32).ravel()
    return {"name": "test", "pos": pos, "nrm": nrm, "uv": uv,
            "color": color, "index": index}


def test_compact_glb():
    layer = synthetic_layer()
    with tempfile.TemporaryDirectory() as td:
        p_full = os.path.join(td, "full.glb")
        p_compact = os.path.join(td, "compact.glb")
        tex = b"\xff\xd8\xff" + b"0" * 12000
        a = glbmod.write_glb(p_full, [layer], tex, "full", compact=False)
        b = glbmod.write_glb(p_compact, [layer], tex, "compact", compact=True)
        ga, gb = glbread.load(p_full), glbread.load(p_compact)
        la, lb = ga["layers"][0], gb["layers"][0]
        check("compact GLB 体积更小", b["bytes"] < a["bytes"],
              "%d -> %d bytes" % (a["bytes"], b["bytes"]))
        check("compact GLB 顶点位置逐位一致",
              np.array_equal(la["pos"], lb["pos"]))
        check("compact GLB 索引一致",
              np.array_equal(la["index"], lb["index"]))
        check("compact GLB UV 误差 < 2e-5",
              float(np.abs(la["uv"] - lb["uv"]).max()) < 2e-5,
              "max=%.2e" % float(np.abs(la["uv"] - lb["uv"]).max()))
        check("compact GLB 顶点色误差 < 4e-3",
              float(np.abs(la["color"] - lb["color"]).max()) < 4e-3)


def test_edge_fix():
    h = w = 4
    eye = np.zeros(3, dtype=np.float32)
    dirs = np.zeros((h, w, 3), dtype=np.float32)
    dirs[..., 2] = -1.0
    z = np.ones((h, w), dtype=np.float32)
    z[0, :] = 10.0
    pos = eye + dirs * z[..., None]
    uv = np.stack(np.meshgrid((np.arange(w) + 0.5) / w,
                              (np.arange(h) + 0.5) / h, indexing="xy"), axis=-1).astype(np.float32)
    safe = np.ones((h, w), dtype=bool)
    safe[0, :] = False
    safe_uv = uv.copy()
    safe_uv[0, :, 1] = uv[1, :, 1]
    m = frustum.grid_mesh(pos, np.ones((h, w), bool), uv, dirs, eye,
                          cliff=1.8, skirt=0.0, safe_mask=safe, safe_uv=safe_uv)
    check("断崖角污染计数 = 6", m.get("edge_fixed") == 6, str(m.get("edge_fixed")))
    base = h * w
    added = m["uv"][base:]
    expected = float(uv[1, 0, 1])
    check("断崖角 UV 改用最近前景内部像素",
          bool(np.allclose(added[:, 1], expected, atol=1e-6)),
          "v=%.4f" % float(added[:, 1].mean()))


def test_horizon_align():
    h, w = 256, 512
    target = h // 2
    shifted = 10
    yy = np.arange(h)[:, None]
    img = np.zeros((h, w, 3), np.float32)
    # 天空亮、地面暗；生成区地平线整体下移 shifted 行
    known = np.zeros((h, w), bool)
    known[:, w // 2 - 24:w // 2 + 24] = True
    for c in range(w):
        row = target + (0 if known[:, c].any() else shifted)
        img[:, c, :] = np.where(yy < row, 0.85, 0.18).astype(np.float32)
    out, shift = pano.align_horizon(img, known, max_shift_frac=0.06)
    gen_cols = ~known[0]
    peak_before = np.array([target + np.argmax(np.abs(np.diff(img[:, c, 0]))) for c in np.where(gen_cols)[0]])
    peak_after = np.array([np.argmax(np.abs(np.diff(out[:, c, 0]))) for c in np.where(gen_cols)[0]])
    check("地平线检测位移符号正确", bool(np.median(shift[gen_cols]) < -5),
          "median=%.1f" % float(np.median(shift[gen_cols])))
    check("对齐后生成区地平线回到目标行 ±2px",
          bool(np.abs(np.median(peak_after) - target) <= 2),
          "before=%d after=%d target=%d" % (int(np.median(peak_before)),
                                            int(np.median(peak_after)), target))
    check("已知区逐像素不变", bool(np.array_equal(out[:, known.any(axis=0), :],
                                               img[:, known.any(axis=0), :])))


def test_view_reconcile():
    rng = np.random.default_rng(0)
    n = 500
    r = rng.uniform(35, 80, n)
    az = rng.uniform(0, 2 * np.pi, n)
    base = np.stack([r * np.sin(az), -0.4 * np.ones(n), -r * np.cos(az)], axis=1)
    p = np.concatenate([base, base.copy()], axis=0)
    p[n:, 1] += 3.0
    yaw = np.concatenate([np.zeros(n), np.ones(n)], axis=0)
    out, bias = terrain.reconcile_view_bias(p, yaw, 1.7)
    m0 = float(np.median(out[:n, 1]))
    m1 = float(np.median(out[n:, 1]))
    check("8 视角高度偏差已消除", abs(m0 - m1) < 0.2,
          "med=%.3f/%.3f bias=%s" % (m0, m1, bias))


def test_js_routing():
    src = open(os.path.join(ROOT, "js", "app.js"), encoding="utf-8").read()
    check("?pipe=v3 路由存在", "q.get('pipe') === 'v3'" in src)
    check("v3 资产标记写入 dataset.pipe", "dataset.pipe = 'v3'" in src)
    check("v3 GLB 缺失时回退 v2", "回退 v2 在线路径" in src)


if __name__ == "__main__":
    test_compact_glb()
    test_edge_fix()
    test_horizon_align()
    test_view_reconcile()
    test_js_routing()
    print("\n%d 通过，%d 失败" % (len(PASS), len(FAIL)))
    if FAIL:
        print("失败项：" + ", ".join(FAIL))
    raise SystemExit(0 if not FAIL else 1)
