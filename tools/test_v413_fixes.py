#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v4.1.3 回归：极线残差方向、RMSE 外点污染、离群点毁高度场、填充率假门槛。

纯 numpy/标准库即可；端到端那组需要带 cv2 的 Python，找不到就跳过。

    python tools/test_v413_fixes.py
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

PASS = 0
SKIP = 0


def ok(label: str) -> None:
    global PASS
    PASS += 1
    print("PASS  %s" % label)


def skip(label: str) -> None:
    global SKIP
    SKIP += 1
    print("SKIP  %s" % label)


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_epipolar_source():
    src = read("pipeline/v4/sfm.py")
    assert "np.linalg.inv(cams[j].K).T @ E @ np.linalg.inv(cams[i].K)" in src, \
        "极线残差的内参矩阵顺序仍是反的"
    assert "K1^-T" in src and "K2^-T" in src, "缺少参数顺序说明"
    ok("极线残差：F = K2^-T E K1^-1，i/j 顺序已修正")


def test_rmse_uses_inliers():
    src = read("pipeline/v4/sfm.py")
    assert "matched(p.i, p.j)[inlier_mask]" in src, "重投影仍把 RANSAC 外点算进 RMSE"
    ok("重投影 RMSE：只用本质矩阵的 RANSAC 内点")


def test_robust_span():
    """少数三角化离群点不能把高度场范围拉大、把高度缩成近似 0。"""
    import numpy as np
    from v4 import fuse

    rng = np.random.default_rng(0)
    a = np.column_stack([rng.random(60) * 0.1, np.full(60, 10.0), rng.random(60) * 0.1])
    b = np.column_stack([0.8 + rng.random(60) * 0.1, np.full(60, 1.0),
                         0.8 + rng.random(60) * 0.1])
    # 1 个离群点（<2%）：旧版 min/max 会让它把 span 从 1 拉到 120
    bad = np.array([[60.0, 0.0, 60.0]])
    pts = np.vstack([a, b, bad]).astype(np.float32)
    cols = np.ones((len(pts), 3), np.float32)

    w = fuse.to_world(pts, cols, cells=24, smooth=0)
    assert w.extent[1] < 5, w.extent
    heights = w.pos[:, 1]
    assert 0.8 < float(heights.max()) < 12, float(heights.max())
    assert float(heights.min()) >= 0.0
    ok("高度场：离群点不再拉大 span；高度 10/1 保持（实测 %.2f/%.2f）"
       % (float(heights.max()), float(heights.min())))


def test_fill_thresholds():
    from types import SimpleNamespace
    import numpy as np
    from v4.qc import summarize

    def fake(fill):
        pairs = [SimpleNamespace(inliers=100, parallax=3.0) for _ in range(8)]
        return SimpleNamespace(localized=9, reproj_rmse=0.5, pairs=pairs,
                               pts=np.zeros((2000, 3), np.float32))

    assert summarize(fake(0), 9, fill=0.018)["verdict"] == "FAIL"
    assert summarize(fake(0), 9, fill=0.05)["verdict"] == "WARN"
    assert summarize(fake(0), 9, fill=0.103)["verdict"] == "PASS"
    ok("填充率门槛：0.018 FAIL / 0.05 WARN / 0.103 PASS（对照实测值）")


def _build_python():
    try:
        from serve_api import _find_build_python
        return _find_build_python()
    except Exception:
        return None


def test_end_to_end_pass():
    """生成一段有基线的走动素材，确认 PASS 真的能出 GLB。"""
    py = _build_python()
    if not py:
        skip("端到端出图（找不到带 cv2 的 Python）")
        return
    argv = py.split() if " " in py else [py]
    with tempfile.TemporaryDirectory(prefix="pw413_") as td:
        inp = os.path.join(td, "images")
        out = os.path.join(td, "world")
        gen = subprocess.run(argv + [str(ROOT / "tools" / "make_synthetic_v4.py"),
                                     "--out", inp],
                             cwd=str(ROOT), capture_output=True, text=True, timeout=120)
        assert gen.returncode == 0, gen.stderr
        build = subprocess.run(argv + [str(ROOT / "pipeline" / "v4" / "build.py"),
                                       "--input", inp, "--out", out],
                               cwd=str(ROOT), capture_output=True, text=True, timeout=300)
        assert build.returncode == 0, build.stdout + "\n" + build.stderr
        glb = os.path.join(out, "world-v4.glb")
        assert os.path.exists(glb), "PASS 了却没出 GLB"
        report = json.load(open(os.path.join(out, "qc.json"), encoding="utf-8"))
        assert report["verdict"] == "PASS", report
        assert report["fill_ratio"] >= 0.08, report
        b = open(glb, "rb").read()
        assert b[:4] == b"glTF", "GLB 头不对"
        json_len, _ = struct.unpack_from("<II", b, 12)
        meta = json.loads(b[20:20 + json_len])
        assert meta["meshes"] and meta["accessors"], "GLB 没有网格"
        ok("端到端：合成走动素材 PASS → GLB %d 字节 / %d 个 mesh"
           % (len(b), len(meta["meshes"])))


def test_original_sets_still_blocked():
    """好素材能出图，不能以放宽质量门为代价。"""
    py = _build_python()
    if not py:
        skip("坏素材仍被拦（找不到带 cv2 的 Python）")
        return
    argv = py.split() if " " in py else [py]
    with tempfile.TemporaryDirectory(prefix="pw413neg_") as td:
        for name in ("v4_terrain", "v4_walk"):
            r = subprocess.run(argv + [str(ROOT / "pipeline" / "v4" / "build.py"),
                                       "--input", str(ROOT / "testdata" / name),
                                       "--qc-only", "--out", os.path.join(td, name)],
                               cwd=str(ROOT), capture_output=True, text=True, timeout=180)
            assert r.returncode == 1, (name, r.stdout)
            report = json.load(open(os.path.join(td, name, "qc.json"), encoding="utf-8"))
            assert report["verdict"] == "FAIL", (name, report)
    ok("质量门没有放宽：纯旋转 / 远景素材依然 FAIL")


if __name__ == "__main__":
    test_epipolar_source()
    test_rmse_uses_inliers()
    test_robust_span()
    test_fill_thresholds()
    test_end_to_end_pass()
    test_original_sets_still_blocked()
    print("\n通过 %d 组，跳过 %d 组" % (PASS, SKIP))
    raise SystemExit(0 if SKIP == 0 else 0)
