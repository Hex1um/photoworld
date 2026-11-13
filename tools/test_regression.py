#!/usr/bin/env python3
"""无 GPU / 无 torch 的回归：训练数据层、尺度门、验收门、深度上限。

  python tools/test_regression.py

覆盖过去的事故：
  * `worldmodel/data/` 缺失导致训练入口 ModuleNotFoundError；
  * 自检 PSNR/SSIM 对全局尺度、相机内参完全不敏感（曾实测坐标 ×0.5
    仍是 99.0dB / SSIM 1.0），任何尺度改动都验收不到；
  * 双锚点声明 1500m、metric_depth 却硬截断 600m，46.1% 像素压成同一平面。
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import tarfile
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

import eval as ev  # noqa: E402
import exif  # noqa: E402
import geom  # noqa: E402
from worldmodel.data.dataset import (  # noqa: E402
    TarWindows, _stack, collate, split_dataset,
)


def test_dataset() -> None:
    w, h, n = 32, 24, 3
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(os.path.join(td, "shard-00000.tar"), "w") as tf:
            for s in range(n):
                key = "traj_0001_%06d" % s
                parts = {}
                for role in ["init", "target"] + ["h%d" % k for k in range(4)]:
                    buf = io.BytesIO()
                    Image.new("RGB", (w, h), (30 + s * 10, 60, 90)).save(buf, "JPEG")
                    parts[key + "." + role + ".jpg"] = buf.getvalue()
                parts[key + ".json"] = json.dumps({
                    "action": [0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.0],
                    "action_dsl": "forward",
                }).encode("utf-8")
                for name, data in parts.items():
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))

        ds = TarWindows(td, w, h, n_hist=4)
        assert len(ds) == n, "样本数不对：%d" % len(ds)
        tr, va = split_dataset(ds, 0.33, seed=0)
        assert len(tr) == n - 1 and len(va) == 1
        one = ds[0]
        assert one["init"].shape == (3, h, w)
        assert one["hist"].shape == (4, 3, h, w)
        assert one["target"].shape == (3, h, w)
        assert one["action"].shape == (7,)
        assert -1.001 <= float(one["init"].min()) and float(one["init"].max()) <= 1.001
        batch = [tr[i] for i in range(len(tr))]
        assert _stack(batch, "init").shape == (len(tr), 3, h, w)
        assert _stack(batch, "hist").shape == (len(tr), 4, 3, h, w)
        assert _stack(batch, "action").shape == (len(tr), 7)
        try:
            import torch  # noqa: F401
        except ImportError:
            print("PASS  P0  dataset 读取/切分/形状（torch 未安装，跳过 collate）")
        else:
            out = collate(batch)
            assert tuple(out["init"].shape) == (len(tr), 3, h, w)
            assert tuple(out["action"].shape) == (len(tr), 7)
            print("PASS  P0  dataset 读取/切分/collate（含 torch）")
        ds.close()


def _synthetic_mesh(cam):
    """地面严格满足 y=-eye 的合成场景，其余是固定远景。"""
    rays = cam.rays()
    ry = rays[..., 1]
    z = np.where(ry < -0.01, -1.7 / np.minimum(ry, -0.01), 60.0)
    z = np.clip(z, 0.5, 1800.0).astype(np.float32)
    return geom.build_mesh(z, cam, edge_ratio=1e6, max_tris=200000)


def test_eval_gates() -> None:
    w, h = 96, 64
    hfov = math.radians(62.0)
    vfov = exif.vfov_of(hfov, w, h)
    cam = geom.Camera(hfov, vfov, w, h)
    m = _synthetic_mesh(cam)
    tex = np.random.default_rng(0).random((h, w, 3), dtype=np.float32)
    ref = (tex * 255).astype(np.uint8)

    base = ev.report(m["pos"], m["uv"], tex, cam, ref, eye=1.7)
    half = ev.report(m["pos"] * 0.5, m["uv"], tex, cam, ref, eye=1.7)
    assert base["ok"] and base["scale"]["ok"], "正常场景不应被判失败：%s" % base
    assert abs(base["psnr"] - half["psnr"]) < 0.01, "泼点自检对尺度仍应不敏感（这是它的固有性质）"
    assert not half["ok"] and not half["scale"]["ok"], "尺度门没拦住整体 ×0.5：%s" % half
    print("PASS  P1  尺度门：坐标 ×0.5 时 PSNR 仍 %.1fdB，但尺度门 rel_error=%.2f → 判失败"
          % (half["psnr"], half["scale"]["rel_error"]))

    cam_bad = geom.Camera(hfov * 1.25, vfov * 1.25, w, h)
    mb = _synthetic_mesh(cam_bad)
    self_ok = ev.report(mb["pos"], mb["uv"], tex, cam_bad, ref)
    cross = ev.report(mb["pos"], mb["uv"], tex, cam_bad, ref, cam_verify=cam)
    assert self_ok["psnr"] > 30, "自洽自检应该仍是高分：%s" % self_ok
    assert cross["cross"]["ok"] is False and cross["ok"] is False, \
        "独立相机交叉门没拦住 FOV 错 25%%：%s" % cross
    print("PASS  P1  交叉门：FOV 错 25%% 自洽自检 %.1fdB → 独立相机 %.1fdB / 覆盖 %.2f → 判失败"
          % (self_ok["psnr"], cross["cross"]["psnr"], cross["cross"]["coverage"]))


def test_metric_depth() -> None:
    h, w = 64, 96
    hfov = math.radians(62.0)
    cam = geom.Camera(hfov, exif.vfov_of(hfov, w, h), w, h)
    yy = np.linspace(0, 1, h)[:, None]
    xx = np.linspace(0, 1, w)[None, :]
    d = np.clip(0.02 + 0.92 * yy + 0.02 * np.sin(xx * 12), 0, 1).astype(np.float32)
    cal = geom.calibrate_scale(d, cam, eye=1.7, horizon_far=1500.0)
    p = float(cal["p"])
    far = 1.2 * 1500.0
    r0 = float((cal["A"] / far) ** (1.0 / p))
    r_near = max(float(cal.get("r_near", 0.8)), 1e-3)
    a_eff = cal["A"] * (1.0 + r0 / r_near) ** p
    z_new = geom.metric_depth(d, a_eff, p=p, far=far, r0=r0)
    z_old = np.clip(cal["A"] * np.maximum(d, 1e-3) ** (-p), 0.5, 600.0)

    assert np.isfinite(z_new).all(), "软饱和仍出现 inf/nan"
    new_top = float((z_new >= far - 0.5).mean())
    old_top = float((z_old >= 599.9).mean())
    assert new_top < old_top, "触顶比例没下降：%.3f → %.3f" % (old_top, new_top)
    z_at_near = a_eff * (r_near + r0) ** (-p)
    assert abs(z_at_near - cal["z_near"]) / max(cal["z_near"], 1e-6) < 1e-5, \
        "A 修正后近端锚点不再精确"
    assert np.unique(np.round(z_new[d < 0.05], 3)).size > 5, "远景仍被压成同一个硬平面"
    print("PASS  P1  深度上限：far 600→%.0fm，触顶像素 %.1f%%→%.1f%%，"
          "近端锚点 %.3fm 保持不变，远景不再是单一平面"
          % (far, old_top * 100, new_top * 100, cal["z_near"]))


if __name__ == "__main__":
    test_dataset()
    test_eval_gates()
    test_metric_depth()
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import test_regression_p2  # noqa: E402
    test_regression_p2.main()
    print("\n全部回归通过（P0/P1 + P2）")
