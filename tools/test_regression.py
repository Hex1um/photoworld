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
    h, w = 96, 160
    hfov = math.radians(62.0)
    cam = geom.Camera(hfov, exif.vfov_of(hfov, w, h), w, h)
    # v1.22（T2）：合成数据必须是**物理一致**的 z ∝ 1/r 场。
    #   旧版用 yy 的线性梯度当"视差"，它根本不是 1/r —— 三参数拟合会把 p 顶到上限，
    #   "近端锚点精确"这类断言就变成在验一个不成立的假设（实测 p=3.0、A·r^-p=154m
    #   而 z_near=5.15m）。这里按相机几何造地面 + 天空，z 由 -eye/ry 给出。
    rays = cam.rays()
    ry = rays[..., 1]
    ground = ry < -0.02
    z_true = np.full((h, w), 1500.0, np.float32)
    z_true[ground] = (-1.7 / np.minimum(ry, -0.004))[ground]
    d = np.clip(2.0 / np.maximum(z_true, 1e-3), 1e-4, 0.999).astype(np.float32)
    cal = geom.calibrate_scale(d, cam, eye=1.7, horizon_far=1500.0)
    p = float(cal["p"])
    far = 1.2 * 1500.0
    # v1.22（T2）：标定/应用统一走 geom.apply_calibration —— 测试不再手抄内联公式，
    # 否则生产改了式子、测试还在验旧式（这一版之前就是这样，一改 b 就炸）。
    z_new, cal2 = geom.apply_calibration(d, cal, far=far)
    r_near = max(float(cal2.get("r_near", 0.8)), 1e-3)
    b_fit = float(cal2.get("b", 0.0))
    r_near_eff = max(1e-3, r_near + b_fit)
    r0 = float(cal2["r0"])
    a_eff = float(cal2["a_effective"])
    z_old = np.clip(cal["A"] * np.maximum(d, 1e-3) ** (-p), 0.5, 600.0)

    assert np.isfinite(z_new).all(), "软饱和仍出现 inf/nan"
    new_top = float((z_new >= far - 0.5).mean())
    old_top = float((z_old >= 599.9).mean())
    assert new_top < old_top, "触顶比例没下降：%.3f → %.3f" % (old_top, new_top)
    z_at_near = a_eff * (r_near_eff + r0) ** (-p)
    assert abs(z_at_near - cal["z_near"]) / max(cal["z_near"], 1e-6) < 1e-5, \
        "A 修正后近端锚点不再精确"
    # 旧 r0=(A/far)^(1/p) 是在这里被抓到的：z(0)=4137m 而 far=1800m
    z_at_zero = a_eff * r0 ** (-p)
    assert abs(z_at_zero - far) / far < 1e-6, \
        "软饱和未收敛到 far：z(0)=%.1fm far=%.1fm" % (z_at_zero, far)
    assert np.unique(np.round(z_new[d < 0.05], 3)).size > 5, "远景仍被压成同一个硬平面"
    print("PASS  P1  深度上限：far 600→%.0fm，触顶像素 %.1f%%→%.1f%%，"
          "近端锚点 %.3fm 保持不变，远景不再是单一平面"
          % (far, old_top * 100, new_top * 100, cal["z_near"]))


def test_action_scale() -> None:
    """T6（v1.22）：推理侧 parse_actions 必须按训练侧的 action_scale 缩放。"""
    from worldmodel.action_codec import apply_action_scale, dsl_action
    import numpy as np
    base = np.asarray(dsl_action("forward"), dtype=np.float32)
    assert base.size == 7, "dsl_action 维度不是 7（%d）" % base.size
    s2 = apply_action_scale(base, 2.5)
    assert np.allclose(s2, base * 2.5, atol=1e-6), "标量 action_scale 未生效"
    v2 = apply_action_scale(base, [1, 2, 3, 4, 5, 6, 7])
    assert np.allclose(v2, base * np.arange(1, 8), atol=1e-6), "7 维 action_scale 未逐维生效"
    assert np.allclose(apply_action_scale(base, None), base), "action_scale=None 不应改动"
    try:
        apply_action_scale(base, [1, 2, 3])
        raise AssertionError("长度不匹配的 action_scale 应当报错")
    except ValueError:
        pass
    print("PASS  T6  action_scale：标量/7 维逐维相乘、None 不动、长度不符报错（纯 numpy，无需 torch）")


def test_ckpt_mark() -> None:
    """G3（v1.22）：旧 checkpoint 必须被判出来（不静默使用）。"""
    from worldmodel.ckpt_meta import checkpoint_warning
    assert checkpoint_warning({"data": {}}) is not None, "缺标记却没有警告"
    assert checkpoint_warning({"data": {"dit_slice_fix": True}}) is None, "新档被误报"
    msg = checkpoint_warning({"data": {"dit_slice_fix": False}}) or ""
    assert "dit_slice_fix" in msg, "警告里没有指出缺失的键"
    print("PASS  G3  旧 checkpoint 判定：缺标记/False → 警告，True → 放行")


def test_hist_frames() -> None:
    """T9-3（v1.22）：历史帧数以 checkpoint 配置为准，不一致要给出警告。"""
    from worldmodel.ckpt_meta import resolve_hist_frames
    n, w = resolve_hist_frames({"data": {"hist_frames": 6}}, 4)
    assert n == 6 and w, "应覆盖为 checkpoint 的 6 帧并警告"
    n2, w2 = resolve_hist_frames({"data": {"hist_frames": 4}}, 4)
    assert n2 == 4 and w2 is None, "一致时不应警告"
    n3, w3 = resolve_hist_frames({"data": {}}, 5)
    assert n3 == 5 and w3 is None, "无配置时沿用调用方取值"
    print("PASS  T9-3 hist_frames：以 checkpoint 为准（6 vs 4 → 覆盖并警告；一致/缺省不警告）")


def test_extend_clamp() -> None:
    """T9-5（v1.22）：视场延拓的深度不得超过 far。"""
    import numpy as np
    cam = geom.Camera(hfov=math.radians(60.0), vfov=math.radians(37.0), w=64, h=40)
    img = np.zeros((40, 64, 3), np.float32)
    dep = np.full((40, 64), 1750.0, np.float32)
    _, d2, _ = geom.extend_fov(img, dep, cam, target_hfov=math.radians(100.0), far=1800.0)
    assert float(d2.max()) <= 1800.0 + 1e-3, "延拓深度超过 far：%.1f" % float(d2.max())
    print("PASS  T9-5 延拓深度夹到 far：max %.1fm ≤ 1800m" % float(d2.max()))


def test_edge_step() -> None:
    """T9-6（v1.22）：structure tensor 梯度按 step 归一后，跳面比例不应随 step 暴涨。"""
    import numpy as np
    h, w = 120, 160
    hfov = math.radians(62.0)
    cam = geom.Camera(hfov, exif.vfov_of(hfov, w, h), w, h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # 左半 8m、右半 40m：一条**真断崖** + 高频纹理（最容易触发跳面判据的组合）。
    #   第一版没放断崖，跳面数恒为 0，断言等于空跑。
    z = np.where(xx < w // 2, 8.0, 40.0) + 0.3 * np.sin(xx * 0.9) * np.cos(yy * 0.7)
    z = z.astype(np.float32)
    m1 = geom.build_mesh(z, cam, max_tris=10 ** 9)          # step = 1
    m2 = geom.build_mesh(z, cam, max_tris=(h * w) // 2)     # step ≈ 2
    s1, s2 = float(m1["skipped"]), float(m2["skipped"])
    # 归一化正确时两者应在同一量级；旧实现 step=2 会把 edge_str 放大 2×、
    # 跳面比例显著上升。
    assert s2 < s1 * 1.6 + 0.05, ("跳面比例随 step 暴涨：%.3f（step1）→ %.3f（step2）" % (s1, s2))
    print("PASS  T9-6 梯度按 step 归一：跳面比例 %.3f（step1）→ %.3f（step2）" % (s1, s2))


def test_walkable() -> None:
    """T9-7（v1.22）：每格取中位；没有采样点的格子必须是 None（不是造一片平地）。"""
    import numpy as np
    # 同一格塞三个高度（含一个"树叶"离群值），另一片区域留空
    pos = np.array([
        [-1.0, -1.70, -10.0], [-0.9, -1.71, -10.1], [-1.1, -0.20, -9.9],   # 同格：中位应≈ -1.70
        [2.0, -1.60, -20.0],
    ], dtype=np.float32)
    w = geom.build_walkable(pos, span=60.0, gh=8, gw=8)
    filled = [v for row in w["height"] for v in row if v is not None]
    assert len(filled) == 2, "应只有 2 个格子有采样点，实际 %d" % len(filled)
    assert w["empty_cells"] == 8 * 8 - 2, "空格数不对：%d" % w["empty_cells"]
    assert -1.75 < filled[0] < -1.65 or -1.75 < filled[1] < -1.65, \
        "同格应取中位（≈-1.70），实际 %s" % filled
    assert not any(v == 0 for v in filled), "空格不应被填成 0/平地"
    print("PASS  T9-7 可走网格：每格中位 %s，空格 %d 个写 null" % (filled, w["empty_cells"]))


def test_normals() -> None:
    """T8（v1.22）：GLB 顶点法线必须都是单位向量，边界也必须有法线。"""
    import numpy as np
    h, w = 48, 64
    hfov = math.radians(62.0)
    cam = geom.Camera(hfov, exif.vfov_of(hfov, w, h), w, h)
    z = np.full((h, w), 12.0, np.float32)
    z[int(h * 0.6):] = 6.0
    mesh = geom.build_mesh(z, cam)
    nrm = mesh["nrm"] if "nrm" in mesh else mesh.get("normal")
    assert nrm is not None, "build_mesh 没有返回法线"
    ln = np.linalg.norm(np.asarray(nrm), axis=1)
    assert ln.size and abs(float(np.median(ln)) - 1.0) < 0.01, \
        "法线非单位长度（中位 %.3f）" % float(np.median(ln))
    assert float(ln.min()) > 0.99, "存在零/退化法线（min %.3f）" % float(ln.min())
    print("PASS  T8  法线：单位长度中位 %.3f / 最小 %.3f（边界已用单侧差分）"
          % (float(np.median(ln)), float(ln.min())))


if __name__ == "__main__":
    test_dataset()
    test_eval_gates()
    test_metric_depth()
    test_action_scale()
    test_ckpt_mark()
    test_hist_frames()
    test_extend_clamp()
    test_edge_step()
    test_walkable()
    test_normals()
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import test_regression_p2  # noqa: E402
    test_regression_p2.main()
    print("\n全部回归通过（P0/P1 + P2）")
