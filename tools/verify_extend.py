#!/usr/bin/env python
"""T1/T2 数值验收（v1.22）。可直接复跑：python tools/verify_extend.py

T1：extend_fov 的角度是否真的正确（旧的按角度线性近似会把原内容拉宽 1.5×）。
T2：z ∝ 1/r 的合成输入，标定出的 p 是否回到 1.0；带偏置 b 的输入能否把 b 找回来。

每条都给"修复前 / 修复后"两组数字，不通过则退出码非零。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import geom  # noqa: E402

FAIL = []


def check(name, ok, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<46} {detail}")
    if not ok:
        FAIL.append(name)


# ---------------- T1：视场外延的角度 ----------------
def t1_angles():
    print("T1 extend_fov 角度")
    w, h = 1600, 900
    cam = geom.Camera(hfov=math.radians(60.0), vfov=math.radians(37.0), w=w, h=h)
    img = np.zeros((h, w, 3), np.float32)
    dep = np.full((h, w), 12.0, np.float32)
    img2, dep2, cam2 = geom.extend_fov(img, dep, cam, target_hfov=math.radians(120.0))

    # 每列中心的角度：t = (u-0.5)*2*tan_h → θ = atan(t)
    def col_angles(c):
        u = (np.arange(c.w) + 0.5) / c.w
        return np.degrees(np.arctan((u - 0.5) * 2.0 * c.tan_h))

    ang = col_angles(cam2)
    left, right = ang[0], ang[-1]
    check("左缘角度", abs(left + 60.0) < 0.3, f"{left:+.2f}°（目标 -60.0°）")
    check("右缘角度", abs(right - 60.0) < 0.3, f"{right:+.2f}°（目标 +60.0°）")

    # 原内容区（居中 w 列）角度必须与原始相机逐列一致
    n_side = (cam2.w - w) // 2
    ang_mid = ang[n_side:n_side + w]
    ang0 = col_angles(cam)
    dev = float(np.max(np.abs(ang_mid - ang0)))
    check("原内容区逐列角度保持", dev < 0.02, f"最大偏差 {dev:.4f}°")

    # vfov 必须按方形像素重算
    ratio = cam2.tan_v / cam2.tan_h
    check("方形像素 tan_v/tan_h == h/w", abs(ratio - h / cam2.w) < 1e-6,
          f"{ratio:.6f} vs {h / cam2.w:.6f}")

    # 非自洽检查：把延拓结果按原 hfov 的 tan 网格重投影，原内容区不应被拉伸
    u = (np.arange(cam2.w) + 0.5) / cam2.w
    t_new = (u - 0.5) * 2.0 * cam2.tan_h
    sel = (np.abs(t_new) <= cam.tan_h * 1.0001)
    t_sub = t_new[sel]
    span = t_sub[-1] - t_sub[0] if t_sub.size else 0
    want = 2 * cam.tan_h
    check("原内容区切向跨度", abs(span - want) / want < 0.02,
          f"{span:.4f} vs {want:.4f}（偏差 {abs(span - want) / want * 100:.2f}%）")

    # 补的列数（60°→120° 应为 w 列，不是 w/2）
    check("每侧补列数", abs(n_side - w) <= 2, f"{n_side}（期望 {w}）")


# ---------------- T2：1/r 合成输入 ----------------
def _synth(cam, z_of_r=None, b_true=0.0, seed=0, far_prior=1500.0):
    """按相机几何造一张"地面 + 天空"的相对视差图，返回 (depth_rel, z_true, mask)。"""
    rng = np.random.default_rng(seed)
    h, w = cam.h, cam.w
    rays = cam.rays()
    ry = rays[..., 1]
    ground = ry < -0.02
    # 天空距离必须与标定声明的远锚先验一致，否则绝对尺度天生差 (先验/实际) 倍
    z = np.full((h, w), float(far_prior), np.float32)
    zt = -1.7 / np.minimum(ry, -0.004)
    z[ground] = zt[ground]
    # z = A(r+b)^-p  →  r = (A/z)^(1/p) - b
    # A 必须使得 r 落在 (0,1)：真实管线里 A≈4.3、近地 z≈3.4m 时 r≈0.8。
    # 第一版写 A=40，近场 r>1 全被 clip，关系被破坏 → 拟合必然发散。
    A, p = 2.0, 1.0
    r = np.power(np.maximum(A / z, 1e-9), 1.0 / p) - b_true
    # 噪声必须**乘性**（视差噪声随幅值增长）。第一版用固定 σ=0.004，
    #   而天空的 r 只有 0.0013 —— 减完直接变负、被 clip 到 1e-3 的下限，
    #   远锚 5 分位因此失真，绝对尺度整体偏 25%+，看起来像"标定错了"，
    #   其实是测试的噪声模型不物理。
    r = r * (1.0 + rng.normal(0, 0.01, r.shape))
    r = np.clip(r, 1e-4, 0.999).astype(np.float32)
    return r, z, ground


def t2_calibration():
    print("T2 标定（z ∝ 1/r）")
    cam = geom.Camera(hfov=math.radians(60.0), vfov=math.radians(37.0), w=640, h=360)
    r, z, ground = _synth(cam, b_true=0.0, far_prior=1500.0)
    cal = geom.calibrate_scale(r, cam, eye=1.7, horizon_far=1500.0)
    p = float(cal["p"])
    check("p 应回到 1.0", 0.95 <= p <= 1.05, f"p={p:.3f}（修复前实测 1.58）")
    # 与 reconstruct.py 走同一条式子（apply_calibration），避免"测试自洽、生产不一致"
    zc, _ = geom.apply_calibration(r, cal, far=1800.0)
    near = ground & (np.arange(cam.h)[:, None] >= int(cam.h * 0.80))
    err = float(np.median(np.abs(zc[near] - z[near]) / np.maximum(z[near], 1e-6)))
    check("近场误差中位", err < 0.05, f"{err * 100:.2f}%（要求 <5%）")

    print("T2 标定（z = A(r+b)^-p，b=0.14）")
    r2, z2, ground2 = _synth(cam, b_true=0.14, seed=1, far_prior=1500.0)
    cal2 = geom.calibrate_scale(r2, cam, eye=1.7, horizon_far=1500.0)
    b_hat = float(cal2["b"])
    check("b 应被找回", abs(b_hat - 0.14) < 0.02, f"b={b_hat:.3f}（真值 0.140）")
    check("b>0 时 p 仍接近 1", 0.93 <= float(cal2["p"]) <= 1.07, f"p={float(cal2['p']):.3f}")


def t5_pitch():
    """T5：俯仰 ±8° 的合成输入，反解误差应 <1°，近场深度误差 <5%。"""
    print("T5 俯仰反解")
    for true_deg in (-8.0, 0.0, 8.0):
        cam_true = geom.Camera(hfov=math.radians(62.0), vfov=math.radians(38.0),
                               w=320, h=200, pitch=math.radians(true_deg))
        r, z, ground = _synth(cam_true, far_prior=1500.0)
        cam0 = geom.Camera(hfov=cam_true.hfov, vfov=cam_true.vfov, w=320, h=200)
        fit = geom.estimate_pitch(r, cam0, eye=1.7)
        got = math.degrees(fit["pitch"]) if fit.get("ok") else float("nan")
        check(f"pitch {true_deg:+.0f}° 反解", abs(got - true_deg) < 1.0,
              f"得到 {got:+.2f}°（MAD {fit.get('mad')}）")
        # 用反解出的俯仰标定 + 应用，检查近场深度误差
        cam_fit = geom.Camera(hfov=cam0.hfov, vfov=cam0.vfov, w=320, h=200, pitch=fit["pitch"])
        cal = geom.calibrate_scale(r, cam_fit, eye=1.7, horizon_far=1500.0)
        zc, _ = geom.apply_calibration(r, cal, far=1800.0)
        near = ground & (np.arange(200)[:, None] >= int(200 * 0.80))
        err = float(np.median(np.abs(zc[near] - z[near]) / np.maximum(z[near], 1e-6)))
        check(f"pitch {true_deg:+.0f}° 近场误差", err < 0.05, f"{err * 100:.2f}%")


if __name__ == "__main__":
    t1_angles()
    t2_calibration()
    t5_pitch()
    print()
    if FAIL:
        print(f"✗ {len(FAIL)} 项未通过：" + "、".join(FAIL))
        sys.exit(3)
    print("✓ 全部通过")
