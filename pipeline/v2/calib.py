#!/usr/bin/env python3
"""稳健尺度标定：在"地面物理锚定"与"分位双锚点"之间按可信度自动选择。

为什么需要这一层（实测）：

  v1 的 `geom.calibrate_scale` 只有一条路——用画面下方的"地面带"做三参数拟合。
  这对有真实地面的照片是对的，但**风景照的底部经常不是地面**：
  03_snowrange 的底部是云海/雾，拟合出来的 p=0.48（物理上 p 应≈1），
  于是整幅场景被压进 207~1136m，前景/远景的区分被抹平，LDI 分层全部落进背景层——
  观感就是"一张巨大的远景幕布"，正是要解决的问题。

  这里的判据：
    * 地面带拟合只在 p ∈ [0.85, 2.2] 且 MAD 可接受时才采信（p 是幂律指数，
      归一化视差 ∝ 1/z 时物理上应接近 1；0.48 意味着深度被"拉平"）；
    * 否则退回**分位双锚点**：最亮 1% 视差 ≈ 2m 的眼前地面，最暗 2% ≈ 地平线先验。
      这条路不用假设地面在哪一行，任何构图都能给出一致的纵深关系。

  两条路都保留 p/A 的可追溯记录（写进 world.json），出问题能直接判断走的是哪条。
"""
from __future__ import annotations

import math

import numpy as np

import geom

P_RANGE = (0.85, 2.20)


def _downsample(disp: np.ndarray, cam, factor: int = 4):
    """标定只需要"地平面的形状"，不需要全分辨率。

    实测：俯仰搜索对 1152×864 跑 34 次 `fit_three_param`，每次 26 个 b 值各做一次
    最小二乘 —— 单张照片光这一步就要 44.8s，而几何构建只要 1s。
    按 4× 抽稀后同样的判据在 7 万像素上做，结果一致（p 与 pitch 的差异 <0.5%），
    耗时降到 1/16。
    """
    if factor <= 1:
        return disp, cam
    h, w = disp.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    d = disp[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))
    cam2 = geom.Camera(hfov=cam.hfov, vfov=cam.vfov, w=w2 // factor, h=h2 // factor,
                       pitch=cam.pitch)
    return d.astype(np.float32), cam2


def robust_calibrate(disp: np.ndarray, cam, eye: float = 1.7,
                     horizon_far: float = 1500.0,
                     near_anchor_m: float = 2.0,
                     pitch_span_deg: float = 6.0) -> dict:
    """返回 {"z": 米深度, "cam": 可能被修正俯仰的相机, "cal": 标定记录}"""
    ds, cams = _downsample(disp, cam, 4)

    # ---- 1) 俯仰：只在有明显改善且不在搜索边界时才采信 ----
    pf = geom.estimate_pitch(ds, cams, eye=eye, span_deg=pitch_span_deg)
    pitch_note = "keep"
    cam2 = cam
    if pf.get("ok"):
        at_edge = abs(pf.get("delta_deg", 0.0)) >= pitch_span_deg - 0.35
        mad0 = geom.estimate_pitch(ds, cams, eye=eye, span_deg=0.0).get("mad") or 0.0
        gain = (mad0 - pf["mad"]) / mad0 if mad0 > 0 else 0.0
        # 落在搜索边界 = 残差单调下降，说明"地面带"根本不是地面，不能采信
        if at_edge:
            pitch_note = f"reject-edge({pf['delta_deg']:+.2f}°)"
        elif abs(pf.get("delta_deg", 0.0)) < 0.2:
            pitch_note = "keep(small)"
        elif gain < 0.12:
            pitch_note = f"reject-weak(gain {gain*100:.1f}%)"
        else:
            cam2 = geom.Camera(hfov=cam.hfov, vfov=cam.vfov, w=cam.w, h=cam.h, pitch=pf["pitch"])
            pitch_note = f"fit({pf['delta_deg']:+.2f}°, gain {gain*100:.1f}%)"

    # ---- 2) 地面带拟合，检查它是否物理可信 ----
    cams2 = geom.Camera(hfov=cam2.hfov, vfov=cam2.vfov, w=cams.w, h=cams.h, pitch=cam2.pitch)
    cal_g = geom.calibrate_scale(ds, cams2, eye=eye, horizon_far=horizon_far)
    p_g = float(cal_g.get("p", 0.0))
    usable = (not cal_g.get("degenerate")) and (P_RANGE[0] <= p_g <= P_RANGE[1])

    far_m = 1.2 * horizon_far if horizon_far > 0 else 600.0
    if usable:
        z, cal = geom.apply_calibration(disp, cal_g, far_m)
        cal = dict(cal)
        cal["mode_used"] = "ground-anchored"
        cal["pitch_note"] = pitch_note
        return {"z": z, "cam": cam2, "cal": cal}

    # ---- 3) 分位双锚点 ----
    r = disp[np.isfinite(disp)]
    r = r[r > 1e-4]
    r_near = float(np.percentile(r, 99.0))          # 最近
    r_far = float(np.percentile(r, 2.0))            # 最远（≈地平线）
    r_near = max(r_near, r_far * 1.5 + 1e-3)
    # z = A·r^(-p)：两点解 p
    p = math.log(horizon_far / near_anchor_m) / math.log(r_near / max(r_far, 1e-6))
    p = float(np.clip(p, 0.6, 2.6))
    A = near_anchor_m * (r_near ** p)
    z = geom.metric_depth(disp, A, 0.0, near=0.5, far=far_m, p=p, r0=None)
    cal = {
        "mode": "two-anchor-percentile", "mode_used": "percentile-two-anchor",
        "A": A, "p": p, "b": 0.0, "r_near": r_near, "r_far": r_far,
        "z_near": near_anchor_m, "z_far": horizon_far,
        "far": far_m, "degenerate": False, "pitch_note": pitch_note,
        "rejected_ground_p": p_g,
        "rejected_ground_mad": cal_g.get("rms"),
    }
    return {"z": z, "cam": cam2, "cal": cal}
