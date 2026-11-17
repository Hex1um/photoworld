"""度量深度 + 尺度标定：把单目相对视差变成"米"，并反解相机俯仰。

分工（与 v2 相同的判断，但输出口径不同）
----------------------------------------
* 深度网络只负责**相对**视差（大 = 近）。v2 的 `depth_gpu.estimate` 已经做了
  高分辨率推理 + guided-filter 上采样 + 分位归一化，这里直接复用，不重写。
* "米"由 `calib.robust_calibrate` 给：地面带物理锚定优先，不可信时退分位双锚点。
  它对 v2 的 LDI 已经验证过，v3 只换坐标系（+Z → -Z、俯仰符号），
  几何量（俯仰角、z 的数值）完全同源，因此标定结论可以直接搬过来。
* v3 新增：把标定后的相机俯仰写进 `Camera3`，**并要求前端把出生俯仰角设成它**。
  否则"从原点看"永远差一个角度 —— 这是 v2 里"站在原点看比原片暗/歪"的一类
  隐性来源（ISSUES #1 只查到曝光，俯仰那一半没被查）。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image

from . import common

_MODEL = None
_PROC = None


def _load(device: str = "cuda", long_side: int = 1540):
    global _MODEL, _PROC
    if _MODEL is None:
        import depth_gpu
        _PROC, _MODEL = depth_gpu.load_model(device=device)
        _MODEL._pw_long_side = long_side
    return _PROC, _MODEL


def unload() -> None:
    """把深度模型从显存里放掉。

    为什么需要：v3 一张图要过两个大模型（DA-V2-Large ≈1.3GB + SDXL unet ≈4.8GB）。
    笔记本 8GB 卡上两者常驻会只剩几百兆，SDXL 的 CPU-offload 就会来回搬权重 ——
    实测一张图的烘焙从 90s 拖到 **15 分钟以上**（GPU 99% 但全在等 PCIe）。
    深度用完立刻放，扩散要用的显存才够。
    """
    global _MODEL, _PROC
    _MODEL = None
    _PROC = None
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def disparity(img_rgb: np.ndarray, device: str = "cuda", long_side: int = 1540) -> np.ndarray:
    """(H,W,3) sRGB 0..1 → (H,W) 视差（大 = 近），与输入同分辨率。"""
    import depth_gpu
    proc, model = _load(device, long_side)
    pil = Image.fromarray((np.clip(img_rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    return depth_gpu.estimate(pil, proc, model, device=device, long_side=long_side)


def _two_anchor(disp: np.ndarray, cam: common.Camera3, horizon_far: float,
                near_anchor_m: float = 2.0) -> tuple[np.ndarray, dict]:
    """分位双锚点：最近的 1% 视差 ≈ 2m，最远的 2% ≈ 地平线先验。"""
    import geom
    r = disp[np.isfinite(disp)]
    r = r[r > 1e-4]
    r_near = float(np.percentile(r, 99.0))
    r_far = float(np.percentile(r, 2.0))
    r_near = max(r_near, r_far * 1.5 + 1e-3)
    p = math.log(horizon_far / near_anchor_m) / math.log(r_near / max(r_far, 1e-6))
    p = float(np.clip(p, 0.6, 2.6))
    A = near_anchor_m * (r_near ** p)
    far_m = 1.2 * horizon_far
    z = geom.metric_depth(disp, A, 0.0, near=0.5, far=far_m, p=p, r0=None)
    return z, {"mode": "two-anchor-percentile", "mode_used": "percentile-two-anchor(guard)",
               "A": A, "p": p, "r_near": r_near, "r_far": r_far, "far": far_m,
               "z_near": near_anchor_m, "z_far": horizon_far, "degenerate": False}


def calibrate(disp: np.ndarray, cam: common.Camera3, far_prior: float = 900.0) -> dict:
    """视差 → 米。返回 {"z", "cam"(俯仰可能被修正), "cal"}。"""
    import geom
    import calib as calibmod

    gcam = geom.Camera(hfov=cam.hfov, vfov=cam.vfov, w=cam.w, h=cam.h, pitch=cam.pitch)
    try:
        res = calibmod.robust_calibrate(disp, gcam, eye=cam.eye, horizon_far=far_prior)
        z = np.asarray(res["z"], dtype=np.float32)
        cam2 = common.Camera3(cam.w, cam.h, cam.hfov, cam.vfov,
                              pitch=float(res["cam"].pitch), eye=cam.eye)
        cal = dict(res["cal"])
        # 标定钳位是承重件（ISSUES #25：去掉它整片山会塌到地平面），
        # 但必须留痕：p 打到边界就说明"这张图的地面先验不成立"。
        p_used = float(cal.get("p", 1.0))
        cal["p_clamped"] = bool(abs(p_used - 3.2) < 1e-6 or abs(p_used - 0.6) < 1e-6)

        # ---- 合理性闸门：地面锚定有时会给出"连最近的像素都在 57m 外"的标定 ----
        # 手持照片的底部几乎总有 2~10m 内的地面。实测 desert 那张被地面锚定
        # 拉到 57~511m，几何整体退到远处：原点视角仍然像（颜色走 UV），
        # 但一走两步就看到"眼前一片空旷的平板"（multiview 的 lowVoid 16%）。
        z_near2 = float(np.percentile(z, 2))
        if z_near2 > 10.0 and cal.get("mode_used") == "ground-anchored":
            z, cal2 = _two_anchor(disp, cam2, far_prior)
            cal2["guard"] = "ground-anchored 给出 z_near=%.1fm，退分位双锚点" % z_near2
            cal2["p_clamped"] = False
            return {"z": np.clip(z, 0.4, far_prior * 1.6), "cam": cam2, "cal": cal2}
        return {"z": np.clip(z, 0.4, far_prior * 1.6), "cam": cam2, "cal": cal}
    except Exception as e:                                  # 极端构图/退化图
        z = 2.0 + 240.0 * np.power(1.0 - np.clip(disp, 0, 1), 1.7)
        return {"z": z.astype(np.float32), "cam": cam,
                "cal": {"mode_used": "fallback", "error": str(e)[:200]}}
