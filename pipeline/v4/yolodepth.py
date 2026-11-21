"""Ultralytics YOLO26 米制深度：逐像素**绝对深度（米）**。

为什么值得接（与 Depth-Anything 对比）：
  * Depth-Anything 输出的是**相对视差**，尺度要靠别的信息猜（我们之前用 SfM 点拟合
    `1/z = a·d + b`，两参数，容易退化）；
  * `yolo26*-depth.pt` 直接给**米**，输出头是 `exp(logit)` 无界，且权重内置了全局尺度
    校准（官方文档：室内+室外混合 219 万张训练，NYU v2 δ1=0.882、RMSE=0.414m）；
  * 鼻尖上的好处：我们拟合只剩**一个尺度参数**（COLMAP 的任意单位 ↔ 米），
    世界也能顺势标定成米制。

实测（lego 桌面场景 800×800）：输出 1.04~2.54m，主体中位 1.17m、背景 1.66m，
遮挡边界干净（见 `out/v4/yolo_depth_vis.jpg`）。

注意：权重需手动下载（本机代理会让 ultralytics 自动下载报 SSL 证书错误）：
    python tools/hf_fetch.py <assets 里的 yolo26n-depth.pt> tools/weights/yolo26n-depth.pt
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
_MODEL = None
_WEIGHT = os.environ.get("YOLO_DEPTH_WEIGHT", str(ROOT / "tools" / "weights" / "yolo26n-depth.pt"))


def available() -> bool:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return Path(_WEIGHT).exists()


def _load():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        if not Path(_WEIGHT).exists():
            raise SystemExit("缺少深度权重：%s（用 tools/hf_fetch.py 手动下载）" % _WEIGHT)
        _MODEL = YOLO(_WEIGHT)
    return _MODEL


def depth_m(rgb: np.ndarray) -> np.ndarray:
    """RGB(H,W,3 uint8) → 深度(H,W) float32，单位米。分辨率与输入对齐。"""
    m = _load()
    r = m(rgb, verbose=False)[0]
    d = r.depth.data.cpu().numpy().astype(np.float32)
    if d.shape[:2] != rgb.shape[:2]:
        import cv2
        d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.clip(d, 1e-3, 1e4)
