"""质量验收：把重建结果"泼点"回渲染到原相机，和输入照片比对。

纯 numpy 向量化，不需要 GPU / 渲染器。任何尺度、视场、等比处理上的错误
都会直接反映成 PSNR 下降，所以它能当作产线的自检门。
"""
from __future__ import annotations

import math

import numpy as np


def splat_render(pos: np.ndarray, uv: np.ndarray, tex: np.ndarray,
                 cam, out_hw: tuple[int, int] | None = None) -> np.ndarray:
    """正交投影到原相机（z 已知，直接用相机模型反投影），返回渲染图。"""
    h, w = out_hw if out_hw else (tex.shape[0], tex.shape[1])
    fwd = np.maximum(-pos[:, 2], 1e-6)          # glTF 约定：视线朝 -Z
    xs = (pos[:, 0] / fwd) / cam.tan_h * 0.5 + 0.5
    ys = 0.5 - (pos[:, 1] / fwd) / cam.tan_v * 0.5
    px = np.clip((xs * w).astype(np.int32), 0, w - 1)
    py = np.clip((ys * h).astype(np.int32), 0, h - 1)

    order = np.argsort(-fwd)               # 远的先画
    px, py, z = px[order], py[order], fwd[order]

    th, tw = tex.shape[:2]
    tu = np.clip((uv[order, 0] * tw).astype(np.int32), 0, tw - 1)
    tv = np.clip((uv[order, 1] * th).astype(np.int32), 0, th - 1)
    col = tex[tv, tu]

    img = np.zeros((h, w, 3), dtype=np.float32)
    zbuf = np.full((h, w), np.inf, dtype=np.float32)
    # 从远到近覆盖
    for i in range(px.size):
        yy, xx = py[i], px[i]
        if z[i] < zbuf[yy, xx]:
            zbuf[yy, xx] = z[i]
            img[yy, xx] = col[i]
    return img


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """简化版 SSIM（灰度、11x11 均值滤波）。"""
    if a.ndim == 3:
        a = a.mean(axis=2)
    if b.ndim == 3:
        b = b.mean(axis=2)
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    def blur(x):
        k = 11
        pad = k // 2
        xp = np.pad(x, pad, mode="edge")
        cs = np.cumsum(np.cumsum(xp, axis=0), axis=1)
        cs = np.pad(cs, ((1, 0), (1, 0)))
        s = cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]
        return s / (k * k)

    mu1, mu2 = blur(a), blur(b)
    s1 = blur(a * a) - mu1 * mu1
    s2 = blur(b * b) - mu2 * mu2
    s12 = blur(a * b) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2)
    return float(np.mean(num / np.maximum(den, 1e-8)))


def report(pos, uv, tex, cam, ref, coverage_min=0.55, step=1) -> dict:
    """返回 PSNR/SSIM + 覆盖率 + 尺度诊断。覆盖率过低说明网格有洞或相机参数错。"""
    ref_f = ref.astype(np.float32) / 255.0 if ref.dtype == np.uint8 else ref.astype(np.float32)
    if step > 1:
        # 与网格采样点严格对齐：网格每隔 step 像素取一个顶点
        ref_f = ref_f[::step, ::step]
    h, w = ref_f.shape[:2]
    img = splat_render(pos, uv, tex, cam, (h, w))
    cov = float(np.mean(np.any(img > 1e-6, axis=2)))
    # 只在前景覆盖处算指标，避免黑洞拉低分数造成误判
    mask = np.any(img > 1e-6, axis=2)
    if mask.sum() < 100:
        return {"coverage": cov, "psnr": 0.0, "ssim": 0.0, "ok": False,
                "reason": "coverage too low"}
    p = psnr(img[mask], ref_f[mask])
    s = ssim(img * mask[..., None], ref_f * mask[..., None])
    return {
        "coverage": round(cov, 4),
        "psnr": round(p, 2),
        "ssim": round(s, 4),
        "ok": bool(cov >= coverage_min and p >= 14.0),
        "reason": "" if cov >= coverage_min else "coverage below %.2f" % coverage_min,
    }
