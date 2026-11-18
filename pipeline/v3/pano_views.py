"""多视角分段外推：把 360° 拆成 N 个透视视角分别生成，再拼回等距柱状图。

为什么：一次性对整张等距柱状图做 inpaint，未知区占 96%，SD 只能凭提示词瞎编 ——
出来是竖直拉丝 + 一块亮斑；而且等距柱状图在两极被极度压缩，模型在那里的输出必然是糊的。
拆成"每个 90° 左右的透视视角"之后：
  * 每个视角内的几何是正常的透视，SD 的训练分布与它一致；
  * 相邻视角有重叠，可以按重叠度做羽化拼接；
  * 之后再让 SD 在**整张**全景上做一次低强度精修，把接缝抹平。

代价：N 次推理。N=6 时约为单次的 6 倍 —— 所以必须在性能模式下跑。
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image
from scipy import ndimage


def _view_dirs(yaw: float, hfov: float, vfov: float, w: int, h: int) -> np.ndarray:
    """透视视角的逐像素方向（世界坐标，yaw 绕 Y，与管线约定一致）。"""
    xs = (np.arange(w) + 0.5) / w * 2.0 - 1.0
    ys = 1.0 - (np.arange(h) + 0.5) / h * 2.0
    X, Y = np.meshgrid(xs * math.tan(hfov * 0.5), ys * math.tan(vfov * 0.5), indexing="xy")
    Z = -np.ones_like(X)
    n = np.sqrt(X * X + Y * Y + Z * Z)
    X, Y, Z = X / n, Y / n, Z / n
    c, s = math.cos(yaw), math.sin(yaw)
    # 绕 Y 旋转（与 common.Camera3 的 yaw 约定一致：yaw>0 向右）
    Xr = X * c - Z * s
    Zr = X * s + Z * c
    return np.stack([Xr, Y, Zr], axis=-1).astype(np.float32)


def _dir_to_uv(d: np.ndarray, pw: int, ph: int) -> tuple[np.ndarray, np.ndarray]:
    u = (np.arctan2(d[..., 0], -d[..., 2]) / (2 * math.pi) + 0.5) * pw
    v = (0.5 - np.arcsin(np.clip(d[..., 1], -1, 1)) / math.pi) * ph
    return u, v


def build_multiview(rgb: np.ndarray, cam, pano0: np.ndarray, known: np.ndarray,
                    prompt: str, cfg: dict) -> np.ndarray:
    """N 个透视视角 → 等距柱状图。返回补全后的全景（已知区保持原像素）。"""
    import torch

    from . import pano as _p

    # 说明：这里用的是 SDXL-**inpaint** 管线，它没有 strength 参数；mask 全白时
    # 等价于纯文生图（参考图完全不起作用）。所以这条路径目前只能当"多视角独立生成"
    # 实验，不能当"参考图精修"。真要好用必须换 StableDiffusionXLImg2ImgPipeline。
    pipe = _p._load_pipe()
    h, w = pano0.shape[:2]
    n_views = int(cfg.get("pano_views", 6))
    vw = int(cfg.get("view_w", 896))
    vh = int(cfg.get("view_h", 576))
    hfov = float(cfg.get("view_hfov", math.radians(100)))
    vfov = float(cfg.get("view_vfov", math.radians(76)))
    steps = int(cfg.get("pano_steps", 26))
    cfg_scale = float(cfg.get("pano_cfg", 5.2))
    seed = int(cfg.get("seed", 0))

    acc = np.zeros((h, w, 3), dtype=np.float32)
    wsum = np.zeros((h, w, 1), dtype=np.float32)

    for i in range(n_views):
        yaw = 2 * math.pi * i / n_views
        dirs = _view_dirs(yaw, hfov, vfov, vw, vh)
        # 该视角在全景上的"参考底图"（镜像初值 + 已知区），作为 img2img 的 image
        u, v = _dir_to_uv(dirs, w, h)
        ref = np.stack([
            ndimage.map_coordinates(pano0[..., c], [v, u], order=1, mode="wrap")
            for c in range(3)], axis=-1).astype(np.float32)
        ref_img = Image.fromarray((np.clip(ref, 0, 1) * 255 + 0.5).astype(np.uint8))
        # 已知区在该视角内的占比决定强度：已知多 → 低强度（保留原图），否则高一些
        uu = np.clip(u.astype(np.int64), 0, w - 1)
        vv = np.clip(v.astype(np.int64), 0, h - 1)
        kfrac = float(known[vv, uu].mean())
        strength = float(np.clip(0.85 - 0.55 * kfrac, 0.30, 0.85))
        gen = torch.Generator(device="cpu").manual_seed(seed + i)
        out = pipe(prompt=prompt, negative_prompt="text, watermark, blurry, distorted, seam",
                   image=ref_img, mask_image=Image.new("L", (vw, vh), 255),
                   height=vh, width=vw, num_inference_steps=max(8, int(steps * strength)),
                   guidance_scale=cfg_scale, strength=strength,
                   generator=gen).images[0]
        view = np.asarray(out, dtype=np.float32) / 255.0

        # 拼回全景：按"该视角中心的方向"加权（边缘权重低，天然羽化）
        cx = math.cos(yaw)
        wgt = np.clip(dirs[..., 2] * (-cx) + dirs[..., 0] * math.sin(yaw), 0.0, 1.0) ** 1.5
        np.add.at(acc, (vv.ravel(), uu.ravel()), (view * wgt[..., None]).reshape(-1, 3))
        np.add.at(wsum, (vv.ravel(), uu.ravel()), wgt.reshape(-1, 1))
        print("    视角 %d/%d yaw=%.0f° 已知占比 %.0f%% strength=%.2f"
              % (i + 1, n_views, math.degrees(yaw), kfrac * 100, strength))

    full = acc / np.maximum(wsum, 1e-4)
    # 完全没被任何视角覆盖的极区：用最近有效行补齐
    miss = (wsum[..., 0] < 1e-3)
    if miss.any():
        rows = np.nonzero(~miss.all(axis=1))[0]
        if rows.size:
            fill = np.median(full[rows], axis=0, keepdims=True)
            full[miss] = np.broadcast_to(fill, full.shape)[miss]
    full = np.where(known[..., None], pano0, full).astype(np.float32)

    # 收尾：整张全景做一次低强度精修，把接缝抹平
    try:
        gen = torch.Generator(device="cpu").manual_seed(seed + 999)
        img = Image.fromarray((np.clip(full, 0, 1) * 255 + 0.5).astype(np.uint8))
        pol = pipe(prompt=prompt, negative_prompt="seam, banding, blurry",
                   image=img, mask_image=Image.new("L", (w, h), 255),
                   height=h, width=w, num_inference_steps=14,
                   guidance_scale=cfg_scale, strength=0.18, generator=gen).images[0]
        polished = np.asarray(pol, dtype=np.float32) / 255.0
        full = np.where(known[..., None], pano0, polished).astype(np.float32)
    except Exception as e:
        print("    （收尾精修跳过：%s）" % e)
    return full
