"""渐进式全景外推：按视角**依次**生成，后一个视角用前一个的成果做上下文。

为什么这么改（实测）：一次性给整张等距柱状图打遮罩，未知区占 96%，SD 只能凭空编 ——
出竖条 + 亮斑。而 SDXL-inpaint 的上下文只来自**未遮罩**的部分，所以关键是：
  1. 先用镜像初值把全景填成"有结构"的底图；
  2. 每次只对一个视角里**尚未生成**的区域打遮罩，已生成的部分作为上下文；
  3. 视角按"已知占比高的先做"排序，保证上下文一开始就是真照片；
  4. 最后整张做一次低强度精修，把视角接缝抹平。

这样全程都有上下文，不需要额外下载 img2img 权重（inpaint 管线本身就支持 strength）。
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage


def _view_dirs(yaw: float, hfov: float, vfov: float, w: int, h: int) -> np.ndarray:
    xs = (np.arange(w) + 0.5) / w * 2.0 - 1.0
    ys = 1.0 - (np.arange(h) + 0.5) / h * 2.0
    X, Y = np.meshgrid(xs * math.tan(hfov * 0.5), ys * math.tan(vfov * 0.5), indexing="xy")
    Z = -np.ones_like(X)
    n = np.sqrt(X * X + Y * Y + Z * Z)
    X, Y, Z = X / n, Y / n, Z / n
    c, s = math.cos(yaw), math.sin(yaw)
    Xr = X * c - Z * s
    Zr = X * s + Z * c
    return np.stack([Xr, Y, Zr], axis=-1).astype(np.float32)


def _dir_to_uv(d: np.ndarray, pw: int, ph: int) -> tuple[np.ndarray, np.ndarray]:
    u = (np.arctan2(d[..., 0], -d[..., 2]) / (2 * math.pi) + 0.5) * pw
    v = (0.5 - np.arcsin(np.clip(d[..., 1], -1, 1)) / math.pi) * ph
    return u, v


def _sample(pano: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack([ndimage.map_coordinates(pano[..., c], [v, u], order=1, mode="wrap")
                     for c in range(3)], axis=-1).astype(np.float32)



_PIPE_I2I = None


def _load_img2img(cfg: dict):
    """优先用 SDXL **img2img** 管线：它接受 strength，能让"参考图"真正起作用。

    为什么必须有它（实测）：inpaint 管线在遮罩外没有上下文时等于纯文生图，
    未知区 96% 的全景只能靠它编 —— 出来是竖条。img2img 不一样：strength 控制
    "保留多少原图"，我们就能拿上一视角的结果当底、只做 45% 的改写，接缝与结构
    都连贯。没有这份权重时回退到 inpaint（效果会差，但不会崩）。
    """
    global _PIPE_I2I
    if _PIPE_I2I is not None:
        return _PIPE_I2I
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline
    try:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16, variant="fp16", local_files_only=True,
            use_safetensors=True)
        torch.cuda.empty_cache()
        pipe.enable_model_cpu_offload()
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        _PIPE_I2I = (pipe, "img2img")
        print("    （用 SDXL img2img 管线）")
        return _PIPE_I2I
    except Exception as e:
        print("    （没有 img2img 权重，回退 inpaint：%s）" % e)
        _PIPE_I2I = (_p_pipe(), "inpaint")
        return _PIPE_I2I


def _p_pipe():
    from . import pano as _p
    return _p._load_pipe()


def build_multiview(rgb: np.ndarray, cam, pano0: np.ndarray, known: np.ndarray,
                    prompt: str, cfg: dict) -> np.ndarray:
    import torch

    from . import pano as _p

    pipe, kind = _load_img2img(cfg)
    h, w = pano0.shape[:2]
    n_views = int(cfg.get("pano_views", 6))
    vw = int(cfg.get("view_w", 896))
    vh = int(cfg.get("view_h", 576))
    hfov = float(cfg.get("view_hfov", math.radians(100)))
    vfov = float(cfg.get("view_vfov", math.radians(78)))
    steps = int(cfg.get("pano_steps", 30))
    cfg_scale = float(cfg.get("pano_cfg", 5.2))
    seed = int(cfg.get("seed", 0))

    full = _p.mirror_fill(pano0, known)
    covered = known.copy()

    views = []
    for i in range(n_views):
        yaw = 2 * math.pi * i / n_views
        dirs = _view_dirs(yaw, hfov, vfov, vw, vh)
        u, v = _dir_to_uv(dirs, w, h)
        uu = np.clip(u.astype(np.int64), 0, w - 1)
        vv = np.clip(v.astype(np.int64), 0, h - 1)
        views.append((float(known[vv, uu].mean()), yaw, dirs, uu, vv))
    views.sort(key=lambda t: -t[0])          # 已知多的先做

    for idx, (kfrac, yaw, dirs, uu, vv) in enumerate(views):
        need = ~covered[vv, uu]              # 该视角里还没生成的像素
        frac = float(need.mean())
        if frac < 0.03:
            print("    视角 %d/%d yaw=%.0f° 已覆盖，跳过"
                  % (idx + 1, n_views, math.degrees(yaw)))
            continue
        u, v = _dir_to_uv(dirs, w, h)
        img_view = np.clip(_sample(full, u, v), 0, 1).astype(np.float32)
        gen = torch.Generator(device="cpu").manual_seed(seed + idx)
        img_pil = Image.fromarray((img_view * 255 + 0.5).astype(np.uint8))
        neg = "text, watermark, blurry, distorted, seam, mirrored, duplicate"
        if kind == "img2img":
            # strength 越低越像原图；未知区多一些就多改一点
            st = float(np.clip(0.62 - 0.25 * kfrac, 0.34, 0.62))
            out = pipe(prompt=prompt, negative_prompt=neg, image=img_pil,
                       height=vh, width=vw,
                       num_inference_steps=max(12, int(steps * st)),
                       guidance_scale=cfg_scale, strength=st,
                       generator=gen).images[0]
        else:
            out = pipe(prompt=prompt, negative_prompt=neg, image=img_pil,
                       mask_image=Image.fromarray(np.where(need, 255, 0).astype(np.uint8)),
                       height=vh, width=vw, num_inference_steps=steps,
                       guidance_scale=cfg_scale, strength=1.0,
                       generator=gen).images[0]
        view = np.asarray(out, dtype=np.float32) / 255.0
        # 余弦羽化：视角中心权重 1，边缘 0，相邻视角自然过渡；
        # 已经"确定的照片像素"永远不被覆盖。
        cx = math.cos(yaw)
        wgt = (np.clip(dirs[..., 2] * (-cx) + dirs[..., 0] * math.sin(yaw), 0.0, 1.0) ** 1.2)
        wgt = wgt * (~known[vv, uu])
        acc = np.zeros((h, w, 3), dtype=np.float32)
        wsum = np.zeros((h, w, 1), dtype=np.float32)
        np.add.at(acc, (vv.ravel(), uu.ravel()), (view * wgt[..., None]).reshape(-1, 3))
        np.add.at(wsum, (vv.ravel(), uu.ravel()), wgt.reshape(-1, 1))
        sel = wsum[..., 0] > 1e-4
        blendw = np.clip(wsum / (wsum + 1.0), 0.0, 1.0)
        full = np.where(sel[..., None], full * (1 - blendw) + acc / np.maximum(wsum, 1e-5) * blendw, full)
        covered |= (wsum[..., 0] > 1e-4)
        print("    视角 %d/%d yaw=%.0f° 已知 %.0f%% mode=%s"
              % (idx + 1, n_views, math.degrees(yaw), kfrac * 100, kind))

    # 收尾：未生成区域做一次低强度统一，抹掉视角接缝
    try:
        miss = ~known
        gen = torch.Generator(device="cpu").manual_seed(seed + 999)
        out = pipe(prompt=prompt, negative_prompt="seam, banding, blurry",
                   image=Image.fromarray((np.clip(full, 0, 1) * 255 + 0.5).astype(np.uint8)),
                   mask_image=Image.fromarray(np.where(miss, 255, 0).astype(np.uint8)),
                   height=h, width=w, num_inference_steps=max(10, steps // 2),
                   guidance_scale=cfg_scale, strength=0.28, generator=gen).images[0]
        pol = np.asarray(out, dtype=np.float32) / 255.0
        full = np.where(miss[..., None], pol, full).astype(np.float32)
    except Exception as e:
        print("    （收尾精修跳过：%s）" % e)

    full = np.where(known[..., None], pano0, full).astype(np.float32)
    return full
