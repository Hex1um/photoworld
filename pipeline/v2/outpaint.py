#!/usr/bin/env python3
"""视锥外扩：用扩散模型把照片"补"到更宽的视野，消除重建世界的硬边与空白。

为什么必须做这一步 —— 从实测截图看，v2 世界的三个显眼缺陷都来自同一件事：
**照片只有那一个梯形**，四周什么都没有。

  1. 视锥外一片灰白（画面左右下三块纯色）；
  2. 照片边界处一条刀切般的硬边（几何到此为止）；
  3. 近处地面被拉成长条 —— 因为它正好在照片边缘，UV 已经到边界，
     再往外没有像素可采样，只能把最后一行像素无限拉长。

做法：把照片放进更大的画布（左右各 +35%、下 +22%、上 +8%），
用 Stable Diffusion 2 Inpainting 生成新增区域，然后：
  * **中心原图区域严格保留原始像素**（合成时只取生成结果的边缘区域），
    保证"原视角看到的还是那张照片"，不被扩散模型改写；
  * 生成前用镜像/拉伸做初始化 —— 扩散模型在纯噪声的边界上更爱长出新东西，
    给一个连续的先验能显著减少接缝。

视野换算：像素密度保持不变，所以
    hfov_new = 2·atan( tan(hfov_old/2) · W_new/W_old )
即中心那块照片的角分辨率与原图完全一致，只有四周是新补的角度。
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageFilter

MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
_PIPE = None
_MODEL_DIR = None


def _model_dir(root: str) -> str:
    """优先用仓库内的 models/sd2_inpaint；否则回落到 HF 缓存里的快照目录。

    为什么不强制拷进仓库：SDXL-inpaint 是 6.5GB，拷一份纯属浪费磁盘。
    缓存目录里带着完整目录结构（unet/vae/text_encoder*），diffusers 能直接读。
    """
    import glob
    local = os.path.join(root, "models", "sd2_inpaint")
    if os.path.exists(os.path.join(local, "model_index.json")):
        return local
    cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1/snapshots")
    for snap in sorted(glob.glob(os.path.join(cache, "*")), reverse=True):
        if os.path.exists(os.path.join(snap, "model_index.json")):
            return snap
    return local


def available(model_dir: str) -> dict:
    """权重在不在、diffusers 装没装。缺任何一项就走边缘延展的降级路径。"""
    ok = True
    why = []
    try:
        import diffusers  # noqa: F401
        import torch     # noqa: F401
    except Exception as e:
        ok = False
        why.append("缺少 diffusers/torch: %s" % e)
    if not os.path.exists(os.path.join(model_dir, "model_index.json")):
        ok = False
        why.append("没找到 inpainting 权重（models/sd2_inpaint 或 HF 缓存，见 pipeline/v2/README）")
    return {"ok": ok, "reason": "; ".join(why)}


def _load(model_dir: str, device: str = "cuda"):
    """AutoPipeline 同时认 SD1.5 / SDXL 的 inpainting 结构。

    8GB 显存跑 SDXL-inpaint 会不够：先用整卡加载，OOM 再退回 CPU offload
    （速度慢一档，但能跑）。这条分支是实测需要 —— 不写就只能报 CUDA OOM。
    """
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    from diffusers import AutoPipelineForInpainting
    import torch
    dtype = torch.float16 if device == "cuda" else torch.float32
    # 只下了 fp16 变体（6.5GB vs 13GB），必须显式 variant="fp16"，
    # 否则 diffusers 会去找 model.safetensors 而报"no file named …"
    kw = dict(torch_dtype=dtype, local_files_only=True, use_safetensors=True,
              variant="fp16")
    pipe = AutoPipelineForInpainting.from_pretrained(model_dir, **kw)
    try:
        pipe = pipe.to(device)
    except Exception:
        # 8GB 装不下 SDXL 时退到 CPU offload（慢一点，但能跑）
        pipe.enable_model_cpu_offload()
    # 这两个优化在不同 diffusers 版本里方法名/可用性不一致（实测 SDXL 管线
    # 没有 enable_vae_slicing），逐个 try，绝不能因为"优化手段不存在"而整体失败
    for fn in ("enable_attention_slicing", "enable_vae_slicing"):
        try:
            getattr(pipe, fn)()
        except Exception:
            pass
    _PIPE = pipe
    return pipe


def _padding(w: int, h: int, sx: float, sy: float, bias_down: float = 0.72):
    """各边扩展像素数。bias_down>0.5 表示下方多补一些（地面比天空更难凭空长）。"""
    extra_w = int(round(w * (sx - 1.0)))
    left = extra_w // 2
    right = extra_w - left
    extra_h = int(round(h * (sy - 1.0)))
    bottom = int(round(extra_h * bias_down))
    top = extra_h - bottom
    return left, top, right, bottom


def _init_canvas(img: Image.Image, pad) -> tuple[Image.Image, Image.Image]:
    """扩展画布 + 初始化值（镜像边）+ 掩码（新增区域=255）。"""
    left, top, right, bottom = pad
    w, h = img.size
    canvas = Image.new("RGB", (w + left + right, h + top + bottom))
    # 四角与四边用镜像填充：比纯色好，扩散模型不用从零"发明"结构
    m = img.transpose(Image.FLIP_LEFT_RIGHT)
    v = img.transpose(Image.FLIP_TOP_BOTTOM)
    canvas.paste(img, (left, top))
    if left:  canvas.paste(img.crop((0, 0, min(left, w), h)).transpose(Image.FLIP_LEFT_RIGHT), (0, top))
    if right:
        cw = min(right, w)
        canvas.paste(img.crop((w - cw, 0, w, h)).transpose(Image.FLIP_LEFT_RIGHT), (left + w, top))
    if top:   canvas.paste(v.crop((0, 0, w, min(top, h))), (left, 0))
    if bottom:
        ch = min(bottom, h)
        canvas.paste(v.crop((0, h - ch, w, h)), (left, top + h))
    # 四角：用最近的边角拉伸铺满（细节不重要，反正要被重画）
    if left and top:      canvas.paste(img.crop((0, 0, 1, 1)).resize((left, top)), (0, 0))
    if right and top:     canvas.paste(img.crop((w - 1, 0, w, 1)).resize((right, top)), (left + w, 0))
    if left and bottom:   canvas.paste(img.crop((0, h - 1, 1, h)).resize((left, bottom)), (0, top + h))
    if right and bottom:  canvas.paste(img.crop((w - 1, h - 1, w, h)).resize((right, bottom)), (left + w, top + h))

    mask = Image.new("L", canvas.size, 255)
    mask.paste(0, (left, top, left + w, top + h))
    # 掩码边缘羽化：硬边会让扩散模型在接缝处画出一圈明显的"框"
    mask = mask.filter(ImageFilter.GaussianBlur(6))
    return canvas, mask


def outpaint(img: Image.Image, root: str, scale_x: float = 1.70, scale_y: float = 1.30,
             device: str = "cuda", use_model: bool = True, steps: int = 30,
             guidance: float = 7.5, seed: int = 0, gen_side: int = 960,
             prompt: str = "", force_fallback: bool = False) -> tuple[Image.Image, dict]:
    """返回 (扩展后的图, 元信息)。失败或没权重时退回镜像延展（依然比空白好）。"""
    w, h = img.size
    pad = _padding(w, h, scale_x, scale_y)
    left, top, right, bottom = pad
    Wext, Hext = w + left + right, h + top + bottom
    canvas, mask = _init_canvas(img, pad)

    info = {"pad": [left, top, right, bottom], "size": [Wext, Hext],
            "scale": [scale_x, scale_y], "mode": "fallback", "model": ""}

    mdir = _model_dir(root)
    if not force_fallback and use_model:
        av = available(mdir)
        if not av["ok"]:
            info["reason"] = av["reason"]
        else:
            try:
                import torch
                pipe = _load(mdir, device)
                # 生成分辨率：长边压到 gen_side 附近（SD2-inpaint 是 512 训练的，
                # 768~960 是"细节与结构"的折中；再大就会开始出现重复纹理）
                k = gen_side / float(max(Wext, Hext))
                gw, gh = max(512, int(round(Wext * k / 8)) * 8), max(512, int(round(Hext * k / 8)) * 8)
                c_small = canvas.resize((gw, gh), Image.LANCZOS)
                m_small = mask.resize((gw, gh), Image.LANCZOS)
                gen = torch.Generator(device=device).manual_seed(seed)
                p = prompt or ("a wide angle landscape photograph, seamless natural "
                               "continuation of the same scene, consistent lighting and colors, "
                               "high detail, photographic")
                n = ("blurry, seam, border, frame, watermark, text, logo, people, "
                     "distorted geometry, duplicate horizon")
                res = pipe(prompt=p, negative_prompt=n, image=c_small, mask_image=m_small,
                           height=gh, width=gw, num_inference_steps=steps,
                           guidance_scale=guidance, generator=gen).images[0]
                gen_full = res.resize((Wext, Hext), Image.LANCZOS)

                # 只取边缘：中心原图区域严格保留原始像素
                comp = gen_full.copy()
                comp.paste(img, (left, top))
                # 接缝 1~2px 用软过渡，避免"贴上去"的硬线
                band = 3
                if left or top or right or bottom:
                    alpha = Image.new("L", (w, h), 0)
                    alpha.paste(255, (band, band, w - band, h - band))
                    alpha = alpha.filter(ImageFilter.GaussianBlur(band))
                    comp.paste(img, (left, top), alpha)
                info.update({"mode": "sd-inpaint", "model": MODEL_ID,
                             "gen_size": [gw, gh], "steps": steps, "seed": seed})
                return comp, info
            except Exception as e:
                info["reason"] = "推理失败: %s" % e

    # ---- 降级：镜像延展 + 渐隐到雾色（没有模型时至少不留白，也不留刀切边） ----
    fog = np.asarray(img.resize((1, 1), Image.BOX), dtype=np.float32)[0, 0]
    arr = np.asarray(canvas, dtype=np.float32)
    yy, xx = np.mgrid[0:Hext, 0:Wext]
    # 到原图矩形的距离（归一化）
    dx = np.maximum(np.maximum(left - xx, xx - (left + w - 1)), 0) / max(1, max(left, right))
    dy = np.maximum(np.maximum(top - yy, yy - (top + h - 1)), 0) / max(1, max(top, bottom))
    d = np.clip(np.maximum(dx, dy), 0, 1)[..., None]
    arr = arr * (1 - 0.55 * d) + fog.reshape(1, 1, 3) * (0.55 * d)
    info["mode"] = "fallback-mirror"
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), info


def hfov_of_extended(hfov: float, w_src: int, w_ext: int) -> float:
    """像素密度不变时，扩展画布对应的水平视场。"""
    return 2.0 * math.atan(math.tan(hfov * 0.5) * (w_ext / float(w_src)))
