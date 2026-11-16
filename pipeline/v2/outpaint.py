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

# 优先用 SD1.5 系（Lykon/dreamshaper-8-inpainting，约 2.6GB）：8GB 笔记本显存上
# 单张 SDXL-inpaint 15 分钟都跑不完（实测），SD1.5 系几秒钟，画质对"补四周风景"够用。
MODEL_ID = "Lykon/dreamshaper-8-inpainting"
# 备选（更大更慢；显存 >=16GB 再考虑）
FALLBACK_IDS = ["diffusers/stable-diffusion-xl-1.0-inpainting-0.1"]
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
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    for mid in [MODEL_ID] + FALLBACK_IDS:
        cache = os.path.join(hub, "models--" + mid.replace("/", "--"), "snapshots")
        for snap in sorted(glob.glob(os.path.join(cache, "*")), reverse=True):
            unet = os.path.join(snap, "unet")
            # 必须确认 unet 权重真的在（下载中断时会只剩 config.json）
            if os.path.exists(os.path.join(snap, "model_index.json")) and \
               any(f.endswith(".safetensors") for f in os.listdir(unet)) if os.path.isdir(unet) else False:
                return snap
    return local


def free_vram_gb(device: str = "cuda") -> float:
    """当前可用显存（GB）。查不到时返回 -1 表示"未知，按可用来处理"。"""
    if device != "cuda":
        return -1.0
    # ① torch 的 mem_get_info（新版本才有）
    try:
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / (1024 ** 3)
    except Exception:
        pass
    # ② nvidia-smi 兜底 —— 这台机器的 torch 就没有 mem_get_info，
    #    早先返回 -1 被当成"未知→随便用"，于是守卫形同虚设，加载卡死 400s 无输出。
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=8).decode("ascii", "ignore")
        vals = [float(x) for x in out.split() if x.strip()]
        if vals:
            return max(vals) / 1024.0
    except Exception:
        pass
    return -1.0


def available(model_dir: str, device: str = "cuda", need_gb: float = 3.2) -> dict:
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
    # 显存守卫：模型加载要 ~2.5GB，若卡上已被别的程序占满，加载会**长时间卡住**
    # （不是报错，是一直在换页 —— 实测 400s 不出结果）。宁可立刻退回镜像延展。
    fv = free_vram_gb(device)
    if ok and fv >= 0 and fv < need_gb:
        ok = False
        why.append("显存不足：可用 %.1fGB < 需要 %.1fGB（关掉些占用显存的程序，或关掉模型外扩）"
                   % (fv, need_gb))
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
        av = available(mdir, device)
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

                # ---- 1) 色调对齐 --------------------------------------
                # 扩散模型生成区的曝光/白平衡和原片常差一截，直接拼上去会看到
                # 一圈明显的矩形硬边（实测：左半偏暗、底部偏亮）。
                # 用生成区与原始区的均值/标准差做一次 Reinhard 迁移，
                # 把生成区拉到原片的色彩统计上。
                ga = np.asarray(gen_full, dtype=np.float32)
                oa = np.asarray(img, dtype=np.float32)
                gmask = np.asarray(mask.resize((Wext, Hext), Image.LANCZOS), dtype=np.float32) > 127
                if gmask.sum() > 1000:
                    for c in range(3):
                        gv = ga[..., c][gmask]
                        ov = oa[..., c]
                        gs = max(1e-3, float(gv.std()))
                        os_ = max(1e-3, float(ov.std()))
                        ga[..., c] = (ga[..., c] - gv.mean()) * (os_ / gs) * 0.85 + ov.mean() + (gv.mean() - ov.mean()) * 0.15
                gen_full = Image.fromarray(np.clip(ga, 0, 255).astype(np.uint8))

                # ---- 有效性判定：这次到底有没有"生成" --------------
                # 扩散模型偶尔会基本不动（输出≈镜像初始值），此时两侧是原图的镜像
                # 复制，看起来像"同一张照片贴了三遍"（实测灯柱那张）。用"输出与初始
                # 画布在掩码区的平均绝对差"判定：太小就认为没生成，直接放弃外扩。
                init_arr = np.asarray(canvas, dtype=np.float32)
                diff = float(np.abs(np.asarray(gen_full, dtype=np.float32)[gmask]
                                    - init_arr[gmask]).mean()) if gmask.any() else 0.0
                # 更直接的判据：**左右两条生成带是不是互为镜像**。
                # 扩散模型有时几乎原样保留镜像初始值，于是整幅看起来像"同一张照片
                # 贴了三遍"（实测校园灯柱那张）。把左带翻转后与右带算相关系数，
                # 接近 1 就说明没生成出东西。
                corr = 0.0
                if left > 8 and right > 8:
                    Lb = np.asarray(gen_full.crop((0, top, left, top + h))
                                    .convert("L").resize((64, 64)), dtype=np.float32).ravel()
                    Rb = np.asarray(gen_full.crop((left + w, top, Wext, top + h))
                                    .convert("L").resize((64, 64)), dtype=np.float32)
                    Rb = np.fliplr(Rb).ravel()
                    if Lb.std() > 1e-3 and Rb.std() > 1e-3:
                        corr = float(np.corrcoef(Lb, Rb)[0, 1])
                info["mirror_corr"] = round(corr, 3)
                info["gen_diff"] = round(diff, 1)
                if diff < 4.0 or corr > 0.90:
                    info["reason"] = ("生成无效（镜像相关 %.2f / 平均差 %.1f）" % (corr, diff))
                    raise RuntimeError(info["reason"])

                # ---- 2) 宽羽化合成 ------------------------------------
                # 中心保留原始像素，但从边界往里 feather 像素做线性过渡，
                # 3px 的羽化挡不住色调差（实测就是一条硬线）。
                feather = max(12, int(min(w, h) * 0.05))
                comp = gen_full.copy()
                alpha = Image.new("L", (w, h), 0)
                alpha.paste(255, (feather, feather, max(feather + 1, w - feather),
                                  max(feather + 1, h - feather)))
                alpha = alpha.filter(ImageFilter.GaussianBlur(feather * 0.6))
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


def outpaint_two_pass(img: Image.Image, root: str, scale_x: float = 1.70,
                      scale_y: float = 1.30, **kw) -> tuple[Image.Image, dict]:
    """分两趟扩：先左右、再上下。

    为什么要两趟：一次生成 1.7×1.3 的画布时，长边压到 1280 后高度只剩 600 左右，
    底部那条"又宽又扁"的条带会让模型横向**重复平铺**（实测底边出现 4 段重复的岩壁）。
    拆成两趟后每趟的宽高比都接近模型熟悉的 1:1~16:9，重复消失，
    而且第二趟是拿"已扩好的图"当输入，上下方向也能接上第一趟生成的岩壁。
    """
    info = {}
    cur = img
    if scale_x > 1.01:
        cur, i1 = outpaint(cur, root, scale_x=scale_x, scale_y=1.0, **kw)
        info["pass1"] = i1
    if scale_y > 1.01:
        cur, i2 = outpaint(cur, root, scale_x=1.0, scale_y=scale_y, **kw)
        info["pass2"] = i2
    if not info:
        return img, {"mode": "off"}
    modes = {v.get("mode") for v in info.values()}
    info["mode"] = "sd-inpaint-2pass" if modes == {"sd-inpaint"} else "fallback-2pass"
    info["size"] = list(cur.size)
    info["model"] = MODEL_ID
    return cur, info
