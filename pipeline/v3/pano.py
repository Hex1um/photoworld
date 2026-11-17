"""视锥外 ①：把照片扩成 360° 全景（等距柱状 / equirect）。

为什么先做全景，而不是直接"往外生成地形"
------------------------------------------
视锥外既没有纹理也没有几何。凭空造地形只能得到"风格相似"的噪声（v1.16 的 fBm
就是这条路，观感上是"程序化色块"）。而**扩散模型擅长的是图像**：给它一张
"中间是真的、四周是空的"全景图，它会把地貌、天空、植被在语义上接下去。
拿到全景之后再跑深度，就有了"视锥外的地形与地物"的真实依据。

实现要点
--------
* 先把照片按**真实内参**投到等距柱状图上：这一块是"已知区"，像素来自原图，
  不做任何生成（否则视锥内就不是原图了）。
* 已知区外的部分交给 SDXL inpainting。提示词从照片的色彩统计自动推
  （雪/荒漠/森林/海/城市），不写死。
* 生成后：已知区**逐像素还原**成原图；左右接缝做环绕交叉淡化，
  否则转身 180° 会看到一条竖缝。
"""
from __future__ import annotations

import math
import os

import numpy as np
from scipy import ndimage

from PIL import Image

from . import common

_PIPE = None


def equirect_dirs(w: int, h: int) -> np.ndarray:
    """等距柱状图每个像素的世界方向（lon=0 → -Z 正前方，y 向上）。"""
    lon = ((np.arange(w) + 0.5) / w - 0.5) * 2.0 * math.pi
    lat = (0.5 - (np.arange(h) + 0.5) / h) * math.pi
    L, A = np.meshgrid(lon, lat)
    d = np.stack([np.sin(L) * np.cos(A), np.sin(A), -np.cos(L) * np.cos(A)], axis=-1)
    return d.astype(np.float32)


def project_dirs(dirs: np.ndarray, cam: common.Camera3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """世界方向 → 照片像素坐标 (u,v) + 是否落在画幅内。"""
    d = np.asarray(dirs, dtype=np.float64)
    if abs(cam.pitch) > 1e-12:
        c, s = math.cos(cam.pitch), math.sin(cam.pitch)
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        d = np.stack([x, y * c + z * s, -y * s + z * c], axis=-1)
    depth = -d[..., 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    u = (d[..., 0] / (safe * cam.tan_h)) * 0.5 + 0.5
    v = 0.5 - (d[..., 1] / (safe * cam.tan_v)) * 0.5
    inside = (depth > 0) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    return u * cam.w - 0.5, v * cam.h - 0.5, inside


def photo_to_equirect(rgb: np.ndarray, cam: common.Camera3,
                      pw: int, ph: int) -> tuple[np.ndarray, np.ndarray]:
    """照片 → (全景图, 已知区掩码)。已知区像素 = 原图像素的双线性采样。"""
    dirs = equirect_dirs(pw, ph)
    px, py, inside = project_dirs(dirs, cam)
    coords = np.stack([np.clip(py, 0, cam.h - 1), np.clip(px, 0, cam.w - 1)], axis=0)
    pano = np.stack([ndimage.map_coordinates(rgb[..., c], coords, order=1, mode="nearest")
                     for c in range(3)], axis=-1).astype(np.float32)

    # 已知区之外不能留"边缘拖影"：map_coordinates 的 nearest 会把照片四边的像素
    # 沿等距柱状图横向拉成条纹，扩散模型会照着条纹继续画（实测出现左右镜像的
    # 重复地貌）。这里改成"天在上、地在下"的垂直渐变，颜色取自照片顶/底缘 ——
    # 模型拿到的先验合理，遮罩仍然要求它重画。
    sky_col = rgb[: max(1, rgb.shape[0] // 25)].reshape(-1, 3).mean(axis=0)
    grd_col = rgb[-max(1, rgb.shape[0] // 25):].reshape(-1, 3).mean(axis=0)
    lat = (0.5 - (np.arange(ph) + 0.5) / ph) * math.pi          # +π/2 顶 → -π/2 底
    t = np.clip(lat / (math.pi * 0.5), -1, 1)
    w = ((t + 1) * 0.5) ** 1.3                                   # 地平线附近过渡更快
    grad = grd_col[None, None, :] * (1 - w[:, None, None]) + sky_col[None, None, :] * w[:, None, None]
    grad = np.repeat(grad, pw, axis=1).astype(np.float32)
    pano = np.where(inside[..., None], pano, grad)
    return pano, inside


def _prompt_from_photo(rgb: np.ndarray, cam: common.Camera3) -> str:
    """从照片统计推提示词。不写死题材：这套管线要吃任意风景/室内照。"""
    h = rgb.shape[0]
    luma = float(rgb.mean())
    top = rgb[: h // 3].reshape(-1, 3).mean(axis=0)
    bot = rgb[-h // 4:].reshape(-1, 3).mean(axis=0)
    sat = float((rgb.max(axis=2) - rgb.min(axis=2)).mean())
    # 判据顺序按"最容易误判的排后面"：先看有没有雪/亮天空，再看水、荒漠、森林。
    top_lum = float(top @ np.array([0.299, 0.587, 0.114], np.float32))
    bot_lum = float(bot @ np.array([0.299, 0.587, 0.114], np.float32))
    if luma > 0.62 and sat < 0.16:
        scene = "snowy mountain landscape with pine forest and calm water"
    elif top_lum > 0.55 and bot_lum < top_lum * 0.75 and bot[2] >= bot[0]:
        # 亮天空 + 更暗的下半幅 + 偏蓝 → 湖/海面（不要判成"海岸悬崖"）
        scene = "alpine lake landscape with mirrored water, forest and mountains at dawn"
    elif bot[2] > bot[0] * 1.08 and bot_lum < 0.42:
        scene = "coastal cliff landscape with dark sea and overcast sky"
    elif top[0] > top[2] * 1.12 and sat > 0.22:
        scene = "red rock desert canyon landscape under a warm sky"
    elif bot[1] > bot[0] * 1.06 and sat > 0.14:
        scene = "green mountain meadow with forest and distant peaks"
    elif luma < 0.3:
        scene = "night landscape with dark sky and distant lights"
    else:
        scene = "natural landscape with mountains, sky and open ground"
    return (f"a photorealistic 360 degree equirectangular panorama of {scene}, "
            "seamless continuation of the same terrain, consistent lighting, "
            "natural colors, no text, no watermark, high detail")


def _load_pipe(device: str = "cuda"):
    global _PIPE
    if _PIPE is None:
        import torch
        from diffusers import StableDiffusionXLInpaintPipeline
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16, variant="fp16", local_files_only=True)
        # 8GB 笔记本卡装不下 unet+双 text encoder 的 fp16 全量，用 CPU offload 换显存。
        # **只能用 model 级 offload**：sequential 档是一步一搬权重，实测把 20 步的
        # 全景从 30s 拖到 15 分钟以上（GPU 显示 99% 但全在等 PCIe，很容易误判成"在算"）。
        torch.cuda.empty_cache()
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        _PIPE = pipe
    return _PIPE


def inpaint(pano: np.ndarray, known: np.ndarray, prompt: str, steps: int = 20,
            guidance: float = 6.5, seed: int = 0) -> np.ndarray:
    """已知区保留、其余交给 SDXL 补全。返回补全后的全景。"""
    import torch
    from PIL import Image

    pipe = _load_pipe()
    h, w = pano.shape[:2]
    # mask：255 = 要重画
    mask = np.where(known, 0, 255).astype(np.uint8)
    # 已知区外扩 2% 作为羽化带，避免硬边
    feather = ndimage.gaussian_filter((~known).astype(np.float32), 2.0)
    mask_img = Image.fromarray(np.clip(np.maximum(mask / 255.0, feather) * 255, 0, 255)
                              .astype(np.uint8))
    img = Image.fromarray((np.clip(pano, 0, 1) * 255 + 0.5).astype(np.uint8))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = pipe(prompt=prompt, negative_prompt="text, watermark, blurry, distorted, seam",
               image=img, mask_image=mask_img, height=h, width=w,
               num_inference_steps=steps, guidance_scale=guidance,
               generator=gen).images[0]
    gen_arr = np.asarray(out, dtype=np.float32) / 255.0

    # 已知区逐像素还原成原图（视锥内必须是原图，不能是"重画过的原图"）
    res = gen_arr.copy()
    res[known] = pano[known]
    # 羽化带内按距离混合
    band = (~known) & (feather > 0.02)
    a = np.clip(feather[band][:, None], 0, 1)
    res[band] = gen_arr[band] * a + pano[band] * (1 - a)

    # 左右环绕接缝：最后 6% 列与最前 6% 列交叉淡化，转身时才不会看到竖缝
    k = max(4, int(w * 0.06))
    ramp = (np.linspace(0, 1, k, dtype=np.float32) ** 2)[None, :, None]
    left = res[:, :k].copy()
    right = res[:, -k:].copy()
    res[:, :k] = right * (1 - ramp) + left * ramp
    res[:, -k:] = left * ramp + right * (1 - ramp)
    return np.clip(res, 0, 1)


def _cache_key(rgb: np.ndarray, cam: common.Camera3, prompt: str, cfg: dict) -> str:
    """全景缓存键：只跟"输入照片 + 内参 + 提示词 + 采样参数"有关。

    为什么要缓存：全景是整条链路最贵的一步（20 步 SDXL）。调几何参数
    （cliff / skirt / ground_tol）时全景完全不变，重跑纯属浪费 —— 参数搜索时
    这一步能把一轮从几十分钟压到几分钟。实测笔记本卡在省电档下一次全景
    能拖到十分钟以上（GPU 显存 99% 但全在等 PCIe）。
    """
    import hashlib
    h = hashlib.sha1()
    h.update(np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).tobytes())
    h.update(("%.4f|%.4f|%.3f|%d|%d|%d|%s" % (
        cam.hfov, cam.pitch, cam.eye, int(cfg.get("pano_w", 1024)),
        int(cfg.get("pano_steps", 20)), int(cfg.get("seed", 0)), prompt)).encode("utf-8"))
    return h.hexdigest()[:16]



def bg_plate(rgb: np.ndarray, fg: np.ndarray, prompt: str, cam: common.Camera3,
             cfg: dict) -> np.ndarray:
    """把前景抠掉、用扩散补出一张**背景板**（与照片同尺寸、同像素网格）。

    为什么值得多跑一次扩散：`fill` 层是"侧移时露出来的那一层"。之前它取的是
    "最近的非前景像素"的颜色 —— 前景背后就是一片糊色，侧看是拉伸的色带。
    补出一张真正的背景图之后，那一层的颜色是有内容的：栏杆后面是路，树后面是坡。

    几何不动（仍然用 push-pull 出来的 z_bg），只换颜色 —— 也就是用户说的
    "先建模，再着色"。
    """
    import torch
    from PIL import Image
    from scipy import ndimage

    cache_dir = cfg.get("cache_dir") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "cache", "v3")
    key = _cache_key(rgb, cam, prompt + "|bg", cfg)
    path = os.path.join(cache_dir, "bg_%s.jpg" % key)
    if os.path.exists(path):
        im = Image.open(path).convert("RGB")
        if im.size != (rgb.shape[1], rgb.shape[0]):
            im = im.resize((rgb.shape[1], rgb.shape[0]))
        return np.asarray(im, dtype=np.float32) / 255.0

    # 要重画的区域：前景外扩 9px 再羽化 —— 抠得太紧会在物体边缘留一圈原物体的颜色
    paint = ndimage.binary_dilation(fg, np.ones((9, 9), bool)).astype(np.float32)
    paint = ndimage.gaussian_filter(paint, 2.0)

    pipe = _load_pipe()
    h, w = rgb.shape[:2]
    pw = int(cfg.get("bg_w", 1024))
    ph = max(64, int(round(pw * h / float(w) / 8.0)) * 8)
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).resize((pw, ph))
    mask = Image.fromarray((np.clip(paint, 0, 1) * 255).astype(np.uint8)).resize((pw, ph))
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 0)))
    out = pipe(prompt=prompt + ", clean empty background, no foreground objects, "
                              "natural continuation of ground and sky",
               negative_prompt="object, person, animal, text, watermark, blurry, duplicate",
               image=img, mask_image=mask, height=ph, width=pw,
               num_inference_steps=int(cfg.get("bg_steps", 18)),
               guidance_scale=float(cfg.get("pano_cfg", 6.5)),
               generator=gen).images[0]
    plate = np.asarray(out.resize((w, h)), dtype=np.float32) / 255.0
    # 前景之外的像素保持原图：只有"被挡住的那块"需要扩散，别的区域原图更可信
    keep = paint < 0.5
    plate[keep] = rgb[keep]
    try:
        os.makedirs(cache_dir, exist_ok=True)
        common.save_rgb(path, plate, quality=95)
    except Exception:
        pass
    return plate


def build(rgb: np.ndarray, cam: common.Camera3, cfg: dict) -> dict:
    pw = int(cfg.get("pano_w", 1024))
    ph = int(cfg.get("pano_h", 512))
    pano0, known = photo_to_equirect(rgb, cam, pw, ph)
    prompt = cfg.get("prompt") or _prompt_from_photo(rgb, cam)

    src = cfg.get("pano_from")
    cache_dir = cfg.get("cache_dir") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "cache", "v3")
    key = _cache_key(rgb, cam, prompt, cfg)
    cache_path = os.path.join(cache_dir, "pano_%s.jpg" % key)

    def _load(path):
        im = Image.open(path).convert("RGB")
        if im.size != (pw, ph):
            im = im.resize((pw, ph))
        return np.asarray(im, dtype=np.float32) / 255.0

    if src and os.path.exists(src):
        full, cache_hit = _load(src), "explicit"
    elif os.path.exists(cache_path):
        full, cache_hit = _load(cache_path), "cache"
    elif cfg.get("no_diffusion"):
        full, cache_hit = pano0, "none"
    else:
        full = inpaint(pano0, known, prompt,
                       steps=int(cfg.get("pano_steps", 20)),
                       guidance=float(cfg.get("pano_cfg", 6.5)),
                       seed=int(cfg.get("seed", 0)))
        cache_hit = "generated"
        try:
            os.makedirs(cache_dir, exist_ok=True)
            common.save_rgb(cache_path, full, quality=95)
        except Exception:
            pass

    # 无论全景从哪来，**已知区都逐像素还原成原图**：
    # 复用缓存/外部全景不能把"视锥内必须等于原图"这条硬要求弄丢。
    full = np.where(known[..., None], pano0, full).astype(np.float32)
    return {"pano": full, "known": known, "prompt": prompt,
            "known_frac": float(known.mean()), "cache": cache_hit}
