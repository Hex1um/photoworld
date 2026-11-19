"""输入读取：图片目录 / 视频文件 → 一组 RGB 帧。

视频的坑：直接按帧号抽会抽到大量"几乎一样"的帧（手持视频 30fps，相邻帧位移不到 1px），
SfM 在这种帧上三角化会退化（基线太小，深度噪声爆炸）。所以这里按**平均哈希去重**：
只保留和上一帧差异超过阈值的帧，同时限制最大帧数。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass
class Frame:
    name: str
    rgb: np.ndarray            # uint8, (H, W, 3), RGB
    index: int


def _ahash(img_rgb: np.ndarray, size: int = 16) -> np.ndarray:
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (g > g.mean()).astype(np.uint8).ravel()


def _read_image(path: str, max_side: int) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_images(folder: str, max_side: int = 1280, limit: int = 40,
                min_gap: float = 0.04) -> list[Frame]:
    """读目录里的图片。min_gap 是"相邻帧最小哈希差异"，用来去掉连拍重复。"""
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(IMG_EXT))
    if not names:
        raise SystemExit("目录里没有图片：%s" % folder)
    out: list[Frame] = []
    last = None
    for n in names:
        rgb = _read_image(os.path.join(folder, n), max_side)
        if rgb is None:
            print("  跳过（读不出）", n)
            continue
        h = _ahash(rgb)
        if last is not None and float(np.mean(h != last)) < min_gap:
            print("  跳过（与上一张太像）", n)
            continue
        out.append(Frame(n, rgb, len(out)))
        last = h
        if len(out) >= limit:
            break
    print("读入 %d 张图片（目录 %s）" % (len(out), folder))
    return out


def load_video(path: str, max_side: int = 1280, fps: float = 1.5,
               limit: int = 60, min_gap: float = 0.06) -> list[Frame]:
    """按 fps 抽帧 + 哈希去重。手机随手拍的一段环绕视频就能直接喂进来。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("打不开视频：%s" % path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / max(0.1, fps))))
    out: list[Frame] = []
    last = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            s = max_side / max(h, w)
            if s < 1.0:
                rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            hh = _ahash(rgb)
            if last is None or float(np.mean(hh != last)) >= min_gap:
                out.append(Frame("f%05d" % i, rgb, len(out)))
                last = hh
            if len(out) >= limit:
                break
        i += 1
    cap.release()
    print("视频 %.1ffps 共 %d 帧 → 抽出 %d 帧（%.1f fps 抽样 + 去重）"
          % (src_fps, total, len(out), fps))
    return out


def load_inputs(path: str, **kw) -> list[Frame]:
    if os.path.isdir(path):
        return load_images(path, **{k: v for k, v in kw.items()
                                    if k in ("max_side", "limit", "min_gap")})
    return load_video(path, **{k: v for k, v in kw.items()
                               if k in ("max_side", "fps", "limit", "min_gap")})
