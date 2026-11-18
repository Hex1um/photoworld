#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理全景天空里的"大亮斑/大圆盘"（对 .pano.jpg 做后处理，不需要重新烘焙）。

为什么单独一个工具：全景是等距柱状图，天空那半边是扩散外推出来的，容易画出一大团
低频率的亮斑 —— 贴到天空球上就是屏幕正中一个圆盘（用户反复指出的那个）。
而"天空只该随仰角变化"是硬约束，所以在**方位方向**做高通：
    col' = 行中值 + (col - 方位低通(col))
小尺度的云、霞光保留，大尺度的亮斑被抹平。

因为地形（mode 7）与天空球（mode 8）都是在**运行时**采这张 jpg，改完立刻生效，
不用重跑 12 分钟的烘焙。

    python tools/v3_pano_clean.py assets/world-v3-alpine.pano.jpg
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image
from scipy import ndimage


def clean(path: str, sky_frac: float = 0.34, sigma_frac: float = 0.09,
          strength: float = 0.85, pole_frac: float = 0.22) -> None:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    a = np.asarray(im, dtype=np.float32) / 255.0
    sky_rows = int(h * sky_frac)              # 上面的部分是天空
    sky = a[:sky_rows]
    # 方位向低通（wrap：等距柱状图左右相接）
    low = ndimage.gaussian_filter(sky, sigma=(0.0, max(2.0, w * sigma_frac), 0.0),
                                 mode=("nearest", "wrap", "nearest"))
    band = np.median(low, axis=1, keepdims=True)
    flat = band + (sky - low)
    out = a.copy()
    out[:sky_rows] = np.clip(a[:sky_rows] * (1.0 - strength) + flat * strength, 0.0, 1.0)
    # 天顶再压一层竖直平滑，避免镜像平铺留下的横向硬条
    top = max(4, int(sky_rows * 0.35))
    out[:top] = ndimage.gaussian_filter(out[:top], sigma=(3.0, 6.0, 0.0),
                                        mode=("nearest", "wrap", "nearest"))
    # 极区（天顶/天底）压成"逐行均值"：抬头/低头时那片是镜像拉伸出来的条纹，
    # 谁也不需要看清天底长什么样，压平反而干净。
    npole = max(2, int(h * pole_frac))
    for y in list(range(0, npole)) + list(range(h - npole, h)):
        row = out[y].mean(axis=0, keepdims=True)
        t = 1.0 - abs(y - (0 if y < h / 2 else h - 1)) / float(npole)
        out[y] = out[y] * (1 - t) + row * t
    Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(path, quality=95)
    print("已清理天空低频谱：%s（天空行 %d/%d）" % (path, sky_rows, h))


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "assets/world-v3-alpine.pano.jpg"
    clean(p)
