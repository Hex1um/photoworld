#!/usr/bin/env python
"""把收集到的风景/室内图集整理成可训练的干净数据集。

  python train/prepare_landscape_dataset.py --src D:/datasets/raw --out data/landscapes
  python train/prepare_landscape_dataset.py --src raw --out data/landscapes --panorama

做四件事：
  1. 递归收集 jpg/png/webp，跳过损坏图与过小图；
  2. 用 8×8 均值哈希去重（汉明距离 ≤ 4 视为重复）；
  3. 统一成 512×384 JPEG（等比缩放 + 中心裁切），保留原图比例信息；
  4. 按哈希确定性切 train/val，写 manifest.json。

--panorama 会把 2:1 等距柱状全景切成 4 个 90° 透视视角（每 90° 一个），
这对「无限地貌 / 世界模型」训练特别有价值：同一场景天然有 360° 监督。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
from PIL import Image

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def ahash(img: Image.Image, n: int = 8) -> int:
    g = img.convert("L").resize((n, n), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    bits = (a > a.mean()).reshape(-1)
    v = 0
    for b in bits:
        v = (v << 1) | int(bool(b))
    return v


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def fit_512x384(img: Image.Image) -> Image.Image:
    """等比缩放到覆盖 512×384，再中心裁切；小图先放大。"""
    tw, th = 512, 384
    w, h = img.size
    s = max(tw / w, th / h)
    nw, nh = max(tw, int(round(w * s))), max(th, int(round(h * s)))
    img = img.resize((nw, nh), Image.LANCZOS)
    x = (nw - tw) // 2
    y = (nh - th) // 2
    return img.crop((x, y, x + tw, y + th))


def equirect_to_perspective(img: Image.Image, yaw_deg: float, fov_deg: float = 90.0,
                            out_w: int = 512, out_h: int = 384) -> Image.Image:
    """等距柱状全景 → 单个透视视图（针孔相机，pitch=0）。"""
    src = np.asarray(img.convert("RGB"), dtype=np.float32)
    sh, sw, _ = src.shape
    f = (out_w / 2) / math.tan(math.radians(fov_deg) / 2)
    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    dx = (xs - out_w / 2)
    dy = (ys - out_h / 2)
    # 相机朝向 yaw：世界方向
    yaw = math.radians(yaw_deg)
    # 视线方向（相机系 -> 世界）
    rx = dx / f
    ry = -dy / f
    rz = np.ones_like(rx)
    # 绕 Y 旋转 yaw
    wx = rx * math.cos(yaw) + rz * math.sin(yaw)
    wz = -rx * math.sin(yaw) + rz * math.cos(yaw)
    wy = ry
    lon = np.arctan2(wx, wz)                 # -pi..pi
    lat = np.arctan2(wy, np.sqrt(wx * wx + wz * wz))
    u = (lon / (2 * math.pi) + 0.5) * (sw - 1)
    v = (0.5 - lat / math.pi) * (sh - 1)
    u0 = np.clip(np.floor(u).astype(np.int32), 0, sw - 1)
    v0 = np.clip(np.floor(v).astype(np.int32), 0, sh - 1)
    out = src[v0, u0]
    return Image.fromarray(out.astype(np.uint8), "RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/landscapes")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--min-side", type=int, default=480)
    ap.add_argument("--panorama", action="store_true", help="2:1 全景切成 4 个透视视角")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = []
    for root, _, names in os.walk(a.src):
        for n in names:
            if n.lower().endswith(EXTS):
                files.append(os.path.join(root, n))
    files.sort()
    print("发现 %d 张图" % len(files))

    for sub in ("train", "val"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    seen = []
    manifest = []
    kept = dup = bad = 0
    for i, p in enumerate(files):
        if a.limit and kept >= a.limit:
            break
        try:
            img = Image.open(p)
            img.load()
            img = img.convert("RGB")
        except Exception as e:
            bad += 1
            print("  [bad] %s (%s)" % (p, e))
            continue
        if min(img.size) < a.min_side:
            bad += 1
            continue
        h = ahash(img)
        if any(hamming(h, s) <= 4 for s in seen):
            dup += 1
            continue
        seen.append(h)

        # 确定性切分：哈希奇偶 + 比例
        key = "%s|%d" % (os.path.basename(p), h)
        r = int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000 / 1000.0
        split = "val" if r < a.val_ratio else "train"
        stem = "%06d_%s" % (i, hashlib.md5(key.encode()).hexdigest()[:8])

        outs = []
        if a.panorama and img.size[0] >= img.size[1] * 1.8:
            for k in range(4):
                view = equirect_to_perspective(img, k * 90.0)
                view = fit_512x384(view)
                name = "%s_p%d.jpg" % (stem, k)
                view.save(os.path.join(a.out, split, name), quality=92)
                outs.append(name)
        else:
            out = fit_512x384(img)
            name = "%s.jpg" % stem
            out.save(os.path.join(a.out, split, name), quality=92)
            outs.append(name)

        for name in outs:
            manifest.append({
                "file": os.path.join(split, name),
                "split": split,
                "source": p,
                "hash": "%016x" % h,
            })
        kept += 1
        if kept % 200 == 0:
            print("  kept=%d dup=%d bad=%d" % (kept, dup, bad))

    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"count": len(manifest), "kept": kept, "dup": dup, "bad": bad,
                   "entries": manifest}, f, ensure_ascii=False, indent=2)
    print("\n完成：保留 %d 张（展开 %d 个样本），重复 %d，损坏/过小 %d"
          % (kept, len(manifest), dup, bad))
    print("输出：%s  (train/ val/ manifest.json)" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
