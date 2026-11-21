#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 HuggingFace 拉一段**真实多视角**测试素材（NeRF-Synthetic / lego），带真值位姿。

为什么用它：仓库里此前的合成集都是"纯旋转 / 远景"，只能验证"坏素材被拦"，
从来没验证过"好素材能重建出来"。lego 是绕物体 360° 的 100 个视角，视差大、真值齐，
正好用来量化 v4 的位姿精度。

    python tools/fetch_lego_sample.py --n 36 --out testdata/lego
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import urllib.request

BASE = "https://hf-mirror.com/datasets/phuckstnk63/nerf-synthetic/resolve/main/nerf_synthetic/lego/"
RAW = "https://raw.githubusercontent.com/Hex1um/photoworld/main/"


def fetch(rel: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return
    url = BASE + rel
    req = urllib.request.Request(url, headers={"User-Agent": "pw/1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=36, help="抽多少张（从 100 帧里等间隔取）")
    ap.add_argument("--out", default="testdata/lego")
    args = ap.parse_args()

    import numpy as np
    from PIL import Image

    os.makedirs(args.out, exist_ok=True)
    tj = os.path.join(args.out, "_transforms.json")
    fetch("transforms_train.json", tj)
    meta = json.load(open(tj, encoding="utf-8"))
    frames = meta["frames"]
    ang = float(meta["camera_angle_x"])
    # NeRF-Synthetic 的 train 帧是**乱序**的（方位角 -2.5°、-67.4°、66.3°… 跳着来），
    # 直接按文件名取相邻帧会拿去匹配"相隔 50°"的两张，SIFT 只有十几个匹配（实测）。
    # 这里按"用户的拍法"重构一条环绕路径：挑同一高度带、按方位角排序、等间隔取。
    import numpy as np
    C = np.array([np.array(f["transform_matrix"], dtype=np.float64)[:3, 3] for f in frames])
    az = np.degrees(np.arctan2(C[:, 0], C[:, 2]))
    el = np.degrees(np.arcsin(C[:, 1] / (np.linalg.norm(C, axis=1) + 1e-9)))
    band = np.abs(el - np.median(el)) < 15.0
    order = np.argsort(az[band])
    idx = np.array(np.nonzero(band)[0])[order]
    print("高度带内 %d 帧，按方位角排序后取 %d 帧" % (len(idx), min(args.n, len(idx))))
    step = max(1, len(idx) // args.n)
    picked = [frames[i] for i in idx[::step][: args.n]]
    print("transforms: %d 帧，取 %d 帧；camera_angle_x=%.4f rad (%.1f°)"
          % (len(frames), len(picked), ang, math.degrees(ang)))

    views = []
    for i, fr in enumerate(picked):
        rel = os.path.basename(fr["file_path"]) + ".png"
        tmp = os.path.join(args.out, "_raw_" + rel)
        fetch("train/" + rel, tmp)
        img = Image.open(tmp).convert("RGBA")
        bg = Image.new("RGBA", img.size, (120, 124, 130, 255))   # 透明背景压成中性灰
        flat = Image.alpha_composite(bg, img).convert("RGB")
        name = "img_%03d.jpg" % i
        flat.save(os.path.join(args.out, name), quality=92)
        os.remove(tmp)

        # transform_matrix 是 **camera→world**（OpenGL，相机看 -z），
        # v4 要的是 world→cam：R = C[:3,:3]^T，t = -R·C_tr
        C = np.array(fr["transform_matrix"], dtype=np.float64)
        Rwc = C[:3, :3].T
        twc = -Rwc @ C[:3, 3]
        views.append({"R": Rwc.tolist(), "t": twc.reshape(3, 1).tolist(),
                      "center": C[:3, 3].tolist(), "file": name})

    w, h = 800, 800
    f = (w / 2.0) / math.tan(ang / 2.0)
    K = [[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]]
    with open(os.path.join(args.out, "gt.json"), "w", encoding="utf-8") as fh:
        json.dump({"K": K, "w": w, "h": h, "views": views}, fh, indent=1)
    print("写入 %d 张图 + gt.json（K 焦距 %.1f，focal_ratio=%.3f）"
          % (len(views), f, f / max(w, h)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
