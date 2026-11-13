#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把语料统一成"分析尺寸"的基线 JPEG（420 宽），
这样 Node 端 jpeg-js 能全量解码，且与浏览器 canvas 缩放后的输入规模一致。
"""
import json, os, sys
from PIL import Image

SRC = sys.argv[1] if len(sys.argv) > 1 else "corpus"
W = 420
OUT = os.path.join(SRC, "_norm")

os.makedirs(OUT, exist_ok=True)
man = json.load(open(os.path.join(SRC, "manifest.json"), encoding="utf-8"))
ok, bad = 0, 0
index = []

for it in man:
    src = os.path.join(SRC, it["file"])
    if not os.path.exists(src):
        bad += 1
        continue
    dst = os.path.join(OUT, os.path.splitext(it["file"])[0] + ".jpg")
    try:
        im = Image.open(src).convert("RGB")
        w, h = im.size
        if w < 700:
            bad += 1
            continue
        nh = max(120, round(W * h / w))
        im = im.resize((W, nh), Image.LANCZOS)
        im.save(dst, quality=88, optimize=True)
        index.append({"file": os.path.basename(dst), "cat": it.get("cat", "unknown"),
                      "src_file": it["file"], "w": W, "h": nh})
        ok += 1
    except Exception as e:
        bad += 1

json.dump(index, open(os.path.join(OUT, "index.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("归一化 %d 张，跳过 %d 张 -> %s" % (ok, bad, OUT))
