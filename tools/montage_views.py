#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把一次 multiview 的 32 机位截图拼成「每图一张 3×2 联系表」，便于逐图目视。

    python tools/montage_views.py --shots out/shots --out out/montage [--per 12]

为什么需要它：32 机位的指标（stripe/void）无法反映"几何立不立得起来"
（见 docs/ISSUES.md §25 那次塌平事故），必须人眼看。逐张点开 12×32 张图不现实，
拼成联系表之后一张图就能看出"正面 / 侧面 / 背面 / 照片外"四种机位的情况。
"""
from __future__ import annotations

import argparse
import glob
import os

from PIL import Image, ImageDraw

# 32 机位套餐：A 圈 12 个（r=10，平视）、B 圈 12 个（r=26，微仰）、C 圈 8 个（r=75，区域外）
PICK = ["A00", "A03", "A06", "A09", "B18", "C28"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=460)
    ap.add_argument("--h", type=int, default=288)
    ap.add_argument("--pick", default="",
                    help="逗号分隔的机位后缀（如 A00,A03,A06,A09,A11,B18,C28）；"
                         "留空用默认 6 机位")
    ap.add_argument("--cols", type=int, default=0, help="列数；留空按 pick 数量自动")
    ap.add_argument("--only", default="", help="只拼这些图（逗号分隔的子串）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pick = [t.strip() for t in args.pick.split(",") if t.strip()] or PICK
    pat = os.path.join(args.shots, "*_%s.png" % pick[0])
    names = sorted({os.path.basename(p).rsplit("_", 1)[0] for p in glob.glob(pat)})
    if not names:
        print("没有找到截图：%s" % pat)
        return 1
    if args.only:
        keep = [s.strip() for s in args.only.split(",") if s.strip()]
        names = [n for n in names if any(k in n for k in keep)]

    cols = args.cols or (3 if len(pick) <= 6 else 4)
    rows = (len(pick) + cols - 1) // cols
    for name in names:
        sheet = Image.new("RGB", (args.w * cols, args.h * rows), (18, 18, 20))
        dr = ImageDraw.Draw(sheet)
        got = 0
        for i, tag in enumerate(pick):
            p = os.path.join(args.shots, "%s_%s.png" % (name, tag))
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB")
            im = im.resize((args.w, args.h), Image.LANCZOS)
            x, y = (i % cols) * args.w, (i // cols) * args.h
            sheet.paste(im, (x, y))
            dr.rectangle([x, y, x + 52, y + 20], fill=(0, 0, 0))
            dr.text((x + 6, y + 4), tag, fill=(255, 220, 120))
            got += 1
        out = os.path.join(args.out, name + "_sheet.png")
        sheet.save(out)
        print("%-34s %d 机位 → %s" % (name, got, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
