#!/usr/bin/env python3
"""多视角质检结果汇总器：按圈（A/B/C）拆指标，列出最差的图与最差视角。

  python tools/mv_report.py reports/all32.json
  python tools/mv_report.py reports/all32.json --top 8 --ring C

从 mv_report.mjs 翻译而来，输出格式保持一致；.mjs 保留为薄包装转调本脚本。
"""
from __future__ import annotations

import json
import os
import sys

# 与 tools/clarity.py 的 CLARITY_KEYS 保持一致；test_clarity_parity.py
# 会同时校验 Python / JS 两侧常量，改任一处都要跑该测试。
CLARITY_KEYS = ['hfEnergy', 'farHF', 'centerHF', 'lapVar', 'centerLapVar', 'localContrast']

# 与原 mjs 输出字节一致：Windows 下强制 LF，避免 CRLF 造成 diff
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(newline="\n")


def mean(a: list) -> float:
    return sum(a) / len(a) if a else 0.0


def main(argv: list[str]) -> int:
    args = argv[1:]
    file = args[0] if args else "reports/all32.json"

    def arg(name: str, default: str) -> str:
        try:
            return args[args.index("--" + name) + 1]
        except (ValueError, IndexError):
            return default

    top = int(arg("top", "8"))
    ring = arg("ring", "")

    with open(file, encoding="utf-8") as f:
        d = json.load(f)
    ok = [r for r in (d.get("results") or []) if r.get("ok")]

    keys = ['voidFrac', 'innerVoid', 'lowVoid', 'borderVoid', 'darkFrac',
            'stripe', 'detail'] + CLARITY_KEYS
    rings = sorted({v.get("ring", "A") for r in ok for v in r.get("views", [])})

    print("样本 %d 张 · 每张 %d 视角 · 圈 %s" %
          (len(ok), len(ok[0].get("views", [])) if ok else 0, "/".join(rings)))
    print("圈   " + "".join(k.ljust(16) for k in keys))
    for rg in rings:
        vs = [v for r in ok for v in r.get("views", []) if v.get("ring", "A") == rg]
        print(rg.ljust(4) + "".join(
            ("%.4f" % mean([v.get(k, 0) for v in vs])).ljust(16) for k in keys))

    # 单视角最差：按指标排序
    flat = []
    for r in ok:
        for v in r.get("views", []):
            if ring and v.get("ring", "A") != ring:
                continue
            row = {"image": r.get("image"), "ring": v.get("ring"), "a": v.get("a")}
            row.update(v)
            flat.append(row)

    def bad(k: str):
        return sorted(flat, key=lambda x: x.get(k, 0), reverse=True)[:top]

    for k in ['voidFrac', 'lowVoid', 'innerVoid', 'stripe', 'darkFrac']:
        print("\n最差 %s：" % k)
        for v in bad(k):
            image = str(v["image"]).split("/")[-1]
            fn = str(v.get("file", "")).split("/")[-1]
            print("  %.4f  %s  %s%s°  %s" % (v.get(k, 0), image, v.get("ring"), v.get("a"), fn))

    print("\n竖幕/拉丝最重的文件：")
    for v in bad('stripe')[:5]:
        print("  " + str(v.get("file", "")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
