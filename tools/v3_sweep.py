#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 参数搜索（这里的"训练"）。

v3 没有可以反传的网络 —— 重建是"确定性管线 + 十来个几何旋钮"。所以"训练"在这里
等于**用验收指标搜参数**：每个组合真的跑一遍 bake，再用 `verify` 光栅化回原点视角，
拿 PSNR/SSIM/覆盖率打分。这就是 v2 里 skillopt 在做的事，只是 v3 先把"确定性那块"
跑通：先把参数搜明白，再谈让 LLM 来改准则。

搜索目标（按优先级）：
  1. `psnr_core`  —— 原点视角、只统计照片定义过的像素。**这是硬指标**。
  2. `coverage`   —— 有几何覆盖的比例（穹顶/地形补洞之后应接近 1）。
  3. `ms`         —— 单张耗时（同分时取快的那组）。

用法：
  python tools/v3_sweep.py                                  # 内置 4 个世界 × 默认网格
  python tools/v3_sweep.py --corpus worlds --corpus corpus_test --limit 6
  python tools/v3_sweep.py --grid "cliff=1.4,1.8,2.4;skirt=0.3,0.5" --step 8
  python tools/v3_sweep.py --with-diffusion                 # 连全景/地形一起评（慢 4 倍）

默认 **不跑扩散**：视锥内的保真度与全景无关，关掉能让一轮搜索从 40 分钟压到几分钟。
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import statistics
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "pipeline"), os.path.join(_ROOT, "pipeline", "v2"),
           os.path.join(_ROOT, "pipeline", "v3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from v3 import bake as v3bake, common  # noqa: E402

DEFAULT_GRID = {
    "cliff": [1.4, 1.8, 2.4],
    "skirt": [0.30, 0.50],
}


def parse_grid(spec: str) -> dict:
    grid = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        vals = []
        for tok in v.split(","):
            tok = tok.strip()
            if not tok:
                continue
            vals.append(float(tok) if ("." in tok or "e" in tok.lower()) else int(tok))
        if k and vals:
            grid[k.strip()] = vals
    return grid or DEFAULT_GRID


def collect(dirs: list[str], limit: int) -> list[str]:
    files = []
    for d in dirs:
        if os.path.isdir(d):
            files += sorted(glob.glob(os.path.join(d, "*.jpg")))
        else:
            files += sorted(glob.glob(d))
    files = [f for f in files if "thumb" not in os.path.basename(f).lower()]
    return files[:limit] if limit else files


def main() -> int:
    ap = argparse.ArgumentParser(description="v3 参数搜索：用验收指标选几何参数")
    ap.add_argument("--corpus", action="append", default=None,
                    help="目录或通配符，可多次给；默认 worlds/ + corpus_test/")
    ap.add_argument("--limit", type=int, default=4, help="最多几张图（默认 4）")
    ap.add_argument("--grid", default="", help='如 "cliff=1.4,1.8,2.4;skirt=0.3,0.5"')
    ap.add_argument("--step", type=int, default=6, help="网格抽稀（搜索阶段用粗网格）")
    ap.add_argument("--with-diffusion", action="store_true")
    ap.add_argument("--pano-steps", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(_ROOT, "reports", "v3_sweep"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dirs = args.corpus or [os.path.join(_ROOT, "worlds"), os.path.join(_ROOT, "corpus_test")]
    files = collect(dirs, args.limit)
    if not files:
        print("没有输入图片", file=sys.stderr)
        return 2
    grid = parse_grid(args.grid)
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]

    os.makedirs(args.out, exist_ok=True)
    print("v3 参数搜索：%d 张图 × %d 组参数 = %d 次烘焙（step=%d，扩散=%s）"
          % (len(files), len(combos), len(files) * len(combos), args.step,
             "开" if args.with_diffusion else "关"))

    rows = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        for ci, combo in enumerate(combos):
            tag = "_".join("%s%s" % (k, combo[k]) for k in keys).replace(".", "")
            out_dir = os.path.join(args.out, name, tag)
            cfg = {"step": args.step, "device": args.device,
                   "no_diffusion": not args.with_diffusion,
                   "pano_steps": args.pano_steps, "jpeg": 85}
            cfg.update(combo)
            t0 = time.time()
            try:
                rep = v3bake.bake_one(f, out_dir, cfg)
            except Exception as e:
                print("  失败 %s %s：%s" % (name, tag, str(e)[:120]))
                continue
            v = rep["verify"]
            rows.append({
                "image": name, "params": combo, "tag": tag,
                "psnr_core": round(v.get("psnr_core", 0.0), 2),
                "psnr": round(v.get("psnr", 0.0), 2),
                "ssim": round(v.get("ssim", 0.0), 4),
                "coverage": round(v.get("coverage", 0.0), 4),
                "ms": int((time.time() - t0) * 1000),
            })
            print("  %-22s %-22s psnr_core %5.2f  ssim %.3f  cov %.3f  %.1fs"
                  % (name, tag, rows[-1]["psnr_core"], rows[-1]["ssim"],
                     rows[-1]["coverage"], (time.time() - t0)))

    if not rows:
        print("全部失败", file=sys.stderr)
        return 1

    # 每个参数组合的跨图表现（中位数比均值稳，单张崩掉不该拉偏结论）
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["tag"], []).append(r)
    summary = []
    for tag, rs in by_tag.items():
        summary.append({
            "tag": tag, "params": rs[0]["params"],
            "psnr_core_median": round(statistics.median(x["psnr_core"] for x in rs), 2),
            "psnr_core_min": round(min(x["psnr_core"] for x in rs), 2),
            "ssim_median": round(statistics.median(x["ssim"] for x in rs), 4),
            "coverage_median": round(statistics.median(x["coverage"] for x in rs), 4),
            "ms_median": int(statistics.median(x["ms"] for x in rs)),
            "n": len(rs),
        })
    summary.sort(key=lambda s: (-s["psnr_core_median"], -s["coverage_median"], s["ms_median"]))

    best = summary[0]
    common.json_dump(os.path.join(args.out, "sweep.json"),
                     {"files": files, "grid": grid, "rows": rows, "summary": summary})

    md = ["# v3 参数搜索报告", "",
          "- 图：%d 张（%s）" % (len(files), ", ".join(os.path.basename(f) for f in files)),
          "- 网格：%s" % grid,
          "- step=%d，扩散=%s" % (args.step, "开" if args.with_diffusion else "关"),
          "", "## 排名（按 psnr_core 中位数）", "",
          "| 参数 | psnr_core 中位 | 最差 | SSIM | 覆盖 | 耗时 |",
          "|---|---|---|---|---|---|"]
    for s in summary:
        md.append("| %s | %.2f | %.2f | %.4f | %.4f | %.1fs |" % (
            " ".join("%s=%s" % (k, v) for k, v in s["params"].items()),
            s["psnr_core_median"], s["psnr_core_min"], s["ssim_median"],
            s["coverage_median"], s["ms_median"] / 1000.0))
    md += ["", "## 逐图最优", "",
           "| 图 | 最优参数 | psnr_core | SSIM |", "|---|---|---|---|"]
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        rs = [r for r in rows if r["image"] == name]
        if not rs:
            continue
        b = max(rs, key=lambda r: (r["psnr_core"], r["coverage"]))
        md.append("| %s | %s | %.2f | %.4f |" % (
            name, " ".join("%s=%s" % (k, v) for k, v in b["params"].items()),
            b["psnr_core"], b["ssim"]))
    md += ["", "## 结论", "",
           "全局最优：`%s`（psnr_core 中位 %.2f）。" % (
               " ".join("%s=%s" % (k, v) for k, v in best["params"].items()),
               best["psnr_core_median"]), "",
           "注意：这一轮只评**视锥内保真度**（关扩散）。地形/地物/转身观感要另跑",
           "`node tools/multiview.mjs --glb-list <list> --plan v3 --views 32`。"]
    with open(os.path.join(args.out, "sweep.md"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(md) + "\n")

    print("\n全局最优：%s  psnr_core 中位 %.2f  SSIM %.4f"
          % (" ".join("%s=%s" % (k, v) for k, v in best["params"].items()),
             best["psnr_core_median"], best["ssim_median"]))
    print("报告：%s" % os.path.join(args.out, "sweep.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
