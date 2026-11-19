#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v4 入口：多张图片 / 一段视频 → 质量报告 → 世界（GLB）。

    # 只评估，不出世界（先看能不能用）
    python pipeline/v4/build.py --input 我的照片目录 --qc-only

    # 评估 + 出世界
    python pipeline/v4/build.py --input 视频.mp4 --out out/v4/worlds/myscene

判定 FAIL 时**默认不出世界**（避免拿一堆垃圾几何浪费下游时间）；要强行出加 --force。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> int:
    ap = argparse.ArgumentParser(description="多图/视频 → 质量报告 → 世界")
    ap.add_argument("--input", required=True, help="图片目录 或 视频文件")
    ap.add_argument("--out", default="out/v4/worlds/scene", help="输出目录")
    ap.add_argument("--qc-only", action="store_true", help="只出报告，不建世界")
    ap.add_argument("--force", action="store_true", help="判定 FAIL 也照样出世界")
    ap.add_argument("--max-side", type=int, default=1100, help="单帧最长边（越小越快）")
    ap.add_argument("--limit", type=int, default=40, help="最多用多少帧")
    ap.add_argument("--fps", type=float, default=1.5, help="视频抽帧频率")
    ap.add_argument("--focal-ratio", type=float, default=1.2,
                    help="f = ratio * max(w,h)；手机广角常见 1.1~1.25")
    ap.add_argument("--cells", type=int, default=200, help="高度场分辨率")
    ap.add_argument("--voxel", type=float, default=0.02, help="体素降采样尺寸（SfM 单位）")
    args = ap.parse_args()

    from v4 import fuse, ingest, qc, sfm

    t0 = time.monotonic()
    frames = ingest.load_inputs(args.input, max_side=args.max_side,
                                limit=args.limit, fps=args.fps, min_gap=0.0)
    if len(frames) < 2:
        print("至少需要 2 帧（现在 %d）" % len(frames))
        return 2

    print("[1/3] 多视图几何…")
    res = sfm.run(frames, focal_ratio=args.focal_ratio)
    rep = qc.summarize(res, len(frames))
    print("[2/3] 质量报告…")
    os.makedirs(args.out, exist_ok=True)
    qc.write(rep, args.out)
    print("      判定 %s  定位 %d/%d  重投影 %.2fpx  点 %d  视差 %.2f°"
          % (rep["verdict"], rep["localized"], rep["frames"],
             rep["reproj_rmse_px"], rep["points"], rep["parallax_median_deg"]))
    if rep["verdict"] == "FAIL" and not args.force:
        print("      报告：%s/qc.md（判定 FAIL，未生成世界；加 --force 可强行生成）"
              % args.out)
        for a in rep["advice"]:
            print("      · " + a)
        return 1
    if args.qc_only:
        print("      只评估（--qc-only），用时 %.1fs" % (time.monotonic() - t0))
        return 0

    if res.pts is None or len(res.pts) < 50:
        print("      稀疏点太少（%d），无法成世界" % (0 if res.pts is None else len(res.pts)))
        return 1

    print("[3/3] 拼世界…")
    pts, cols = fuse.voxel_downsample(res.pts, res.colors, args.voxel)
    world = fuse.to_world(pts, cols, cells=args.cells)
    rep2 = qc.summarize(res, len(frames), fill=world.fill)
    qc.write(rep2, args.out)
    glb = os.path.join(args.out, "world-v4.glb")
    info = fuse.to_glb(world, glb)
    if rep2["verdict"] == "FAIL" and not args.force:
        # 填充率这类指标只有建完之后才知道：如果不达标，就别把半成品留在目录里
        os.remove(glb)
        print("      拼完后判定降为 FAIL（填充率 %.0f%%），已删除 %s"
              % (world.fill * 100, os.path.basename(glb)))
        for a in rep2["advice"]:
            print("      · " + a)
        return 1
    print("      %s  顶点 %d  三角 %d  覆盖 %.0f%%  用时 %.1fs"
          % (glb, len(world.pos), len(world.idx),
             world.fill * 100, time.monotonic() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
