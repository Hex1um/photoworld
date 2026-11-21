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
    # 默认用 heightfield：它在"素材一般"时更稳（tsdf 在稀疏输入上会出大块黑面）。
    # 素材好（稠密、位姿准）时用 --mesh tsdf 才有垂直面。
    ap.add_argument("--dense", action="store_true",
                    help="用 COLMAP patch_match_stereo 做真多视图立体（替代单目深度反投影）")
    # 默认仍是 ours：COLMAP 一旦可用就"悄悄换实现"，会让既有回归（和用户的历史结果）
    # 突然变样。要更好的效果显式加 --poses colmap（前端服务会自动加）。
    ap.add_argument("--poses", choices=["auto", "ours", "colmap"], default="ours",
                    help="位姿来源：ours=手写 SfM（默认）；colmap=用 COLMAP（更准，需装）；auto=有就用")
    ap.add_argument("--splat", action="store_true",
                    help="额外训练一个可微泼溅（3DGS-lite），并从插值位姿渲新视角")
    ap.add_argument("--splat-iters", type=int, default=600)
    ap.add_argument("--mesh", choices=["heightfield", "tsdf", "poisson"], default="heightfield",
                    help="世界表示：heightfield=高度场（默认，适合地形）；"
                         "tsdf=体素融合（物体/室内推荐，薄壳）；"
                         "poisson=泊松（实验：MVS 法线不一致时会拉丝）")
    ap.add_argument("--poisson-depth", type=int, default=9)
    args = ap.parse_args()

    from v4 import fuse, ingest, qc, sfm

    t0 = time.monotonic()
    frames = ingest.load_inputs(args.input, max_side=args.max_side,
                                limit=args.limit, fps=args.fps, min_gap=0.0)
    if len(frames) < 2:
        print("至少需要 2 帧（现在 %d）" % len(frames))
        return 2

    print("[1/3] 多视图几何…")
    res = None
    use_colmap = args.poses == "colmap"
    if args.poses in ("auto", "colmap"):
        from v4 import colmap as v4colmap
        if v4colmap.available():
            use_colmap = True
        elif args.poses == "colmap":
            raise SystemExit("找不到 colmap 可执行文件：设 COLMAP_EXE，或把 COLMAP 放到 tools/colmap/")
    if use_colmap:
        import types
        from PIL import Image as _Im
        from v4 import colmap as v4colmap
        img_dir = os.path.join(args.out, "colmap", "images")
        os.makedirs(img_dir, exist_ok=True)
        for f in frames:
            _Im.fromarray(f.rgb).save(os.path.join(img_dir, f.name if hasattr(f, "name") else "img.jpg"))
        model = v4colmap.run(img_dir, os.path.join(args.out, "colmap"),
                             matcher="exhaustive" if len(frames) <= 60 else "sequential")
        cams, poses, pts, cols = v4colmap.load_model(model, frames)
        dense_normals = None
        if args.dense:
            ply = v4colmap.run_dense(img_dir, os.path.join(args.out, "colmap"), model)
            if ply is not None:
                got = v4colmap.read_ply(ply)
                dp, dc = got[0], got[1]
                dn = got[2] if len(got) > 2 else None
                print("      稠密融合：%d 点（稀疏 %d）" % (len(dp), len(pts)))
                if len(dp) > len(pts):
                    pts, cols = dp, dc
                    dense_normals = dn
        res = types.SimpleNamespace(cams=cams, poses=poses, pts=pts, colors=cols,
                                    pairs=[], reproj_rmse=0.0, localized=sum(1 for p in poses if p is not None),
                                    notes=["位姿来自 COLMAP"], mode="colmap", parallax_deg=0.0)
        res.normals = dense_normals
        _, par = qc.cloud_stats(frames, res)
        res.parallax_deg = 0.0 if par != par else par
    else:
        res = sfm.run(frames, focal_ratio=args.focal_ratio)
    photo = qc.photometric(frames, res)
    rep = qc.summarize(res, len(frames), photo_l1=None if not np.isfinite(photo) else photo)
    print("[2/3] 质量报告…")
    os.makedirs(args.out, exist_ok=True)
    qc.write(rep, args.out)
    print("      判定 %s  定位 %d/%d  重投影 %.2fpx  点 %d  视差 %.2f°  光度 %.3f"
          % (rep["verdict"], rep["localized"], rep["frames"],
             rep["reproj_rmse_px"], rep["points"], rep["parallax_median_deg"],
             -1 if rep["photo_l1"] is None else rep["photo_l1"]))
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

    if args.splat:
        from PIL import Image
        from v4 import splat as v4splat
        print("[3/4] 训练可微泼溅（3DGS-lite）…")
        W, H = 320, 200
        from v4 import tsdf as v4tsdf
        dp, dc = v4tsdf.dense_points(frames, res, step=4, max_points=120000)
        print("      稠密初始化：%d 点（SfM 稀疏点 %d）" % (len(dp), len(res.pts)))
        spl, hist = v4splat.train(frames, res, iters=args.splat_iters, W=W, H=H,
                                 verbose=False, init=(dp, dc) if len(dp) > 1000 else None)
        if spl is not None:
            l80 = hist[int(len(hist) * 0.8)] if len(hist) > 5 else hist[0]
            print("      loss %.4f → %.4f（末 20%% 均值 %.4f）"
                  % (hist[0], hist[-1], float(np.mean(hist[int(len(hist) * 0.8):]))))
            k0 = res.cams[0].K
            sx, sy = W / res.cams[0].w, H / res.cams[0].h
            import torch as _t
            K = _t.tensor([[k0[0, 0] * sx, 0, k0[0, 2] * sx],
                           [0, k0[1, 1] * sy, k0[1, 2] * sy], [0, 0, 1]],
                          dtype=_t.float32, device="cuda")
            # 新视角：相邻两个已定位帧之间插值（位置线性、旋转取反后线性再正交化）
            valid = [i for i, p in enumerate(res.poses) if p is not None]
            os.makedirs(os.path.join(args.out, "splat"), exist_ok=True)
            made = 0
            for k in range(min(3, max(1, len(valid) - 1))):
                ia, ib = valid[k], valid[k + 1]
                Ra, ta = res.poses[ia]; Rb, tb2 = res.poses[ib]
                ca = -np.asarray(Ra).T @ np.asarray(ta).ravel()
                cb = -np.asarray(Rb).T @ np.asarray(tb2).ravel()
                # 点云已被归一化（s, c），位姿也要跟着变
                cN = np.median(res.pts, axis=0).astype(np.float32)
                span = float(np.percentile(np.linalg.norm(res.pts - cN, axis=1), 90)) or 1.0
                sc = 1.6 / span
                ca = (ca - cN) * sc; cb = (cb - cN) * sc
                mid = (ca + cb) * 0.5
                # 新相机要**看向场景中心**（归一化后的原点），而不是沿着基线看——
                # 沿基线看会拍到空处，渲出来只有背景色（踩过）。
                fwd = -mid
                n = np.linalg.norm(fwd)
                fwd = fwd / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
                up = np.array([0.0, 1.0, 0.0])
                right = np.cross(fwd, up); right /= max(1e-6, np.linalg.norm(right))
                up2 = np.cross(right, fwd)
                Rm = np.stack([right, up2, -fwd]).astype(np.float32)
                tm = -Rm @ mid
                img = v4splat.render_novel(spl, Rm, tm, K, W, H)
                p_out = os.path.join(args.out, "splat", "novel_%02d.png" % k)
                Image.fromarray(img).save(p_out)
                made += 1
            print("      新视角 %d 张 → %s/splat/" % (made, args.out))

    print("[3/3] 拼世界（%s）…" % args.mesh)
    if args.mesh == "poisson":
        try:
            from v4 import poisson as v4poisson
            mesh = v4poisson.reconstruct(res.pts, res.colors, depth=args.poisson_depth,
                                         normals=getattr(res, "normals", None))
        except Exception as e:
            print("      Poisson 失败（%s），回退 heightfield" % e)
            mesh = None
        if mesh is not None:
            glb = os.path.join(args.out, "world-v4.glb")
            v4poisson.to_glb(mesh, glb)
            print("      %s  顶点 %d  三角 %d  用时 %.1fs"
                  % (glb, len(mesh.pos), len(mesh.idx), time.monotonic() - t0))
            return 0
        args.mesh = "heightfield"
    if args.mesh == "tsdf":
        from v4 import tsdf as v4tsdf
        mesh = v4tsdf.fuse_tsdf(frames, res, cells=args.cells)
        if mesh is None:
            print("      体素融合没产出网格（深度/位姿不一致），回退到高度场")
            args.mesh = "heightfield"
        else:
            glb = os.path.join(args.out, "world-v4.glb")
            info = v4tsdf.to_glb(mesh, glb)
            print("      %s  顶点 %d  三角 %d  用时 %.1fs"
                  % (glb, len(mesh.pos), len(mesh.idx), time.monotonic() - t0))
            return 0

    pts, cols = fuse.voxel_downsample(res.pts, res.colors, args.voxel)
    world = fuse.to_world(pts, cols, cells=args.cells)
    rep2 = qc.summarize(res, len(frames), fill=world.fill,
                        photo_l1=None if not np.isfinite(photo) else photo)
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
