#!/usr/bin/env python3
"""v2 烘焙：一张照片 → 分层 GLB（可直接喂前端 ?glb=）。

    python pipeline/v2/bake.py corpus_test/03_snowrange.jpg --out out/v2/03
    python pipeline/v2/bake.py "corpus_test/*.jpg" --out out/v2 --batch

与 v1（pipeline/reconstruct.py）的差别，只有三处是结构性的：

  1. 深度前端换成 GPU 高分辨率 + 引导上采样（pipeline/v2/depth_gpu.py）；
  2. 几何从"一张连续薄片"换成 LDI 分层 + 裙边（pipeline/v2/ldi.py）；
  3. 地面用几何判据压平，不再靠画面行号。

**其余（EXIF 视场、俯仰反解、地面双锚点标定、断崖判据、GLB 写出）全部复用 v1**：
那部分是踩过坑的，重写没有收益。
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
ROOT = os.path.dirname(PIPE)
sys.path.insert(0, PIPE)
sys.path.insert(0, HERE)

import geom                      # noqa: E402  v1 已验证的标定/相机
import glb as glb_mod            # noqa: E402
from exif import hfov_from_image, vfov_of   # noqa: E402
import ldi                       # noqa: E402
import calib                     # noqa: E402


def load_photo(path: str, max_side: int):
    # P1（v2.15.8）：按 EXIF 摆正（手机竖拍 orientation=6/8），
    # 否则深度模型看到的是躺倒的图，建出来的世界横竖颠倒。
    _raw = Image.open(path)
    _stored_size = _raw.size
    # P1（本轮修）：transpose 之前读 Orientation；竖拍 FOV 要按传感器短边换算。
    _ori = 1
    try:
        _ori = int(_raw.getexif().get(0x0112, 1))
    except Exception:
        pass
    _rotated90 = _ori in (5, 6, 7, 8)
    from PIL import ImageOps
    img = ImageOps.exif_transpose(_raw).convert("RGB")
    src = img.size
    hfov, fov_src, meta = hfov_from_image(img, sensor_size=_stored_size,
                                          rotated90=_rotated90)
    # 缩放按显示朝向（transpose 之后）的宽高，竖拍才不会被压成横图。
    if max(src) > max_side:
        s = max_side / float(max(src))
        img = img.resize((max(1, round(src[0] * s)), max(1, round(src[1] * s))), Image.LANCZOS)
    return img, hfov, fov_src, meta, src


def depth_for(img: Image.Image, cache_dir: str, name: str, args, dg, model_bundle):
    """优先读缓存（同一张图反复调几何参数时不必重跑 GPU）。"""
    key = f"{name}_{max(img.size)}.npy"
    p = os.path.join(cache_dir, key)
    if os.path.exists(p) and not args.force_depth:
        return np.load(p), True, 0.0
    proc, model = model_bundle
    t0 = time.time()
    est = dg.estimate(img, proc, model, args.device, args.long_side, not args.no_guide)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(p, est)
    return est, False, time.time() - t0


def sky_heuristic(z: np.ndarray, rays: np.ndarray, far: float, rgb: np.ndarray) -> np.ndarray:
    """兜底天空掩码：远端 + 地平线以上 + 低细节。

    没有语义模型时也能工作，但会把"很远的山"误判成天空 —— 所以只在没有
    SegFormer 时使用，且判据里要求"在最远那一档"（z ≥ 0.97·far）。
    """
    ry = rays[..., 1]
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gray = lum
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    flat = (gx + gy) < 0.02
    return (z >= 0.97 * far) & (ry > 0.02) & flat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="out/v2")
    ap.add_argument("--cache", default="cache/depth")
    ap.add_argument("--work-side", type=int, default=1536, help="几何工作分辨率（长边）")
    ap.add_argument("--tex-max", type=int, default=2048)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--eye", type=float, default=1.7)
    ap.add_argument("--max-tris", type=int, default=520000)
    ap.add_argument("--edge-ratio", type=float, default=2.4)
    ap.add_argument("--far-split", type=float, default=70.0, help="背景层分界（米）")
    ap.add_argument("--near-split", type=float, default=0.0, help="近层分界（米），0=自适应")
    ap.add_argument("--horizon-far", type=float, default=1500.0)
    ap.add_argument("--no-flatten", action="store_true")
    ap.add_argument("--strong-median", type=int, default=7)
    ap.add_argument("--quantize", type=float, default=0.0, help="深度量化倍数，0=关（枝叶类内容可试 1.3）")
    ap.add_argument("--repr", default="auto", choices=["auto", "mesh", "points"],
                    help="表示方式：auto 按散布度自动选，mesh 强制网格，points 强制点云")
    ap.add_argument("--scatter-thr", type=float, default=0.12,
                    help="散布度阈值（auto 模式下超过就用点云）")
    ap.add_argument("--stretch-clamp", type=float, default=12.0,
                    help="纹理拉伸超过该倍数就用更强中值平滑深度（0=关）")
    ap.add_argument("--no-skirt", action="store_true", help="关掉裙边（A/B 用）")
    ap.add_argument("--no-guide", action="store_true")
    ap.add_argument("--force-depth", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--long-side", type=int, default=1540)
    ap.add_argument("--depth-model", default="", help="本地权重目录或 HF id，空=用默认")
    ap.add_argument("--save-layers", action="store_true", help="额外存分层可视化")
    ap.add_argument("--debug-layers", action="store_true", help="每层单独导一个 GLB，用于定位缺陷在哪层")
    args = ap.parse_args()

    import importlib
    dg = importlib.import_module("depth_gpu")
    model_id = args.depth_model or os.path.join(ROOT, "models", "da2_large")
    if not (os.path.isdir(model_id) or "/" in model_id):
        model_id = dg.MODEL_ID

    proc, model = dg.load_model(model_id, args.device)
    bundle = (proc, model)

    jobs = []
    import glob as globmod
    for pat in args.images:
        jobs.extend(sorted(globmod.glob(pat)) if any(c in pat for c in "*?[") else [pat])
    jobs = [j for j in jobs if os.path.exists(j)]
    if not jobs:
        print("没有输入图片", file=sys.stderr)
        return 2

    rows = []
    for j in jobs:
        t0 = time.time()
        name = os.path.splitext(os.path.basename(j))[0]
        print(f"\n=== {name} ===")
        work, hfov, fov_src, meta, src = load_photo(j, args.work_side)
        tw, th = work.size
        vfov = vfov_of(hfov, tw, th)
        cam = geom.Camera(hfov=hfov, vfov=vfov, w=tw, h=th)
        print(f"  工作 {tw}x{th}  hfov={math.degrees(hfov):.1f}°({fov_src})  原图 {src[0]}x{src[1]}")

        disp, cached, dms = depth_for(work, args.cache, name, args, dg, bundle)
        if disp.shape != (th, tw):
            disp = np.asarray(Image.fromarray((disp * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR),
                              dtype=np.float32) / 255.0
        print(f"  深度 {disp.shape}  {'缓存' if cached else f'{dms*1000:.0f}ms'}")

        pf = None
        rb = calib.robust_calibrate(disp, cam, eye=args.eye,
                                    horizon_far=args.horizon_far)
        cam = rb["cam"]
        z, cal = rb["z"], rb["cal"]
        print(f"  标定 {cal['mode_used']}  A={cal['A']:.2f} p={cal['p']:.2f} "
              f"俯仰 {cal['pitch_note']}  z p10/p50/p90="
              f"{np.percentile(z,10):.1f}/{np.percentile(z,50):.1f}/{np.percentile(z,90):.1f}m")

        rgb = np.asarray(work, dtype=np.float32) / 255.0
        rays = cam.rays()
        sky = sky_heuristic(z, rays, cal["far"], rgb)

        step = 1
        while (tw // step) * (th // step) * 2 > args.max_tris:
            step += 1
        near_split = args.near_split
        if near_split <= 0:
            fwd_all = z[np.isfinite(z)]
            near_split = float(np.clip(np.percentile(fwd_all, 40), 3.0, 30.0))

        res = ldi.build(z, cam, step=step, eye=args.eye,
                        far_split=args.far_split, near_split=near_split,
                        sky=sky, edge_ratio=args.edge_ratio,
                        flatten=not args.no_flatten,
                        skirt_min=(0.0 if args.no_skirt else 0.12), skirt_max=(0.0 if args.no_skirt else 6.0),
                        stretch_clamp=args.stretch_clamp, strong_median=args.strong_median,
                        quantize=args.quantize, rgb=rgb,
                        repr_mode=args.repr, scatter_thr=args.scatter_thr)
        g = res["stats"]["_global"]
        print(f"  网格 step={g['step']} 三角 {g['triangles']} 顶点 {g['vertices']} "
              f"断崖 {g['cliff_frac']*100:.1f}% 地面 {g['ground_frac']*100:.1f}% 天空 {sky.mean()*100:.1f}%")
        for k, v in res["stats"].items():
            if k.startswith("_"):
                continue
            print(f"    {k:9} 像素 {v['pixels']:7d} 三角 {v['triangles']:7d} "
                  f"裙 {v['skirt_quads']:5d} 深度 {v['depth_p10']:.1f}~{v['depth_p90']:.1f}m")

        # ---- 贴图 & 写出 ----
        tex = Image.open(j).convert("RGB")
        if max(tex.size) > args.tex_max:
            s = args.tex_max / float(max(tex.size))
            tex = tex.resize((round(tex.size[0] * s), round(tex.size[1] * s)), Image.LANCZOS)
        buf = io.BytesIO()
        tex.save(buf, format="JPEG", quality=args.jpeg_quality, optimize=True)

        out_dir = os.path.join(args.out, name) if len(jobs) > 1 else args.out
        os.makedirs(out_dir, exist_ok=True)
        glb_path = os.path.join(out_dir, "world.glb")
        cam_extras = {"camera": {
            "hfov": cam.hfov, "vfov": cam.vfov, "w": tw, "h": th,
            "pitch": cam.pitch, "eyeHeight": args.eye,
            "pointPx": (res["stats"].get("splat") or {}).get("point_px", 0.0),
            "yaw": 0.0, "generator": "photoworld-v2",
        }}
        info = glb_mod.write_glb(glb_path, res["layers"], buf.getvalue(),
                                 name="photoworld", extras=cam_extras)
        if args.debug_layers:
            for L in res["layers"]:
                glb_mod.write_glb(os.path.join(out_dir, f"only_{L['name']}.glb"), [L],
                                  buf.getvalue(), name="debug")
            print("  调试层：" + ", ".join(L["name"] for L in res["layers"]))
        walk = geom.build_walkable(np.concatenate([l["pos"] for l in res["layers"]]),
                                   span=60.0, gh=64, gw=64)
        elapsed = time.time() - t0
        manifest = {
            "version": "v2", "source": os.path.basename(j),
            "source_size": list(src), "work_size": [tw, th], "texture_size": list(tex.size),
            "depth": {"model": dg.MODEL_ID, "long_side": args.long_side,
                      "cached": cached, "guided": not args.no_guide},
            "camera": {"hfov_deg": round(math.degrees(cam.hfov), 3),
                       "vfov_deg": round(math.degrees(cam.vfov), 3),
                       "pitch_deg": round(math.degrees(cam.pitch), 3),
                       "fov_source": fov_src},
            "calibration": {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                            for k, v in cal.items()},
            "layers": res["stats"], "glb": info,
            "walkable_empty_cells": walk["empty_cells"],
            "elapsed_s": round(elapsed, 2),
        }
        with open(os.path.join(out_dir, "world.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        print(f"  → {glb_path}  {info['bytes']/1e6:.2f}MB  {info['layers']} 层 "
              f"{info['triangles']} 三角  {elapsed:.1f}s")
        rows.append({"name": name, "glb": glb_path, "mb": round(info["bytes"] / 1e6, 2),
                     "tris": info["triangles"], "layers": info["layers"],
                     "elapsed_s": round(elapsed, 2)})
    if len(rows) > 1:
        with open(os.path.join(args.out, "_batch.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
