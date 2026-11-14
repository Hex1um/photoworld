#!/usr/bin/env python3
"""photoworld 重建管线

  python pipeline/reconstruct.py photo.jpg                  → out/world.glb
  python pipeline/reconstruct.py a.jpg b.jpg --backend vggt → 双图（需 GPU 依赖）
  python pipeline/reconstruct.py photo.jpg --target-fov 120 --max-tris 400000

产物：
  out/world.glb     几何 + 贴图（PBR，可直接被 three.js / 前端加载）
  out/world.json    元数据（相机内参、尺度标定、包围盒、可走高度网格、验收指标）

设计约定：
  * 尺度来自"地面平面标定"，不使用任何经验曲线；
  * 照片等比缩放 + padding，绝不拉伸；
  * 视场外按边界深度连续延拓到目标 FOV，不做镜像平铺；
  * 跨深度断崖的四边形直接跳过，保持轮廓锐利。
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import depth as depth_mod          # noqa: E402
import eval as eval_mod            # noqa: E402
import geom                        # noqa: E402
import glb                         # noqa: E402
from exif import hfov_from_image, vfov_of   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "models", "model_quantized.onnx")


def load_photo(path: str, max_side: int = 1600) -> tuple[Image.Image, np.ndarray, str, dict]:
    img = Image.open(path).convert("RGB")
    src_w, src_h = img.size
    hfov, src, meta = hfov_from_image(img)
    if max(src_w, src_h) > max_side:          # 等比缩放，保持宽高比
        s = max_side / float(max(src_w, src_h))
        img = img.resize((max(1, round(src_w * s)), max(1, round(src_h * s))), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    meta["source_size"] = [src_w, src_h]
    meta["used_size"] = list(img.size)
    return img, arr, src, {"fov_source": src, **meta}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--out", default="out")
    ap.add_argument("--backend", default="onnx", choices=["onnx", "depthpro", "vggt"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--eye", type=float, default=1.7, help="相机高度（米）")
    ap.add_argument("--fov", type=float, default=0, help="覆盖 EXIF 的水平视场角（度）")
    ap.add_argument("--target-fov", type=float, default=120.0, help="延拓到的水平视场角（度）")
    ap.add_argument("--max-tris", type=int, default=480_000)
    ap.add_argument("--edge-ratio", type=float, default=1.6, help="多大深度比视为断崖")
    ap.add_argument("--split", type=float, default=70.0,
                    help="近景/远景分层距离（米）：远景做成无视差背景层")
    ap.add_argument("--horizon-far", type=float, default=1500.0,
                    help="地平线先验距离（米），0=只用地面带拟合")
    ap.add_argument("--far", type=float, default=0.0,
                    help="度量深度上限（米）；0=自动取 1.2×horizon-far（或 600）")
    ap.add_argument("--verify-fov", type=float, default=0.0,
                    help="独立测量得到的水平视场角（度），用于交叉验收；未提供时只做尺度门")
    ap.add_argument("--tex-max", type=int, default=2048)
    ap.add_argument("--jpeg-quality", type=int, default=88)
    args = ap.parse_args()

    t0 = time.time()
    jobs = [os.path.abspath(p) for p in args.photos]
    for p in jobs:
        if not os.path.exists(p):
            print("找不到照片：" + p, file=sys.stderr)
            return 2
    if len(jobs) > 2:
        print("最多支持两张照片", file=sys.stderr)
        return 2

    img, arr, fov_src, meta = load_photo(jobs[0])
    if args.fov > 0:
        hfov = math.radians(args.fov)
        meta["fov_source"] = "cli"
    else:
        hfov, _, m = hfov_from_image(img)
        meta = {**meta, **m}
    vfov = vfov_of(hfov, img.size[0], img.size[1])
    cam = geom.Camera(hfov=hfov, vfov=vfov, w=img.size[0], h=img.size[1])
    print(f"[1/5] 输入 {img.size[0]}x{img.size[1]}  hfov={math.degrees(hfov):.1f}°  ({fov_src})")

    print(f"[2/5] 深度估计（{args.backend}）…")
    backend = depth_mod.build_backend(args.backend, args.model)
    res = backend(img)
    print(f"      {res.backend}  metric={res.metric}  {time.time() - t0:.1f}s")

    print("[3/5] 尺度标定（近处地面带锚定，z = a / r）…")
    cal = geom.calibrate_scale(res.depth, cam, eye=args.eye, horizon_far=args.horizon_far)
    # 上限必须与远端锚点联动：旧实现用 far=600 配 horizon_far=1500，
    # 实测 46.1% 像素被压到同一平面。软饱和 r0 让 r→0 时收敛到 far 而不发散。
    far_m = args.far if args.far > 0 else (1.2 * args.horizon_far if args.horizon_far > 0 else 600.0)
    pv = float(cal["p"])
    r_near = max(float(cal.get("r_near", 0.8)), 1e-3)
    # 联合求解 r0：metric_depth 收到的是 a_eff = A·(1+r0/r_near)^p，
    # 于是 z(0) = A·(1/r0 + 1/r_near)^p。令 z(0) = far 得闭式解
    #   r0 = 1 / ((far/A)^(1/p) - 1/r_near)
    # 旧解 r0=(A/far)^(1/p) 用的是未修正的 A：实测 z(0)=4137m 而 far=1800m
    # （2.298×），天空仍被 clip 成一块硬平面。分母非正时退回旧解，不发散。
    denom = ((far_m / cal["A"]) ** (1.0 / pv) - 1.0 / r_near
             if pv > 0 and cal["A"] > 0 else 0.0)
    r0 = (1.0 / denom if denom > 1e-6 else
          (float((cal["A"] / far_m) ** (1.0 / pv))
           if pv > 0 and cal["A"] > 0 else 0.0))
    a_eff = cal["A"] * (1.0 + r0 / r_near) ** pv   # 保持 z(r_near) = z_near
    z_m = geom.metric_depth(res.depth, a_eff, cal.get("b", 0.0),
                            p=pv, far=far_m, r0=r0)
    ground_rows = z_m[int(z_m.shape[0] * 0.95):]
    touched = float((z_m >= far_m - 0.5).mean() * 100)
    print(f"      模式={cal.get('mode', '-')} A={a_eff:.2f}(标定 {cal['A']:.2f}) p={pv:.2f} "
          f"上限={far_m:.0f}m r0={r0:.4f} 触顶={touched:.1f}% "
          f"近地中位距={float(np.median(ground_rows)):.2f}m "
          f"{'（退化：地平线假设可能不成立）' if cal.get('degenerate') else ''}")
    if pv > 2.0:
        print(f"      警告：p={pv:.2f} > 2，深度 10% 误差会放大成距离 "
              f"{(0.9 ** -pv - 1) * 100:.1f}% 误差；建议用度量深度后端或调小 horizon-far")
    cal["far"] = far_m
    cal["r0"] = r0
    cal["a_effective"] = a_eff

    print(f"[4/5] 视场外延拓 → {args.target_fov:.0f}° …")
    img2, z2, cam2 = geom.extend_fov(arr, z_m, cam, target_hfov=math.radians(args.target_fov))
    print(f"      {img2.shape[1]}x{img2.shape[0]}")

    print("[5/5] 建网格 + 导出 …")
    mesh = geom.build_mesh(z2, cam2, edge_ratio=args.edge_ratio, max_tris=args.max_tris)
    print(f"      顶点 {mesh['pos'].shape[0]}  三角 {mesh['index'].size // 3}  "
          f"跳过的断崖格 {mesh['skipped'] * 100:.1f}%")

    os.makedirs(args.out, exist_ok=True)
    tex_img = Image.fromarray((np.clip(img2, 0, 1) * 255).astype(np.uint8))
    if max(tex_img.size) > args.tex_max:
        s = args.tex_max / float(max(tex_img.size))
        tex_img = tex_img.resize((round(tex_img.size[0] * s), round(tex_img.size[1] * s)),
                                 Image.LANCZOS)
    buf = io.BytesIO()
    tex_img.save(buf, format="JPEG", quality=args.jpeg_quality, optimize=True)
    tex_bytes = buf.getvalue()

    hh, ww, mstep = mesh["grid"]
    tex_f = np.asarray(tex_img, dtype=np.float32) / 255.0
    ref_u8 = (np.clip(img2, 0, 1) * 255).astype(np.uint8)

    # 独立来源的相机内参：只有显式给 --verify-fov，或 CLI 覆盖了 EXIF
    # （此时 EXIF 就是独立来源）时才存在。没有独立信息就不声称能验视场。
    cam_verify = None
    hfov_ref = None
    if args.verify_fov > 0:
        hfov_ref = math.radians(args.verify_fov)
    elif args.fov > 0:
        hfov_ref, _, _ = hfov_from_image(img)
    if hfov_ref is not None:
        vfov_ref = vfov_of(hfov_ref, img.size[0], img.size[1])
        cam_verify = geom.Camera(hfov=cam2.hfov, vfov=vfov_ref,
                                 w=cam2.w, h=cam2.h, pitch=cam.pitch)

    chk = eval_mod.report(mesh["pos"], mesh["uv"], tex_f, cam2, ref_u8,
                          step=mstep, eye=args.eye, cam_verify=cam_verify, cal=cal)
    print(f"      自检：覆盖 {chk['coverage'] * 100:.1f}%  PSNR {chk['psnr']}dB  "
          f"SSIM {chk['ssim']}  {'OK' if chk['ok'] else '注意：' + chk['reason']}")
    sc = chk.get("scale")
    if sc:
        print(f"      尺度门：地面 y 中位 {sc['ground_y_median']}m / 期望 {sc['expected']}m  "
              f"相对误差 {sc['rel_error']}  {'OK' if sc['ok'] else '注意：' + sc['reason']}")
    cr = chk.get("cross")
    if cr:
        print(f"      交叉门：覆盖 {cr['coverage'] * 100:.1f}%  PSNR {cr['psnr']}dB  "
              f"SSIM {cr['ssim']}  {'OK' if cr['ok'] else '注意：相机内参不一致'}")
    warn = chk.get("calibration", {}).get("warning")
    if warn:
        print(f"      标定告警：{warn}")

    # 分层：近景（真几何，走动正确）+ 远景（无视差背景层，避免"沙盘感"）
    pos_m, nrm_m, uv_m, idx_m = mesh["pos"], mesh["nrm"], mesh["uv"], mesh["index"]
    fwd = -pos_m[:, 2]
    tri = idx_m.reshape(-1, 3)
    cz = (fwd[tri[:, 0]] + fwd[tri[:, 1]] + fwd[tri[:, 2]]) / 3.0
    near_mask = cz < args.split

    def submesh(mask):
        t = tri[mask]
        if t.size == 0:
            return None
        flat = t.reshape(-1)
        used, inv = np.unique(flat, return_inverse=True)
        return {"pos": pos_m[used], "nrm": nrm_m[used], "uv": uv_m[used],
                "index": inv.astype(np.uint32)}

    layers = []
    near = submesh(near_mask)
    far = submesh(~near_mask)
    if near is not None:
        near["name"] = "near"
        layers.append(near)
    if far is not None:
        far["name"] = "backdrop"
        layers.append(far)

    out_glb = os.path.join(args.out, "world.glb")
    info = glb.write_glb(out_glb, layers, tex_bytes, name="photoworld")
    print(f"      {out_glb}  {info['bytes'] / 1e6:.1f}MB  {info['layers']} 层 "
          f"({info['vertices']} 顶点 / {info['triangles']} 三角)")
    print(f"      近景 {int(near_mask.sum())} 三角 / 远景 {int((~near_mask).sum())} 三角"
          f"（分界 {args.split:.0f}m）")

    # 可走区域高度网格（前端碰撞用）：在相机前方 60m×60m 上采样表面高度
    gh, gw = 64, 64
    pos = mesh["pos"]
    ext = np.abs(pos[:, 0]).max()
    grid = np.full((gh, gw), np.nan, dtype=np.float32)
    span = 60.0
    gx = (pos[:, 0] + span * 0.5) / span * (gw - 1)
    fwd = -pos[:, 2]                            # 朝前的距离（z 为负）
    gz = fwd / span * (gh - 1)
    ok = (pos[:, 0] > -span * 0.5) & (pos[:, 0] < span * 0.5) & (fwd > 0.2) & (fwd < span)
    ix = np.clip(gx[ok].astype(np.int32), 0, gw - 1)
    iz = np.clip(gz[ok].astype(np.int32), 0, gh - 1)
    for a, b, y in zip(ix, iz, pos[ok, 1]):
        if np.isnan(grid[b, a]):
            grid[b, a] = y
    grid = np.nan_to_num(grid, nan=float(np.nanmedian(grid)))

    meta_out = {
        "world": "photoworld",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": [os.path.basename(p) for p in jobs],
        "backend": res.backend,
        # ONNX 是相对深度 + 几何标定，不能把标定后的米写成"度量深度后端"
        "metric": bool(res.metric),
        "depth_kind": "metric" if res.metric else "relative-calibrated",
        "camera": {"hfov": hfov, "vfov": vfov, "w": img.size[0], "h": img.size[1],
                   "eye_height": args.eye, "fov_source": meta.get("fov_source")},
        "extended": {"hfov": cam2.hfov, "w": cam2.w, "h": cam2.h},
        "calibration": {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                        for k, v in cal.items()},
        "bbox": {"min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist()},
        "mesh": {"vertices": info["vertices"], "triangles": info["triangles"],
                 "layers": info.get("layers", 1),
                 "near_triangles": int(near_mask.sum()),
                 "split_m": args.split,
                 "skipped_quads": round(mesh["skipped"], 4)},
        "check": chk,
        "walkable": {"span": span, "grid": gh, "height": grid.round(3).tolist()},
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "world.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=1)
    print(f"完成，用时 {meta_out['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
