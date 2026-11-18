#!/usr/bin/env python3
"""v3 主入口：一张照片 → 可漫游世界（GLB）+ 原点视角保真报告。

管线（顺序即依赖）：
  1. 深度：`depth.py`        照片 → 米
  2. 视锥内：`frustum.py`    每像素 3D → 双四边形连续曲面 + 天空壳
  3. 视锥外：`pano.py`       照片 → 360° 全景（SDXL 补全）
  4. 地形地物：`terrain.py`  全景 → 8 视角深度 → 极坐标高度场 + 树石
  5. 统一着色：`shade.py`    太阳/天光 + 接缝拉色
  6. 打包 + 验证：GLB + 原点 PSNR/SSIM + 侧后视图

用法：
  python pipeline/v3/bake.py worlds/alpine.jpg --out out/v3/alpine
  python pipeline/v3/bake.py "corpus_test/*.jpg" --out out/v3/all --step 4
  # 没有 GPU / 不想跑扩散时（只做视锥内，视锥外用程序化兜底）：
  python pipeline/v3/bake.py photo.jpg --out out/x --no-diffusion

产物：
  <out>/world.glb          给前端 `?glb=` 直接用
  <out>/world.json         参数、统计、验证指标（可追溯）
  <out>/pano.jpg           360° 全景（扩散产物，人工目检用）
  <out>/origin.png         原点视角的**光栅化结果**（不是照片）
  <out>/diff.png           渲染与原图的差异图
  <out>/views.png          侧/后/俯视联系表
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, os.path.join(_ROOT, "pipeline"), os.path.join(_ROOT, "pipeline", "v2"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from v3 import common, depth as v3depth, frustum, pano as panomod, terrain, shade, verify


def _contact_sheet(imgs: list[np.ndarray], cols: int = 3) -> np.ndarray:
    h, w = imgs[0].shape[:2]
    rows = (len(imgs) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.float32)
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return sheet


def bake_one(path: str, out_dir: str, cfg: dict) -> dict:
    t_all = time.time()
    os.makedirs(out_dir, exist_ok=True)
    rgb, meta = common.load_photo(path)
    hfov = float(np.clip(math.radians(meta["hfov_deg"]), math.radians(25), math.radians(120)))
    cam = common.Camera3(rgb.shape[1], rgb.shape[0], hfov, eye=float(cfg.get("eye", 1.7)))
    print(f"[1/6] 照片 {rgb.shape[1]}x{rgb.shape[0]}  hfov={math.degrees(hfov):.1f}° ({meta['fov_source']})")

    t = time.time()
    disp = v3depth.disparity(rgb, device=cfg.get("device", "cuda"),
                             long_side=int(cfg.get("long_side", 1540)))
    cal = v3depth.calibrate(disp, cam)
    z, cam = cal["z"], cal["cam"]
    print(f"[2/6] 深度 {time.time()-t:.1f}s  标定={cal['cal'].get('mode_used')} "
          f"pitch={math.degrees(cam.pitch):+.2f}° z={np.percentile(z,2):.1f}~{np.percentile(z,98):.0f}m")

    t = time.time()
    # 白条 UV 只跟照片高度有关（贴图底部固定 8px 白条），提前算好给 fill 层用
    v_scale0 = rgb.shape[0] / float(rgb.shape[0] + common.WHITE_STRIP_PX)
    cfg["white_v"] = (v_scale0 + 1.0) * 0.5
    cfg["white_u"] = 0.5

    # 背景板要在 frustum.build 之前算：fill 层要用它的颜色（几何仍用 push-pull 的深度）
    bg_plate = None
    if not cfg.get("no_diffusion") and cfg.get("bg_fill", True):
        t_bg = time.time()
        sky0 = frustum.sky_mask(rgb, z, cam)
        fg0 = frustum.foreground_mask(z, sky0)
        prompt0 = panomod._prompt_from_photo(rgb, cam)
        bg_plate = panomod.bg_plate(rgb, fg0, prompt0, cam, cfg)
        print(f"[2.5/6] 背景板（扩散）{time.time()-t_bg:.1f}s  前景 {fg0.mean()*100:.0f}%")

    fr = frustum.build(z, rgb, cam, {
        "step": int(cfg.get("step", 3)),
        "cliff": float(cfg.get("cliff", 1.8)),
        # 裙边回到 0.35：加厚（0.9）确实让侧看更"实"，但裙边是用边界像素的 UV 挤出来的，
        # 俯视时那一圈会变成**竖条拉丝**（投影纹理只会夹到照片边缘色）。0.35 足够挡住
        # 壳的背面，又不至于在俯视图里拉出一整片条纹。
        # 裙边收到 0.06：它沿射线挤出、UV 取边界像素，转身 180° 时那圈侧壁正对着
        # 相机 → 一片竖条拉丝（对比图"转身 180°"那格）。0.06 只够挡住壳的边缘。
        "skirt": float(cfg.get("skirt", 0.06)),
        # 下面三个默认值就是 v3 的推荐值；参数搜索（tools/v3_sweep.py）会覆盖它们
        "ground_tol": float(cfg.get("ground_tol", 0.40)),
        "sky_grad": float(cfg.get("sky_grad", 0.045)),
        "fg_near": float(cfg.get("fg_near", 70.0)),
        "skirt_bg": float(cfg.get("skirt_bg", 0.15)),
        "cliff_bg": float(cfg.get("cliff_bg", 6.0)),
        "white_u": cfg.get("white_u", 0.5),
        "white_v": cfg.get("white_v", 0.9959),
    }, bg_rgb=bg_plate)
    print(f"[3/6] 视锥内 {time.time()-t:.1f}s  {fr['stats']}")

    # 深度用完了：先把 1.3GB 的 DA-V2-Large 放掉，扩散才有足够的显存
    if not cfg.get("no_diffusion"):
        v3depth.unload()

    edge, _ = shade.edge_from_near(fr["layers"], bins=terrain.AZ_BINS)
    floor_y = shade.floor_from_near(fr["layers"], bins=terrain.AZ_BINS)
    # 地物原型：从照片的前景物体里量轮廓与配色（外面长的东西跟里面同一类）
    protos = terrain.extract_prototypes(rgb, fr["debug"]["fg"], max_n=3)
    edge_col = shade.ground_edge_color(rgb, bins=terrain.AZ_BINS)

    pcfg = {"pano_w": cfg.get("pano_w", 1024), "pano_h": cfg.get("pano_h", 512),
            "pano_steps": cfg.get("pano_steps", 20), "seed": cfg.get("seed", 0),
            "no_diffusion": cfg.get("no_diffusion", False),
            "prompt": cfg.get("prompt"), "pano_from": cfg.get("pano_from"),
            "cache_dir": cfg.get("cache_dir"),
            # 全景生成方式必须一起传下去：漏了它 build() 会走默认的单次 inpaint
            # （实测就是"传了 --pano-mode views 却打印 generated"）。
            "pano_w": int(cfg.get("pano_w", 1024)), "pano_h": int(cfg.get("pano_h", 512)),
            "pano_mode": cfg.get("pano_mode"), "pano_views": cfg.get("pano_views", 6),
            "pano_cfg": cfg.get("pano_cfg", 5.2)}
    # 逐视角高度偏差由 diffusion 分支里的 point_cloud(reconcile) 算出；
    # 必须在分支前给默认值，否则 report 里的 terrain.view_bias 恒为空。
    view_bias: dict = {}
    if cfg.get("no_diffusion"):
        print("[4/6] 视锥外：跳过扩散（--no-diffusion），全景=原图填入的已知区")
        pn = panomod.build(rgb, cam, pcfg)
        layers_far = []
    else:
        t = time.time()
        pn = panomod.build(rgb, cam, pcfg)
        print(f"[4/6] 全景 {time.time()-t:.1f}s（{pn.get('cache', '?')}）  "
              f"已知区 {pn['known_frac']*100:.0f}%  prompt={pn['prompt'][:56]}…")
        t = time.time()
        views = terrain.render_views(pn["pano"], n=int(cfg.get("views", 8)), size=int(cfg.get("view_size", 448)))
        cloud = terrain.point_cloud(views, device=cfg.get("device", "cuda"),
                                   long_side=int(cfg.get("view_size", 448)),
                                   reconcile=bool(cfg.get("view_reconcile", True)))
        view_bias = cloud.get("view_bias", {})
        field = terrain.polar_heightfield(cloud, pn["pano"], cam.eye, near_edge=edge)
        fov_half = cam.hfov * 0.5
        far = terrain.heightfield_mesh(field, cam.eye, edge, (0.5, 0.5),
                                       fov_half=fov_half,
                                       # floor_y=None：足迹内不再铺"平板地板"，改用**实测高度场 + 起伏**。
                                       # 原点视角由 conform_to_photo 把关（把落在照片表面之前的顶点
                                       # 沿射线推到表面之后），所以不需要一块死的平板来兜底；
                                       # 而侧视/背视能看到真正有起伏的地面，而不是一张水平的板。
                                       floor_y=None if cfg.get("no_floor", True) else floor_y,
                                       seed=int(cfg.get("seed", 0)),
                                       amp_far=float(cfg.get("relief_amp", 7.0)))
        prop = terrain.props(pn["pano"], field, edge, cam.eye, (0.5, 0.5),
                            seed=int(cfg.get("seed", 0)),
                            density=float(cfg.get("prop_density", 2.0)),
                            fov_half=fov_half, protos=protos)
        dome = terrain.sky_dome(pn["pano"], cam=cam)
        layers_far = [far] + ([prop] if prop else []) + [dome]
        print(f"[5/6] 地形/地物 {time.time()-t:.1f}s  点云 {cloud['pos'].shape[0]} "
              f"far {far['index'].size//3} 三角  prop {(prop['index'].size//3) if prop else 0} 三角  "
              f"dome {dome['index'].size//3} 三角")

    # ---- 视锥外几何与照片对齐（压到照片表面之后 / 剔掉落在天空的部分）----
    if layers_far:
        z_ref = fr["debug"]["z_snap"]
        sky_ref = fr["debug"]["sky"]
        layers_far = [terrain.conform_to_photo(l, z_ref, sky_ref, cam, photo=rgb)
                      for l in layers_far]
        print("       对齐照片：剔除 %d 顶点、推后 %d 顶点" % (
            sum(l.get("culled", 0) for l in layers_far),
            sum(l.get("pushed", 0) for l in layers_far)))

    # ---- 统一着色 ----
    light = shade.sun_from_pano(pn["pano"])
    # 天空壳只做很轻的明暗（天空本身不该有地形那种坡面明暗）
    # 天空壳完全不烘焙光照：它是"远处本身"，再打光只会让天空出现球面明暗
    # （实测：侧视里天空穹顶变成一个圆盘，边界肉眼可见）。
    layers_far = [shade.bake_lighting(l, light,
                                      strength=0.0 if l["name"] == "backdrop" else 0.85)
                  for l in layers_far]
    # 着色对齐视锥内：用照片**底缘地面色**当目标亮度，只做乘性亮度匹配（不动色相）。
    # 只对地形/地物做；天空壳用全景自己的渐变。
    ground_ref = rgb[int(rgb.shape[0] * 0.88):int(rgb.shape[0] * 0.99)].reshape(-1, 3).mean(axis=0)
    layers_far = [shade.match_to_photo(l, ground_ref) if l["name"] in ("far", "prop") else l
                  for l in layers_far]
    layers_far = [shade.seam_blend(l, edge, edge_col, band=float(cfg.get("seam_band", 28.0)))
                  if l.get("name") == "far" else l for l in layers_far]

    # ---- 打包 ----
    tex_jpeg, vscale = common.pack_texture(rgb, quality=int(cfg.get("jpeg", 90)))
    uv_white = common.white_uv(vscale)
    # far/prop 的 UV 必须指到白条上（贴图 × 顶点色 = 顶点色）
    for l in layers_far:
        l["uv"] = np.tile(np.array(uv_white, dtype=np.float32), (l["pos"].shape[0], 1))
    # near/fill/backdrop 的 UV 是照片口径，乘上 v 比例（避开白条）
    for l in fr["layers"]:
        l["uv"] = l["uv"].copy()
        l["uv"][:, 1] *= vscale

    layers = fr["layers"] + layers_far
    bbox = _bbox(fr["layers"])
    extras = {
        "camera": {**cam.fov_meta(), "v3": True, "hfov": cam.hfov, "vfov": cam.vfov,
                   "pitch": cam.pitch, "w": cam.w, "h": cam.h},
        "v3": {"pipeline": "v3", "sun": [float(v) for v in light["dir"]],
               "amb": [float(v) for v in light["amb"]],
               "bbox": bbox, "prompt": pn["prompt"],
               "white_uv": list(uv_white),
               "fog": [float(v) for v in np.clip(
                   rgb[int(rgb.shape[0] * 0.55):].reshape(-1, 3).mean(axis=0), 0, 1)]},
        "pointPx": 2.2,
    }
    glb_path = os.path.join(out_dir, "world.glb")
    info = common.write_world_glb(glb_path, layers, tex_jpeg, extras,
                                  name=os.path.basename(out_dir),
                                  compact=bool(cfg.get("compact", True)))

    # ---- 验证：真的从原点光栅化一遍 ----
    tex_full = _texture_array(rgb)
    all_layers = fr["layers"] + layers_far
    rend = verify.rasterize(all_layers, tex_full, cam, rgb.shape[1], rgb.shape[0])
    rep = verify.report(rend, rgb, cam)

    # **原图保真度**单独算一遍：只统计"照片自己定义过的像素"（near + 照片天空壳）。
    # 为什么分开：360° 穹顶会把照片 FOV 边缘那 0.4% 的像素也盖上（用扩散出来的
    # 天空），那些像素原来是无覆盖、不计入 PSNR 的 —— 混在一起算，穹顶一加
    # PSNR 就从 46dB 掉到 34dB，读起来像"退步"，其实是"补全了但只有近似值"。
    core = verify.rasterize([l for l in fr["layers"] if l["name"] != "fill"],
                            tex_full, cam, rgb.shape[1], rgb.shape[0])
    rep["psnr_core"] = verify.psnr(core["rgb"], rgb, core["coverage"])
    rep["coverage_core"] = float(core["coverage"].mean())
    rep["reproj"] = verify.reprojection(fr["layers"], cam, vscale)

    common.save_rgb(os.path.join(out_dir, "origin.png"), rend["rgb"])
    err = np.abs(rend["rgb"] - rgb).max(axis=2)
    common.save_gray(os.path.join(out_dir, "diff.png"), np.where(rend["coverage"], err, 0))
    common.save_rgb(os.path.join(out_dir, "pano.jpg"), pn["pano"], quality=88)

    # 侧/后/俯视联系表（人工目检用）
    views_img = []
    for yaw, pitch, eye_y, tag in [(math.pi / 2, 0.0, cam.eye, "left"),
                                   (math.pi, 0.0, cam.eye, "back"),
                                   (-math.pi / 2, 0.0, cam.eye, "right"),
                                   (math.pi, -0.45, cam.eye + 55.0, "air-back"),
                                   (0.0, 0.35, cam.eye, "up"),
                                   (math.pi * 0.75, 0.0, cam.eye, "back-left")]:
        r2 = verify.rasterize(fr["layers"] + layers_far, tex_full, cam,
                              rgb.shape[1], rgb.shape[0], yaw=yaw, pitch=pitch, eye_y=eye_y)
        views_img.append(r2["rgb"])
    if views_img:
        common.save_rgb(os.path.join(out_dir, "views.png"),
                        _contact_sheet(views_img, cols=3), quality=88)

    report = {
        "input": os.path.relpath(path, _ROOT),
        "photo": {"w": rgb.shape[1], "h": rgb.shape[0], "hfov_deg": round(math.degrees(hfov), 2),
                  "fov_source": meta["fov_source"], "pitch_deg": round(math.degrees(cam.pitch), 2)},
        "calib": cal["cal"],
        "frustum": fr["stats"],
        "pano": {"prompt": pn["prompt"], "known_frac": pn["known_frac"],
                 "diffusion": not cfg.get("no_diffusion", False),
                 "horizon_shift_med": pn.get("horizon_shift_med", 0.0)},
        "terrain": {"view_bias": view_bias},
        "prototypes": protos,
        "layers": [{"name": l["name"], "verts": int(l["pos"].shape[0]),
                    "tris": int(l["index"].size // 3)} for l in layers],
        "glb": info, "verify": rep, "light": light,
        "elapsed_s": round(time.time() - t_all, 1),
    }
    common.json_dump(os.path.join(out_dir, "world.json"), report)
    print(f"[6/6] 完成 {time.time()-t_all:.1f}s  GLB {info['bytes']/1e6:.1f}MB  "
          f"PSNR {rep['psnr']:.2f}dB (原图区 {rep['psnr_core']:.2f}dB)  SSIM {rep['ssim']:.3f}  "
          f"cov {rep['coverage']*100:.1f}%  reproj {rep['reproj']['median_px']:.2e}px")
    return report


def _texture_array(rgb: np.ndarray) -> np.ndarray:
    """与 GLB 里那张贴图同构的数组（照片 + 底部白条），供光栅化采样。"""
    h, w = rgb.shape[:2]
    canvas = np.ones((h + common.WHITE_STRIP_PX, w, 3), dtype=np.float32)
    canvas[:h] = rgb
    return canvas


def _bbox(layers: list) -> dict:
    P = np.concatenate([l["pos"] for l in layers])
    return {"x0": float(P[:, 0].min()), "x1": float(P[:, 0].max()),
            "y0": float(P[:, 1].min()), "y1": float(P[:, 1].max()),
            "z0": float(P[:, 2].min()), "z1": float(P[:, 2].max())}


def main() -> int:
    ap = argparse.ArgumentParser(description="PHOTOWORLD v3：照片 → 连续可漫游世界")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="out/v3/world")
    ap.add_argument("--step", type=int, default=3, help="视锥内网格抽稀（1=原分辨率）")
    ap.add_argument("--no-diffusion", action="store_true", help="不跑扩散（只做视锥内）")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--pano-w", type=int, default=1024)
    ap.add_argument("--pano-steps", type=int, default=20)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jpeg", type=int, default=90, help="内嵌照片贴图的 JPEG 质量")
    # views（多视角分段）暂时是**实验开关**：本机缓存的只有 SDXL-**inpaint** 管线，
    # 它的 __call__ 不接受 strength，mask 全白时等于纯文生图 —— 视角图跟参考图无关，
    # 拼回来就是满屏竖条（实测）。要真正用它，得再下一份 img2img 管线权重。
    ap.add_argument("--pano-mode", choices=["eq", "views"], default="eq",
                    help="全景生成方式：eq=一次 inpaint（默认）；views=多视角分段（实验）")
    ap.add_argument("--pano-views", type=int, default=6, help="多视角分段的数量")
    ap.add_argument("--no-bg-fill", action="store_true",
                    help="不跑【前景抠除 + 扩散补背景板】（GPU 慢时用）")
    ap.add_argument("--relief-amp", type=float, default=7.0,
                    help="视锥外地形的低频起伏幅度（米，默认 7）")
    ap.add_argument("--pano-from", default=None,
                    help="复用一张已有全景（只重算几何，跳过扩散；A/B 用）")
    ap.add_argument("--cliff", type=float, default=1.8, help="断崖判据（越小切得越狠）")
    ap.add_argument("--prop-density", type=float, default=1.0)
    ap.add_argument("--no-compact", action="store_true",
                    help="GLB 退回全 float32（默认用规范化整数，体积更小）")
    ap.add_argument("--no-horizon-align", action="store_true",
                    help="关闭全景地平线对齐（P1-2，默认开）")
    ap.add_argument("--no-view-reconcile", action="store_true",
                    help="关闭 8 视角高度偏差校正（P1-3，默认开）")
    args = ap.parse_args()

    files = []
    for pat in args.images:
        files.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("没有输入图片", file=sys.stderr)
        return 2
    if args.limit:
        files = files[:args.limit]

    cfg = {"step": args.step, "no_diffusion": args.no_diffusion, "prompt": args.prompt,
           "jpeg": args.jpeg, "cliff": args.cliff, "prop_density": args.prop_density,
           "pano_from": args.pano_from,
           "bg_fill": not args.no_bg_fill, "relief_amp": args.relief_amp,
           "pano_mode": args.pano_mode, "pano_views": args.pano_views,
           # 等距柱状图固定 2:1，高度由宽度推出来
           "pano_w": args.pano_w, "pano_h": args.pano_w // 2,
           "pano_steps": args.pano_steps, "views": args.views,
           "device": args.device, "seed": args.seed,
           "compact": not args.no_compact,
           "horizon_align": not args.no_horizon_align,
           "view_reconcile": not args.no_view_reconcile}
    reports = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        out = args.out if len(files) == 1 else os.path.join(args.out, name)
        # 单图模式 "--out 目录" 直接写该目录；如果里面已经有**另一张图**的产物，
        # 会被静默覆盖（实测踩过：连跑三张不同照片都写到同一个 world.glb，
        # 后面所有验证看的都是旧资产）。这里至少吵一声。
        prev = os.path.join(out, "world.json")
        if os.path.exists(prev):
            try:
                old = json.load(open(prev, encoding="utf-8")).get("input", "")
                newrel = os.path.relpath(f, _ROOT)
                if old and old != newrel:
                    print("  [警告] %s 里已有另一张图（%s）的产物，将被覆盖；"
                          "多图请用 --out <父目录>" % (out, old))
            except Exception:
                pass
        try:
            reports.append(bake_one(f, out, cfg))
        except Exception as e:
            import traceback
            traceback.print_exc()
            reports.append({"input": f, "error": str(e)[:300]})
    if len(reports) > 1:
        common.json_dump(os.path.join(args.out, "_summary.json"), reports)
        ok = [r for r in reports if "error" not in r]
        print(f"\n合计 {len(reports)} 张，成功 {len(ok)}")
        for r in ok:
            print("  %-28s PSNR %5.2f  SSIM %.3f  %s" % (
                r["input"], r["verify"]["psnr"], r["verify"]["ssim"],
                "%.1fMB" % (r["glb"]["bytes"] / 1e6)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
