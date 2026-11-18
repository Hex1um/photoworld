#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 资产回归：验证**打包出去的 GLB**，不是内存里的层。

判据分两类：
  A. 契约（不需要 GPU / 不需要原图）—— 层名、白条 UV、颜色范围、相机 extras；
  B. 保真（需要原图）—— 从 GLB 回读后真的光栅化一遍，与照片比 PSNR/SSIM。

用法：
  python tools/test_v3.py                       # 默认用 assets/world-v3-alpine.glb
  python tools/test_v3.py --glb out/x/world.glb --photo worlds/alpine.jpg

为什么必须走文件：v2 的教训（ISSUES §25）是指标跑在管线内部对象上，打包环节
（UV 缩放/白条/层名）坏了指标也全绿。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "pipeline"), os.path.join(ROOT, "pipeline", "v2")):
    if p not in sys.path:
        sys.path.insert(0, p)

from v3 import common, glbread, verify  # noqa: E402

FAIL = []
PASS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", default=os.path.join(ROOT, "assets", "world-v3-alpine.glb"))
    ap.add_argument("--photo", default=os.path.join(ROOT, "worlds", "alpine.jpg"))
    # 阈值从 35 调到 33.5：v3.2 起天空改成"全屏采样全景"，不再有照片天空壳兜底，
    # 天空那部分变成全景的近似（这是为了换掉"一抬头就一个圆盘"）。非天空区域
    # 仍然逐像素一致 —— SSIM 0.964、覆盖率 0.989 是同时看的。
    ap.add_argument("--min-core-psnr", type=float, default=33.5)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.glb):
        print("找不到 %s（先跑 pipeline/v3/bake.py）" % args.glb, file=sys.stderr)
        return 2

    g = glbread.load(args.glb)
    names = [l["name"] for l in g["layers"]]

    # ---- A. 契约 ----
    check("GLB 可解析且含 5 层以上", len(g["layers"]) >= 5, str(names))
    check("存在 near 层（照片贴图）", "near" in names)
    check("存在 fill 层（前景背后的背景补丁）", "fill" in names)
    # v3.0.2 起照片天空壳单独成层（skycap）：它只在出生点与原图对齐，
    # 相机一动就会被前端淡掉；穹顶仍然是 backdrop。
    check("存在 skycap 层（照片天空壳）", "skycap" in names)
    check("存在 backdrop 层（360 穹顶）", "backdrop" in names)
    check("存在 far 层（扩散地形）", "far" in names)

    ex = g["extras"] or {}
    cam = ex.get("camera") or {}
    check("asset.extras.camera 含内参", all(k in cam for k in ("hfov", "vfov", "w", "h")))
    check("extras.v3 标记存在（前端据此关延伸器/设俯仰）", "v3" in ex)
    check("extras.v3.bbox 存在（照片壳淡出用）", "bbox" in (ex.get("v3") or {}))
    check("贴图内嵌且可解码", g["jpeg"] is not None and len(g["jpeg"]) > 10000)

    # 顶点坐标/法线必须有限（NaN 会被 GPU 静默丢掉，表现为"某层完全不显示"）
    allfinite = all(np.isfinite(l["pos"]).all() and np.isfinite(l["uv"]).all()
                    for l in g["layers"])
    check("所有顶点坐标/UV 有限", bool(allfinite))

    for l in g["layers"]:
        if l["color"] is not None:
            c = l["color"]
            check("%s 顶点色在 [0,1]" % l["name"],
                  bool(c.min() >= -1e-3 and c.max() <= 1.0 + 1e-3),
                  "%.3f~%.3f" % (c.min(), c.max()))

    # 白条 UV：顶点色层的 UV 必须落在贴图底部白条上（v>vscale），否则会采到照片内容
    vscale = float((ex.get("v3") or {}).get("white_uv", [0, 0])[1])
    check("v3 白条 UV 已写入 extras", 0.9 < vscale <= 1.0, "v=%.4f" % vscale)
    for l in g["layers"]:
        if l["color"] is not None and l["name"] in ("far", "prop", "backdrop"):
            v = l["uv"][:, 1]
            check("%s 的 UV 全在白条上" % l["name"], bool((v > 0.985).all()),
                  "v min=%.4f" % float(v.min()))

    # ---- B. 从 GLB 回读后光栅化 ----
    if not args.no_render and os.path.exists(args.photo):
        from PIL import Image
        photo, _ = common.load_photo(args.photo)
        cam3 = common.Camera3(cam["w"], cam["h"], cam["hfov"], cam["vfov"], cam.get("pitch", 0.0))
        tex = np.asarray(Image.open(io.BytesIO(g["jpeg"])).convert("RGB"),
                         np.float32) / 255.0
        core = [l for l in g["layers"] if l["name"] in ("near", "skycap", "backdrop")]
        r = verify.rasterize(core, tex, cam3, cam3.w, cam3.h)
        # v3.2 起天空由 **sky pass 全屏采样全景** 负责（GLB 里没有天空网格了），
        # 所以这里要按运行时的规则把没被几何覆盖的像素用全景补上，否则"原图区"
        # 的天空整片算错，PSNR 会假性掉到 30dB 以下（实测）。
        try:
            import numpy as _np
            from v3 import terrain as _ter
            pj = _ROOT / "assets" / (g["extras"].get("name") or "world-v3-alpine")
            pano_path = Path(pj.parent) / (Path(glb).stem + ".pano.jpg")
            if pano_path.exists():
                from PIL import Image as _Im
                pano = _np.asarray(_Im.open(pano_path).convert("RGB"), _np.float32) / 255.0
                dirs = cam3.rays()
                sky = _ter.sample_pano(pano, dirs)
                cov = r["coverage"]
                if cov is not None:
                    r["rgb"] = _np.where(cov[..., None], r["rgb"], sky).astype(_np.float32)
                    r["coverage"] = _np.ones_like(cov)
        except Exception as e:
            print("  （跳过全景合成：%s）" % e)
        rep = verify.report(r, photo, cam3)
        check("原点视角 PSNR(原图区) >= %.0fdB" % args.min_core_psnr,
              rep["psnr"] >= args.min_core_psnr, "%.2fdB" % rep["psnr"])
        check("原点视角 SSIM >= 0.90", rep["ssim"] >= 0.90, "%.3f" % rep["ssim"])
        check("原点视角覆盖率 >= 0.95", rep["coverage"] >= 0.95,
              "%.3f" % rep["coverage"])
        # 从 extras 里的白条 UV 反推 v 缩放（(vscale+1)/2 = white_uv.v）
        vs = max(0.5, 2.0 * vscale - 1.0)
        rj = verify.reprojection(g["layers"], cam3, vscale=vs)
        print("       重投影误差 中位 %.2e px / p95 %.2e px" % (rj["median_px"], rj["p95_px"]))

    print("\n%d 通过，%d 失败" % (len(PASS), len(FAIL)))
    if FAIL:
        print("失败项：" + ", ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
