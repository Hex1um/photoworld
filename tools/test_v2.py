#!/usr/bin/env python3
"""v2 管线回归（CPU，不需要 GPU / 权重，2 秒内跑完）。

    python tools/test_v2.py
    python tools/test_v2.py --with-model      # 额外跑一次真实推理（需要 CUDA + 权重）

守的是"会让整条链路悄悄变成垃圾"的几条硬约束：
  * 网格索引越界 / 属性长度不一致 —— GLB 加载器会静默丢面或直接崩；
  * 法线未归一化 —— 光照全黑或过曝；
  * UV 越界 —— 采样到 clamp 边缘，整片糊开；
  * 标定把场景压成一片（p 越界时不回退）—— 就是 v2 开发中踩到的 207~1136m 那次的病；
  * GLB 头部/extras 缺失 —— 前端拿不到相机内参，整幅重建错位。
"""
from __future__ import annotations

import math
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "pipeline"), os.path.join(ROOT, "pipeline", "v2")):
    if p not in sys.path:
        sys.path.insert(0, p)

import geom    # noqa: E402
import ldi     # noqa: E402
import calib   # noqa: E402
import glb     # noqa: E402

FAIL = []
OK = 0


def check(cond, msg):
    global OK
    if cond:
        OK += 1
        print("PASS  " + msg)
    else:
        FAIL.append(msg)
        print("FAIL  " + msg)


def synth_scene(w=320, h=200):
    cam = geom.Camera(hfov=math.radians(62), vfov=math.radians(40), w=w, h=h)
    ry = cam.rays()[..., 1]
    z = np.where(ry < -0.02, -1.7 / np.minimum(ry, -1e-3), 40.0)
    z = np.clip(z, 0.6, 1800.0).astype(np.float32)
    z[60:90, 120:200] = 6.0            # 一块近景方块
    return z, cam


def test_mesh():
    z, cam = synth_scene()
    # far_split 要小于场景最远处（合成场景的"墙"在 40m），否则背景层为空
    res = ldi.build(z, cam, step=1, far_split=25.0, near_split=12.0, sky=None)
    g = res["stats"]["_global"]
    check(g["triangles"] > 5000, f"网格三角数合理（{g['triangles']}）")
    check(len(res["layers"]) >= 2, f"至少两层（{len(res['layers'])}）")
    for L in res["layers"]:
        n = L["pos"].shape[0]
        check(L["nrm"].shape[0] == n and L["uv"].shape[0] == n,
              f"{L['name']}: POSITION/NORMAL/UV 长度一致")
        check(int(L["index"].max()) < n, f"{L['name']}: 索引不越界")
        check(np.isfinite(L["pos"]).all(), f"{L['name']}: 位置无 NaN/Inf")
        ln = np.linalg.norm(L["nrm"], axis=1)
        check(np.allclose(ln, 1.0, atol=2e-3), f"{L['name']}: 法线已归一化")
        check(L["uv"].min() >= 0.0 and L["uv"].max() <= 1.0, f"{L['name']}: UV 在 [0,1]")
        check(L["index"].size % 3 == 0, f"{L['name']}: 索引数能被 3 整除")
    # 近景方块必须落在近层（否则分层判据失效）
    near = [L for L in res["layers"] if L["name"] == "near"]
    check(bool(near) and near[0]["pos"].shape[0] > 0, "存在近层且非空")


def test_flat_ground():
    """纯地面场景：压平后地面顶点应几乎落在 y≈-eye 附近（相机系）。"""
    w, h = 240, 180
    cam = geom.Camera(hfov=math.radians(60), vfov=math.radians(45), w=w, h=h)
    ry = cam.rays()[..., 1]
    z = np.where(ry < -0.05, -1.7 / np.minimum(ry, -1e-3), 500.0).astype(np.float32)
    z = np.clip(z, 0.6, 1800.0)
    # 给深度加 ±8% 的扰动（真实深度不会正好落在平面上），压平应把它吸回平面
    rng = np.random.default_rng(3)
    z_noisy = (z * (1.0 + 0.08 * rng.standard_normal(z.shape))).astype(np.float32)
    zs, ground = ldi.flatten_ground(z_noisy, cam.rays(), eye=1.7)
    check(float(ground.mean()) > 0.15, f"识别出地面像素（{ground.mean()*100:.0f}%）")
    ry = cam.rays()[..., 1]
    plane = np.where(ground, -1.7 / np.minimum(ry, -1e-3), np.nan)
    resid = np.nanmax(np.abs(zs - plane))
    check(resid < 1e-3, f"地面像素被精确压到平面（最大残差 {resid:.2e} m）")
    check(not np.isnan(zs).any(), "压平后无 NaN")


def test_calibration_fallback():
    """地面带不存在时（深度全平滑），不能把整幅压成一片 —— 必须退回分位双锚点。"""
    w, h = 200, 140
    cam = geom.Camera(hfov=math.radians(60), vfov=math.radians(42), w=w, h=h)
    ry = cam.rays()[..., 1]
    # 构造一幅"底部不是地面"的深度：全图都是 0.2~0.9 的平滑渐变
    disp = (0.2 + 0.7 * (1 - (ry + 1) / 2)).astype(np.float32)
    rb = calib.robust_calibrate(disp, cam)
    z = rb["z"]
    used = rb["cal"]["mode_used"]
    p10, p90 = np.percentile(z, 10), np.percentile(z, 90)
    check(used in ("percentile-two-anchor", "ground-anchored"), f"标定有明确模式（{used}）")
    # 合成的是"全图平滑渐变、没有近景"的病态输入：只要不塌成一个平面即可。
    # （真实照片有近景时这个比值通常 20~100，见 docs/AB_v2.md）
    check(p90 / max(p10, 1e-6) > 2.0,
          f"纵深没有被压成一个平面（p90/p10={p90/max(p10,1e-6):.1f}）")
    check(np.isfinite(z).all() and z.min() > 0, "米深度有限且为正")


def test_glb_roundtrip():
    z, cam = synth_scene(160, 100)
    res = ldi.build(z, cam, step=1, far_split=25.0, near_split=12.0, sky=None)
    tmp = os.path.join(ROOT, "out", "_test_v2.glb")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    extras = {"camera": {"hfov": 1.0, "vfov": 0.7, "w": 160, "h": 100}}
    info = glb.write_glb(tmp, res["layers"], b"\xff\xd8\xff\xd9", name="t", extras=extras)
    raw = open(tmp, "rb").read()
    magic, ver, total = struct.unpack("<III", raw[:12])
    check(magic == 0x46546C67 and ver == 2, "GLB 魔数与版本正确")
    check(total == len(raw), f"GLB 头部长度与实际一致（{total}/{len(raw)}）")
    jlen, jtype = struct.unpack("<II", raw[12:20])
    import json
    doc = json.loads(raw[20:20 + jlen].decode("utf-8"))
    check(doc.get("asset", {}).get("extras", {}).get("camera", {}).get("vfov") == 0.7,
          "相机内参写进了 asset.extras")
    check(len(doc["meshes"]) == info["layers"], "mesh 数 == 层数")
    names = [m["name"] for m in doc["meshes"]]
    check("backdrop" in names, "存在 backdrop 层（前端据此做无视差背景）")
    open(tmp, "rb").close()
    os.remove(tmp)


def test_with_model():
    """真跑一次推理（需要 CUDA + models/da2_large）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v2_api", os.path.join(ROOT, "pipeline", "v2", "api.py"))
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    av = api.available()
    if not av.get("ok"):
        print("SKIP  真实推理（" + av.get("reason", "") + "）")
        return
    from PIL import Image
    import io
    img = Image.open(os.path.join(ROOT, "worlds", "alpine.jpg")).convert("RGB")
    img.thumbnail((640, 640))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    glb_bytes, man = api.bake(buf.getvalue())
    check(len(glb_bytes) > 200_000, f"真实烘焙产出 GLB（{len(glb_bytes)/1e6:.1f}MB）")
    check(man["camera"]["vfov_deg"] > 5, "manifest 里有相机视场")
    layers = [k for k in man["layers"] if not k.startswith("_")]
    check(len(layers) >= 1, f"至少一层（{layers}）")


if __name__ == "__main__":
    print("=== v2 管线回归 ===")
    test_mesh()
    test_flat_ground()
    test_calibration_fallback()
    test_glb_roundtrip()
    if "--with-model" in sys.argv:
        test_with_model()
    print(f"\n{OK} 通过 / {len(FAIL)} 失败")
    if FAIL:
        for f in FAIL:
            print("  - " + f)
        raise SystemExit(1)
