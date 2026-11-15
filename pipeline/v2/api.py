#!/usr/bin/env python3
"""v2 可调用接口：一张照片 → 分层 GLB 字节（带缓存、带常驻深度模型）。

给三条链路共用：
  * `server/app.py` 的 `POST /api/reconstruct`（前端点一张照片就走这里）；
  * 命令行批处理（`bake.py` 内部同样的步骤）；
  * 测试/回归。

为什么不写在 bake.py 里：bake.py 是"命令行批处理"，每次进程启动都要重新加载
1.3GB 权重（约 3s）。服务端必须**常驻**模型，否则单张照片的端到端时间会被
模型加载吞掉。所以这里做模块级单例 + 按图片内容哈希缓存。
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
import threading
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
ROOT = os.path.dirname(PIPE)
for p in (PIPE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import geom          # noqa: E402
import glb as glb_mod  # noqa: E402
import ldi           # noqa: E402
import calib         # noqa: E402
from exif import hfov_from_image, vfov_of  # noqa: E402

DEFAULTS = {
    "work_side": 1536, "tex_max": 2048, "jpeg_quality": 90,
    "eye": 1.7, "max_tris": 520000, "edge_ratio": 2.4,
    "far_split": 120.0, "horizon_far": 1500.0,
    "flatten": True, "skirt_min": 0.12, "skirt_max": 6.0,
    "stretch_clamp": 12.0, "strong_median": 7, "quantize": 0.0,
    "long_side": 1540, "device": "cuda", "median_size": 3,
}

_lock = threading.Lock()
_model = None
_cache: dict[str, tuple[bytes, dict]] = {}
MODEL_DIR = os.path.join(ROOT, "models", "da2_large")


def available() -> dict:
    """探测依赖与权重，供 /health 和前端判断要不要走 v2。"""
    ok = True
    why = []
    for mod in ("torch", "transformers", "scipy"):
        try:
            __import__(mod)
        except Exception:
            ok = False
            why.append("缺少 python 包 " + mod)
    if not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
        ok = False
        why.append("缺少深度权重 models/da2_large（见 pipeline/v2/README）")
    return {"ok": ok, "reason": "; ".join(why)}


def _load_model(device: str = "cuda"):
    global _model
    with _lock:
        if _model is None:
            import depth_gpu
            _model = depth_gpu.load_model(MODEL_DIR, device)
        return _model


def _hash(data: bytes, opts: dict) -> str:
    h = hashlib.sha256()
    h.update(data)
    h.update(json.dumps({k: opts.get(k) for k in sorted(DEFAULTS)}, sort_keys=True).encode())
    return h.hexdigest()


def bake(data: bytes, **over) -> tuple[bytes, dict]:
    """照片字节 → (GLB 字节, manifest)。异常向上抛，由调用方决定怎么回。"""
    opts = dict(DEFAULTS)
    opts.update({k: v for k, v in over.items() if v is not None})
    key = _hash(data, opts)
    if key in _cache:
        return _cache[key]

    import depth_gpu

    t0 = time.time()
    img_src = Image.open(io.BytesIO(data)).convert("RGB")
    src_size = img_src.size
    hfov, fov_src, meta = hfov_from_image(img_src)

    work = img_src
    if max(work.size) > opts["work_side"]:
        s = opts["work_side"] / float(max(work.size))
        work = work.resize((max(1, round(work.size[0] * s)), max(1, round(work.size[1] * s))),
                           Image.LANCZOS)
    tw, th = work.size
    cam = geom.Camera(hfov=hfov, vfov=vfov_of(hfov, tw, th), w=tw, h=th)

    proc, model = _load_model(opts["device"])
    disp = depth_gpu.estimate(work, proc, model, opts["device"], opts["long_side"], True)

    rb = calib.robust_calibrate(disp, cam, eye=opts["eye"], horizon_far=opts["horizon_far"])
    cam, z, cal = rb["cam"], rb["z"], rb["cal"]

    rgb = np.asarray(work, dtype=np.float32) / 255.0
    far_m = cal.get("far", 1800.0)
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1]))
    sky = (z >= 0.97 * far_m) & (cam.rays()[..., 1] > 0.02) & ((gx + gy) < 0.02)

    step = 1
    while (tw // step) * (th // step) * 2 > opts["max_tris"]:
        step += 1

    res = ldi.build(z, cam, step=step, eye=opts["eye"], far_split=opts["far_split"],
                    near_split=0.0, sky=sky, edge_ratio=opts["edge_ratio"],
                    flatten=opts["flatten"], skirt_min=opts["skirt_min"],
                    skirt_max=opts["skirt_max"], median_size=opts["median_size"],
                    stretch_clamp=opts["stretch_clamp"],
                    strong_median=opts["strong_median"], quantize=opts["quantize"])

    tex = img_src
    if max(tex.size) > opts["tex_max"]:
        s = opts["tex_max"] / float(max(tex.size))
        tex = tex.resize((round(tex.size[0] * s), round(tex.size[1] * s)), Image.LANCZOS)
    buf = io.BytesIO()
    tex.save(buf, format="JPEG", quality=opts["jpeg_quality"], optimize=True)

    out = io.BytesIO()
    tmp = os.path.join(HERE, "_tmp_out.glb")
    cam_extras = {"camera": {"hfov": cam.hfov, "vfov": cam.vfov, "w": tw, "h": th,
                             "pitch": cam.pitch, "eyeHeight": opts["eye"], "yaw": 0.0,
                             "generator": "photoworld-v2"}}
    info = glb_mod.write_glb(tmp, res["layers"], buf.getvalue(),
                             name="photoworld", extras=cam_extras)
    with open(tmp, "rb") as f:
        glb_bytes = f.read()
    try:
        os.remove(tmp)
    except OSError:
        pass

    manifest = {
        "version": "v2", "source_size": list(src_size), "work_size": [tw, th],
        "texture_size": list(tex.size), "fov_source": fov_src,
        "camera": {"hfov_deg": round(math.degrees(cam.hfov), 2),
                   "vfov_deg": round(math.degrees(cam.vfov), 2),
                   "pitch_deg": round(math.degrees(cam.pitch), 2)},
        "calibration": {k: v for k, v in cal.items() if isinstance(v, (int, float, str, bool))},
        "layers": res["stats"], "glb": info,
        "depth_ms": None, "elapsed_s": round(time.time() - t0, 2),
    }
    _cache[key] = (glb_bytes, manifest)
    if len(_cache) > 24:                       # 简单上限，避免长跑吃满内存
        _cache.pop(next(iter(_cache)))
    return glb_bytes, manifest
