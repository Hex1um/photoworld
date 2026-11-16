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
    "repr": "auto", "scatter_thr": 0.002, "cliff_thr": 0.010, "ground_tol": 0.30,
    # 视锥外扩（模型补全）：倍数 >1 才启用。1.7 = 左右各多 35%，
    # 纵向 1.3 = 上 8% / 下 22%（地面比天空更需要内容）
    # 外扩默认**关闭**（1.0）：模型能生成可信的左右地貌（实测岩壁、树都接得上），
    # 但扩完之后 ① 视场从 69° 变成 99°，② 标定/取景没跟着重配，
    # 结果画面变暗、地平线抬高、地面占满 2/3 —— 比不扩更差。
    # 打开需要同时把"扩展后的相机与标定"一起调好，这是一项独立工作。
    # 开启模型外扩（只扩左右）：生成区能长出树/岩壁，正是"要有地物"要求的来源。
    "outpaint": 1.70, "outpaint_y": 1.0, "outpaint_steps": 26,
    "outpaint_side": 1024, "outpaint_seed": 0,
    # 默认**不**用扩散模型：本机缓存里只有 SDXL-inpaint，8GB 显存上单张 15 分钟都跑不完
    # （实测）。不打开时走"镜像延展 + 渐隐到雾色"，同样能消掉视锥外的空白与硬边，
    # 只是没有新内容。等快速权重（dreamshaper-8-inpainting）下完整后，再开这个开关。
    # 显存充足时开启（有守卫：可用显存 <3.2GB 会自动退回快速路径，不会卡死）
    "outpaint_model": True,
    "ground_radius": 400.0, "semantics": True,
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



def _sem_obj_aligned(sem, seg, work_size=None):
    """语义物体掩码，必要时缩放到工作图（可能被外扩过）的分辨率。

    语义是在**原图**上算的，而 ldi 拿到的世界是基于工作图（外扩后更宽）。
    尺寸不一致时如果不缩放，掩码会整体错位（左侧物体被当成右侧）。
    """
    if seg is None:
        return None
    try:
        m = sem.object_mask(seg)
    except Exception:
        return None
    if m is None or not m.any():
        return m
    if work_size is None:
        return m
    from PIL import Image as _Im
    if (m.shape[1], m.shape[0]) == tuple(work_size):
        return m
    up = _Im.fromarray((m.astype(np.uint8) * 255), mode="L").resize(
        tuple(work_size), _Im.NEAREST)
    return np.asarray(up) > 127

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
    hfov0 = hfov

    # ---- 视锥外扩：把"照片只有一块梯形"补成"周围也有世界" --------------
    # 不做这一步，画面左右下三块是纯色，照片边界是一条刀切硬边。
    oinfo = {"mode": "off"}
    wall = False
    wall_why = ""
    seg = None
    if opts["semantics"]:
        try:
            import importlib
            import sys as _sys
            # 语义模块在 pipeline/v2/ 下；api 被以文件路径加载时，那个目录不在 sys.path，
            # import_module("semantics") 会抛 ImportError，而这段在 try 里 →
            # **静默降级成 None**，于是"非风景照判定""物体封闭化"全都拿不到语义结果
            # （实测：灯柱/lake 的语义一直是 None）。这里显式补路径。
            _p2 = os.path.dirname(os.path.abspath(__file__))
            if _p2 not in _sys.path:
                _sys.path.insert(0, _p2)
            sem = importlib.import_module("semantics")
            if sem.available(ROOT):
                # 这里**不能**用 rgb：它在后面（外扩之后）才定义，
                # 之前直接引用会 NameError，而这段在 try 里 → 语义静默降级成 None，
                # 导致"非风景判定""urban 门控""物体封闭化"全部失效（长期潜伏的 bug）。
                _rgb_sem = np.asarray(img_src, dtype=np.float32) / 255.0
                seg = sem.segment(_rgb_sem, ROOT)
                ok, wall_why = sem.looks_like_scenery(seg)
                wall = not ok
            else:
                wall_why = "缺少语义模型（models/segformer_b0_ade20k.onnx）"
        except Exception as e:              # 语义只是加分项，失败不能中断烘焙
            wall_why = "语义判定失败: %s" % e

    # 外扩的"适用性"门控：城市/近景物体多的照片，扩散模型会把画面挤成中间一条、
    # 两侧变成重复边条（实测校园灯柱那张）。这类图改走地面裙边。
    urban = 0.0
    if seg is not None:
        urban = seg["frac"]["building"] + seg["frac"]["person"] + seg["frac"]["vehicle"]
    if urban > 0.22 and not wall:
        wall_why = (wall_why + f" | 建筑/人物/车辆占比 {urban*100:.0f}%，不启用模型外扩").strip(" |")

    # 外扩只对**横构图**有意义：竖构图（宽高比 <1.15）横向扩 1.7 倍后，
    # 照片内容会被挤成中间一条、两侧变成重复的边条（实测灯柱那张就是）。
    # 竖构图继续走地面裙边。
    aspect0 = src_size[0] / float(max(1, src_size[1]))
    if float(opts["outpaint"]) > 1.01 and aspect0 >= 1.15 and urban <= 0.22:
        import importlib
        op = importlib.import_module("outpaint")
        img_src, oinfo = op.outpaint_two_pass(
            img_src, ROOT,
            scale_x=float(opts["outpaint"]), scale_y=float(opts["outpaint_y"]),
            device=opts["device"], use_model=bool(opts["outpaint_model"]),
            steps=int(opts["outpaint_steps"]), gen_side=int(opts["outpaint_side"]),
            seed=int(opts["outpaint_seed"]))
        if img_src.size[0] != src_size[0]:
            hfov = op.hfov_of_extended(hfov0, src_size[0], img_src.size[0])

    work = img_src
    if max(work.size) > opts["work_side"]:
        s = opts["work_side"] / float(max(work.size))
        work = work.resize((max(1, round(work.size[0] * s)), max(1, round(work.size[1] * s))),
                           Image.LANCZOS)
    tw, th = work.size
    cam = geom.Camera(hfov=hfov, vfov=vfov_of(hfov, tw, th), w=tw, h=th)

    proc, model = _load_model(opts["device"])

    # ---- 外扩时的深度：**在原始照片上推理，再按列外延** -------------------
    # 为什么不直接在扩展图上跑深度：生成区是扩散模型"画"出来的，
    # 单目深度对它给出的估计只是"看起来合理"，与照片那侧的深度在接缝处对不上。
    # 建面时那条深度跳变会被剪成一条竖缝，边缘像素被拉成一大片浅色幕
    # （实测：校园灯柱那张的灯柱被拉成一根浅色柱子 + 横贯画面的白幕）。
    # 正确做法是让几何**连续**：深度只在原图上算，左右两侧用边界列外延
    # （并轻微加大远端，让两侧读作"远处"），纹理仍用生成出来的像素。
    pad_lr = None
    if isinstance(oinfo, dict):
        p1 = oinfo.get("pass1") or {}
        if p1.get("pad") and len(p1["pad"]) == 4:
            pad_lr = p1["pad"]
    if pad_lr and max(pad_lr) > 0:
        base = img_src
        # 原图区域（扩展画布去 pad 后的部分）
        l, t, r0, b = pad_lr
        core = base.crop((l, t, base.size[0] - r0, base.size[1] - b))
        if max(core.size) > opts["work_side"]:
            s = opts["work_side"] / float(max(core.size))
            core = core.resize((max(1, round(core.size[0] * s)),
                                max(1, round(core.size[1] * s))), Image.LANCZOS)
        disp_core = depth_gpu.estimate(core, proc, model, opts["device"],
                                       opts["long_side"], True)
        if max(work.size) > opts["work_side"]:
            s = opts["work_side"] / float(max(work.size))
            tw2, th2 = (max(1, round(work.size[0] * s)), max(1, round(work.size[1] * s)))
        else:
            tw2, th2 = work.size
        # 把 core 的深度放到整幅网格里，左右用边界列外延
        disp = np.zeros((th2, tw2), dtype=np.float32)
        cw = max(1, min(tw2, int(round(tw2 * core.size[0] / float(work.size[0])))))
        ch = max(1, min(th2, int(round(th2 * core.size[1] / float(work.size[1])))))
        core_r = np.asarray(Image.fromarray((np.clip(disp_core, 0, 1) * 255).astype(np.uint8))
                            .resize((cw, ch), Image.BILINEAR), dtype=np.float32) / 255.0
        y0 = max(0, (th2 - ch) // 2)
        x0 = max(0, (tw2 - cw) // 2)
        disp[y0:y0 + ch, x0:x0 + cw] = core_r[:min(ch, th2 - y0), :min(cw, tw2 - x0)]
        if x0 > 0:
            # 左边：复制第 x0 列，并向左轻微衰减（视差上读作"更远一点"）
            colL = disp[y0:y0 + ch, x0]
            ramp = np.linspace(0.85, 1.0, x0, dtype=np.float32)[None, :]
            disp[y0:y0 + ch, :x0] = colL[:, None] * ramp
        if x0 + cw < tw2:
            colR = disp[y0:y0 + ch, x0 + cw - 1]
            n = tw2 - (x0 + cw)
            ramp = np.linspace(1.0, 0.85, n, dtype=np.float32)[None, :]
            disp[y0:y0 + ch, x0 + cw:] = colR[:, None] * ramp
        # 上下留空的行（正常不会有，扩展只做左右）用最近行补齐
        if y0 > 0:
            disp[:y0] = disp[y0]
        if y0 + ch < th2:
            disp[y0 + ch:] = disp[y0 + ch - 1]
    else:
        disp = depth_gpu.estimate(work, proc, model, opts["device"], opts["long_side"], True)

    # 外扩之后深度图变宽了：如果还在**整幅**上做标定，生成区的天空/地面会把
    # 分位锚点带偏，尺度整体漂移 —— 实测表现是"画面变暗、地平线抬高、地面占满"。
    # 所以标定只在原照片那一块做（像素密度不变 → 与原图同一套视场），
    # 再把结果套到整幅扩展深度上。
    disp_c, cam_c = disp, cam
    pad = (oinfo or {}).get("pass1", {}).get("pad") if isinstance(oinfo, dict) else None
    if pad and len(pad) == 4 and max(pad) > 0:
        l, t, r, b = pad
        h0, w0 = disp.shape[0] - t - b, disp.shape[1] - l - r
        if h0 > 32 and w0 > 32:
            disp_c = disp[t:t + h0, l:l + w0]
            cam_c = geom.Camera(hfov=hfov0, vfov=vfov_of(hfov0, w0, h0), w=w0, h=h0)
    rb = calib.robust_calibrate(disp_c, cam_c, eye=opts["eye"], horizon_far=opts["horizon_far"])
    cal = rb["cal"]
    cam = rb["cam"]
    if disp_c is not disp:
        # 同一套 A/p 应用到整幅；相机用扩展后的（视场更宽，像素密度一致）
        z, cal = geom.apply_calibration(disp, cal, cal.get("far", 1800.0))
        cam = geom.Camera(hfov=hfov, vfov=vfov_of(hfov, tw, th), w=tw, h=th,
                          pitch=rb["cam"].pitch)
    else:
        z = rb["z"]

    rgb = np.asarray(work, dtype=np.float32) / 255.0

    # ---- 语义判定：这是不是一张"可以建三维"的风景照 -----------------------
    # 云海 / 星空 / 太空地球这类图，深度没有物理意义，硬建三维只会得到一团
    # 黑色拉丝（验收批次里最惨的两张）。语义是唯一可靠的信号 —— 数值统计试过两次
    # 都不成立。命中就退回"照片平贴"，观感等于一张高分辨率照片挂在正前方。
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
                    # 语义"物体"掩码：用它给近景物体建封闭体（比深度屏障可靠得多 ——
                    # 实测 lake/alpine 用深度分割一个候选都分不出来）
                    # rgb 就是工作图（可能已外扩），掩码按它的分辨率对齐
                    sem_obj=_sem_obj_aligned(sem, seg, (rgb.shape[1], rgb.shape[0])),
                    flatten=opts["flatten"], skirt_min=opts["skirt_min"],
                    skirt_max=opts["skirt_max"], median_size=opts["median_size"],
                    stretch_clamp=opts["stretch_clamp"],
                    strong_median=opts["strong_median"], quantize=opts["quantize"],
                    rgb=rgb, repr_mode=opts["repr"], scatter_thr=opts["scatter_thr"],
                    cliff_thr=opts["cliff_thr"], ground_tol=opts["ground_tol"],
                    # 裙边**始终**铺：外扩只覆盖左右两侧的 99° 视场，
                    # 转身看背后仍然是空的（实测：背后只有雾色和一条远处瓦楞）。
                    # 之前"有外扩就关裙边"是怕同心环带 —— 那个环带来自每格取不同色板，
                    # 现在改成单一色 + 低频起伏，不会再有环。
                    ground_radius=float(opts["ground_radius"]),
                    photo_wall=wall)

    tex = img_src
    if max(tex.size) > opts["tex_max"]:
        s = opts["tex_max"] / float(max(tex.size))
        tex = tex.resize((round(tex.size[0] * s), round(tex.size[1] * s)), Image.LANCZOS)
    buf = io.BytesIO()
    tex.save(buf, format="JPEG", quality=opts["jpeg_quality"], optimize=True)

    out = io.BytesIO()
    tmp = os.path.join(HERE, "_tmp_out.glb")
    splat_stat = res["stats"].get("splat") or {}
    cam_extras = {"camera": {"hfov": cam.hfov, "vfov": cam.vfov, "w": tw, "h": th,
                             "pitch": cam.pitch, "eyeHeight": opts["eye"], "yaw": 0.0,
                             "pointPx": splat_stat.get("point_px", 0.0),
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
        "outpaint": oinfo, "hfov_deg": round(math.degrees(hfov), 2),
        "hfov_source_deg": round(math.degrees(hfov0), 2),
        "texture_size": list(tex.size), "fov_source": fov_src,
        "camera": {"hfov_deg": round(math.degrees(cam.hfov), 2),
                   "vfov_deg": round(math.degrees(cam.vfov), 2),
                   "pitch_deg": round(math.degrees(cam.pitch), 2)},
        "calibration": {k: v for k, v in cal.items() if isinstance(v, (int, float, str, bool))},
        "repr": res["stats"]["_global"].get("repr", "mesh"),
        "photo_wall": wall, "photo_wall_reason": wall_why,
        "scatter": res["stats"]["_global"].get("scatter", 0.0),
        "cliff": res["stats"]["_global"].get("cliff", 0.0),
        "layers": res["stats"], "glb": info,
        "depth_ms": None, "elapsed_s": round(time.time() - t0, 2),
    }
    _cache[key] = (glb_bytes, manifest)
    if len(_cache) > 24:                       # 简单上限，避免长跑吃满内存
        _cache.pop(next(iter(_cache)))
    return glb_bytes, manifest
