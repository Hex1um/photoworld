#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多视角批量质检（语料 → 16 方位 + 俯视）—— Python 权威实现。

由 tools/multiview.mjs 迁移而来（v3.0.6）。逻辑逐条对齐原 JS：
  * 机位套餐 A/B/C(32) / PLAN_V2(32) / PLAN_V3(32) 与默认 N 视角圈；
  * 空洞 / 拉丝 / 清晰度指标（clarity 六项直接复用 clarity.py，与 JS 镜像 parity 锁定）；
  * 聚合口径：判据项取最差机位，展示项取均值；
  * 基线 A/B：captureDiff 复用 clarity.py，采集条件不一致时退出码 2。

浏览器链路复用 tools/cdp.py（标准库手写 WebSocket），PNG 解码用 Pillow；
旧 `node tools/multiview.mjs ...` 入口保留为薄包装，参数/输出/退出码透传。

    python tools/multiview.py --base http://127.0.0.1:8099 \
      --list corpus/sample.txt --views 16 --out shots/mv --json reports/mv.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import cdp  # noqa: E402
from clarity import (  # noqa: E402
    CLARITY_KEYS, HIGHER_IS_BETTER, clarity_metrics, capture_diff, _fixed,
)

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


# ---------------- 参数 ----------------
def parse_args(argv=None):
    a = argparse.ArgumentParser(description="多视角批量质检（Python）")
    a.add_argument("--base", default="http://127.0.0.1:8099")
    a.add_argument("--list", default="")
    a.add_argument("--corpus", default="")
    a.add_argument("--glb-list", default="")
    a.add_argument("--views", type=int, default=16)
    a.add_argument("--out", default="shots/mv")
    a.add_argument("--json", dest="json_out", default="reports/multiview.json")
    a.add_argument("--radius", type=float, default=11.0)
    a.add_argument("--center", default="0,-26")
    a.add_argument("--pitch", type=float, default=-0.10)
    a.add_argument("--wait", type=int, default=100)
    a.add_argument("--settle", type=int, default=260)
    a.add_argument("--w", type=int, default=960)
    a.add_argument("--h", type=int, default=600)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--plan", default="")
    a.add_argument("--query", default="")
    a.add_argument("--edge", default=cdp.DEFAULT_EDGE)
    a.add_argument("--gl", default="hw", choices=["hw", "sw"])
    a.add_argument("--baseline", default="")
    return a.parse_args(argv)


def make_plans():
    """复刻 RINGS_32 / PLAN_V2 / PLAN_V3 —— 与 JS 逐机位一致（含俯仰与眼高）。"""
    plan_v2 = []
    xs, zs = [-6, -2, 2, 6], [0, -1.5, -3, -5]
    for zi, z in enumerate(zs):
        for x in xs:
            for yaw in ([0, -0.44] if zi % 2 == 0 else [0.44, 0]):
                plan_v2.append(dict(ring="V", x=x, z=z, yaw=yaw,
                                    pitch=-0.04 + zi * 0.02, eye=None))

    plan_v3 = []
    pos = [[-5, 0], [-1.5, -3], [1.5, -3], [5, 0]]
    for x, z in pos:
        plan_v3 += [dict(ring="F", x=x, z=z, yaw=-0.35, pitch=-0.05),
                    dict(ring="F", x=x, z=z, yaw=0.35, pitch=-0.05)]
    for x, z in pos:
        plan_v3 += [dict(ring="S", x=x, z=z, yaw=1.40, pitch=-0.05),
                    dict(ring="S", x=x, z=z, yaw=-1.40, pitch=-0.05)]
    for x, z in pos:
        plan_v3 += [dict(ring="B", x=x, z=z, yaw=math.pi, pitch=-0.05),
                    dict(ring="B", x=x, z=z, yaw=math.pi * 0.9, pitch=0.04)]
    for x, z in pos:
        plan_v3 += [dict(ring="U", x=x, z=z, yaw=0, pitch=0.55),
                    dict(ring="U", x=x, z=z, yaw=math.pi, pitch=-0.55)]

    rings32 = []
    for i in range(12):
        rings32.append(dict(ring="A", a=(i / 12) * math.pi * 2, r=10, pitch=-0.10, eye=None))
    for i in range(12):
        rings32.append(dict(ring="B", a=(i / 12) * math.pi * 2, r=26, pitch=0.05, eye=None))
    for i in range(8):
        elevated = i % 2 == 1
        rings32.append(dict(ring="C", a=(i / 8) * math.pi * 2, r=75,
                            pitch=-0.35 if elevated else -0.02,
                            eye=14 if elevated else None))
    return rings32, plan_v2, plan_v3


# ---------------- PNG 解码（CDP 截图 8bit RGBA，无隔行） ----------------
def decode_png(buf: bytes):
    """返回 (w, h, bpp, data)，data 为原始通道字节（与 JS decodePNG 同布局）。"""
    if Image is not None:
        im = Image.open(io.BytesIO(buf))
        if im.mode not in ("RGBA", "RGB", "L"):
            im = im.convert("RGBA")
        bpp = {"RGBA": 4, "RGB": 3, "L": 1}[im.mode]
        return im.width, im.height, bpp, im.tobytes()
    # 无 Pillow 时用标准库 zlib 解 8bit 无隔行 PNG（与 JS decodePNG 同算法）
    import zlib
    import struct
    if buf[:4] != b"\x89PNG":
        raise ValueError("不是 PNG")
    off, w, h, depth, ctype, interlace, idat = 8, 0, 0, 8, 6, 0, []
    while off < len(buf):
        ln = struct.unpack_from(">I", buf, off)[0]
        typ = buf[off + 4:off + 8]
        chunk = buf[off + 8:off + 8 + ln]
        if typ == b"IHDR":
            w, h = struct.unpack_from(">II", chunk, 0)
            depth, ctype, interlace = chunk[8], chunk[9], chunk[12]
        elif typ == b"IDAT":
            idat.append(chunk)
        elif typ == b"IEND":
            break
        off += 12 + ln
    if depth != 8 or interlace != 0:
        raise ValueError("只支持 8bit 无隔行 PNG")
    bpp = {6: 4, 2: 3, 0: 1}.get(ctype, 0)
    if not bpp:
        raise ValueError("不支持的 colorType %d" % ctype)
    raw = zlib.decompress(b"".join(idat))
    stride = w * bpp
    out = bytearray(h * stride)
    prev = bytearray(stride)
    for y in range(h):
        ft = raw[y * (stride + 1)]
        line = raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)]
        cur = out[y * stride:(y + 1) * stride]
        for x in range(stride):
            a = cur[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            v = line[x]
            if ft == 1:
                v += a
            elif ft == 2:
                v += b
            elif ft == 3:
                v += (a + b) >> 1
            elif ft == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                v += a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            cur[x] = v & 0xFF
        prev = cur
    return w, h, bpp, bytes(out)


# ---------------- 指标 ----------------
VOID_TOL = 6
LUMA = (0.2126, 0.7152, 0.0722)


def _luma(r, g, b):
    return (LUMA[0] * r + LUMA[1] * g + LUMA[2] * b) / 255.0


def blob_stats(mask, w, h, min_area):
    seen = bytearray(w * h)
    stack = [0] * (w * h)
    count = max_area = total_area = 0
    for i in range(w * h):
        if not mask[i] or seen[i]:
            continue
        sp, area = 0, 0
        stack[sp] = i
        sp += 1
        seen[i] = 1
        while sp > 0:
            sp -= 1
            k = stack[sp]
            area += 1
            x, y = k % w, k // w
            for nk, ok in ((k - 1, x > 0), (k + 1, x < w - 1),
                           (k - w, y > 0), (k + w, y < h - 1)):
                if ok and mask[nk] and not seen[nk]:
                    seen[nk] = 1
                    stack[sp] = nk
                    sp += 1
        if area >= min_area:
            count += 1
            total_area += area
            if area > max_area:
                max_area = area
    return count, max_area, total_area


def image_metrics(png, ref_bg=None):
    w, h, bpp, data = png
    total = w * h
    L = [0.0] * total

    hist = [0] * (16 * 16 * 16)
    for i in range(total):
        o = i * bpp
        q = (data[o] >> 4) * 256 + (data[o + 1] >> 4) * 16 + (data[o + 2] >> 4)
        hist[q] += 1
    best_q, best_n = 0, -1
    for q in range(len(hist)):
        if hist[q] > best_n:
            best_n, best_q = hist[q], q
    mr = ((best_q >> 8) & 15) * 16 + 8
    mg = ((best_q >> 4) & 15) * 16 + 8
    mb = (best_q & 15) * 16 + 8

    void_n = dark_n = inner_n = inner_void = 0
    low_n = low_void = border_n = border_void = 0
    x0, x1 = int(w * 0.25), int(w * 0.75)
    y0, y1 = int(h * 0.25), int(h * 0.75)
    bx, by = int(w * 0.06), int(h * 0.06)
    for y in range(h):
        for x in range(w):
            o = (y * w + x) * bpp
            r, g, b = data[o], data[o + 1], data[o + 2]
            l = _luma(r, g, b)
            L[y * w + x] = l
            if l < 0.05:
                dark_n += 1
            near = (abs(r - ref_bg[0]) <= VOID_TOL and abs(g - ref_bg[1]) <= VOID_TOL
                    and abs(b - ref_bg[2]) <= VOID_TOL) if ref_bg else (
                    abs(r - mr) <= 4 and abs(g - mg) <= 4 and abs(b - mb) <= 4)
            is_void = near
            if near:
                o_r = o + bpp if x < w - 1 else o
                o_l = o - bpp if x > 0 else o
                o_u = o - w * bpp if y > 0 else o
                o_d = o + w * bpp if y < h - 1 else o
                vals = (l, _luma(data[o_r], data[o_r + 1], data[o_r + 2]),
                        _luma(data[o_l], data[o_l + 1], data[o_l + 2]),
                        _luma(data[o_u], data[o_u + 1], data[o_u + 2]),
                        _luma(data[o_d], data[o_d + 1], data[o_d + 2]))
                if max(vals) - min(vals) > 0.02:
                    is_void = False
            if is_void:
                void_n += 1
            if x0 <= x < x1 and y0 <= y < y1:
                inner_n += 1
                if is_void:
                    inner_void += 1
            if y >= h * 0.6:
                low_n += 1
                if is_void:
                    low_void += 1
            if x < bx or x >= w - bx or y < by or y >= h - by:
                border_n += 1
                if is_void:
                    border_void += 1

    vy = vx = detail = cells = 0.0
    for y in range(1, h, 2):
        for x in range(1, w, 2):
            k = y * w + x
            vy += abs(L[k] - L[k - w])
            vx += abs(L[k] - L[k - 1])
            detail += abs(L[k] - L[k - 1]) + abs(L[k] - L[k - w])
            cells += 1

    sw, sh = max(8, w >> 2), max(6, h >> 2)
    vmask, smask = bytearray(sw * sh), bytearray(sw * sh)
    scale = w / sw
    for y in range(sh):
        for x in range(sw):
            sx = min(w - 1, int((x + 0.5) * scale))
            sy = min(h - 1, int((y + 0.5) * scale))
            o = (sy * w + sx) * bpp
            r, g, b = data[o], data[o + 1], data[o + 2]
            col_near = (abs(r - ref_bg[0]) <= VOID_TOL and abs(g - ref_bg[1]) <= VOID_TOL
                        and abs(b - ref_bg[2]) <= VOID_TOL) if ref_bg else (
                        abs(r - mr) <= 6 and abs(g - mg) <= 6 and abs(b - mb) <= 6)
            flat = col_near
            if col_near:
                li = sy * w + sx
                a1 = L[li]
                a2 = L[li + 1 if sx < w - 1 else li]
                a3 = L[li - w if sy > 0 else li]
                if max(a1, a2, a3) - min(a1, a2, a3) > 0.02:
                    flat = False
            vmask[y * sw + x] = 1 if flat else 0
            if sy > 0 and sx > 0:
                k = sy * w + sx
                dv, dh = abs(L[k] - L[k - w]), abs(L[k] - L[k - 1])
                smask[y * sw + x] = 1 if (dv > 0.055 and dv > 2.2 * dh) else 0
    vb = blob_stats(vmask, sw, sh, 3)
    sb = blob_stats(smask, sw, sh, 3)

    run_max = 0
    for y in range(0, h, 2):
        run = 0
        for x in range(w):
            o = (y * w + x) * bpp
            inv = (abs(data[o] - ref_bg[0]) <= VOID_TOL
                   and abs(data[o + 1] - ref_bg[1]) <= VOID_TOL
                   and abs(data[o + 2] - ref_bg[2]) <= VOID_TOL) if ref_bg else (
                   abs(data[o] - mr) <= 4 and abs(data[o + 1] - mg) <= 4
                   and abs(data[o + 2] - mb) <= 4)
            if inv:
                run += 1
                if run > run_max:
                    run_max = run
            else:
                run = 0

    out = {
        "voidBlobs": vb[0],
        "voidBlobMax": _fixed(vb[1] / max(1, sw * sh), 4),
        "voidRunMax": _fixed(run_max / max(1, w), 4),
        "streakBlobs": sb[0],
        "streakBlobFrac": _fixed(sb[2] / max(1, sw * sh), 4),
        "voidFrac": _fixed(void_n / total, 4),
        "innerVoid": _fixed(inner_void / inner_n if inner_n else 0, 4),
        "lowVoid": _fixed(low_void / low_n if low_n else 0, 4),
        "borderVoid": _fixed(border_void / border_n if border_n else 0, 4),
        "darkFrac": _fixed(dark_n / total, 4),
        "stripe": _fixed(vy / max(1e-6, vx), 3),
        "detail": _fixed(detail / max(1, cells), 4),
    }
    out.update(clarity_metrics(L, w, h))
    return out


def mean_of(views, key):
    return _fixed(sum(v[key] for v in views) / len(views), 4)


def worst_of(views, key):
    return _fixed(max(v[key] for v in views), 4)


def aggregate(rec):
    v = rec["views"]
    return {
        "voidFrac": worst_of(v, "voidFrac"), "darkFrac": mean_of(v, "darkFrac"),
        "innerVoid": worst_of(v, "innerVoid"), "lowVoid": worst_of(v, "lowVoid"),
        "borderVoid": mean_of(v, "borderVoid"), "stripe": worst_of(v, "stripe"),
        "detail": mean_of(v, "detail"),
        **{k: mean_of(v, k) for k in CLARITY_KEYS},
        "worstInner": _fixed(max(x["innerVoid"] for x in v), 4),
        "worstLow": _fixed(max(x["lowVoid"] for x in v), 4),
        "worstStripe": _fixed(max(x["stripe"] for x in v), 3),
        "voidBlobs": max(x.get("voidBlobs", 0) for x in v),
        "streakBlobs": max(x.get("streakBlobs", 0) for x in v),
        "voidRunMax": _fixed(max(x.get("voidRunMax", 0) for x in v), 4),
        "voidBlobMax": _fixed(max(x.get("voidBlobMax", 0) for x in v), 4),
        "streakBlobFrac": _fixed(max(x.get("streakBlobFrac", 0) for x in v), 4),
    }


# ---------------- 主流程 ----------------
def run(a):
    base = a.base.rstrip("/")
    is_glb = bool(a.glb_list)
    if a.glb_list:
        images = [s.strip() for s in open(a.glb_list, encoding="utf-8").read().splitlines()
                  if s.strip() and not s.strip().startswith("#")]
    elif a.list:
        images = [s.strip() for s in open(a.list, encoding="utf-8").read().splitlines()
                  if s.strip() and not s.strip().startswith("#")]
    elif a.corpus:
        images = [a.corpus.replace("\\", "/").rstrip("/") + "/" + f
                  for f in sorted(os.listdir(a.corpus)) if f.lower().endswith((".jpg", ".jpeg"))]
    else:
        images = []
    if not images:
        print("没有输入图片：用 --list / --corpus / --glb-list 指定", file=sys.stderr)
        return 2
    if a.limit > 0:
        images = images[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    out_dir_json = os.path.dirname(a.json_out) or "."
    os.makedirs(out_dir_json, exist_ok=True)

    center_parts = a.center.split(",")
    try:
        cx = float(center_parts[0])
    except ValueError:
        cx = 0.0
    try:
        cz = float(center_parts[1])
    except ValueError:
        cz = -26.0
    rings32, plan_v2, plan_v3 = make_plans()
    capture = {"w": a.w, "h": a.h, "dpr": 1, "gl": a.gl, "views": a.views,
               "radius": a.radius, "center": center_parts, "pitch": a.pitch,
               "settle": a.settle, "query": a.query, "clarity": 1}

    gpu = "d3d11" if a.gl == "hw" else "swiftshader"
    browser = cdp.launch(prefix="pwmv-", port=random.randint(9600, 9899), w=a.w, h=a.h,
                         edge=a.edge, gpu=gpu, url="about:blank", no_proxy=True)
    results = []
    exit_code = 0
    try:
        cli = browser.cdp
        cdp.enable_common(cli, a.w, a.h, metrics=True)
        t0 = time.monotonic()
        for n, rel in enumerate(images):
            if is_glb:
                url = ("%s/?glb=%s&instant=1&cb=%d" % (base, cdp.urllib.parse.quote(rel, safe=""), int(time.time() * 1000)))
            else:
                name = os.path.splitext(os.path.basename(rel))[0]
                url = "%s/?img=%s&instant=1&name=%s" % (base, cdp.urllib.parse.quote(rel, safe=""),
                                                        cdp.urllib.parse.quote(name, safe=""))
            if a.query:
                url += "&" + a.query
            rec = {"image": rel, "views": [], "ok": False}
            ref_bg = None
            cli.send("Page.navigate", {"url": url})
            ts = time.monotonic()
            state = {}
            while (time.monotonic() - ts) * 1000 < a.wait * 1000:
                try:
                    s = cli.ev("JSON.stringify(Object.assign({}, document.body.dataset, "
                               "{ errs: (window.__errs || []).join(';') }))")
                    state = json.loads(s or "{}")
                except Exception:
                    state = {}
                if state.get("ready") == "1" or state.get("err"):
                    break
                time.sleep(0.4)
            if state.get("ready") == "1":
                try:
                    raw = cli.ev("JSON.stringify((window.__pw && window.__pw.world && "
                                 "window.__pw.world.fogColor) || null)")
                    col = json.loads(raw)
                    if isinstance(col, list) and len(col) == 3:
                        ref_bg = [max(0, min(255, round(float(v) * 255))) for v in col]
                        rec["refBg"] = ref_bg
                except Exception:
                    pass
            if state.get("ready") != "1":
                rec["error"] = state.get("err") or "ready 超时"
                rec["errs"] = state.get("errs", "")
                results.append(rec)
                print("✗ %s  %s" % (rel, rec["error"]))
                exit_code = 1
                continue
            rec["meta"] = {k: state.get(k, "" if k in ("objects", "version") else None)
                           for k in ("mode", "horizon", "expo", "sem", "objects", "version")}
            if state.get("dbg"):
                try:
                    rec["relief"] = json.loads(state["dbg"])
                except Exception:
                    pass

            plan_name = a.plan or ("v2" if is_glb else "rings")
            if a.views == 32:
                plan = plan_v3 if plan_name == "v3" else (plan_v2 if is_glb else rings32)
            else:
                plan = [dict(ring="A", a=(i / a.views) * math.pi * 2, r=a.radius,
                             pitch=a.pitch, eye=None) for i in range(a.views)]
            for i, v in enumerate(plan):
                ang = v.get("a", 0)
                x = v["x"] if "x" in v else cx + math.sin(ang) * v["r"]
                z = v["z"] if "z" in v else cz + math.cos(ang) * v["r"]
                yaw = v.get("yaw", -ang)
                eye = "undefined" if v.get("eye") is None else str(v["eye"])
                ok = cli.ev("window.__setCam ? window.__setCam(%s, %s, %s, %s, %s) : false" % (
                    round(x, 2), round(z, 2), round(yaw, 4), v["pitch"], eye))
                if not ok:
                    raise RuntimeError("页面缺少 window.__setCam")
                time.sleep(a.settle / 1000.0)
                shot = cli.send("Page.captureScreenshot", {"format": "png"})
                buf = base64.b64decode(shot["data"])
                base0 = os.path.splitext(os.path.basename(rel))[0]
                tag = (os.path.basename(os.path.dirname(rel)) if base0 == "world" else base0) \
                    if is_glb else os.path.splitext(os.path.basename(rel))[0]
                path_out = os.path.join(a.out, "%s_%s%02d.png" % (tag, v["ring"], i))
                with open(path_out, "wb") as f:
                    f.write(buf)
                m = image_metrics(decode_png(buf), ref_bg)
                vv = {"i": i, "ring": v["ring"], "r": v.get("r"),
                      "a": round(ang * 180 / math.pi), **m,
                      "file": path_out.replace("\\", "/")}
                rec["views"].append(vv)
            if a.views != 32:
                cli.ev("window.__setCam(%s, %s, 0, -1.0, 24)" % (cx, round(cz - a.radius * 0.55, 2)))
                time.sleep(a.settle / 1000.0)
                shot = cli.send("Page.captureScreenshot", {"format": "png"})
                buf = base64.b64decode(shot["data"])
                top_file = os.path.join(a.out, "%s_top.png" % os.path.splitext(os.path.basename(rel))[0])
                with open(top_file, "wb") as f:
                    f.write(buf)
                rec["top"] = {**image_metrics(decode_png(buf), ref_bg),
                              "file": top_file.replace("\\", "/")}
            rec["ok"] = True
            rec["agg"] = aggregate(rec)
            results.append(rec)
            el = int(time.monotonic() - t0)
            g = rec["agg"]
            print("✓ %s  void %s inner %s low %s stripe %s vB %s sB %s run %s detail %s "
                  "hf %s lap %s  [%d/%d, %ds]" % (
                      os.path.basename(rel), g["voidFrac"], g["innerVoid"], g["lowVoid"],
                      g["stripe"], g["voidBlobs"], g["streakBlobs"], g["voidRunMax"],
                      g["detail"], g["centerHF"], g["centerLapVar"], n + 1, len(images), el))
            with open(a.json_out, "w", encoding="utf-8") as f:
                json.dump({"base": base, "views": a.views, "capture": capture, "results": results},
                          f, indent=1, ensure_ascii=False)
    except Exception as e:
        print("失败：" + str(e), file=sys.stderr)
        exit_code = 1
    finally:
        browser.close()

    ok_results = [r for r in results if r["ok"]]
    if ok_results:
        avg = lambda k: _fixed(sum(r["agg"][k] for r in ok_results) / len(ok_results), 4)
        worst = sorted((dict(image=r["image"], **r["agg"]) for r in ok_results),
                       key=lambda r: r["worstLow"] + r["worstInner"], reverse=True)[:10]
        summary = {
            "images": len(results), "ok": len(ok_results), "failed": len(results) - len(ok_results),
            "views": a.views,
            "avg": {"voidFrac": avg("voidFrac"), "innerVoid": avg("innerVoid"),
                    "lowVoid": avg("lowVoid"), "borderVoid": avg("borderVoid"),
                    "stripe": avg("stripe"), "darkFrac": avg("darkFrac"),
                    "detail": avg("detail"), **{k: avg(k) for k in CLARITY_KEYS}},
            "worstInner": worst,
        }
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump({"base": base, "views": a.views, "capture": capture,
                       "summary": summary, "results": results}, f, indent=1, ensure_ascii=False)
        print("汇总 " + json.dumps(summary["avg"], ensure_ascii=False) + "  失败 " + str(summary["failed"]))
        print("最差 inner/low void：" + " | ".join(
            "%s %s/%s" % (os.path.basename(w["image"]), w["worstInner"], w["worstLow"])
            for w in worst[:5]))
        print("明细 → " + a.json_out + "，截图 → " + a.out)

    if a.baseline and os.path.isfile(a.json_out):
        with open(a.json_out, encoding="utf-8") as f:
            cur = json.load(f)
        if not os.path.exists(a.baseline):
            os.makedirs(os.path.dirname(a.baseline) or ".", exist_ok=True)
            with open(a.baseline, "w", encoding="utf-8") as f:
                json.dump(cur, f, indent=1, ensure_ascii=False)
            if exit_code:
                print("注意：本次有失败样本，基线不完整，建议修复后重跑", file=sys.stderr)
            print("基线已建立 → %s（同参数再跑一次即为 A/B）" % a.baseline)
        else:
            with open(a.baseline, encoding="utf-8") as f:
                baseline = json.load(f)
            diff = capture_diff(baseline.get("capture"), cur.get("capture"))
            if diff:
                print("采集条件不一致，拒绝对比：" + ", ".join(diff), file=sys.stderr)
                exit_code = 2
            else:
                keys = CLARITY_KEYS + ["detail", "voidFrac", "innerVoid", "lowVoid",
                                       "borderVoid", "darkFrac", "stripe"]
                map_of = lambda d: {r["image"]: r["agg"] for r in (d.get("results") or [])
                                    if r.get("ok") and r.get("agg")}
                bm, cm = map_of(baseline), map_of(cur)
                shared = [k for k in cm if k in bm]
                print("\nA/B 对比：基线 %s · 共同样本 %d 张 · 采集条件一致" % (
                    os.path.basename(a.baseline), len(shared)))
                print("指标".ljust(16) + "基线".ljust(11) + "当前".ljust(11) + "变化".ljust(11) + "改善/退化")
                for k in keys:
                    bs = cs = better = worse = 0
                    for img in shared:
                        b, c = bm[img].get(k), cm[img].get(k)
                        if not (isinstance(b, (int, float)) and isinstance(c, (int, float))):
                            continue
                        bs += b
                        cs += c
                        hi = HIGHER_IS_BETTER.get(k, True)
                        if (c > b) if hi else (c < b):
                            better += 1
                        elif (c < b) if hi else (c > b):
                            worse += 1
                    n = shared or [1]
                    pct = ((cs - bs) / bs) * 100 if bs else 0
                    print(k.ljust(16) + ("%.5f" % (bs / len(n))).ljust(11)
                          + ("%.5f" % (cs / len(n))).ljust(11)
                          + (("%+.1f%%" % pct)).ljust(11) + "%d/%d" % (better, worse))
    return exit_code


def main():
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
