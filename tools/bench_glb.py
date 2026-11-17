#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真值基准（Python 版）：把 GLB 场景当"摄影棚"，拍一张照片 → 喂进重建 → 逐视角对比。

    python tools/bench_glb.py --glb assets/world-alpine.glb \\
        --poses "0,0,0,-0.02|5,-8,0.5,-0.02|-6,-14,-0.6,0.05|0,-22,3.14,-0.02"

像素解码用 Pillow（替代 JS 版 jpeg-js），PSNR / 六项指标公式与 multiview / clarity 同源。
流程：真值 A（作照片）→ 重建自洽 → 每个 hold-out 视角 __setCam 两边各拍一张 →
比较 PSNR 与 void/dark/stripe/detail 指标，输出 JSON。
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import cdp as C

try:
    from PIL import Image
except ImportError:  # 解码失败时给出可操作提示
    Image = None


def _fixed(x: float, digits: int) -> float:
    scale = 10 ** digits
    return math.floor(x * scale + 0.5) / scale


def decode_jpeg(buf: bytes):
    """返回 (width, height, rgba_bytes)，与 jpeg-js decode(useTArray) 布局一致。"""
    if Image is None:
        raise RuntimeError("需要 Pillow：pip install pillow")
    im = Image.open(io.BytesIO(buf)).convert("RGBA")
    return im.width, im.height, im.tobytes()


def psnr(a, b) -> float:
    n = min(len(a[2]), len(b[2]))
    da, db = a[2], b[2]
    se = 0
    i = 0
    while i < n:
        for k in range(3):
            d = da[i + k] - db[i + k]
            se += d * d
        i += 4
    mse = se / max(1e-9, (n / 4) * 3)
    return 99.0 if mse <= 1e-6 else 10 * math.log10((255 * 255) / mse)


def view_metrics(img) -> dict:
    w, h, data = img
    hist = [0] * (16 * 16 * 16)
    for i in range(0, w * h * 4, 4):
        hist[(data[i] >> 4) * 256 + (data[i + 1] >> 4) * 16 + (data[i + 2] >> 4)] += 1
    bq = 0
    bn = -1
    for q in range(len(hist)):
        if hist[q] > bn:
            bn, bq = hist[q], q
    mr = ((bq >> 8) & 15) * 16 + 8
    mg = ((bq >> 4) & 15) * 16 + 8
    mb = (bq & 15) * 16 + 8
    vy = vx = det = 0.0
    cells = void_n = dark = inner_n = inner_v = low_n = low_v = 0
    for y in range(h):
        for x in range(w):
            o = (y * w + x) * 4
            r, g, bl = data[o], data[o + 1], data[o + 2]
            l = (0.2126 * r + 0.7152 * g + 0.0722 * bl) / 255
            if l < 0.05:
                dark += 1
            is_void = abs(r - mr) <= 4 and abs(g - mg) <= 4 and abs(bl - mb) <= 4
            if is_void:
                void_n += 1
            if w * 0.25 <= x < w * 0.75 and h * 0.25 <= y < h * 0.75:
                inner_n += 1
                if is_void:
                    inner_v += 1
            if y >= h * 0.6:
                low_n += 1
                if is_void:
                    low_v += 1
            if y > 0 and x > 0 and ((x + y) & 1) == 0:
                o_l = (y * w + x - 1) * 4
                o_u = ((y - 1) * w + x) * 4
                l_l = (0.2126 * data[o_l] + 0.7152 * data[o_l + 1] + 0.0722 * data[o_l + 2]) / 255
                l_u = (0.2126 * data[o_u] + 0.7152 * data[o_u + 1] + 0.0722 * data[o_u + 2]) / 255
                vy += abs(l - l_u)
                vx += abs(l - l_l)
                det += abs(l - l_u) + abs(l - l_l)
                cells += 1
    return {
        "voidFrac": _fixed(void_n / (w * h), 4),
        "innerVoid": _fixed(inner_v / inner_n if inner_n else 0, 4),
        "lowVoid": _fixed(low_v / low_n if low_n else 0, 4),
        "darkFrac": _fixed(dark / (w * h), 4),
        "stripe": _fixed(vy / max(1e-6, vx), 3),
        "detail": _fixed(det / max(1, cells), 4),
    }


def _ready(c):
    r = c.eval_raw("(document.body.dataset||{}).ready === '1'")
    return (r.get("result") or {}).get("value") is True


def open_page(c, url: str, wait_ms: int, extra_check=None):
    c.send("Page.navigate", {"url": url})
    for _ in range(int(math.ceil(wait_ms / 250.0))):
        time.sleep(0.25)
        try:
            if _ready(c):
                break
        except Exception:
            pass
    if extra_check:
        try:
            c.ev(extra_check)
        except Exception:
            pass
    time.sleep(0.6)


def shot(c, path: str, w: int, h: int):
    data = c.screenshot(path, fmt="jpeg", quality=92)
    return decode_jpeg(data)


def main() -> int:
    a = C.args_dict()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = a.get("base", "http://127.0.0.1:8099")
    glb = a.get("glb", "assets/world-alpine.glb")
    poses = [[float(v) for v in seg.split(",")]
             for seg in a.get("poses", "0,0,0,-0.02|5,-8,0.5,-0.02|-6,-14,-0.6,0.05|0,-22,3.14,-0.02").split("|")]
    w, h = int(a.get("w", "640")), int(a.get("h", "420"))
    out = a.get("out", "reports/bench_glb.json")
    shots = a.get("shots", "shots/bench")
    wait = int(a.get("wait", "13000"))
    q = a.get("q", "").strip()

    os.makedirs(shots, exist_ok=True)
    bench_dir = os.path.join(root, "corpus", "bench")
    os.makedirs(bench_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(glb))[0]

    cam_expr = lambda p: "window.__setCam ? window.__setCam(%s) : false" % ", ".join(str(v) for v in p)

    truth = rec = None
    try:
        truth = C.launch(prefix="pwbench-t-", port=9507, w=w, h=h, gpu="d3d11", url="about:blank")
        rec = C.launch(prefix="pwbench-r-", port=9514, w=w, h=h, gpu="d3d11", url="about:blank")
        tc, rc = truth.cdp, rec.cdp
        C.enable_common(tc, w, h)
        C.enable_common(rc, w, h)

        qs = ("&" + q) if q else ""
        import urllib.parse
        open_page(tc, "%s/?glb=%s&instant=1%s" % (base, urllib.parse.quote(glb, safe=""), qs), wait)
        photo_file = os.path.join(bench_dir, name + "-a.jpg")
        photo_img = shot(tc, photo_file, w, h)
        print("照片（真值 A）已存：" + os.path.relpath(photo_file, root))

        open_page(rc, "%s/?img=%s&instant=1%s"
                  % (base, urllib.parse.quote("corpus/bench/" + name + "-a.jpg", safe=""), qs), wait)
        rows = [{"pose": "A (输入视角)", "truthFile": os.path.relpath(photo_file, root), "recon": None}]
        rec_a = shot(rc, os.path.join(shots, name + "_A_rec.jpg"), w, h)
        t_a = shot(tc, os.path.join(shots, name + "_A_truth.jpg"), w, h)
        rows[0]["psnr"] = _fixed(psnr(rec_a, t_a), 2)
        rows[0]["metrics"] = view_metrics(rec_a)
        print("A 自洽：PSNR %s dB" % rows[0]["psnr"])

        for i in range(1, len(poses)):
            p = poses[i]
            tc.ev(cam_expr(p))
            time.sleep(0.7)
            t_img = shot(tc, os.path.join(shots, "%s_P%d_truth.jpg" % (name, i)), w, h)
            rc.ev(cam_expr(p))
            time.sleep(0.9)
            r_img = shot(rc, os.path.join(shots, "%s_P%d_rec.jpg" % (name, i)), w, h)
            r = {
                "pose": ",".join(str(v) for v in p),
                "psnr": _fixed(psnr(r_img, t_img), 2),
                "truth": view_metrics(t_img), "recon": view_metrics(r_img),
                "files": ["%s_P%d_truth.jpg" % (name, i), "%s_P%d_rec.jpg" % (name, i)],
            }
            rows.append(r)
            print("  P%d [%s]  PSNR %s dB  stripe 真值 %s → 重建 %s"
                  % (i, r["pose"], r["psnr"], r["truth"]["stripe"], r["recon"]["stripe"]))
    except Exception as e:
        print("失败：" + str(e), file=sys.stderr)
        return 1
    finally:
        if truth:
            truth.close()
        if rec:
            rec.close()

    novel = [r for r in rows if r.get("psnr") is not None and not str(r["pose"]).startswith("A")]
    mean = sum(r["psnr"] for r in novel) / len(novel) if novel else 0
    report = {
        "glb": glb, "size": [w, h],
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputViewPsnr": rows[0].get("psnr") if rows else None,
        "novelViewPsnr": _fixed(mean, 2), "rows": rows,
    }
    out_full = os.path.join(root, out)
    os.makedirs(os.path.dirname(out_full) or ".", exist_ok=True)
    with open(out_full, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("\n输入视角 PSNR %s dB · 新视角平均 PSNR %s dB" % (report["inputViewPsnr"], report["novelViewPsnr"]))
    print("明细 → " + out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
