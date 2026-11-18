#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多视角对比图：一机位一进程截一张，再拼成带标签的对比板。

为什么一机位一进程：`window.__setCam` 在同一个浏览器会话里**第二次调用常常不生效**
（HUD 坐标不动、截图重复），连着换机位会得到同一帧 —— 这个坑已经骗过好几轮验收。

用法：
    python tools/v3_sheet.py --glb assets/world-v3-alpine.glb --out shots/sheet_alpine
    python tools/v3_sheet.py --glb ... --poses default --cols 3
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import cdp as C

# (标签, x, z, yaw, pitch, 飞行高度 or None)
POSES = [
    ("出生点 · 站定",        0.0,  0.0,  0.00,  0.02, None),
    ("出生点 · 抬头",        0.0,  0.0,  0.00,  0.55, None),
    ("左移 5m",             -5.0, -2.0,  1.20,  0.00, None),
    ("后退 6m · 回头看",     0.0,  4.0,  math.pi, 0.00, None),
    ("右移 6m",              6.0, -4.0, -1.20,  0.00, None),
    ("起飞 6m",              0.0, -3.0,  0.00, -0.30, 6.0),
    ("起飞 9m · 俯视",       0.0, -3.0,  0.00, -0.75, 9.0),
    ("起飞 12m · 俯视原点",  0.0, -3.0,  0.00, -1.00, 12.0),
    ("起飞 12m · 回头",      0.0, -3.0,  math.pi, -0.85, 12.0),
]


def shot(base, glb, tag, x, z, yaw, pitch, fly, out, w, h, wait) -> str | None:
    url = "%s/?pipe=v3&glb=%s&instant=1" % (base, glb)
    b = None
    try:
        b = C.launch(prefix="pws-", w=w, h=h, url="about:blank", no_cache=True, no_proxy=True)
        c = b.cdp
        C.enable_common(c, w, h)
        c.send("Page.navigate", {"url": url})
        t0 = time.monotonic()
        ok = False
        while (time.monotonic() - t0) < wait:
            if (c.ev("document.body.dataset.ready") or "") == "1":
                ok = True
                break
            time.sleep(0.4)
        if not ok:
            print("  超时：" + tag)
            return None
        expr = "(()=>{const p=window.__pw.player;p.state.flying=%s;" % ("true" if fly else "false")
        if fly:
            expr += "p.state.flyY=%f;p.state.y=%f;" % (fly, fly)
        expr += "window.__setCam(%f,%f,%f,%f,undefined);" % (x, z, yaw, pitch)
        if fly:
            expr += "p.state.flying=true;p.state.flyY=%f;p.state.y=%f;" % (fly, fly)
        expr += "return 1})()"
        c.ev(expr)
        time.sleep(1.0)
        info = json.loads(c.ev("JSON.stringify({y:+window.__pw.player.state.y.toFixed(2),"
                               "hide:+(window.__pwWorld.shellHide||0).toFixed(2)})") or "{}")
        path = os.path.join(out, "shot_%02d.png" % len(os.listdir(out)))
        c.screenshot(path)
        print("  %-16s y=%-5s hide=%-4s → %s" % (tag, info.get("y"), info.get("hide"),
                                                  os.path.basename(path)))
        return path
    except Exception as e:
        print("  失败 %s：%s" % (tag, e))
        return None
    finally:
        if b:
            b.close()


def compose(items, out_png, cols=3, cell_w=520, title_h=34):
    from PIL import Image, ImageDraw, ImageFont
    rows = (len(items) + cols - 1) // cols
    first = Image.open(items[0][1]).convert("RGB")
    cell_h = int(cell_w * first.height / first.width)
    sheet = Image.new("RGB", (cols * cell_w, rows * (cell_h + title_h)), (18, 20, 24))
    d = ImageDraw.Draw(sheet)
    font = None
    for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(cand):
            try:
                font = ImageFont.truetype(cand, 20)
                break
            except Exception:
                pass
    for i, (label, path) in enumerate(items):
        r, c = divmod(i, cols)
        im = Image.open(path).convert("RGB").resize((cell_w, cell_h))
        x0 = c * cell_w
        y0 = r * (cell_h + title_h)
        sheet.paste(im, (x0, y0 + title_h))
        d.rectangle([x0, y0, x0 + cell_w, y0 + title_h], fill=(30, 34, 40))
        d.text((x0 + 10, y0 + 7), "%d. %s" % (i + 1, label), fill=(230, 232, 236), font=font)
    sheet.save(out_png)
    return out_png


def main() -> int:
    a = C.args_dict()
    base = a.get("base", "http://127.0.0.1:8199").rstrip("/")
    glb = a.get("glb", "assets/world-v3-alpine.glb")
    out = a.get("out", "shots/sheet")
    w = int(a.get("w", "1040"))
    h = int(a.get("h", "650"))
    wait = int(a.get("wait", "90"))
    cols = int(a.get("cols", "3"))
    os.makedirs(out, exist_ok=True)
    items = []
    for pose in POSES:
        tag, x, z, yaw, pitch, fly = pose
        p = shot(base, glb, tag, x, z, yaw, pitch, fly, out, w, h, wait)
        if p:
            items.append((tag, p))
    if not items:
        print("没有截到任何画面", file=sys.stderr)
        return 1
    png = compose(items, os.path.join(out, "sheet.png"), cols=cols)
    print("对比图：" + png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
