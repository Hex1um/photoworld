#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 定点截图：出生点 / 侧视 / 背后 / 起飞 —— 用来看"用户会看到什么"。

为什么单独一个工具：multiview 的机位是给指标用的（960×600、固定套餐），
而"起飞是不是全白""转头有没有大圆圈""侧看是不是拉伸"这类判断必须用
**接近真实使用**的机位与分辨率人眼看。这里就是这几个固定机位。

    python tools/v3_shots.py --glb assets/world-v3-alpine.glb --out shots/v3shots
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
    ("spawn",   0.0,  0.0,  0.0,  0.0,   None),
    ("left",   -4.0, -2.0,  1.45, 0.0,   None),
    ("back",    0.0, -2.0,  math.pi, -0.02, None),
    ("fly20",   0.0, -4.0,  math.pi, -0.55, 20.0),
    ("fly60",   0.0, -6.0,  0.0,   -0.95, 60.0),
    ("walk6",  -6.0, -6.0,  0.0,  0.0,   None),
]


def main() -> int:
    a = C.args_dict()
    base = a.get("base", "http://127.0.0.1:8199").rstrip("/")
    glb = a.get("glb", "assets/world-v3-alpine.glb")
    out = a.get("out", "shots/v3shots")
    w = int(a.get("w", "1280"))
    h = int(a.get("h", "800"))
    wait = int(a.get("wait", "120"))
    os.makedirs(out, exist_ok=True)

    url = "%s/?glb=%s&instant=1" % (base, glb)
    print("打开", url)
    b = None
    try:
        b = C.launch(prefix="pwv3-", w=w, h=h, edge=a.get("edge"),
                     profile=a.get("profile"), url="about:blank", no_cache=True)
        c = b.cdp
        C.enable_common(c, w, h)
        c.send("Page.navigate", {"url": url})
        t0 = time.monotonic()
        state = {}
        while (time.monotonic() - t0) < wait:
            s = c.ev("JSON.stringify(Object.assign({}, document.body.dataset,"
                     "{ errs: (window.__errs || []).join(';') }))") or "{}"
            try:
                state = json.loads(s)
            except Exception:
                state = {}
            if state.get("ready") == "1":
                break
            time.sleep(0.4)
        print("state=" + json.dumps(state, ensure_ascii=False))
        if state.get("ready") != "1":
            print("进世界超时", file=sys.stderr)
            return 1

        out_files = []
        for tag, x, z, yaw, pitch, fly in POSES:
            expr = ("(() => { const p = window.__pw && window.__pw.player;"
                    " if (!p) return false;"
                    " window.__setCam(%f, %f, %f, %f, undefined);" % (x, z, yaw, pitch))
            if fly is not None:
                expr += (" p.state.flying = true; p.state.flyY = %f; p.state.y = %f;"
                         % (fly, fly))
            else:
                expr += " p.state.flying = false;"
            expr += " return true; })()"
            ok = c.ev(expr)
            time.sleep(0.9)
            path = os.path.join(out, "%s.png" % tag)
            c.screenshot(path)
            out_files.append(path)
            print("  截图 %-6s x=%g z=%g yaw=%.2f pitch=%.2f fly=%s"
                  % (tag, x, z, yaw, pitch, fly))
        print(json.dumps({"files": out_files}, ensure_ascii=False))
        return 0
    except Exception as e:
        print("失败：" + str(e), file=sys.stderr)
        return 1
    finally:
        if b:
            b.close()


if __name__ == "__main__":
    raise SystemExit(main())
