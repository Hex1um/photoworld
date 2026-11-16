#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轨道 / 侧拍质检（Python 版）：绕世界中心拍 N 个角度 + 一张俯视侧拍。

    python tools/orbit_shot.py --url "http://127.0.0.1:8099/?w=alpine&instant=1" \
        --out shots/alpine --angles 8 --radius 14 --center 0,-30 --pitch -0.18
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import cdp as C


def main() -> int:
    a = C.args_dict()
    url = a.get("url", "http://127.0.0.1:8099/?instant=1")
    outdir = a.get("out", "shots/orbit")
    angles = int(a.get("angles", "8"))
    radius = float(a.get("radius", "14"))
    try:
        center = [float(x) for x in a.get("center", "0,-30").split(",")]
    except Exception:
        center = [0.0, -30.0]
    pitch = float(a.get("pitch", "-0.18"))
    wait = int(a.get("wait", "120"))
    w = int(a.get("w", "1280"))
    h = int(a.get("h", "800"))
    prefix = a.get("prefix", "orbit")
    top_h = float(a.get("topheight", "26"))
    os.makedirs(outdir, exist_ok=True)

    b = None
    try:
        b = C.launch(prefix="pworbit-", w=w, h=h, edge=a.get("edge"), url="about:blank")
        c = b.cdp
        C.enable_common(c, w, h)
        c.send("Page.navigate", {"url": url})

        t0 = time.monotonic()
        ready = False
        state = "{}"
        while (time.monotonic() - t0) * 1000 < wait * 1000:
            state = c.ev(
                "JSON.stringify(Object.assign({}, document.body.dataset, "
                "{ errs: (window.__errs || []).join(';') }))") or "{}"
            try:
                st = json.loads(state)
            except Exception:
                st = {}
            if st.get("ready") == "1":
                ready = True
                break
            if st.get("err"):
                break
            time.sleep(0.5)
        print("state=" + state)
        if not ready:
            raise C.CDPError("世界未就绪（见上方 state）")

        cx = center[0] if len(center) > 0 and math.isfinite(center[0]) else 0.0
        cz = center[1] if len(center) > 1 and math.isfinite(center[1]) else -30.0
        for i in range(angles):
            ang = (i / angles) * math.pi * 2
            x = cx + math.sin(ang) * radius
            z = cz + math.cos(ang) * radius
            yaw = -ang
            ok = c.ev("window.__setCam ? window.__setCam(%s, %s, %s, %s) : false"
                      % (x, z, yaw, pitch))
            if not ok:
                raise C.CDPError("页面缺少 window.__setCam，请确认 app.js 已更新")
            time.sleep(0.32)
            f = os.path.join(outdir, "%s_%02d.png" % (prefix, i))
            c.screenshot(f)
            print("  %s  (%d°)" % (f, round(ang * 180 / math.pi)))

        c.ev("window.__setCam(%s, %s, 0, -1.0, %s)" % (cx, cz - radius * 0.6, top_h))
        time.sleep(0.32)
        f = os.path.join(outdir, "%s_top.png" % prefix)
        c.screenshot(f)
        print("  %s  (俯视)" % f)
        return 0
    except Exception as e:
        print("失败：" + str(e), file=sys.stderr)
        return 1
    finally:
        if b:
            b.close()


if __name__ == "__main__":
    sys.exit(main())
