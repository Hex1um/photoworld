#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""俯视原点：检查"照片壳 → 地形"的过渡是否平滑。

为什么单独一个脚本：
  * `window.__setCam` 在同一个会话里**第二次调用常常不生效**（HUD 坐标不动，截图重复），
    所以这里**每个机位单开一个浏览器进程**，一机位一进程；
  * 俯视必须真的"俯"下去：给一组固定高度（4/6/8/10/12m）和两个朝向（朝前 / 朝后），
    这样能同时看到壳还在（低空）与壳淡完（高空）两张画面，过渡问题一眼可见。

用法：
    python tools/v3_topdown.py --glb assets/world-v3-alpine.glb --out shots/topdown
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import cdp as C

# (标签, x, z, yaw, pitch, 高度m)
SHOTS = [
    ("a4_front", 0.0, -2.0, 0.0, -0.62, 4.0),
    ("b6_front", 0.0, -2.0, 0.0, -0.75, 6.0),
    ("c9_front", 0.0, -2.0, 0.0, -0.90, 9.0),
    ("d12_front", 0.0, -2.0, 0.0, -1.05, 12.0),
    ("e9_back", 0.0, -2.0, math.pi, -0.90, 9.0),
    ("f12_back", 0.0, -2.0, math.pi, -1.05, 12.0),
]


def one(base: str, glb: str, tag: str, x: float, z: float, yaw: float,
        pitch: float, fly: float, out: str, w: int, h: int, wait: int) -> bool:
    url = "%s/?pipe=v3&glb=%s&instant=1" % (base, glb)
    b = None
    try:
        b = C.launch(prefix="pwtd-", w=w, h=h, url="about:blank", no_cache=True, no_proxy=True)
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
            return False
        c.ev("(()=>{const p=window.__pw.player;p.state.flying=true;"
             "p.state.flyY=%f;p.state.y=%f;"
             "window.__setCam(%f,%f,%f,%f,undefined);"
             "p.state.flying=true;p.state.flyY=%f;p.state.y=%f;return 1})()"
             % (fly, fly, x, z, yaw, pitch, fly, fly))
        time.sleep(1.0)
        info = c.ev("JSON.stringify({y:+window.__pw.player.state.y.toFixed(2),"
                    "hide:+(window.__pwWorld.shellHide||0).toFixed(2),"
                    "exp:+(window.__pwWorld.exposure||0).toFixed(2)})")
        c.screenshot(os.path.join(out, tag + ".png"))
        print("  %-10s %s" % (tag, info))
        return True
    except Exception as e:
        print("  失败 %s：%s" % (tag, e))
        return False
    finally:
        if b:
            b.close()


def main() -> int:
    a = C.args_dict()
    base = a.get("base", "http://127.0.0.1:8199").rstrip("/")
    glb = a.get("glb", "assets/world-v3-alpine.glb")
    out = a.get("out", "shots/topdown")
    w = int(a.get("w", "1000"))
    h = int(a.get("h", "640"))
    wait = int(a.get("wait", "90"))
    os.makedirs(out, exist_ok=True)
    n = 0
    for (tag, x, z, yaw, pitch, fly) in SHOTS:
        if one(base, glb, tag, x, z, yaw, pitch, fly, out, w, h, wait):
            n += 1
    print("完成 %d/%d → %s" % (n, len(SHOTS), out))
    return 0 if n == len(SHOTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
