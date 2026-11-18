#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B 多机位截图台（Python 版）：一次启动浏览器，在同一页面里连续 enterWorld 多张图。

    python tools/abshots.py --base http://127.0.0.1:8199 --list testlist.txt --out C:/tmp/ab/before --hw
    python tools/abshots.py --base ... --list ... --out ... --only 00_grass

为什么不用 tools/shot.py 循环：每次 shot 都要新 profile → 重新下载 45MB 模型 + WASM 初始化，
20s/张全花在冷启动上；这里复用同一次会话，第二张起只需推理时间。

机位默认三个：front 正视 / side 右移 8m 暴露遮挡区 / back 后退 3m 转身 180°。
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse

import cdp as C

POSE = {
    "front": {"x": 0, "z": 0, "yaw": 0, "pitch": -0.02},
    "side": {"x": 8, "z": 0, "yaw": 0, "pitch": -0.02},
    "side16": {"x": 16, "z": 0, "yaw": 0, "pitch": -0.02},
    "back": {"x": 0, "z": -3, "yaw": 3.141592653589793, "pitch": -0.02},
    "up": {"x": 0, "z": 0, "yaw": 0, "pitch": 0.28},
}


def _jstr(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def main() -> int:
    a = C.args_dict()
    if not a.get("list"):
        print("需要 --list", file=sys.stderr)
        return 2
    base = a.get("base", "http://127.0.0.1:8199").rstrip("/")
    out = a.get("out", "C:/tmp/ab/out")
    only = a.get("only", "")
    per_timeout = int(a.get("wait", "120"))
    w, h = int(a.get("w", "1280")), int(a.get("h", "800"))
    views = a.get("views", "front,side,back").split(",")
    append = ("&" + a["append"].lstrip("&")) if a.get("append") else ""
    hw = "--hw" in sys.argv
    glb_mode = a.get("mode") == "glb"

    images = []
    with open(a["list"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            src = parts[0]
            name = parts[1] if len(parts) > 1 else os.path.splitext(os.path.basename(src))[0]
            if only and only not in name:
                continue
            images.append({"src": src, "name": name})
    os.makedirs(out, exist_ok=True)

    rows = []
    b = None
    try:
        b = C.launch(prefix="pwsess-", w=w, h=h, edge=a.get("edge"),
                     gpu="hw" if hw else "swiftshader", no_cache=True, url="about:blank")
        c = b.cdp
        C.enable_common(c, w, h)
        c.send("Page.navigate", {"url": "%s/?instant=1%s&cb=%d" % (base, append, int(time.time() * 1000))})

        ok = False
        t0 = time.monotonic()
        while (time.monotonic() - t0) * 1000 < 120 * 250:
            if c.ev("!!(window.__pw && window.__pw.enterWorld)", False):
                ok = True
                break
            time.sleep(0.25)
        if not ok:
            raise C.CDPError("__pw 未就绪")
        c.ev("(() => { const l=document.getElementById('landing'); if(l){l.style.display='none';} return 1; })()", False)

        for it in images:
            t0 = time.monotonic()
            if glb_mode:
                enc = urllib.parse.quote(it["src"], safe="")
                c.send("Page.navigate", {"url": "%s/?glb=%s&instant=1%s&cb=%d"
                                          % (base, enc, append, int(time.time() * 1000))})
                time.sleep(0.3)
                for _ in range(200):
                    if c.ev("!!(window.__pw && window.__pw.world)", False):
                        break
                    time.sleep(0.25)
            else:
                expr = ('(() => { document.body.dataset.ready="0"; document.body.dataset.mode=""; '
                        'window.__errs=[]; window.__pw.enterWorld({name:%s, sub:"", src:%s, thumb:%s}); '
                        'return 1; })()') % (_jstr(it["name"]), _jstr(it["src"]), _jstr(it["src"]))
                c.ev(expr, False)

            st = {}
            deadline = time.monotonic() + per_timeout
            while time.monotonic() < deadline:
                raw = c.ev('JSON.stringify(Object.assign({}, document.body.dataset, '
                           '{ errs: (window.__errs||[]).join(";") }))', False)
                try:
                    st = json.loads(raw or "{}")
                except Exception:
                    st = {}
                if st.get("ready") == "1" or st.get("errs"):
                    break
                time.sleep(0.4)
            load_ms = int((time.monotonic() - t0) * 1000)

            for v in views:
                p = POSE.get(v, POSE["front"])
                expr = ('(() => { const pl=window.__pw.player; if(!pl) return 0; '
                        'pl.state.x=%s; pl.state.z=%s; pl.state.yaw=%s; pl.state.pitch=%s; '
                        'pl.state.vx=0; pl.state.vz=0; pl.state.bob=0; return 1; })()'
                        ) % (p["x"], p["z"], p["yaw"], p["pitch"])
                c.ev(expr, False)
                time.sleep(0.5)
                c.screenshot(os.path.join(out, "%s__%s.png" % (it["name"], v)))

            mode = st.get("mode") or "-"
            errs = st.get("errs") or ""
            tex = ("%sx%s" % (st.get("texW"), st.get("texH"))) if st.get("texW") else "-"
            rows.append({"name": it["name"], "mode": mode, "loadS": "%.1f" % (load_ms / 1000.0),
                         "errs": errs, "tex": tex})
            print("%-22s mode=%-7s %.1fs %s" % (it["name"], mode, load_ms / 1000.0,
                                                 "ERR " + errs if errs else ""))
    except Exception as e:
        print("失败：" + str(e), file=sys.stderr)
        return 1
    finally:
        if b:
            b.close()
        if rows:
            with open(os.path.join(out, "_summary.json"), "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=1)
            print("\n共 %d 张，截图 %d 张 → %s" % (len(rows), len(rows) * len(views), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
