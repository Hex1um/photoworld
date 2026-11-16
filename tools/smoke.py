#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHOTOWORLD 冒烟测试（Python 版）：一条命令验证页面能否进世界。

    python tools/smoke.py --base http://127.0.0.1:8137
    python tools/smoke.py --shots /tmp/smoke --only alpine
"""
from __future__ import annotations

import json
import os
import sys
import time

import cdp as C

CASES = [
    {"name": "alpine", "url": "/?w=alpine&instant=1"},
    {"name": "desert", "url": "/?w=desert&instant=1"},
    {"name": "asset-glb", "url": "/?glb=assets/world-alpine.glb&instant=1"},
    {"name": "ring-on", "url": "/?w=alpine&ring=1&instant=1"},
    {"name": "skip-sky", "url": "/?w=alpine&skip=sky&instant=1"},
]


def main() -> int:
    a = C.args_dict()
    base = a.get("base", "http://127.0.0.1:8099").rstrip("/")
    wait = int(a.get("wait", "90"))
    shots = a.get("shots", "")
    only = a.get("only", "")
    cases = [c for c in CASES if not only or only in c["name"]]

    rows = []
    b = None
    try:
        if shots:
            os.makedirs(shots, exist_ok=True)
        b = C.launch(prefix="pwsmoke-", w=1280, h=800, edge=a.get("edge"),
                     gpu="swiftshader", url="about:blank", no_proxy=True)
        c = b.cdp
        C.enable_common(c, 1280, 800)
        for case in cases:
            t0 = time.monotonic()
            c.send("Page.navigate", {"url": base + case["url"]})
            st = {}
            while (time.monotonic() - t0) * 1000 < wait * 1000:
                raw = c.ev(
                    "JSON.stringify(Object.assign({}, document.body && document.body.dataset, "
                    "{errs: (window.__errs||[]).join(';')}))") or "{}"
                try:
                    st = json.loads(raw)
                except Exception:
                    st = {}
                if st.get("ready") == "1" or st.get("err"):
                    break
                time.sleep(0.3)
            ms = int((time.monotonic() - t0) * 1000)
            if shots:
                try:
                    c.screenshot(os.path.join(shots, case["name"] + ".png"))
                except Exception:
                    pass
            ok = st.get("ready") == "1" and not st.get("err") and not str(st.get("errs", "")).strip()
            rows.append({"name": case["name"], "ok": ok, "ms": ms,
                         "mode": st.get("mode") or "-", "err": st.get("err") or st.get("errs") or ""})
    except Exception as e:
        print("冒烟失败：" + str(e), file=sys.stderr)
        rows.append({"name": "(harness)", "ok": False, "ms": 0, "mode": "-", "err": str(e)})
    finally:
        if b:
            b.close()

    def pad(s, n):
        return str(s).ljust(n)

    print("\n" + pad("用例", 14) + pad("结果", 6) + pad("耗时", 9) + pad("mode", 10) + "错误")
    for r in rows:
        print(pad(r["name"], 14) + pad("PASS" if r["ok"] else "FAIL", 6)
              + pad(C.fmt_ms(r["ms"]), 9) + pad(r["mode"], 10) + str(r["err"]))
    bad = sum(1 for r in rows if not r["ok"])
    print("\n%d/%d 通过%s" % (len(rows) - bad, len(rows), "，%d 例失败" % bad if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
