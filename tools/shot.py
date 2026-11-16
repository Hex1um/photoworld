#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实等待式截图（Python 版）。

    python tools/shot.py --url "http://127.0.0.1:8099/?w=alpine" --out shot.png [--wait 90]
"""
from __future__ import annotations

import json
import sys
import time

import cdp as C


def main() -> int:
    a = C.args_dict()
    url = a.get("url", "http://127.0.0.1:8099/")
    out = a.get("out", "shot.png")
    wait = int(a.get("wait", "90"))
    w = int(a.get("w", "1440"))
    h = int(a.get("h", "900"))

    b = None
    try:
        b = C.launch(prefix="pwshot-", w=w, h=h, edge=a.get("edge"),
                     profile=a.get("profile"), url="about:blank", no_cache=True)
        c = b.cdp
        C.enable_common(c, w, h)
        c.send("Page.navigate", {"url": url})

        t0 = time.monotonic()
        state = "{}"
        while (time.monotonic() - t0) * 1000 < wait * 1000:
            state = c.ev(
                "JSON.stringify(Object.assign({}, document.body.dataset, "
                "{ errs: (window.__errs || []).join(';') }))") or "{}"
            try:
                st = json.loads(state)
            except Exception:
                st = {}
            if st.get("ready") in ("1", "0", 1, 0):
                break
            time.sleep(0.5)
        print("state=" + state)

        time.sleep(0.6)
        c.screenshot(out)
        print("截图保存 %s  %s" % (out, C.fmt_ms(int((time.monotonic() - t0) * 1000))))
        return 0
    except Exception as e:
        print("失败：" + str(e), file=sys.stderr)
        return 1
    finally:
        if b:
            b.close()


if __name__ == "__main__":
    sys.exit(main())
