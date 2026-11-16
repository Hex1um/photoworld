#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用 CDP 探针（Python 版）。

    python tools/cdp_eval.py --url "http://127.0.0.1:8199/probe.html" \
        --expr "JSON.stringify(navigator.gpu?1:0)"

    python tools/cdp_eval.py --url "..." --exprfile tools/gpu_probe.js \
        --wait 60 --hw --shot shots/gpu.png

原 cdp_eval.mjs 只保留薄包装；业务逻辑在此。参数/输出/退出码与旧入口一致。
"""
from __future__ import annotations

import sys

import cdp as C


def main() -> int:
    a = C.args_dict()
    url = a.get("url", "about:blank")
    wait = int(a.get("wait", "30"))
    headless = not a.get("headful")
    expr = None
    if a.get("exprfile"):
        with open(a["exprfile"], encoding="utf-8") as f:
            expr = f.read()
    expr = expr if expr is not None else a.get("expr", "document.title")

    extra = a.get("flags", "").split("|") if a.get("flags") else None
    b = None
    try:
        b = C.launch(
            prefix="pwcdp-", w=1280, h=800, edge=a.get("edge"),
            headless=headless, gpu="hw" if a.get("hw") else "swiftshader",
            extra=extra, url=url, no_cache=True,
        )
        c = b.cdp
        C.enable_common(c, 1280, 800, metrics=False)
        c.send("Page.navigate", {"url": url})
        import time
        time.sleep(1.2)
        wrapped = (
            "(async () => { try { return JSON.stringify(await (async () => { return %s; })()); }"
            " catch (e) { return JSON.stringify({ error: String(e && e.stack || e) }); } })()"
        ) % expr
        r = c.eval_raw(wrapped, timeout_ms=wait * 1000)
        out = (r.get("result") or {}).get("value") or "(空)"
        print(out)
        if a.get("shot"):
            c.screenshot(a["shot"])
            print("截图 → " + a["shot"])
        return 0
    except Exception as e:
        print("失败：" + str(e), file=sys.stderr)
        return 1
    finally:
        if b:
            b.close()


if __name__ == "__main__":
    sys.exit(main())
