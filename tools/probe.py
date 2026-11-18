#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行时探针（Python 版）：加载页面 → 等 ready → 执行任意表达式 → 打印结果。

    python tools/probe.py --url "http://127.0.0.1:8099/?w=winter&instant=1" \
        --expr "JSON.stringify(window.__pw.extender && window.__pw.extender.stats())"

    python tools/probe.py --exprfile tools/diag_jumps.js --url "...&dbgpos=1"
"""
from __future__ import annotations

import json
import os
import sys
import time

import cdp as C


def main() -> int:
    a = C.args_dict()
    url = a.get("url", "http://127.0.0.1:8099/")
    if a.get("exprfile"):
        with open(a["exprfile"], encoding="utf-8") as f:
            expr = f.read()
    else:
        expr = a.get("expr", "JSON.stringify({ ready: document.body.dataset.ready })")
    wait = int(a.get("wait", "90"))
    settle = int(a.get("settle", "3000"))
    w = int(a.get("w", "1440"))
    h = int(a.get("h", "900"))
    profile = a.get("profile")

    errs = []

    def on_msg(msg):
        if msg.get("method") == "Runtime.exceptionThrown":
            d = msg.get("params", {}).get("exceptionDetails", {})
            errs.append((d.get("exception") or {}).get("description") or d.get("text") or "unknown")
        elif msg.get("method") == "Runtime.consoleAPICalled" and msg["params"].get("type") == "error":
            errs.append(" ".join(str(x.get("value") or x.get("description") or "")
                                 for x in msg["params"].get("args", [])))

    b = None
    try:
        b = C.launch(prefix="pwprobe-", w=w, h=h, edge=a.get("edge"),
                     profile=profile, url="about:blank")
        c = b.cdp
        C.enable_common(c, w, h)
        c.on_event(on_msg)
        c.send("Page.navigate", {"url": url})

        t0 = time.monotonic()
        state = {}
        while (time.monotonic() - t0) * 1000 < wait * 1000:
            try:
                state = json.loads(c.ev("JSON.stringify(document.body.dataset)") or "{}")
            except Exception:
                state = {}
            if state.get("ready") in ("1", "0", 1, 0):
                break
            time.sleep(0.5)
        print("dataset=" + json.dumps(state, ensure_ascii=False))
        # v3.0.5 修：页面从未就绪（服务未启动 / 错误页）时不能再执行表达式并返回
        # 成功——错误页上 dataset 为空、多数表达式仍能求到值，会伪装成探测成功。
        # 保留 ready="0" 的合法语义（应用侧中途状态）。
        if state.get("ready") not in ("1", "0", 1, 0):
            print("页面在 %ss 内未就绪（url=%s），拒绝探测" % (wait, url), file=sys.stderr)
            return 1
        time.sleep(settle / 1000.0)

        r = c.eval_raw(expr)
        if r.get("exceptionDetails"):
            d = r["exceptionDetails"]
            desc = (d.get("exception") or {}).get("description") or d.get("text")
            print("表达式抛错：" + str(desc), file=sys.stderr)
            rc = 1
        else:
            v = (r.get("result") or {}).get("value")
            print("result=" + (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)))
            rc = 0

        if a.get("out"):
            time.sleep(int(a.get("shotdelay", "900")) / 1000.0)
            c.screenshot(a["out"])
            print("截图 " + a["out"])
        if errs:
            print("page-errors:\n  " + "\n  ".join(errs[:12]), file=sys.stderr)
        print("elapsed=" + "%.1fs" % ((time.monotonic() - t0)))
        return rc
    except Exception as e:
        print("失败：" + str(e), file=sys.stderr)
        return 1
    finally:
        if b:
            b.close()


if __name__ == "__main__":
    sys.exit(main())
