#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""深度模型加载体检（Python 版）：限速 + 采样载入文案。

    python tools/probe_depth.py --url "http://127.0.0.1:8099/?w=alpine&instant=1" \
        --mbps 1.6 --latency 150 --out reports/depth_probe.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import cdp as C


def main() -> int:
    a = C.args_dict()
    url = a.get("url", "http://127.0.0.1:8099/?w=alpine&instant=1")
    mbps = float(a.get("mbps", "1.6"))
    lat = int(a.get("latency", "150"))
    wait = int(a.get("wait", "180"))
    out = a.get("out", "reports/depth_probe.json")
    profile = a.get("profile", "")

    timeline = []
    ready = False
    err = ""
    b = None
    try:
        b = C.launch(prefix="pwdepth-", w=900, h=600, edge=a.get("edge"),
                     gpu="d3d11", profile=profile or None, url="about:blank")
        c = b.cdp
        c.send("Page.enable")
        c.send("Runtime.enable")
        c.send("Network.enable")
        if mbps > 0:
            c.send("Network.emulateNetworkConditions", {
                "offline": False,
                "latency": lat,
                "downloadThroughput": mbps * 1024 * 1024 / 8,
                "uploadThroughput": mbps * 1024 * 1024 / 8 * 0.4,
                "connectionType": "cellular3g",
            })
        c.send("Page.navigate", {"url": url})
        t0 = time.monotonic()
        last = None
        while (time.monotonic() - t0) * 1000 < wait * 1000:
            st_raw = c.ev(
                "JSON.stringify({"
                "step: (document.getElementById('loaderSteps')||{}).textContent || '',"
                "bar: (document.getElementById('loaderBar')||{}).style ? "
                "(document.getElementById('loaderBar')||{}).style.width : '',"
                "ready: (document.body.dataset||{}).ready || '',"
                "err: (document.body.dataset||{}).err || ''"
                "})")
            try:
                s = json.loads(st_raw or "{}")
            except Exception:
                s = {}
            label = str(s.get("step") or "").strip()
            t = int((time.monotonic() - t0) * 1000)
            if label and (not last or last["label"] != label):
                timeline.append({"t": t, "label": label, "bar": s.get("bar")})
                last = {"label": label, "t": t}
            if s.get("ready") == "1":
                ready = True
                timeline.append({"t": t, "label": "[ready]", "bar": ""})
                break
            if s.get("err"):
                err = str(s["err"])
                break
            time.sleep(0.25)
    except Exception as e:
        err = str(e)
    finally:
        if b:
            b.close()

    max_stall = 0
    stall_label = ""
    for i, row in enumerate(timeline):
        end = timeline[i + 1]["t"] if i + 1 < len(timeline) else timeline[-1]["t"]
        dt = end - row["t"]
        if dt > max_stall:
            max_stall = dt
            stall_label = row["label"]
    distinct = len({r["label"] for r in timeline})
    result = {
        "url": url, "mbps": mbps, "latency": lat, "profile": profile,
        "ready": ready, "err": err, "samples": distinct,
        "maxStallMs": max_stall, "stallLabel": stall_label,
        "totalMs": timeline[-1]["t"] if timeline else 0,
        "timeline": timeline,
    }
    parent = os.path.dirname(os.path.abspath(out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    print("ready=%s  文案取样=%d 个  总耗时=%ss" %
          (ready, distinct, C.fmt_ms(result["totalMs"])))
    print("最长停留 %s → 「%s」" % (C.fmt_ms(max_stall), stall_label))
    for row in timeline:
        print("  %6ss  %s" % (("%.1f" % (row["t"] / 1000)).rjust(6), row["label"]))
    if err:
        print("错误：" + err, file=sys.stderr)
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
