#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐层定位（Python 版）：把 world.meshes 里的某一层置空后截图。

    python tools/probe_layer.py --url "http://127.0.0.1:8099/?img=...&instant=1" \
        --outdir shots/layer
"""
from __future__ import annotations

import os
import sys
import time

import cdp as C


def main() -> int:
    a = C.args_dict()
    url = a.get("url", "http://127.0.0.1:8099/?w=alpine&instant=1")
    outdir = a.get("outdir", "shots/layer")
    w = int(a.get("w", "480"))
    h = int(a.get("h", "1040"))
    layers = (a.get("layers", "reliefBackdrop,reliefBg,skinSkirt,relief,worldBackdrop")
              or "").split(",")
    os.makedirs(outdir, exist_ok=True)

    b = None
    try:
        b = C.launch(prefix="pwl-", w=w, h=h, edge=a.get("edge"), gpu="d3d11",
                     url="about:blank")
        c = b.cdp
        C.enable_common(c, w, h)
        c.send("Page.navigate", {"url": url})
        time.sleep(14)

        names = c.ev(
            "JSON.stringify(Object.keys((window.__pw&&window.__pw.world&&"
            "window.__pw.world.meshes)||{}))")
        print("可用层：" + str(names))

        for layer in layers:
            if not layer:
                continue
            if layer == "extraMeshes":
                c.ev("(function(){window.__pw.savedExtra=window.__pw.world.extraMeshes||[]; "
                     "window.__pw.world.extraMeshes=[]; return 1})()")
            else:
                c.ev("(function(){var m=window.__pw.world.meshes; if(m) "
                     "window.__pw.saved_%s=m.%s||null, m.%s=null; return 1})()"
                     % (layer, layer, layer))
            time.sleep(0.7)
            f = os.path.join(outdir, "hide_%s.png" % layer)
            c.screenshot(f)
            print("  " + f)
            if layer == "extraMeshes":
                c.ev("(function(){window.__pw.world.extraMeshes=window.__pw.savedExtra||[]; "
                     "return 1})()")
            else:
                c.ev("(function(){var m=window.__pw.world.meshes; if(m) "
                     "m.%s=window.__pw.saved_%s||null; return 1})()" % (layer, layer))
            time.sleep(0.3)
        c.screenshot(os.path.join(outdir, "all_on.png"))
        return 0
    except Exception as e:
        print("失败：" + str(e), file=sys.stderr)
        return 1
    finally:
        if b:
            b.close()


if __name__ == "__main__":
    sys.exit(main())
