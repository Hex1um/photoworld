#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""移动边界体检（Python 版）：把玩家推到远处，看有没有空气墙。

    python tools/probe_walk.py --url "http://127.0.0.1:8099/?w=desert&instant=1" \
        --to -200 --out shots/wall.png
"""
from __future__ import annotations

import os
import sys
import time

import cdp as C


def ev(c: C.CDP, expr: str) -> str:
    r = c.eval_raw(expr)
    if r.get("exceptionDetails"):
        d = r["exceptionDetails"]
        desc = (d.get("exception") or {}).get("description") or d.get("text")
        return "EXC " + str(desc)[:300]
    v = (r.get("result") or {}).get("value")
    if v is None:
        return "NO VALUE"
    return str(v)


def main() -> int:
    a = C.args_dict()
    url = a.get("url", "http://127.0.0.1:8099/?w=desert&instant=1")
    to = float(a.get("to", "-200"))
    out = a.get("out", "")
    w = int(a.get("w", "480"))
    h = int(a.get("h", "1040"))

    b = None
    try:
        b = C.launch(prefix="pwwalk-", w=w, h=h, edge=a.get("edge"), gpu="d3d11",
                     url="about:blank")
        c = b.cdp
        C.enable_common(c, w, h)
        c.send("Page.navigate", {"url": url})
        time.sleep(14)

        print("before :", ev(c, """(function(){try{
  var w=window.__pw&&window.__pw.world, p=window.__pw&&window.__pw.player;
  return JSON.stringify({hasPw:!!window.__pw, mode:w&&w.mode, walkRadius:w&&w.walkRadius,
    bbox:w&&w.bbox, y:p&&+p.state.y.toFixed(2),
    err:(document.body.dataset||{}).err||'', errs:(window.__errs||[]).slice(0,3)});
}catch(e){return 'ERR '+e.message}})()"""))

        ev(c, """(function(){try{
  var p=window.__pw.player, w=window.__pw.world;
  p.state.z=%s; p.state.y=w.heightAt(0,%s)+1.7; return 1;
}catch(e){return 0}})()""" % (to, to))
        time.sleep(3)

        print("after  :", ev(c, """(function(){try{
  var p=window.__pw.player, w=window.__pw.world, ext=window.__pw.extender;
  return JSON.stringify({x:+p.state.x.toFixed(1), z:+p.state.z.toFixed(1), y:+p.state.y.toFixed(2),
    h:+w.heightAt(0,%s).toFixed(2), tiles:(w.extraMeshes||[]).length, ext:ext&&ext.stats?ext.stats():null});
}catch(e){return 'ERR '+e.message}})()""" % to))

        if out:
            c.screenshot(out)
            print("截图：" + out)
        return 0
    except Exception as e:
        print("失败：" + str(e), file=sys.stderr)
        return 1
    finally:
        if b:
            b.close()


if __name__ == "__main__":
    sys.exit(main())
