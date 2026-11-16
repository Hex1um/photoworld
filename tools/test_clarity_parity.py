#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python / JS 清晰度指标一致性回归。

clarity.mjs 仍被 multiview.mjs 同步 import（浏览器侧链路），无法直接换成
子进程调用 Python；本测试用同一组输入同时跑两边，锁住公式不漂移。

    python tools/test_clarity_parity.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
from clarity import CLARITY_KEYS, HIGHER_IS_BETTER, capture_diff, clarity_metrics  # noqa: E402

JS = os.path.join(TOOLS, "clarity.mjs")

NODE_SRC = """import fs from 'node:fs';
import { clarityMetrics, CLARITY_KEYS, HIGHER_IS_BETTER, captureDiff } from '%s';
const d = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify({
  m: clarityMetrics(d.L, d.w, d.h),
  keys: CLARITY_KEYS,
  dir: HIGHER_IS_BETTER,
  diff: captureDiff(d.a, d.b)
}));
""" % Path(JS).as_uri()


def _random_field(w, h, seed):
    vals = []
    s = seed
    for _ in range(w * h):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        vals.append(s / 4294967296.0)
    return vals


def _cases():
    w, h = 96, 64
    sharp = _random_field(w, h, 12345)
    # 用确定性的盒式模糊代替随机，测的是两边公式一致而非指标好坏
    soft = []
    for y in range(h):
        for x in range(w):
            acc = n = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        acc += sharp[yy * w + xx]
                        n += 1
            soft.append(acc / n)
    fog = [0.2 + 0.5 * (x / w) for y in range(h) for x in range(w)]
    base = {"w": 960, "h": 600, "dpr": 1, "gl": "hw", "views": 32, "radius": 11,
            "center": [0, -26], "pitch": -0.1, "settle": 260, "clarity": 1}
    return [
        (w, h, sharp, base, dict(base, w=1440)),
        (w, h, soft, base, dict(base, gl="sw")),
        (w, h, fog, base, dict(base)),
        (2, 2, [0.1] * 4, base, dict(base)),
    ]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="pwclarity-") as td:
        runner = os.path.join(td, "run.mjs")
        with open(runner, "w", encoding="utf-8", newline="") as f:
            f.write(NODE_SRC)
        for w, h, L, a, b in _cases():
            payload = json.dumps({"L": L, "w": w, "h": h, "a": a, "b": b})
            p = subprocess.run(["node", runner], input=payload.encode("utf-8"),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            if p.returncode != 0:
                print("Node 端失败：" + p.stderr.decode("utf-8", "replace")[:500], file=sys.stderr)
                return 1
            got = json.loads(p.stdout.decode("utf-8"))
            mine = clarity_metrics(L, w, h)
            for k in CLARITY_KEYS:
                if not math.isclose(mine[k], got["m"][k], rel_tol=1e-9, abs_tol=1e-9):
                    print("%s 不一致：Python=%s JS=%s" % (k, mine[k], got["m"][k]), file=sys.stderr)
                    return 1
            if got["keys"] != CLARITY_KEYS or got["dir"] != HIGHER_IS_BETTER:
                print("指标键或方向常量不一致", file=sys.stderr)
                return 1
            if capture_diff(a, b) != got["diff"]:
                print("captureDiff 不一致：Python=%s JS=%s"
                      % (capture_diff(a, b), got["diff"]), file=sys.stderr)
                return 1
    print("✓ Python / JS 清晰度指标 4 组输入逐项一致（含键、方向与 captureDiff）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
