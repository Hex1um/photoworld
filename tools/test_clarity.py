#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""无浏览器自测：证明清晰度指标能分辨「糊片」和「雾/渐变」（Python 版）。

    python tools/test_clarity.py
"""
from __future__ import annotations

import sys

from clarity import capture_diff, clarity_metrics


def main() -> int:
    w, h = 128, 96
    sharp = [0.0] * (w * h)
    seed = 12345

    def rnd():
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return seed / 4294967296.0

    for y in range(h):
        for x in range(w):
            check = ((x >> 2) + (y >> 2)) % 2
            sharp[y * w + x] = min(1.0, max(0.0, 0.25 + check * 0.5 + (rnd() - 0.5) * 0.18))

    def blur3(src):
        out = [0.0] * len(src)
        for y in range(h):
            for x in range(w):
                s = n = 0.0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w:
                            s += src[yy * w + xx]
                            n += 1
                out[y * w + x] = s / n
        return out

    soft = sharp
    for _ in range(4):
        soft = blur3(soft)
    fog = [0.2 + 0.5 * (x / w) for y in range(h) for x in range(w)]

    A = clarity_metrics(sharp, w, h)
    B = clarity_metrics(soft, w, h)
    C = clarity_metrics(fog, w, h)
    print("清晰图  ", A)
    print("模糊图  ", B)
    print("渐变/雾 ", C)

    for k in ("hfEnergy", "centerHF", "lapVar", "centerLapVar", "localContrast"):
        assert k in A and A[k] >= 0, k + " 应为非负有限数"
    assert A["hfEnergy"] > B["hfEnergy"] * 2, "hfEnergy 未拉开差距"
    assert A["centerLapVar"] > B["centerLapVar"] * 2, "centerLapVar 未拉开差距"
    assert C["lapVar"] < 0.05, "纯渐变不该有拉普拉斯方差"
    assert C["centerLapVar"] < 0.05, "纯渐变不该有中央拉普拉斯方差"
    assert A["localContrast"] > C["localContrast"], "清晰图局部对比度应高于纯渐变"

    cap = {"w": 960, "h": 600, "dpr": 1, "gl": "hw", "views": 32, "radius": 11,
           "center": [0, -26], "pitch": -0.1, "settle": 260, "clarity": 1}
    assert capture_diff(cap, dict(cap)) == [], "相同采集条件应判为一致"
    assert capture_diff(cap, dict(cap, w=1440)) == ["w"], "分辨率不同必须拒绝对比"
    assert capture_diff(cap, dict(cap, gl="sw")) == ["gl"], "光栅后端不同必须拒绝对比"

    print("✓ 6/6 组断言通过：清晰度指标可分辨糊片与雾，A/B 采集条件校验生效")
    return 0


if __name__ == "__main__":
    sys.exit(main())
