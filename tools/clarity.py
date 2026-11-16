#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清晰度指标（Python 版，零依赖）。

tools/clarity.mjs 是同一组公式的 JS 镜像（multiview.mjs 在浏览器侧同步调用，
不能改成子进程）；本文件是测试 / 报告侧的单一来源，test_clarity_parity.py
会逐项比对两者输出，防止一边改了另一边没跟。

用法：
    from clarity import clarity_metrics, capture_diff
    c = clarity_metrics(L, w, h)   # L = 亮度 0..1 的扁平序列
"""
from __future__ import annotations

import json
import math

HIGHER_IS_BETTER = {
    "hfEnergy": True, "farHF": True, "centerHF": True,
    "lapVar": True, "centerLapVar": True, "localContrast": True, "detail": True,
    "voidFrac": False, "innerVoid": False, "lowVoid": False, "borderVoid": False,
    "darkFrac": False, "stripe": False,
}
CLARITY_KEYS = ["hfEnergy", "farHF", "centerHF", "lapVar", "centerLapVar", "localContrast"]
CAPTURE_KEYS = ["w", "h", "dpr", "gl", "views", "radius", "center", "pitch", "settle", "clarity"]


def _fixed(x: float, digits: int) -> float:
    """JS Number.prototype.toFixed 的非负版本（避开 Python 银行家舍入）。"""
    scale = 10 ** digits
    return math.floor(x * scale + 0.5) / scale


def _box_blur3(L, w: int, h: int):
    n = w * h
    tmp = [0.0] * n
    out = [0.0] * n
    for y in range(h):
        row = y * w
        for x in range(w):
            tmp[row + x] = (L[row + (x - 1 if x > 0 else 0)] + L[row + x]
                            + L[row + (x + 1 if x < w - 1 else 0)]) / 3.0
    for y in range(h):
        row = y * w
        up = (y - 1 if y > 0 else 0) * w
        dn = (y + 1 if y < h - 1 else 0) * w
        for x in range(w):
            out[row + x] = (tmp[up + x] + tmp[row + x] + tmp[dn + x]) / 3.0
    return out


def _local_contrast(L, w: int, h: int, x0: int, x1: int, y0: int, y1: int) -> float:
    sw = w + 1
    S = [0.0] * (sw * (h + 1))
    S2 = [0.0] * (sw * (h + 1))
    for y in range(h):
        rs = rs2 = 0.0
        for x in range(w):
            v = L[y * w + x]
            rs += v
            rs2 += v * v
            i = (y + 1) * sw + (x + 1)
            S[i] = S[y * sw + (x + 1)] + rs
            S2[i] = S2[y * sw + (x + 1)] + rs2
    r, step = 2, 2
    total = count = 0.0
    ys = range(max(r, y0), min(h - r, y1), step)
    xs = range(max(r, x0), min(w - r, x1), step)
    for y in ys:
        ya, yb = y - r, y + r + 1
        for x in xs:
            xa, xb = x - r, x + r + 1
            area = (xb - xa) * (yb - ya)
            s = S[yb * sw + xb] - S[ya * sw + xb] - S[yb * sw + xa] + S[ya * sw + xa]
            s2 = S2[yb * sw + xb] - S2[ya * sw + xb] - S2[yb * sw + xa] + S2[ya * sw + xa]
            m = s / area
            total += math.sqrt(max(0.0, s2 / area - m * m))
            count += 1
    return total / max(1.0, count)


def _variance(s: float, sq: float, n: int) -> float:
    return max(0.0, sq / n - (s / n) ** 2) if n else 0.0


def clarity_metrics(L, w: int, h: int) -> dict[str, float]:
    if not L or w < 3 or h < 3 or len(L) < w * h:
        return {k: 0 for k in CLARITY_KEYS}
    B = _box_blur3(L, w, h)
    cx0, cx1 = int(w * 0.25), int(w * 0.75)
    cy0, cy1 = int(h * 0.25), int(h * 0.75)
    fy0, fy1 = int(h * 0.20), int(h * 0.55)

    hf_sum = hf_n = far_sum = far_n = c_hf_sum = c_n = 0.0
    lap_s = lap_sq = c_lap_s = c_lap_sq = 0.0
    lap_n = c_lap_n = 0
    for y in range(1, h - 1, 2):
        in_far = fy0 <= y < fy1
        in_center_y = cy0 <= y < cy1
        for x in range(1, w - 1, 2):
            k = y * w + x
            hf = abs(L[k] - B[k])
            hf_sum += hf
            hf_n += 1
            if in_far:
                far_sum += hf
                far_n += 1
            in_center = in_center_y and cx0 <= x < cx1
            if in_center:
                c_hf_sum += hf
                c_n += 1
            lap = 4 * L[k] - L[k - 1] - L[k + 1] - L[k - w] - L[k + w]
            lap_s += lap
            lap_sq += lap * lap
            lap_n += 1
            if in_center:
                c_lap_s += lap
                c_lap_sq += lap * lap
                c_lap_n += 1
    lc = _local_contrast(L, w, h, cx0, cx1, cy0, cy1)
    return {
        "hfEnergy": _fixed(hf_sum / max(1, hf_n), 5),
        "farHF": _fixed(far_sum / max(1, far_n), 5),
        "centerHF": _fixed(c_hf_sum / max(1, c_n), 5),
        "lapVar": _fixed(_variance(lap_s, lap_sq, lap_n) * 1000, 4),
        "centerLapVar": _fixed(_variance(c_lap_s, c_lap_sq, c_lap_n) * 1000, 4),
        "localContrast": _fixed(lc, 5),
    }


def _js_json(v):
    """近似 JSON.stringify 的标量归一化：1.0 -> 1，键顺序保持原样。"""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, list):
        return [_js_json(x) for x in v]
    if isinstance(v, dict):
        return {k: _js_json(x) for k, x in v.items()}
    return v


def _js_dump(v) -> str:
    return json.dumps(_js_json(v), separators=(",", ":"), ensure_ascii=False)


def capture_diff(a, b) -> list[str]:
    if not a or not b:
        return list(CAPTURE_KEYS)
    return [k for k in CAPTURE_KEYS if _js_dump(a.get(k)) != _js_dump(b.get(k))]
