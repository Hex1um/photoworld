#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成带**真值位姿**的合成场景，用来验证 v4 的 SfM（没有真值就没法量化"重建得怎么样"）。

场景：一张带程序纹理的地面 + 两个立方体；渲染用逐像素射线求交（几十行，无需渲染引擎）。
输出：
    <out>/view_XX.png       渲染图
    <out>/gt.json           K、每帧的 R/t（world→cam）、场景 bbox

    python tools/make_synth_scene.py --out testdata/synth1 --views 8
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


def hash3(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    v = np.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453
    return v - np.floor(v)


def noise2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """多倍频值噪声（给地面一点结构，SIFT 才有东西可抓）。"""
    acc = np.zeros_like(x)
    amp, freq = 0.5, 1.0
    for _ in range(4):
        acc += amp * hash3(np.floor(x * freq), np.floor(y * freq), np.zeros_like(x))
        amp *= 0.5
        freq *= 2.0
    return acc


def render(K, R, t, w, h, boxes):
    """world→cam 的 R,t；返回 (h,w,3) uint8。"""
    ys, xs = np.mgrid[0:h, 0:w]
    pix = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs)], axis=-1).astype(np.float64)
    Kinv = np.linalg.inv(K)
    ray_cam = pix @ Kinv.T                       # (h,w,3)
    # 本项目的相机约定：**朝 -z 看**（投影用 z=-dc.z），而 pix 的 z=+1 是 OpenCV 约定，
    # 所以这里必须翻一下，否则所有射线都朝相机背后打（第一版渲出来全屏天空就是这个 bug）
    ray_cam[..., 2] *= -1.0
    ray_cam /= np.linalg.norm(ray_cam, axis=-1, keepdims=True)
    ray_world = ray_cam @ R                      # R^T · d
    o = -R.T @ t.ravel()                         # 相机中心（world）
    d = ray_world

    # 地面：有起伏的曲面（纯平面会让本质矩阵退化 —— 单应就能解释所有对应点）
    # 迭代 3 次求交：y = -1.2 + h(x,z)
    tt = np.where(d[..., 1] < -1e-6, (-1.2 - o[1]) / np.where(d[..., 1] < -1e-6, d[..., 1], -1), 1e9)
    for _ in range(3):
        P0 = o[None, None, :] + d * np.where(np.isfinite(tt), tt, 0)[..., None]
        hgt = -1.2 + 0.35 * (noise2(P0[..., 0] * 1.1, P0[..., 2] * 1.1) - 0.5) * 2
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (hgt - o[1]) / np.where(d[..., 1] < -1e-6, d[..., 1], -1)
    hit = (tt > 0) & np.isfinite(tt) & (d[..., 1] < -1e-6)
    P = o[None, None, :] + d * np.where(hit, tt, 0)[..., None]
    dist = np.where(hit, tt, 1e9)

    nz = noise2(P[..., 0] * 0.55, P[..., 2] * 0.55)
    col = np.stack([0.25 + 0.55 * nz, 0.28 + 0.45 * nz, 0.22 + 0.30 * (1 - nz)], axis=-1)
    # 棋盘格（高频，利于匹配）
    chk = ((np.floor(P[..., 0] * 1.2) + np.floor(P[..., 2] * 1.2)) % 2)
    col = col * (0.72 + 0.55 * chk[..., None])

    for (c, half) in boxes:
        c = np.asarray(c, np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (c - half - o) / d
            t2 = (c + half - o) / d
        tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
        inb = (tmax > np.maximum(tmin, 0)) & (tmin > 0) & (tmin < dist)
        if inb.any():
            n = np.stack([hash3(P[..., 0] * 3, P[..., 1] * 3, P[..., 2] * 3)] * 3, -1) * 0.6 + 0.3
            col = np.where(inb[..., None], n, col)
            dist = np.where(inb, tmin, dist)

    sky = np.clip(col, 0, 1)
    sky = np.where((dist > 1e8)[..., None], np.array([0.55, 0.62, 0.72]), sky)
    return (np.clip(sky, 0, 1) * 255 + 0.5).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="testdata/synth1")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=640)
    ap.add_argument("--radius", type=float, default=3.0)
    args = ap.parse_args()

    from PIL import Image

    os.makedirs(args.out, exist_ok=True)
    w, h = args.w, args.h
    f = 1.15 * max(w, h)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
    boxes = [((-1.6, 0.25, -2.2), 0.75), ((1.7, 0.15, -1.0), 0.6), ((0.2, 0.05, -3.0), 0.5)]
    gt = {"K": K.tolist(), "w": w, "h": h, "views": []}

    for k in range(args.views):
        ang = 2 * math.pi * k / args.views
        # 相机绕原点走一圈，始终看向场景中心（有真实平移 → 有视差）
        c = np.array([args.radius * math.sin(ang), 0.9, -args.radius * math.cos(ang)])
        target = np.array([0.0, 0.55, -1.8])
        fwd = target - c
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0])); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        R = np.stack([right, up, -fwd])          # world→cam
        t = -R @ c
        img = render(K, R, t, w, h, boxes)
        Image.fromarray(img).save(os.path.join(args.out, "view_%02d.png" % k))
        gt["views"].append({"R": R.tolist(), "t": t.tolist(), "center": c.tolist()})
        print("渲染 view_%02d  相机 (%.2f, %.2f, %.2f)" % (k, *c))

    with open(os.path.join(args.out, "gt.json"), "w", encoding="utf-8") as fh:
        json.dump(gt, fh, indent=1)
    print("真值写入 %s/gt.json" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
