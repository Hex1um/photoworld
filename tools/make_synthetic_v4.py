#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""造一段"能通过 v4 质量门"的走动素材，用来端到端验证 PASS → 出 GLB 的链路。

仓库自带的 testdata 全是纯旋转 / 远景素材，按 v4.1.2 的判据应全部 FAIL；
这保证了"坏素材被拦住"，但没人验证过"好素材真的能出世界"。这个生成器补上另一半：

    python tools/make_synthetic_v4.py --out out/_selftest_good

场景是一块起伏地形（非共面，本质矩阵不会退化）+ 非周期随机散斑纹理
（棋盘格跨帧描述子重复，交叉匹配会大量误配）+ 少量小立柱。相机沿弧线走动，
内参与 build.py 默认 focal_ratio=1.2 一致。

    --arc 108   相机扫过的角度；调小到 18 可造"只拍一面"的对照组
    --frames 13 帧数
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

W, H = 960, 640
F = 1.2 * max(W, H)
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]], np.float64)


def terrain_y(x, z):
    return (0.16 * np.sin(1.35 * x) * np.cos(1.15 * z)
            + 0.07 * np.sin(2.4 * x + 0.8 * z))


def build_scene(seed=11):
    rng = np.random.default_rng(seed)
    faces = []                       # (4x3 顶点, BGR 颜色)

    N, half = 110, 4.8
    xs = np.linspace(-half, half, N + 1)
    for i in range(N):
        for j in range(N):
            x0, x1, z0, z1 = xs[i], xs[i + 1], xs[j], xs[j + 1]
            p = np.array([[x0, terrain_y(x0, z0), z0],
                          [x1, terrain_y(x1, z0), z0],
                          [x1, terrain_y(x1, z1), z1],
                          [x0, terrain_y(x0, z1), z1]], np.float64)
            # 非周期散斑，避免棋盘格的跨帧描述子重复
            base = int(rng.integers(35, 225))
            faces.append((p, (base, base, base)))
            # 每格一个位置/亮度随机的小方块，让局部特征可区分
            a, b = rng.uniform(0.2, 0.8, 2)
            mx0 = x0 + a * (x1 - x0)
            mz0 = z0 + b * (z1 - z0)
            mx1, mz1 = mx0 + 0.45 * (x1 - x0), mz0 + 0.45 * (z1 - z0)
            my = terrain_y(mx0, mz0) + 0.002
            mark = np.array([[mx0, my, mz0], [mx1, my, mz0],
                             [mx1, my, mz1], [mx0, my, mz1]], np.float64)
            mc = int(rng.choice([20, 60, 200, 245]))
            faces.append((mark, (mc, mc, mc)))

    # 少量小立柱：补垂直结构，但不遮挡主视野
    for _ in range(24):
        cx, cz = rng.uniform(-2.2, 2.2, 2)
        s, h = rng.uniform(0.06, 0.13), rng.uniform(0.10, 0.30)
        y0 = terrain_y(cx, cz)
        x0, x1, z0, z1 = cx - s, cx + s, cz - s, cz + s
        hue = tuple(int(v) for v in rng.integers(40, 235, 3))
        faces.append((np.array([[cx - s, y0 + h, cz - s], [cx + s, y0 + h, cz - s],
                                [cx + s, y0 + h, cz + s], [cx - s, y0 + h, cz + s]],
                               np.float64), hue))
        for quad in ([[x0, y0, z0], [x1, y0, z0], [x1, y0 + h, z0], [x0, y0 + h, z0]],
                     [[x1, y0, z0], [x1, y0, z1], [x1, y0 + h, z1], [x1, y0 + h, z0]],
                     [[x1, y0, z1], [x0, y0, z1], [x0, y0 + h, z1], [x1, y0 + h, z1]],
                     [[x0, y0, z1], [x0, y0, z0], [x0, y0 + h, z0], [x0, y0 + h, z1]]):
            c = tuple(int(max(5, min(250, v * 0.8))) for v in hue)
            faces.append((np.array(quad, np.float64), c))
    return faces


def render(faces, cam, target):
    fwd = np.array(target, np.float64) - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.vstack([right, -up, fwd])          # 相机 y 轴朝下
    t = (-R @ cam).reshape(3, 1)

    items = []
    for verts, col in faces:
        pc = (R @ verts.T + t).T
        if np.any(pc[:, 2] <= 0.05):
            continue
        uv = (K @ pc.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        items.append((float(pc[:, 2].mean()), np.round(uv).astype(np.int32), col))
    items.sort(key=lambda x: -x[0])            # painter：远的先画

    img = np.full((H, W, 3), 215, np.uint8)
    for _depth, poly, col in items:
        cv2.fillConvexPoly(img, poly, tuple(int(v) for v in col))
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 v4 端到端自测素材")
    ap.add_argument("--out", default="out/_selftest_good", help="输出目录")
    ap.add_argument("--arc", type=float, default=108.0, help="相机扫过的角度（度）")
    ap.add_argument("--frames", type=int, default=13)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    faces = build_scene(args.seed)
    for k in range(args.frames):
        ang = np.deg2rad(-args.arc / 2 + k * args.arc / max(1, args.frames - 1))
        cam = np.array([3.1 * np.sin(ang), 1.55, 3.1 * np.cos(ang)])
        cv2.imwrite(os.path.join(args.out, "w%02d.png" % k),
                    render(faces, cam, [0.0, 0.0, 0.0]))
    print("wrote %d frames to %s" % (args.frames, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
