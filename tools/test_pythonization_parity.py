#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python 化后的纯计算函数回归：不依赖浏览器/网络，锁住与 JS 原版一致的公式。

    python tools/test_pythonization_parity.py

覆盖本轮从 .mjs 迁到 .py 的四个工具里的「无副作用」核心：
  - bench_glb.py : psnr / view_metrics 的极值行为（全同图、纯色底、条纹梯度）
  - render_traj.py : make_pos 的确定性（同种子逐位一致）+ trs_array 单位阵/平移
  - harvest_github.py : looks_like_landscape 的天空启发式（蓝天在上 vs 倒挂）
四项脚本的 .mjs 入口仍为薄包装（静态断言），参数/输出/退出码透传。
"""
from __future__ import annotations

import math
import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)

import bench_glb as B  # noqa: E402
import harvest_github as H  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import render_traj as R  # noqa: E402

PASS = 0


def ok(msg):
    global PASS
    PASS += 1
    print("PASS  " + msg)


def eq(a, b, msg, tol=1e-6):
    assert abs(a - b) <= tol, "%s: %r != %r" % (msg, a, b)
    ok(msg)


# ---------- 1) bench_glb 像素指标 ----------
def solid(w, h, rgb):
    px = bytes([rgb[0], rgb[1], rgb[2], 255]) * (w * h)
    return (w, h, px)


# 用 16 级直方图 bin 中心色（200>>4=12，中心 12*16+8=200），
# 否则像 128 这种值会被重建到 136，|128-136|=8>4 → 不算 void（与 JS 一致）。
A = solid(8, 8, (200, 200, 200))
eq(B.psnr(A, A), 99.0, "bench_glb.psnr 全同图 = 99dB")
m = B.view_metrics(A)
eq(m["voidFrac"], 1.0, "bench_glb.view_metrics 纯色图 voidFrac=1")
eq(m["innerVoid"], 1.0, "bench_glb.view_metrics 纯色图 innerVoid=1")
eq(m["stripe"], 0.0, "bench_glb.view_metrics 纯色图 stripe=0")
eq(m["detail"], 0.0, "bench_glb.view_metrics 纯色图 detail=0")
C = solid(8, 8, (10, 200, 30))
assert B.psnr(A, C) < 30, "不同色图 PSNR 应显著低于 30dB"
ok("bench_glb.psnr 不同色图显著下降")


# ---------- 2) render_traj 位姿确定性 + 变换矩阵 ----------
p1, a1 = R.make_pos(1, 8)
p2, a2 = R.make_pos(1, 8)
assert p1 == p2 and a1 == a2, "同种子位姿必须逐位一致"
ok("render_traj.make_pos 同种子确定性")
assert R.make_pos(2, 8)[0] != p1, "不同种子位姿应不同"
ok("render_traj.make_pos 不同种子不同")
m0 = R.trs_array(0, 0, 0, 0, 0)
assert m0 == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], "单位变换矩阵"
ok("render_traj.trs_array 单位阵")
mt = R.trs_array(3, 1.7, -2, 0, 0)
assert mt[12:15] == [3, 1.7, -2], "平移列"
ok("render_traj.trs_array 平移列")


# ---------- 3) harvest_github 天空启发式 ----------
def synth_landscape():
    """顶部蓝亮平（天），底部暗（地）→ 应判为风景。"""
    w = h = 40
    buf = bytearray()
    for y in range(h):
        for x in range(w):
            if y < h * 0.5:
                r, g, b = 120, 170, 230  # 亮蓝天空，方差 0
            else:
                r, g, b = 40, 55, 35     # 暗绿地
            buf += bytes([r, g, b, 255])
    return bytes(buf), w, h


data, w, h = synth_landscape()
assert H.looks_like_landscape(data, w, h) is True, "蓝天在上应判为风景"
ok("harvest_github.looks_like_landscape 蓝天在上=True")

# 倒挂：底部亮天、顶部暗地 → 顶部不比底部亮 → False
data2 = bytearray()
for y in range(h):
    for x in range(w):
        if y < h * 0.5:
            r, g, b = 40, 55, 35
        else:
            r, g, b = 120, 170, 230
        data2 += bytes([r, g, b, 255])
assert H.looks_like_landscape(bytes(data2), w, h) is False, "天在下应拒绝"
ok("harvest_github.looks_like_landscape 天在下=False")


# ---------- 4) 四个入口仍是薄包装 ----------
for rel, py in [
    ("tools/abshots.mjs", "abshots.py"),
    ("tools/bench_glb.mjs", "bench_glb.py"),
    ("scripts/render_traj.mjs", "render_traj.py"),
    ("tools/harvest_github.mjs", "harvest_github.py"),
]:
    src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
    assert "spawnSync" in src and py in src and "process.exit(r.status" in src, \
        rel + " 不是薄包装"
    assert os.path.isfile(os.path.join(ROOT, os.path.dirname(rel), py)), \
        rel + " 对应的 .py 不存在"
ok("四个 .mjs 入口均为薄包装且 .py 就位")

print("\nPython 化纯计算回归全部通过：%d 项" % PASS)
