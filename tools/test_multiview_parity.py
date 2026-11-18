#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""multiview.py 与 multiview.mjs 的指标口径一致性回归（v3.0.6）。

multiview.mjs 已改为薄包装，其 metrics()/blobStats()/机位套餐逻辑迁到 Python。
本测试做两件事：
  1) 从仓库保存的 JS 原版算法（tools/_multiview_js_ref.json 里记录的输入→输出基准）
     逐项比对 Python image_metrics，确保迁移没漂移；
  2) 直接锚定几个关键不变量（纯色图 voidFrac=1、条纹 stripe 高、机位套餐数量与首末机位），
     即使基准文件缺失也能抓住回归。

    python tools/test_multiview_parity.py
"""
from __future__ import annotations

import base64
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import multiview as M  # noqa: E402

PASS = 0


def ok(msg):
    global PASS
    PASS += 1
    print("PASS  " + msg)


def png(w, h, fn):
    """构造 RGBA 字节（与 CDP 截图同布局）。"""
    buf = bytearray()
    for y in range(h):
        for x in range(w):
            buf += bytes(fn(x, y))
    return w, h, 4, bytes(buf)


def near(a, b, tol=1e-9):
    assert abs(float(a) - float(b)) <= tol, "%r != %r" % (a, b)


# ---------- 1) 不变量：纯色图 ----------
solid = png(24, 24, lambda x, y: (200, 200, 200, 255))
m = M.image_metrics(solid)
near(m["voidFrac"], 1.0)
near(m["innerVoid"], 1.0)
near(m["lowVoid"], 1.0)
near(m["borderVoid"], 1.0)
near(m["darkFrac"], 0.0)
near(m["stripe"], 0.0)
ok("纯色主色图 voidFrac/inner/low/border=1, stripe=0")

# refBg 精确匹配时同样判空（即便主色算法被纹理干扰）
m2 = M.image_metrics(solid, ref_bg=[200, 200, 200])
near(m2["voidFrac"], 1.0)
ok("refBg 精确匹配 voidFrac=1")

# ---------- 2) 不变量：全黑图 darkFrac=1 ----------
black = png(24, 24, lambda x, y: (0, 0, 0, 255))
m3 = M.image_metrics(black)
near(m3["darkFrac"], 1.0)
ok("全黑图 darkFrac=1")

# ---------- 3) 竖条纹 stripe > 横条纹 ----------
# stripe = vy/vx：沿 y 方向的差分能量 / 沿 x 方向的差分能量。
# 横向带（逐行变色、行内一致）→ vy 大、vx≈0 → stripe 极大；
# 竖向带（逐列变色、列内一致）→ vy≈0、vx 大 → stripe≈0。
horz = png(24, 24, lambda x, y: (250, 250, 250, 255) if y % 2 else (20, 20, 20, 255))
vert = png(24, 24, lambda x, y: (250, 250, 250, 255) if x % 2 else (20, 20, 20, 255))
mh, mv = M.image_metrics(horz), M.image_metrics(vert)
assert mh["stripe"] > mv["stripe"] * 3, "横向带 stripe 应远大于竖向带"
ok("横带 stripe 显著高于竖带 (%s vs %s)" % (mh["stripe"], mv["stripe"]))

# ---------- 4) 有纹理的"接近背景色"不算空底 ----------
# 背景色 200，但相邻像素有颗粒（max-min > 0.02）→ 不应大面积判空
textured = png(24, 24, lambda x, y: (200 + (7 if (x + y) % 2 else 0),) * 3 + (255,))
mt = M.image_metrics(textured, ref_bg=[200, 200, 200])
assert mt["voidFrac"] < 0.9, "有纹理的近背景色图不应被整片判空"
ok("近背景色但有纹理 voidFrac=%.4f (<0.9)" % mt["voidFrac"])

# ---------- 5) blobStats 连通域口径 ----------
mask = bytearray(20)
# 3 连通块：{0,1,2} 面积 3、{10} 面积 1、{15,16,17,18} 面积 4
for i in (0, 1, 2, 10, 15, 16, 17, 18):
    mask[i] = 1
count, max_area, total = M.blob_stats(mask, 20, 1, 3)
assert count == 2 and max_area == 4 and total == 7, (count, max_area, total)
ok("blobStats 最小面积过滤 + maxArea/totalArea")

# ---------- 6) 机位套餐逐项锚定（与 JS makePlan 一致）----------
rings32, plan_v2, plan_v3 = M.make_plans()
assert len(rings32) == 32 and len(plan_v2) == 32 and len(plan_v3) == 32
ok("三套 32 机位套餐数量正确")

near(rings32[0]["a"], 0.0)
near(rings32[0]["r"], 10)
near(rings32[11]["a"], (11 / 12) * math.pi * 2)
near(rings32[12]["r"], 26)
assert rings32[24]["ring"] == "C" and rings32[24]["r"] == 75
near(rings32[25]["eye"], 14)
ok("RINGS_32 首末/圈层半径/俯视眼高正确")

assert plan_v2[0]["x"] == -6 and plan_v2[0]["z"] == 0
near(plan_v2[0]["yaw"], 0)
near(plan_v2[1]["yaw"], -0.44)
near(plan_v2[0]["pitch"], -0.04)
assert plan_v2[-1]["x"] == 6 and plan_v2[-1]["z"] == -5
ok("PLAN_V2 首末机位/偏航/俯仰正确")

assert all(v["ring"] in ("F", "S", "B", "U") for v in plan_v3)
assert sum(1 for v in plan_v3 if v["ring"] == "B") == 8
near(plan_v3[0]["yaw"], -0.35)
near(plan_v3[-1]["pitch"], -0.55)
ok("PLAN_V3 四组各 8 机位/偏航俯仰正确")

# ---------- 7) 聚合口径：判据取最差、展示取均值 ----------
rec = {"views": [
    {"voidFrac": 0.1, "darkFrac": 0.2, "innerVoid": 0.05, "lowVoid": 0.1,
     "borderVoid": 0.0, "stripe": 1.0, "detail": 0.5, "centerHF": 10, "farHF": 8,
     "hfEnergy": 9, "lapVar": 3, "centerLapVar": 4, "localContrast": 2,
     "voidBlobs": 1, "streakBlobs": 2, "voidRunMax": 0.1, "voidBlobMax": 0.2, "streakBlobFrac": 0.3},
    {"voidFrac": 0.4, "darkFrac": 0.4, "innerVoid": 0.3, "lowVoid": 0.5,
     "borderVoid": 0.2, "stripe": 3.0, "detail": 0.7, "centerHF": 20, "farHF": 12,
     "hfEnergy": 15, "lapVar": 7, "centerLapVar": 10, "localContrast": 6,
     "voidBlobs": 5, "streakBlobs": 9, "voidRunMax": 0.6, "voidBlobMax": 0.8, "streakBlobFrac": 0.9},
]}
agg = M.aggregate(rec)
near(agg["voidFrac"], 0.4)        # 最差
near(agg["stripe"], 3.0)          # 最差
near(agg["darkFrac"], 0.3)        # 均值
near(agg["centerHF"], 15)         # 均值
assert agg["voidBlobs"] == 5 and agg["streakBlobs"] == 9
ok("聚合：判据取最差机位 / 展示与 clarity 取均值")

# ---------- 8) .mjs 仍是薄包装 ----------
src = open(os.path.join(HERE, "multiview.mjs"), encoding="utf-8").read()
assert "spawnSync" in src and "multiview.py" in src and "process.exit(r.status" in src, \
    "multiview.mjs 不是薄包装"
ok("multiview.mjs 为薄包装且 multiview.py 就位")

# ---------- 9) 保存的 JS 基准（若存在）逐项比对 ----------
ref_path = os.path.join(HERE, "_multiview_js_ref.json")
if os.path.isfile(ref_path):
    with open(ref_path, encoding="utf-8") as f:
        cases = json.load(f)
    for name, c in cases.items():
        w, h = c["w"], c["h"]
        buf = base64.b64decode(c["data_b64"])
        got = M.image_metrics((w, h, 4, buf), c.get("refBg"))
        for k, v in c["expect"].items():
            # 极端值（如除零得到的 stripe ~1e8）用相对容差：JS toFixed 与
            # Python math.floor 在 1e8 量级的浮点尾数不同，相对误差 <1e-6。
            tol = max(2e-4, abs(float(v)) * 1e-6)
            near(got[k], v, tol=tol)
        ok("JS 基准 %s 全部指标一致" % name)
else:
    print("（无 _multiview_js_ref.json，跳过 JS 基准比对 —— 不变量回归仍有效）")

print("\nmultiview Python 化回归全部通过：%d 项" % PASS)
