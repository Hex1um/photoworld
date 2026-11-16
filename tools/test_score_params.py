#!/usr/bin/env python3
"""评分器回归测试（无 GPU / 无 torch / 无浏览器）：判分契约 + multiview 源码契约。

  python tools/test_score_params.py

覆盖 2026-11-16 v2.13.2（R1）修掉的问题，全部有真实观测支撑：

  1. **空指标 = 满分**：旧 `score({})` 返回 hard=1 soft=1.0 —— 渲染坏了却拿满分，
     是奖励作弊入口。现在必须返回 invalid（hard=0, soft=0.0, fail 带 invalid: 前缀）。
  2. **缺一项 = 满分**：同上，逐项检查（少 voidRunMax 等任意一项都要 invalid）。
  3. **非数值指标**：字符串/NaN 之类的脏值按无效处理，不让比较抛 TypeError。
  4. **voidBlobMax 语义错**：旧版用 totalArea（与 voidFrac 重复），文档声称的
     "单块大洞"没人守；现取 blobStats.maxArea。
  5. **空洞容差不齐**：掩码 ±8、主循环/横段 ±6 → 统一为单一常量 VOID_TOL=6。
  6. **俯视口径不齐**：俯视图未传 refBg，改用画面主色 → 与主视角统一。

只依赖标准库；不对 render_score / bake 做端到端调用（那需要 GPU 和浏览器）。
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "pwscore_test", os.path.join(ROOT, "tools", "score_params.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pwscore_test"] = mod
    spec.loader.exec_module(mod)
    return mod


SP = _load()

# 八项全达标的样本（数值均低于 THRESH）
GOOD = {
    "stripe": 1.30, "innerVoid": 0.01, "voidFrac": 0.02, "lowVoid": 0.03,
    "voidBlobs": 20, "streakBlobs": 15, "voidRunMax": 0.10, "voidBlobMax": 0.03,
}


def test_score_full():
    hard, soft, fail = SP.score(dict(GOOD))
    assert hard == 1 and soft == 1.0 and fail == [], (hard, soft, fail)
    print("PASS  S1  八项全达标 → hard=1 soft=1.0 无失败项")


def test_score_empty_is_invalid():
    for agg in ({}, None, [], "x"):
        hard, soft, fail = SP.score(agg)
        assert hard == 0 and soft == 0.0, (agg, hard, soft)
        assert fail and fail[0].startswith("invalid:"), (agg, fail)
    print("PASS  S2  空指标（{}/None/[]/字符串）→ invalid，不再判满分")


def test_score_missing_key_is_invalid():
    for k in SP.THRESH:
        agg = dict(GOOD)
        agg.pop(k)
        hard, soft, fail = SP.score(agg)
        assert hard == 0 and soft == 0.0, (k, hard, soft)
        assert any(f.startswith("invalid:missing:") and k in f for f in fail), (k, fail)
    print("PASS  S3  缺任意一项（逐项 %d 次）→ invalid 且失败项指明缺哪个键" % len(SP.THRESH))


def test_score_nonnumeric_is_invalid():
    for dirty in ("abc", float("nan"), object()):
        agg = dict(GOOD, stripe=dirty)
        try:
            hard, soft, fail = SP.score(agg)
        except Exception as e:                       # noqa: BLE001
            raise AssertionError("非数值指标不应抛异常: %r" % (e,))
        assert hard == 0 and soft == 0.0, (dirty, hard, soft)
        assert any(f.startswith("invalid:") for f in fail), (dirty, fail)
    print("PASS  S4  非数值指标（字符串/NaN/对象）→ invalid 且不抛异常")


def test_score_over_threshold():
    agg = dict(GOOD, stripe=4.0, lowVoid=0.30)
    hard, soft, fail = SP.score(agg)
    assert hard == 0 and set(fail) == {"stripe", "lowVoid"}, (hard, fail)
    assert 0.0 < soft < 1.0, soft
    # 恰等于阈值视为通过（避免浮点边界导致"擦线即失败"）
    hard2, _, fail2 = SP.score(dict(GOOD, stripe=SP.THRESH["stripe"]))
    assert hard2 == 1 and not fail2, (hard2, fail2)
    print("PASS  S5  超阈值才失败（%s），soft 扣减 %.4f，恰等阈值视为通过"
          % ("/".join(sorted(fail)), soft))


def test_score_weights_sum_to_one():
    assert math.isclose(sum(SP.WEIGHT.values()), 1.0, abs_tol=1e-9), SP.WEIGHT
    assert set(SP.WEIGHT) == set(SP.THRESH), "阈值与权重键集合必须一致"
    print("PASS  S6  权重合计=1.0，阈值/权重键集合一致（%d 项）" % len(SP.THRESH))


def test_void_blobmax_semantics():
    src = open(os.path.join(ROOT, "tools", "multiview.mjs"), encoding="utf-8").read()
    assert "voidBlobMax: +(vb.maxArea" in src, "voidBlobMax 必须取单块最大面积 maxArea"
    assert "voidBlobMax: +(vb.totalArea" not in src, "voidBlobMax 不能再用总占比 totalArea"
    print("PASS  S7  voidBlobMax 取 maxArea（单块大洞），不再复用 totalArea")


def test_void_tolerance_unified():
    src = open(os.path.join(ROOT, "tools", "multiview.mjs"), encoding="utf-8").read()
    assert "const VOID_TOL = 6;" in src, "缺少统一容差常量 VOID_TOL=6"
    # 三处 isVoid / vmask 都必须用 VOID_TOL，不能残留硬编码 ±8 的 refBg 判定
    assert "refBg[0]) <= 8" not in src, "掩码仍残留 ±8 的 refBg 判定"
    n_tol = src.count("<= VOID_TOL")
    assert n_tol >= 9, "三处（整幅/掩码/横段 ×3 通道）应共用 VOID_TOL，实际 %d 处" % n_tol
    print("PASS  S8  空洞容差统一为 VOID_TOL=6（三处共用，无残留 ±8）")


def test_top_view_uses_refbg():
    src = open(os.path.join(ROOT, "tools", "multiview.mjs"), encoding="utf-8").read()
    assert "metrics(decodePNG(top), refBg)" in src, "俯视图必须传 refBg，与主视角口径一致"
    assert "metrics(decodePNG(top))" not in src, "俯视图不能退回画面主色判定"
    print("PASS  S9  俯视图传 refBg，与主视角空洞判据口径一致")


def test_callsite_contract():
    src = open(os.path.join(ROOT, "tools", "score_params.py"), encoding="utf-8").read()
    assert 'score(agg) if agg else' not in src, "空 agg 的守卫应收敛到 score() 内部"
    print("PASS  S10 调用点把空/缺项判定统一交给 score()，无旁路")


if __name__ == "__main__":
    test_score_full()
    test_score_empty_is_invalid()
    test_score_missing_key_is_invalid()
    test_score_nonnumeric_is_invalid()
    test_score_over_threshold()
    test_score_weights_sum_to_one()
    test_void_blobmax_semantics()
    test_void_tolerance_unified()
    test_top_view_uses_refbg()
    test_callsite_contract()
    print("\n评分器回归全部通过（10 项）")
