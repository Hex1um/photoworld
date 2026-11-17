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


def _multiview_src():
    """指标公式的权威来源已从 multiview.mjs 迁到 multiview.py（v3.0.6）；
    .mjs 现在只是薄包装。这里读 .py 源做静态断言。"""
    return open(os.path.join(ROOT, "tools", "multiview.py"), encoding="utf-8").read()


def test_void_blobmax_semantics():
    src = _multiview_src()
    assert '"voidBlobMax": _fixed(vb[1]' in src or '"voidBlobMax"' in src and 'vb[1]' in src, \
        "voidBlobMax 必须取单块最大面积 maxArea（blob_stats 返回 (count,maxArea,total)）"
    assert 'vb[2]' not in src.split('"voidBlobMax"')[1].split(',')[0], \
        "voidBlobMax 不能再用总占比 totalArea"
    print("PASS  S7  voidBlobMax 取 maxArea（单块大洞），不再复用 totalArea")


def test_void_tolerance_unified():
    src = _multiview_src()
    assert "VOID_TOL = 6" in src, "缺少统一容差常量 VOID_TOL=6"
    # 三处 isVoid / vmask 都必须用 VOID_TOL，不能残留硬编码 ±8 的 refBg 判定
    assert "ref_bg[0]) <= 8" not in src, "掩码仍残留 ±8 的 refBg 判定"
    n_tol = src.count("<= VOID_TOL")
    assert n_tol >= 9, "三处（整幅/掩码/横段 ×3 通道）应共用 VOID_TOL，实际 %d 处" % n_tol
    print("PASS  S8  空洞容差统一为 VOID_TOL=6（三处共用，无残留 ±8）")


def test_top_view_uses_refbg():
    src = _multiview_src()
    assert "image_metrics(decode_png(buf), ref_bg)" in src, "俯视图必须传 refBg，与主视角口径一致"
    assert 'rec["top"]' in src, "俯视结果必须落到 rec.top"
    print("PASS  S9  俯视图传 refBg，与主视角空洞判据口径一致")


def test_callsite_contract():
    src = open(os.path.join(ROOT, "tools", "score_params.py"), encoding="utf-8").read()
    assert 'score(agg) if agg else' not in src, "空 agg 的守卫应收敛到 score() 内部"
    print("PASS  S10 调用点把空/缺项判定统一交给 score()，无旁路")


def test_aggregate_views_worst_camera():
    """判据 8 项必须取最差机位，不能被其他视角的均值摊薄。"""
    base = dict(stripe=0.5, innerVoid=0.01, voidFrac=0.02, lowVoid=0.01,
                voidBlobs=10, streakBlobs=20, voidRunMax=0.1, voidBlobMax=0.05,
                detail=0.4, darkFrac=0.01, borderVoid=0.02, centerHF=0.3, centerLapVar=0.2)
    bad = dict(base)
    bad.update(stripe=3.5, innerVoid=0.20, voidFrac=0.30, lowVoid=0.25,
               voidBlobs=400, streakBlobs=350, voidRunMax=0.8, voidBlobMax=0.5)
    good = [dict(base, detail=0.6), dict(base, detail=0.2)]
    agg = SP.aggregate_views(good + [bad])
    for k in SP.THRESH:
        assert agg[k] == bad[k], "判据 %s 应取最差机位 %s，实际 %s" % (k, bad[k], agg[k])
    # 展示项取均值（detail: (0.6+0.2+0.4)/3 = 0.4）
    assert math.isclose(agg["detail"], 0.4, abs_tol=1e-4), agg["detail"]
    # 最差机位进入 score 后确实判失败（旧均值口径下 stripe=(0.5*2+3.5)/3≈1.5<2.0 会漏判）
    hard, soft, fail = SP.score(agg)
    assert hard == 0 and "stripe" in fail, (hard, soft, fail)
    print("PASS  S11 aggregate_views 判据取最差机位（单视角缺陷不被摊薄），展示项取均值")


def test_aggregate_views_missing_is_none():
    """某视角缺判据项时该判据返回 None，交 score() 判 invalid。"""
    views = [dict(stripe=1.0, innerVoid=0.01)]  # 只有两项，其余全缺
    agg = SP.aggregate_views(views)
    assert agg["voidFrac"] is None, "缺项应为 None，不能补 0 或跳过"
    assert SP.score(agg)[0] == 0, "缺项 agg 必须判 invalid（hard=0）"
    assert SP.aggregate_views([]) == {}
    print("PASS  S12 aggregate_views 缺项返回 None → score 判 invalid，空 views 返回 {}")


def test_mv_agg_worst_contract():
    """multiview 展示用 agg 的 4 个判据项也须取 worst，与判据口径一致（v3.0.6 权威在 .py）。"""
    src = _multiview_src()
    for k in ("voidFrac", "innerVoid", "lowVoid", "stripe"):
        assert '"%s": worst_of(v, "%s")' % (k, k) in src, "%s 展示口径未同步为 worst" % k
    for k in ("darkFrac", "borderVoid", "detail"):
        assert '"%s": mean_of(v, "%s")' % (k, k) in src, "%s 应保留 mean（展示项）" % k
    print("PASS  S13 multiview 展示 agg：4 判据取 worst，展示项保留 mean")


def test_render_score_uses_aggregate_views():
    """render_score 必须走 aggregate_views 重算，而非直接返回 multiview 的均值 agg。"""
    src = open(os.path.join(ROOT, "tools", "score_params.py"), encoding="utf-8").read()
    assert "return aggregate_views(per_view)" in src, "render_score 应用 views 重算"
    assert 'per_view = rec.get("views")' in src, "应读逐视角 views"
    print("PASS  S14 render_score 用 aggregate_views 从 views 重算，不用均值 agg")


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
    test_aggregate_views_worst_camera()
    test_aggregate_views_missing_is_none()
    test_mv_agg_worst_contract()
    test_render_score_uses_aggregate_views()
    print("\n评分器回归全部通过（14 项）")
