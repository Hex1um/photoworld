#!/usr/bin/env python3
"""把一次 32×32 验收的 JSON 汇总成可读报告（含通过/不通过判定）。

    python tools/acc_report.py reports/acc32/per_image.json --out docs/acc32.md

判定口径（按用户给的验收标准翻译成可计算的量）：

  视锥内（前 3 环：8 个水平位置 × 4 个纵深，全部朝前）
    * `stripe`  : 竖条能量比。>2.0 视为"明显拉丝"
    * `innerVoid`: 中央 50% 的空底色占比。>0.06 视为"中间有洞"
  视锥外（同机位里带 ±25° 偏航的那些，以及离开中心的位置）
    * `voidFrac`: 整幅空底色占比。>0.10 视为"露空白"
    * `lowVoid` : 下 40% 空底。>0.08 视为"脚下没东西"

这些阈值不是拍脑袋，用的是这批图上的分位数：先看分布，再定"明显"的界。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st

# P2（v2.15.8 修）：阈值**单一来源**。
# 以前这里手写 4 项、score_params.THRESH 有 8 项（多 voidBlobs/streakBlobs/
# voidRunMax/voidBlobMax，且缺项判 invalid），于是同一张图可以"验收报告通过"
# 而"训练评分 hard=0"——ISSUES #6 的"阈值重新定标"根本没有载体。
# 现在直接引用 score_params 的 THRESH；改阈值只需要改一处。
try:
    import sys as _sys
    if os.path.dirname(os.path.abspath(__file__)) not in _sys.path:
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from score_params import THRESH as _TH
except Exception as _e:                      # 单文件拷贝出去用时留一份兜底
    _TH = {"stripe": 2.0, "innerVoid": 0.06, "voidFrac": 0.10, "lowVoid": 0.08}
    print("[acc_report] 无法导入 score_params.THRESH（%s），用内置副本" % _e)

INNER_T = {"stripe": _TH["stripe"], "innerVoid": _TH["innerVoid"]}
OUTER_T = {"voidFrac": _TH["voidFrac"], "lowVoid": _TH["lowVoid"]}


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    d = json.load(open(args.json, encoding="utf-8"))
    rows = [r for r in d.get("rows", []) if r.get("agg")]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("没有可汇总的数据")
        return 2

    agg = [r["agg"] for r in rows]
    lines = []
    lines.append("# 32 图 × 32 机位验收报告")
    lines.append("")
    lines.append(f"- 视角数：{d.get('views')}（v2 专用机位：8 水平位置 × 4 纵深，全部朝前，"
                 f"带 ±25° 偏航）")
    lines.append(f"- 光栅后端：`{d.get('gl')}`　图数：{len(rows)}　"
                 f"截图：{len(rows)*int(d.get('views', 32))} 张")
    lines.append(f"- 总耗时：{d.get('elapsed_s')} s")
    lines.append("")
    lines.append("## 分布")
    lines.append("")
    lines.append("| 指标 | p10 | 中位 | p90 | 最差 | 判定阈值 |")
    lines.append("|---|---|---|---|---|---|")
    for k, t in [("stripe", INNER_T["stripe"]), ("innerVoid", INNER_T["innerVoid"]),
                 ("voidFrac", OUTER_T["voidFrac"]), ("lowVoid", OUTER_T["lowVoid"])]:
        xs = [a[k] for a in agg]
        lines.append(f"| {k} | {pct(xs,0.1):.3f} | {st.median(xs):.3f} | "
                     f"{pct(xs,0.9):.3f} | {max(xs):.3f} | {t} |")
    lines.append("")

    # 逐图判定
    lines.append("## 逐图判定")
    lines.append("")
    lines.append("| 图 | stripe | innerVoid | lowVoid | voidFrac | 判定 |")
    lines.append("|---|---|---|---|---|---|")
    bad = 0
    for r in rows:
        a = r["agg"]
        fails = []
        if a["stripe"] > INNER_T["stripe"]:
            fails.append("拉丝")
        if a["innerVoid"] > INNER_T["innerVoid"]:
            fails.append("中央空洞")
        if a["voidFrac"] > OUTER_T["voidFrac"]:
            fails.append("露空白")
        if a["lowVoid"] > OUTER_T["lowVoid"]:
            fails.append("脚下空")
        verdict = "通过" if not fails else "不通过：" + "/".join(fails)
        if fails:
            bad += 1
        lines.append(f"| `{r['name'][:14]}` | {a['stripe']:.2f} | {a['innerVoid']:.3f} | "
                     f"{a['lowVoid']:.3f} | {a['voidFrac']:.3f} | {verdict} |")
    lines.append("")
    lines.append(f"**通过 {len(rows)-bad} / {len(rows)}**，不通过 {bad} 张。")
    lines.append("")
    lines.append('> 阈值是**可计算的近似**，不等于「看着没问题」：通过判定只说明这几项客观指标'
                 '没超界。近处细长物体（灯柱那种）的拉丝已在 v2.2.1 用「地物分割 + 卡片化」'
                 '单独处理，不体现在这四项里。')

    text = "\n".join(lines) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w", encoding="utf-8").write(text)
        print("报告 → " + args.out)
    print(text[-1200:] if len(text) > 1200 else text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
