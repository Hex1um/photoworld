#!/usr/bin/env python3
"""按图自适应：每张照片烘焙多种表示 → 32 视角质检 → 按指标选最好的那个。

    python tools/tune_per_image.py --list corpus_test/list.txt --out out/tune
    python tools/tune_per_image.py --corpus corpus_test --variants mesh,points
    python tools/tune_per_image.py --list ... --views 32 --gl hw --apply

为什么要"每张图独立"：同一套参数在不同内容上最优解相反 ——
山脊/建筑要网格（点云侧面是点雾），而枝叶贴脸的内容用网格必然被剪成碎片。
全局阈值试过（相邻深度比 > 4 的像素占比），在这批图上只有 0.0000 vs 0.0054 的
差距，**薄到不可靠**。所以改成"用渲染结果投票"：同一机位、同一指标口径，
谁赢就用谁。

评分用 **Borda 计数（名次和）**，不做加权求和：各指标量纲差 1~2 个数量级
（void ~0.01、stripe ~1.2、centerHF ~0.03），硬凑权重等于用魔法数掩盖判断。
名次和只依赖"这组里谁更好"，对量纲免疫。硬约束：空洞率超过最优者 1.6 倍的
变体直接淘汰（洞比糊更不能接受）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "pipeline", "v2"))

LOWER_IS_BETTER = ["voidFrac", "innerVoid", "lowVoid", "borderVoid", "darkFrac", "stripe"]
HIGHER_IS_BETTER = ["centerHF", "centerLapVar", "detail", "localContrast", "hfEnergy"]


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"命令失败({r.returncode}): {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


def bake_variants(images, out_root, variants, extra):
    """每张图每个变体烘焙一个 GLB，返回 [(name, variant, glb_rel)]"""
    import api  # pipeline/v2/api.py
    rows = []
    for img in images:
        name = os.path.splitext(os.path.basename(img))[0]
        data = open(img, "rb").read()
        for v in variants:
            t0 = time.time()
            glb, man = api.bake(data, repr=v, **extra)
            d = os.path.join(out_root, name, v)
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, "world.glb")
            with open(p, "wb") as f:
                f.write(glb)
            with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(man, f, ensure_ascii=False, indent=1)
            rows.append({
                "name": name, "variant": v,
                "glb": os.path.relpath(p, ROOT).replace("\\", "/"),
                "mb": round(len(glb) / 1e6, 2),
                "repr": man.get("repr"), "scatter": man.get("scatter"),
                "secs": round(time.time() - t0, 1),
            })
            print(f"  {name:22} {v:7} {rows[-1]['mb']:6.2f}MB "
                  f"repr={rows[-1]['repr']:6} scatter={rows[-1]['scatter']} "
                  f"{rows[-1]['secs']}s")
    return rows


def analyze(rows, out_dir, views, gl, base):
    """一次调起 multiview，把所有变体放在**同一批机位与指标口径**下测。"""
    os.makedirs(out_dir, exist_ok=True)
    lst = os.path.join(out_dir, "_glb_list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r["glb"] + "\n")
    js = os.path.join(out_dir, "multiview.json")
    run(["node", os.path.join(ROOT, "tools", "multiview.mjs"),
         "--base", base, "--glb-list", lst, "--views", str(views),
         "--out", os.path.join(out_dir, "shots"), "--json", js, "--gl", gl])
    return json.load(open(js, encoding="utf-8"))


def borda(variants, metrics):
    """名次和：每个指标排序，名次累加。返回 {tag: score}，小者优。"""
    score = {t: 0 for t in variants}
    for m in metrics:
        vals = [(t, variants[t].get(m)) for t in variants if variants[t].get(m) is not None]
        if len(vals) < 2:
            continue
        # 排序方向统一成"越小越好"
        vals.sort(key=lambda kv: kv[1] if m in LOWER_IS_BETTER else -kv[1])
        for rank, (t, _) in enumerate(vals):
            score[t] += rank
    return score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="")
    ap.add_argument("--corpus", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="out/tune")
    ap.add_argument("--variants", default="mesh,points")
    ap.add_argument("--views", type=int, default=32)
    ap.add_argument("--gl", default="hw", choices=["hw", "sw"])
    ap.add_argument("--base", default="http://127.0.0.1:8199")
    ap.add_argument("--report", default="reports/tune")
    ap.add_argument("--far-split", type=float, default=120.0)
    ap.add_argument("--edge-ratio", type=float, default=2.4)
    args = ap.parse_args()

    images = []
    if args.list:
        images = [l.strip() for l in open(args.list, encoding="utf-8")
                  if l.strip() and not l.startswith("#")]
        images = [os.path.join(ROOT, i.split("|")[0]) for i in images]
    elif args.corpus:
        images = sorted(glob.glob(os.path.join(args.corpus, "*.jpg")))
    if not images:
        print("没有输入图片：--list 或 --corpus", file=sys.stderr)
        return 2
    if args.limit:
        images = images[:args.limit]

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"=== 按图自适应：{len(images)} 张 × {len(variants)} 变体 × {args.views} 视角 ===")

    extra = {"far_split": args.far_split, "edge_ratio": args.edge_ratio}
    t0 = time.time()
    print(f"\n[1/3] 烘焙（{len(images)*len(variants)} 个世界）")
    rows = bake_variants(images, args.out, variants, extra)
    print(f"      用时 {time.time()-t0:.0f}s")

    print(f"\n[2/3] 32 视角质检（{len(rows)*args.views} 张截图）")
    rep_dir = os.path.join(ROOT, args.report)
    data = analyze(rows, rep_dir, args.views, args.gl, args.base)
    # 用截图文件名把 multiview 的结果对回 (name, variant)：tag = <变体>_<图片名>
    by_tag = {}
    for r in rows:
        tag = os.path.basename(os.path.dirname(r["glb"])) + "_" + \
              os.path.basename(os.path.dirname(os.path.dirname(r["glb"])))
        by_tag[tag] = r
    for rec in data.get("results", []):
        f = (rec.get("views") or [{}])[0].get("file", "")
        tag = os.path.basename(f).rsplit("_", 1)[0] if f else ""
        if tag in by_tag and rec.get("agg"):
            by_tag[tag]["agg"] = rec["agg"]
            by_tag[tag]["ok"] = rec.get("ok")

    print(f"\n[3/3] 选优（Borda 名次和 + 空洞硬约束）")
    per_image = {}
    for r in rows:
        per_image.setdefault(r["name"], {})[r["variant"]] = r

    decisions, table = [], []
    for name, group in per_image.items():
        usable = {v: r for v, r in group.items() if r.get("agg")}
        if len(usable) < 2:
            best = next(iter(group))
            decisions.append({"name": name, "chosen": best, "reason": "只有一个变体可用"})
            continue
        voids = {v: r["agg"]["voidFrac"] for v, r in usable.items()}
        vmin = min(voids.values())
        # 硬约束：空洞率超过最优者 1.6 倍的直接淘汰（洞比糊更不能接受）。
        # 只剩一个幸存者时**直接选它** —— 早期实现这里"退回全量比较"，
        # 等于让约束完全失效（秋叶那张点云空洞高 8 倍仍然被选中）。
        survivors = {v: r for v, r in usable.items() if voids[v] <= max(0.004, vmin * 1.6)}
        if len(survivors) == 1:
            chosen = next(iter(survivors))
            score = {chosen: 0}
            decisions.append({"name": name, "chosen": chosen,
                              "reason": "空洞约束淘汰其余变体",
                              "void": {v: round(voids[v], 4) for v in usable},
                              "stripe": {v: usable[v]["agg"]["stripe"] for v in usable},
                              "centerHF": {v: usable[v]["agg"]["centerHF"] for v in usable}})
        else:
            pool = survivors or usable
            score = borda({v: r["agg"] for v, r in pool.items()},
                          LOWER_IS_BETTER + HIGHER_IS_BETTER)
            chosen = min(score, key=score.get)
        if len(survivors) != 1:
            decisions.append({
                "name": name, "chosen": chosen, "score": score,
                "void": {v: round(voids[v], 4) for v in usable},
                "stripe": {v: usable[v]["agg"]["stripe"] for v in usable},
                "centerHF": {v: usable[v]["agg"]["centerHF"] for v in usable},
            })
        row = [name, chosen] + [f"{v}:{'✓' if v == chosen else ' '}"
                                f" void {voids[v]:.4f} stripe {usable[v]['agg']['stripe']:.2f} "
                                f"hf {usable[v]['agg']['centerHF']:.4f}" for v in usable]
        table.append(row)

    for row in table:
        print("  " + "  ".join(row))

    os.makedirs(rep_dir, exist_ok=True)
    out_json = os.path.join(rep_dir, "per_image.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"views": args.views, "gl": args.gl, "variants": variants,
                   "decisions": decisions, "rows": rows,
                   "elapsed_s": round(time.time() - t0, 1)}, f, ensure_ascii=False, indent=1)
    print(f"\n报告 → {out_json}")

    # 从结果反推自适应阈值：散布度在"点云赢"与"网格赢"两组里的分布
    print(f"\n散布度可分离性：点云胜出 {sum(1 for d in decisions if d.get('chosen')=='points')}"
          f" / 网格胜出 {sum(1 for d in decisions if d.get('chosen')=='mesh')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
