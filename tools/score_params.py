#!/usr/bin/env python
"""给 SkillOpt 用的单图快速评分器（rollout 的评测器）。

流程：按参数烘焙 → 用 4 个机位渲染 → 算验收指标 → 返回 hard/soft 分数。

与 `tools/tune_per_image.py` + `tools/acc_report.py` 的关系：
  · 那套是"32 图 × 32 机位"的完整验收，一张图要 1~2 分钟；
  · 这里为了训练循环能跑起来，用 4 机位（前/侧/后/俯视），单图 10~20 秒。
  · 阈值与 acc_report 完全一致，所以分数可以和 `docs/acc32b.md` 对读。

用法：
  python tools/score_params.py --image worlds/desert.jpg --base http://127.0.0.1:8199
  python tools/score_params.py --image worlds/desert.jpg --params '{"far_split": 90}'

输出（stdout 一行 JSON）：
  {"ok": true, "hard": 1, "soft": 0.87, "metrics": {...}, "fail": ["..."], "secs": 12.3}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 tools/acc_report.py 一致的判定阈值（改这里要同步改那边）
THRESH = {
    "stripe": 2.0,
    "innerVoid": 0.06,
    "voidFrac": 0.10,
    "lowVoid": 0.08,
}
# soft 分数里各项的权重（拉丝最影响观感，权重最高）
WEIGHT = {
    "stripe": 0.40,
    "innerVoid": 0.20,
    "voidFrac": 0.20,
    "lowVoid": 0.20,
}


def bake(image: str, params: dict, out_glb: str) -> dict:
    """按参数烘焙一张图，返回 manifest。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "v2api", os.path.join(ROOT, "pipeline", "v2", "api.py"))
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)

    with open(os.path.join(ROOT, image) if not os.path.isabs(image) else image, "rb") as f:
        data = f.read()
    glb, man = api.bake(data, **params)
    with open(out_glb, "wb") as f:
        f.write(glb)
    return man


def render_score(glb_rel: str, base: str, views: int, out_dir: str) -> dict:
    """调 multiview 渲染并返回 agg 指标。"""
    lst = os.path.join(out_dir, "_list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        f.write(glb_rel + "\n")
    js = os.path.join(out_dir, "mv.json")
    cmd = ["node", os.path.join(ROOT, "tools", "multiview.mjs"),
           "--base", base, "--glb-list", lst, "--views", str(views),
           "--out", os.path.join(out_dir, "shots"), "--json", js, "--gl", "hw"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"multiview 失败: {r.stderr[-800:]}")
    d = json.load(open(js, encoding="utf-8"))
    res = d.get("results") or []
    if not res:
        raise RuntimeError("multiview 没有返回结果")
    return res[0].get("agg") or {}


def score(agg: dict) -> tuple[int, float, list[str]]:
    """把指标折成 (hard, soft, 失败项)。"""
    fail = []
    for k, t in THRESH.items():
        v = agg.get(k)
        if v is not None and v > t:
            fail.append(k)
    hard = 1 if not fail else 0
    # soft：每项按"超阈值程度"扣分，越界越多扣得越多（用阈值归一）
    soft = 1.0
    for k, t in THRESH.items():
        v = agg.get(k)
        if v is None:
            continue
        over = max(0.0, v - t) / max(1e-6, t)
        soft -= WEIGHT[k] * min(1.0, over)
    return hard, round(max(0.0, soft), 4), fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--params", default="{}", help="烘焙参数字典（JSON）")
    ap.add_argument("--base", default=os.environ.get("PW_BASE", "http://127.0.0.1:8199"))
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="保留中间产物便于排查")
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "out", "_score"), exist_ok=True)

    t0 = time.time()
    params = json.loads(args.params or "{}")
    # 训练循环里默认关掉 SD 外扩：它一张图要 40~150 秒，会把 rollout 拖成不可用。
    params.setdefault("outpaint", 1.0)

    # 产物必须落在**站点根目录内**：静态服务只服务 ROOT，放到系统 temp 里
    # 转出来的相对路径是 ../../AppData/...，multiview 拿不到（实测报错）。
    tmp = tempfile.mkdtemp(prefix="score-", dir=os.path.join(ROOT, "out", "_score"))
    try:
        glb = os.path.join(tmp, "world.glb")
        man = bake(args.image, params, glb)
        rel = os.path.relpath(glb, ROOT).replace("\\", "/")
        agg = render_score(rel, args.base, args.views, tmp)
        hard, soft, fail = score(agg)
        print(json.dumps({
            "ok": True, "hard": hard, "soft": soft,
            "metrics": {k: round(float(v), 4) for k, v in agg.items() if isinstance(v, (int, float))},
            "fail": fail, "repr": man.get("repr"), "secs": round(time.time() - t0, 1),
        }, ensure_ascii=False))
    except Exception as e:                                  # noqa: BLE001
        print(json.dumps({"ok": False, "hard": 0, "soft": 0.0, "error": str(e)[:400],
                          "secs": round(time.time() - t0, 1)}, ensure_ascii=False))
    finally:
        if not args.keep:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
