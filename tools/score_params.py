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
    "stripe": 2.0,          # 拉丝强度
    "innerVoid": 0.06,      # 中景空洞占比
    "voidFrac": 0.10,       # 露空白占比
    "lowVoid": 0.08,        # 脚下空洞占比
    "voidBlobs": 260,       # **空白块数**：同样占比，碎成很多小块比一大块好，但要有限度
    "streakBlobs": 200,     # **拉丝块数**：竖条/拉花的连通域个数
    "voidRunMax": 0.50,     # **最长横向空段**（按画面宽度归一）：贯穿式的缝最难受
    "voidBlobMax": 0.22,    # 单块空白最大占比：避免"一块就占了半屏"
}
# soft 分数里各项的权重（拉丝与贯穿缝最影响观感）
WEIGHT = {
    "stripe": 0.24,
    "innerVoid": 0.12,
    "voidFrac": 0.12,
    "lowVoid": 0.12,
    "voidBlobs": 0.12,
    "streakBlobs": 0.16,
    "voidRunMax": 0.08,
    "voidBlobMax": 0.04,
}



def _gpu_lock():
    """全局文件锁：保证同一时刻只有一个 multiview 在跑。

    multiview.mjs 启动浏览器用固定调试端口，两个实例并发会互相抢端口并失败
    （症状就是"multiview 失败:" 后面 stderr 空）。训练里一个 batch 有 2 张图，
    并发是常态，所以必须串行化。
    """
    import contextlib
    import time as _t

    lock_path = os.path.join(ROOT, "out", "_score", ".mv.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    @contextlib.contextmanager
    def cm():
        fh = None
        for _ in range(1800):                      # 最多等 30 分钟（别的任务可能很慢）
            try:
                fh = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fh, str(os.getpid()).encode())
                break
            except FileExistsError:
                # 陈旧锁必须清掉，否则一次崩溃就把后续所有评分永久卡死。
                # 判据两条：① 超过 300 秒必定陈旧（单次 multiview ≤ 120s）；
                #          ② 锁里记的 PID 已经不在了。
                stale = False
                try:
                    age = _t.time() - os.path.getmtime(lock_path)
                    if age > 300:
                        stale = True
                    else:
                        try:
                            pid = int(open(lock_path).read().strip() or 0)
                        except Exception:
                            pid = 0
                        if pid:
                            import subprocess as _sp
                            out = _sp.run(["tasklist", "/FI", f"PID eq {pid}"],
                                          capture_output=True, text=True).stdout
                            if str(pid) not in out:      # 进程没了 → 锁是遗产
                                stale = True
                except OSError:
                    stale = True
                if stale:
                    try:
                        os.remove(lock_path)
                        continue
                    except OSError:
                        pass
                _t.sleep(1.0)
        try:
            yield
        finally:
            if fh is not None:
                try:
                    os.close(fh)
                except OSError:
                    pass
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
    return cm()


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
    ap.add_argument("--server", default=os.environ.get("PW_SCORE_SERVER", ""),
                    help="常驻评分服务地址（如 http://127.0.0.1:8300）；给了就走它，不再本地烘焙")
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "out", "_score"), exist_ok=True)

    t0 = time.time()
    params = json.loads(args.params or "{}")
    if args.server:
        # 常驻服务：模型只加载一次，单图 ~15s（本地起进程要 60~80s）
        import urllib.parse
        import urllib.request
        qs = urllib.parse.urlencode({"image": args.image, "params": json.dumps(params),
                                     "base": args.base, "views": args.views})
        try:
            with urllib.request.urlopen(f"{args.server.rstrip('/')}/score?{qs}", timeout=1800) as r:
                d = json.loads(r.read().decode("utf-8"))
        except Exception as e:                          # noqa: BLE001
            print(json.dumps({"ok": False, "hard": 0, "soft": 0.0, "error": str(e)[:300],
                              "secs": round(time.time() - t0, 1)}, ensure_ascii=False))
            return 0
        agg = d.get("metrics") or {}
        hard, soft, fail = score(agg) if agg else (0, 0.0, ["no-metrics"])
        print(json.dumps({"ok": bool(d.get("ok")), "hard": hard, "soft": soft,
                          "metrics": {k: round(float(v), 4) for k, v in agg.items()
                                      if isinstance(v, (int, float))},
                          "fail": fail, "repr": d.get("repr"),
                          "secs": round(time.time() - t0, 1)}, ensure_ascii=False))
        return 0
    # 训练循环里默认关掉 SD 外扩：它一张图要 40~150 秒，会把 rollout 拖成不可用。
    params.setdefault("outpaint", 1.0)

    # 产物必须落在**站点根目录内**：静态服务只服务 ROOT，放到系统 temp 里
    # 转出来的相对路径是 ../../AppData/...，multiview 拿不到（实测报错）。
    # 固定工作目录（见 score_server.py 的说明：频繁 rmtree 会触发批量删除保护）
    tmp = os.path.join(ROOT, "out", "_score", "cli")
    os.makedirs(tmp, exist_ok=True)
    try:
        with _gpu_lock():
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
            for name in ("world.glb", "_list.txt", "mv.json"):
                try:
                    os.remove(os.path.join(tmp, name))
                except OSError:
                    pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
