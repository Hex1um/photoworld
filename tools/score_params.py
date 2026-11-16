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
import math
import os
import subprocess
import sys
import tempfile
import threading
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
                            # tasklist 在中文 Windows 上输出 **GBK**，
                            # text=True 会抛 UnicodeDecodeError（0xcf），
                            # 而这个异常发生在子线程里，主线程只看到 None →
                            # 评分直接失败（实测整轮训练全 0 分）。
                            _r = _sp.run(["tasklist", "/FI", f"PID eq {pid}"],
                                         capture_output=True)
                            out = (_r.stdout or b"").decode("cp936", "replace")
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


# api.py 模块单例。它顶层持有两个**进程级**状态：_model（1.3GB 深度权重）
# api.py 进程内单例：旧实现每次 bake() 都 exec_module 重建 api.py，把它的
# _model（1.3GB 权重）和 _cache（内容哈希 GLB 缓存）清零——每张图重载模型、
# 缓存永远命中不了（单步 290~380 秒的主要来源）。现在模块只加载一次，
# 并用 (mtime_ns, size) 指纹在 api.py 变更时重载。模块级变量即可：训练侧
# adapter 缓存本模块、score_server 走自己的 api 缓存，都不会重建本模块后调 bake()。
_API_CACHE: dict = {"fingerprint": None, "module": None, "loads": 0}
_API_LOAD_LOCK = threading.Lock()


def _get_api():
    """返回进程内共享的 api.py 模块；文件变更时自动重建。"""
    import importlib.util

    path = os.path.join(ROOT, "pipeline", "v2", "api.py")
    st = os.stat(path)
    fingerprint = (st.st_mtime_ns, st.st_size)
    with _API_LOAD_LOCK:
        if _API_CACHE["module"] is not None and _API_CACHE["fingerprint"] == fingerprint:
            return _API_CACHE["module"]
        spec = importlib.util.spec_from_file_location("v2api", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _API_CACHE.update(fingerprint=fingerprint, module=mod)
        _API_CACHE["loads"] += 1
        return mod


def bake(image: str, params: dict, out_glb: str) -> dict:
    """按参数烘焙一张图，返回 manifest。"""
    api = _get_api()

    with open(os.path.join(ROOT, image) if not os.path.isabs(image) else image, "rb") as f:
        data = f.read()
    glb, man = api.bake(data, **params)
    with open(out_glb, "wb") as f:
        f.write(glb)
    return man


def aggregate_views(views: list) -> dict:
    """把逐视角指标聚合成 agg：判据项取「最差机位」，展示项取均值。

    旧口径把 voidFrac/innerVoid/lowVoid/stripe 四个判据取均值，单个机位的严重
    缺陷（贯穿缝、大洞、拉丝）会被另外 7 个好视角摊薄到约 1/8，训练目标里
    根本看不到真实失败。改为：THRESH 的 8 项一律取所有视角的最大值（最差），
    与块数类（voidBlobs/voidBlobMax/voidRunMax，multiview 侧本就取 max）统一。
    detail/darkFrac/borderVoid/centerHF/centerLapVar 等只用于展示/诊断，继续取均值。

    缺项的判据返回 None——交给 score() 判 invalid，不给"缺项=满分"留口子。
    """
    if not views:
        return {}
    out: dict = {}
    for k in THRESH:
        vals = [v[k] for v in views if isinstance(v, dict) and isinstance(v.get(k), (int, float))]
        out[k] = max(vals) if len(vals) == len(views) else None
    for k in ("detail", "darkFrac", "borderVoid", "centerHF", "centerLapVar"):
        vals = [v[k] for v in views if isinstance(v, dict) and isinstance(v.get(k), (int, float))]
        if vals:
            out[k] = round(sum(vals) / len(vals), 4)
    return out


def render_score(glb_rel: str, base: str, views: int, out_dir: str) -> dict:
    """调 multiview 渲染并返回 agg 指标。"""
    lst = os.path.join(out_dir, "_list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        f.write(glb_rel + "\n")
    js = os.path.join(out_dir, "mv.json")
    cmd = ["node", os.path.join(ROOT, "tools", "multiview.mjs"),
           "--base", base, "--glb-list", lst, "--views", str(views),
           "--out", os.path.join(out_dir, "shots"), "--json", js, "--gl", "hw"]
    # **不要用 text=True / encoding=**：multiview（node）经由管道吐出的字节里
    # 混着 GBK 片段（实测 position 4 的 0xcf），text 模式会在 reader 线程里
    # 直接抛 UnicodeDecodeError，而且异常发生在子线程、主线程只看到一堆 None。
    # 收字节、自己按 utf-8 + replace 解码，稳。
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    out = (r.stdout or b"").decode("utf-8", "replace")
    err = (r.stderr or b"").decode("utf-8", "replace")
    if r.returncode != 0:
        raise RuntimeError(f"multiview 失败({r.returncode}): {(err or out)[-800:]}")
    d = json.load(open(js, encoding="utf-8"))
    res = d.get("results") or []
    if not res:
        raise RuntimeError("multiview 没有返回结果")
    rec = res[0]
    # 用逐视角指标重算 agg：判据项取最差机位（见 aggregate_views），
    # 不再直接信任 multiview.mjs 里取均值的 agg——那是观察口径，不是判据口径。
    per_view = rec.get("views") or []
    if per_view:
        return aggregate_views(per_view)
    return rec.get("agg") or {}


def score(agg: dict) -> tuple[int, float, list[str]]:
    """把指标折成 (hard, soft, 失败项)。

    契约（2026-11-16 v2.13.2）：**指标缺项 = 无效，不是满分**。
    agg 必须给出 THRESH 的全部键；少任意一项（或为空）一律返回
    hard=0 / soft=0.0，并在失败项里标 `invalid:missing:<key>`。
    旧实现里 `v is None` 被跳过 → 渲染坏了却算不出指标时反而拿满分，
    是奖励作弊入口。失败项统一带 `invalid:` 前缀，便于上层把"无效"
    和"真实低分"区分开（无效样本不该进反馈/择优）。
    """
    if not isinstance(agg, dict) or not agg:
        return 0, 0.0, ["invalid:empty"]
    missing = [k for k in THRESH if agg.get(k) is None]
    if missing:
        return 0, 0.0, ["invalid:missing:" + k for k in missing]
    # 非数值 / NaN / Inf 同样按无效处理：
    # NaN 让所有 `v > t` 为假 → 旧逻辑会判 hard=1 soft=1.0（满分），必须堵住。
    bad = []
    for k in THRESH:
        try:
            if not math.isfinite(float(agg[k])):
                bad.append(k)
        except (TypeError, ValueError):
            bad.append(k)
    if bad:
        return 0, 0.0, ["invalid:nonnumeric:" + k for k in bad]
    fail = []
    for k, t in THRESH.items():
        v = agg[k]
        if v > t:
            fail.append(k)
    hard = 1 if not fail else 0
    # soft：每项按"超阈值程度"扣分，越界越多扣得越多（用阈值归一）
    soft = 1.0
    for k, t in THRESH.items():
        v = agg[k]
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
        hard, soft, fail = score(agg)   # 空/缺项由 score() 统一判 invalid
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
