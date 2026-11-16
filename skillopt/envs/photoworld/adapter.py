"""photoworld 的 EnvAdapter：把"烘焙参数"当作可训练的技能内容。

rollout 做的事：
  1. 从 skill 文档里取 ```yaml 代码块中的 `params:`；
  2. 对 batch 里每张图跑 `photoworld/tools/score_params.py`（烘焙 + 渲染 + 打分）；
  3. 返回 [{"id", "hard", "soft", "metrics", ...}]。

reflect 用父类的通用实现（它拿 rollout 的结果去让优化器模型提议文本编辑）。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.photoworld.dataloader import PhotoWorldDataLoader

try:                                     # 目录名映射（与 reflect 读轨迹时用的一致）
    from skillopt.envs.task_output import task_output_segment
except Exception:                        # 旧版本没有这个模块时退化
    def task_output_segment(tid: str) -> str:
        return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(tid))

# 判定阈值（只用于把"为什么失败"写成一句人话，权威定义在 tools/score_params.py）
_THRESH_HINT = {
    "stripe": "拉丝强度", "streakBlobs": "拉丝块数", "innerVoid": "中景空洞占比",
    "voidFrac": "露空白占比", "voidBlobs": "空白块数", "voidBlobMax": "单块空白最大占比",
    "voidRunMax": "最长横向空段", "lowVoid": "脚下空洞占比",
}

# PHOTOWORLD 仓库根（可用环境变量覆盖，默认按本文件位置推：tools_skillopt 的兄弟目录）
_DEFAULT_ROOT = os.environ.get(
    "PHOTOWORLD_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "photoworld")),
)

_YAML_BLOCK = re.compile(r"```ya?ml\s*\n(.*?)```", re.S)


def parse_params(skill_content: str) -> dict[str, Any]:
    """从 skill 文档的 yaml 块里取 `params:`。

    取不到就返回空字典（= 用流水线默认值）。这样即使优化器把文档改坏了，
    rollout 也不会崩，只是退回默认参数 —— 训练循环宁可变差也不要中断。
    """
    try:
        import yaml
    except ImportError:
        return {}
    for block in _YAML_BLOCK.findall(skill_content or ""):
        try:
            doc = yaml.safe_load(block)
        except Exception:                                   # noqa: BLE001
            continue
        if isinstance(doc, dict) and isinstance(doc.get("params"), dict):
            return {k: v for k, v in doc["params"].items() if v is not None}

    # 兜底：分析器产出的是**散文式编辑**（"把 far_split 调到 90 会更稳"），
    # 未必保留 yaml 块。实测训练出的 best_skill.md 里 params 与种子完全一样，
    # 但正文多了一堆建议 —— 只认 yaml 块等于训练白做。
    # 这里再扫一遍正文里的 `key: value` 与 `key = value`。
    KNOWN = {"far_split": float, "ground_radius": float, "seg_edge_tau": float,
             "skirt_cap": float, "card_flatten": bool, "cliff_thr": float,
             "quad_stretch_max": float}
    found: dict[str, Any] = {}
    for key, cast in KNOWN.items():
        # 三种写法都要认：`key: 90`、`key = 90`、以及中文散文里的"把 key 调到 90"
        m = re.search(
            rf"{key}\s*(?:[:=]|调到|改为|设为|调整到|->|→)\s*([0-9.]+|true|false)",
            skill_content or "", re.I)
        if not m:
            continue
        raw = m.group(1).lower()
        try:
            found[key] = (raw == "true") if cast is bool else cast(raw)
        except ValueError:
            continue
    return found


class PhotoWorldAdapter(EnvAdapter):
    def __init__(self, split_dir: str = "", data_path: str = "", split_mode: str = "split_dir",
                 split_ratio: str = "2:1:7", split_seed: int = 42, split_output_dir: str = "",
                 workers: int = 1, analyst_workers: int = 2, failure_only: bool = False,
                 minibatch_size: int = 1, edit_budget: int = 3, seed: int = 42, limit: int = 0,
                 max_completion_tokens: int = 4096, base_url: str = "", views: int = 4,
                 root: str = "", score_server: str = "", inproc: bool = True) -> None:
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.max_completion_tokens = int(max_completion_tokens)
        self.base_url = base_url or os.environ.get("PW_BASE", "http://127.0.0.1:8199")
        self.views = views
        self.root = root or _DEFAULT_ROOT
        self.score_server = score_server or os.environ.get("PW_SCORE_SERVER", "")
        # inproc=True：不依赖任何常驻服务，训练进程内直接烘焙（默认，最稳）
        self.inproc = bool(inproc) and os.environ.get("PW_SCORE_INPROC", "1") != "0"
        self._mod = None
        self.dataloader = PhotoWorldDataLoader(
            split_dir=split_dir, data_path=data_path, split_mode=split_mode,
            split_ratio=split_ratio, split_seed=split_seed,
            split_output_dir=split_output_dir, seed=seed, limit=limit,
        )

    # ── 生命周期 ────────────────────────────────────────────────────
    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    # ── rollout ────────────────────────────────────────────────────
    def rollout(self, env_manager, skill_content: str, out_dir: str, **kwargs) -> list[dict]:
        params = parse_params(skill_content)
        os.makedirs(out_dir, exist_ok=True)
        rows: list[dict] = []
        pred_dir = os.path.join(out_dir, "predictions")
        for item in env_manager:
            row = self._score_one(item, params)
            # **写自然语言轨迹**：分析器读的是 predictions/<id>/conversation.json，
            # 而且里面得是"执行过程 + 反馈"。早先只把指标数字塞进结果字典，
            # 分析器拿到的是一堆裸数字 → 每次都产出 0 条编辑（[skip] no usable patches）。
            self._write_trajectory(pred_dir, item, params, row)
            rows.append(row)
            print(f"[photoworld] {item.get('id')} hard={row['hard']} soft={row['soft']} "
                  f"{row.get('metrics', {})}", flush=True)
        # 把这次用到的参数写下来，方便回溯"哪个版本的 skill 产出了什么分数"
        with open(os.path.join(out_dir, "rollout.json"), "w", encoding="utf-8") as f:
            json.dump({"params": params, "rows": rows}, f, ensure_ascii=False, indent=1)
        return rows

    def _score_one(self, item: dict, params: dict) -> dict:
        """打分。优先走**常驻评分服务**（模型只加载一次，单图 ~15s）。

        早先用子进程调 `score_params.py`：一张图 60~80 秒（每次都重新加载深度模型），
        而且在训练进程里会失败（multiview 报空 stderr，单跑却正常）。
        常驻服务把这层子进程去掉，两个问题一起解决。
        """
        d: dict = {}
        # ① 在**训练进程内**直接烘焙（最稳的一条路）：
        #    不走 HTTP、不走子进程，CUDA 调用发生在主线程；
        #    渲染仍交给 node（multiview 是独立进程，本来就不碰 CUDA）。
        #    为什么不用评分服务：在这套环境里后台服务反复被回收
        #    （表现为 Remote end closed / WinError 10061），把整轮 rollout 变成 0 分。
        if self.inproc:
            try:
                d = self._score_inproc(item, params)
            except Exception as e:                       # noqa: BLE001
                d = {"ok": False, "error": f"进程内评分失败: {str(e)[:240]}"}
        if not d and self.score_server:
            import urllib.parse
            import urllib.request
            qs = urllib.parse.urlencode({"image": item["image"],
                                         "params": json.dumps(params, ensure_ascii=False),
                                         "base": self.base_url, "views": self.views})
            try:
                url = f"{self.score_server.rstrip('/')}/score?{qs}"
                with urllib.request.urlopen(url, timeout=1800) as r:
                    d = json.loads(r.read().decode("utf-8"))
            except Exception as e:                           # noqa: BLE001
                d = {"ok": False, "error": f"评分服务不可用: {str(e)[:240]}"}
        if not d:
            cmd = [os.sys.executable, os.path.join(self.root, "tools", "score_params.py"),
                   "--image", item["image"], "--base", self.base_url,
                   "--views", str(self.views), "--params", json.dumps(params, ensure_ascii=False)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=self.root)
                line = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
                d = json.loads(line) if line.startswith("{") else {}
            except Exception as e:                           # noqa: BLE001
                d = {"ok": False, "error": str(e)[:300]}
        return {
            "id": item["id"],
            "hard": int(d.get("hard", 0)),
            "soft": float(d.get("soft", 0.0)),
            "metrics": d.get("metrics") or {},
            "fail": d.get("fail") or [],
            "error": d.get("error", ""),
            "task_type": item.get("task_type", "landscape"),
            "note": item.get("note", ""),
        }



    def _score_inproc(self, item: dict, params: dict) -> dict:
        """与 tools/score_params.py 同一条链路，但在本进程里跑（共享已加载的模型）。"""
        import importlib.util
        if self._mod is None:
            spec = importlib.util.spec_from_file_location(
                "pwscore_inproc", os.path.join(self.root, "tools", "score_params.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._mod = mod
        mod = self._mod

        p = dict(params or {})
        p.setdefault("outpaint", 1.0)          # 训练里关 SD 外扩（一张 40~150s）
        tmp = os.path.join(self.root, "out", "_score", "inproc")
        os.makedirs(tmp, exist_ok=True)
        import time as _t
        t0 = _t.time()
        # 全局文件锁：两个 multiview 并发会抢浏览器端口（失败时 stderr 为空）
        with mod._gpu_lock():
            glb_path = os.path.join(tmp, "world.glb")
            man = mod.bake(item["image"], p, glb_path)
            rel = os.path.relpath(glb_path, self.root).replace("\\", "/")
            agg = mod.render_score(rel, self.base_url, self.views, tmp)
        hard, soft, fail = mod.score(agg)
        return {"ok": True, "metrics": agg, "hard": hard, "soft": soft, "fail": fail,
                "repr": (man or {}).get("repr"), "secs": round(_t.time() - t0, 1)}

    # ── 轨迹：把"这次做了什么、哪里坏了"写成分析器能读的对话 ──────────
    def _write_trajectory(self, pred_dir: str, item: dict, params: dict, row: dict) -> None:
        seg = task_output_segment(item["id"])
        d = os.path.join(pred_dir, seg)
        os.makedirs(d, exist_ok=True)
        m = row.get("metrics") or {}
        fail = row.get("fail") or []

        lines = []
        for k, v in sorted(m.items()):
            if not isinstance(v, (int, float)):
                continue
            mark = "FAIL" if k in fail else "ok"
            lines.append(f"  {k:16s} = {v:<10.4f} [{mark}]")
        table = "\n".join(lines) if lines else "  (没有拿到指标)"

        conv = [
            {"step": 1,
             "action": f"按技能参数烘焙 {item['image']}（{item.get('task_type', 'landscape')}）",
             "env_feedback": "参数：" + json.dumps(params, ensure_ascii=False)},
            {"step": 2,
             "action": "渲染 4 个机位并计算判据（阈值见技能文档）",
             "env_feedback": "指标：\n" + table},
            {"step": 3,
             "action": "按阈值判定",
             "env_feedback": (f"hard={row['hard']} soft={row['soft']:.4f}；"
                              + ("全部通过" if not fail else "未通过：" + "、".join(fail)))},
        ]
        if row.get("error"):
            conv.append({"step": 4, "action": "评分失败", "env_feedback": row["error"]})
        with open(os.path.join(d, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(conv, f, ensure_ascii=False, indent=1)

        # 结果字典里补上分析器会读的字段
        row["task_description"] = (
            f"为这张 {item.get('task_type', 'landscape')} 照片挑选烘焙参数，"
            f"要求不出现空洞/拉丝/露空白。图片：{item['image']}")
        row["task_type"] = item.get("task_type", "landscape")
        row["n_turns"] = len(conv)
        row["fail_reason"] = self._fail_reason(m, fail)
        row["prediction"] = f"hard={row['hard']} soft={row['soft']}"

    @staticmethod
    def _fail_reason(metrics: dict, fail: list) -> str:
        """把失败项写成一句人话，带"超了几倍"和这些指标意味着什么。"""
        if not fail:
            return ""
        parts = []
        for k in fail:
            v = metrics.get(k)
            hint = _THRESH_HINT.get(k, k)
            if isinstance(v, (int, float)):
                parts.append(f"{k}={v:.4g}（{hint}）未达标")
            else:
                parts.append(f"{k}（{hint}）未达标")
        return "；".join(parts)

    def get_task_types(self) -> list[str]:
        return ["landscape", "architecture", "people", "indoor"]
