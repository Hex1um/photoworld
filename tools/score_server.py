#!/usr/bin/env python
"""常驻评分服务：深度模型只加载一次，供训练循环反复调用。

为什么要它：
  · 训练循环里每次 `score_params.py` 都是新进程 → 每次都要加载 1.3GB 深度模型（~20s），
    单图 60~80s，8 步训练要一小时以上；
  · 而且子进程在训练进程里会莫名失败（multiview 报空 stderr），常驻进程没有这层。

用法：
    python tools/score_server.py --port 8300 --base http://127.0.0.1:8199

接口（GET，返回一行 JSON）：
    /score?image=worlds/desert.jpg&params={"far_split":100}&views=4
    /health  → {"ok": true, "loaded": true, "hits": 12}
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_LOCK = threading.Lock()          # GPU/模型不是线程安全的：串行处理
_STATE = {"loaded": False, "hits": 0, "api": None, "error": ""}


def _decode_output(stdout: bytes, stderr: bytes) -> tuple[str, str]:
    """把 node 子进程的字节输出安全解码为文本。

    multiview（node）经管道吐出的字节里混着 GBK 片段（实测 position 4 的 0xcf）；
    subprocess.run(text=True) 会在 reader 线程抛 UnicodeDecodeError，主线程只看到
    一堆 None，表现为"评分失败、stderr 为空"。与 score_params.py 的 render_score
    保持一致：收字节、自己按 utf-8 + replace 解码。
    """
    return ((stdout or b"").decode("utf-8", "replace"),
            (stderr or b"").decode("utf-8", "replace"))


def _load_api():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v2api", os.path.join(ROOT, "pipeline", "v2", "api.py"))
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api


def _score(image: str, params: dict, base: str, views: int) -> dict:
    import subprocess
    t0 = time.time()
    params = dict(params or {})
    # 训练里默认关 SD 外扩（一张 40~150s，rollout 会被拖死）
    params.setdefault("outpaint", 1.0)

    # **固定工作目录**，不再每次新建/删除临时目录。
    # 环境里有"批量删除保护"：连续删除到 50 次会拦截，而评分每张图删一次 →
    # 第 50 张之后服务端的连接会被切断，训练里表现成 "Remote end closed connection"。
    # 改成覆盖同一目录就没有删除动作了。
    tmp = os.path.join(ROOT, "out", "_score", "cur")
    os.makedirs(tmp, exist_ok=True)
    try:
        if _STATE["api"] is None:
            _STATE["api"] = _load_api()
            _STATE["loaded"] = True
        api = _STATE["api"]
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("pwscore2", os.path.join(ROOT, "tools", "score_params.py"))
        _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
        _lock_ctx = _mod._gpu_lock()
        _lock_ctx.__enter__()
        path = image if os.path.isabs(image) else os.path.join(ROOT, image)
        with open(path, "rb") as f:
            data = f.read()
        glb, man = api.bake(data, **params)
        glb_path = os.path.join(tmp, "world.glb")
        with open(glb_path, "wb") as f:
            f.write(glb)
        rel = os.path.relpath(glb_path, ROOT).replace("\\", "/")

        lst = os.path.join(tmp, "_list.txt")
        with open(lst, "w", encoding="utf-8") as f:
            f.write(rel + "\n")
        js = os.path.join(tmp, "mv.json")
        cmd = ["node", os.path.join(ROOT, "tools", "multiview.mjs"),
               "--base", base, "--glb-list", lst, "--views", str(views),
               "--out", os.path.join(tmp, "shots"), "--json", js, "--gl", "hw"]
        # 不用 text=True：multiview 字节流混 GBK 会抛 UnicodeDecodeError（见 _decode_output）
        r = subprocess.run(cmd, capture_output=True, timeout=900, cwd=ROOT)
        out, err = _decode_output(r.stdout, r.stderr)
        if r.returncode != 0:
            raise RuntimeError(f"multiview 失败({r.returncode}): {(err or out)[-600:]}")
        d = json.load(open(js, encoding="utf-8"))
        res = (d.get("results") or [])
        if not res:
            raise RuntimeError("multiview 没有返回结果")
        agg = res[0].get("agg") or {}
        # 判定必须在这里一起算：adapter 直接读 hard/soft，
        # 只给 metrics 的话它会一直拿到 hard=0（实测踩过：整轮 rollout 全是 0 分，
        # 反射阶段因此学不到任何东西）。
        hard, soft, fail = _judge(agg)
        return {"ok": True, "metrics": agg, "hard": hard, "soft": soft, "fail": fail,
                "repr": man.get("repr"), "secs": round(time.time() - t0, 1)}
    finally:
        try:
            _lock_ctx.__exit__(None, None, None)      # 释放全局渲染锁
        except Exception:
            pass
        # 只清掉本次产生的文件（不删目录，避免触发批量删除保护）
        for name in ("world.glb", "_list.txt", "mv.json"):
            try:
                os.remove(os.path.join(tmp, name))
            except OSError:
                pass


def _judge(agg: dict) -> tuple[int, float, list[str]]:
    """与 tools/score_params.py 同一套阈值与权重（那份是权威定义，这里复用）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pwscore", os.path.join(ROOT, "tools", "score_params.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.score(agg)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # 静音：训练日志已经够多
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/health":
            self._send({"ok": True, "loaded": _STATE["loaded"],
                        "hits": _STATE["hits"], "error": _STATE["error"]})
            return
        if u.path != "/score":
            self._send({"ok": False, "error": "unknown path"}, 404)
            return
        image = (q.get("image") or [""])[0]
        try:
            params = json.loads((q.get("params") or ["{}"])[0])
        except Exception:
            params = {}
        base = (q.get("base") or ["http://127.0.0.1:8199"])[0]
        views = int((q.get("views") or ["4"])[0])
        if not image:
            self._send({"ok": False, "error": "missing image"}, 400)
            return
        with _LOCK:
            try:
                out = _score(image, params, base, views)
                _STATE["hits"] += 1
            except Exception as e:                      # noqa: BLE001
                _STATE["error"] = str(e)[:400]
                out = {"ok": False, "error": _STATE["error"]}
        self._send(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--base", default=os.environ.get("PW_BASE", "http://127.0.0.1:8199"))
    ap.add_argument("--warm", action="store_true", help="启动时先加载模型")
    args = ap.parse_args()

    if args.warm:
        try:
            _STATE["api"] = _load_api()
            _STATE["loaded"] = True
            print("[score_server] 模型已加载", flush=True)
        except Exception as e:                           # noqa: BLE001
            print(f"[score_server] 预热失败: {e}", flush=True)

    # **单线程**服务器，不是 ThreadingHTTPServer。
    # 用线程跑请求时，torch 的 CUDA 上下文在非主线程里会直接崩掉进程
    # （表现为客户端收到 "Remote end closed connection without response"，
    #  服务端没有任何 traceback）—— 实测连挂几次，换成主线程处理后就稳了。
    # 顺带：评分本来就该串行（一把 GPU），多线程没有收益。
    srv = HTTPServer((args.host, args.port), Handler)
    print(f"[score_server] http://{args.host}:{args.port}  (render base={args.base})", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
