#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v4 本地重建服务：静态站 + 上传/重建/查询三个接口（纯标准库，不引依赖）。

为什么要它：静态站（`python -m http.server`）只能读，不能算。SfM 必须在本机跑，
所以加一个**只监听 127.0.0.1** 的小服务，页面把照片传进来、这里起子进程跑
`pipeline/v4/build.py`，页面轮询进度，完成后直接加载产出的 GLB。

    python tools/serve_api.py --port 8200 --static .

接口
    POST /api/v4/job                      → {"job": "j<uuid>", "token": "..."}
    POST /api/v4/upload?job=..&token=..&name=..    raw body 即文件内容
    POST /api/v4/build?job=..&token=..    → 起子进程；可重复调用（会先杀旧的）
    GET  /api/v4/status?job=..[&token=..] → {"state":..., "log":..., "qc":..., "glb":...}

安全：只绑定 127.0.0.1；CORS 只放行本机静态站（:8199），不回显任意 Origin；
job 名用 uuid + 一次性 token，避免同秒碰撞和跨站触发；上传大小上限 512MB；
静态文件按白名单暴露，不把 uploads/ / configs/ / logs/ 挂到网上。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
OUT = ROOT / "out" / "v4" / "worlds"
MAX_UPLOAD = 512 * 1024 * 1024
JOB_RE = re.compile(r"^[A-Za-z0-9_-]{4,64}$")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".mp4", ".mov", ".avi", ".mkv"}
ALLOWED_ORIGINS = {"http://127.0.0.1:8199", "http://localhost:8199"}
ALLOW_STATIC_DIRS = ("js/", "css/", "assets/", "worlds/", "out/", "vendor/")
ALLOW_STATIC_FILES = {"index.html", "gpu.html", "traj.html",
                      "favicon-32.png", "icon-192.png", "icon-512.png"}
BUILD_MODULES = ("numpy", "cv2", "scipy", "PIL")

# RLock：_state 会在已持锁的分支里被调用
_lock = threading.RLock()
_jobs: dict[str, dict] = {}
BUILD_PY: str | None = None      # main() 里解析，形如 "py -3.11" 或 "C:/.../python.exe"


def _find_build_python() -> str | None:
    """找一个装了 numpy/cv2/scipy/PIL 的解释器。

    v4.1.0 的坑：start.py 用"零依赖 Python"拉起本服务，本服务又用 sys.executable
    跑 build.py；build.py 顶层 import numpy → 干净环境必现 ModuleNotFoundError，
    前端只看到 rc=1，误报成"未通过质量门"。这里换成**显式挑选** capable 解释器。
    """
    try:
        from start import find_python_with  # tools/ 已在 sys.path，单一事实来源
        found = find_python_with(list(BUILD_MODULES))
        if found:
            return found
    except Exception:
        pass
    candidates: list[str] = []
    py = shutil.which("py")
    if py:
        candidates += [py + " -3.11", py + " -3"]
    for name in ("python", "python3"):
        p = shutil.which(name)
        if p and p not in candidates:
            candidates.append(p)
    code = ("import importlib.util as u,sys;"
            "m=" + repr(list(BUILD_MODULES)) + ";"
            "sys.exit(0 if all(u.find_spec(x) for x in m) else 1)")
    for cmd in candidates:
        argv = cmd.split() if " " in cmd else [cmd]
        try:
            if subprocess.run(argv + ["-c", code], capture_output=True, timeout=8).returncode == 0:
                return cmd
        except Exception:
            continue
    return None


def _cleanup_old_jobs(max_age_hours: float = 24.0) -> int:
    """清掉超过 24 小时的上传/产物任务目录：v4 每建一次任务就留一份图+日志，
    不清理会长期占盘（一次十几张照片就是几十 MB）。只动符合 job 命名的目录。"""
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for base in (UPLOADS, OUT):
        if not base.is_dir():
            continue
        for d in base.iterdir():
            if not d.is_dir() or not JOB_RE.match(d.name):
                continue
            try:
                newest = max((p.stat().st_mtime for p in d.rglob("*") if p.is_file()),
                             default=d.stat().st_mtime)
            except OSError:
                newest = d.stat().st_mtime
            if newest < cutoff:
                if shutil.rmtree(d, ignore_errors=True) is None and not d.exists():
                    removed += 1
                    print("[api] 清理过期任务：" + str(d))
    return removed


def _job_dir(job: str) -> Path:
    d = UPLOADS / job
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state(job: str) -> dict:
    with _lock:
        return _jobs.setdefault(job, {"state": "idle", "started": 0, "proc": None,
                                      "gen": 0, "token": None, "log": "", "exit": None,
                                      "error": None, "logpath": None})


def _tail(path: Path, n: int = 4000) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return data[-n:]


def _run_build(job: str, gen: int) -> None:
    st = _state(job)
    out = OUT / job
    out.mkdir(parents=True, exist_ok=True)
    log = out / ("build.log" if gen <= 1 else "build-%d.log" % gen)
    argv = BUILD_PY.split() if " " in BUILD_PY else [BUILD_PY]  # type: ignore[union-attr]
    cmd = argv + [str(ROOT / "pipeline" / "v4" / "build.py"),
                  "--input", str(_job_dir(job)), "--out", str(out)]
    try:
        with open(log, "w", encoding="utf-8") as f:
            f.write("$ " + " ".join(cmd) + "\n")
            f.flush()
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
            with _lock:
                if st["gen"] != gen:      # 已被新一轮重建取代
                    p.kill()
                    return
                st["proc"] = p
                st["state"] = "running"
                st["logpath"] = str(log)
            rc = p.wait()
        with _lock:
            if st["gen"] == gen:
                st["exit"] = rc
                st["state"] = "done" if rc == 0 else "fail"
                st["proc"] = None
    except Exception as e:                # 线程里任何异常都必须落到状态，不能永久 running
        try:
            with open(log, "a", encoding="utf-8") as f:
                f.write("\n[serve_api] 启动重建失败：%s\n" % e)
        except Exception:
            pass
        with _lock:
            if st["gen"] == gen:
                st["exit"] = -1
                st["state"] = "fail"
                st["error"] = str(e)
                st["proc"] = None


class Handler(SimpleHTTPRequestHandler):
    server_version = "photoworld-v4/1.1"

    # ---------- 工具 ----------
    def _origin_ok(self) -> bool:
        o = self.headers.get("Origin")
        return (not o) or o in ALLOWED_ORIGINS

    def _cors(self):
        # 页面在 :8199，接口在 :8200 → 跨端口，必须带 CORS 头，否则浏览器读不到响应。
        # v4.1.1：只回显白名单 Origin，不再用 "*" 给任意网页开本机重建入口。
        o = self.headers.get("Origin")
        if o in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", o)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _qs(self) -> dict:
        q = urllib.parse.urlparse(self.path).query
        return {k: v[0] for k, v in urllib.parse.parse_qs(q).items()}

    def log_message(self, fmt, *args):
        print("[api] " + (fmt % args))

    # 静态文件：显式白名单，绝不把 uploads/out/logs/configs 整个根目录暴露出去
    def translate_path(self, path: str) -> str:
        rel = urllib.parse.unquote(urllib.parse.urlparse(path).path).lstrip("/").replace("\\", "/")
        if not rel or rel.endswith("/"):
            rel += "index.html"
        root = ROOT.resolve()
        try:
            full = (root / rel).resolve()
            if os.path.commonpath([str(root), str(full)]) != str(root):
                return str(root / "index.html")
        except (OSError, ValueError):
            return str(root / "index.html")
        allowed = rel in ALLOW_STATIC_FILES or any(rel.startswith(d) for d in ALLOW_STATIC_DIRS)
        return str(full) if allowed else str(root / "index.html")

    # ---------- GET ----------
    def do_GET(self):
        if self.path.startswith("/api/v4/status"):
            q = self._qs()
            job = q.get("job", "")
            if not JOB_RE.match(job):
                return self._json({"error": "bad job"}, 400)
            st = _state(job)
            # token 可选：带上了就必须匹配；本机页面会带上，curl 排查可以不带
            tok = q.get("token")
            if tok and tok != st.get("token"):
                return self._json({"error": "bad token"}, 403)
            out = OUT / job
            qc = None
            p = out / "qc.json"
            if p.exists():
                try:
                    qc = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    qc = None
            glb = out / "world-v4.glb"
            log_path = Path(st["logpath"]) if st.get("logpath") else (out / "build.log")
            return self._json({
                "job": job,
                "state": st["state"],
                "exit": st["exit"],
                "error": st.get("error"),
                "log": _tail(log_path),
                "qc": qc,
                "glb": (str(glb.relative_to(ROOT)).replace("\\", "/")
                        if glb.exists() else None),
            })
        return super().do_GET()

    # ---------- POST ----------
    def do_POST(self):
        if not self.path.startswith("/api/v4/"):
            return self._json({"error": "not found"}, 404)
        if not self._origin_ok():
            return self._json({"error": "origin not allowed"}, 403)
        q = self._qs()
        if self.path.startswith("/api/v4/job"):
            job = "j" + uuid.uuid4().hex[:16]
            token = secrets.token_urlsafe(18)
            with _lock:
                st = _state(job)
                st["token"] = token
            _job_dir(job)
            return self._json({"job": job, "token": token})

        job = q.get("job", "")
        if not JOB_RE.match(job):
            return self._json({"error": "bad job"}, 400)
        st = _state(job)
        if q.get("token") != st.get("token"):
            return self._json({"error": "bad token"}, 403)

        if self.path.startswith("/api/v4/upload"):
            name = os.path.basename(q.get("name", "file.bin"))
            name = re.sub(r"[^A-Za-z0-9._-]", "_", name) or "file.bin"
            if os.path.splitext(name)[1].lower() not in IMG_EXT:
                return self._json({"error": "unsupported file type"}, 400)
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_UPLOAD:
                return self._json({"error": "bad size"}, 400)
            dest = _job_dir(job) / name
            left = n
            with open(dest, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            if left:
                return self._json({"error": "upload truncated"}, 400)
            return self._json({"ok": True, "size": n, "name": name})

        if self.path.startswith("/api/v4/build"):
            if not BUILD_PY:
                msg = ("找不到装了 %s 的 Python：请安装依赖后重启服务"
                       % ", ".join(BUILD_MODULES))
                with _lock:
                    st["state"] = "fail"
                    st["error"] = msg
                return self._json({"error": msg}, 400)
            files = [p for p in _job_dir(job).iterdir() if p.is_file()]
            if not files:
                return self._json({"error": "还没有上传任何文件"}, 400)
            with _lock:
                old = st.get("proc")
                if old and old.poll() is None:
                    old.kill()
                    old.wait(timeout=3)
                st["gen"] = int(st.get("gen") or 0) + 1
                gen = st["gen"]
                st["state"] = "running"
                st["exit"] = None
                st["error"] = None
                st["logpath"] = None
            threading.Thread(target=_run_build, args=(job, gen), daemon=True).start()
            return self._json({"ok": True, "files": len(files)})

        return self._json({"error": "not found"}, 404)


def main() -> int:
    global BUILD_PY
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--static", default=".")
    args = ap.parse_args()
    BUILD_PY = _find_build_python()
    if BUILD_PY:
        print("v4 构建解释器：" + BUILD_PY)
    else:
        print("警告：没找到装了 %s 的 Python，上传面板会给出明确错误。"
              % ", ".join(BUILD_MODULES), file=sys.stderr)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    n = _cleanup_old_jobs()
    if n:
        print("已清理 %d 个过期重建任务目录（>24h）" % n)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("v4 重建服务：http://127.0.0.1:%d  （静态目录 %s）" % (args.port, Path(args.static).resolve()))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
