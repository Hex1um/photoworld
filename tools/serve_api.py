#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v4 本地重建服务：静态站 + 上传/重建/查询三个接口（纯标准库，不引依赖）。

为什么要它：静态站（`python -m http.server`）只能读，不能算。SfM 必须在本机跑，
所以加一个**只监听 127.0.0.1** 的小服务，页面把照片传进来、这里起子进程跑
`pipeline/v4/build.py`，页面轮询进度，完成后直接加载产出的 GLB。

    python tools/serve_api.py --port 8200 --static .

接口
    POST /api/v4/job                      → {"job": "j20261119-123456"}
    POST /api/v4/upload?job=..&name=..    raw body 即文件内容
    POST /api/v4/build?job=..             → 起子进程；可重复调用（会先杀旧的）
    GET  /api/v4/status?job=..            → {"state":..., "log":..., "qc":..., "glb":...}

安全：只绑定 127.0.0.1；job 名只允许 [A-Za-z0-9_-]；上传大小上限 512MB；
不提供任何删除/执行任意命令的入口。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
OUT = ROOT / "out" / "v4" / "worlds"
MAX_UPLOAD = 512 * 1024 * 1024
JOB_RE = re.compile(r"^[A-Za-z0-9_-]{4,64}$")

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _job_dir(job: str) -> Path:
    d = UPLOADS / job
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state(job: str) -> dict:
    st = _jobs.setdefault(job, {"state": "idle", "started": 0, "proc": None,
                                "log": "", "exit": None})
    return st


def _tail(path: Path, n: int = 4000) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return data[-n:]


def _run_build(job: str) -> None:
    st = _state(job)
    out = OUT / job
    log = out / "build.log"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "pipeline" / "v4" / "build.py"),
           "--input", str(_job_dir(job)), "--out", str(out)]
    with open(log, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        st["proc"] = p
        st["state"] = "running"
        rc = p.wait()
    st["exit"] = rc
    st["state"] = "done" if rc == 0 else "fail"


class Handler(SimpleHTTPRequestHandler):
    server_version = "photoworld-v4/1.0"

    # ---------- 工具 ----------
    def _cors(self):
        # 页面在 :8199，接口在 :8200 → 跨端口，必须带 CORS 头，否则浏览器读不到响应
        # （请求其实到了，但 fetch 会直接 reject，表现成"服务没起"）
        self.send_header("Access-Control-Allow-Origin", "*")
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

    # ---------- GET ----------
    def do_GET(self):
        if self.path.startswith("/api/v4/status"):
            q = self._qs()
            job = q.get("job", "")
            if not JOB_RE.match(job):
                return self._json({"error": "bad job"}, 400)
            st = _state(job)
            out = OUT / job
            qc = None
            p = out / "qc.json"
            if p.exists():
                try:
                    qc = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    qc = None
            glb = out / "world-v4.glb"
            return self._json({
                "job": job,
                "state": st["state"],
                "exit": st["exit"],
                "log": _tail(out / "build.log"),
                "qc": qc,
                "glb": (str(glb.relative_to(ROOT)).replace("\\", "/")
                        if glb.exists() else None),
            })
        return super().do_GET()

    # ---------- POST ----------
    def do_POST(self):
        if not self.path.startswith("/api/v4/"):
            return self._json({"error": "not found"}, 404)
        q = self._qs()
        if self.path.startswith("/api/v4/job"):
            job = "j" + time.strftime("%Y%m%d-%H%M%S")
            _job_dir(job)
            _state(job)
            return self._json({"job": job})

        job = q.get("job", "")
        if not JOB_RE.match(job):
            return self._json({"error": "bad job"}, 400)

        if self.path.startswith("/api/v4/upload"):
            name = os.path.basename(q.get("name", "file.bin"))
            name = re.sub(r"[^A-Za-z0-9._-]", "_", name) or "file.bin"
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
            return self._json({"ok": True, "size": n, "name": name})

        if self.path.startswith("/api/v4/build"):
            st = _state(job)
            if st["state"] == "running" and st["proc"] and st["proc"].poll() is None:
                st["proc"].kill()
                time.sleep(0.2)
            files = [p for p in _job_dir(job).iterdir() if p.is_file()]
            if not files:
                return self._json({"error": "还没有上传任何文件"}, 400)
            st["state"] = "running"
            st["exit"] = None
            threading.Thread(target=_run_build, args=(job,), daemon=True).start()
            return self._json({"ok": True, "files": len(files)})

        return self._json({"error": "not found"}, 404)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--static", default=".")
    args = ap.parse_args()
    os.chdir(args.static)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("v4 重建服务：http://127.0.0.1:%d  （静态目录 %s）" % (args.port, Path(args.static).resolve()))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
