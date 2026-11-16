#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""极简 CDP 客户端：只依赖标准库，供 tools/ 下的无头浏览器脚本共用。

原来每个 .mjs 都复制一份「 spawn Edge → 轮询 /json/list → WebSocket →
Runtime.evaluate → taskkill 」的样板（8 个文件、约 2 万行重复逻辑）。
本模块把公共链路收成一处，Python 脚本只写自己的业务判断。

设计取舍：
- 不引入 websocket-client / playwright：本机 Python 环境常常是裸解释器，
  pip 装包会拖慢首次运行；CDP 只需要文本帧 + 大 base64 截图，手写帧足够。
- 支持分片帧、ping/pong、64 位长度，截图（几百 KB）不会截断。
- 只连接本机调试端口，显式绕过环境变量里的 HTTP 代理。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

DEFAULT_EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class CDPError(RuntimeError):
    pass


class CDP:
    """与一个 Chrome/Edge 页面建立 CDP WebSocket 会话。"""

    def __init__(self, ws_url: str, timeout: float = 30.0):
        u = urllib.parse.urlparse(ws_url)
        if u.scheme not in ("ws", "wss"):
            raise CDPError("不支持的 WebSocket 地址: %s" % ws_url)
        host = u.hostname or "127.0.0.1"
        port = u.port or (443 if u.scheme == "wss" else 80)
        path = u.path or "/"
        if u.query:
            path += "?" + u.query
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.buf = b""
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            "GET %s HTTP/1.1\r\n"
            "Host: %s:%d\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Key: %s\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ) % (path, host, port, key)
        self.sock.sendall(req.encode("ascii"))
        while b"\r\n\r\n" not in self.buf:
            self._fill()
        head, self.buf = self.buf.split(b"\r\n\r\n", 1)
        status = head.split(b"\r\n", 1)[0].decode("ascii", "replace")
        if " 101" not in status:
            raise CDPError("WebSocket 握手失败: %s" % status)
        self._id = 0
        self._pending: dict[int, tuple] = {}
        self._listeners = []
        self._frag = bytearray()
        self._frag_opcode = None
        self.closed = False

    # ---------- 底层帧 ----------
    def _fill(self):
        chunk = self.sock.recv(65536)
        if not chunk:
            raise CDPError("WebSocket 连接已关闭")
        self.buf += chunk

    def _exact(self, n: int) -> bytes:
        while len(self.buf) < n:
            self._fill()
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _send_frame(self, payload: bytes, opcode: int = 1):
        mask = os.urandom(4)
        n = len(payload)
        head = bytearray([0x80 | opcode])
        if n < 126:
            head.append(0x80 | n)
        elif n < 65536:
            head.append(0x80 | 126)
            head += struct.pack("!H", n)
        else:
            head.append(0x80 | 127)
            head += struct.pack("!Q", n)
        head += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes(head) + masked)

    def _read_frame(self):
        b1, b2 = self._exact(2)
        fin = bool(b1 & 0x80)
        opcode = b1 & 0x0F
        masked = bool(b2 & 0x80)
        length = b2 & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._exact(8))[0]
        mask = self._exact(4) if masked else None
        data = self._exact(length)
        if mask:
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        if opcode == 9:  # ping -> pong
            self._send_frame(data, 10)
            return
        if opcode == 10:
            return
        if opcode == 8:
            self.closed = True
            raise CDPError("WebSocket 收到 close 帧")
        if opcode in (1, 2):
            self._frag_opcode = opcode
            self._frag = bytearray(data)
        elif opcode == 0 and self._frag_opcode is not None:
            self._frag += data
        else:
            return
        is_text = self._frag_opcode == 1
        if not fin:
            return
        raw = bytes(self._frag)
        self._frag = bytearray()
        self._frag_opcode = None
        if is_text and raw:
            try:
                msg = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                return
            self._dispatch(msg)

    def _dispatch(self, msg: dict):
        if "method" in msg:
            for fn in list(self._listeners):
                try:
                    fn(msg)
                except Exception:
                    pass
        mid = msg.get("id")
        if mid in self._pending:
            # 不要在这里 pop：send() 还在轮询同一个 dict 项，pop 会让它 KeyError。
            self._pending[mid].append(msg)

    # ---------- 对外 API ----------
    def on_event(self, fn):
        self._listeners.append(fn)

    def send(self, method: str, params: dict | None = None):
        self._id += 1
        mid = self._id
        payload = json.dumps(
            {"id": mid, "method": method, "params": params or {}},
            separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        self._send_frame(payload)
        self._pending[mid] = []
        while not self._pending[mid]:
            self._read_frame()
        msg = self._pending.pop(mid)[0]
        if msg.get("error"):
            raise CDPError(method + " 失败: " + json.dumps(msg["error"], ensure_ascii=False))
        return msg.get("result", {})

    def eval_raw(self, expr: str, await_promise: bool = True, timeout_ms: int | None = None):
        params = {
            "expression": expr,
            "returnByValue": True,
            "awaitPromise": bool(await_promise),
        }
        if timeout_ms is not None:
            params["timeout"] = timeout_ms
        return self.send("Runtime.evaluate", params)

    def ev(self, expr: str, await_promise: bool = True, timeout_ms: int | None = None):
        r = self.eval_raw(expr, await_promise, timeout_ms)
        if r.get("exceptionDetails"):
            d = r["exceptionDetails"]
            desc = (d.get("exception") or {}).get("description") or d.get("text") or "unknown"
            raise CDPError("表达式抛错: " + str(desc))
        return (r.get("result") or {}).get("value")

    def screenshot(self, path: str, fmt: str = "png", quality: int | None = None):
        params = {"format": fmt}
        if quality is not None:
            params["quality"] = quality
        data = self.send("Page.captureScreenshot", params).get("data", "")
        if not data:
            raise CDPError("截图返回为空")
        if path:
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "wb") as f:
                f.write(base64.b64decode(data))
        return base64.b64decode(data)

    def close(self):
        if not self.closed:
            try:
                self._send_frame(b"", 8)
            except Exception:
                pass
            self.closed = True
        try:
            self.sock.close()
        except Exception:
            pass


def _json_opener():
    # 会话里可能带 HTTP_PROXY；CDP 端口永远只在本机，不能走代理。
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def ws_url(port: int, retries: int = 120, interval: float = 0.25) -> str:
    for _ in range(retries):
        try:
            with _json_opener().open("http://127.0.0.1:%d/json/list" % port, timeout=2) as r:
                items = json.loads(r.read().decode("utf-8", "replace"))
            for it in items:
                if it.get("type") == "page" and it.get("webSocketDebuggerUrl"):
                    return it["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(interval)
    raise CDPError("CDP 未就绪（端口 %d）" % port)


class Browser:
    """启动一个临时 profile 的 Edge/Chrome，并在需要时建立 CDP 会话。"""

    def __init__(self, proc, port, profile, owns_profile=True, edge=DEFAULT_EDGE):
        self.proc = proc
        self.port = port
        self.profile = profile
        self.owns_profile = owns_profile
        self.edge = edge
        self.cdp: CDP | None = None

    def connect(self, retries: int = 120) -> CDP:
        self.cdp = CDP(ws_url(self.port, retries=retries))
        return self.cdp

    def close(self):
        if self.cdp:
            self.cdp.close()
            self.cdp = None
        if self.proc and self.proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                self.proc.terminate()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.kill()
            except Exception:
                pass
        if self.owns_profile and self.profile:
            shutil.rmtree(self.profile, ignore_errors=True)


def _gpu_flags(gpu: str) -> list[str]:
    if gpu == "swiftshader":
        return ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
    if gpu == "d3d11":
        return ["--use-gl=angle", "--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"]
    if gpu == "hw":
        return ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU",
                "--force_high_performance_gpu", "--enable-gpu-rasterization"]
    return []


def launch(prefix="pwcdp-", port=None, w=1280, h=800, edge=None, headless=True,
           gpu="swiftshader", extra=None, url="about:blank", no_cache=False,
           no_proxy=False, profile=None, retries=120) -> Browser:
    edge = edge or DEFAULT_EDGE
    port = port or random.randint(9300, 9950)
    owns = profile is None
    profile = profile or tempfile.mkdtemp(prefix=prefix)
    flags = []
    if headless:
        flags.append("--headless=new")
    flags += ["--remote-debugging-port=%d" % port, "--user-data-dir=%s" % profile]
    if no_cache:
        flags += ["--disk-cache-dir=/dev/null", "--disk-cache-size=1",
                  "--disable-application-cache"]
    if no_proxy:
        flags.append("--no-proxy-server")
    flags += _gpu_flags(gpu)
    flags += ["--hide-scrollbars", "--no-first-run", "--no-default-browser-check",
              "--window-size=%dx%d" % (w, h)]
    if extra:
        flags += list(extra)
    flags.append(url)
    try:
        proc = subprocess.Popen([edge] + flags, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except Exception as e:
        if owns:
            shutil.rmtree(profile, ignore_errors=True)
        raise CDPError("启动浏览器失败: %s" % e)
    b = Browser(proc, port, profile, owns, edge)
    try:
        b.connect(retries=retries)
    except Exception:
        b.close()
        raise
    return b


def enable_common(cdp: CDP, w: int, h: int, metrics=True):
    cdp.send("Page.enable")
    cdp.send("Runtime.enable")
    if metrics:
        cdp.send("Emulation.setDeviceMetricsOverride",
                 {"width": w, "height": h, "deviceScaleFactor": 1, "mobile": False})


def wait_ready(cdp: CDP, timeout_s: int, interval: float = 0.5,
               keep_err=False) -> tuple[dict, int]:
    """轮询 body.dataset.ready，返回 (state, 耗时毫秒)。"""
    expr = ("JSON.stringify(Object.assign({}, document.body && document.body.dataset, "
            "{ errs: (window.__errs || []).join(';') }))")
    t0 = time.monotonic()
    state = {}
    while (time.monotonic() - t0) * 1000 < timeout_s * 1000:
        try:
            raw = cdp.ev(expr)
            state = json.loads(raw or "{}")
        except Exception:
            state = {}
        if state.get("ready") in ("1", 0, 1):
            break
        time.sleep(interval)
    ms = int((time.monotonic() - t0) * 1000)
    if not keep_err:
        state.pop("errs", None)
    return state, ms


def args_dict(argv=None) -> dict[str, str]:
    """复刻原 .mjs 的 `--key value` 解析（下一个 token 无条件当值）。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    out: dict[str, str] = {}
    for i, v in enumerate(argv):
        if v.startswith("--") and i + 1 < len(argv):
            out[v[2:]] = argv[i + 1]
    return out


def fmt_ms(ms: int) -> str:
    return "%.1fs" % (ms / 1000.0)
