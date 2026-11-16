#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CDP 最小客户端自测：用本地假 WebSocket 服务验证握手/帧/调用/事件/截图。

    python tools/test_cdp.py
不需要 Edge，也不需要第三方包。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdp import CDPError, CDP  # noqa: E402

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _recv_exact(c, n):
    buf = b""
    while len(buf) < n:
        part = c.recv(n - len(buf))
        if not part:
            raise EOFError
        buf += part
    return buf


def _read_msg(c):
    b1, b2 = _recv_exact(c, 2)
    n = b2 & 0x7F
    if n == 126:
        n = struct.unpack("!H", _recv_exact(c, 2))[0]
    elif n == 127:
        n = struct.unpack("!Q", _recv_exact(c, 8))[0]
    mask = _recv_exact(c, 4) if b2 & 0x80 else None
    data = _recv_exact(c, n)
    if mask:
        data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    return b1 & 0x0F, data


def _send_msg(c, text):
    payload = text.encode("utf-8")
    n = len(payload)
    head = bytearray([0x81])
    if n < 126:
        head.append(n)
    elif n < 65536:
        head.append(126)
        head += struct.pack("!H", n)
    else:
        head.append(127)
        head += struct.pack("!Q", n)
    c.sendall(bytes(head) + payload)


def fake_server(ln, errors):
    try:
        conn, _ = ln.accept()
        with conn:
            req = b""
            while b"\r\n\r\n" not in req:
                req += conn.recv(4096)
            line = [x for x in req.split(b"\r\n") if x.lower().startswith(b"sec-websocket-key:")][0]
            key = line.split(b":", 1)[1].strip()
            accept = base64.b64encode(hashlib.sha1(key + GUID.encode()).digest()).decode()
            conn.sendall(("HTTP/1.1 101 Switching Protocols\r\n"
                          "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                          "Sec-WebSocket-Accept: %s\r\n\r\n" % accept).encode())
            seen = []
            answers = {
                "Page.enable": {},
                "Runtime.evaluate": {"result": {"type": "number", "value": 2}},
                "Page.captureScreenshot": {"data": base64.b64encode(b"PNG-BYTES").decode()},
            }
            for i in range(3):
                op, data = _read_msg(conn)
                if op != 1:
                    raise AssertionError("客户端帧类型应为文本: %d" % op)
                msg = json.loads(data.decode("utf-8"))
                seen.append(msg["method"])
                if i == 1:
                    _send_msg(conn, json.dumps({"method": "Runtime.consoleAPICalled",
                                                "params": {"type": "error",
                                                           "args": [{"value": "boom"}]}}))
                _send_msg(conn, json.dumps({"id": msg["id"], "result": answers[msg["method"]]}))
            if seen != ["Page.enable", "Runtime.evaluate", "Page.captureScreenshot"]:
                raise AssertionError("调用顺序不符: %s" % seen)
    except Exception as e:
        errors.append(e)


def main() -> int:
    ln = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ln.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ln.bind(("127.0.0.1", 0))
    ln.listen(1)
    port = ln.getsockname()[1]
    errors = []
    th = threading.Thread(target=fake_server, args=(ln, errors), daemon=True)
    th.start()
    events = []
    try:
        c = CDP("ws://127.0.0.1:%d/devtools/page/test" % port, timeout=10)
        c.on_event(events.append)
        assert c.send("Page.enable") == {}
        assert c.ev("1+1") == 2
        with tempfile.TemporaryDirectory(prefix="pwcdptest-") as td:
            p = os.path.join(td, "shot.png")
            c.screenshot(p)
            with open(p, "rb") as f:
                assert f.read() == b"PNG-BYTES"
        c.close()
        th.join(timeout=5)
        assert not errors, errors
        assert len(events) == 1 and events[0]["method"] == "Runtime.consoleAPICalled"
        print("✓ CDP 握手 / 文本帧 / 调用匹配 / 事件分发 / 截图落盘 全通过")
        return 0
    except Exception as e:
        print("CDP 自测失败：%s" % e, file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        ln.close()


if __name__ == "__main__":
    sys.exit(main())
