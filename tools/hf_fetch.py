#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""直连下载 HuggingFace 大文件（绕过 hf_hub 的 HEAD 头校验）。

为什么需要它：本机走了系统代理，`resolve/main/...` 的响应**丢了 X-Repo-Commit 头**，
hf_hub 2.4 会因此判定"不是 HuggingFace 端点"并拒绝下载（小文件侥幸过了，5GB 的 UNet 过不去）。
这里直接用 urllib + Range 续传，落到 diffusers 期望的 snapshots/<rev>/<path> 上。

    python tools/hf_fetch.py <url> <dest> [--proxy-off]
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request


def fetch(url: str, dest: str, proxy_off: bool = False) -> int:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"User-Agent": "photoworld/1.0"}
    if have:
        headers["Range"] = "bytes=%d-" % have
    handlers = []
    if proxy_off:
        handlers.append(urllib.request.ProxyHandler({}))
    op = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    with op.open(req, timeout=60) as r:
        total = r.headers.get("Content-Length")
        total = int(total) + have if total else 0
        mode = "ab" if have and r.status == 206 else "wb"
        if mode == "wb":
            have = 0
        got = have
        last = 0
        with open(part, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total and got - last > total * 0.02:
                    last = got
                    sp = got / max(0.1, time.time() - t0) / 1e6
                    print("%.1f%%  %.2f/%.2f GB  %.1f MB/s"
                          % (got * 100.0 / total, got / 1e9, total / 1e9, sp), flush=True)
    if total and os.path.getsize(part) != total:
        print("不完整：%d / %d" % (os.path.getsize(part), total), flush=True)
        return 1
    os.replace(part, dest)
    print("完成 %.1f 分钟 → %s" % ((time.time() - t0) / 60, dest), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(fetch(sys.argv[1], sys.argv[2], "--proxy-off" in sys.argv))
