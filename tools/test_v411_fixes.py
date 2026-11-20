#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v4.1.1 回归：修掉的两个数据错误 + 上传链路的假失败/竞态/越权。

纯标准库 + numpy；服务集成测试会临时起一个 127.0.0.1 上的 serve_api，跑完即关。

    python tools/test_v411_fixes.py
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

PASS = 0


def ok(label: str) -> None:
    global PASS
    PASS += 1
    print("PASS  %s" % label)


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_fuse_median_height():
    """回归：中位高度曾被二次除以点数，60 点格子高度 10 → 0.167。"""
    import numpy as np
    from v4 import fuse

    rng = np.random.default_rng(0)
    a = np.column_stack([rng.random(60) * 0.1, np.full(60, 10.0), rng.random(60) * 0.1])
    b = np.column_stack([0.8 + rng.random(60) * 0.1, np.full(60, 1.0),
                         0.8 + rng.random(60) * 0.1])
    pts = np.vstack([a, b]).astype(np.float32)
    w = fuse.to_world(pts, np.ones((len(pts), 3), np.float32), cells=4, smooth=0)
    hmax, hmin = float(w.pos[:, 1].max()), float(w.pos[:, 1].min())
    assert 8.0 < hmax < 12.0, "高台高度被缩放：%.3f（应为 ~10）" % hmax
    assert 0.5 < hmin < 1.5, "低台高度被缩放：%.3f（应为 ~1）" % hmin
    ok("v4 高度场：中位高度不再除以点数（10/1 输入 → %.2f/%.2f）" % (hmax, hmin))


def test_fuse_normal_axes():
    src = read("pipeline/v4/fuse.py")
    assert "gx_, gz_ = np.gradient(grid_y)" in src, "法线梯度 axes 未修正"
    assert "-gx_, np.ones_like(grid_y) * (span / cells), -gz_" in src, "法线 x/z 分量仍互换"
    ok("v4 高度场：法线 x/z 轴序修正")


def test_tsdf_blackface():
    src = read("pipeline/v4/tsdf.py")
    assert "distance_transform_edt" in src, "TSDF 空体素仍未取最近命中体素颜色"
    assert "np.maximum(hn[:, None], 1.0)" in src, "黑面除数仍是 1e-6"
    ok("TSDF：等值面空体素不再取到纯黑")


def test_serve_api_static_source():
    src = read("tools/serve_api.py")
    assert 'Access-Control-Allow-Origin", "*"' not in src, "CORS 仍是通配符"
    assert "ALLOWED_ORIGINS" in src and "uuid.uuid4" in src, "job/Origin 未加固"
    assert "_find_build_python" in src and "BUILD_MODULES" in src, "构建解释器未自检"
    ok("serve_api 源码：CORS 白名单 + uuid job + 解释器自检")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _http(url: str, method: str = "GET", origin: str | None = None, data=None):
    h = {}
    if origin:
        h["Origin"] = origin
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, dict(r.headers), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode("utf-8", "replace")


def test_serve_api_integration():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "serve_api.py"),
         "--port", str(port), "--static", str(ROOT)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = "http://127.0.0.1:%d" % port
    try:
        up = False
        for _ in range(40):
            try:
                if _http(base + "/index.html")[0] == 200:
                    up = True
                    break
            except Exception:
                pass
            time.sleep(0.25)
        assert up, "serve_api 没起来"

        # 1) CORS 不回显任意 Origin（CORS 头只加在 API 响应上，页面本身同源）
        api_probe = base + "/api/v4/status?job=probe"
        _, hdr, _ = _http(api_probe, origin="http://evil.example")
        assert not hdr.get("Access-Control-Allow-Origin"), "对任意 Origin 回显了 CORS 头"
        _, hdr, _ = _http(api_probe, origin="http://127.0.0.1:8199")
        assert hdr.get("Access-Control-Allow-Origin") == "http://127.0.0.1:8199"
        ok("serve_api 集成：CORS 只放行本机 8199")

        # 2) 静态白名单：configs / uploads 不应被读出来
        _, _, body = _http(base + "/configs/photoworld.yaml")
        assert "llm_target" not in body, "configs/ 被静态服务暴露"
        _, _, body = _http(base + "/uploads/")
        assert "llm_target" not in body
        ok("serve_api 集成：静态目录白名单生效")

        # 3) job token：猜错 token 不能触发重建，且不会把状态卡在 running
        code, _, body = _http(base + "/api/v4/job", method="POST", data=b"")
        assert code == 200, (code, body)
        j = json.loads(body)
        assert re.match(r"^j[0-9a-f]{16}$", j["job"]), j["job"]
        assert j.get("token")
        q = "?job=" + j["job"] + "&token="
        code, _, body = _http(base + "/api/v4/build" + q + "bad", method="POST", data=b"")
        assert code == 403, (code, body)
        _, _, st = _http(base + "/api/v4/status?job=" + j["job"])
        assert json.loads(st)["state"] == "idle", "拒绝请求后状态被误置为 running"
        code, _, body = _http(base + "/api/v4/build" + q + j["token"], method="POST", data=b"")
        assert code == 400 and "还没有上传" in body, (code, body)
        ok("serve_api 集成：token 校验 + 拒绝后不卡 running")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_start_stop_v4():
    src = read("tools/start.py")
    assert "V4_PORT = 8200" in src
    assert '"v4": v4_pid' in src, "--stop 不记录 v4 PID"
    assert 'for key in ("static", "api", "v4")' in src
    assert '("v4", V4_PORT)' in src
    assert "find_python_with([\"numpy\", \"cv2\", \"scipy\", \"PIL\"])" in src
    ok("start.py：v4 服务纳入生命周期 + 用 capable Python 启动")


def test_frontend():
    src = read("js/v4upload.js")
    assert ".replace('http://', 'http://')" not in src, "https 页面仍会拼错 API 地址"
    assert "setBar((b) => b);" not in src, "进度条仍会写成 NaN%"
    assert "setBar(78)" in src and "POLL_TIMEOUT_MS" in src
    assert "auth('/api/v4/status')" in src and "auth('/api/v4/build')" in src
    assert "s.error ?" in src and "退出码" in src, "失败原因仍混为一谈"
    ok("前端：API 地址/进度条/token/超时/失败分类均已修")


def test_version_docs():
    # 版本号会随后续版本继续升；本测试只锁"v4.1.1 的修复没有被回退"，
    # 并要求 index.html 缓存串与 version.json 始终同源。
    ver = json.loads(read("version.json"))["version"]
    assert tuple(int(x) for x in ver.split(".")) >= (4, 1, 1), ver
    idx = read("index.html")
    vs = set(re.findall(r"\?v=(\d+\.\d+\.\d+)", idx))
    assert vs == {ver}, (vs, ver)
    assert read("CHANGELOG.md").startswith("# 更新日志") and "## [4.1.1]" in read("CHANGELOG.md")
    v4 = read("docs/v4.md")
    assert "前端上传未接" not in v4, "v4 文档仍有「前端上传未接」旧口径"
    ok("版本/缓存串/CHANGELOG/v4 文档一致（%s）" % ver)


if __name__ == "__main__":
    test_fuse_median_height()
    test_fuse_normal_axes()
    test_tsdf_blackface()
    test_serve_api_static_source()
    test_serve_api_integration()
    test_start_stop_v4()
    test_frontend()
    test_version_docs()
    print("\nv4.1.1 回归测试全部通过（%d 组）" % PASS)
