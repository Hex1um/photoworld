#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v4.1.2 回归：质量门不再放过"没有基线但 RMSE 很小"的素材；SfM 去重与任务清理。

纯 numpy/标准库即可；唯一的可选集成测试会找装有 cv2 的解释器，找不到就跳过。

    python tools/test_v412_fixes.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

PASS = 0
SKIP = 0


def ok(label: str) -> None:
    global PASS
    PASS += 1
    print("PASS  %s" % label)


def skip(label: str) -> None:
    global SKIP
    SKIP += 1
    print("SKIP  %s" % label)


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def fake_result(localized: int, rmse: float, parallax: list[float],
                pairs: int | None = None, npts: int = 500):
    n = pairs if pairs is not None else len(parallax)
    ps = [SimpleNamespace(inliers=100, parallax=(parallax[i] if i < len(parallax) else 0.0))
          for i in range(n)]
    import numpy as np
    pts = np.zeros((npts, 3), np.float32)
    return SimpleNamespace(localized=localized, reproj_rmse=rmse, pairs=ps, pts=pts)


def test_qc_no_baseline_fails():
    """复现：4/7 定位、1.40px、0.25° 的纯旋转素材，v4.1.1 判 WARN 并出世界。"""
    from v4.qc import summarize

    r = fake_result(4, 1.40, [0.25] * 6)
    rep = summarize(r, 7)
    assert rep["verdict"] == "FAIL", rep
    assert any("视差不足" in a for a in rep["advice"]), rep["advice"]
    ok("质量门：纯旋转素材（0.25°）硬 FAIL")


def test_qc_zero_parallax_in_median():
    """可解配对里视差为 0 的也必须进中位，否则 1 组 5° 就能冒充整体质量。"""
    from v4.qc import summarize

    rep = summarize(fake_result(6, 1.0, [5.0] + [0.0] * 5), 6)
    assert rep["parallax_median_deg"] < 1.0, rep
    assert rep["verdict"] == "FAIL", rep
    ok("质量门：视差中位包含 0°，不再被单组高视差拉高")


def test_qc_points_and_levels():
    from v4.qc import summarize

    assert summarize(fake_result(9, 1.0, [3.0] * 8, npts=0), 9)["verdict"] == "FAIL"
    assert summarize(fake_result(9, 1.0, [3.0] * 8, npts=10), 9)["verdict"] == "FAIL"
    assert summarize(fake_result(9, 1.0, [3.0] * 8), 9)["verdict"] == "PASS"
    assert summarize(fake_result(6, 2.0, [0.8] * 6), 9)["verdict"] == "WARN"
    assert summarize(fake_result(6, 7.0, [3.0] * 6), 9)["verdict"] == "FAIL"
    # v4.1.3：门槛按实测重设（0.1 已是完整绕拍的正常值，不能再判 FAIL）
    assert summarize(fake_result(9, 1.0, [3.0] * 8), 9, fill=0.018)["verdict"] == "FAIL"
    assert summarize(fake_result(9, 1.0, [3.0] * 8), 9, fill=0.103)["verdict"] == "PASS"
    ok("质量门：PASS/WARN/FAIL 三档边界与点数/填充率硬门槛")


def test_sfm_source():
    src = read("pipeline/v4/sfm.py")
    assert "_match_cache: dict[tuple[int, int], np.ndarray]" in src, "匹配未缓存"
    assert src.count("_match(dess[") == 1, "仍有重复 BFMatcher 调用"
    assert "g2f: dict[int, int]" in src, "PnP 对应点仍是线性扫描"
    assert "m[m[:, 0] == kg]" not in src
    ok("SfM：成对匹配只算一次 + PnP 对应点哈希查找")


def test_cleanup_source():
    src = read("tools/serve_api.py")
    assert "_cleanup_old_jobs" in src and "max_age_hours" in src
    ok("serve_api 源码：过期任务清理已接入")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_cleanup_integration():
    job = "jtestcleanup0001"
    old = time.time() - 48 * 3600
    made = []
    for base in (ROOT / "uploads", ROOT / "out" / "v4" / "worlds"):
        d = base / job
        d.mkdir(parents=True, exist_ok=True)
        f = d / "stale.txt"
        f.write_text("stale", encoding="utf-8")
        os.utime(f, (old, old))
        os.utime(d, (old, old))
        made.append(d)
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "serve_api.py"), "--port", str(port)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/index.html" % port, timeout=2)
                break
            except Exception:
                time.sleep(0.25)
        for d in made:
            assert not d.exists(), "过期任务目录没被清理：" + str(d)
        ok("serve_api 集成：>24h 的上传/产物目录被清理")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        for d in made:
            shutil.rmtree(d, ignore_errors=True)


def test_synthetic_gate_end_to_end():
    """可选：用真 cv2 跑一遍 --qc-only，确认纯旋转集确实被拦。"""
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from serve_api import _find_build_python
    except Exception:
        _find_build_python = lambda: None  # noqa: E731
    py = _find_build_python()
    if not py:
        skip("合成集端到端（找不到带 cv2 的 Python）")
        return
    argv = py.split() if " " in py else [py]
    r = subprocess.run(argv + [str(ROOT / "pipeline" / "v4" / "build.py"),
                               "--input", str(ROOT / "testdata" / "v4_terrain"),
                               "--qc-only", "--out", str(ROOT / "out" / "v412_gate_check")],
                       cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=180)
    assert r.returncode == 1, (r.returncode, r.stdout[-600:], r.stderr[-400:])
    assert "判定 FAIL" in r.stdout, r.stdout[-600:]
    ok("合成集端到端：v4_terrain 被质量门拦下（退出码 1）")


def test_version():
    # 与 v411 测试同理：本组锁的是 v4.1.2 的修复没被回退，不锁死具体版本号。
    ver = json.loads(read("version.json"))["version"]
    assert tuple(int(x) for x in ver.split(".")) >= (4, 1, 2), ver
    vs = set(re.findall(r"\?v=(\d+\.\d+\.\d+)", read("index.html")))
    assert vs == {ver}, (vs, ver)
    assert "## [4.1.2]" in read("CHANGELOG.md")
    v4 = read("docs/v4.md")
    assert "< 0.5°" in v4 and "稀疏点" in v4
    assert "0.25° | 1.40px" not in v4, "实测表还是旧视差数值"
    ok("版本/缓存串/文档一致（≥4.1.2，当前 %s）" % ver)


if __name__ == "__main__":
    test_qc_no_baseline_fails()
    test_qc_zero_parallax_in_median()
    test_qc_points_and_levels()
    test_sfm_source()
    test_cleanup_source()
    test_cleanup_integration()
    test_synthetic_gate_end_to_end()
    test_version()
    print("\nv4.1.2 回归测试全部通过（%d 组，跳过 %d）" % (PASS, SKIP))
