#!/usr/bin/env python3
"""bump_cache 入口测试：mjs 薄包装与 py 直调，在各自沙箱里输出必须一致。

  python tools/test_bump_cache_parity.py

bump_cache.mjs 已是转调 bump_cache.py 的薄包装；本测试确认两条入口
（node mjs / python py）在临时沙箱里行为一致，且版本串不一致时都能检出。
脚本都基于「自身所在目录的父级」定位 index.html，故复制进独立沙箱 tools/ 下跑。
Python 实现与原 mjs 实现的逐字节等价性已在改写时用旧 mjs 验证过。
只依赖标准库 + node。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJS = os.path.join(ROOT, "tools", "bump_cache.mjs")
PY = os.path.join(ROOT, "tools", "bump_cache.py")

INDEX_SAMPLE = """<!doctype html>
<link rel="stylesheet" href="a.css?v=1.0.0">
<script src="js/app.js?v=1.0.0"></script>
<script src="js/renderer.js?v=1.0.0"></script>
<img src="x.png?v=1.0.0">
"""
APP_SAMPLE = "const VERSION = '1.0.0';\n// app\n"


def _sandbox(script: str) -> str:
    """造一个含 tools/ 的沙箱；mjs 是薄包装，会把两个脚本都放进去。"""
    d = tempfile.mkdtemp(prefix="pw_bump_")
    os.makedirs(os.path.join(d, "js"))
    os.makedirs(os.path.join(d, "tools"))
    with open(os.path.join(d, "index.html"), "w", encoding="utf-8", newline="") as f:
        f.write(INDEX_SAMPLE)
    with open(os.path.join(d, "js", "app.js"), "w", encoding="utf-8", newline="") as f:
        f.write(APP_SAMPLE)
    shutil.copyfile(MJS, os.path.join(d, "tools", "bump_cache.mjs"))
    shutil.copyfile(PY, os.path.join(d, "tools", "bump_cache.py"))
    return d


def _run(d: str, script: str, arg: str):
    cmd = (["node", os.path.join(d, "tools", "bump_cache.mjs"), arg]
           if script.endswith(".mjs")
           else [sys.executable, os.path.join(d, "tools", "bump_cache.py"), arg])
    return subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=30)


def _version_json(d: str):
    p = os.path.join(d, "version.json")
    if not os.path.exists(p):
        return None
    obj = json.load(open(p, encoding="utf-8"))
    obj.pop("builtAt", None)          # 时间戳无法逐字节比，只比结构/版本
    return obj


def test_bump_parity():
    a, b = _sandbox(MJS), _sandbox(PY)
    try:
        ra, rb = _run(a, MJS, "2.5.9"), _run(b, PY, "2.5.9")
        assert ra.returncode == rb.returncode == 0, (ra.stdout, ra.stderr, rb.stdout, rb.stderr)
        for name in ("index.html", "js/app.js"):
            assert open(os.path.join(a, name), "rb").read() == \
                   open(os.path.join(b, name), "rb").read(), "%s 字节不一致" % name
        assert _version_json(a) == _version_json(b), "version.json 结构不一致"
        print("PASS  B1  升版产物逐字节一致（index.html / app.js / version.json，除时间戳）")
    finally:
        shutil.rmtree(a, ignore_errors=True)
        shutil.rmtree(b, ignore_errors=True)


def test_check_parity():
    a, b = _sandbox(MJS), _sandbox(PY)
    try:
        ra, rb = _run(a, MJS, "--check"), _run(b, PY, "--check")
        assert ra.returncode == rb.returncode == 0, (ra.stderr, rb.stderr)
        assert ra.stdout.strip() == rb.stdout.strip(), \
            "--check 输出不一致:\n  mjs=%r\n  py =%r" % (ra.stdout, rb.stdout)
        print("PASS  B2  --check 输出一致：%s" % ra.stdout.strip())
    finally:
        shutil.rmtree(a, ignore_errors=True)
        shutil.rmtree(b, ignore_errors=True)


def test_inconsistent_detected_both():
    for script in (MJS, PY):
        d = _sandbox(script)
        try:
            p = os.path.join(d, "index.html")
            s = open(p, encoding="utf-8").read().replace(
                "js/renderer.js?v=1.0.0", "js/renderer.js?v=9.9.9")
            with open(p, "w", encoding="utf-8", newline="") as f:
                f.write(s)
            r = _run(d, script, "--check")
            assert r.returncode == 1 and "版本串不一致" in r.stderr, \
                "%s 未检出版本串不一致: %r" % (os.path.basename(script), r.stderr)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print("PASS  B3  版本串不一致时 mjs/py 的 --check 都正确报错退出")


if __name__ == "__main__":
    test_bump_parity()
    test_check_parity()
    test_inconsistent_detected_both()
    print("\nbump_cache 翻译等价性全部通过（3 项）")
