#!/usr/bin/env python3
"""api.py 模块单例缓存回归测试（无 GPU / 无 torch / 无浏览器）。

  python tools/test_api_cache.py

覆盖 v2.15.2（P0 性能修复，因远端在 v2.13.2 后推进到 v2.15.1 而重做）：旧 `score_params.bake()` 每次调用都
`spec_from_file_location + exec_module` 重建 api.py，把它的进程级
`_model`（1.3GB 权重）和 `_cache`（内容哈希 GLB 缓存）清零。
现在 `_get_api()` 用模块级 `_API_CACHE` 缓存模块实例（简化自旧的 sys.modules 注册表），
并用 (mtime_ns, size) 指纹自动失效。

用临时假 api.py 替身，避免真的加载深度模型；只依赖标准库。
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_sp():
    spec = importlib.util.spec_from_file_location(
        "pwscore_cache_test", os.path.join(ROOT, "tools", "score_params.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pwscore_cache_test"] = mod
    spec.loader.exec_module(mod)
    return mod


SP = _load_sp()


def _clear_registry():
    SP._API_CACHE.update(fingerprint=None, module=None, loads=0)


def _make_fake_root(value: int = 42):
    """造一个最小 api.py 替身，返回 (root, fake_path)。"""
    tmp = tempfile.mkdtemp(prefix="pw_api_cache_")
    d = os.path.join(tmp, "pipeline", "v2")
    os.makedirs(d)
    fake = os.path.join(d, "api.py")
    with open(fake, "w", encoding="utf-8") as f:
        f.write("VALUE = %d\n" % value)
    return tmp, fake


def test_cache_hit_same_module():
    old = SP.ROOT
    tmp, fake = _make_fake_root(42)
    try:
        SP.ROOT = tmp
        _clear_registry()
        a = SP._get_api()
        b = SP._get_api()
        assert a is b, "同指纹下必须返回同一模块实例（模型/缓存才能保住）"
        assert a.VALUE == 42, "假 api.py 应正常执行"
        reg = SP._API_CACHE
        assert reg["loads"] == 1, "只应加载一次，实际 %d 次" % reg["loads"]
        assert reg["fingerprint"] == (os.stat(fake).st_mtime_ns, os.stat(fake).st_size)
        print("PASS  C1  同指纹复用同一 api 模块实例，仅加载一次")
    finally:
        SP.ROOT = old
        _clear_registry()
        shutil.rmtree(tmp, ignore_errors=True)


def test_fingerprint_change_reloads():
    old = SP.ROOT
    tmp, fake = _make_fake_root(42)
    try:
        SP.ROOT = tmp
        _clear_registry()
        a = SP._get_api()
        # 改内容 + 显式推进 mtime_ns（不用 sleep，确定性强）
        with open(fake, "w", encoding="utf-8") as f:
            f.write("VALUE = 99\n")
        st = os.stat(fake)
        os.utime(fake, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        b = SP._get_api()
        reg = SP._API_CACHE
        assert b is not a, "文件变更后必须重建模块"
        assert b.VALUE == 99, "新模块应加载新代码，实际 VALUE=%r" % getattr(b, "VALUE", None)
        assert reg["loads"] == 2, "应恰好重载一次，实际 %d 次" % reg["loads"]
        print("PASS  C2  api.py 指纹(mtime_ns,size)变化后自动重载")
    finally:
        SP.ROOT = old
        _clear_registry()
        shutil.rmtree(tmp, ignore_errors=True)


def test_bake_uses_get_api():
    src = open(os.path.join(ROOT, "tools", "score_params.py"), encoding="utf-8").read()
    # 定位 bake() 函数体（到下一个顶层 def 为止），确认不再裸建模块
    i = src.index("def bake(")
    j = src.find("\ndef ", i + 1)
    body = src[i:j if j != -1 else len(src)]
    assert "api = _get_api()" in body, "bake() 必须走 _get_api()"
    assert "spec_from_file_location" not in body, "bake() 不应再直接重建模块"
    assert "_API_CACHE" in src and "_API_REGISTRY_KEY" not in src, \
        "应用模块级 _API_CACHE 单例，不再用 sys.modules 注册表"
    print("PASS  C3  bake() 走 _get_api()（模块级单例），不再每次 exec_module 重建")


if __name__ == "__main__":
    test_cache_hit_same_module()
    test_fingerprint_change_reloads()
    test_bake_uses_get_api()
    print("\napi 模块缓存回归全部通过（3 项）")
