#!/usr/bin/env python3
"""R5 回归：depth_cap 接线 + iters 默认值统一（无 numpy / 纯静态检查）。

  python tools/test_depth_cap.py

覆盖：
  1. api.DEFAULTS 必须含 depth_cap（否则配置链传不到，只能吃 ldi 函数默认值）
  2. api 的 build 调用必须显式传 depth_cap=opts["depth_cap"]
  3. ldi.cap_depth_gradient 的默认 iters 与实际调用 iters 一致（此前 10 vs 12）
"""
from __future__ import annotations

import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = os.path.join(ROOT, "pipeline", "v2", "api.py")
LDI = os.path.join(ROOT, "pipeline", "v2", "ldi.py")


def _defaults(src: str, fn_name: str) -> dict:
    """取模块级函数的 {参数名: 默认值}（纯 AST，不执行）。"""
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn_name)
    names = [a.arg for a in fn.args.args]
    # defaults 对应参数列表末尾的 len(defaults) 个
    out = {}
    off = len(names) - len(fn.args.defaults)
    for name, d in zip(names[off:], fn.args.defaults):
        try:
            out[name] = ast.literal_eval(d)
        except Exception:
            out[name] = ast.dump(d)
    return out


def test_api_defaults_has_depth_cap():
    src = open(API, encoding="utf-8").read()
    tree = ast.parse(src)
    d = next(n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "DEFAULTS" for t in n.targets))
    keys = {k.value for k in d.value.keys if isinstance(k, ast.Constant)}
    assert "depth_cap" in keys, "api.DEFAULTS 缺 depth_cap，配置链无法开关限幅"
    val = next((ast.literal_eval(v) for k, v in zip(d.value.keys, d.value.values)
                if isinstance(k, ast.Constant) and k.value == "depth_cap"), None)
    assert val == 1.35, "depth_cap 默认应为 1.35（与 ldi.build 一致），实际 %r" % val
    print("PASS  D1  api.DEFAULTS 含 depth_cap=1.35（可配置开关）")


def test_api_passes_depth_cap():
    src = open(API, encoding="utf-8").read()
    assert 'depth_cap=float(opts["depth_cap"])' in src, \
        "build() 调用须显式传 depth_cap，否则 DEFAULTS 形同虚设"
    print("PASS  D2  build() 调用显式传 depth_cap=opts[depth_cap]")


def test_iters_default_matches_call():
    ldi = open(LDI, encoding="utf-8").read()
    defaults = _defaults(ldi, "cap_depth_gradient")
    assert defaults.get("iters") == 12, \
        "cap_depth_gradient 默认 iters 应与实际调用(12)统一，实际 %r" % defaults.get("iters")
    assert "cap_depth_gradient(zs, k=depth_cap, iters=12)" in ldi, \
        "build 调用应继续用 iters=12"
    print("PASS  D3  cap_depth_gradient 默认 iters=12，与 build 调用一致")


if __name__ == "__main__":
    test_api_defaults_has_depth_cap()
    test_api_passes_depth_cap()
    test_iters_default_matches_call()
    print("\ndepth_cap 接线回归全部通过（3 项）")
