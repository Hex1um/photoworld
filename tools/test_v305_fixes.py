#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3.0.5 六项 bug 修复回归（纯 AST + 源码断言，无第三方依赖 / 无需服务或模型）。

    python tools/test_v305_fixes.py

纪律沿用 test_backend_fixes.py：**每条修复都必须有一条会 fail 的断言**。
本文件用静态断言锁定下列回退：

  1. pipeline/v3/frustum.py  uv_photo 赋值早于引用（否则 v3 烘焙 UnboundLocalError）
  2. pipeline/reconstruct.py  默认 --fov=0 不再引用未赋值的 hfov（否则默认调用必崩）
  3. pipeline/v3/bake.py      view_bias 不被统一着色段重置为空（否则报告恒空）
  4. tools/shot.py            页面从未就绪时拒绝截图并失败退出
  5. tools/probe.py           页面从未就绪时拒绝探测并失败退出
  6. tools/bench_glb.py       页面从未就绪时抛错（否则对错误页截图得 PSNR 99dB 假满分）
  7. tools/smoke.py           --only 空集时失败退出（否则 0/0 通过伪装成功）
"""
from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("  PASS  " if ok else "  FAIL  ") + name + (("  · " + detail) if detail else ""))


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def code_only(rel: str) -> str:
    """剥离行注释后的源码——静态断言只看代码，不被修复说明注释绊倒。"""
    out = []
    for line in read(rel).splitlines():
        out.append(line.split("#", 1)[0])
    return "\n".join(out)


def fn_node(rel: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(read(rel))
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    raise AssertionError("%s 中找不到函数 %s" % (rel, name))


# ---- 1. frustum.py：uv_photo 赋值必须在引用之前 ----
def test_frustum_uv_photo_order():
    n = fn_node("pipeline/v3/frustum.py", "build")
    assigns, loads = [], []
    for st in ast.walk(n):
        if isinstance(st, ast.Assign):
            for tg in st.targets:
                if isinstance(tg, ast.Name) and tg.id == "uv_photo":
                    assigns.append(st.lineno)
        if isinstance(st, ast.Name) and st.id == "uv_photo" and isinstance(st.ctx, ast.Load):
            loads.append(st.lineno)
    check("frustum uv_photo 有赋值", bool(assigns), "赋值行=%s" % sorted(assigns))
    check("frustum uv_photo 赋值早于首次引用",
          bool(assigns) and bool(loads) and min(assigns) < min(loads),
          "赋值=%s 引用=%s" % (sorted(assigns), sorted(loads)))


# ---- 2. reconstruct.py：默认分支不能引用未赋值的 hfov ----
def test_reconstruct_hfov():
    src = code_only("pipeline/reconstruct.py")
    check("reconstruct load_photo 写入 meta['hfov_rad']",
          'meta["hfov_rad"] = hfov' in src)
    check("reconstruct 默认分支从 meta 读 hfov",
          'meta.get("hfov_rad")' in src)
    check("reconstruct 不再有自引用写法 `hfov = hfov if hfov`",
          "hfov = hfov if hfov" not in src)


# ---- 3. bake.py：view_bias 初始化唯一且不被重置 ----
def test_bake_view_bias():
    src = code_only("pipeline/v3/bake.py")
    n_init = src.count("view_bias = {}") + src.count("view_bias: dict = {}")
    check("bake view_bias 初始化恰好一处（不被重置）", n_init == 1, "出现 %d 次" % n_init)
    check("bake view_bias 来自 point_cloud",
          'view_bias = cloud.get("view_bias", {})' in src)
    check("bake 报告写出 terrain.view_bias",
          '"terrain": {"view_bias": view_bias}' in src)


# ---- 4-6. 就绪检测：超时/未就绪必须失败，不能静默继续 ----
def test_readiness_guards():
    shot = code_only("tools/shot.py")
    check("shot 未就绪时拒绝截图", "拒绝截图" in read("tools/shot.py")
          and "return 1" in shot)

    probe = code_only("tools/probe.py")
    check("probe 未就绪时拒绝探测", "拒绝探测" in read("tools/probe.py")
          and "return 1" in probe)

    bench = code_only("tools/bench_glb.py")
    check("bench open_page 未就绪时抛错",
          "页面未就绪" in read("tools/bench_glb.py") and "raise RuntimeError" in bench)
    check("bench open_page 记录 ready 标志（非无条件 break）",
          "ready = False" in bench and "ready = True" in bench and "if not ready" in bench)


# ---- 7. smoke.py：--only 空集必须失败 ----
def test_smoke_empty_only():
    src = code_only("tools/smoke.py")
    check("smoke --only 空集返回非 0", "if not cases" in src and "return 2" in src)


def main() -> int:
    print("v3.0.5 修复回归（AST + 源码断言）")
    test_frustum_uv_photo_order()
    test_reconstruct_hfov()
    test_bake_view_bias()
    test_readiness_guards()
    test_smoke_empty_only()
    print("\n%d 通过，%d 失败" % (len(PASS), len(FAIL)))
    if FAIL:
        print("失败项：" + ", ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
