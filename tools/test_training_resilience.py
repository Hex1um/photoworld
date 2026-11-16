#!/usr/bin/env python3
"""训练兜底三连回归测试（无 GPU / 无 skillopt / 无浏览器）。

  python tools/test_training_resilience.py

覆盖 v2.15.3 修掉的三处"整轮全 0 分 / 白等烘焙"问题：

  1. **inproc 失败不降级**：adapter._score_one 用 `if not d` 判断兜底，
     而失败路径返回的 `{"ok": False, ...}` 是 truthy → score_server 和子进程
     两级兜底永不触发，一次瞬时异常就把整图记成 0 分。
  2. **score_server GBK 崩溃**：`subprocess.run(text=True)` 解不了 multiview
     字节流里混的 GBK（0xcf）→ reader 线程抛 UnicodeDecodeError → 0 分。
  3. **LLM 重试无早停**：参数未变（temperature 0.2 + 同 prompt 几乎必同输出）
     或 soft 提升 <1e-4 时仍做第二次完整烘焙，每图白费 70~95 秒；
     且 `>=` 让平局覆盖旧 best。

用 AST 从源码精确提取纯函数执行，不 import adapter/score_server（避免依赖），
并配源码契约检查调用点确实改了。只依赖标准库。
"""
from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER = os.path.join(ROOT, "skillopt", "envs", "photoworld", "adapter.py")
SCORE_SERVER = os.path.join(ROOT, "tools", "score_server.py")


def _method_body(src: str, name: str) -> str:
    """在模块的任意类体里找方法并返回其源码（适配器方法都是类方法）。"""
    t = ast.parse(src)
    for n in ast.walk(t):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(src, n)
    raise AssertionError("找不到方法 %s" % name)


def _extract_functions(path: str, names: list[str]) -> dict:
    """从源码 AST 提取指定纯函数的定义并在隔离命名空间执行。"""
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    wanted = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(wanted) == len(names), "缺函数: %s" % (
        set(names) - {n.name for n in wanted})
    mod = ast.Module(body=wanted, type_ignores=[])
    ns: dict = {}
    exec(compile(mod, path, "exec"), ns)
    return ns


AD = _extract_functions(ADAPTER, ["_score_usable", "_params_unchanged", "_gain_too_small"])
SS = _extract_functions(SCORE_SERVER, ["_decode_output"])


def test_score_usable():
    f = AD["_score_usable"]
    assert f({"ok": True, "hard": 1, "soft": 0.9}) is True
    assert f({"ok": False, "error": "进程内评分失败"}) is False, "失败字典是 truthy，必须显式判 ok"
    assert f({}) is False
    assert f(None) is False
    assert f({"hard": 1, "soft": 0.9}) is False, "缺 ok 字段不可当作成功"
    print("PASS  T1  _score_usable：只有 ok is True 才可用，失败字典正确判否")


def test_fallback_call_sites():
    src = open(ADAPTER, encoding="utf-8").read()
    # 定位 _score_one 函数体
    body = _method_body(src, "_score_one")
    assert "_score_usable(d) and self.score_server" in body, "score_server 兜底须用 _score_usable"
    assert "if not _score_usable(d):" in body, "子进程兜底须用 _score_usable"
    assert "if not d:" not in body, "不应残留 `if not d` 的错误判定"
    print("PASS  T2  _score_one 两处兜底判定均改用 _score_usable，无 `if not d` 残留")


def test_gbk_decode():
    f = SS["_decode_output"]
    # 模拟 multiview 混 GBK 字节：合法 utf-8 + 孤立 0xcf（旧代码在此抛异常）
    good = '{"ok":true}'.encode("utf-8")
    bad = b"\xcf\x20tail"
    out, err = f(good + bad, b"\xcf error")
    assert "ok" in out and "\ufffd" in out, "非法字节应替换为 U+FFFD 而非抛异常"
    assert "\ufffd" in err
    out2, err2 = f(None, None)
    assert out2 == "" and err2 == ""
    print("PASS  T3  _decode_output：GBK 混字节按 replace 解码，不抛异常")


def test_no_text_true_in_score_server():
    src = open(SCORE_SERVER, encoding="utf-8").read()
    # 只看真实调用，不看注释（注释里解释"不用 text=True"是允许的）
    t = ast.parse(src)
    calls = [n for n in ast.walk(t)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "run"]
    for c in calls:
        assert not any(k.arg == "text" for k in c.keywords), "subprocess.run 不应再用 text=True"
    assert "_decode_output(r.stdout, r.stderr)" in src, "应改用字节解码"
    print("PASS  T4  score_server 真实调用无 text=True，改走 _decode_output")


def test_early_stop_helpers():
    unchanged, gain = AD["_params_unchanged"], AD["_gain_too_small"]
    assert unchanged({"a": 1}, {"a": 1}) is True
    assert unchanged({"a": 1}, {"a": 2}) is False
    assert unchanged({"a": 1}, None) is False
    assert gain(0.70, 0.70) is True, "平局应判为提升不足"
    assert gain(0.60, 0.70) is True, "负提升应判为提升不足"
    assert gain(0.85, 0.70) is False, "显著提升应继续"
    assert gain(0.5, None) is False, "首轮无 prev_soft 不适用"
    print("PASS  T5  早停纯函数：参数未变/提升不足(含负与平局)正确判定")


def test_early_stop_call_site():
    src = open(ADAPTER, encoding="utf-8").read()
    body = _method_body(src, "_rollout_llm")
    assert "_params_unchanged(params, prev_params)" in body
    assert "_gain_too_small(row[\"soft\"], prev_soft)" in body
    assert "row[\"soft\"] > best[\"soft\"]" in body, "平局不应覆盖 best（>= 改 >）"
    assert ">= best[\"soft\"]" not in body
    print("PASS  T6  _rollout_llm 两处早停已接入，best 覆盖改为严格 >")


if __name__ == "__main__":
    test_score_usable()
    test_fallback_call_sites()
    test_gbk_decode()
    test_no_text_true_in_score_server()
    test_early_stop_helpers()
    test_early_stop_call_site()
    print("\n训练兜底三连回归全部通过（6 项）")
