#!/usr/bin/env python3
"""训练兜底三连回归测试（无 GPU / 无浏览器 / 无真实 skillopt）。

  python tools/test_training_resilience.py

覆盖 v2.15.3 修掉的三处"整轮全 0 分 / 白等烘焙"问题：

  1. **inproc 失败不降级**：adapter 用 `if not d` 判断兜底，而失败路径返回的
     `{"ok": False, ...}` 是 truthy → 两级兜底永不触发，一次瞬时异常就整图 0 分。
  2. **score_server GBK 崩溃**：`subprocess.run(text=True)` 解不了 multiview
     字节流里混的 GBK → 0 分。
  3. **LLM 重试无早停**：参数未变或 soft 提升 <1e-4 时仍做第二次完整烘焙；
     且 `>=` 让平局覆盖旧 best。

本机没有 SkillOpt 包，所以用最小 stub 注入 sys.modules 后**正常 import** adapter
（不再用 AST 从源码抠函数）；score_server 只依赖标准库，直接 import。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stub(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _load_adapter():
    """用最小 stub 满足 skillopt 依赖，正常 import adapter。"""
    if "skillopt.envs.photoworld.adapter" in sys.modules:
        return sys.modules["skillopt.envs.photoworld.adapter"]
    for pkg in ("skillopt", "skillopt.datasets", "skillopt.envs", "skillopt.envs.photoworld"):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))

    class EnvAdapter:               # PhotoWorldAdapter 的基类
        def __init__(self, *a, **k):
            pass

    class BatchSpec:                # 仅类型注解用
        pass

    class SplitDataLoader:          # dataloader.PhotoWorldDataLoader 的基类
        def __init__(self, *a, **k):
            pass

    _stub("skillopt.datasets.base", BatchSpec=BatchSpec)
    _stub("skillopt.envs.base", EnvAdapter=EnvAdapter)
    _stub("skillopt.datasets", SplitDataLoader=SplitDataLoader)
    _stub("skillopt.envs.photoworld.dataloader",
          PhotoWorldDataLoader=SplitDataLoader)
    # task_output 缺失时 adapter 自带退化实现，无需 stub

    spec = importlib.util.spec_from_file_location(
        "pw_adapter_test",
        os.path.join(ROOT, "skillopt", "envs", "photoworld", "adapter.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pw_adapter_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_score_server():
    if "pw_score_server_test" in sys.modules:
        return sys.modules["pw_score_server_test"]
    spec = importlib.util.spec_from_file_location(
        "pw_score_server_test", os.path.join(ROOT, "tools", "score_server.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pw_score_server_test"] = mod
    spec.loader.exec_module(mod)
    return mod


AD = _load_adapter()
SS = _load_score_server()


def test_score_usable():
    f = AD._score_usable
    assert f({"ok": True, "hard": 1, "soft": 0.9}) is True
    assert f({"ok": False, "error": "进程内评分失败"}) is False, "失败字典是 truthy，必须显式判 ok"
    assert f({}) is False
    assert f(None) is False
    assert f({"hard": 1, "soft": 0.9}) is False, "缺 ok 字段不可当作成功"
    print("PASS  T1  _score_usable：只有 ok is True 才可用，失败字典正确判否")


def test_fallback_call_sites():
    """用 inspect 读方法源码，确认两处兜底判定都用了 _score_usable。"""
    import inspect
    body = inspect.getsource(AD.PhotoWorldAdapter._score_one)
    assert "_score_usable(d) and self.score_server" in body, "score_server 兜底须用 _score_usable"
    assert "if not _score_usable(d):" in body, "子进程兜底须用 _score_usable"
    assert "if not d:" not in body, "不应残留 `if not d` 的错误判定"
    print("PASS  T2  _score_one 两处兜底判定均改用 _score_usable，无 `if not d` 残留")


def test_gbk_decode():
    f = SS._decode_output
    good = '{"ok":true}'.encode("utf-8")
    bad = b"\xcf\x20tail"           # 孤立 0xcf：旧 text=True 在此抛异常
    out, err = f(good + bad, b"\xcf error")
    assert "ok" in out and "\ufffd" in out, "非法字节应替换为 U+FFFD 而非抛异常"
    assert "\ufffd" in err
    out2, err2 = f(None, None)
    assert out2 == "" and err2 == ""
    print("PASS  T3  _decode_output：GBK 混字节按 replace 解码，不抛异常")


def test_no_text_true_in_score_server():
    """AST 确认真实 subprocess.run 调用无 text=True（注释里提到不算）。"""
    import ast
    src = open(os.path.join(ROOT, "tools", "score_server.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "run" \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
            found.append(node)
    assert found, "没找到 subprocess.run 调用"
    for call in found:
        assert not any(kw.arg == "text" for kw in call.keywords), \
            "subprocess.run 不应再用 text=True（GBK 崩溃）"
    assert "_decode_output" in src, "应改用字节解码"
    print("PASS  T4  score_server subprocess.run 无 text=True，改走 _decode_output")


def test_early_stop_predicates():
    p_unchanged, p_gain = AD._params_unchanged, AD._gain_too_small
    assert p_unchanged({"a": 1}, {"a": 1}) is True
    assert p_unchanged({"a": 1}, {"a": 2}) is False
    assert p_unchanged({"a": 1}, None) is False
    assert p_gain(0.80, 0.80) is True, "平局应视为无提升（>= 的旧逻辑会覆盖 best）"
    assert p_gain(0.80001, 0.80) is True, "提升 1e-5 < eps(1e-4)，应早停"
    assert p_gain(0.805, 0.80) is False, "提升 5e-3 > eps，应继续重试"
    assert p_gain(0.79, 0.80) is True, "负提升应早停"
    assert p_gain(0.80, None) is False, "首轮无 prev，不应早停"
    print("PASS  T5  LLM 早停谓词：参数未变 / 平局 / 负提升 → 早停，首轮不早停")


def test_rollout_uses_strict_gt():
    import inspect
    body = inspect.getsource(AD.PhotoWorldAdapter._rollout_llm)
    assert "row[\"soft\"] > best[\"soft\"]" in body, "best 应用严格 >，平局不覆盖"
    assert "_params_unchanged" in body and "_gain_too_small" in body, "两处早停谓词都应接入"
    print("PASS  T6  _rollout_llm 用严格 >，且两处早停谓词已接入")


if __name__ == "__main__":
    test_score_usable()
    test_fallback_call_sites()
    test_gbk_decode()
    test_no_text_true_in_score_server()
    test_early_stop_predicates()
    test_rollout_uses_strict_gt()
    print("\n训练兜底回归全部通过（6 项）")
