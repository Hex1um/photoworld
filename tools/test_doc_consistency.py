#!/usr/bin/env python3
"""R7 回归：文档/配置/代码口径一致（纯文本 + AST，无第三方依赖）。

  python tools/test_doc_consistency.py

核对 target=LLM、机位数=8、阈值/权重表与 score_params 一致，且旧口径措辞不残留。
"""
from __future__ import annotations

import ast
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel: str) -> str:
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


def _thresh_weights() -> tuple[dict, dict]:
    src = _read("tools/score_params.py")
    tree = ast.parse(src)
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ("THRESH", "WEIGHT") for t in n.targets):
            name = next(t.id for t in n.targets if isinstance(t, ast.Name))
            out[name] = {k.value: ast.literal_eval(v)
                         for k, v in zip(n.value.keys, n.value.values)}
    return out["THRESH"], out["WEIGHT"]


def test_config_target_and_views():
    cfg = _read("skillopt/configs/photoworld.yaml")
    assert re.search(r"^\s*llm_target:\s*true", cfg, re.M), "config 必须 llm_target: true"
    assert re.search(r"^\s*views:\s*8", cfg, re.M), "config 必须 views: 8"
    print("PASS  C1  config: llm_target=true, views=8")


def test_no_old_target_claim():
    for rel in ("docs/skillopt.md", "skillopt/configs/photoworld.yaml",
                "skillopt/envs/photoworld/__init__.py", "tools/sync_skillopt.py"):
        txt = _read(rel)
        assert "target 不是 LLM" not in txt and "target **不是 LLM**" not in txt, \
            "%s 仍残留「target 不是 LLM」旧口径" % rel
        assert "不是 LLM" not in txt or "target" not in txt, \
            "%s 仍残留 target 不是 LLM 的措辞" % rel
    print("PASS  C2  4 处文档/注释均无「target 不是 LLM」旧口径")


def test_doc_table_matches_code():
    thresh, weight = _thresh_weights()
    doc = _read("docs/skillopt.md")
    assert len(thresh) == 8, "判据应为 8 项"
    for k, t in thresh.items():
        # 表格行形如 | `stripe` | ≤ 2.0 | 0.24 |
        m = re.search(r"\|\s*`%s`\s*\|\s*≤\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|" % re.escape(k), doc)
        assert m, "文档判定表缺 %s 行" % k
        assert float(m.group(1)) == t, "%s 阈值文档=%s 代码=%s" % (k, m.group(1), t)
        assert float(m.group(2)) == weight[k], "%s 权重文档=%s 代码=%s" % (k, m.group(2), weight[k])
    print("PASS  C3  文档 8 项阈值/权重表与 score_params.THRESH/WEIGHT 逐项一致")


def test_doc_no_old_metric_count():
    doc = _read("docs/skillopt.md")
    assert "四项全过" not in doc, "仍写「四项全过」旧口径"
    assert "八项全过" in doc, "应写「八项全过」"
    assert "4 机位" not in doc and "→ 4 机位" not in doc, "仍残留 4 机位"
    assert "最差值" in doc, "应说明最差机位聚合口径"
    print("PASS  C4  文档机位数=8、判据=8 项、说明最差机位口径")


if __name__ == "__main__":
    test_config_target_and_views()
    test_no_old_target_claim()
    test_doc_table_matches_code()
    test_doc_no_old_metric_count()
    print("\n文档口径一致性回归全部通过（4 项）")
