"""photoworld env for SkillOpt.

把 PHOTOWORLD 的烘焙参数当作"技能内容"来训练：skill 文档里有一段 yaml
（`params:` 下的一组烘焙参数），rollout 就是"按这段参数烘焙 + 渲染 + 评分"。

这里 target **是 LLM**（config 的 `llm_target: true`）：每轮由它读 skill 准则、
输出一组烘焙参数，再跑确定性流水线（烘焙+渲染+评分）；不达标时把指标喂回
target 再要一组。优化器模型（DeepSeek / OpenAI 兼容）负责**改 skill 文本**。
"""
from __future__ import annotations

from skillopt.envs.photoworld.adapter import PhotoWorldAdapter

__all__ = ["PhotoWorldAdapter"]
