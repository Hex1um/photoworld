"""photoworld env for SkillOpt.

把 PHOTOWORLD 的烘焙参数当作"技能内容"来训练：skill 文档里有一段 yaml
（`params:` 下的一组烘焙参数），rollout 就是"按这段参数烘焙 + 渲染 + 评分"。

与常规 SkillOpt 基准的区别：这里的 target **不是 LLM**，而是一条确定性流水线。
所以 rollout 不需要调用模型 —— 它只读 skill 里的参数、跑流水线、返回分数。
优化器模型（DeepSeek / OpenAI 兼容）只负责**改 skill 文本**。
"""
from __future__ import annotations

from skillopt.envs.photoworld.adapter import PhotoWorldAdapter

__all__ = ["PhotoWorldAdapter"]
