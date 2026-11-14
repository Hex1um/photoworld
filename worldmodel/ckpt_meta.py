"""checkpoint 元信息校验（无 torch 依赖，便于回归测试）。G3（v1.22）"""
from __future__ import annotations

#: 旧 checkpoint 的判定标记（DiT 输出切片修复于 v1.21.x）
DIT_SLICE_MARK = "dit_slice_fix"


def checkpoint_warning(cfg: dict) -> str | None:
    """旧 checkpoint 必须给出**显式**警告，而不是静默错位。

    DiT 输出切片的修复改变了隐变量的拆法；修复前训出来的档，切出来的
    latent 与当前代码不是同一段数据，续训/评测都会"能跑但结果不对"。
    """
    data = (cfg or {}).get("data", {}) or {}
    if data.get(DIT_SLICE_MARK) is True:
        return None
    return ("checkpoint 缺少 %s 标记：可能是 DiT 输出切片修复（v1.21.x）之前的产物，"
            "隐变量切法与当前代码不一致，评测/续训结果不可信。"
            "请用 v1.22+ 重新训练，或在 cfg.data 里显式写 %s: true。"
            % (DIT_SLICE_MARK, DIT_SLICE_MARK))


def resolve_hist_frames(cfg: dict, requested: int) -> tuple[int, str | None]:
    """T9-3（v1.22）：历史帧数以 **checkpoint 训练时的配置**为准。

    返回 (实际使用的帧数, 警告文案或 None)。抽成纯函数便于无 torch 回归。
    """
    hist_cfg = ((cfg or {}).get("data") or {}).get("hist_frames")
    if hist_cfg is None:
        return int(requested), None
    if int(hist_cfg) != int(requested):
        return int(hist_cfg), ("hist_frames=%d 与 checkpoint 的 %d 不一致，已按 checkpoint 覆盖"
                               % (int(requested), int(hist_cfg)))
    return int(requested), None
