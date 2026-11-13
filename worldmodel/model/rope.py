"""三轴 RoPE 的角度表（纯 numpy，便于无 torch 环境验证等变性）。

为什么单独放一层
----------------
`apply_rope` 用 half-split 旋转：`rotated = [-x[d/2:], x[:d/2]]`，
它要求角度表满足 `ang[..., :d/2] == ang[..., d/2:]`，第 j 与第 j+d/2
通道才共享同一频率，旋转才有 `<RoPE(q,p1),RoPE(k,p2)> = f(p1-p2)` 的等变性。

旧实现把 (t,y,x) 各 20 维频率**拼接**成 60 维再补 4 个零：
前半 = t 的 20 + y 的前 12，后半 = y 的后 8 + x 的 20 + 常数，
两半完全不配对，三轴的相对位置信息实际是错乱的。

这里把 d/2 个频率通道分给三轴，再 `cat([f, f])` 复制成 d 维，
严格保证前后半一致。多出来的通道（d/2 不能被 3 整除时）按顺序补给 t、y。
"""
from __future__ import annotations

import numpy as np


def rope_angles(dim: int, n_t: int, gh: int, gw: int, extra: int = 0) -> np.ndarray:
    """返回 (n_t*gh*gw + extra, dim) 的角度（弧度），前半与后半严格相同。"""
    if dim % 2:
        raise ValueError("RoPE 的 head dim 必须是偶数，收到 %d" % dim)
    half = dim // 2
    per = half // 3
    sizes = [per + (1 if i < half % 3 else 0) for i in range(3)]

    def freqs(n: int, k: int) -> np.ndarray:
        inv = 1.0 / (10000.0 ** (np.arange(k, dtype=np.float64) / max(k, 1)))
        return np.outer(np.arange(n, dtype=np.float64), inv)  # (n, k)

    ft, fy, fx = freqs(n_t, sizes[0]), freqs(gh, sizes[1]), freqs(gw, sizes[2])
    ang = np.concatenate([
        np.broadcast_to(ft[:, None, None, :], (n_t, gh, gw, sizes[0])),
        np.broadcast_to(fy[None, :, None, :], (n_t, gh, gw, sizes[1])),
        np.broadcast_to(fx[None, None, :, :], (n_t, gh, gw, sizes[2])),
    ], axis=-1).reshape(n_t * gh * gw, half)
    full = np.concatenate([ang, ang], axis=-1)  # (L, dim)

    if extra:
        # 动作 token 插在目标帧之前，沿用目标帧的起始位置，
        # 与 _causal_mask 里"动作 token 归入最后一帧"的约定一致。
        n = gh * gw
        full = np.concatenate([
            full[:-n],
            np.repeat(full[-n:-n + 1], extra, axis=0),
            full[-n:],
        ], axis=0)
    return full


def apply_rope_np(x: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """与 dit.apply_rope 完全相同的 half-split 旋转，供纯 numpy 单测使用。"""
    d = x.shape[-1]
    h = d // 2
    rotated = np.concatenate([-x[..., h:], x[..., :h]], axis=-1)
    return x * np.cos(ang) + rotated * np.sin(ang)
