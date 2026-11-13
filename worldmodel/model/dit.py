"""Causal World DiT：潜空间自回归帧预测

关键设计
--------
* 潜变量 patchify(2×2) → token；位置编码用 RoPE，三轴 (t, y, x)
* 注意力因果沿**时间**维，同一帧内双向
* 条件：frame_0 前缀（永久参与）、最近 N 帧历史、动作 token、域 token
* 目标：flow matching（rectified flow），预测速度场 v = x1 - x0
  采样单步：x ← x + dt * v(x, t)，从 t=0（噪声）积分到 t=1（数据）
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiTConfig:
    in_ch: int = 4
    latent_h: int = 32
    latent_w: int = 48
    patch: int = 2
    dim: int = 768
    depth: int = 12
    heads: int = 12
    n_hist: int = 4
    act_dim: int = 7
    n_domains: int = 4
    mlp_ratio: float = 4.0

    @property
    def gh(self): return self.latent_h // self.patch

    @property
    def gw(self): return self.latent_w // self.patch

    @property
    def per_frame(self): return self.gh * self.gw


# ---------------- RoPE（三轴） ----------------
def rope_cache(dim: int, n_t: int, gh: int, gw: int, device, dtype):
    """把通道维均分给 (t, y, x) 三个轴。"""
    per = dim // 3
    per -= per % 2
    def freqs(n, axis_dim):
        half = axis_dim // 2
        inv = 1.0 / (10000 ** (torch.arange(half, device=device, dtype=torch.float32) / half))
        pos = torch.arange(n, device=device, dtype=torch.float32)
        ang = torch.outer(pos, inv)                      # (n, half)
        return torch.cat([ang, ang], dim=-1)             # (n, axis_dim)
    ft = freqs(n_t, per)
    fy = freqs(gh, per)
    fx = freqs(gw, per)
    # 每个 token 的 cos/sin
    cos = torch.cat([ft[:, None, None, :].expand(n_t, gh, gw, per),
                     fy[None, :, None, :].expand(n_t, gh, gw, per),
                     fx[None, None, :, :].expand(n_t, gh, gw, per)], dim=-1)
    cos = cos.reshape(n_t * gh * gw, -1)
    # 通道数不整除时，剩余通道不做旋转（cos=1, sin=0）
    pad = dim - cos.shape[-1]
    if pad > 0:
        cos = torch.cat([cos, torch.zeros(cos.shape[0], pad, device=device)], dim=-1)
    return cos.cos().to(dtype), cos.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, L, H, Dh)"""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos[:, None, :] + rotated * sin[:, None, :]


class Attention(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.h = cfg.heads
        self.dh = cfg.dim // cfg.heads
        self.qkv = nn.Linear(cfg.dim, cfg.dim * 3)
        self.out = nn.Linear(cfg.dim, cfg.dim)

    def forward(self, x, cos, sin, mask=None):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.h, self.dh)
        k = k.view(B, L, self.h, self.dh)
        v = v.view(B, L, self.h, self.dh)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))     # (B,H,L,Dh)
        # 用 SDPA（sm_120 上走 flash/mem-efficient，显存低得多）
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, L, -1)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.n1 = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.attn = Attention(cfg)
        self.n2 = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, int(cfg.dim * cfg.mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(cfg.dim * cfg.mlp_ratio), cfg.dim),
        )
        self.mod = nn.Linear(cfg.dim, cfg.dim * 6)   # adaLN: shift/scale/gate ×2

    def forward(self, x, c, cos, sin, mask):
        m = self.mod(F.silu(c))[:, None, :]
        s1, sc1, g1, s2, sc2, g2 = m.chunk(6, dim=-1)
        h = self.n1(x) * (1 + sc1) + s1
        x = x + g1 * self.attn(h, cos, sin, mask)
        h = self.n2(x) * (1 + sc2) + s2
        x = x + g2 * self.mlp(h)
        return x


class CausalWorldDiT(nn.Module):
    """输入：初始帧、历史帧、动作、噪声目标帧；输出：速度场。"""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(cfg.in_ch, cfg.dim, kernel_size=cfg.patch, stride=cfg.patch)
        self.t_embed = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.act_embed = nn.Sequential(nn.Linear(cfg.act_dim, cfg.dim), nn.SiLU(),
                                       nn.Linear(cfg.dim, cfg.dim))
        self.dom_embed = nn.Embedding(cfg.n_domains, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.head = nn.Linear(cfg.dim, cfg.in_ch * cfg.patch * cfg.patch)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ---- 组装 token 序列 ----
    def _tokens(self, x_frame: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, gh*gw, dim)"""
        return self.patch(x_frame).flatten(2).transpose(1, 2)

    def forward(self, init_lat, hist_lat, x_t, t, action, domain=None, drop_act=None, drop_init=None):
        """
        init_lat : (B, C, H, W) 初始帧潜变量
        hist_lat : (B, N, C, H, W) 历史帧（旧→新）
        x_t      : (B, C, H, W) 目标帧在当前 t 的噪声样本
        t        : (B,) 0..1
        action   : (B, act_dim)
        """
        B, C, H, W = x_t.shape
        cfg = self.cfg
        n_frames = 2 + cfg.n_hist            # init + hist + target
        L = n_frames * cfg.per_frame
        dev = x_t.device

        toks = []
        init_tok = self._tokens(init_lat)
        if drop_init is not None:
            keep = (~drop_init).float().view(B, 1, 1)
            init_tok = init_tok * keep
        toks.append(init_tok)
        for i in range(cfg.n_hist):
            toks.append(self._tokens(hist_lat[:, i]))
        toks.append(self._tokens(x_t))
        seq = torch.cat(toks, dim=1)                                  # (B, L, D)

        # 条件向量：时间步 + 动作 + 域
        c = self.t_embed(self._sinusoid(t, cfg.dim, dev))
        a = self.act_embed(action)
        if drop_act is not None:
            a = a * (~drop_act).float().view(B, 1)
        c = c + a
        if domain is not None:
            c = c + self.dom_embed(domain.clamp(0, cfg.n_domains - 1))

        # 动作 token 插在目标帧前 → 会打乱索引，这里改为把动作并进 adaLN 条件
        # （spec 允许"action token 插在目标帧前"，此处用等价且更省 token 的做法）
        cos, sin = rope_cache(cfg.dim // cfg.heads, n_frames, cfg.gh, cfg.gw, dev, seq.dtype)
        mask = self._causal_mask(n_frames, cfg.per_frame, dev, seq.dtype)

        x = seq
        for blk in self.blocks:
            x = blk(x, c, cos, sin, mask)
        x = self.norm(x)
        out = self.head(x[:, -cfg.per_frame:])                        # 只预测目标帧
        out = out.view(B, cfg.gh, cfg.gw, cfg.patch, cfg.patch, C)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return out

    @staticmethod
    def _sinusoid(t: torch.Tensor, dim: int, device) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        ang = t.view(-1, 1) * freqs.view(1, -1) * 1000.0
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    @staticmethod
    def _causal_mask(n_frames: int, per_frame: int, device, dtype):
        """时间维因果、帧内双向：token i 可看 frame(i) 及之前的帧。"""
        idx = torch.arange(n_frames * per_frame, device=device)
        fid = idx // per_frame
        allow = fid.view(-1, 1) >= fid.view(1, -1)          # (L, L) True=可见
        m = torch.zeros_like(allow, dtype=dtype)
        m = m.masked_fill(~allow, -1e4)   # bf16 下 -inf 会在 softmax 里产生 NaN
        return m.view(1, 1, *m.shape)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
