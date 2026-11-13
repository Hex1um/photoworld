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
def rope_cache(dim: int, n_t: int, gh: int, gw: int, device, dtype, extra: int = 0):
    """角度表由纯 numpy 的 rope_angles 生成：前后半频率严格配对。

    旧实现把三轴各 20 维拼接成 60 维再补 4 个零，与 apply_rope 的
    half-split 旋转不配对，破坏了旋转等变性。详见 worldmodel/model/rope.py。
    """
    from worldmodel.model.rope import rope_angles
    ang = torch.as_tensor(rope_angles(dim, n_t, gh, gw, extra),
                          dtype=torch.float32, device=device)
    return ang.cos().to(dtype), ang.sin().to(dtype)


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
        # 标准 DiT 初始化：残差分支的 gate 从 0 起步，block 初始为恒等映射。
        # chunk(6) 的顺序是 (s1, sc1, g1, s2, sc2, g2)，g1/g2 是第 3、6 段。
        # 与"输出头零初始化"相反，这样第一步梯度就能贯通所有层。
        with torch.no_grad():
            for lo in (2 * cfg.dim, 5 * cfg.dim):
                self.mod.weight[lo:lo + cfg.dim].zero_()
                self.mod.bias[lo:lo + cfg.dim].zero_()

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
        # CFG 的"无条件"必须是独立学习的嵌入，不能拿 idle(动作=0) 冒充
        self.null_act = nn.Parameter(torch.zeros(cfg.dim))
        self.null_dom = nn.Parameter(torch.zeros(cfg.dim))
        self.dom_embed = nn.Embedding(cfg.n_domains, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.head = nn.Linear(cfg.dim, cfg.in_ch * cfg.patch * cfg.patch)
        # 不再零初始化输出头：W_head=0 会让 dL/dx = W_head^T·grad = 0，
        # 训练最初若干步所有 block 的梯度恒为零，只有 head 在拟合常数速度场。
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # ---- 组装 token 序列 ----
    def _tokens(self, x_frame: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, gh*gw, dim)"""
        return self.patch(x_frame).flatten(2).transpose(1, 2)

    def forward(self, init_lat, hist_lat, x_t, t, action, domain=None, uncond=False):
        """
        init_lat : (B, C, H, W) 初始帧潜变量
        hist_lat : (B, N, C, H, W) 历史帧（旧→新）
        x_t      : (B, C, H, W) 目标帧在当前 t 的噪声样本
        t        : (B,) 0..1
        action   : (B, act_dim)
        uncond   : bool 或 (B,) bool。True 时**同时**丢弃动作、初始帧与域标签，
                   与训练侧的联合 dropout 走完全相同的路径。推理的 CFG 无条件
                   分支必须传 True，不能再只把 action 置零（act_embed 有 bias，
                   且训练还丢了 init/domain，两条路径根本不等价）。
        """
        B, C, H, W = x_t.shape
        cfg = self.cfg
        n_frames = 2 + cfg.n_hist            # init + hist + target
        dev = x_t.device
        u = self._uncond_mask(uncond, B, dev).view(B, 1)   # 0=有条件, 1=无条件

        toks = []
        init_tok = self._tokens(init_lat) * (1.0 - u).view(B, 1, 1)
        toks.append(init_tok)
        for i in range(cfg.n_hist):
            toks.append(self._tokens(hist_lat[:, i]))
        # 动作 token：插在目标帧前，注意力层可直接"读"到动作
        a = self.act_embed(action)                                    # (B, D)
        a = (1.0 - u) * a + u * self.null_act.view(1, -1)
        toks.append(a.unsqueeze(1))
        toks.append(self._tokens(x_t))
        seq = torch.cat(toks, dim=1)

        # 条件向量只留时间步 + 域；动作走 token
        c = self.t_embed(self._sinusoid(t, cfg.dim, dev))
        if domain is not None:
            dom = self.dom_embed(domain.clamp(0, cfg.n_domains - 1))
        else:
            dom = torch.zeros_like(c)
        c = c + (1.0 - u) * dom + u * self.null_dom.view(1, -1)

        cos, sin = rope_cache(cfg.dim // cfg.heads, n_frames, cfg.gh, cfg.gw, dev, seq.dtype,
                              extra=1)
        mask = self._causal_mask(n_frames, cfg.per_frame, dev, seq.dtype)

        x = seq
        for blk in self.blocks:
            x = blk(x, c, cos, sin, mask)
        x = self.norm(x)
        out = self.head(x[:, -(cfg.per_frame + 1):-1])                # 跳过动作 token
        out = out.view(B, cfg.gh, cfg.gw, cfg.patch, cfg.patch, C)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return out

    @staticmethod
    def _uncond_mask(uncond, B: int, device) -> torch.Tensor:
        """bool / (B,) bool → (B,) float 0/1，供条件与无条件分支共用一条前向路径。"""
        if isinstance(uncond, bool):
            return torch.full((B,), 1.0 if uncond else 0.0, device=device)
        return torch.as_tensor(uncond, dtype=torch.float32, device=device).reshape(-1)

    @staticmethod
    def _sinusoid(t: torch.Tensor, dim: int, device) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        ang = t.view(-1, 1) * freqs.view(1, -1) * 1000.0
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    @staticmethod
    def _causal_mask(n_frames: int, per_frame: int, device, dtype):
        """时间维因果、帧内双向。序列末尾多一个动作 token，归入最后一帧。"""
        idx = torch.arange(n_frames * per_frame + 1, device=device)
        fid = torch.clamp(idx // per_frame, max=n_frames - 1)
        # 用 bool 掩码（True=可见）：SDPA 对 bool 掩码走高效路径，
        # 浮点加法掩码会退回 math 回退实现，反向极慢
        allow = fid.view(-1, 1) >= fid.view(1, -1)
        return allow.view(1, 1, *allow.shape)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
