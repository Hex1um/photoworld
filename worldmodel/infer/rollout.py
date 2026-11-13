"""自回归 rollout：初始图 + 动作序列 → 视频帧。

单步采样（Euler）：
    x_{t+dt} = x_t + dt * v(x_t, t)
    t 从 0（纯噪声）走到 1（数据）；CFG：
    v = v_cond + w * (v_cond - v_uncond)，uncond = drop 动作条件
"""
from __future__ import annotations

import math

import torch

from worldmodel.model.dit import DiTConfig
from worldmodel.train.trainer import make_model
from worldmodel.model.tokenizer import Tokenizer

DSL = {
    "forward": (0.0, -1.0, 0.0),      # (dyaw, dforward, dpitch)
    "back": (0.0, 1.0, 0.0),
    "turn_left": (1.0, 0.0, 0.0),
    "turn_right": (-1.0, 0.0, 0.0),
    "look_up": (0.0, 0.0, 1.0),
    "look_down": (0.0, 0.0, -1.0),
    "idle": (0.0, 0.0, 0.0),
}


def parse_actions(spec: list) -> list[torch.Tensor]:
    """['forward 20', 'turn_left 30'] → 每帧一个 7 维动作张量。"""
    out = []
    for item in spec:
        if isinstance(item, (list, tuple)):
            name, n = item[0], int(item[1])
        else:
            parts = str(item).split()
            name, n = parts[0], int(parts[1]) if len(parts) > 1 else 1
        yaw, fwd, pitch = DSL.get(name, (0.0, 0.0, 0.0))
        for _ in range(n):
            a = torch.zeros(7)
            a[1] = yaw * 0.05
            a[3] = fwd * 0.12
            a[5] = pitch * 0.03
            out.append(a)
    return out


@torch.no_grad()
def rollout(image: torch.Tensor, actions: list[torch.Tensor], ckpt_path: str,
            steps: int = 8, cfg_scale: float = 4.0, device: str = "cuda",
            use_ema: bool = True, hist_frames: int = 4):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = make_model(cfg, device)
    sd = ck["ema"] if use_ema and "ema" in ck else ck["model"]
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in sd.items()})
    model.eval()

    tok = Tokenizer(cfg["data"]["vae"], device=device)
    z_init = tok.encode(image.to(device))
    hist = [z_init.clone() for _ in range(hist_frames)]
    frames = []

    for i, a in enumerate(actions):
        ab = a.view(1, 7).to(device)
        z = torch.randn_like(z_init)
        dt = 1.0 / steps
        for s in range(steps):
            t = torch.full((1,), s * dt, device=device)
            zt = z
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                v_cond = model(z_init, torch.stack(hist, dim=1), zt, t, ab,
                               torch.zeros(1, dtype=torch.long, device=device))
                if cfg_scale > 1.0:
                    v_null = model(z_init, torch.stack(hist, dim=1), zt, t, ab * 0,
                                   torch.zeros(1, dtype=torch.long, device=device))
                    v = v_cond + cfg_scale * (v_cond - v_null)
                else:
                    v = v_cond
            z = z + dt * v.float()
        hist = hist[1:] + [z.clone()]
        img = tok.decode(z)[0]
        frames.append(img.cpu())
    return frames
