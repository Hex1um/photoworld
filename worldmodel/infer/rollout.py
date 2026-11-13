"""自回归 rollout：初始图 + 动作序列 → 视频帧。

单步采样（Euler 中点）：
    t_s = (s + 0.5) * dt
    x_{t+dt} = x_t + dt * v(x_t, t_s)
    t 从 0（纯噪声）走到 1（数据）；CFG：
    v = v_cond + w * (v_cond - v_uncond)

无条件分支必须走 `uncond=True`（与训练侧联合 dropout 同一路径），
不能再只把 action 置零：act_embed 有 bias，且训练还同时丢了 init/domain。
"""
from __future__ import annotations

import numpy as np
import torch

from worldmodel.action_codec import dsl_action, parse_spec
from worldmodel.model.dit import DiTConfig
from worldmodel.train.trainer import make_model
from worldmodel.model.tokenizer import Tokenizer

# 可解析的动作名，与 action_codec.DSL_NAMES 保持一致
DSL_ACTIONS = ("forward", "back", "turn_left", "turn_right",
               "look_up", "look_down", "idle")


def parse_actions(spec, *, speed=None, move_step=None,
                  yaw_step=None, pitch_step=None, fov=None) -> list[torch.Tensor]:
    """['forward 20', 'turn_left 15'] → 每帧一个 **7 维**动作张量。

    动作由 `worldmodel.action_codec.dsl_action` 生成，与训练标签共用同一个
    `rel_action`：轴角 3 + 相机系平移 3（按速度归一化）+ logFOV 1。
    旧实现只填 a[1]/a[3]/a[5] 三个 DSL 标量，维度与符号都是错的。

    可选参数用于对齐具体数据集的动作幅度；默认取合成轨迹的典型幅度。
    """
    kw = {}
    for key, val in (("speed", speed), ("move_step", move_step),
                     ("yaw_step", yaw_step), ("pitch_step", pitch_step),
                     ("fov", fov)):
        if val is not None:
            kw[key] = float(val)
    out = []
    for name, n in parse_spec(spec):
        a = torch.from_numpy(np.asarray(dsl_action(name, **kw), dtype=np.float32))
        for _ in range(n):
            out.append(a.clone())
    return out


_CACHE = {}


def _load(ckpt_path: str, device: str, use_ema: bool):
    """按 checkpoint 记忆模型与 VAE：评测要跑十几次 rollout，每次重载要白等 15s。"""
    key = (ckpt_path, device, use_ema)
    if key in _CACHE:
        return _CACHE[key]
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = make_model(cfg, device)
    sd = ck["ema"] if use_ema and "ema" in ck else ck["model"]
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in sd.items()})
    model.eval()
    tok = Tokenizer(cfg["data"]["vae"], device=device)
    _CACHE[key] = (model, tok, cfg)
    return _CACHE[key]


@torch.no_grad()
def rollout(image: torch.Tensor, actions: list[torch.Tensor], ckpt_path: str,
            steps: int = 8, cfg_scale: float = 4.0, device: str = "cuda",
            use_ema: bool = True, hist_frames: int = 4):
    model, tok, cfg = _load(ckpt_path, device, use_ema)
    z_init = tok.encode(image.to(device))
    hist = [z_init.clone() for _ in range(hist_frames)]
    frames = []

    for i, a in enumerate(actions):
        ab = a.view(1, 7).to(device)
        z = torch.randn_like(z_init)
        dt = 1.0 / steps
        for s in range(steps):
            # 中点代替左端点：最后一步 t 从 0.875 抬到 0.9375，更贴近积分区间末端
            t = torch.full((1,), (s + 0.5) * dt, device=device)
            hb = torch.stack(hist, dim=1)
            dom = torch.zeros(1, dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                v_cond = model(z_init, hb, z, t, ab, dom)
                if cfg_scale > 1.0:
                    v_null = model(z_init, hb, z, t, ab, dom, uncond=True)
                    v = v_cond + cfg_scale * (v_cond - v_null)
                else:
                    v = v_cond
            z = z + dt * v.float()
        hist = hist[1:] + [z.clone()]
        img = tok.decode(z)[0]
        frames.append(img.cpu())
    return frames
