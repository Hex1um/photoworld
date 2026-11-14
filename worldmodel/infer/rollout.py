"""自回归 rollout：初始图 + 动作序列 → 视频帧。

单步采样（Euler 中点）：
    t_s = (s + 0.5) * dt
    x_{t+dt} = x_t + dt * v(x_t, t_s)
    t 从 0（纯噪声）走到 1（数据）；CFG（标准式）：
    v = v_null + (1 + w) * (v_cond - v_null)
    其中 w 是 `cfg_scale` 参数：w=0 等价于关闭引导；w=4 实际引导强度 s=5。

无条件分支必须走 `uncond=True`（与训练侧联合 dropout 同一路径），
不能再只把 action 置零：act_embed 有 bias，且训练还同时丢了 init/domain。
"""
from __future__ import annotations

import numpy as np
import torch

from worldmodel.action_codec import apply_action_scale, dsl_action, parse_spec
from worldmodel.ckpt_meta import checkpoint_warning, resolve_hist_frames
from worldmodel.model.dit import DiTConfig
from worldmodel.train.trainer import make_model
from worldmodel.model.tokenizer import Tokenizer

# 可解析的动作名，与 action_codec.DSL_NAMES 保持一致
DSL_ACTIONS = ("forward", "back", "turn_left", "turn_right",
               "look_up", "look_down", "idle")


def parse_actions(spec, *, speed=None, move_step=None,
                  yaw_step=None, pitch_step=None, fov=None,
                  action_scale=None) -> list[torch.Tensor]:
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
    # T6（v1.22）：**训练侧的 action_scale 必须在推理侧同样乘一次**。
    #   训练时 dataset.py 会对 rel_action 再乘一次 action_scale（标量或 7 维向量），
    #   而推理侧旧实现在这里直接返回未缩放的 dsl_action —— 幅度差 2.1~30×，
    #   等于让模型看"几乎不动的动作"，rollout 自然学不到位移。
    out = []
    for name, n in parse_spec(spec):
        # 逐维同序相乘（与 dataset 侧一致），逻辑在 action_codec 里以便无 torch 也能测
        v = apply_action_scale(dsl_action(name, **kw), action_scale)
        a = torch.from_numpy(v.astype(np.float32, copy=False))
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
    # G3（v1.22）：DiT 输出切片修复之前的旧 checkpoint 与现在的隐变量切法不一致，
    #   拿它续训/评测会**静默错位**（不报错，只是画面对不上）。
    #   这里给出显式警告，并且允许用 cfg 里的标记覆盖（新档写 sliced=True）。
    msg = checkpoint_warning(cfg)
    if msg:
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    model = make_model(cfg, device)
    sd = ck["ema"] if use_ema and "ema" in ck else ck["model"]
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in sd.items()})
    model.eval()
    tok = Tokenizer(cfg["data"]["vae"], device=device)
    _CACHE[key] = (model, tok, cfg)
    return _CACHE[key]


@torch.no_grad()
def action_scale_of(ckpt_path: str, device: str = "cpu") -> float | list | None:
    """读出 checkpoint 训练时用的 action_scale（T6）。没有就返回 None。"""
    try:
        _, _, cfg = _load(ckpt_path, device, True)
    except Exception:
        return None
    return cfg.get("data", {}).get("action_scale")


def rollout(image: torch.Tensor, actions: list[torch.Tensor], ckpt_path: str,
            steps: int = 8, cfg_scale: float = 4.0, device: str = "cuda",
            use_ema: bool = True, hist_frames: int = 4,
            solver: str = "euler", cfg_interval: float = 1.0,
            batch_cfg: bool = True, stats: dict | None = None):
    """自回归生成。

    v1.26 三项推理优化，默认行为与 8 步中点 Euler × 串行 CFG 数学等价：

    * ``batch_cfg=True``（默认）：把 v_cond / v_null 沿 batch 拼成 B=2 走一次
      DiT 前向，替代串行两次 model(...)。权重只读取一次、kernel 启动减半，
      uncond 仍走同一个 ``uncond`` 参数（[False, True]），CFG 语义不变。
      A/B 用 ``batch_cfg=False`` 回到旧串行路径。
    * ``solver="heun"``（opt-in）：4 步 Heun 预测-校正，每步 2 次评估；
      配合 batch 合并后 4 步 = 8 次单样本等效前向（旧 8 步 Euler = 16 次）。
    * ``cfg_interval``（opt-in，默认 1.0）：仅前 x 比例的步数启用 CFG，
      末期只算 v_cond（B=1）。会改变采样轨迹，必须跑 scripts/eval.py 验证 PSNR。

    ``stats`` 传入空 dict 可收每帧的模型调用数与等效单样本前向数。
    本机无 torch/CUDA，这些次数是解析计数，不是实测耗时。
    """
    model, tok, cfg = _load(ckpt_path, device, use_ema)
    # T9-3（v1.22）：hist_frames 必须与 checkpoint 训练时的历史长度一致。
    #   训练用 N 帧历史、推理给 4 帧，位置编码/因果掩码的时序长度就对不上，
    #   表现是"能跑但动作响应变糊"。以前只取函数默认值，没人校验。
    hist_frames, hist_warn = resolve_hist_frames(cfg, hist_frames)
    if hist_warn:
        import warnings
        warnings.warn(hist_warn, RuntimeWarning, stacklevel=2)
    z_init = tok.encode(image.to(device))
    hist = [z_init.clone() for _ in range(hist_frames)]
    frames = []
    solver = str(solver or "euler").lower()
    if solver not in ("euler", "heun"):
        raise ValueError("solver 只能是 euler / heun，收到 %r" % solver)
    interval = max(0.0, min(1.0, float(cfg_interval)))
    calls = 0        # 实际 model(...) 调用次数
    equiv = 0        # 折算成 B=1 前向次数（B=2 记 2）

    for i, a in enumerate(actions):
        ab = a.view(1, 7).to(device)
        z = torch.randn_like(z_init)
        dt = 1.0 / steps
        hb = torch.stack(hist, dim=1)
        dom = torch.zeros(1, dtype=torch.long, device=device)

        def velocity(zz, tt, want_cfg):
            nonlocal calls, equiv
            ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))
            if not want_cfg or cfg_scale <= 1.0:
                calls += 1
                equiv += 1
                with ac:
                    return model(z_init, hb, zz, tt, ab, dom).float()

            if batch_cfg:
                # 一次 B=2 前向取代两次 B=1：uncond 逐样本给 [False, True]
                calls += 1
                equiv += 2
                u2 = torch.tensor([False, True], device=device)
                with ac:
                    vv = model(
                        torch.cat([z_init, z_init], dim=0),
                        torch.cat([hb, hb], dim=0),
                        torch.cat([zz, zz], dim=0),
                        torch.cat([tt, tt], dim=0),
                        torch.cat([ab, ab], dim=0),
                        torch.cat([dom, dom], dim=0),
                        uncond=u2,
                    ).float()
                vc, vn = vv[:1], vv[1:]
                # T9-4（v1.22）：统一到**标准 CFG 形式** v = v_null + s·(v_cond - v_null)，
                #   s = 1 + cfg_scale。与旧写法 v_cond + w·(v_cond - v_null) 数学等价
                #   （展开后 s = 1 + w），但语义不会被误读成 s=w。
                s_guid = 1.0 + float(cfg_scale)
                return vn + s_guid * (vc - vn)

            # A/B 回退：保留串行 CFG 路径，uncond=True 与训练侧同源
            calls += 2
            equiv += 2
            with ac:
                v_cond = model(z_init, hb, zz, tt, ab, dom)
                v_null = model(z_init, hb, zz, tt, ab, dom, uncond=True)
            v_cond, v_null = v_cond.float(), v_null.float()
            s_guid = 1.0 + float(cfg_scale)
            return v_null + s_guid * (v_cond - v_null)

        for s in range(steps):
            # 中点代替左端点：最后一步 t 从 0.875 抬到 0.9375，更贴近积分区间末端
            t = torch.full((1,), (s + 0.5) * dt, device=device)
            want_cfg = ((s + 0.5) / steps) <= interval
            v1 = velocity(z, t, want_cfg)
            if solver == "heun":
                z_pred = z + dt * v1
                t2 = torch.full((1,), min(1.0, (s + 1.5) * dt), device=device)
                v2 = velocity(z_pred, t2, want_cfg)
                z = z + dt * 0.5 * (v1 + v2)
            else:
                z = z + dt * v1
        hist = hist[1:] + [z.clone()]
        img = tok.decode(z)[0]
        frames.append(img.cpu())

    if stats is not None:
        n = max(1, len(frames))
        stats.update({
            "steps": steps, "solver": solver, "cfg_interval": interval,
            "batch_cfg": bool(batch_cfg), "frames": len(frames),
            "model_calls": calls, "equivalent_forwards": equiv,
            "model_calls_per_frame": calls / n,
            "equivalent_forwards_per_frame": equiv / n,
        })
    return frames
