"""训练循环：rectified flow + CFG dropout + 历史帧噪声增强 + EMA。"""
from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from worldmodel.data.dataset import Subset, TarWindows, collate, split_dataset
from worldmodel.model.dit import CausalWorldDiT, DiTConfig, count_params
from worldmodel.model.tokenizer import Tokenizer


def raw_model(m: torch.nn.Module) -> torch.nn.Module:
    """torch.compile 包装后 state_dict 会带 '_orig_mod.' 前缀。

    EMA shadow、checkpoint 与续训必须始终操作裸模型，否则旧 last.pt 读不出来
    （B9：不改 checkpoint 格式）。
    """
    return getattr(m, "_orig_mod", m)


class EMA:
    """带 warmup 的 EMA。

    固定 decay=0.999 在短训练里是灾难：300 步后影子权重仍有 74% 是初始随机值，
    rollout 拿它推理等于用了个没训过的模型。warmup 让早期紧跟当前权重。

    `every > 1` 时每 N 步才更新一次，等价衰减用 decay**N 补偿
    （每步都更新 → 时间常数 1/(1-d)；每 N 步更新 → 1/(1-d**N)，两者对齐）。
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, every: int = 1):
        self.max_decay = decay
        self.every = max(1, int(every))
        self.step = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in raw_model(model).state_dict().items()}

    @property
    def decay(self):
        return min(self.max_decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.step += 1
        if self.every > 1 and (self.step % self.every) != 0:
            return
        d = self.decay ** self.every
        for k, v in raw_model(model).state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                s.copy_(v)

    def copy_to(self, model: torch.nn.Module):
        m = raw_model(model)
        sd = m.state_dict()
        m.load_state_dict({k: self.shadow[k].to(sd[k].dtype) for k in sd})


def make_model(cfg: dict, device: str) -> CausalWorldDiT:
    mcfg = DiTConfig(
        latent_h=cfg["data"]["height"] // 8,
        latent_w=cfg["data"]["width"] // 8,
        dim=cfg["model"]["dim"], depth=cfg["model"]["depth"], heads=cfg["model"]["heads"],
        n_hist=cfg["data"]["n_hist"],
    )
    # v1.26 吞吐开关全部 opt-in（默认 0/False）：不改权重形状、不改 checkpoint 键。
    return CausalWorldDiT(
        mcfg,
        pad_multiple=int(cfg["model"].get("pad_to_multiple", 0) or 0),
        activation_checkpointing=bool(cfg["train"].get("activation_checkpointing", False)),
    ).to(device)


@torch.no_grad()
def make_self_frame(model, z_init, z_hist, act, domain, z1, device,
                    scaler_dtype, dt_lo: float = 0.35, dt_hi: float = 0.65):
    """T9-1（v1.22）：scheduled sampling 的"自产帧"—— 从 x_0 出发的单步 Euler。

    自洽要求：速度必须在**当前时刻**求值。旧实现拿 t=0.35 处的速度作用在 x_0 上，
    z_self 落在错误位置（v 依赖 t 时落点偏 70%，实测），SS 教给模型的是错分布。
    抽成函数是为了能单测（断言传进模型的 t 全 0、domain 是 zeros 而不是 None）。
    """
    B = z1.shape[0]
    z_noise = torch.randn_like(z1)
    t_ss = torch.zeros(B, device=device)
    dt_ss = dt_lo + torch.rand(B, 1, 1, 1, device=device) * (dt_hi - dt_lo)
    with torch.autocast("cuda", dtype=scaler_dtype, enabled=(device == "cuda")):
        v_self = model(z_init, z_hist, z_noise, t_ss, act, domain)
    return z_noise + v_self.float() * dt_ss


def _augment_hist(z_hist: torch.Tensor, B: int, device: str,
                  p: float, sigma: float) -> torch.Tensor:
    """历史帧噪声增强（对抗自回归漂移）。z_hist 只是输入，不参与梯度。"""
    if p <= 0:
        return z_hist
    sig = torch.rand(B, 1, 1, 1, 1, device=device) * sigma
    mask = (torch.rand(B, 1, 1, 1, 1, device=device) < p).float()
    return z_hist + mask * sig * torch.randn_like(z_hist)


def enable_fast_math() -> None:
    """TF32 + cuDNN benchmark：不开的话 fp32 conv 在 sm_120 上会走极慢的回退路径
    （实测 VAE 编码 6 帧从 10.4s 降到 0.3s，200 倍差距）。"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # benchmark=True 会在 sm_120 上对每个 conv 形状做算法搜索，首步要跑几分钟，关掉


def resolve_shard_dir(cfg: dict) -> str:
    """选分片目录：`data.prefer_latent=true` 时优先用预编码 latent 分片。

    VAE 编码是冻结的、结果只跟图片有关，所以「离线编一次 → 训练直读 .latent」
    与「每步在线编码」在数学上完全等价，只是把 6 次 encode/样本 从训练循环里搬走。
    没有 latent 分片时退回原目录并明确打印，不静默改变行为（默认 false）。
    """
    base = cfg["data"]["shard_dir"]
    if not cfg["data"].get("prefer_latent"):
        return base
    cand = cfg["data"].get("latent_dir") or (base.rstrip("/\\") + "_latent")
    if os.path.isdir(cand) and any(
            f.startswith("shard-") and f.endswith(".tar") for f in os.listdir(cand)):
        print("[data] 使用预编码 latent 分片：%s（训练侧跳过 VAE encode）" % cand, flush=True)
        return cand
    print("[data] prefer_latent=true 但没找到 %s 的分片，回退在线编码" % cand, flush=True)
    return base


def coerce_train_cfg(cfg: dict) -> dict:
    """超参强制转数值：PyYAML 1.1 只认「带小数点的指数」，`lr: 1e-3` 会被读成字符串，
    然后在 AdamW 里抛 `TypeError: '<=' not supported between float and str`——
    错误位置离原因隔了半个训练脚本。这里统一转一次，报错就报在配置上。
    """
    T = cfg.get("train", {})
    for k in ("lr", "clip", "ema", "drop_cond", "hist_noise_prob", "hist_noise_sigma",
              "scheduled_sampling"):
        if T.get(k) is not None:
            T[k] = float(T[k])
    for k in ("steps", "batch_size", "grad_accum", "warmup", "log_every", "val_every",
              "val_max_samples", "ema_every", "num_workers", "prefetch_factor"):
        if T.get(k) is not None:
            T[k] = int(T[k])
    # v1.26 新增开关：YAML 里写成 "true" 字符串时也按布尔处理
    for k in ("pin_memory", "persistent_workers", "activation_checkpointing",
              "compile", "fused_adam"):
        v = T.get(k)
        if isinstance(v, str):
            T[k] = v.strip().lower() in ("1", "true", "yes", "on")
    M = cfg.get("model", {})
    if M.get("pad_to_multiple") is not None:
        M["pad_to_multiple"] = int(M["pad_to_multiple"])
    return cfg


def train(cfg: dict) -> None:
    cfg = coerce_train_cfg(cfg)
    enable_fast_math()
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = cfg["train"]["out"]
    os.makedirs(out_dir, exist_ok=True)

    # cache=False：缓存解码后的张量会把内存吃爆（5160×6 帧 ≈ 36GB），导致疯狂换页
    shard_dir = resolve_shard_dir(cfg)
    ds = TarWindows(shard_dir, cfg["data"]["width"], cfg["data"]["height"],
                    cfg["data"]["n_hist"], cache=False,
                    action_scale=cfg["data"].get("action_scale"))
    tr, va = split_dataset(ds, cfg["data"].get("val_ratio", 0.05), seed)
    bs = cfg["train"]["batch_size"]
    # v1.26 opt-in：多进程预取把 data 阶段（作者机器实测 0.11s/step）摊到后台。
    # 默认 num_workers=0 与旧行为逐字一致；pin_memory 只在 CUDA 下有意义。
    nw = int(cfg["train"].get("num_workers", 0) or 0)
    loader_kw = dict(
        batch_size=bs, collate_fn=collate, num_workers=nw,
        pin_memory=bool(cfg["train"].get("pin_memory", True)) and device == "cuda",
        persistent_workers=bool(cfg["train"].get("persistent_workers", False)) and nw > 0,
    )
    if nw > 0:
        loader_kw["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 2) or 2)
    dl = DataLoader(tr, shuffle=True, drop_last=True, **loader_kw)
    dl_va = DataLoader(va, shuffle=False, **loader_kw)

    # VAE 惰性加载：全量使用 .latent 预编码分片时，训练机完全不需要 VAE 权重
    tok = None

    def get_tok():
        nonlocal tok
        if tok is None:
            tok = Tokenizer(cfg["data"]["vae"], device=device)
        return tok

    model = make_model(cfg, device)
    # torch.compile：融合 adaLN 的逐元素算子。默认关闭 —— 本机无 CUDA/torch，
    # 未实测数值与吞吐；reduce-overhead 还需固定的输入形状，训练机需自行 A/B。
    if device == "cuda" and cfg["train"].get("compile", False):
        try:
            mode = str(cfg["train"].get("compile_mode", "reduce-overhead"))
            model = torch.compile(model, mode=mode)
            print("[train] torch.compile 已启用（mode=%s）" % mode, flush=True)
        except Exception as e:                       # 老版本 torch / 不支持的后端
            print("[train] torch.compile 不可用，回退 eager：%s" % e, flush=True)
    print("参数量 %.1fM  训练样本 %d  验证 %d" % (count_params(model) / 1e6, len(tr), len(va)))

    opt_kw = dict(lr=cfg["train"]["lr"], betas=(0.9, 0.95), weight_decay=0.01)
    if device == "cuda" and cfg["train"].get("fused_adam", True):
        try:
            # fused AdamW 与逐参数 AdamW 数学等价，只是把上百次小 kernel 合成一次
            opt = torch.optim.AdamW(model.parameters(), fused=True, **opt_kw)
        except TypeError:                            # 旧 torch 没有 fused 参数
            opt = torch.optim.AdamW(model.parameters(), **opt_kw)
    else:
        opt = torch.optim.AdamW(model.parameters(), **opt_kw)
    total = cfg["train"]["steps"]
    warm = cfg["train"]["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) *
        (0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))))
    ema = EMA(model, cfg["train"].get("ema", 0.999), cfg["train"].get("ema_every", 1))
    scaler_dtype = torch.bfloat16

    # 验证集子集化：val_every 一到就遍历全量 val（10% 数据），前向开销能顶 5 个训练 step。
    # 固定 seed 抽固定样本 + 固定 (t, z0)，val loss 曲线方差更小，前后可比。
    val_max = int(cfg["train"].get("val_max_samples", 0) or 0)
    # T9-2（v1.22）：注释一直说"固定 (t, z0)"，但验证循环里每次都 randn/rand 重抽，
    #   val loss 的方差里有相当一部分来自噪声抽样而不是模型进步。
    #   这里给一个**专用 generator**，每次评测前重置到同一 seed：
    #   批顺序固定（shuffle=False）+ 形状固定 ⇒ (t,z0) 每次完全一致。
    val_gen = torch.Generator(device="cpu")
    val_seed = int(seed) * 104729 + 3
    if val_max > 0 and len(va) > val_max:
        import random as _random
        order = list(range(len(va)))
        _random.Random(int(seed) * 7919 + 17).shuffle(order)
        va = Subset(va, order[:val_max])
        dl_va = DataLoader(va, shuffle=False, **loader_kw)
        print("[val] 固定子集 %d / %d 条" % (val_max, len(ds) - len(tr)), flush=True)

    # 联合条件 dropout：训练与 CFG 的无条件分支共用同一条前向路径。
    # 旧的 drop_action / drop_init 是两次独立 10% dropout，而推理只把 action
    # 置零，三条路径互不等价（act_embed 有 bias，且 init/domain 没被丢）。
    p_uncond = cfg["train"].get("drop_cond", cfg["train"].get("drop_action", 0.1))
    p_hist_noise = cfg["train"].get("hist_noise_prob", 0.5)
    hist_sigma = cfg["train"].get("hist_noise_sigma", 0.3)
    accum = cfg["train"].get("grad_accum", 1)
    # "forward"（默认，旧行为：多跑一次前向造自产帧）/ "noise_proxy"（零额外前向）
    ss_mode = cfg["train"].get("ss_mode", "forward")

    step = 0
    resume = cfg["train"].get("resume")
    if resume:
        path = os.path.join(out_dir, "last.pt")
        if os.path.exists(path):
            ck = torch.load(path, map_location=device, weights_only=False)
            if hasattr(model, "_orig_mod"):
                raw_model(model).load_state_dict(ck["model"])
            else:
                model.load_state_dict(ck["model"])
            if "ema" in ck:
                ema.shadow = {k: v.to(device).float() for k, v in ck["ema"].items()}
                ema.step = ck.get("step", 0)
            step = int(ck.get("step", 0))
            print("从 %s 续训，step=%d" % (path, step), flush=True)
        else:
            print("没有找到可续训的 checkpoint，从头开始", flush=True)

    t0 = time.time()
    it = iter(dl)
    model.train()
    while step < total:
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(accum):
            _t0 = time.time()
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)

            _t1 = time.time()
            act = batch["action"].to(device, non_blocking=True)
            if "z_init" in batch:
                # 预编码分片：VAE 编码在离线阶段完成，训练时不再每步重复编码
                z_init = batch["z_init"].to(device, non_blocking=True)
                z_hist = batch["z_hist"].to(device, non_blocking=True)
                z1 = batch["z_target"].to(device, non_blocking=True)
            else:
                init = batch["init"].to(device, non_blocking=True)
                hist = batch["hist"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                with torch.no_grad():
                    z_init = get_tok().encode(init)
                    z_hist = get_tok().encode(hist.reshape(-1, *hist.shape[2:])).reshape(
                        init.shape[0], hist.shape[1], -1, *z_init.shape[2:])
                    z1 = get_tok().encode(target)
            B = z1.shape[0]
            z_hist = _augment_hist(z_hist, B, device, p_hist_noise, hist_sigma)
            # T9-1（v1.22）：domain 必须在 SS 前向之前就定义好 ——
            #   旧代码 SS 里传 `domain=None`，与训练主路径的 zeros 不是同一条路，
            #   而且 DiT 里 None 会被当成"没有 domain 嵌入"，等于第三种前向。
            domain = torch.zeros(B, dtype=torch.long, device=device)

            _t2 = time.time()
            # ---- scheduled sampling：消除暴露偏差 ----
            # 训练时历史帧永远是"真值"，推理时历史帧是自己产的 → 误差按帧累积（rollout 塌黑）。
            # 这里以概率 p 用模型自产的一帧替换最近的历史帧。
            p_ss = cfg["train"].get("scheduled_sampling", 0.0)
            if p_ss > 0 and torch.rand(1).item() < p_ss:
                if ss_mode == "noise_proxy":
                    # 零成本代理：不再多跑一次 DiT 前向（p=0.5 时那是 +16.7% MACs），
                    # 直接把真值 z1 沿 flow 轨迹往噪声方向推一点当"自产帧"。
                    # 单步从纯噪声出发的 z_self 本来就离流形极远，既贵又不真。
                    alpha = 0.05 + torch.rand(B, 1, 1, 1, device=device) * 0.20
                    z_self = (1 - alpha) * z1 + alpha * torch.randn_like(z1)
                else:
                    # T9-1：单步 Euler 自洽版（时刻与状态必须配对），实现见 make_self_frame
                    with torch.no_grad():
                        z_self = make_self_frame(model, z_init, z_hist, act, domain, z1,
                                                 device, scaler_dtype)
                z_hist = torch.cat([z_hist[:, 1:], z_self.unsqueeze(1)], dim=1)

            z0 = torch.randn_like(z1)
            t = torch.rand(B, device=device)
            zt = (1 - t.view(B, 1, 1, 1)) * z0 + t.view(B, 1, 1, 1) * z1
            v_target = z1 - z0

            uncond = torch.rand(B, device=device) < p_uncond

            with torch.autocast("cuda", dtype=scaler_dtype, enabled=(device == "cuda")):
                v = model(z_init, z_hist, zt, t, act, domain, uncond=uncond)
                loss = F.mse_loss(v.float(), v_target.float()) / accum
            _t3 = time.time()
            loss.backward()
            loss_acc += loss.item()
            if step < 3:
                print("  [prof] data %.2fs vae %.2fs fwd %.2fs bwd %.2fs" %
                      (_t1 - _t0, _t2 - _t1, _t3 - _t2, time.time() - _t3), flush=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip"])
        opt.step()
        sched.step()
        ema.update(model)
        step += 1

        if step % cfg["train"]["log_every"] == 0:
            sps = step / (time.time() - t0)
            print("[%5d/%d] loss %.4f  lr %.2e  %.2f step/s" %
                  (step, total, loss_acc, sched.get_last_lr()[0], sps), flush=True)

        if step % cfg["train"]["val_every"] == 0 or step == total:
            model.eval()
            # T9-2：每次评测前把专用 generator 重置到同一 seed，
            # 于是每轮评测用的 (t, z0) 序列完全相同，曲线只反映模型变化。
            val_gen.manual_seed(val_seed)
            # val loss 必须与 rollout 同源：临时换上 EMA 权重，评估完恢复训练权重。
            # 旧代码用非 EMA 模型算 val，而 rollout 默认加载 EMA，指标不同源。
            train_sd = {k: v.detach().clone() for k, v in raw_model(model).state_dict().items()}
            try:
                ema.copy_to(model)
                with torch.no_grad():
                    vl = []
                    for vb in dl_va:
                        va_ = vb["action"].to(device)
                        if "z_init" in vb:
                            zi = vb["z_init"].to(device)
                            zh = vb["z_hist"].to(device)
                            z1 = vb["z_target"].to(device)
                        else:
                            vi = vb["init"].to(device); vh = vb["hist"].to(device)
                            vt = vb["target"].to(device)
                            zi = get_tok().encode(vi)
                            zh = get_tok().encode(vh.reshape(-1, *vh.shape[2:])).reshape(
                                vi.shape[0], vh.shape[1], -1, *zi.shape[2:])
                            z1 = get_tok().encode(vt)
                        Bv = z1.shape[0]
                        z0 = torch.randn(z1.shape, generator=val_gen).to(device)
                        tt = torch.rand(Bv, generator=val_gen).to(device)
                        zt = (1 - tt.view(-1, 1, 1, 1)) * z0 + tt.view(-1, 1, 1, 1) * z1
                        with torch.autocast("cuda", dtype=scaler_dtype, enabled=(device == "cuda")):
                            vv = model(zi, zh, zt, tt, va_,
                                       torch.zeros(Bv, dtype=torch.long, device=device))
                        vl.append(F.mse_loss(vv.float(), (z1 - z0).float()).item())
                    print("  val loss %.4f (EMA)" % (sum(vl) / max(1, len(vl))), flush=True)
            finally:
                raw_model(model).load_state_dict(train_sd)
            ck = {"model": raw_model(model).state_dict(), "ema": ema.shadow, "cfg": cfg, "step": step}
            torch.save(ck, os.path.join(out_dir, "last.pt"))
            torch.save(ck, os.path.join(out_dir, "step-%06d.pt" % step))
            model.train()

    meta = {"steps": total, "params": count_params(model),
            "step_per_sec": round(total / (time.time() - t0), 2),
            "samples": len(tr), "seed": seed}
    json.dump(meta, open(os.path.join(out_dir, "train_meta.json"), "w"), indent=1)
    print("训练完成：" + json.dumps(meta, ensure_ascii=False))
