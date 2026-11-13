"""训练循环：rectified flow + CFG dropout + 历史帧噪声增强 + EMA。"""
from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from worldmodel.data.dataset import TarWindows, collate, split_dataset
from worldmodel.model.dit import CausalWorldDiT, DiTConfig, count_params
from worldmodel.model.tokenizer import Tokenizer


class EMA:
    """带 warmup 的 EMA。

    固定 decay=0.999 在短训练里是灾难：300 步后影子权重仍有 74% 是初始随机值，
    rollout 拿它推理等于用了个没训过的模型。warmup 让早期紧跟当前权重。
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.max_decay = decay
        self.step = 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @property
    def decay(self):
        return min(self.max_decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.step += 1
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                s.copy_(v)

    def copy_to(self, model: torch.nn.Module):
        sd = model.state_dict()
        model.load_state_dict({k: self.shadow[k].to(sd[k].dtype) for k in sd})


def make_model(cfg: dict, device: str) -> CausalWorldDiT:
    mcfg = DiTConfig(
        latent_h=cfg["data"]["height"] // 8,
        latent_w=cfg["data"]["width"] // 8,
        dim=cfg["model"]["dim"], depth=cfg["model"]["depth"], heads=cfg["model"]["heads"],
        n_hist=cfg["data"]["n_hist"],
    )
    return CausalWorldDiT(mcfg).to(device)


def enable_fast_math() -> None:
    """TF32 + cuDNN benchmark：不开的话 fp32 conv 在 sm_120 上会走极慢的回退路径
    （实测 VAE 编码 6 帧从 10.4s 降到 0.3s，200 倍差距）。"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # benchmark=True 会在 sm_120 上对每个 conv 形状做算法搜索，首步要跑几分钟，关掉


def train(cfg: dict) -> None:
    enable_fast_math()
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = cfg["train"]["out"]
    os.makedirs(out_dir, exist_ok=True)

    # cache=False：缓存解码后的张量会把内存吃爆（5160×6 帧 ≈ 36GB），导致疯狂换页
    ds = TarWindows(cfg["data"]["shard_dir"], cfg["data"]["width"], cfg["data"]["height"],
                    cfg["data"]["n_hist"], cache=False,
                    action_scale=cfg["data"].get("action_scale"))
    tr, va = split_dataset(ds, cfg["data"].get("val_ratio", 0.05), seed)
    bs = cfg["train"]["batch_size"]
    dl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=0,
                    collate_fn=collate, drop_last=True)
    dl_va = DataLoader(va, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate)

    tok = Tokenizer(cfg["data"]["vae"], device=device)
    model = make_model(cfg, device)
    print("参数量 %.1fM  训练样本 %d  验证 %d" % (count_params(model) / 1e6, len(tr), len(va)))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            betas=(0.9, 0.95), weight_decay=0.01)
    total = cfg["train"]["steps"]
    warm = cfg["train"]["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) *
        (0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))))
    ema = EMA(model, cfg["train"].get("ema", 0.999))
    scaler_dtype = torch.bfloat16

    p_act = cfg["train"].get("drop_action", 0.1)
    p_init = cfg["train"].get("drop_init", 0.1)
    p_hist_noise = cfg["train"].get("hist_noise_prob", 0.5)
    hist_sigma = cfg["train"].get("hist_noise_sigma", 0.3)
    accum = cfg["train"].get("grad_accum", 1)

    step = 0
    resume = cfg["train"].get("resume")
    if resume:
        path = os.path.join(out_dir, "last.pt")
        if os.path.exists(path):
            ck = torch.load(path, map_location=device, weights_only=False)
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
            init = batch["init"].to(device, non_blocking=True)
            hist = batch["hist"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            act = batch["action"].to(device, non_blocking=True)
            B = init.shape[0]

            with torch.no_grad():
                z_init = tok.encode(init)
                z_hist = tok.encode(hist.reshape(-1, *hist.shape[2:])).reshape(
                    B, hist.shape[1], -1, *z_init.shape[2:])
                z1 = tok.encode(target)
                # 历史帧噪声增强（对抗自回归漂移）
                if p_hist_noise > 0:
                    sig = torch.rand(B, 1, 1, 1, 1, device=device) * hist_sigma
                    mask = (torch.rand(B, 1, 1, 1, 1, device=device) < p_hist_noise).float()
                    z_hist = z_hist + mask * sig * torch.randn_like(z_hist)

            _t2 = time.time()
            z0 = torch.randn_like(z1)
            t = torch.rand(B, device=device)
            zt = (1 - t.view(B, 1, 1, 1)) * z0 + t.view(B, 1, 1, 1) * z1
            v_target = z1 - z0

            drop_act = torch.rand(B, device=device) < p_act
            drop_init = torch.rand(B, device=device) < p_init
            domain = torch.zeros(B, dtype=torch.long, device=device)

            with torch.autocast("cuda", dtype=scaler_dtype, enabled=(device == "cuda")):
                v = model(z_init, z_hist, zt, t, act, domain, drop_act=drop_act, drop_init=drop_init)
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
            with torch.no_grad():
                vl = []
                for vb in dl_va:
                    vi = vb["init"].to(device); vh = vb["hist"].to(device)
                    vt = vb["target"].to(device); va_ = vb["action"].to(device)
                    zi = tok.encode(vi)
                    zh = tok.encode(vh.reshape(-1, *vh.shape[2:])).reshape(
                        vi.shape[0], vh.shape[1], -1, *zi.shape[2:])
                    z1 = tok.encode(vt)
                    z0 = torch.randn_like(z1)
                    tt = torch.rand(vi.shape[0], device=device)
                    zt = (1 - tt.view(-1, 1, 1, 1)) * z0 + tt.view(-1, 1, 1, 1) * z1
                    with torch.autocast("cuda", dtype=scaler_dtype, enabled=(device == "cuda")):
                        vv = model(zi, zh, zt, tt, va_, torch.zeros(vi.shape[0], dtype=torch.long, device=device))
                    vl.append(F.mse_loss(vv.float(), (z1 - z0).float()).item())
                print("  val loss %.4f" % (sum(vl) / max(1, len(vl))), flush=True)
            ck = {"model": model.state_dict(), "ema": ema.shadow, "cfg": cfg, "step": step}
            torch.save(ck, os.path.join(out_dir, "last.pt"))
            torch.save(ck, os.path.join(out_dir, "step-%06d.pt" % step))
            model.train()

    meta = {"steps": total, "params": count_params(model),
            "step_per_sec": round(total / (time.time() - t0), 2),
            "samples": len(tr), "seed": seed}
    json.dump(meta, open(os.path.join(out_dir, "train_meta.json"), "w"), indent=1)
    print("训练完成：" + json.dumps(meta, ensure_ascii=False))
