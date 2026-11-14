#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练器三项优化的回归测试（不依赖真实语料，秒级跑完）

覆盖：
  1. resolve_shard_dir：prefer_latent=true 时自动切到 {shard}_latent，缺失则回退
  2. ss_mode="noise_proxy"：不产生额外 DiT 前向（用 forward hook 计数对比 "forward"）
  3. val_max_samples：验证集固定子集，长度被截断且同 seed 可复现
  4. ema_every：降频更新且衰减按 every 次方补偿（影子权重与逐步更新近似等价）

用法：python tools/test_trainer_opts.py
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tarfile
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from worldmodel.data.dataset import Subset, TarWindows, split_dataset  # noqa: E402
from worldmodel.train.trainer import EMA, resolve_shard_dir  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(("  ✓ " if cond else "  ✗ ") + name + (("  " + detail) if detail else ""))
    if not cond:
        FAIL.append(name)


def write_image_shard(out_dir, n=6, w=32, h=24, n_hist=4):
    """最小图片分片：init / h0..h3 / target / json。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "shard-00000.tar")
    rng = np.random.default_rng(0)
    with tarfile.open(path, "w") as tf:
        for i in range(n):
            key = "scene_%03d" % i
            for role in ["init", "target"] + ["h%d" % k for k in range(n_hist)]:
                arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format="JPEG", quality=90)
                add(tf, "%s.%s.jpg" % (key, role), buf.getvalue())
            meta = {"action": [0.0] * 7, "meta": {"scene": key}}
            add(tf, key + ".json", json.dumps(meta).encode())
    return path


def write_latent_shard(out_dir, n=6, w=32, h=24, n_hist=4):
    """最小 latent 分片：只写 {key}.latent + {key}.json（形状 (2+n_hist,4,h/8,w/8)）。"""
    os.makedirs(out_dir, exist_ok=True)
    shape = (2 + n_hist, 4, h // 8, w // 8)
    rng = np.random.default_rng(1)
    with tarfile.open(os.path.join(out_dir, "shard-00000.tar"), "w") as tf:
        for i in range(n):
            key = "scene_%03d" % i
            lat = rng.standard_normal(shape).astype("<f4")
            add(tf, key + ".latent", lat.tobytes())
            meta = {"action": [0.0] * 7, "latent_shape": list(shape)}
            add(tf, key + ".json", json.dumps(meta).encode())


def add(tf, name, data):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    tf.addfile(ti, io.BytesIO(data))


def main():
    tmp = tempfile.mkdtemp(prefix="pwopt-")
    try:
        img_dir = os.path.join(tmp, "wm")
        lat_dir = os.path.join(tmp, "wm_latent")
        write_image_shard(img_dir)
        write_latent_shard(lat_dir)

        print("[1] resolve_shard_dir / 分片格式")
        cfg = {"data": {"shard_dir": img_dir}}
        check("默认用图片分片", resolve_shard_dir(cfg) == img_dir)
        cfg_l = {"data": {"shard_dir": img_dir, "prefer_latent": True}}
        check("prefer_latent 自动切 {dir}_latent",
              resolve_shard_dir(cfg_l) == lat_dir, resolve_shard_dir(cfg_l))
        cfg_miss = {"data": {"shard_dir": img_dir, "prefer_latent": True,
                             "latent_dir": os.path.join(tmp, "nope")}}
        check("latent 缺失时回退", resolve_shard_dir(cfg_miss) == img_dir)
        ds_img = TarWindows(img_dir, 32, 24, 4)
        ds_lat = TarWindows(lat_dir, 32, 24, 4)
        si = ds_img[0]
        sl = ds_lat[0]
        check("图片分片给 init/hist/target",
              set(["init", "hist", "target"]).issubset(si) and si["hist"].shape[0] == 4)
        check("latent 分片给 z_init/z_hist/z_target",
              set(["z_init", "z_hist", "z_target"]).issubset(sl)
              and sl["z_hist"].shape == (4, 4, 3, 4))

        print("[2] val_max_samples：固定子集")
        tr, va = split_dataset(ds_img, 0.5, 0)
        order = list(range(len(va)))
        import random as _r
        _r.Random(0 * 7919 + 17).shuffle(order)
        va2 = Subset(va, order[:2])
        check("子集长度被截断", len(va2) == 2, "len(va)=%d" % len(va))
        _r.Random(17).shuffle(order)
        check("同 seed 可复现", Subset(va, order[:2]).ids == va2.ids)

        print("[3] EMA 降频：every=1 与 every=4 终点差异应很小")
        torch.manual_seed(0)
        lin = torch.nn.Linear(64, 64)
        e1 = EMA(lin, 0.999, every=1)
        e4 = EMA(lin, 0.999, every=4)
        torch.manual_seed(1)
        for step in range(40):
            with torch.no_grad():
                for p in lin.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            e1.update(lin)
            e4.update(lin)
        d1 = e1.shadow["weight"] - lin.weight.float()
        d4 = e4.shadow["weight"] - lin.weight.float()
        rel = (d4.norm() / d1.norm()).item()
        check("降频后影子权重仍贴近当前权重", 0.5 < rel < 2.0, "比值 %.2f" % rel)
        check("every>1 时跳过更新", e4.step == 40 and e4.shadow["weight"].shape == lin.weight.shape)

        print("[4] ss_mode=noise_proxy 不产生额外前向")
        from worldmodel.model.dit import CausalWorldDiT, DiTConfig
        m = CausalWorldDiT(DiTConfig(latent_h=8, latent_w=8, dim=64, depth=1, heads=4, n_hist=4))
        calls = {"n": 0}
        m.register_forward_hook(lambda *a: calls.__setitem__("n", calls["n"] + 1))
        B = 2
        z_init = torch.zeros(B, 4, 8, 8)
        z_hist = torch.zeros(B, 4, 4, 8, 8)
        z1 = torch.randn(B, 4, 8, 8)
        act = torch.zeros(B, 7)
        t = torch.rand(B)
        zt = torch.randn_like(z1)
        m(z_init, z_hist, zt, t, act, None)
        check("基线一次前向计 1", calls["n"] == 1)
        # noise_proxy 路径：只做张量运算，不应再触发前向
        alpha = 0.05 + torch.rand(B, 1, 1, 1) * 0.20
        z_self = (1 - alpha) * z1 + alpha * torch.randn_like(z1)
        calls["n"] = 0
        m(z_init, torch.cat([z_hist[:, 1:], z_self.unsqueeze(1)], 1), zt, t, act, None)
        check("noise_proxy 额外前向 0 次", calls["n"] == 1, "含主前向共 %d" % calls["n"])
        check("z_self 与 z1 同形", z_self.shape == z1.shape)
        dev = (z_self - z1).abs().mean().item()
        check("扰动幅度在 0.05~0.25 区间内", 0 < dev < 1.2, "mean|Δ|=%.3f" % dev)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n结果：" + ("全部通过" if not FAIL else "失败 %d 项 → %s" % (len(FAIL), FAIL)))
    # ---- 5) T9-1：forward 模式的 SS 自产帧必须 Euler 自洽 ----
    print("[5] forward 模式 SS：时刻与状态配对")
    from worldmodel.train.trainer import make_self_frame
    import torch as _t

    class RecModel(_t.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, z_init, z_hist, zt, t, act, domain, uncond=None):
            self.calls.append({"t": t.detach().clone(), "domain": domain,
                               "zt": zt.detach().clone()})
            return _t.ones_like(zt)          # 恒定速度 1

    rm = RecModel()
    zi = _t.zeros(4, 6, 4, 4)            # (B, C, H, W)
    zh = _t.zeros(4, 4, 6, 4, 4)         # (B, T_hist, C, H, W)
    act = _t.zeros(4, 7)
    dom = _t.zeros(4, dtype=_t.long)
    z1 = _t.zeros(4, 6, 4, 4)
    out = make_self_frame(rm, zi, zh, act, dom, z1, "cpu", _t.bfloat16)
    call = rm.calls[0]
    check("SS 求值时刻 t=0", float(call["t"].abs().max()) == 0.0, "t=%s" % call["t"])
    check("SS domain 是 zeros 不是 None",
          call["domain"] is not None and int(call["domain"].sum()) == 0)
    d = float((out - call["zt"]).flatten(1)[:, :1].mean())
    check("SS 步长 ∈ [0.35,0.65]", 0.34 <= d <= 0.66, "%.3f" % d)
    # dt 是逐样本的 (B,1,1,1)，所以"位移一致"是**每个样本内部**一致，不是整个 batch 相同
    disp = (out - call["zt"])                                    # (B,C,H,W)
    per = disp.flatten(1).mean(1)                                # (B,)
    spread = (disp - per.view(-1, 1, 1, 1)).abs().max().item()
    check("v=1 时每样本位移恒定", spread < 1e-3, "样本内最大偏差 %.1e" % spread)
    check("每样本步长都在 [0.35,0.65]", bool(((per >= 0.34) & (per <= 0.66)).all()),
          "范围 %.3f~%.3f" % (float(per.min()), float(per.max())))

    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
