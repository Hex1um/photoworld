#!/usr/bin/env python
"""训练循环冒烟：打桩数据 + 微型 DiT，真跑 2 步（含 SS 与 val）。

为什么需要它：本机没有 data/wm 分片，端到端训练跑不起来；但 T9-1（SS Euler 自洽）
和 T9-2（val 固定 generator）都改在**训练循环内部**，只做函数级单测不足以证明
"把改动放进循环里还能跑"。这里用打桩数据集 + 假 tokenizer + 微型真模型，
把 train() 完整跑两步，能抓到形状/作用域/分支类错误。

    python tools/test_train_smoke.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from worldmodel.train import trainer as T  # noqa: E402

C = 4                       # SD VAE 通道数
IMG_H, IMG_W = 64, 64       # 假图片边长
H, W = IMG_H // 8, IMG_W // 8   # latent 形状（VAE 下采样 8×）→ 8×8


class FakeTok:
    """假 tokenizer：不加载 VAE，直接造固定形状 latent。"""

    def encode(self, x):
        n = x.shape[0]
        return torch.zeros(n, C, H, W, device=x.device, dtype=torch.float32)

    def decode(self, z):
        return torch.zeros(z.shape[0], 3, IMG_H, IMG_W, device=z.device)


class FakeDS:
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return {
            "init": torch.zeros(3, IMG_H, IMG_W),
            "hist": torch.zeros(2, 3, IMG_H, IMG_W),
            "target": torch.zeros(3, IMG_H, IMG_W),
            "action": torch.zeros(7),
        }


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="pw-train-smoke-")
    # ---- 打桩 ----
    T.TarWindows = lambda *a, **k: FakeDS()
    # 必须返回"可索引的数据集"，返回 index 列表会让 DataLoader 吐出 int
    T.split_dataset = lambda ds, ratio, seed: (ds, ds)
    T.collate = lambda batch: {
        "init": torch.stack([b["init"] for b in batch]),
        "hist": torch.stack([b["hist"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
    }
    T.Tokenizer = lambda *a, **k: FakeTok()

    cfg = {
        "seed": 0,
        "data": {"shard_dir": tmp, "width": IMG_W, "height": IMG_H, "n_hist": 2,
                 "vae": "fake", "val_ratio": 0.5, "action_scale": 2.0},
        "model": {"dim": 64, "depth": 2, "heads": 4},
        "train": {"steps": 2, "batch_size": 2, "grad_accum": 1, "clip": 1.0,
                  "log_every": 1, "val_every": 1, "warmup": 1, "lr": 1e-3,
                  "out": os.path.join(tmp, "run"), "drop_cond": 0.1,
                  "scheduled_sampling": 1.0,   # 强制走 SS 分支
                  "ss_mode": "forward",        # T9-1 改的正是这条
                  "val_max_samples": 2, "ema": 0.9, "ema_every": 1},
    }
    try:
        T.train(cfg)
    except Exception as e:                     # noqa: BLE001
        print("FAIL  训练循环抛异常：%s: %s" % (type(e).__name__, e))
        import traceback
        traceback.print_exc()
        return 3

    ck = os.path.join(tmp, "run", "last.pt")
    ok = os.path.exists(ck)
    print("%s  训练循环 2 步完成（SS=forward 路径 + val），checkpoint %s"
          % ("PASS" if ok else "FAIL", "已写出" if ok else "缺失"))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
