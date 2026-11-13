"""VAE tokenizer（冻结）。默认 stabilityai/sd-vae-ft-mse，4 通道潜变量。"""
from __future__ import annotations

import torch

SCALING = 0.18215


class Tokenizer:
    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse", device: str = "cuda"):
        from diffusers import AutoencoderKL
        self.device = device
        # 优先用本地缓存：每次联网校验会被 HF 镜像 429 限流，白等 100s+
        vae = None
        for kwargs in ({"local_files_only": True},):   # 只走本地缓存，避免 429 限流卡死
            try:
                vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=torch.float32, **kwargs)
                break
            except Exception:
                continue
        if vae is None:
            raise RuntimeError("VAE 权重不可用：请先 HF_ENDPOINT=https://hf-mirror.com 联网下载一次")
        self.vae = vae.to(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.scale = SCALING

    @property
    def downsample(self) -> int:
        return 8

    @torch.no_grad()
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B,3,H,W) in [-1,1] → (B,4,H/8,W/8)"""
        return self.vae.encode(img).latent_dist.mode() * self.scale

    @torch.no_grad()
    def decode(self, lat: torch.Tensor, chunk: int = 8) -> torch.Tensor:
        """lat: (B,4,h,w) → (B,3,H,W) in [-1,1]；分块解码省显存。"""
        outs = []
        for i in range(0, lat.shape[0], chunk):
            outs.append(self.vae.decode(lat[i:i + chunk] / self.scale).sample)
        return torch.cat(outs, dim=0).clamp(-1, 1)
