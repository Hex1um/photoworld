#!/usr/bin/env python3
"""v2 深度前端：GPU 上的高分辨率单目深度 + 图像引导上采样。

为什么要换掉旧的 518×518 小模型链路（`pipeline/depth.py`）：

  旧链路把整张照片压到 518×518 再推理，相对原图（常见 1500–2000 宽）**深度分辨率
  只有图像的 1/3~1/4**；放大回原图用的是最近邻/双立方，于是：
    * 树枝、草丛、栏杆这类细结构没有各自的深度 → 整片塌到同一个深度，
      在几何里变成一条"横带"（REVIEW.md 里 08 秋叶那张整体塌成地毯就是这个）；
    * 物体轮廓被平均掉，剪影不锐利，侧面看是"糊边"。

  这里做两件事：
    1. 用 Depth-Anything-V2-**Large**（ViT-L，Apache-2.0）在 GPU 上跑到接近原图的分辨率
       （长边默认 1540，14 的整数倍）——参数量是老 Small 的 6 倍，细节本身更好；
    2. 残差 **guided filter** 上采样：以原图为引导，把深度边缘吸附到图像边缘上
       （He et al. 2010 的快速引导滤波，积分图等价实现，O(N) 可分离）。
       双立方上采样会把深度边缘糊开 3~5 像素，引导滤波之后边缘与原图对齐。

输出统一为 **视差口径**（大 = 近），并归一化到稳定区间，供 `geom.calibrate_scale`
的 z = A·(r+b)^(-p) 使用。归一化必须做：不同深度模型输出的数值范围完全不同，
旧链路直接喂原始值，标定参数 A 就在不同模型间漂移。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
# 归一化后的视差区间：下界不能取 0（幂律 z=A·r^-p 在 r→0 处发散），
# 上界 1.0 对应"最近"。
# **分位窗口不能窄**：先前用 2/98，最近的那片前景（雪山图左下角的雪脊盘山公路）
# 只占 2% 像素，被整块削到 1.0 → 几何塌成 2m 的平板、纹理沿竖向拉成竖条。
# 实测 2/98 时底部 15% 行里有 12.5% 像素饱和；0.5/99.85 只裁 0.15%，保留前景层次。
LO_PCT, HI_PCT = 0.5, 99.85


def _hf(name: str = MODEL_ID):
    return name


def load_model(model_id: str = MODEL_ID, device: str = "cuda"):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForDepthEstimation.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device).eval()
    return proc, model


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    """(N,1,H,W) 的盒式均值，边界按 edge padding —— 与积分图实现等价。"""
    k = 2 * r + 1
    x = F.pad(x, (r, r, r, r), mode="replicate")
    return F.avg_pool2d(x, k, stride=1)


def guided_upsample(depth_low: torch.Tensor, guide: torch.Tensor, size: tuple[int, int],
                    radius_frac: float = 0.006, eps: float = 1e-4) -> torch.Tensor:
    """深度按引导图上采样到 size=(H,W)。

    depth_low: (1,1,h,w)   guide: (1,3,H,W)（0..1）
    走快速引导滤波：先把深度双线性放大到目标分辨率再滤波，
    让局部线性模型把深度边缘拉到与引导图边缘一致。
    """
    h, w = size
    p = F.interpolate(depth_low, size=(h, w), mode="bicubic", align_corners=False)
    I = (0.299 * guide[:, 0:1] + 0.587 * guide[:, 1:2] + 0.114 * guide[:, 2:3])
    # 半径随分辨率缩放，但限制在 2..24：太小吸不住边缘，太大把细节抹平
    r = int(max(2, min(24, round(radius_frac * max(h, w)))))
    mI, mp = _box(I, r), _box(p, r)
    corr_I = _box(I * I, r)
    corr_Ip = _box(I * p, r)
    var_I = corr_I - mI * mI
    cov_Ip = corr_Ip - mI * mp
    a = cov_Ip / (var_I + eps)
    b = mp - a * mI
    return _box(a, r) * I + _box(b, r)


@torch.no_grad()
def estimate(img: Image.Image, proc, model, device: str = "cuda", long_side: int = 1540,
             guide_upsample: bool = True) -> np.ndarray:
    """返回与 img 同尺寸的 float32 视差图（大 = 近）。"""
    W, H = img.size
    # 模型输入：保持宽高比、14 的整数倍（ViT patch 14），长边对齐 long_side
    scale = long_side / float(max(W, H))
    th = int(round(H * scale / 14.0)) * 14
    tw = int(round(W * scale / 14.0)) * 14
    th = max(14, th); tw = max(14, tw)
    inp = proc(images=img, size={"height": th, "width": tw}, keep_aspect_ratio=True,
               ensure_multiple_of=14, return_tensors="pt")
    inp = {k: (v.half().to(device) if v.dtype.is_floating_point else v.to(device))
           for k, v in inp.items()}
    with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        out = model(**inp).predicted_depth
    d = out.float()                                   # (1,h,w)
    raw_hw = (d.shape[-2], d.shape[-1])

    if guide_upsample:
        g = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        g = g.permute(2, 0, 1)[None].to(device)
        d2 = guided_upsample(d[:, None], g, (H, W))
    else:
        d2 = F.interpolate(d[:, None], size=(H, W), mode="bicubic", align_corners=False)
    d2 = d2[0, 0]

    # 归一化到 [LO,HI] 分位 → 固定视差区间
    flat = d2.flatten()
    lo = torch.quantile(flat, LO_PCT / 100.0)
    hi = torch.quantile(flat, HI_PCT / 100.0)
    r = ((d2 - lo) / torch.clamp(hi - lo, min=1e-6)).clamp(0.0, 1.0)
    r = 0.02 + 0.98 * r          # 下界 0.02：避免 1/r^p 发散
    est = r.detach().cpu().numpy().astype(np.float32)
    return est


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="cache/depth")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--long-side", type=int, default=1540)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-guide", action="store_true", help="关掉引导上采样（A/B 用）")
    args = ap.parse_args()

    files = []
    for pat in args.images:
        files.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("没有输入图片", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)

    t = time.time()
    proc, model = load_model(args.model, args.device)
    print(f"模型 {args.model} 加载 {(time.time()-t):.1f}s  device={args.device}")

    rows = []
    for f in files:
        img = Image.open(f).convert("RGB")
        t0 = time.time()
        est = estimate(img, proc, model, args.device, args.long_side, not args.no_guide)
        dt = time.time() - t0
        name = os.path.splitext(os.path.basename(f))[0]
        np.save(os.path.join(args.out, name + ".npy"), est)
        Image.fromarray((np.clip(est, 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(args.out, name + "_vis.png"))
        rows.append({"image": f, "size": list(img.size), "ms": round(dt * 1000)})
        print(f"{name:26} {img.size[0]}x{img.size[1]}  {dt*1000:6.0f}ms  "
              f"视差 {est.min():.3f}~{est.max():.3f} 中位 {np.median(est):.3f}")
    with open(os.path.join(args.out, "_index.json"), "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=1)
    print(f"共 {len(rows)} 张 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
