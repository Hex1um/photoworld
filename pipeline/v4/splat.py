"""3DGS-lite：纯 PyTorch 的可微点泼溅（不依赖 CUDA 扩展，8GB 卡可训）。

为什么不直接用 `gsplat`：本机**没有 CUDA toolkit**，gsplat 的 kernels 装不上
（`import gsplat.cuda` 能过，但 `rasterization` 直接报 `No CUDA toolkit found`）。
所以这里用纯 torch 实现一个"够用"的可微泼溅：

  场景 = N 个各向同性高斯（位置 xyz / 颜色 / 半径 / 不透明度）
  渲染 = 投影到屏幕 → 双线性泼到 3×3 邻域 → 按 (color·α, α) 累加并归一化
  训练 = L1 + (1-SSIM)，Adam，支持任意已知位姿（来自 v4.sfm）

和真 3DGS 的差距（明确写出来，避免误判）：没有各向异性协方差、没有球谐、
没有 densify/prune、没有真正的 α 合成排序。它的定位是"视频 → 可训的三维表示 →
能渲新视角"，不是论文级质量。
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _ssim(x: torch.Tensor, y: torch.Tensor, c1=0.01 ** 2, c2=0.03 ** 2) -> torch.Tensor:
    """单尺度 SSIM（5×5 均值池化），用来给 L1 补结构损失。"""
    k = torch.ones(1, 1, 5, 5, device=x.device) / 25.0
    mu_x = torch.nn.functional.conv2d(x, k, padding=2)
    mu_y = torch.nn.functional.conv2d(y, k, padding=2)
    sx = torch.nn.functional.conv2d(x * x, k, padding=2) - mu_x ** 2
    sy = torch.nn.functional.conv2d(y * y, k, padding=2) - mu_y ** 2
    sxy = torch.nn.functional.conv2d(x * y, k, padding=2) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2))
    return s.mean()


class Splats:
    def __init__(self, xyz, rgb, scale, opacity, device="cuda"):
        self.xyz = torch.nn.Parameter(torch.as_tensor(xyz, dtype=torch.float32, device=device))
        # 颜色用 logit 初始化：渲染端是 sigmoid(rgb)，直接塞 0..1 会把整张图压到
        # 0.5~0.73 的灰白（实测渲出来一片白），logit 之后初始颜色就是原色。
        c0 = torch.as_tensor(rgb, dtype=torch.float32, device=device).clamp(0.02, 0.98)
        self.rgb = torch.nn.Parameter(torch.logit(c0))
        # 用 log 半径，保证恒正
        self.log_s = torch.nn.Parameter(
            torch.log(torch.as_tensor(scale, dtype=torch.float32, device=device)))
        self.logit_o = torch.nn.Parameter(
            torch.logit(torch.clamp(torch.as_tensor(opacity, dtype=torch.float32, device=device),
                                    1e-3, 1 - 1e-3)))

    def params(self):
        return [self.xyz, self.rgb, self.log_s, self.logit_o]


def _project(xyz, R, t, K, W, H):
    Xc = xyz @ R.T + t.reshape(1, 3)
    z = -Xc[:, 2]                       # 相机朝 -z
    ok = z > 0.05
    zs = torch.where(ok, z, torch.ones_like(z))
    u = K[0, 0] * (Xc[:, 0] / zs) + K[0, 2]
    v = K[1, 1] * (Xc[:, 1] / zs) + K[1, 2]
    return u, v, z, ok


def render(spl: Splats, R, t, K, W, H, bg=None):
    """返回 (3,H,W) 图像（0..1）。α 用累加近似，最后归一化。"""
    dev = spl.xyz.device
    u, v, z, ok = _project(spl.xyz, R, t, K, W, H)
    scale = torch.exp(spl.log_s)
    alpha = torch.sigmoid(spl.logit_o)
    col = torch.sigmoid(spl.rgb)
    # 像素半径：f·scale/z（各向同性）
    rad = (K[0, 0] * scale / z.clamp(min=1e-3)).clamp(0.6, 40.0)
    alpha = (alpha * ok.float()).clamp(0, 1)

    acc_c = torch.zeros(3, H * W, device=dev)
    acc_a = torch.zeros(1, H * W, device=dev)
    # 3×3 邻域双线性泼溅（9 次 index_put，全都可微）
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            px = torch.round(u + dx).long()
            py = torch.round(v + dy).long()
            inside = (px >= 0) & (px < W) & (py >= 0) & (py < H) & ok
            if not inside.any():
                continue
            d2 = ((u - px.float()) ** 2 + (v - py.float()) ** 2) / (rad.clamp(min=1.0) ** 2)
            w = torch.exp(-2.0 * d2) * alpha
            w = torch.where(inside, w, torch.zeros_like(w))
            idx = (py.clamp(0, H - 1) * W + px.clamp(0, W - 1))
            # 用 scatter_add：可微、自动累加重复索引。
            # 别用 index_put_（对零张量做原地写会把整条图标记成不需要梯度），
            # 也别用 Tensor.index_put（它只吃张量索引，传 slice 会 TypeError）。
            # 显式 float()：中间量有可能被 numpy 的 float64 带偏，scatter_add 要求同 dtype
            vals = (col * w[:, None]).T.float()
            acc_c = acc_c.scatter_add(1, idx.expand(3, idx.numel()), vals)
            acc_a = acc_a.scatter_add(1, idx[None, :], w.float()[None, :])
    # 如果这一帧所有点都在相机背后（未定位帧 / 位姿异常），上面 9 个偏移全被跳过，
    # acc_* 还是初始的常零张量 —— 那样 img 就没有 grad_fn，backward 会报
    # "element 0 of tensors does not require grad"。这里补一条零梯度连接到参数上。
    if acc_c.grad_fn is None:
        zero = (col.sum() + alpha.sum()) * 0.0
        acc_c = acc_c + zero
        acc_a = acc_a + zero
    a = acc_a.clamp(0.05, 1.0)
    img = (acc_c / a).clamp(0, 1).reshape(3, H, W)
    if bg is not None:
        img = img * acc_a.clamp(0, 1).reshape(1, H, W) + bg.reshape(3, 1, 1) * \
              (1 - acc_a.clamp(0, 1)).reshape(1, H, W)
    return img


def train(frames, res, iters: int = 400, lr: float = 0.02, W: int = 320, H: int = 200,
          device: str = "cuda", seed: int = 0, verbose: bool = True, init=None):
    """用已知位姿 + 输入图训练泼溅；返回 Splats 与损失曲线。

    `init=(pts, cols)` 可直接给一份**稠密**点云（来自 `tsdf.dense_points`）当代起点：
    SfM 的稀疏点只有几千个，渲出来画面基本是空的（实测 loss 卡在背景色上不动）。
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pts, cols = (init if init is not None else (res.pts, res.colors))
    if pts is None or len(pts) < 200:
        return None, []

    # 尺度归一（训练更稳）：把场景缩到 ±2 单位内
    c = np.median(pts, axis=0).astype(np.float32)
    span = float(np.percentile(np.linalg.norm(pts - c, axis=1), 90)) or 1.0
    s = 1.6 / span
    # 统一 float32：numpy 的 median 是 float64，混进来会让中间的 t/w 都变 float64，
    # 进而让 scatter_add 的 dtype 对不上、梯度链在某些路径上中断（踩过）
    P = ((pts - c) * s).astype(np.float32)
    # 位姿跟着同一变换：X' = s(X - c) ⇒ 世界点与相机的相对关系保持不变，
    # 即 t' = s·t - s·R·c（相机中心不动、点云缩放）。
    Rb, tb = [], []
    for p in res.poses:
        if p is None:
            Rb.append(np.eye(3, dtype=np.float32)); tb.append(np.zeros(3, dtype=np.float32)); continue
        R0 = np.asarray(p[0], np.float32)
        t0 = np.asarray(p[1], np.float32).ravel()
        Rb.append(R0)
        tb.append((s * t0 - s * (R0 @ c)).astype(np.float32))

    n = len(P)
    spl = Splats(P, cols, np.full(n, 0.02, np.float32), np.full(n, 0.85, np.float32), device)
    opt = torch.optim.Adam(spl.params(), lr=lr)

    # 预渲染输入图（缩到 W×H）
    import cv2
    tgts = []
    for f in frames:
        im = cv2.resize(f.rgb, (W, H), interpolation=cv2.INTER_AREA)
        tgts.append(torch.as_tensor(im.astype(np.float32) / 255.0, device=device).permute(2, 0, 1))
    # 直接把 K 按输出分辨率重算（等比缩放焦距与主点）
    k0 = res.cams[0].K
    sx, sy = W / res.cams[0].w, H / res.cams[0].h
    K = torch.tensor([[k0[0, 0] * sx, 0, k0[0, 2] * sx],
                      [0, k0[1, 1] * sy, k0[1, 2] * sy],
                      [0, 0, 1]], dtype=torch.float32, device=device)

    hist = []
    valid = [i for i, p in enumerate(res.poses) if p is not None]
    if not valid:
        return None, []
    for it in range(iters):
        fi = int(valid[int(rng.integers(0, len(valid)))])
        R = torch.tensor(Rb[fi], device=device)
        t = torch.tensor(tb[fi], device=device)
        img = render(spl, R, t, K, W, H, bg=torch.tensor([0.55, 0.6, 0.66], device=device))
        gt = tgts[fi]
        l1 = (img - gt).abs().mean()
        ssim = _ssim(img.mean(0, keepdim=True), gt.mean(0, keepdim=True))
        loss = l1 + 0.35 * (1 - ssim)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append(float(loss))
        if verbose and (it % 50 == 0 or it == iters - 1):
            print("        iter %4d  loss %.4f  (l1 %.4f, ssim %.3f)" % (it, float(loss), float(l1), float(ssim)))
    return spl, hist


def render_novel(spl: Splats, pose_R, pose_t, K, W, H, bg_color=(0.55, 0.6, 0.66)):
    """渲一帧新视角（返回 H×W×3 uint8）。"""
    R = torch.as_tensor(pose_R, dtype=torch.float32, device=spl.xyz.device)
    t = torch.as_tensor(pose_t, dtype=torch.float32, device=spl.xyz.device)
    bg = torch.tensor(bg_color, dtype=torch.float32, device=spl.xyz.device)
    with torch.no_grad():
        img = render(spl, R, t, K, W, H, bg=bg)
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
