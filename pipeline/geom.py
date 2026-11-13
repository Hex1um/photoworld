"""几何：相对深度 → 度量尺度 → 点云 → 连续网格（不粘连、不镜像）。

核心是把"魔法曲线"换成**地面平面标定**：
地平线以下的像素，其沿视线的距离由"相机高度 + 射线俯角"唯一确定；
用它和网络输出的相对视差做最小二乘，得到 disparity→米 的两个参数，
再套到**所有**像素上（包括天空）。于是尺度来自几何而不是拍脑袋。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FOG_FADE = 0.62


@dataclass
class Camera:
    hfov: float
    vfov: float
    w: int
    h: int
    pitch: float = 0.0      # 俯仰（>0 抬头），由地平线位置反解

    @property
    def tan_h(self): return math.tan(self.hfov * 0.5)

    @property
    def tan_v(self): return math.tan(self.vfov * 0.5)

    def rays(self) -> np.ndarray:
        """返回 (H,W,3) 的归一化视线（x 右, y 上, z 前）。含相机俯仰。"""
        u = (np.arange(self.w) + 0.5) / self.w
        v = (np.arange(self.h) + 0.5) / self.h
        dx = (u - 0.5) * 2.0 * self.tan_h
        dy = (0.5 - v) * 2.0 * self.tan_v
        X, Y = np.meshgrid(dx, dy)
        d = np.stack([X, Y, np.ones_like(X)], axis=-1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        if abs(self.pitch) > 1e-6:
            c, s = math.cos(self.pitch), math.sin(self.pitch)
            x, y, z = d[..., 0], d[..., 1], d[..., 2]
            d = np.stack([x, y * c + z * s, -y * s + z * c], axis=-1)
        return d


# 历史实验：早期「搜索俯仰角 + 地面带最小二乘」标定已由下方 calibrate_scale
# 的双锚点方案取代；estimate_pitch / _fit_scale 全仓库无调用点，v1.10.0 移除。


def calibrate_scale(depth_rel: np.ndarray, cam: Camera, eye: float = 1.7,
                    horizon_far: float = 0.0) -> dict:
    """相对视差 → 米，分两种模式：

    A. horizon_far > 0（默认 1500m）：**双锚点**
       近端锚点来自地面平面（俯角最大处，z=eye/sinθ，物理精确）；
       远端锚点来自地平线先验（地平线≈无穷远，取可配置的 horizon_far 米）。
       两个锚点定下 z = A·r^(-p) 的两个参数，中景因此被真正拉开，
       不会挤成"微缩沙盘"。

    B. horizon_far = 0：只用地面带做最小二乘（物理最硬，但中远景会被模型压缩）。

    两条路都不用任何"看心情"的常数：A 的远端是明确声明的先验，B 全是几何约束。
    """
    h, w = depth_rel.shape
    rays = cam.rays()
    ry = rays[..., 1]

    # 近端：画面最下方的地面带
    near = np.zeros((h, 1), dtype=bool)
    near[int(h * 0.93):] = True
    m_near = (ry < -0.10) & near
    if m_near.sum() < 200:
        m_near = ry < -0.06
    z_near = -eye / np.minimum(ry[m_near], -0.01)
    r_near = depth_rel[m_near]
    ok = np.isfinite(z_near) & np.isfinite(r_near) & (r_near > 1e-3)
    r_n = float(np.median(r_near[ok])) if ok.sum() > 50 else 0.8
    z_n = float(np.median(z_near[ok])) if ok.sum() > 50 else eye / 0.34

    if horizon_far <= 0:
        # 模式 B：整条地面带拟合
        rows = np.zeros((h, 1), dtype=bool)
        rows[int(h * 0.42):int(h * 0.97)] = True
        mask = (ry < -0.02) & rows
        if mask.sum() < 2000:
            mask = ry < -0.01
        zt = -eye / np.minimum(ry[mask], -0.004)
        rr = depth_rel[mask]
        good = np.isfinite(zt) & np.isfinite(rr) & (zt > 1.0) & (zt < 4000) & (rr > 0.02)
        rr, zt = rr[good], zt[good]
        if rr.size < 500:
            return {"A": 4.3, "p": 1.0, "n": int(rr.size), "rms": None, "degenerate": True}
        lx, ly = np.log(rr), np.log(zt)
        M = np.stack([-lx, np.ones_like(lx)], axis=1)
        sol, *_ = np.linalg.lstsq(M, ly, rcond=None)
        p_, lnA = float(sol[0]), float(sol[1])
        for _ in range(2):
            res = ly - (-p_ * lx + lnA)
            mad = float(np.median(np.abs(res - np.median(res)))) + 1e-6
            keep = np.abs(res) < 3.0 * mad
            if keep.sum() < 500:
                break
            sol, *_ = np.linalg.lstsq(M[keep], ly[keep], rcond=None)
            p_, lnA = float(sol[0]), float(sol[1])
        A_ = float(math.exp(lnA)); p_ = float(np.clip(p_, 0.25, 3.0))
        rel = np.abs(A_ * np.power(rr, -p_) - zt) / np.maximum(zt, 1e-6)
        return {"A": A_, "p": p_, "b": 0.0, "n": int(rr.size),
                "rms": float(np.median(rel)), "mode": "ground-fit",
                "r_near": float(np.median(rr)) if rr.size else 0.8, "z_near": z_n,
                "degenerate": bool(np.median(rel) > 0.5)}

    # 模式 A：远端锚点取地平线附近的"最远地形"视差
    sky_rows = np.zeros((h, 1), dtype=bool)
    sky_rows[: int(h * 0.6)] = True
    m_far = (ry >= -0.02) & (ry < 0.06) & sky_rows       # 贴着地平线的一条带
    r_far = depth_rel[m_far]
    r_far = r_far[np.isfinite(r_far) & (r_far > 1e-3)]
    r_f = float(np.percentile(r_far, 20)) if r_far.size > 50 else max(0.05, r_n * 0.3)

    r_n = max(1e-3, r_n)
    r_f = max(1e-4, min(r_f, r_n * 0.85))
    if abs(math.log(r_n / r_f)) < 1e-3:
        return {"A": 4.3, "p": 1.0, "n": 0, "rms": None, "degenerate": True}
    p_ = math.log(horizon_far / max(z_n, 0.1)) / math.log(r_n / r_f)
    p_ = float(np.clip(p_, 0.1, 4.0))
    A_ = z_n * (r_n ** p_)
    return {"A": A_, "p": p_, "b": 0.0, "n": int(r_n), "rms": None,
            "mode": "two-anchor", "r_near": r_n, "z_near": z_n,
            "r_far": r_f, "z_far": horizon_far, "degenerate": False}


def metric_depth(depth_rel: np.ndarray, a: float, b: float = 0.0,
                 near: float = 0.5, far: float = 1800.0, p: float = 1.0,
                 r0: float | None = None) -> np.ndarray:
    """相对视差 → 米：带软饱和的幂律 z = A·(r + r0)^(-p)。

    旧的 `clip(A·r^(-p), near, far)` 有两个问题：
      * far 默认 600m，而双锚点标定的远端先验是 1500m —— 实测 46.1%
        像素被压到同一个 600m 平面，"拉开中景"的设计被上限吃掉；
      * r→0 时幂律发散，直接放开 far 会让天空深度爆到 1e5m。
    这里取 r0 = (A/far)^(1/p)：z(0) 恰好收敛到 far，中间段随 r 连续
    变化而不是硬平面；r >> r0 时与原来的标定一致。调用方应把 A 按
    (1 + r0/r_near)^p 修正，让近端锚点仍精确（见 reconstruct.py）。
    """
    d = np.maximum(depth_rel - b, 0.0)
    if p > 0 and a > 0 and far > 0:
        r0 = float(np.power(a / far, 1.0 / p)) if r0 is None else float(r0)
        z = a * np.power(d + r0, -p)
    else:
        z = a * np.power(np.maximum(d, 1e-3), -p)
    return np.clip(z, near, far).astype(np.float32)


def extend_fov(img: np.ndarray, depth_m: np.ndarray, cam: Camera,
               target_hfov: float = math.radians(120.0)) -> tuple[np.ndarray, np.ndarray, Camera]:
    """把视场横向延展到 target_hfov（按边界深度连续延伸，不镜像）。

    做法：左右各补若干列；每列的深度取边界列并沿角度做小斜率外推，
    颜色取边界像素并向雾色渐变，避免出现"墙"。
    """
    if cam.hfov >= target_hfov:
        return img, depth_m, cam

    extra = (target_hfov - cam.hfov) * 0.5
    # 每侧新增列数：按角度线性映射（近似，局部角度小）
    n_side = int(round(cam.w * (math.tan(extra) / cam.tan_h) * 0.5))
    n_side = max(1, min(n_side, cam.w * 2))

    h, w = depth_m.shape
    left_d = np.repeat(depth_m[:, :1], n_side, axis=1)
    right_d = np.repeat(depth_m[:, -1:], n_side, axis=1)
    left_c = np.repeat(img[:, :1], n_side, axis=1)
    right_c = np.repeat(img[:, -1:], n_side, axis=1)

    # 沿角度做少量外推：越靠外越远一点（地形连续感），并轻微加雾
    ramp = np.linspace(0.0, 1.0, n_side, dtype=np.float32)[None, :]   # (1, n_side)
    depth_ext_l = left_d * (1.0 + 0.25 * ramp)
    depth_ext_r = right_d * (1.0 + 0.25 * ramp)
    fog = np.array([0.72, 0.76, 0.80], dtype=np.float32)
    col_l = left_c * (1 - FOG_FADE * ramp[..., None]) + fog * (FOG_FADE * ramp[..., None])
    col_r = right_c * (1 - FOG_FADE * ramp[..., None]) + fog * (FOG_FADE * ramp[..., None])

    img2 = np.concatenate([col_l, img, col_r], axis=1)
    d2 = np.concatenate([depth_ext_l, depth_m, depth_ext_r], axis=1)
    new_w = img2.shape[1]
    cam2 = Camera(hfov=target_hfov, vfov=cam.vfov, w=new_w, h=h, pitch=cam.pitch)
    return img2, d2, cam2


def build_mesh(depth_m: np.ndarray, cam: Camera, edge_ratio: float = 2.4,
               edge_tau: float = 0.055, max_tris: int = 480_000) -> dict:
    """深度图 → 顶点/法线/UV/索引。跨深度断崖的四边形直接跳过，保持轮廓锐利。"""
    h, w = depth_m.shape
    step = 1
    while (w // step) * (h // step) * 2 > max_tris:
        step += 1

    rays = cam.rays()[::step, ::step]
    z = depth_m[::step, ::step]
    hh, ww = z.shape

    P = rays * z[..., None]                      # 相机系：x 右，y 上，z 前
    P[..., 2] *= -1.0                            # 转成 glTF 相机约定：视线朝 -Z

    uv = np.zeros((hh, ww, 2), dtype=np.float32)
    uv[..., 0] = ((np.arange(ww) * step + 0.5) / cam.w)[None, :]
    uv[..., 1] = ((np.arange(hh) * step + 0.5) / cam.h)[:, None]

    # 法线：用邻居差分（不跨界）
    dzdx = np.zeros_like(P)
    dzdx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dzdy = np.zeros_like(P)
    dzdy[1:-1] = P[2:] - P[:-2]
    n = np.cross(dzdx, dzdy)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(ln, 1e-8)
    # 朝向相机
    flip = np.sum(n * P, axis=-1) > 0
    n[flip] *= -1.0

    # --- 断崖判定：结构张量（见《传统视觉与深度学习视觉的数学基础》ch.07）---
    # 相对深度梯度比"绝对比值"更稳：|∇log z| 与尺度无关，细密纹理不会被误杀。
    lz = np.log(np.maximum(z, 1e-3))
    gx = np.zeros_like(lz); gy = np.zeros_like(lz)
    gx[:, 1:-1] = (lz[:, 2:] - lz[:, :-2]) * 0.5
    gy[1:-1] = (lz[2:] - lz[:-2]) * 0.5
    # 结构张量 J = [[gx², gxgy],[gxgy, gy²]]，3×3 均值平滑
    def box(a):
        k = np.pad(a, 1, mode="edge")
        return (k[:-2, :-2] + k[:-2, 1:-1] + k[:-2, 2:] +
                k[1:-1, :-2] + k[1:-1, 1:-1] + k[1:-1, 2:] +
                k[2:, :-2] + k[2:, 1:-1] + k[2:, 2:]) / 9.0
    Jxx, Jyy, Jxy = box(gx * gx), box(gy * gy), box(gx * gy)
    lam1 = 0.5 * (Jxx + Jyy) + np.sqrt(np.maximum(0.0, 0.25 * (Jxx - Jyy) ** 2 + Jxy ** 2))
    edge_str = np.sqrt(lam1)                       # 主方向的梯度强度（相对量）

    ii, jj = np.meshgrid(np.arange(ww - 1), np.arange(hh - 1), indexing="xy")
    z00 = z[:-1, :-1]; z10 = z[:-1, 1:]; z01 = z[1:, :-1]; z11 = z[1:, 1:]
    zmax = np.maximum(np.maximum(z00, z10), np.maximum(z01, z11))
    zmin = np.minimum(np.minimum(z00, z10), np.minimum(z01, z11))
    ratio = zmax / np.maximum(zmin, 1e-6)
    # 四边形内的最大边缘强度
    e = np.maximum(np.maximum(edge_str[:-1, :-1], edge_str[:-1, 1:]),
                   np.maximum(edge_str[1:, :-1], edge_str[1:, 1:]))
    # 两个条件都要满足才跳过：几何上有断崖 + 张量确认是"一条边"而不是噪声
    okq = ~((ratio > edge_ratio) & (e > edge_tau))
    a = (jj * ww + ii)[okq]
    b = a + ww
    quad = np.stack([a, b, a + 1, a + 1, b, b + 1], axis=1).astype(np.uint32)

    return {
        "pos": P.reshape(-1, 3).astype(np.float32),
        "nrm": n.reshape(-1, 3).astype(np.float32),
        "uv": uv.reshape(-1, 2),
        "index": quad.reshape(-1),
        "grid": (hh, ww, step),
        "skipped": float(1.0 - okq.mean()),
        "edge_tau": edge_tau,
    }
