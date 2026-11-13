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


def estimate_pitch(depth_rel: np.ndarray, cam: Camera, eye: float = 1.7) -> tuple[float, float, float]:
    """在地平线位置未知时，搜索使"地面平面假设"最成立的俯仰角。"""
    best = (0.0, 1e9, 0.5)
    for hz in np.arange(0.32, 0.72, 0.02):
        pitch = math.atan((hz - 0.5) * 2.0 * cam.tan_v)
        c = Camera(cam.hfov, cam.vfov, cam.w, cam.h, pitch=pitch)
        cal = _fit_scale(depth_rel, c, eye)
        if cal["rms"] is not None and cal["rms"] < best[1]:
            best = (pitch, cal["rms"], float(hz))
    return best


def _fit_scale(depth_rel: np.ndarray, cam: Camera, eye: float = 1.7) -> dict:
    h, w = depth_rel.shape
    rays = cam.rays()
    ry = rays[..., 1]
    rows = np.zeros((h, 1), dtype=bool)
    rows[int(h * 0.55):int(h * 0.99)] = True
    mask = (ry < -0.03) & rows
    if mask.sum() < 500:
        mask = ry < -0.03

    t_geom = -eye / np.minimum(ry, -0.004)
    r = depth_rel[mask]
    z = t_geom[mask]
    ok = np.isfinite(z) & (z > 0.5) & (z < 3000) & np.isfinite(r)
    r, z = r[ok], z[ok]
    if r.size < 300:
        k = float(np.median(r * z)) if r.size else 20.0
        return {"a": k, "b": 0.0, "n": int(r.size), "rms": None, "degenerate": True}

    inv = 1.0 / z
    A = np.stack([inv, np.ones_like(inv)], axis=1)
    sol, *_ = np.linalg.lstsq(A, r, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    # 稳健：剔除 3*MAD 之外的样本再拟合一次
    for _ in range(2):
        res = r - (a * inv + b)
        mad = float(np.median(np.abs(res - np.median(res)))) + 1e-6
        keep = np.abs(res) < 3.5 * mad
        if keep.sum() < 200:
            break
        r2, inv2 = r[keep], inv[keep]
        A2 = np.stack([inv2, np.ones_like(inv2)], axis=1)
        sol, *_ = np.linalg.lstsq(A2, r2, rcond=None)
        a, b = float(sol[0]), float(sol[1])
    if not np.isfinite(a) or a <= 0:
        a, b = float(np.median(r * z)), 0.0
    res = r - (a * inv + b)
    rms = float(np.sqrt(np.mean(res ** 2)))
    return {"a": a, "b": b, "n": int(r.size), "rms": rms,
            "degenerate": bool(rms > 0.22)}


def calibrate_scale(depth_rel: np.ndarray, cam: Camera, eye: float = 1.7) -> dict:
    """相对视差 → 米：用**整条地面带**拟合 z = A * r^(-p)。

    约束来自真实几何：地平线以下所有像素都落在同一个水平面上，
    它们的距离 z = eye / sinθ 是精确已知的。于是「模型的 r」和「真实的 z」
    在很宽的范围内一一对应，拟合出的幂律才能外推到中景。
    这不是经验曲线：参数完全由这张图的地面决定。
    """
    h, w = depth_rel.shape
    rays = cam.rays()
    ry = rays[..., 1]
    # 地面带：地平线以下、避开最底部两行（可能入镜遮挡物）
    rows = np.zeros((h, 1), dtype=bool)
    rows[int(h * 0.42):int(h * 0.97)] = True
    mask = (ry < -0.02) & rows
    if mask.sum() < 2000:
        mask = ry < -0.01

    z_true = -eye / np.minimum(ry[mask], -0.004)
    r = depth_rel[mask]
    ok = np.isfinite(z_true) & np.isfinite(r) & (z_true > 1.0) & (z_true < 4000) & (r > 0.02)
    r, z_true = r[ok], z_true[ok]
    if r.size < 500:
        return {"A": 4.3, "p": 1.0, "n": int(r.size), "rms": None, "degenerate": True}

    lx, ly = np.log(r), np.log(z_true)
    A2 = np.stack([-lx, np.ones_like(lx)], axis=1)   # ly = -p*lx + lnA
    sol, *_ = np.linalg.lstsq(A2, ly, rcond=None)
    p_, lnA = float(sol[0]), float(sol[1])
    for _ in range(2):                                # 稳健剔除
        pred = -p_ * lx + lnA
        res = ly - pred
        mad = float(np.median(np.abs(res - np.median(res)))) + 1e-6
        keep = np.abs(res) < 3.0 * mad
        if keep.sum() < 500:
            break
        sol, *_ = np.linalg.lstsq(A2[keep], ly[keep], rcond=None)
        p_, lnA = float(sol[0]), float(sol[1])

    A_ = float(math.exp(lnA))
    p_ = float(np.clip(p_, 0.25, 3.0))
    pred = A_ * np.power(r, -p_)
    rel = np.abs(pred - z_true) / np.maximum(z_true, 1e-6)
    rms = float(np.median(rel))
    return {"A": A_, "p": p_, "b": 0.0, "n": int(r.size), "rms": rms,
            "degenerate": bool(rms > 0.5)}


def metric_depth(depth_rel: np.ndarray, a: float, b: float = 0.0,
                 near: float = 0.5, far: float = 600.0, p: float = 1.0) -> np.ndarray:
    """z = A * r^(-p)。p=1 时退化为视差模型。"""
    d = np.maximum(depth_rel - b, 1e-3)
    z = a * np.power(d, -p)
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


def build_mesh(depth_m: np.ndarray, cam: Camera, edge_ratio: float = 1.6,
               max_tris: int = 480_000) -> dict:
    """深度图 → 顶点/法线/UV/索引。跨深度断崖的四边形直接跳过，保持轮廓锐利。"""
    h, w = depth_m.shape
    step = 1
    while (w // step) * (h // step) * 2 > max_tris:
        step += 1

    rays = cam.rays()[::step, ::step]
    z = depth_m[::step, ::step]
    hh, ww = z.shape

    P = rays * z[..., None]                      # 相机系：x 右，y 上，z 前

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

    idx = []
    ii, jj = np.meshgrid(np.arange(ww - 1), np.arange(hh - 1), indexing="xy")
    z00 = z[:-1, :-1]; z10 = z[:-1, 1:]; z01 = z[1:, :-1]; z11 = z[1:, 1:]
    zmax = np.maximum(np.maximum(z00, z10), np.maximum(z01, z11))
    zmin = np.minimum(np.minimum(z00, z10), np.minimum(z01, z11))
    okq = (zmax / np.maximum(zmin, 1e-6)) < edge_ratio        # 断崖处跳过
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
    }
