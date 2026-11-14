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


def fit_three_param(rr: np.ndarray, zt: np.ndarray, b_max: float = 0.5,
                    b_step: float = 0.02) -> dict:
    """T2（v1.22）：拟合 z = A·(r + b)^(-p)，b 用网格搜索。

    为什么必须带 b：归一化后的相对视差与原图的乘性关系只在"不减 lo"时成立；
    但模型输出本身常有偏置（天空/近景饱和），强行 b=0 会把这个偏置压进 p，
    实测 p 从正确 1.0 偏到 1.58（近场误差中位 21%）。
    做法：对每个 b 做对数空间线性最小二乘（ly = -p·lx + lnA），取残差 MAD 最小者。
    """
    rr = np.asarray(rr, dtype=np.float64)
    zt = np.asarray(zt, dtype=np.float64)
    ok = np.isfinite(rr) & np.isfinite(zt) & (zt > 1e-3) & (rr > -b_max + 1e-6)
    rr, zt = rr[ok], zt[ok]
    if rr.size < 500:
        return {"A": 4.3, "p": 1.0, "b": 0.0, "n": int(rr.size), "mad": None,
                "degenerate": True}
    ly = np.log(zt)
    best = None
    for c in np.arange(0.0, b_max + 1e-9, b_step):
        r = np.maximum(rr + c, 1e-6)
        lx = np.log(r)
        M = np.stack([-lx, np.ones_like(lx)], axis=1)
        sol, *_ = np.linalg.lstsq(M, ly, rcond=None)
        res = ly - M @ sol
        mad = float(np.median(np.abs(res - np.median(res))))
        if best is None or mad < best[0]:
            best = (mad, float(sol[0]), float(sol[1]), float(c))
    mad, p_, lnA, c = best
    # 一轮 IRLS 去离群（3×MAD 之外的点权重 0），用定好的 b 再解一次
    r = np.maximum(rr + c, 1e-6)
    lx = np.log(r)
    M = np.stack([-lx, np.ones_like(lx)], axis=1)
    kept = np.ones_like(ly, dtype=bool)
    for _ in range(2):
        sol, *_ = np.linalg.lstsq(M[kept], ly[kept], rcond=None)
        res = ly - M @ sol
        mad2 = float(np.median(np.abs(res[kept] - np.median(res[kept])))) + 1e-6
        kept = np.abs(res) < 3.0 * mad2
        if kept.sum() < 500:
            break
    p_ = float(np.clip(sol[0], 0.25, 3.0))
    A_ = float(math.exp(sol[1]))
    return {"A": A_, "p": p_, "b": float(c), "n": int(kept.sum()), "mad": mad,
            "degenerate": bool(mad > 0.5 or not np.isfinite(A_))}


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
        fit = fit_three_param(rr, zt)
        if fit["degenerate"]:
            return {"A": fit["A"], "p": fit["p"], "b": fit["b"], "n": fit["n"],
                    "rms": fit["mad"], "mode": "ground-fit", "degenerate": True,
                    "r_near": float(np.median(rr)) if rr.size else 0.8, "z_near": z_n}
        A_, p_, b_ = fit["A"], fit["p"], fit["b"]
        rel = np.abs(A_ * np.power(np.maximum(rr + b_, 1e-6), -p_) - zt) / np.maximum(zt, 1e-6)
        return {"A": A_, "p": p_, "b": b_, "n": fit["n"],
                "rms": float(np.median(rel)), "mode": "ground-fit-3p",
                "r_near": float(np.median(rr)) if rr.size else 0.8, "z_near": z_n,
                "degenerate": False}

    # 模式 A：远端锚点 = **视差最小的一批像素**（= 最远）。
    #   T2/T5（v1.22）：旧实现按"贴着地平线的射线带（|ry|<0.06）"选像素，
    #   一旦俯仰角非 0，这条带会落到**地面**上（+8° 时对应 35m 的地面），
    #   于是把"35m 的东西当成 1500m"，A 被放大 42×（实测 85.4 vs 正确 2.0）。
    #   按视差排序与俯仰无关，才是"最远"的正确判据；同时剔除天空以外的
    #   近景（r 太大）不必要 —— 取 5 分位即可。
    upper = np.zeros((h, 1), dtype=bool)
    upper[: int(h * 0.75)] = True
    r_cand = depth_rel[upper & np.isfinite(depth_rel)]
    r_cand = r_cand[r_cand > 1e-4]
    r_f = float(np.percentile(r_cand, 5)) if r_cand.size > 200 else max(0.02, r_n * 0.1)

    r_n = max(1e-3, r_n)
    r_f = max(1e-4, min(r_f, r_n * 0.85))

    # T2：形状（p、b）由**地面带逐像素**拟合决定，远锚只提供绝对尺度。
    #   旧的 "两点解 p" 把 p 绑死在两个中位值上，任何一个中位被天空/近景污染
    #   都会让整幅深度系统性偏斜（实测 p 1.073→1.584）。
    rows = np.zeros((h, 1), dtype=bool)
    rows[int(h * 0.42):int(h * 0.97)] = True
    gm = (ry < -0.02) & rows
    zt_g = -eye / np.minimum(ry[gm], -0.004)
    rr_g = depth_rel[gm]
    good_g = np.isfinite(zt_g) & np.isfinite(rr_g) & (zt_g > 1.0) & (rr_g > 0.02)
    fit = fit_three_param(rr_g[good_g], zt_g[good_g])
    if fit["degenerate"] or r_f <= 1e-3:
        p_ = float(np.clip(math.log(horizon_far / max(z_n, 0.1))
                           / max(1e-6, math.log(r_n / r_f)), 0.1, 4.0))
        A_ = z_n * (r_n ** p_)
        return {"A": A_, "p": p_, "b": 0.0, "n": int(r_n), "rms": None,
                "mode": "two-anchor", "r_near": r_n, "z_near": z_n,
                "r_far": r_f, "z_far": horizon_far, "degenerate": False}
    p_, b_ = fit["p"], fit["b"]
    # 让 z(r_f) = horizon_far：A = horizon_far·(r_f + b)^p
    A_ = float(horizon_far * math.pow(r_f + b_, p_))
    return {"A": A_, "p": p_, "b": b_, "n": fit["n"], "rms": fit["mad"],
            "mode": "ground-3p+far-anchor", "r_near": r_n, "z_near": z_n,
            "r_far": r_f, "z_far": horizon_far, "degenerate": False}


def metric_depth(depth_rel: np.ndarray, a: float, b: float = 0.0,
                 near: float = 0.5, far: float = 1800.0, p: float = 1.0,
                 r0: float | None = None, radd: float = 0.0) -> np.ndarray:
    """相对视差 → 米：带软饱和的幂律 z = A·(r + r0)^(-p)。

    旧的 `clip(A·r^(-p), near, far)` 有两个问题：
      * far 默认 600m，而双锚点标定的远端先验是 1500m —— 实测 46.1%
        像素被压到同一个 600m 平面，"拉开中景"的设计被上限吃掉；
      * r→0 时幂律发散，直接放开 far 会让天空深度爆到 1e5m。
    调用方按 a_eff = A·(1+r0/r_near)^p 修正幅度后，
    z(0) = A·(1/r0 + 1/r_near)^p，令其等于 far 的闭式解为
    r0 = 1 / ((far/A)^(1/p) - 1/r_near)（见 reconstruct.py）；
    这样中间段随 r 连续变化而不是硬平面，且 r >> r0 时与原有标定一致。
    r0=None 的分支只用于没有近端锚点的独立调用（a 未修正，口径不同）。
    """
    # radd（T2）：三参数拟合得到的加性偏移 b_fit，对应 z = A(r + b_fit)^(-p)。
    # 保留旧的 b（减性）语义不动，避免影响既有调用；两者同时存在时先加后减。
    d = np.maximum(depth_rel + radd - b, 0.0)
    if p > 0 and a > 0 and far > 0:
        r0 = float(np.power(a / far, 1.0 / p)) if r0 is None else float(r0)
        z = a * np.power(d + r0, -p)
    else:
        z = a * np.power(np.maximum(d, 1e-3), -p)
    return np.clip(z, near, far).astype(np.float32)


def estimate_pitch(depth_rel: np.ndarray, cam: Camera, eye: float = 1.7,
                   span_deg: float = 12.0, coarse: float = 1.0,
                   fine: float = 0.1) -> dict:
    """T5（v1.22）：用地面带拟合残差反解俯仰角（旧实现 pitch 恒 0）。

    原理：俯仰角错 θ 时，"地平线以下像素的物理距离 -eye/ry" 会整体错位，
    于是 z = A(r+b)^-p 对地面带的拟合残差显著变大。所以**最小化拟合 MAD**
    就是最直接的判据，不需要额外假设地面在哪一行。
    粗搜 ±12°（步长 1°）→ 细搜 ±1°（步长 0.1°），代价约 O(2·h·w) 每档。
    """
    base = cam.pitch
    h, w = depth_rel.shape
    rows = np.zeros((h, 1), dtype=bool)
    rows[int(h * 0.42):int(h * 0.97)] = True

    def mad_at(pitch: float) -> float | None:
        c = Camera(hfov=cam.hfov, vfov=cam.vfov, w=cam.w, h=cam.h, pitch=pitch)
        ry = c.rays()[..., 1]
        m = (ry < -0.02) & rows
        if m.sum() < 500:
            return None
        zt = -eye / np.minimum(ry[m], -0.004)
        rr = depth_rel[m]
        ok = np.isfinite(zt) & np.isfinite(rr) & (zt > 1.0) & (rr > 0.02)
        if ok.sum() < 500:
            return None
        fit = fit_three_param(rr[ok], zt[ok])
        return fit.get("mad")

    best, best_mad = base, None
    lim = math.radians(span_deg)
    p = base - lim
    while p <= base + lim + 1e-9:
        m = mad_at(p)
        if m is not None and (best_mad is None or m < best_mad):
            best, best_mad = p, m
        p += math.radians(coarse)
    if best_mad is None:
        return {"pitch": base, "mad": None, "ok": False, "reason": "地面带不足，无法反解"}
    p2 = best - math.radians(coarse)
    end = best + math.radians(coarse) + 1e-9
    while p2 <= end:
        m = mad_at(p2)
        if m is not None and m < best_mad:
            best, best_mad = p2, m
        p2 += math.radians(fine)
    return {"pitch": float(best), "mad": float(best_mad), "ok": True,
            "delta_deg": round(math.degrees(best - base), 2)}


def apply_calibration(depth_rel: np.ndarray, cal: dict, far: float,
                      near: float = 0.5) -> tuple[np.ndarray, dict]:
    """把标定结果套到深度图上：z = a_eff·(r + b + r0)^(-p)，并返回更新后的 cal。

    T2（v1.22）：这段以前内联在 reconstruct.py 里（r0 闭式解 + a_eff 修正），
    测试再手抄一遍 —— 两边一旦不同步，"验收通过"就没有意义。抽成函数后
    reconstruct 和 verify_*.py 走同一条式子。
    """
    p = float(cal["p"])
    A = float(cal["A"])
    b = float(cal.get("b", 0.0))
    r_near = max(float(cal.get("r_near", 0.8)), 1e-3)
    r_near_eff = max(1e-3, r_near + b)
    denom = ((far / A) ** (1.0 / p) - 1.0 / r_near_eff) if (p > 0 and A > 0) else 0.0
    r0 = (1.0 / denom if denom > 1e-6 else
          ((A / far) ** (1.0 / p) if (p > 0 and A > 0) else 0.0))
    a_eff = A * (1.0 + r0 / r_near_eff) ** p if p > 0 else A
    z = metric_depth(depth_rel, a_eff, 0.0, near=near, far=far, p=p, r0=r0, radd=b)
    out = dict(cal)
    out.update({"far": far, "r0": r0, "a_effective": a_eff, "b": b})
    return z, out


def extend_fov(img: np.ndarray, depth_m: np.ndarray, cam: Camera,
               target_hfov: float = math.radians(120.0)) -> tuple[np.ndarray, np.ndarray, Camera]:
    """把视场横向延展到 target_hfov（按边界深度连续延伸，不镜像）。

    做法：左右各补若干列；每列的深度取边界列并沿角度做小斜率外推，
    颜色取边界像素并向雾色渐变，避免出现"墙"。
    """
    if cam.hfov >= target_hfov:
        return img, depth_m, cam

    # T1（v1.22）：每侧新增列数必须按**切向**算，不是按角度线性近似。
    #   旧的 w·(tan(extra)/tan_h)·0.5 在 60°→120° 时给 800 列（正确 1600），
    #   缺口被「把原内容拉宽」填上：左缘角度 -30°→-40.9°，切向放大 1.500×。
    #   像素网格在切向上是均匀的（u-0.5 ∝ tanθ），原内容的列在新网格里只是
    #   居中平移、不需要重采样；两侧补 n_side 列即可：
    #       tan(target/2) = (w/2 + n_side) / (w/2) · tan_h
    #   解得 n_side = w·(tan(target/2)/tan_h − 1)/2。
    scale_t = math.tan(target_hfov * 0.5) / cam.tan_h
    n_side = int(round(cam.w * (scale_t - 1.0) * 0.5))
    n_side = max(1, min(n_side, cam.w * 4))

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
    # T1：宽度变了，竖直角分辨率必须跟着重算（方形像素：tan_v/h = tan_h/w），
    # 否则横竖角度每像素不一致，天空/远景会被横向压扁。
    tan_v2 = math.tan(target_hfov * 0.5) * (h / float(new_w))
    vfov2 = 2.0 * math.atan(tan_v2)
    cam2 = Camera(hfov=target_hfov, vfov=vfov2, w=new_w, h=h, pitch=cam.pitch)
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
    # T8（v1.22）：**边界用单侧差分**。原来左右/上下第一列保持全 0，
    # cross(0, 0) = 0 → 归一化后是零法线，glTF 里这类顶点会导致黑边/抖动。
    dzdx = np.zeros_like(P)
    dzdx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dzdx[:, 0] = P[:, 1] - P[:, 0]
    dzdx[:, -1] = P[:, -1] - P[:, -2]
    dzdy = np.zeros_like(P)
    dzdy[1:-1] = P[2:] - P[:-2]
    dzdy[0] = P[1] - P[0]
    dzdy[-1] = P[-1] - P[-2]
    n = np.cross(dzdx, dzdy)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    # 退化面（三点共线）保底：给一个朝向相机的常量法线，别留零向量
    bad = (ln[..., 0] < 1e-8)
    if bad.any():
        n[bad] = np.array([0.0, 0.0, 1.0], dtype=n.dtype)
        ln = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(ln, 1e-8)
    # 朝向相机
    # T8：阈值从 0 抬到 1e-4 —— 近似切向的面（n·P≈0）会在两个符号之间抖动，
    # 相邻顶点一翻一不翻就出现"法线闪烁"。留一条死区。
    flip = np.sum(n * P, axis=-1) > 1e-4
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
