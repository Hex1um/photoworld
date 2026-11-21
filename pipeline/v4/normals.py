"""全局法线一致化：用相机可见性投票，把所有点的法线统一朝"相机一侧"。

为什么需要（实测）：COLMAP `stereo_fusion` 给出的法线只在**每个参考视图内**自洽，
整体不是全局一致朝向。泊松重建对法线方向极其敏感，正反混着输入就会解出一堆
互不相连的片（拉丝面条）。先做这一层，泊松才有意义。

做法（标准做法，不是启发式）：
  1. 把每个三维点投影进每一台相机；落在画面内且深度为正 ⇒ 这台相机"看得见"它；
  2. 对每台看得见的相机，期望法线朝向"从点指向相机"：`dot(n, c - p) > 0`；
  3. 按票数多数决定是否翻转；
  4. 可见相机数 < min_views 的点视为不可靠，直接剔除（MVS 边缘/反光/误匹配）。
"""
from __future__ import annotations

import numpy as np


def orient_by_visibility(P: np.ndarray, N: np.ndarray, cams, poses,
                         min_views: int = 2, max_points: int = 300000,
                         verbose: bool = True):
    """返回 (P, N, idx)：统一朝向后的点/法线，以及它们在**原始数组**里的下标
    （调用方用它同步裁剪颜色，别按长度硬切）。"""
    P0 = np.asarray(P, np.float32)
    N0 = np.asarray(N, np.float32)
    sel = np.arange(len(P0))
    if len(P0) > max_points:
        sel = np.sort(np.random.default_rng(0).choice(len(P0), max_points, replace=False))
    P, N = P0[sel], N0[sel]

    # 位姿预处理：只保留已定位的相机，一次算好 R/t/K/中心
    Rs, ts, Ks, whs, centers = [], [], [], [], []
    for i, p in enumerate(poses):
        if p is None:
            continue
        R = np.asarray(p[0], np.float64)
        t = np.asarray(p[1], np.float64).ravel()
        Rs.append(R); ts.append(t)
        Ks.append(np.asarray(cams[i].K, np.float64))
        whs.append((cams[i].w, cams[i].h))
        centers.append(-R.T @ t)
    if not centers:
        return P, N, sel

    Pf = P.astype(np.float64)
    Nf = N.astype(np.float64)
    votes = np.zeros(len(Pf), np.int32)
    seen = np.zeros(len(Pf), np.int32)

    for k in range(len(Rs)):
        R, t, K = Rs[k], ts[k], Ks[k]
        w, h = whs[k]
        Xc = Pf @ R.T + t
        z = Xc[:, 2]
        ok = z > 0.05
        if not ok.any():
            continue
        zs = np.where(ok, z, 1.0)
        uv = Xc @ K.T
        u = uv[:, 0] / zs
        v = uv[:, 1] / zs
        inside = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not inside.any():
            continue
        # 视线方向（点 → 相机）；法线与之同向 = 朝外（相机在外侧）
        d = centers[k][None, :] - Pf
        dot = np.einsum("ij,ij->i", Nf, d)
        votes += np.where(inside, np.where(dot >= 0, 1, -1), 0)
        seen += inside.astype(np.int32)

    keep = seen >= max(1, min_views)
    if not keep.any():
        return P, N, sel
    N2 = N.copy()
    flip = (votes < 0) & keep
    N2[flip] = -N2[flip]
    N2 = N2 / np.maximum(np.linalg.norm(N2, axis=1, keepdims=True), 1e-9)
    if verbose:
        flipped = int((flip & keep).sum())
        print("      法线一致化：保留 %d/%d 点，翻转 %d 个（可见相机 ≥%d）"
              % (int(keep.sum()), len(P), flipped, min_views))
    return P[keep], N2[keep], sel[keep]
