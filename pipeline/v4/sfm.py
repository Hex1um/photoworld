"""增量式 SfM（简化版）：SIFT 匹配 → 本质矩阵 → 锚点对三角化 → PnP 增量定位 → 全局重投影。

不做全局 BA（时间/复杂度不值当），但每一步都可量化：
内点数、极线残差、三角化点数、PnP 内点率、全局重投影 RMSE —— 这些就是 QC 的全部输入。

刻意的简化（都是精度上限的来源，写在这里以免日后误判为 bug）：
  * 内参未知：`f = focal_ratio * max(w, h)`（默认 1.2），主点取画面中心；
  * 尺度不可观：以锚点对的基线为单位 1，最终尺度由 depth.py 用单目深度对齐；
  * 环拍首尾的累积漂移不做闭环平差，只在 QC 里报出来（"闭环残差"）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Cam:
    K: np.ndarray
    w: int
    h: int


@dataclass
class Pair:
    i: int
    j: int
    good: int
    inliers: int
    median_epi: float
    parallax: float             # 三角化点的视差角中位数（度）


@dataclass
class Result:
    cams: list[Cam] = field(default_factory=list)
    poses: list = field(default_factory=list)          # (R, t) world→cam；未定位为 None
    pts: np.ndarray = None                             # (N,3) 稀疏点云
    colors: np.ndarray = None                          # (N,3) 0..1
    pairs: list[Pair] = field(default_factory=list)
    reproj_rmse: float = 0.0
    localized: int = 0
    notes: list[str] = field(default_factory=list)
    kpts: list = field(default_factory=list)
    dess: list = field(default_factory=list)


def estimate_K(w: int, h: int, focal_ratio: float = 1.2) -> np.ndarray:
    f = focal_ratio * max(w, h)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float64)


def _sift(gray, nfeat=8000):
    s = cv2.SIFT_create(nfeatures=nfeat, contrastThreshold=0.005, edgeThreshold=10)
    kp, des = s.detectAndCompute(gray, None)
    pts = np.array([k.pt for k in kp], np.float32) if kp else np.zeros((0, 2), np.float32)
    return pts, des


def _match(da, db, ratio=0.78):
    if da is None or db is None or len(da) < 20 or len(db) < 20:
        return np.zeros((0, 2), np.int32)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    ab = bf.knnMatch(da, db, k=2)
    ba = bf.knnMatch(db, da, k=2)
    ok_ab = {m.queryIdx: m.trainIdx for m, n in ab if m.distance < ratio * n.distance}
    ok_ba = {m.queryIdx: m.trainIdx for m, n in ba if m.distance < ratio * n.distance}
    pairs = [(a, b) for a, b in ok_ab.items() if ok_ba.get(b) == a]
    return np.array(pairs, np.int32).reshape(-1, 2)


def _project(K, R, t, X):
    Xc = (R @ X.T + t).T
    uv = (K @ Xc.T).T
    z = uv[:, 2]
    return uv[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None], z


def run(frames, focal_ratio: float = 1.2, pair_min_inliers: int = 50,
        max_pairs: int = 40, verbose: bool = True) -> Result:
    n = len(frames)
    res = Result()
    if n < 2:
        res.notes.append("只有 1 帧，无法三角化")
        return res

    cams, kpts, dess = [], [], []
    for f in frames:
        g = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2GRAY)
        h, w = g.shape[:2]
        kp, de = _sift(g)
        kpts.append(kp); dess.append(de)
        cams.append(Cam(estimate_K(w, h, focal_ratio), w, h))
    res.cams, res.kpts, res.dess = cams, kpts, dess

    # v4.1.2：配对匹配原先在锚点、增量定位、扩充地图、视差、重投影里被重复算了
    # 4~5 次（每次 BFMatcher.knnMatch 两趟 8000 维描述子）。这里按无序对缓存一次，
    # 反向调用时交换两列，保持原来的 (i 的 keypoint, j 的 keypoint) 语义。
    _match_cache: dict[tuple[int, int], np.ndarray] = {}

    def matched(i: int, j: int) -> np.ndarray:
        if i == j:
            return np.zeros((0, 2), np.int32)
        a, b = (i, j) if i < j else (j, i)
        key = (a, b)
        if key not in _match_cache:
            _match_cache[key] = _match(dess[a], dess[b])
        m = _match_cache[key]
        return m if i < j else m[:, ::-1]

    # ---- 配对：先相邻（视频/环拍顺序），再补远距离对（最多 max_pairs）----
    cand = []
    for i in range(n):
        for d in (1, 2, 3):
            j = (i + d) % n
            if i != j:
                cand.append((min(i, j), max(i, j)))
    cand = sorted(set(cand))
    if len(cand) < max_pairs:
        extra = []
        for i in range(n):
            for j in range(i + 4, n):
                extra.append((i, j))
        cand = sorted(set(cand) | set(extra))[:max_pairs]

    pairs, Emats = [], {}
    for (i, j) in cand:
        m = matched(i, j)
        if len(m) < 20:
            continue
        p1 = kpts[i][m[:, 0]].astype(np.float64)
        p2 = kpts[j][m[:, 1]].astype(np.float64)
        E, mask = cv2.findEssentialMat(p1, p2, cams[i].K, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.2)
        if E is None or E.shape != (3, 3):
            continue
        inl = int(mask.sum())
        mk = mask.ravel() > 0
        # 极线残差
        F = np.linalg.inv(cams[i].K).T @ E @ np.linalg.inv(cams[j].K)
        l = np.c_[p2[mk], np.ones(mk.sum())] @ F.T
        d = np.abs(np.sum(l * np.c_[p1[mk], np.ones(mk.sum())], axis=1)) / np.maximum(
            np.linalg.norm(l[:, :2], axis=1), 1e-9)
        pairs.append(Pair(i, j, len(m), inl, float(np.median(d)), 0.0))
        Emats[(i, j)] = (E, mk)
    good = [p for p in pairs if p.inliers >= pair_min_inliers]
    if verbose:
        med = int(np.median([p.inliers for p in good])) if good else 0
        print("  配对 %d 组，可解 %d 组（内点中位 %d，极线中位 %.2fpx）"
              % (len(pairs), len(good), med,
                 float(np.median([p.median_epi for p in good])) if good else -1))
    if not good:
        res.notes.append("没有一对能解本质矩阵：视角差太小或纹理不足")
        res.pairs = pairs
        return res

    # ---- 锚点对 ----
    a = max(good, key=lambda p: p.inliers)
    i0, j0 = a.i, a.j
    E, mk = Emats[(i0, j0)]
    ma = matched(i0, j0)
    sel0 = mk
    _, R, t, mask2 = cv2.recoverPose(E, kpts[i0][ma[sel0, 0]].astype(np.float64),
                                     kpts[j0][ma[sel0, 1]].astype(np.float64), cams[i0].K)
    poses = [None] * n
    poses[i0] = (np.eye(3), np.zeros((3, 1)))
    poses[j0] = (R, t / max(1e-9, np.linalg.norm(t)))

    P1 = cams[i0].K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = cams[j0].K @ np.hstack([poses[j0][0], poses[j0][1]])
    sel = mask2.ravel() > 0
    m0 = ma[sel0][sel]
    Xh = cv2.triangulatePoints(P1, P2, kpts[i0][m0[:, 0]].astype(np.float64).T,
                               kpts[j0][m0[:, 1]].astype(np.float64).T)
    X = (Xh[:3] / np.where(np.abs(Xh[3]) < 1e-9, 1e-9, Xh[3])).T
    okp = np.isfinite(X).all(1) & (np.linalg.norm(X, axis=1) > 0.02) & (np.linalg.norm(X, axis=1) < 80)
    X = X[okp]
    obs = {}          # (frame, kp) -> point id
    for pid, (u, v) in enumerate(m0[okp]):
        obs[(i0, int(u))] = pid
        obs[(j0, int(v))] = pid
    pts = [X]
    colors = [res_colors(frames, i0, kpts[i0][m0[okp][:, 0]])]

    # ---- 增量定位：反复扫，直到没有新帧能定位 ----
    changed = True
    while changed:
        changed = False
        for f in range(n):
            if poses[f] is not None:
                continue
            # 找一个已定位、且与该帧有足够匹配的伙伴
            best = None
            for g in range(n):
                if poses[g] is None or g == f:
                    continue
                m = matched(g, f)
                if len(m) < 12:
                    continue
                # v4.1.2：旧版对每个 3D 点做一次 m[m[:,0]==kg] 线性扫描，
                # 随地图点增长是 O(obs×matches)。改成一次哈希建表。
                g2f: dict[int, int] = {}
                for a, b in m:
                    ka = int(a)
                    if ka not in g2f:      # 与旧版 hit[0] 保持一致：取第一条匹配
                        g2f[ka] = int(b)
                usable = [(int(kg), pid) for (fg, kg), pid in obs.items()
                          if fg == g and int(kg) in g2f]
                if len(usable) >= 12:
                    score = len(usable)
                    if best is None or score > best[0]:
                        obj = np.array([pts[0][pid] for _, pid in usable], np.float64)
                        img = np.array([kpts[f][g2f[kg]] for kg, _ in usable], np.float64)
                        best = (score, obj, img)
            if best is None:
                continue
            _, obj, img = best
            ok, rvec, tvec, inls = cv2.solvePnPRansac(
                obj, img, cams[f].K, None, reprojectionError=4.0,
                iterationsCount=500, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or inls is None or len(inls) < 10:
                continue
            Rf, _ = cv2.Rodrigues(rvec)
            poses[f] = (Rf, tvec)
            changed = True
            # 新帧与已定位帧三角化，扩充地图
            for g in range(n):
                if g == f or poses[g] is None:
                    continue
                m = matched(g, f)
                if len(m) < 15:
                    continue
                Rg, tg = poses[g]
                Xh = cv2.triangulatePoints(cams[g].K @ np.hstack([Rg, tg]),
                                           cams[f].K @ np.hstack([Rf, tvec]),
                                           kpts[g][m[:, 0]].astype(np.float64).T,
                                           kpts[f][m[:, 1]].astype(np.float64).T)
                Xn = (Xh[:3] / np.where(np.abs(Xh[3]) < 1e-9, 1e-9, Xh[3])).T
                zg = (Rg @ Xn.T + tg).T[:, 2]
                zf = (Rf @ Xn.T + tvec).T[:, 2]
                kn = (zg > 0.02) & (zf > 0.02) & np.isfinite(Xn).all(1) & (np.linalg.norm(Xn, axis=1) < 80)
                if kn.sum() < 10:
                    continue
                base = len(pts[0])
                pts[0] = np.vstack([pts[0], Xn[kn]])
                colors[0] = np.vstack([colors[0], res_colors(frames, g, kpts[g][m[kn][:, 0]])])
                for k, idx in enumerate(np.nonzero(kn)[0]):
                    obs[(g, int(m[idx, 0]))] = base + k
                    obs[(f, int(m[idx, 1]))] = base + k

    res.poses = poses
    res.localized = sum(1 for p in poses if p is not None)
    res.pts = pts[0]
    res.colors = colors[0]
    res.pairs = pairs

    # ---- 视差角（三角化质量的直接指标）----
    for p in pairs:
        if poses[p.i] is None or poses[p.j] is None:
            continue
        Ri, ti = poses[p.i]; Rj, tj = poses[p.j]
        m = matched(p.i, p.j)
        if len(m) < 10:
            continue
        Xh = cv2.triangulatePoints(cams[p.i].K @ np.hstack([Ri, ti]),
                                   cams[p.j].K @ np.hstack([Rj, tj]),
                                   kpts[p.i][m[:, 0]].astype(np.float64).T,
                                   kpts[p.j][m[:, 1]].astype(np.float64).T)
        X = (Xh[:3] / np.where(np.abs(Xh[3]) < 1e-9, 1e-9, Xh[3])).T
        v1 = X - (-Ri.T @ ti.ravel())
        v2 = X - (-Rj.T @ tj.ravel())
        c = np.sum(v1 * v2, axis=1) / np.maximum(np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-9)
        p.parallax = float(np.degrees(np.median(np.arccos(np.clip(c, -1, 1)))))

    # ---- 全局重投影 ----
    errs = []
    for p in good:
        if poses[p.i] is None or poses[p.j] is None:
            continue
        m = matched(p.i, p.j)
        if len(m) < 10:
            continue
        Ri, ti = poses[p.i]; Rj, tj = poses[p.j]
        Xh = cv2.triangulatePoints(cams[p.i].K @ np.hstack([Ri, ti]),
                                   cams[p.j].K @ np.hstack([Rj, tj]),
                                   kpts[p.i][m[:, 0]].astype(np.float64).T,
                                   kpts[p.j][m[:, 1]].astype(np.float64).T)
        X = (Xh[:3] / np.where(np.abs(Xh[3]) < 1e-9, 1e-9, Xh[3])).T
        for f, kk in ((p.i, m[:, 0]), (p.j, m[:, 1])):
            R, t = poses[f]
            proj, z = _project(cams[f].K, R, t, X)
            if z.max() <= 0:
                continue
            e = np.linalg.norm(proj - kpts[f][kk], axis=1)
            ok = (z > 0.02) & np.isfinite(e)
            errs.append(np.minimum(e[ok], 25.0))
    if errs:
        e = np.concatenate(errs)
        res.reproj_rmse = float(np.sqrt(np.mean(e ** 2)))
    return res


def res_colors(frames, fi, uv):
    h, w = frames[fi].rgb.shape[:2]
    u = np.clip(uv.astype(np.int64), [0, 0], [w - 1, h - 1])
    return frames[fi].rgb[u[:, 1], u[:, 0]].astype(np.float32) / 255.0
