#!/usr/bin/env python3
"""v2 几何核心：**分层深度图（LDI）网格**。

要解决的问题（来自实测，不是推测）：

  旧链路把整幅照片做成**一张连续的深度薄片**。一张薄片在深度不连续处必然把
  纹理沿竖直方向拉满，于是得到"幕布"、"钟乳石"、"栅栏"——REVIEW.md 里 D1/D4 那两条。
  更糟的是侧移 8m 之后，前景背后本来就**没有**任何像素，会直接露出空洞或雾。

  LDI 的做法（3d-photo-inpainting, CVPR 2020 的结构）：
    * 按深度把场景切成有限几层（近/中/远背景）；
    * 每层只建自己那一层的四边形，被前景遮住的地方**本来就不建面**（不拉伸）；
    * 每层在被掏空/断崖的边界上，沿视线向后**挤出一圈裙边**，让"薄片的厚度"填住
      洞口——侧移时看到的是物体的"厚度"，而不是穿透过去。

  结果：正视观感不损失（仍是照片贴图），侧视不再出现拉满的幕布和空洞。
"""
from __future__ import annotations

import math

import numpy as np


def _box(a: np.ndarray) -> np.ndarray:
    k = np.pad(a, 1, mode="edge")
    return (k[:-2, :-2] + k[:-2, 1:-1] + k[:-2, 2:] +
            k[1:-1, :-2] + k[1:-1, 1:-1] + k[1:-1, 2:] +
            k[2:, :-2] + k[2:, 1:-1] + k[2:, 2:]) / 9.0


def rays_and_pos(z: np.ndarray, cam) -> tuple[np.ndarray, np.ndarray]:
    """(H,W,3) 世界坐标。相机在原点，x 右 / y 上 / -Z 为视线（glTF 约定）。"""
    rays = cam.rays()
    P = rays * z[..., None]
    P[..., 2] *= -1.0
    return rays, P


def flatten_ground(z: np.ndarray, rays: np.ndarray, eye: float = 1.7,
                   tol: float = 0.30, min_z: float = 0.6, max_z: float = 600.0):
    """把"确实落在地平面上"的像素压回平面，去掉地毯式起伏。

    判据是**几何的**：射线向下的像素，若其深度与"射线与 y=0 平面交点距离"
    吻合（相对误差 < tol），它就属于地面。这是 REVIEW.md D1 指出的正确判据
    ——旧代码用"画面行号"判地面，于是高耸主体只要落在地平线以下就被拍平成 y=0。

    只做**向下修正的确定项**：不改判为非地面的像素，避免把山脚当前景拉平。
    """
    ry = rays[..., 1]
    hit = ry < -0.02
    z_plane = np.where(hit, -eye / np.minimum(ry, -1e-3), np.inf)
    ok = hit & (z_plane > min_z) & (z_plane < max_z)
    rel = np.zeros_like(z)
    np.divide(np.abs(z - z_plane), np.maximum(z_plane, 1e-6), out=rel, where=ok)
    ground = ok & (rel < tol)
    out = z.copy()
    out[ground] = z_plane[ground]
    return out, ground


def cliff_mask(z: np.ndarray, step: int, edge_ratio: float = 2.4,
               edge_tau: float = 0.055) -> np.ndarray:
    """(H-1,W-1) 的四边形断崖掩码：True = 该格是断崖，不连面。

    与 pipeline/geom.py 同一套判据（对数深度梯度的结构张量 + 四角深度比），
    但**两处阈值集中在参数里**，不再和 JS 侧各写一份魔数
    （v1.26.8 的 README 声称"同一判据"，实际 Python 是 2.4+edge_tau、JS 是 2.6 单条件）。
    """
    h, w = z.shape
    lz = np.log(np.maximum(z, 1e-3))
    gx = np.zeros_like(lz); gy = np.zeros_like(lz)
    inv = 1.0 / float(step)
    gx[:, 1:-1] = (lz[:, 2:] - lz[:, :-2]) * 0.5 * inv
    gy[1:-1] = (lz[2:] - lz[:-2]) * 0.5 * inv
    Jxx, Jyy, Jxy = _box(gx * gx), _box(gy * gy), _box(gx * gy)
    lam1 = 0.5 * (Jxx + Jyy) + np.sqrt(np.maximum(0.0, 0.25 * (Jxx - Jyy) ** 2 + Jxy ** 2))
    edge_str = np.sqrt(lam1)

    z00, z10 = z[:-1, :-1], z[1:, :-1]
    z01, z11 = z[:-1, 1:], z[1:, 1:]
    zmax = np.maximum(np.maximum(z00, z10), np.maximum(z01, z11))
    zmin = np.minimum(np.minimum(z00, z10), np.minimum(z01, z11))
    ratio = zmax / np.maximum(zmin, 1e-6)
    e = np.maximum(np.maximum(edge_str[:-1, :-1], edge_str[:-1, 1:]),
                   np.maximum(edge_str[1:, :-1], edge_str[1:, 1:]))
    return (ratio > edge_ratio) & (e > edge_tau), ratio


def segment_layers(zs: np.ndarray, valid: np.ndarray, cliff: np.ndarray,
                   far_split: float, near_frac: float = 0.5) -> np.ndarray:
    """按**深度断崖**分割连通域，再把连通域按远近归到 0/1/2 三层。

    为什么不能按固定距离切层（v2 第一版的实测问题）：等距分层会在**平滑表面中间**
    留下一条等深线，那两侧属于不同层、几何各自独立，于是画面上出现一条横贯的细缝。
    人眼对这种直线极度敏感（雪山那张正中间就有一条亮线）。

    现在层界只落在真实断崖（遮断边）上：两侧本来就不连面，缝藏在物体轮廓里。
    """
    from scipy import ndimage
    # cliff 是 (H-1,W-1) 的"格"掩码，先摊到像素上（一个断崖格传播到它的四个角）
    if cliff.shape != zs.shape:
        cp = np.zeros(zs.shape, dtype=bool)
        cp[:-1, :-1] |= cliff; cp[1:, :-1] |= cliff
        cp[:-1, 1:] |= cliff; cp[1:, 1:] |= cliff
        cliff_px = cp
    else:
        cliff_px = cliff
    barrier = cliff_px | (~valid)
    # 断崖膨胀 1 像素：单像素宽的洞不足以让两侧分开，却足以漏光
    barrier = ndimage.binary_dilation(barrier, np.ones((3, 3), bool))
    free = ~barrier
    labels, n = ndimage.label(free, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if n == 0:
        return np.full(zs.shape, -1, dtype=np.int8)
    # 断崖像素按最近连通域归属，避免沿每条边留下 1px 无主带
    idx = ndimage.distance_transform_edt(~free, return_distances=False, return_indices=True)
    labels_full = labels[idx[0], idx[1]]

    order = np.arange(1, n + 1)
    med = ndimage.median(zs, labels_full, index=order)
    med = np.nan_to_num(np.asarray(med, dtype=np.float64), nan=1e9)
    lay_comp = np.full(n + 1, 1, dtype=np.int8)
    far_ids = order[med >= far_split]
    lay_comp[far_ids] = 2
    rest = order[med < far_split]
    if rest.size:
        # 近/中按连通域的远近排：更近的那一半算近层（仍以真实边缘为界）
        ranked = rest[np.argsort(med[rest - 1])]
        k = max(1, int(round(ranked.size * near_frac)))
        lay_comp[ranked[:k]] = 0
        lay_comp[ranked[k:]] = 1
    lay = lay_comp[labels_full].astype(np.int8)
    lay[~valid] = -1
    return lay


def assign_layers(fwd: np.ndarray, sky: np.ndarray | None, far_split: float,
                  near_split: float) -> np.ndarray:
    """0=近 1=中 2=远背景(sky 也归这里) -1=无效。"""
    lay = np.full(fwd.shape, 1, dtype=np.int8)
    lay[fwd >= far_split] = 2
    lay[fwd < near_split] = 0
    if sky is not None:
        lay[sky] = 2
    return lay


def _vertex_normals(P: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """按掩码内的邻居差分算法线，再统一朝向相机。"""
    n = np.zeros_like(P)
    dx = np.zeros_like(P); dy = np.zeros_like(P)
    valid = mask
    # 横向
    m01 = valid[:, :-1] & valid[:, 1:]
    d = P[:, 1:] - P[:, :-1]
    dx[:, :-1] += np.where(m01[..., None], d, 0)
    dx[:, 1:] += np.where(m01[..., None], d, 0)
    # 纵向
    m10 = valid[:-1] & valid[1:]
    d2 = P[1:] - P[:-1]
    dy[:-1] += np.where(m10[..., None], d2, 0)
    dy[1:] += np.where(m10[..., None], d2, 0)
    n = np.cross(dx, dy)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    bad = (ln[..., 0] < 1e-9)
    n = np.where(bad[..., None], np.array([0.0, 0.0, 1.0], dtype=n.dtype), n)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(ln, 1e-9)
    flip = np.sum(n * P, axis=-1) > 1e-4
    n[flip] *= -1.0
    return n


def build_layer(P: np.ndarray, uv: np.ndarray, mask: np.ndarray, filled: np.ndarray,
                zgrid: np.ndarray, dir_world: np.ndarray, skirt_delta_min: float,
                skirt_delta_max: float) -> dict:
    """单层网格 + 裙边。

    mask    : (H,W) 该层拥有哪些顶点
    filled  : (H-1,W-1) 该层哪些四边形成立（未被断崖剪掉）
    """
    h, w = mask.shape
    vid = np.full((h, w), -1, dtype=np.int64)
    ys, xs = np.nonzero(mask)
    vid[ys, xs] = np.arange(ys.size, dtype=np.int64)
    vcount = int(ys.size)
    if vcount == 0:
        return {"pos": np.zeros((0, 3), np.float32), "nrm": np.zeros((0, 3), np.float32),
                "uv": np.zeros((0, 2), np.float32), "index": np.zeros((0,), np.uint32),
                "skirt_quads": 0}

    pos = [P[ys, xs]]
    nrm = [_vertex_normals(P, mask)[ys, xs]]
    uvs = [uv[ys, xs]]

    # ---- 四边形 ----
    jj, ii = np.nonzero(filled)
    a = vid[jj, ii]; c = vid[jj, ii + 1]
    b = vid[jj + 1, ii]; d = vid[jj + 1, ii + 1]
    tris = np.stack([a, b, c, c, b, d], axis=1).astype(np.uint32)

    # ---- 裙边 ---------------------------------------------------
    # 两种缺口都要堵，但**只从较近的一侧挤出**，否则两个方向各建一面墙会互相穿插：
    #   (a) 层内空洞：两端都在本层，但相邻四边形被断崖剪掉了；
    #   (b) 层间剪影：一端在本层、另一端属于别的层 —— 只有本层更近时才挤
    #       （近景物体因此有了"厚度"，侧移看到的是它的侧面而不是穿透的洞）。
    # fv/fh 记录"这条网格边有没有被本层的四边形用到"
    fv = np.zeros((h - 1, w), dtype=bool)          # 竖边 (j,i)-(j+1,i)
    fv[:, 1:] |= filled
    fv[:, :-1] |= filled
    fh = np.zeros((h, w - 1), dtype=bool)          # 横边 (j,i)-(j,i+1)
    fh[1:, :] |= filled
    fh[:-1, :] |= filled

    sk_pos, sk_nrm, sk_uv, sk_idx = [], [], [], []
    nsk = 0
    for axis in (0, 1):
        if axis == 0:
            z_in, z_out = zgrid[:-1, :], zgrid[1:, :]
            m_in, m_out = mask[:-1, :], mask[1:, :]
            no_quad = ~fv                              # (h-1,w)
            iin, iout = (slice(0, h - 1), slice(None)), (slice(1, h), slice(None))
        else:
            z_in, z_out = zgrid[:, :-1], zgrid[:, 1:]
            m_in, m_out = mask[:, :-1], mask[:, 1:]
            no_quad = ~fh                              # (h,w-1)
            iin, iout = (slice(None), slice(0, w - 1)), (slice(None), slice(1, w))

        # 内端在本层、这条边没被本层四边形用到；且要么外端无效（层内空洞），
        # 要么本层更近（剪影时只从近的一侧挤出）
        want = m_in & no_quad & (~m_out | (z_in <= z_out))
        # **图像四边不挤裙边**：边框外没有任何内容，挤出的是"一堵粉墙"
        # （雪山那张底部那条粉红尖刺带就是这么来的），留着洞没人看得见。
        want[0, :] = False
        want[-1, :] = False
        want[:, 0] = False
        want[:, -1] = False
        jj, ii = np.nonzero(want)
        if jj.size == 0:
            continue
        p1 = P[iin][jj, ii]; p2 = P[iout][jj, ii]
        u1 = uv[iin][jj, ii]; u2 = uv[iout][jj, ii]
        # **逐顶点沿各自射线外推**。用单点方向 dv1 推两个端点，裙边四边形会扭转，
        # 从原始视角就能看见一条被拉长的前景色带（雪山那张底部左侧的粉色尖刺）。
        dv1 = dir_world[iin][jj, ii]
        dv2 = dir_world[iout][jj, ii]
        z1 = z_in[jj, ii]
        z2 = np.maximum(z_out[jj, ii], 0.0)
        # 长度只看**近端**：远端是背景（可能几百米），按它算会把裙边拉成 6m 长的刺。
        d1 = np.clip(0.45 * z1, skirt_delta_min, skirt_delta_max)
        d2 = np.clip(0.45 * np.minimum(z2, z1 * 1.5), skirt_delta_min, skirt_delta_max)
        q1 = p1 + dv1 * d1[..., None]
        q2 = p2 + dv2 * d2[..., None]
        base = vcount + np.arange(jj.size * 4, dtype=np.int64).reshape(jj.size, 4)
        fn = np.cross(q1 - p1, p2 - p1)
        ln = np.linalg.norm(fn, axis=-1, keepdims=True)
        fn = np.where((ln[..., 0] < 1e-9)[..., None], np.array([0.0, 0.0, 1.0], fn.dtype), fn)
        fn = fn / np.maximum(np.linalg.norm(fn, axis=-1, keepdims=True), 1e-9)
        sk_pos.append(np.stack([p1, p2, q2, q1], axis=1))
        sk_nrm.append(np.repeat(fn[:, None, :], 4, axis=1))
        sk_uv.append(np.stack([u1, u2, u2, u1], axis=1))
        b0, b1, b2, b3 = base[:, 0], base[:, 1], base[:, 2], base[:, 3]
        sk_idx.append(np.stack([b0, b1, b2, b0, b2, b3], axis=1).astype(np.uint32))
        nsk += int(jj.size)
        vcount += int(jj.size * 4)

    all_pos = pos + sk_pos
    all_nrm = nrm + sk_nrm
    all_uv = uvs + sk_uv
    all_idx = [tris] + sk_idx
    return {
        "pos": np.concatenate([x.reshape(-1, 3) for x in all_pos]).astype(np.float32),
        "nrm": np.concatenate([x.reshape(-1, 3) for x in all_nrm]).astype(np.float32),
        "uv": np.concatenate([x.reshape(-1, 2) for x in all_uv]).astype(np.float32),
        "index": np.concatenate(all_idx).astype(np.uint32),
        "skirt_quads": nsk,
    }


def build(z: np.ndarray, cam, step: int = 1, eye: float = 1.7,
          far_split: float = 70.0, near_split: float = 12.0,
          sky: np.ndarray | None = None, edge_ratio: float = 2.4,
          flatten: bool = True, skirt_min: float = 0.12, skirt_max: float = 6.0,
          dist_min: float = 0.5, median_size: int = 3,
          stretch_clamp: float = 12.0, strong_median: int = 7,
          quantize: float = 0.0) -> dict:
    """主入口：深度（米）→ 分层网格。

    返回 {"layers": [...], "stats": {...}}，可直接交给 pipeline/glb.write_glb。
    """
    z = np.asarray(z, dtype=np.float32)
    h, w = z.shape
    # 深度量化（0=关）：把连续深度压成若干"层间距为 quantize 倍"的档位。
    # 对枝叶这类"几像素内跨 30 倍深度"的内容，连续深度必然把四边形全部剪碎；
    # 量化后同档内深度相同 → 四边形连成片，读作"一层次"而不是一堆尖刺。
    if quantize and quantize > 1.0:
        lq = np.log(np.maximum(z, 1e-3)) / math.log(quantize)
        z = np.exp(np.round(lq) * math.log(quantize)).astype(np.float32)
    if median_size and median_size >= 3:
        # 对数域中值：压掉单像素尖刺。近地（2m 处一个像素≈2mm）时，
        # 深度上的 1 像素噪声就是半米级起伏，直接在几何里变成"细长尖刺"——
        # 雪山那张底部左侧的粉色尖刺带就是这个。中值不改变大尺度结构。
        from scipy import ndimage
        z = np.exp(ndimage.median_filter(np.log(np.maximum(z, 1e-3)),
                                         size=median_size, mode="nearest"))
    rays_full = cam.rays()
    if flatten:
        z, ground = flatten_ground(z, rays_full, eye=eye)
    else:
        ground = np.zeros_like(z, dtype=bool)

    zs = z[::step, ::step]
    rays = rays_full[::step, ::step]
    sky_s = None
    sky_depth = 0.0
    if sky is not None:
        sky_s = sky[::step, ::step]
        # 天空必须推到**最远**，不能取 200m —— v2 第一版写成 200m，天空比远山还近，
        # 于是天空把山挡在后面。取 far 的上限（外面传进来的 far_m）。
        sky_depth = max(far_split * 2.0, 800.0)
        zs = np.where(sky_s, sky_depth, zs)

    P = rays * zs[..., None]
    P[..., 2] *= -1.0
    hh, ww = zs.shape

    # ---- 掠射面纹理拉伸抑制 ----------------------------------------
    # 单视图反投影的固有病：表面越接近"与视线平行"，同样几个像素就被摊到越长
    # 的世界距离上，贴图沿该方向被拉成条。实测雪山图有 10.5% 的网格超过 12×
    # （最大 2802×），在雪脊这种高对比区域就形成可见的彩色细条。
    # 处理办法：只在这些区域把深度换成更大窗口的中值 —— 相当于承认"这里本来
    # 就分辨不出细节"，用更平滑的表面换更均匀的纹理映射。其余区域一格不动。
    if stretch_clamp > 0:
        from scipy import ndimage
        foot = (2.0 * math.tan(cam.hfov * 0.5) / cam.w) * zs
        ddx = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1) / np.maximum(foot[:, :-1], 1e-6)
        ddy = np.linalg.norm(P[1:] - P[:-1], axis=-1) / np.maximum(foot[:-1], 1e-6)
        st = np.zeros_like(zs)
        st[:, :-1] = np.maximum(st[:, :-1], ddx)
        st[:-1, :] = np.maximum(st[:-1, :], ddy)
        bad = st > stretch_clamp
        if bad.any():
            z_s = np.exp(ndimage.median_filter(np.log(np.maximum(zs, 1e-3)),
                                               size=int(strong_median), mode="nearest"))
            zs = np.where(bad, z_s, zs)
            P = rays * zs[..., None]
            P[..., 2] *= -1.0
            stretch_stats = {"bad_frac": round(float(bad.mean()), 4),
                             "p99": round(float(np.percentile(st, 99)), 1)}
        else:
            stretch_stats = {"bad_frac": 0.0, "p99": round(float(np.percentile(st, 99)), 1)}
    else:
        stretch_stats = {"bad_frac": 0.0, "p99": 0.0}
    uv = np.zeros((hh, ww, 2), dtype=np.float32)
    uv[..., 0] = ((np.arange(ww) * step + 0.5) / w)[None, :]
    uv[..., 1] = ((np.arange(hh) * step + 0.5) / h)[:, None]
    dir_world = rays.copy()
    dir_world[..., 2] *= -1.0

    fwd = -P[..., 2]
    valid = np.isfinite(zs) & (zs > dist_min)
    cliff, ratio = cliff_mask(zs, step, edge_ratio)
    valid_q = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    ok_quad = valid_q & ~cliff

    # 分层：按距离切"近景 / 无视差背景"两层。
    #
    # 曾经试过两条更"聪明"的路，都被实测否掉：
    #   * 等距切三层（近/中/远）→ 等深线横穿平滑表面，画面中间出现一条细缝；
    #   * 按断崖连通域分层   → 深度本身很连续，整幅并成一个连通域，等于没分层。
    # 现在的做法：只切一刀，切在 70m 的雾霭里；并且让背景层**向近景多占 1 像素**，
    # 把等深线上那条 1px 的透光缝盖住（两侧深度相同，几何重合，不会 z-fighting）。
    from scipy import ndimage
    far_m = (fwd >= far_split) & valid
    if sky_s is not None:
        far_m = far_m | sky_s
    if far_m.any():
        far_m = ndimage.binary_dilation(far_m, np.ones((3, 3), bool)) & valid
    lay = np.where(far_m, 2, 0).astype(np.int8)
    lay[~valid] = -1

    names = {0: "near", 1: "mid", 2: "backdrop"}
    layers, stats = [], {}
    for lid in (0, 1, 2):
        mask = (lay == lid)
        if not mask.any():
            continue
        q = (ok_quad & mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:])
        if not q.any():
            continue
        L = build_layer(P, uv, mask, q, zs, dir_world, skirt_min, skirt_max)
        L["name"] = names[lid]
        layers.append(L)
        stats[names[lid]] = {
            "vertices": int(L["pos"].shape[0]),
            "triangles": int(L["index"].size // 3),
            "skirt_quads": int(L["skirt_quads"]),
            "pixels": int(mask.sum()),
            "depth_p10": round(float(np.percentile(fwd[mask], 10)), 2),
            "depth_p90": round(float(np.percentile(fwd[mask], 90)), 2),
        }

    stats["_global"] = {
        "grid": [int(hh), int(ww)], "step": int(step),
        "cliff_frac": round(float(cliff.mean()), 4),
        "stretch": stretch_stats,
        "ground_frac": round(float(ground.mean()), 4),
        "far_split": far_split, "near_split": near_split,
        "triangles": int(sum(int(l["index"].size // 3) for l in layers)),
        "vertices": int(sum(int(l["pos"].shape[0]) for l in layers)),
    }
    return {"layers": layers, "stats": stats}
