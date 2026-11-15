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
    return (ratio > edge_ratio) & (e > edge_tau), ratio, edge_str


def cliff_to_pixels(cliff: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """把 (H-1,W-1) 的"格掩码"摊到 (H,W) 的像素上（一格传播到它的四个角）。

    建面用的是格子，分割用的是像素 —— 两边形状差 1，直接相与会报
    "operands could not be broadcast"。这个转换在两处都要用。
    """
    if cliff.shape == shape:
        return cliff
    out = np.zeros(shape, dtype=bool)
    h, w = cliff.shape
    out[:h, :w] |= cliff
    out[1:h + 1, :w] |= cliff
    out[:h, 1:w + 1] |= cliff
    out[1:h + 1, 1:w + 1] |= cliff
    return out


def segment_layers(zs: np.ndarray, valid: np.ndarray, cliff: np.ndarray,
                   far_split: float, near_frac: float = 0.5) -> np.ndarray:
    """按**深度断崖**分割连通域，再把连通域按远近归到 0/1/2 三层。

    为什么不能按固定距离切层（v2 第一版的实测问题）：等距分层会在**平滑表面中间**
    留下一条等深线，那两侧属于不同层、几何各自独立，于是画面上出现一条横贯的细缝。
    人眼对这种直线极度敏感（雪山那张正中间就有一条亮线）。

    现在层界只落在真实断崖（遮断边）上：两侧本来就不连面，缝藏在物体轮廓里。
    """
    from scipy import ndimage
    barrier = cliff_to_pixels(cliff, zs.shape) | (~valid)
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
                skirt_delta_max: float, skirt_cap: float = 3.0) -> dict:
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
        cap = min(skirt_delta_max, skirt_cap)
        d1 = np.clip(0.40 * z1, skirt_delta_min, cap)
        d2 = np.clip(0.40 * np.minimum(z2, z1 * 1.5), skirt_delta_min, cap)
        q1 = p1 + dv1 * d1[..., None]
        q2 = p2 + dv2 * d2[..., None]
        base = vcount + np.arange(jj.size * 4, dtype=np.int64).reshape(jj.size, 4)
        fn = np.cross(q1 - p1, p2 - p1)
        ln = np.linalg.norm(fn, axis=-1, keepdims=True)
        fn = np.where((ln[..., 0] < 1e-9)[..., None], np.array([0.0, 0.0, 1.0], fn.dtype), fn)
        fn = fn / np.maximum(np.linalg.norm(fn, axis=-1, keepdims=True), 1e-9)
        sk_pos.append(np.stack([p1, p2, q2, q1], axis=1))
        sk_nrm.append(np.repeat(fn[:, None, :], 4, axis=1))
        # 裙边的 UV **两端都用内侧端点**的颜色。
        # 用 (u1,u2,u2,u1) 会让裙边从物体色渐变到背景色 —— 在剪影处等于画了一圈
        # 亮色轮廓（实测：灯柱、灌木、人身上都是一圈发白的描边）。
        # 只用内侧色，裙边读起来是"这个物体自己的厚度"。
        sk_uv.append(np.stack([u1, u1, u1, u1], axis=1))
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


def scatter_ratio(zs: np.ndarray, valid: np.ndarray, ratio_thr: float = 4.0) -> float:
    """"散布型"内容占比：相邻像素深度比超过 ratio_thr 的比例。

    枝叶、栏杆、蛛网这类内容在几像素内跨一个量级的深度。它们**不该**用连续网格
    表示 —— 网格会把这些四边形全部剪掉，剩下的是碎条。这个比值就是判别依据：
    测过的雪山/峡谷/海岸都在 0.02~0.05，而满屏秋叶是 0.3 以上。
    """
    z = np.maximum(zs, 1e-3)
    rx = np.maximum(z[:, 1:], z[:, :-1]) / np.minimum(z[:, 1:], z[:, :-1])
    ry = np.maximum(z[1:], z[:-1]) / np.minimum(z[1:], z[:-1])
    ok = valid[:, 1:] & valid[:, :-1]
    ok2 = valid[1:] & valid[:-1]
    hit = int(((rx > ratio_thr) & ok).sum() + ((ry > ratio_thr) & ok2).sum())
    tot = int(ok.sum() + ok2.sum())
    return hit / max(1, tot)


def build_points(P: np.ndarray, rgb: np.ndarray, valid: np.ndarray,
                 step_used: int = 1, max_points: int = 900_000) -> dict:
    """把整幅场景做成一层点云（每个采样点就是照片里的一个像素）。

    为什么点云在这里是对的：点的颜色直接来自照片像素，点与点之间**没有连接**，
    所以既不会被"剪碎"，也不会像网格那样在掠射角把纹理拉成条。
    代价是侧面看是"点雾"而不是实体表面 —— 对枝叶这类内容这是可接受的，
    对山体/建筑则应该用网格（见调用方的判别）。
    """
    # 三个入参都已经是**网格分辨率**（调用方降采样过了），这里不再二次抽稀
    c = rgb[..., :3]
    col = np.concatenate([c, np.ones(c.shape[:2] + (1,), dtype=np.float32)], axis=-1)
    m = valid.reshape(-1)
    pos = P.reshape(-1, 3)[m]
    col = col.reshape(-1, 4)[m]
    # 每点的**原始距离**：surfel 的正确点尺寸是 "基准像素 × 原距离/当前距离"。
    # 少了它，走近时点距被放大、画面变稀（实测雪山图的近景就是一片点雾）。
    size = np.linalg.norm(pos, axis=1).astype(np.float32)
    if pos.shape[0] > max_points:
        k = int(max_points / pos.shape[0])
        pos = pos[::max(1, k)]
        col = col[::max(1, k)]
    return {"name": "splat", "points": True, "pos": pos.astype(np.float32),
            "color": col.astype(np.float32), "size": size,
            "point_px": float(step_used) * 2.0}


def build_ground_skirt(P: np.ndarray, ground: np.ndarray, uv: np.ndarray,
                       radius: float = 400.0, cells: int = 96,
                       n_palette: int = 12, inner_frac: float = 0.28) -> dict | None:
    """重建区域之外补一整圈地面，让"视锥外的灰白"变成"远处地面 + 雾"。

    为什么需要它：v2 的几何止于照片的投影梯形，梯形外什么都没有 ——
    实测截图里画面左右下三块纯色、以及山脚那条刀切硬边，都是这个边界。
    光靠外扩（模型补全）只能把边界推远，推不远的地方还是要有东西兜住。

    做法：
      * 地面高度取"地面像素的 y 中位数"（相机系里约等于 -eye）；
      * 内圈留空（inner_frac），避免与重建网格重叠打架；
      * 颜色**从照片自己的地面像素里采样**：把地面像素的 UV 聚成 n_palette 个代表值，
        按位置哈希轮流贴到裙边顶点上。这样裙边天然带一点照片的地面杂色，
        不再是纯色板，也不需要额外贴图。
      * 法线一律朝上；远端交给雾（渲染器按距离混雾），所以不需要写衰减。
    """
    if not ground.any():
        return None
    # 高度必须取地面像素的**真实 y**（世界坐标），不能拿沿视线的距离取负 ——
    # 早先写成 -zs 得到的是"负的几十上百米"，裙边整个沉到地下，
    # 画面里表现成"上半是景、下半是一整块地面"的硬地平线。
    # 再压低 0.2m 并往内延伸：裙子要钻到重建网格下面，
    # 否则两者交界（网格远端 vs 平坦裙边）会留一条亮缝。
    gy = float(np.median(P[ground][:, 1])) - 0.20
    # 用地面像素的 UV 做色板
    g_uv = uv[ground]
    if g_uv.shape[0] < 8:
        return None
    idx = np.linspace(0, g_uv.shape[0] - 1, min(n_palette, g_uv.shape[0])).astype(int)
    palette = g_uv[np.argsort(g_uv[:, 1] * 1.0 + g_uv[:, 0] * 0.37 + np.random.rand(g_uv.shape[0]) * 0.0)[idx]]

    n = int(cells)
    lin = np.linspace(-radius, radius, n + 1, dtype=np.float32)
    X, Z = np.meshgrid(lin, lin)                 # Z 向 -Z 是前方，这里用世界 z
    Z = Z - radius * 0.35                        # 略向前铺，画面里主要是前方
    inner = radius * inner_frac
    hole = (np.abs(X) < inner) & (np.abs(Z + radius * 0.35) < inner)

    pos, nrm, uvout, index = [], [], [], []
    vid = np.full(X.shape, -1, dtype=np.int64)
    for j in range(n + 1):
        for i in range(n + 1):
            if hole[j, i]:
                continue
            vid[j, i] = len(pos) // 3
            pos.extend([float(X[j, i]), gy, float(Z[j, i])])
            nrm.extend([0.0, 1.0, 0.0])
            # 按格子哈希挑一个色板 UV：既不是纯色，也不会出现明显重复
            k = (j * 73856093 ^ i * 19349663) % palette.shape[0]
            uvout.extend([float(palette[k, 0]), float(palette[k, 1])])

    for j in range(n):
        for i in range(n):
            a, b = vid[j, i], vid[j + 1, i]
            c, d = vid[j, i + 1], vid[j + 1, i + 1]
            if a < 0 or b < 0 or c < 0 or d < 0:
                continue
            index.extend([a, b, c, c, b, d])

    if not index:
        return None
    return {
        "name": "ground",
        "pos": np.asarray(pos, dtype=np.float32).reshape(-1, 3),
        "nrm": np.asarray(nrm, dtype=np.float32).reshape(-1, 3),
        "uv": np.asarray(uvout, dtype=np.float32).reshape(-1, 2),
        "index": np.asarray(index, dtype=np.uint32),
    }


def build(z: np.ndarray, cam, step: int = 1, eye: float = 1.7,
          far_split: float = 70.0, near_split: float = 12.0,
          sky: np.ndarray | None = None, edge_ratio: float = 2.4,
          flatten: bool = True, skirt_min: float = 0.12, skirt_max: float = 6.0,
          dist_min: float = 0.5, median_size: int = 3, ground_tol: float = 0.30,
          skirt_cap: float = 3.0,
          stretch_clamp: float = 12.0, strong_median: int = 7,
          quantize: float = 0.0, rgb: np.ndarray | None = None,
          repr_mode: str = "auto", scatter_thr: float = 0.002,
          cliff_thr: float = 0.010, ground_radius: float = 400.0,
          seg_edge_tau: float = 0.10, quad_stretch_max: float = 0.0,
          card_flatten: bool = True,
          ground_cells: int = 96) -> dict:
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
        z, ground = flatten_ground(z, rays_full, eye=eye, tol=ground_tol)
    else:
        ground = np.zeros_like(z, dtype=bool)

    z_full = z                      # 点云用全分辨率（见下面的表示选择）
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
    # 纹理拉伸倍数：每个格子"3D 边长 / 一像素应有的世界尺寸"。
    # 它同时服务两件事：① 超标区域平滑深度；② **超标格子直接不连面**。
    # ② 是关键：剪影处的深度是"软过渡"，逐格深度比看着不大（所以 cliff 漏判），
    # 但连着建面就是把近物的暗像素拉成几十像素长的黑带 —— 实测灯柱/人头上的黑弧。
    from scipy import ndimage
    foot = (2.0 * math.tan(cam.hfov * 0.5) / cam.w) * zs
    ddx = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1) / np.maximum(foot[:, :-1], 1e-6)
    ddy = np.linalg.norm(P[1:] - P[:-1], axis=-1) / np.maximum(foot[:-1], 1e-6)
    st = np.zeros_like(zs)
    st[:, :-1] = np.maximum(st[:, :-1], ddx)
    st[:-1, :] = np.maximum(st[:-1, :], ddy)

    if stretch_clamp > 0:
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
    # 断崖掩码要在表示选择之前算：它是"该用网格还是点云"两个判据之一
    cliff, ratio, edge_str = cliff_mask(zs, step, edge_ratio)

    # ---- 表示选择：网格还是点云 -----------------------------------
    # 判据是"散布度"（相邻深度比 > 4 的像素占比），不是画面内容的名字。
    scattered = scatter_ratio(zs, valid)
    # 第二个判据：断崖占比。枝叶类内容在 518²→1540 的高分辨率深度上会留下
    # 大量相邻深度比 > 2.4 的格子（实测秋叶 1.9%，而雪山/峡谷/海岸只有 0.2~0.5%，
    # 有 4 倍以上差距）。散布度（相邻比 > 4）在同样这批图上只有 0.000 vs 0.0025，
    # 太薄；断崖占比与几何构建的真实行为直接对应，更可靠。
    cliff_frac_auto = float(cliff.mean())
    use_points = (repr_mode == "points"
                  or (repr_mode == "auto"
                      and (scattered >= scatter_thr or cliff_frac_auto >= cliff_thr)))
    if use_points:
        # 点云用**全分辨率**：点云没有三角形预算的限制，没理由跟着网格一起降采样
        # （网格的 step 是被三角数压下来的）。只在超过点数预算时才按 2/3/4 倍抽稀。
        Pf = rays_full * z_full[..., None]
        Pf[..., 2] *= -1.0
        rgb_f = rgb if rgb is not None else np.full(z_full.shape + (3,), 0.6, dtype=np.float32)
        valid_f = np.isfinite(z_full) & (z_full > dist_min)
        pt_step = 1
        while (z_full.shape[0] // pt_step) * (z_full.shape[1] // pt_step) > 650_000:
            pt_step += 1
        if pt_step > 1:
            Pf = Pf[::pt_step, ::pt_step]
            rgb_f = rgb_f[::pt_step, ::pt_step]
            valid_f = valid_f[::pt_step, ::pt_step]
        pts = build_points(Pf, rgb_f, valid_f, step_used=pt_step)
        return {"layers": [pts], "stats": {
            "splat": {"vertices": int(pts["pos"].shape[0]), "triangles": 0,
                      "skirt_quads": 0, "pixels": int(valid.sum()),
                      "point_px": pts["point_px"],
                      "depth_p10": round(float(np.percentile(fwd[valid], 10)), 2),
                      "depth_p90": round(float(np.percentile(fwd[valid], 90)), 2)},
            "_global": {"grid": [int(hh), int(ww)], "step": int(step),
                        "cliff_frac": round(cliff_frac_auto, 4), "ground_frac": round(float(ground.mean()), 4),
                        "stretch": stretch_stats, "repr": "points",
                        "scatter": round(float(scattered), 4),
                        "cliff": round(cliff_frac_auto, 4),
                        "triangles": 0, "vertices": int(pts["pos"].shape[0])}}}
    valid_q = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]

    # ---- 地物分割：不同地物之间**永不连面** ----------------------------
    # 实测的坏例子：一张校园照片里有一根近处灯柱，深度确实把它判成了"近"，
    # 但建面时柱子边缘的四边形连到了几十米外的背景上，于是柱子被横向拉成
    # 一大片糊条。逐格比较深度比（cliff 判据）救不了它 —— 阈值调低就剪碎地形，
    # 调高就漏掉细杆。
    # 正确做法不是调阈值，而是**先分割再建面**：
    #   把"深度平滑连通"的区域标成同一块，只在同一块内部建四边形。
    # 这样柱子、人、楼房各自成块，柱子只可能长成一张窄卡片，不可能横着连出去。
    from scipy import ndimage
    # 分割屏障要比建面判据**更严**：建面用 2.4 倍深度比 + 结构张量，太稀疏，
    # 细杆的轮廓线断断续续 → 整幅并成一整块（实测灯柱照片只分出 1 块）。
    # 这里只用"相对深度梯度"，阈值低得多，能沿细杆画出连续边界。
    seg_barrier = ndimage.binary_dilation(
        (edge_str > seg_edge_tau) | cliff_to_pixels(cliff, zs.shape) | (~valid),
        np.ones((3, 3), bool))
    seg_labels, n_seg = ndimage.label(~seg_barrier)
    if n_seg:
        # 断崖像素就近归到相邻地物（否则每条边都留 1px 无主带）
        idx_seg = ndimage.distance_transform_edt(seg_barrier, return_distances=False,
                                                return_indices=True)
        seg = seg_labels[idx_seg[0], idx_seg[1]]
    else:
        seg = np.ones(zs.shape, dtype=np.int32)
    s00 = seg[:-1, :-1]; s10 = seg[1:, :-1]; s01 = seg[:-1, 1:]; s11 = seg[1:, 1:]

    # ---- 前景细物体压成"卡片" ----------------------------------------
    # 灯柱、树枝、行人这类"显著近于周边且很细"的物体，深度在剪影处是 3~6 像素的
    # 软过渡；建面时这段斜坡会把物体的暗像素一路拉到远处，肉眼看就是一条黑弧。
    # 先把它们的深度压成各自的中位数（= 一张正对相机的卡片），斜坡消失，
    # 拉丝也就没了。代价是物体失去自身的微小起伏 —— 对细杆/人来说这是可接受的。
    if card_flatten:
        med_big = ndimage.median_filter(zs, size=31, mode="nearest")
        near_mask = valid & (zs < 0.55 * med_big)
        area_total = max(1, int(valid.sum()))
        for lab in range(1, n_seg + 1):
            m = (seg == lab) & valid
            a = int(m.sum())
            if a == 0 or a > 0.12 * area_total:
                continue
            if float(near_mask[m].mean()) < 0.60:
                continue
            zs[m] = float(np.median(zs[m]))
        P = rays * zs[..., None]
        P[..., 2] *= -1.0
        fwd = -P[..., 2]

    same_seg = (s00 == s10) & (s00 == s01) & (s00 == s11)

    ok_quad = valid_q & ~cliff & same_seg
    if quad_stretch_max > 0:
        # 可选：把"纹理拉伸超过 N 倍"的格子也剪掉。
        # 默认关闭 —— 实测在细杆/人头这类剪影上，剪掉高拉伸格子会让中景塌成
        # 一条浅色空带（比原来的黑带更糟）。真正该走的是"物体卡片 + 模型补背面"。
        cell_st = np.maximum(st[:-1, :-1],
                             np.maximum(st[:-1, 1:], np.maximum(st[1:, :-1], st[1:, 1:])))
        ok_quad = ok_quad & (cell_st <= quad_stretch_max)

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
        L = build_layer(P, uv, mask, q, zs, dir_world, skirt_min, skirt_max,
                        skirt_cap=skirt_cap)
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

    # 远景地面裙边：把"视锥外的空白"变成"远处地面 + 雾"。
    # 必须在 _global 统计之前追加，否则三角/顶点数会少算这一层。
    if ground_radius > 0:
        gmask = ground if ground.shape == zs.shape else ground[::step, ::step]
        if gmask.shape == zs.shape:
            skirt = build_ground_skirt(P, gmask, uv,
                                       radius=ground_radius, cells=ground_cells)
            if skirt is not None:
                layers.append(skirt)
                stats["ground"] = {
                    "vertices": int(skirt["pos"].shape[0]),
                    "triangles": int(skirt["index"].size // 3),
                    "skirt_quads": 0, "pixels": int(ground.sum()),
                    "depth_p10": round(float(ground_radius * 0.3), 1),
                    "depth_p90": round(float(ground_radius), 1),
                }

    stats["_global"] = {
        "grid": [int(hh), int(ww)], "step": int(step),
        "cliff_frac": round(float(cliff.mean()), 4),
        "stretch": stretch_stats,
        "ground_frac": round(float(ground.mean()), 4),
        "far_split": far_split, "near_split": near_split,
        "repr": "mesh", "scatter": round(float(scattered), 4),
        "segments": int(n_seg),
        "cliff": round(cliff_frac_auto, 4),
        "triangles": int(sum(int(l.get("index", np.zeros(0, np.uint32)).size // 3) for l in layers)),
        "vertices": int(sum(int(l["pos"].shape[0]) for l in layers)),
    }
    return {"layers": layers, "stats": stats}
