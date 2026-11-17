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

import os
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


def smooth_ground_rows(zs: np.ndarray, rays: np.ndarray, width: int = 9,
                       horizon_y: float = -0.04) -> np.ndarray:
    """地平线以下的深度做**横向各向异性**低通（列方向平滑，行方向保留）。

    为什么只压横向：脚下那片地面在同一行里深度应该几乎一样（它是一条水平等距线），
    逐列抖动纯粹是模型噪声（草地纹理最容易被判成"忽远忽近"）。而**行方向**的
    深度梯度是真实的（越往画面下方越近），抹掉它反而会把地面拍成一块板。

    压掉列噪声之后，相邻顶点的三维距离变得和像素间距匹配，四边形不再被拉成
    长条 —— 也就是实测截图里"中景草地那条横向暗带"。
    （在线链路 js/world.js 早就有这一手："地平线以下横向 7 列低通"，这里是同一招。）
    """
    from scipy import ndimage
    ry = rays[..., 1]
    below = ry < horizon_y
    if not below.any() or width < 3:
        return zs
    lz = np.log(np.maximum(zs, 1e-3))
    # 只沿 x 方向做盒式均值（size=(1,width)），行方向完全不动
    sm = ndimage.uniform_filter(lz, size=(1, int(width)), mode="nearest")
    # **只在"横向噪声大"的地方平滑**：整片无差别低通会把纹理干净的地面
    # （沙漠那种均匀砂地）也抹成一块色板 —— 实测就是这么把近景纹理做没了。
    # 用横向一阶差分的绝对值当噪声度量。
    gx = np.abs(np.diff(lz, axis=1, prepend=lz[:, :1]))
    noise = ndimage.uniform_filter(gx, size=(1, int(width)), mode="nearest")
    thr = np.percentile(noise[below], 60) if below.any() else 0.0
    mix = np.clip((noise - thr) / max(1e-4, thr * 1.5), 0.0, 1.0)
    out = lz * (1.0 - mix) + sm * mix
    out[~below] = lz[~below]
    return np.exp(out).astype(np.float32)


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
                skirt_delta_max: float, skirt_cap: float = 3.0,
                no_skirt: np.ndarray | None = None) -> dict:
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
        # **外端是天空的边不挤裙边**：那里没有"洞"要堵（天空由天穹画），
        # 挤出来的是把照片亮度像素往后拉的一圈亮边 —— 地面视角里山脊上那条白线。
        if no_skirt is not None:
            want &= ~no_skirt[iout]
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



def cap_depth_gradient(z: np.ndarray, k: float = 1.35, iters: int = 12) -> np.ndarray:
    """限制相邻像素之间的**深度跳变**，把"拉向远方的长尾"截断。

    问题：剪影处一个像素内深度从前景（2m）跳到远景（300m），建面时那个四边形
    只能把前景像素沿射线拉出去 —— 画面上表现成一条从物体尾部射向远处的细丝。

    观察：真实物体是**前后对称**的，看不见的背面最多和正面差不多厚。
    所以这里不让深度一步跳超过 k 倍：反复做"若 z 大于四邻域最小值的 k 倍，
    就把 z 压回 k 倍"，跳变会被摊成一段有限的斜坡（物体的厚度），而不是无限长的尾。

    对真正持续后退的地面（每步比值 ~1.02）完全无影响。
    """
    z = np.array(z, dtype=np.float32, copy=True)
    if k <= 1.0:
        return z
    for _ in range(max(1, int(iters))):
        nb = np.minimum.reduce([
            np.roll(z, 1, 0), np.roll(z, -1, 0),
            np.roll(z, 1, 1), np.roll(z, -1, 1),
        ])
        limit = nb * k
        over = z > limit
        if not over.any():
            break
        z[over] = limit[over]
    return z



def build_solid(P: np.ndarray, uv: np.ndarray, mask: np.ndarray, z: np.ndarray,
                depth_factor: float = 1.0) -> dict | None:
    """把一个近景物体做成**封闭体**：正面 + 镜像背面 + 封边。

    为什么：现在剪影处是"剪掉四边形"，物体只剩一张朝向相机的薄膜；
    为了不露出洞，又用裙边往远处挤一段 —— 短了遮不住，长了就是"长尾"。

    真实物体是**前后对称**的：看不见的背面，最多和正面差不多厚。
    所以这里直接把正面沿"物体中位深度平面"镜像出一层背面（z_back = 2*z_med - z_front），
    再把四周的边缘环连起来封口。这样从侧面看是一个实体，不再需要往远处拉。

    depth_factor < 1 时背面会浅一些（像压扁的实体），用于很薄的物体（栏杆、树枝）。
    """
    ys, xs = np.nonzero(mask)
    if ys.size < 24:
        return None
    sub = np.zeros_like(mask, dtype=bool)
    sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    sub &= mask

    zf = z[sub]
    zmed = float(np.median(zf))
    pos, uvx, nrm, idx = [], [], [], []
    vid = {}
    h, w = mask.shape

    # 正面：用物体自己的深度
    order = [(int(y), int(x)) for y, x in zip(*np.nonzero(sub))]
    for i, (y, x) in enumerate(order):
        vid[(y, x)] = i
        pos.append(P[y, x].copy())
        uvx.append(uv[y, x].copy())
        nrm.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    nf = len(order)

    # 背面：沿视图方向镜像到中位深度平面另一侧
    back = {}
    for i, (y, x) in enumerate(order):
        p0 = P[y, x]
        z0 = float(z[y, x])
        # **镜像深度必须限幅**：树冠/枝叶的深度有噪声，`2*z_med - z` 会把背面
        # 甩到极远（甚至负深度），从空中看就是一堆从中心射出的黑色尖刺（实测截图）。
        # 物体厚度取中位深度的 ±55%，够了。
        dz = (zmed - z0) * float(depth_factor)
        dz = float(np.clip(dz, -0.55 * zmed, 0.55 * zmed))
        zb = zmed + dz
        pb = p0 * (zb / max(1e-6, z0))
        back[(y, x)] = nf + i
        pos.append(pb.astype(np.float32))
        uvx.append(uv[y, x].copy())
        nrm.append(np.array([0.0, 0.0, -1.0], dtype=np.float32))

    # 正面/背面的四边形
    for (y, x) in order:
        if (y + 1, x) in vid and (y, x + 1) in vid and (y + 1, x + 1) in vid:
            a, b, c, d = vid[(y, x)], vid[(y + 1, x)], vid[(y, x + 1)], vid[(y + 1, x + 1)]
            idx += [a, b, c, c, b, d]
            a, b, c, d = back[(y, x)], back[(y + 1, x)], back[(y, x + 1)], back[(y + 1, x + 1)]
            idx += [a, c, b, c, d, b]          # 背面反向绕，保证外法线朝外

    # 封边：把"正面的边界像素"与它对应的背面连成一条环带
    for (y, x) in order:
        for dy, dx in ((1, 0), (0, 1)):
            ny2, nx2 = y + dy, x + dx
            if (ny2, nx2) in vid:
                continue                        # 内部边，不需要封
            f1, f2 = vid[(y, x)], back[(y, x)]
            f3, f4 = vid[(ny2, nx2)] if (ny2, nx2) in vid else None, back[(ny2, nx2)] if (ny2, nx2) in vid else None
            if f3 is None:
                continue
            idx += [f1, f2, f3, f3, f2, f4]

    if not idx:
        return None
    return {
        "name": "solid",
        "pos": np.asarray(pos, dtype=np.float32).reshape(-1, 3),
        "nrm": np.asarray(nrm, dtype=np.float32).reshape(-1, 3),
        "uv": np.asarray(uvx, dtype=np.float32).reshape(-1, 2),
        "index": np.asarray(idx, dtype=np.uint32),
    }


def build_top_skin(P: np.ndarray, uv: np.ndarray, valid: np.ndarray,
                    cell: float = 1.8, max_cells: int = 190,
                    smooth: int = 1, rgb: np.ndarray | None = None,
                    extent: float = 240.0, tile_uv: float = 32.0,
                    hole_poly: tuple | None = None) -> dict | None:
    """从点云生成一张"俯视高度图"顶面，供飞行时从上方看。

    为什么需要它：v2 是单视图重建，只有一层**朝前的壳**。平视没有问题，
    但飞到高处往下看，看到的是这层壳的背面 —— 竖直拉长的岩壁、被剪开的空洞、
    透过去是天空（实测 56m 俯拍就是"一片竖直岩墙 + 白洞"）。

    做法（标准的"点云 → 高度图"）：
      1. 把每个有效像素投影到 (x,z) 网格里，**每格只保留 y 最大**的那个点
         （从上方看，能看到的就是最高的那层表面），同时记住它的 UV；
      2. 空格用最近邻补（`distance_transform_edt` 的索引就是"最近的有点的格"）；
      3. 轻微平滑，避免逐点噪声变成刺；
      4. 用这张网格建三角面，UV 用该格那个点的 UV —— 于是顶面天然带着照片的纹理，
         高空中看下来就是"同一个峡谷的正俯视"，而不是另造一套颜色。

    这只在**高空**可见（前端按视点高度开关），平视时它正好藏在墙顶轮廓里。
    """
    from scipy import ndimage
    if not valid.any() or P.shape[0] < 4:
        return None
    xs = P[..., 0][valid]
    ys = P[..., 1][valid]
    zs = P[..., 2][valid]
    uvs = uv[valid]
    if xs.size < 64:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    z0, z1 = float(zs.min()), float(zs.max())
    # 俯视专用：网格要扩出照片足迹，否则足迹外没有顶面（俯视时会露出照片壳的竖条）
    if extent > 0:
        x0, x1 = x0 - extent, x1 + extent
        z0, z1 = z0 - extent, z1 + extent
    if x1 - x0 < 2 or z1 - z0 < 2:
        return None
    nx = int(np.clip(round((x1 - x0) / cell) + 1, 8, max_cells))
    nz = int(np.clip(round((z1 - z0) / cell) + 1, 8, max_cells))

    ix = np.clip(((xs - x0) / max(1e-6, x1 - x0) * (nx - 1)).astype(np.int32), 0, nx - 1)
    iz = np.clip(((zs - z0) / max(1e-6, z1 - z0) * (nz - 1)).astype(np.int32), 0, nz - 1)

    hgt = np.full((nz, nx), -1e9, dtype=np.float32)
    huv = np.zeros((nz, nx, 2), dtype=np.float32)
    # 按 y 升序写入 → 后写的（更高的）覆盖前面的，天然得到"每格最高点"
    order = np.argsort(ys)
    hgt[iz[order], ix[order]] = ys[order]
    huv[iz[order], ix[order]] = uvs[order]

    filled = hgt > -1e8
    if filled.sum() < 16:
        return None
    # 空格用最近的有值格补（索引数组就是最近邻）
    dist, (iy_nn, ix_nn) = ndimage.distance_transform_edt(
        ~filled, return_indices=True, sampling=(cell, cell))
    hgt_raw = hgt.copy()
    hgt = hgt_raw[iy_nn, ix_nn]
    huv = huv[iy_nn, ix_nn]
    if extent > 0:
        # 区外沿边界高度向"地面基准"缓慢收敛，并叠低频起伏。
        #
        # **距离必须用连续量**：`distance_transform_edt` 给的是逐格的离散距离，
        # 相邻格差一格 → 高度一格格跳 → 从空中看就是一道 V 形台阶（正好是足迹形状）。
        # 这里改成到"照片足迹矩形"的解析距离（区内为 0），完全连续。
        hw = (xs.max() - xs.min()) * 0.5
        hd = (zs.max() - zs.min()) * 0.5
        cx = (xs.max() + xs.min()) * 0.5
        cz = (zs.max() + zs.min()) * 0.5
        gx1 = x0 + (np.arange(nx, dtype=np.float32) / max(1, nx - 1)) * (x1 - x0)
        gz1 = z0 + (np.arange(nz, dtype=np.float32) / max(1, nz - 1)) * (z1 - z0)
        ddx = np.maximum(np.abs(gx1[None, :] - cx) - hw, 0.0)
        ddz = np.maximum(np.abs(gz1[:, None] - cz) - hd, 0.0)
        dist_c = np.sqrt(ddx * ddx + ddz * ddz)          # 区内 0，区外连续递增
        k = 1.0 - np.exp(-dist_c / 120.0)
        base_h = float(np.median(hgt_raw[filled]))
        hgt = hgt * (1.0 - k) + base_h * k
        gx2 = x0 + (np.arange(nx, dtype=np.float32) / max(1, nx - 1)) * (x1 - x0)
        gz2 = z0 + (np.arange(nz, dtype=np.float32) / max(1, nz - 1)) * (z1 - z0)
        # **多层 fBm 起伏**，不是两道正弦波：从空中看，单频波纹/平地都是一张板，
        # 这正是"起飞后地面变白/发灰、毫无内容"的根因。振幅随距离增长，
        # 贴近照片的地方保持安静（那里有真实几何，不该被程序化地形打扰）。
        X2 = gx1[None, :]
        Z2 = gz1[:, None]
        ramp = np.clip(dist_c / 260.0, 0.0, 1.0)
        # 振幅按"从 120m 高空能看出地貌"来定：14m 在 500m 跨度上几乎是平的
        # （实测起飞后是一张均匀的褐色板）。现在最大约 45m，并补了一层中频。
        fbm = (30.0 * np.sin(X2 * 0.0038 + Z2 * 0.0033 + 0.7)
               + 16.0 * np.sin(X2 * 0.0093 - Z2 * 0.0071 + 2.1) * np.cos(Z2 * 0.0088 + 0.3)
               + 7.0 * np.sin(X2 * 0.021 + Z2 * 0.017 + 4.4)
               + 2.6 * np.sin(X2 * 0.047 - Z2 * 0.039 + 1.2)
               # 更高的两档：只加细节不抬整体，否则会变成一团团隆起
               + 1.1 * np.sin(X2 * 0.093 + Z2 * 0.081 + 0.9)
               + 0.45 * np.sin(X2 * 0.19 - Z2 * 0.17 + 3.3))
        hgt = hgt + fbm * ramp
    if smooth > 0:
        hgt = ndimage.uniform_filter(hgt, size=(3, 3), mode="nearest")

    # ---- 高度限幅 + 更强的平滑 -----------------------------------------
    # 森林/枝叶这类场景，深度在像素间噪声很大；"每格取最高点"会把树冠的离群值
    # 当成地面 → 从空中看是一片从中心射出的尖刺（实测 winter 那张）。
    # 做法：按分位截断，再加大平滑；俯视图本来就不需要像素级锐利。
    _h = hgt[np.isfinite(hgt)]
    if _h.size > 32:
        _lo, _hi = np.percentile(_h, 2), np.percentile(_h, 98)
        _span = max(1.0, float(_hi - _lo))
        hgt = np.clip(hgt, _lo - 0.25 * _span, _hi + 0.25 * _span)
        hgt = ndimage.uniform_filter(hgt, size=(5, 5), mode="nearest")

    pos = np.zeros((nz, nx, 3), dtype=np.float32)
    gx = x0 + (np.arange(nx, dtype=np.float32) / max(1, nx - 1)) * (x1 - x0)
    gz = z0 + (np.arange(nz, dtype=np.float32) / max(1, nz - 1)) * (z1 - z0)
    pos[..., 0] = gx[None, :]
    pos[..., 1] = hgt
    pos[..., 2] = gz[:, None]

    P2 = pos.reshape(-1, 3)
    U2 = huv.reshape(-1, 2)

    # ---- UV：把"照片里的地面区域"当作可平铺纹理 ---------------------
    # 直接用 huv（每个格子借一个像素的 UV）有两个问题：
    #   ① 同一片地面来自照片不同位置，颜色跳；② 起伏大处会拉出条纹。
    # 改成按世界坐标平铺，但**限制在照片的地面 bbox 内** —— 这样绝不会把天空采进地面。
    if tile_uv:
        gxs = uv[..., 0][valid]
        gys = uv[..., 1][valid]
        if gxs.size:
            u0, u1 = float(np.percentile(gxs, 5)), float(np.percentile(gxs, 95))
            v0, v1 = float(np.percentile(gys, 5)), float(np.percentile(gys, 95))
            # 保证有足够的采样范围，太窄就退化成一条线
            if (u1 - u0) < 0.08:
                u0, u1 = max(0.0, u0 - 0.04), min(1.0, u1 + 0.04)
            if (v1 - v0) < 0.08:
                v0, v1 = max(0.0, v0 - 0.04), min(1.0, v1 + 0.04)
            # 在"地面 bbox"里挑**最平坦的一块**当纹理源：
            # 直接铺整段地面会把石头/树一起铺满，九米一重复非常扎眼（实测）。
            # 做法：把 bbox 划成若干窗口，按亮度标准差挑最平滑的那个。
            if rgb is not None and rgb.size:
                H2, W2 = rgb.shape[:2]
                _lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
                best = None
                for wv in (0.22, 0.30, 0.38):
                    for wy in (0.22, 0.30):
                        for du in np.linspace(0.0, max(0.0, 1.0 - wv), 5):
                            for dv in np.linspace(0.0, max(0.0, 1.0 - wy), 4):
                                su0, su1 = du, du + wv
                                sv0, sv1 = dv, dv + wy
                                # 只接受"整块都在地面里"的窗口
                                m_win = ((gxs >= su0) & (gxs <= su1) &
                                         (gys >= sv0) & (gys <= sv1))
                                if m_win.mean() < 0.92:
                                    continue
                                iy0, iy1 = int(sv0 * (H2 - 1)), int(sv1 * (H2 - 1))
                                ix0, ix1 = int(su0 * (W2 - 1)), int(su1 * (W2 - 1))
                                if iy1 - iy0 < 16 or ix1 - ix0 < 16:
                                    continue
                                patch = _lum[iy0:iy1, ix0:ix1]
                                sd = float(patch.std())
                                if best is None or sd < best[0]:
                                    best = (sd, su0, su1, sv0, sv1)
                if best is not None:
                    _, u0, u1, v0, v1 = best
            T = float(tile_uv) if isinstance(tile_uv, (int, float)) and tile_uv > 0 else 32.0
            uu = np.mod(P2[:, 0] / T, 1.0) * (u1 - u0) + u0
            vv = np.mod(P2[:, 2] / T, 1.0) * (v1 - v0) + v0
            U2 = np.stack([uu, vv], axis=1).astype(np.float32)
    # 法线用高度场梯度算。法线一律朝上时 ndl 恒定 → 什么起伏都看不出来，
    # 整片就是均匀色块（换色也没用）。有了坡面法线，从空中才有山脊/谷地的明暗。
    gz_, gx_ = np.gradient(hgt, float(cell), float(cell))
    nrm_grid = np.stack([-gx_, np.ones_like(hgt), -gz_], axis=-1).astype(np.float32)
    nrm_grid /= np.maximum(np.linalg.norm(nrm_grid, axis=-1, keepdims=True), 1e-6)
    N2 = nrm_grid.reshape(-1, 3)

    # 顶点色：用**参与该格高度计算的那个像素**的 RGB。
    # 高空看地表本来就该是低频色块；用 UV 去采样照片贴图，相邻格的 UV 差异被放大成
    # 一圈圈放射状拉伸（实测）。顶点色没有这个问题，也不依赖贴图 mip 链。
    C2 = None
    if rgb is not None and rgb.size:
        # 颜色**不逐格借像素**，而是"地面平均色 × 位置噪声 × 高程明暗"。
        # 逐格借像素看似忠实，实际取到的多半是崖壁阴影色 —— 高空看下去是
        # 一片黑色条纹（实测）。俯视本来就该是低频的地貌色块。
        # 注意形状：这里拿到的 valid/uv 是**降采样后**的网格，
        # 不能直接用 rgb[valid]（全分辨率的面具对不上，实测 IndexError）。
        # 用 uv 反查像素取平均色，尺寸就自然对上了。
        H2, W2 = rgb.shape[:2]
        u_v = uv[valid]
        if u_v.size:
            iy2 = np.clip((u_v[:, 1] * (H2 - 1)).astype(np.int32), 0, H2 - 1)
            ix2 = np.clip((u_v[:, 0] * (W2 - 1)).astype(np.int32), 0, W2 - 1)
            # 用 **75 分位亮度**的那一档颜色，而不是均值：
            # 均值会把大片阴影（峡谷照的暗部）算进去，俯视整片发灰发闷（实测是泥浆色）。
            _px = np.clip(rgb[iy2, ix2], 0, 1)
            _lum = 0.2126 * _px[:, 0] + 0.7152 * _px[:, 1] + 0.0722 * _px[:, 2]
            _pick = _px[np.argsort(_lum)[min(len(_lum) - 1, int(len(_lum) * 0.75))]]
            base_c = _pick.astype(np.float32)
        else:
            base_c = np.array([0.5, 0.42, 0.34], dtype=np.float32)
        hmin, hmax = float(hgt.min()), float(hgt.max())
        hr = max(1e-3, hmax - hmin)
        up = np.clip((P2[:, 1] - hmin) / hr, 0, 1)
        n1 = np.sin(P2[:, 0] * 0.013 + 1.1) * np.cos(P2[:, 2] * 0.015 - 0.5)
        n2 = np.sin(P2[:, 0] * 0.06 + P2[:, 2] * 0.045 + 2.0)
        v = (0.94 + 0.10 * n1 + 0.04 * n2) * (0.82 + 0.36 * up)
        if extent > 0:
            v = v * np.clip(1.0 - dist_c.reshape(-1) / (extent * 5.0), 0.85, 1.0)
        # 0.62：俯视是"看地貌"，不是"看原片亮度"。不压回来，亮场景（雪山/灯柱那种
        # 高调照片）会整片糊成白板 —— 即"起飞后地面变全白"。
        # **把坡面明暗烘进顶点色**：场景太阳很低，纯靠 uSunDir 算 ndl 时
        # 朝上的面几乎一样亮，起伏也看不出来（实测起飞后是一张平色板）。
        # 这里用固定的"制图光"方向算一次 lambert，乘进颜色里。
        Ld = np.array([0.45, 0.78, -0.44], dtype=np.float32)
        Ld /= np.linalg.norm(Ld)
        ndl_fixed = np.clip(N2 @ Ld, 0.0, 1.0)
        shade = 0.42 + 0.85 * ndl_fixed
        # **多尺度顶点色噪声**（无接缝、无重复）。
        # 试过"把照片地面区域当纹理平铺"，必然出现拼布感（一块块重复的石头/树，
        # 接缝明显）。顶点色噪声没有这个问题，代价是细节是"统计意义上像"而不是"真实纹理"。
        # 四档尺度：25m 大块色斑 / 8m 斑块 / 3m 颗粒 / 1m 细砂。
        X3 = P2[:, 0]
        Z3 = P2[:, 2]
        n_big = np.sin(X3 * 0.041 + Z3 * 0.037 + 1.1) * np.cos(Z3 * 0.033 - 0.6)
        n_mid = np.sin(X3 * 0.13 - Z3 * 0.11 + 2.4) * np.cos(X3 * 0.09 + Z3 * 0.12 + 0.2)
        n_fine = np.sin(X3 * 0.37 + Z3 * 0.41 + 4.1) * np.cos(Z3 * 0.33 - X3 * 0.29 + 1.3)
        n_sand = np.sin(X3 * 1.13 - Z3 * 0.97 + 0.7) * np.cos(X3 * 0.91 + Z3 * 1.07 + 2.9)
        grain = (1.0
                 + 0.16 * n_big
                 + 0.12 * n_mid
                 + 0.07 * n_fine
                 + 0.035 * n_sand)
        # 轻微冷暖漂移：同一片地不该是同一个色相
        warm = 1.0 + 0.05 * np.sin(X3 * 0.017 + Z3 * 0.013 + 3.1)
        v2 = v * shade * grain
        _c0 = np.clip(base_c[0] * v2 * warm * 0.82, 0, 1)
        _c2 = np.clip(base_c[2] * v2 / warm * 0.82, 0, 1)
        C2 = np.stack([_c0, np.clip(base_c[1] * v2 * 0.82, 0, 1), _c2,
                       np.ones_like(v)], axis=1).astype(np.float32)

    # ---- 照片足迹打洞 --------------------------------------------------
    # 站在原点时，视锥内的地面必须**就是照片本身**（纹理/明暗都要一致）。
    # 这一层如果铺到足迹上，就会把近景壳盖住 —— 表现是"什么纹理都没了"。
    hole_mask = None
    if hole_poly is not None:
        _zn, _zf, _th = hole_poly
        _X = pos[..., 0]
        _Z = pos[..., 2]
        hole_mask = ((-_Z >= _zn * 0.90) & (-_Z <= _zf * 1.02) &
                     (np.abs(_X) <= (-_Z) * _th * 1.04) & (_Z <= 0))
        # 内缩一格：宁可留细缝，也不能盖住照片
        hole_mask = ndimage.binary_erosion(hole_mask, np.ones((3, 3), bool))

    idx2 = []
    for j in range(nz - 1):
        row = j * nx
        for i in range(nx - 1):
            if (hole_mask is not None and hole_mask[j, i] and hole_mask[j, i + 1]
                    and hole_mask[j + 1, i] and hole_mask[j + 1, i + 1]):
                continue                       # 照片足迹：这里归近景壳，地面层不画
            a = row + i
            b = a + 1
            c = a + nx
            d = c + 1
            idx2.extend([a, c, b, b, c, d])
    if not idx2:
        return None
    return {"name": "top", "pos": P2, "nrm": N2, "uv": U2,
            "color": C2, "index": np.asarray(idx2, dtype=np.uint32)}


def build_photo_wall(cam, dist: float = 8.0, expand: float = 1.0) -> dict:
    """把整张照片做成一面"照片墙"（正对视线的四边形）。

    什么时候用：画面里几乎没有地面（云海、星空、从太空拍的地球、纯天空）。
    这类图像的"深度"没有物理意义 —— 硬建三维只会得到一团黑色拉丝，
    实测验收批次里最惨的两张就是这种。诚实的做法是退回"平贴"，观感等于
    一张高分辨率照片挂在正前方，至少不毁。
    """
    rays = cam.rays()
    h, w = rays.shape[:2]
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    pos, uv = [], []
    for (j, i) in corners:
        d = rays[j, i]
        p = d * dist
        pos.extend([float(p[0]), float(p[1]), float(-p[2])])
        # expand>1 会把 UV 推�� [0,1] 之外、由 CLAMP 拉成边条（实测很难看），
        # 所以默认 1.0：墙严格等于照片视场。让墙填满屏幕靠前端把视点放到原点
        # （见 app.js 里 wall 世界的 eyeHeight=0）—— 视锥比照片窄时自然是裁切，不会露灰。
        u = 0.5 + (i / (w - 1.0) - 0.5) * expand
        v = 0.5 + (j / (h - 1.0) - 0.5) * expand
        uv.extend([u, v])
    idx = [0, 2, 1, 1, 2, 3]
    return {
        "name": "backdrop",
        "pos": np.asarray(pos, dtype=np.float32).reshape(-1, 3),
        "nrm": np.tile(np.array([0, 0, 1], dtype=np.float32), (4, 1)),
        "uv": np.asarray(uv, dtype=np.float32).reshape(-1, 2),
        "index": np.asarray(idx, dtype=np.uint32),
    }


def build_ground_skirt(P: np.ndarray, ground: np.ndarray, uv: np.ndarray,
                       radius: float = 400.0, cells: int = 96,
                       n_palette: int = 12, inner_frac: float = 0.06,
                       footprint: tuple | None = None, amp: float = 2.2,
                       rgb: np.ndarray | None = None, eye: float = 1.7) -> dict | None:
    """重建区域之外补一整圈地面，让"视锥外的灰白"变成"远处地面 + 雾"。

    为什么需要它：v2 的几何止于照片的投影梯形，梯形外什么都没有 ——
    实测截图里画面左右下三块纯色、以及山脚那条刀切硬边，都是这个边界。
    光靠外扩（模型补全）只能把边界推远，推不远的地方还是要有东西兜住。

    做法：
      * 地面高度取"地面像素的 y 中位数"（相机系里约等于 -eye）；
      * 内圈很小（inner_frac*radius，默认 24m）：留太大就会在"照片边缘到裙边之间"
        拉出一条空带 —— 实测那张沙漠截图里脚下的大片空白就是这么来的（旧值 112m）；
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
    # 高度要钳在"相机脚下的地面"附近（世界系里约 -eye）。
    # 直接取地面像素 y 的中位数会翻车：峡谷照里被判成"地面"的像素大量在台地顶部，
    # 中位数把裙边抬到几十米高，于是它在画面里变成一张**横贯视野的亮平面**
    # （实测：正视图正中那条亮横线就是它）。
    gy_raw = float(np.median(P[ground][:, 1])) - 0.20
    ref = -float(eye)
    gy = float(np.clip(gy_raw, ref - 1.5, ref + 1.5))
    # 用地面像素的 UV 做色板
    g_uv = uv[ground]
    if g_uv.shape[0] < 8:
        return None
    # 色板均匀抽样，然后**取最亮的那一档**。
    # 取中位数会挑到阴影里的地面（金色时刻一片暗褐），整个裙边变成近黑色的地皮
    # （实测：背后一整片近黑 + 远处白带）。远景地面本来就该是平均日照色。
    order = np.argsort(g_uv[:, 1] * 1.0 + g_uv[:, 0] * 0.37)
    pick = np.linspace(0, g_uv.shape[0] - 1, min(n_palette * 3, g_uv.shape[0])).astype(int)
    palette = g_uv[order[pick]]
    if rgb is not None and rgb.size and palette.size:
        lum = []
        for (u, v) in palette:
            iy = int(np.clip(v * (rgb.shape[0] - 1), 0, rgb.shape[0] - 1))
            ix2 = int(np.clip(u * (rgb.shape[1] - 1), 0, rgb.shape[1] - 1))
            c = rgb[iy, ix2]
            lum.append(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
        order2 = np.argsort(lum)
        palette = palette[[order2[min(len(order2) - 1, int(len(order2) * 0.70))]]]

    n = int(cells)
    lin = np.linspace(-radius, radius, n + 1, dtype=np.float32)
    X, Z = np.meshgrid(lin, lin)                 # Z 向 -Z 是前方，这里用世界 z
    Z = Z - radius * 0.35                        # 略向前铺，画面里主要是前方
    inner = radius * inner_frac
    if footprint is not None:
        # 用**照片投影的梯形**当洞，而不是圆：圆的半径要么盖住照片自己的地面
        # （把细节挡掉），要么在照片边缘外留一条空带（用户截图里脚下那片空白）。
        # footprint = (z_near, z_far, tan_h)，与世界坐标同向（前方 z 为负）。
        z_near, z_far, tan_h = footprint
        half_w = np.maximum(1e-3, -Z) * tan_h
        hole = (-Z >= z_near * 0.8) & (-Z <= z_far * 1.05) & (np.abs(X) <= half_w * 1.02)
        hole &= (Z <= 0)
    else:
        hole = (np.abs(X) < inner) & (np.abs(Z + radius * 0.35) < inner)

    pos, nrm, uvout, index = [], [], [], []
    vid = np.full(X.shape, -1, dtype=np.int64)
    for j in range(n + 1):
        for i in range(n + 1):
            if hole[j, i]:
                continue
            vid[j, i] = len(pos) // 3
            # 低频起伏：纯平面在未来任何视角都读作"空地"，加一点沙丘式起伏
            # 才有了地面的体积感（±amp 米，尺度约 80m）
            dune = (np.sin(X[j, i] * 0.021 + 1.3) * np.cos(Z[j, i] * 0.017 - 0.7)
                    + 0.5 * np.sin(X[j, i] * 0.0071 + Z[j, i] * 0.0093))
            pos.extend([float(X[j, i]), gy + float(dune) * amp, float(Z[j, i])])
            nrm.extend([0.0, 1.0, 0.0])
            # **统一取一个色板 UV**：早先按格子哈希取不同色，结果在掠射角下
            # 每个格子的颜色差异被放大成一圈圈同心橙环（实测），比纯色难看得多。
            uvout.extend([float(palette[palette.shape[0] // 2, 0]),
                          float(palette[palette.shape[0] // 2, 1])])

    for j in range(n):
        for i in range(n):
            a, b = vid[j, i], vid[j + 1, i]
            c, d = vid[j, i + 1], vid[j + 1, i + 1]
            if a < 0 or b < 0 or c < 0 or d < 0:
                continue
            index.extend([a, b, c, c, b, d])

    if not index:
        return None
    # 顶点色：用上面选出的那一个"日照地面"像素的 RGB。
    # 裙边的 UV 只能指向照片里的某一个点，采样出来是整片单色且偏色（实测偏暗红）；
    # 直接给顶点色反而可控 —— 前端用 mode 1 画，颜色就是这里给的值。
    if rgb is not None and rgb.size and palette.size:
        u0, v0 = float(palette[0, 0]), float(palette[0, 1])
        iy = int(np.clip(v0 * (rgb.shape[0] - 1), 0, rgb.shape[0] - 1))
        ix0 = int(np.clip(u0 * (rgb.shape[1] - 1), 0, rgb.shape[1] - 1))
        c = np.clip(np.asarray(rgb[iy, ix0], dtype=np.float32), 0, 1) * 0.88
        # **逐顶点加变化**：整片单色在近处读作"一块色板"（实测）。
        # 低频（~90m）给大片地面分层次，中频（~18m）压掉塑料感，另加轻微冷暖漂移。
        Pv = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
        n1 = np.sin(Pv[:, 0] * 0.011 + 1.3) * np.cos(Pv[:, 2] * 0.013 - 0.6)
        n2 = np.sin(Pv[:, 0] * 0.055 + Pv[:, 2] * 0.041 + 2.2)
        v = 1.0 + 0.13 * n1 + 0.05 * n2
        warm = 1.0 + 0.05 * np.sin(Pv[:, 0] * 0.006 + Pv[:, 2] * 0.004 + 0.8)
        colv = np.stack([c[0] * v * warm, c[1] * v, c[2] * v / warm,
                         np.ones_like(v)], axis=1).astype(np.float32)
    else:
        n = len(pos) // 3
        colv = np.tile(np.array([0.55, 0.45, 0.36, 1.0], dtype=np.float32), (n, 1))
    return {
        "name": "ground",
        "pos": np.asarray(pos, dtype=np.float32).reshape(-1, 3),
        "nrm": np.asarray(nrm, dtype=np.float32).reshape(-1, 3),
        "uv": np.asarray(uvout, dtype=np.float32).reshape(-1, 2),
        "color": colv,
        "index": np.asarray(index, dtype=np.uint32),
    }


def build_scatter_rocks(palette_uv: np.ndarray, ground_y: float, radius: float,
                        inner: float, count: int = 220, seed: int = 7) -> dict | None:
    """在裙边上撒一批低模石块，让"视锥外"有地物而不只是一块地面。

    参考效果（用户给的图）：远景是地貌，近中景散落着岩块、灌木。纯平面无论怎么雾化
    都读作"空"，必须有不规则的立体物提供视差与轮廓。

    做法：每块石头 = 上下两个不规则环 + 顶部一点（10 个三角形），半径 0.3~2.2m，
    颜色/贴图取照片地面色板里的随机一个 UV（平色即可，远处看不清纹理）。
    """
    rng = np.random.default_rng(seed)
    if palette_uv is None or palette_uv.size == 0 or count <= 0:
        return None
    pos, nrm, uv, idx = [], [], [], []
    base = 0
    for k in range(int(count)):
        ang = rng.uniform(0, 2 * np.pi)
        # 半径按面积均匀撒（sqrt），避免全挤在内圈
        # 密度按"近多���少"分配：把 u 做平方，落点集中在中近景
        u = rng.uniform(0.0, 1.0) ** 2.2
        rr = (inner + 4.0) + u * (radius * 0.55 - inner - 4.0)
        cx, cz = np.sin(ang) * rr, np.cos(ang) * rr
        sc = float(np.clip(rng.normal(1.35, 0.7), 0.4, 3.2))        # 尺寸（米）
        h = sc * rng.uniform(0.55, 1.15)
        rot = rng.uniform(0, np.pi)
        nv = 6
        ring_lo = []
        ring_hi = []
        for v in range(nv):
            th = rot + 2 * np.pi * v / nv
            r0 = sc * rng.uniform(0.75, 1.15)
            r1 = sc * rng.uniform(0.35, 0.7)
            ring_lo.append((cx + np.sin(th) * r0, ground_y, cz + np.cos(th) * r0))
            ring_hi.append((cx + np.sin(th) * r1, ground_y + h, cz + np.cos(th) * r1))
        apex = (cx, ground_y + h * 1.15, cz)
        u_pick = palette_uv[rng.integers(0, palette_uv.shape[0])]
        for (x, y, z) in ring_lo + ring_hi + [apex]:
            pos.extend([float(x), float(y), float(z)])
        for i2 in range(nv):                                   # 侧面
            a = base + i2
            b = base + (i2 + 1) % nv
            c = base + nv + i2
            d = base + nv + (i2 + 1) % nv
            idx.extend([a, b, c, b, d, c])
        top = base + 2 * nv
        for i2 in range(nv):                                   # 顶盖
            a = base + nv + i2
            b = base + nv + (i2 + 1) % nv
            idx.extend([a, b, top])
        for i2 in range(2 * nv + 1):
            nrm.extend([0.0, 1.0, 0.0])
            uv.extend([float(u_pick[0]), float(u_pick[1])])
        base += 2 * nv + 1
    if not idx:
        return None
    return {"name": "scatter",
            "pos": np.asarray(pos, dtype=np.float32).reshape(-1, 3),
            "nrm": np.asarray(nrm, dtype=np.float32).reshape(-1, 3),
            "uv": np.asarray(uv, dtype=np.float32).reshape(-1, 2),
            "index": np.asarray(idx, dtype=np.uint32)}


def build(z: np.ndarray, cam, step: int = 1, eye: float = 1.7,
          far_split: float = 70.0, near_split: float = 12.0,
          sky: np.ndarray | None = None, edge_ratio: float = 2.4,
          flatten: bool = True, skirt_min: float = 0.12, skirt_max: float = 6.0,
          dist_min: float = 0.5, median_size: int = 3, ground_tol: float = 0.30,
          ground_smooth: int = 9,
          skirt_cap: float = 3.0,
          stretch_clamp: float = 12.0, strong_median: int = 7,
          quantize: float = 0.0, rgb: np.ndarray | None = None,
          repr_mode: str = "auto", scatter_thr: float = 0.002,
          cliff_thr: float = 0.010, ground_radius: float = 400.0,
          seg_edge_tau: float = 0.10, quad_stretch_max: float = 0.0,
          depth_cap: float = 1.35,
          solid_objs: bool = True, sem_obj: np.ndarray | None = None,
          sem_ground: np.ndarray | None = None,
          # 默认关闭：试过"地面像素占比 < 0.4%"当"非风景照"的判据，实测**不成立** ——
          # 云海/星空/太空地球这几张图的地面占比也有 3.7~5.2%（压平判据把它们误判成地面），
          # 而"远景占比 / 纵深动态范围"在好坏图上完全不分离（远景占比全是 0，
          # 纵深比好的 29、坏的 80~148，但雪山这种好图也有 102）。
          # 结论：这一层要靠**语义**（SegFormer 的 sky 类占比）才能判，不是数值统计能到的。
          photo_wall_frac: float = 0.0, photo_wall: bool = False,
          card_flatten: bool = True,
          ground_cells: int = 140, scatter_rocks: int = 0,
          top_skin: bool = True, top_skin_cell: float = 1.8) -> dict:
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
    if ground_smooth > 1:
        z = smooth_ground_rows(z, rays_full, width=ground_smooth)

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

    from scipy import ndimage as _ndi
    # ---- 补掉"无效深度"造成的一像素洞 -------------------------------
    # 剪影处单目深度偶尔给 0/NaN，这些像素被排除在 valid 之外 → 任何层都不会在那里建面
    # → 透出天穹底色（fogColor 的灰）。地面视角里山脊上那条一格宽、颜色恒定的灰带
    # 就是这么来的（逐项排除过顶面/裙边/分割缝/远景层多占一格）。
    # 只补"离有效像素 2px 以内"的洞：更远的无效区是真的没数据，不该凭空造。
    if (~valid).any():
        dd, (ny, nx2) = _ndi.distance_transform_edt(~valid, return_indices=True,
                                                      sampling=(1.0, 1.0))
        fill = (~valid) & (dd <= 2.0)
        if fill.any():
            zs = zs.astype(np.float32)
            zs[fill] = zs[ny[fill], nx2[fill]]
            valid = valid | fill
    # 断崖掩码要在表示选择之前算：它是"该用网格还是点云"两个判据之一
    cliff, ratio, edge_str = cliff_mask(zs, step, edge_ratio)

    # ---- 深度梯度限幅：去掉剪影处的"长尾" --------------------------
    # 必须在分层、cliff 判定**之前**做：几何本身就是按 zs 建的，
    # 先限幅，后面的分层/剪面也都在更合理的深度上判断。
    if depth_cap < 100.0:
        zs = cap_depth_gradient(zs, k=depth_cap, iters=12)
        # 用当前分辨率的 rays（build 里已经按 step 降采样过），
        # 不能重新调 cam.rays()：那是全分辨率，形状对不上（实测 ValueError）。
        P = rays * zs[..., None]
        P[..., 2] *= -1.0
        fwd = -P[..., 2]

    # ---- 非风景照：退回"照片墙" -----------------------------------
    # 判据用两个便宜的统计量：能识别成地面的像素几乎没有，且地平线以上占绝大多数。
    # 云海、星空、太空地球这类图都会命中。
    gfrac = float(ground.mean())
    skyfrac = float((rays[..., 1] > 0.02).mean())
    # 触发条件由调用方给（语义模型判定"这不是风景照"），不再用便宜的数值判据 ——
    # 那些判据实测不成立，见 photo_wall_frac 的注释。
    if photo_wall:
        wall = build_photo_wall(cam)
        return {"layers": [wall], "stats": {
            "wall": {"vertices": 4, "triangles": 2, "skirt_quads": 0,
                     "pixels": int(valid.sum()),
                     "depth_p10": 8.0, "depth_p90": 8.0},
            "_global": {"grid": [int(hh), int(ww)], "step": int(step), "repr": "photo-wall",
                        "cliff_frac": 0.0, "ground_frac": round(gfrac, 4),
                        "sky_frac": round(skyfrac, 3), "scatter": 0.0,
                        "segments": 0, "triangles": 2, "vertices": 4}}}

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

        # **点云路径也要有地面层**：否则飞到空中只有点云，
        # 俯视时会看到"点从中心炸开"的尖刺（实测截图），四周一片空白。
        # 点云没有"壳"的概念，地面层就成了空中唯一能看的东西。
        p_layers = [pts]
        try:
            _gm = valid
            if sem_ground is not None:
                _g2 = np.asarray(sem_ground)
                # 掩码必须与 valid 同形（= zs.shape）。这里**曾经写的是 seg.shape**，
                # 而 seg 要等地物分割（本函数后半段）才赋值 → 点云分支必然抛
                # UnboundLocalError，被下面宽泛的 except 吞掉 → v2.15.0 的
                # "点云路径也建地面层"在生产路径（api.py 总传 sem_ground）静默失效，
                # 空中俯视时只剩从中心炸开的点云尖刺。
                _tgt = zs.shape
                if _g2.shape != _tgt:
                    _iy2 = (np.arange(_tgt[0]) * _g2.shape[0] // _tgt[0]).astype(np.int32)
                    _ix2 = (np.arange(_tgt[1]) * _g2.shape[1] // _tgt[1]).astype(np.int32)
                    _g2 = _g2[_iy2][:, _ix2]
                if _g2.any() and int((_g2 & valid).sum()) > 500:
                    _gm = valid & _g2
            _fv = fwd[valid]
            _zn = float(np.percentile(_fv, 2)) if _fv.size else 2.0
            _zf = float(np.percentile(_fv, 92)) if _fv.size else 60.0
            _th = float(np.max(np.abs(rays[..., 0]) / np.maximum(np.abs(rays[..., 2]), 1e-6)))
            _skin = build_top_skin(P, uv, _gm, cell=top_skin_cell * 1.6, rgb=rgb,
                                   extent=float(ground_radius) * 1.15, max_cells=300,
                                   hole_poly=(_zn, _zf, _th))
            if _skin is not None:
                p_layers.append(_skin)
        except Exception as e:                      # noqa: BLE001
            print(f"[ldi] 点云路径的地面层构建失败（不影响点云）：{e}", flush=True)

        return {"layers": p_layers, "stats": {
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

    # ---- 近景物体封闭化（solid）------------------------------------
    # 判据保守：面积适中 + 中位深度近 + 明显比周围近。
    # 满足的物体单独做成封闭体（正面+镜像背面+封边），并从近景层里剔除，
    # 这样它不再是一张需要裙边去遮的薄膜，也就没有"往远处拉"的长尾。
    solids: list[dict] = []
    dbg: list[dict] = []
    solid_mask = np.zeros_like(seg, dtype=bool)
    # **优先用语义物体掩码**：深度屏障分割在 lake/alpine 上分不出任何物体（实测），
    # 语义给的是"它是个东西"，与大小无关。按连通域切成一个个物体。
    sem_parts = []
    # 语义掩码是**原图分辨率**，而这里的 zs 是 step 降采样后的网格 —— 形状不一致，
    # 之前直接因 shape 不等而被跳过（实测一个实体都没生成）。按 step 抽稀即可。
    if sem_obj is not None and sem_obj.shape != seg.shape:
        # 最近邻重采样到网格分辨率。不能用 `[::step]` 切片：
        # 尺寸常有 1 像素的舍入差（实测 565 vs 283*2=566），切片会整体错位。
        _iy = (np.arange(seg.shape[0]) * sem_obj.shape[0] // seg.shape[0]).astype(np.int32)
        _ix = (np.arange(seg.shape[1]) * sem_obj.shape[1] // seg.shape[1]).astype(np.int32)
        sem_obj = sem_obj[_iy][:, _ix]
    if os.environ.get("PW_DEBUG_SOLID"):
        print(f"[solid] sem_obj={'None' if sem_obj is None else sem_obj.shape}"
              f" seg={seg.shape} step={step} solid_objs={solid_objs}", flush=True)
    if solid_objs and sem_obj is not None and sem_obj.shape == seg.shape and sem_obj.any():
        lab_s, n_s = ndimage.label(sem_obj, structure=np.ones((3, 3), bool))
        for k in range(1, n_s + 1):
            m = (lab_s == k) & valid
            a = int(m.sum())
            if a < 200 or a > 0.30 * max(1, int(valid.sum())):
                continue
            if float(np.median(zs[m])) > 60.0:      # 太远的当远景，不做实体
                continue
            sem_parts.append(m)
    if solid_objs:
        area_total = max(1, int(valid.sum()))
        med_big = ndimage.median_filter(zs, size=31, mode="nearest")
        for lab in range(1, int(n_seg) + 1):
            m = (seg == lab) & valid
            a = int(m.sum())
            if a < 120 or a > 0.10 * area_total:
                continue
            zmed = float(np.median(zs[m]))
            if zmed > 45.0:
                continue
            nearfrac = float(((zs < 0.72 * med_big) & m).sum()) / max(1, a)
            dbg.append({"area": a, "zmed": round(zmed, 1), "near": round(nearfrac, 2)})
            if nearfrac < 0.28:
                continue
            S = build_solid(P, uv, m, zs)
            if S is not None:
                solids.append(S)
                solid_mask |= m

        for m in sem_parts:            # 语义物体
            S = build_solid(P, uv, m, zs)
            if S is not None:
                solids.append(S)
                solid_mask |= m

    if solids:
        # 合并成一个 solid 层（索引要整体偏移）
        pos = np.concatenate([x["pos"] for x in solids])
        nrm = np.concatenate([x["nrm"] for x in solids])
        uvc = np.concatenate([x["uv"] for x in solids])
        off = 0
        idxs = []
        for x in solids:
            idxs.append(x["index"] + off)
            off += x["pos"].shape[0]
        solid_layer = {"name": "solid", "pos": pos, "nrm": nrm, "uv": uvc,
                       "index": np.concatenate(idxs).astype(np.uint32)}
    else:
        solid_layer = None

    names = {0: "near", 1: "mid", 2: "backdrop"}
    # layers/stats 必须先初始化，再追加 solid 层（顺序反了会 UnboundLocalError）
    layers, stats = [], {}
    if solid_layer is not None:
        layers.append(solid_layer)
        stats["_solid_dbg"] = sorted(dbg, key=lambda d: -d["area"])[:12]
        stats["solid"] = {"vertices": int(solid_layer["pos"].shape[0]),
                          "triangles": int(solid_layer["index"].size // 3),
                          "pixels": int(solid_mask.sum())}
    # 近/远两层在交界处**互相多占一格**：只让远景层多占 1 格还不够，
    # 实测交界处仍留一条底色缝（调试模式下红蓝之间那条深棕色细线）。
    # 两层都膨胀 1 格后重叠，缝被两边一起填上，深度接近也不会看出接缝。
    mask_near_d = ndimage.binary_dilation(lay == 0, np.ones((3, 3), bool)) & valid
    if solids:
        mask_near_d &= ~solid_mask          # 别和封闭体重叠（会 z-fighting）
    mask_far_d = ndimage.binary_dilation(lay == 2, np.ones((3, 3), bool)) & valid

    for lid in (0, 1, 2):
        if lid == 0:
            mask = mask_near_d
        elif lid == 2:
            mask = mask_far_d
        else:
            mask = (lay == lid)
        if not mask.any():
            continue
        m4 = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
        if lid == 2:
            # 远景层是**幕布**，不需要断崖剪面（剪了反而在它与近景的交界处留下一条缝：
            # 交界处的深度比极大 → 两层都被 cliff 剪掉 → 透出 clear color 的深棕线，
            # 用 ?dbg=layers 直接看得到）。这里只要求四角都在掩码内。
            q = valid_q & m4
        else:
            q = ok_quad & m4
        if not q.any():
            continue
        # 该层对应的天空掩码（与 zs 同形状）；没有就传 None
        sky_slice = None
        if sky is not None and sky.shape == zs.shape:
            sky_slice = sky
        L = build_layer(P, uv, mask, q, zs, dir_world, skirt_min, skirt_max,
                        skirt_cap=skirt_cap, no_skirt=sky_slice)
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
            fz = fwd[valid]
            footprint = (float(np.percentile(fz, 2)), float(np.percentile(fz, 92)),
                         math.tan(cam.hfov * 0.5))
            skirt = build_ground_skirt(P, gmask, uv,
                                       radius=ground_radius, cells=ground_cells,
                                       footprint=footprint, rgb=rgb, eye=eye)
            if skirt is not None:
                layers.append(skirt)
                # 地物：在裙边上撒石块（同一套地面色板取色）
                try:
                    g_uv = uv[gmask]
                    if g_uv.shape[0] > 8:
                        pick = np.linspace(0, g_uv.shape[0] - 1,
                                           min(16, g_uv.shape[0])).astype(int)
                        rocks = build_scatter_rocks(g_uv[pick], float(np.median(P[gmask][:, 1])) - 0.15,
                                                    ground_radius, ground_radius * 0.06,
                                                    count=scatter_rocks)
                        if rocks is not None:
                            layers.append(rocks)
                            stats["scatter"] = {"vertices": int(rocks["pos"].shape[0]),
                                                "triangles": int(rocks["index"].size // 3),
                                                "skirt_quads": 0, "pixels": 0,
                                                "depth_p10": 0.0, "depth_p90": 0.0}
                except Exception:
                    pass
                stats["ground"] = {
                    "vertices": int(skirt["pos"].shape[0]),
                    "triangles": int(skirt["index"].size // 3),
                    "skirt_quads": 0, "pixels": int(ground.sum()),
                    "depth_p10": round(float(ground_radius * 0.3), 1),
                    "depth_p90": round(float(ground_radius), 1),
                }

    # 俯视顶面：飞行时从上方看到的是"同一片地貌的正俯视"，而不是单层壳的背面。
    # 放在最后追加，前端按视点高度决定画不画。
    if top_skin:
        # 只用**近景、非天空**的像素建顶面。
        # 不加这个限制，天空像素（y 高达几百米）会把顶面抬到 300m 以上：
        # 阈值随之变成"飞到 313m 才显示"，等于这功能永远不生效（实测命中）。
        near_mask = valid & (fwd < max(60.0, far_split * 1.5))
        if sky is not None and sky.shape == near_mask.shape:
            near_mask &= ~sky
        # 有语义地面掩码时，顶面**只用地面的像素**：否则"每格取最高点"会把崖壁顶
        # 连成一片屋顶，低视角看就是一块悬空的板（实测截图）。
        gm = near_mask
        if sem_ground is not None:
            g2 = sem_ground
            if g2.shape != seg.shape:
                _iy2 = (np.arange(seg.shape[0]) * g2.shape[0] // seg.shape[0]).astype(np.int32)
                _ix2 = (np.arange(seg.shape[1]) * g2.shape[1] // seg.shape[1]).astype(np.int32)
                g2 = g2[_iy2][:, _ix2]
            if g2.any() and int((g2 & near_mask).sum()) > 500:
                gm = near_mask & g2
        # 足迹：用有效像素的前方距离分位 + 水平半视场的 tan
        _fv = fwd[valid]
        _z_near = float(np.percentile(_fv, 2)) if _fv.size else 2.0
        _z_far = float(np.percentile(_fv, 92)) if _fv.size else 60.0
        _tan_h = float(np.max(np.abs(rays[..., 0]) / np.maximum(np.abs(rays[..., 2]), 1e-6)))
        # **两份地面**：
        #   top（完整）—— 飞行/俯视时用，照片足迹也要铺，否则空中看到一个空洞；
        #   topHole（挖洞）—— 地面视角用，足迹处留给近景壳（保证"站在原地像原片"）。
        # 只有一份的话，两个需求必然互相打架（实测：空中一个巨大三角空洞）。
        skin = build_top_skin(P, uv, gm, cell=top_skin_cell * 1.6, rgb=rgb,
                              extent=float(ground_radius) * 1.15, max_cells=300)
        skin_h = build_top_skin(P, uv, gm, cell=top_skin_cell * 1.6, rgb=rgb,
                                extent=float(ground_radius) * 1.15, max_cells=300,
                                hole_poly=(_z_near, _z_far, _tan_h))
        if skin_h is not None:
            skin_h["name"] = "topHole"
        if skin_h is not None:
            layers.append(skin_h)
            stats["topHole"] = {"vertices": int(skin_h["pos"].shape[0]),
                                "triangles": int(skin_h["index"].size // 3)}
        if skin is not None:
            layers.append(skin)
            stats["top"] = {"vertices": int(skin["pos"].shape[0]),
                            "triangles": int(skin["index"].size // 3),
                            "skirt_quads": 0, "pixels": 0,
                            "depth_p10": float(np.percentile(fwd[valid], 10)),
                            "depth_p90": float(np.percentile(fwd[valid], 90))}

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
