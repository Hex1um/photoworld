"""视锥内几何：每个像素一个世界坐标点 → 连续曲面。

这一步是整条 v3 管线的核心，回答三件事：

1. **每个像素落在哪**（"视锥内推出每个像素的位置"）
   `P(u,v) = eye + ray(u,v) · z(u,v)`，z 来自度量深度。因为 `ray` 与渲染器
   共用同一套内参，把 P 反投影回原点相机**必然落回它自己的像素**
   （`verify.reprojection` 实测中位 3e-5 px）。

2. **怎么变成"连续的"而不是一张薄片**
   v1/v2 的通病是"单层薄片"：断崖处要么连成幕布（拉伸），要么跳过四边形
   留一个洞。v3 用**双四边形**：
     * 近端四边形：四角沿各自射线缩到该格最近的深度 → 从原点看显示的是
       **前景自己的像素**（与原图一致），侧移时它就是有厚度的正面；
     * 远端四边形：四角放到最远深度 → 侧移时露出的是**背景**，不是虚空。
   两个都只沿射线移动顶点，所以"从原点看等于原图"这条不变量不被破坏。
   实测：v2 式跳过四边形会让原点视角出现一圈 1 格宽的黑边（PSNR 22.7），
   双四边形之后黑洞消失（见 docs/v3.md 的对照表）。

3. **怎么保证"从原点看等于原图"**
   颜色走**照片 UV**（不是顶点色），贴图就是原图；几何修正只沿射线做。
   两条一起，使 PSNR 只受纹理压缩与遮挡顺序影响。

天空单独处理：天空像素没有"表面"，硬建面会在侧移时变成幕布。v3 把天空建成
一层**无视差远景壳**（推远到 ~900m，层名 backdrop），既保住原点像素一致，
又不会在走动时露馅。
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from . import common


# --------------------------------------------------------------------------- 掩码

def horizon_row(cam: common.Camera3) -> float:
    """地平线所在的行（归一化 0..1）。pitch>0（抬头）时地平线下移。"""
    return 0.5 + math.tan(cam.pitch) / (2.0 * cam.tan_v)


def sky_mask(rgb: np.ndarray, z: np.ndarray, cam: common.Camera3,
             grad_max: float = 0.045) -> np.ndarray:
    """天空 = 地平线以上 + 远 + 低梯度，且**与画面上边缘连通**。

    连通性是关键判据：远处的雪山也满足"远 + 低梯度"，但它不连顶边。
    v2 里"岩壁被当成天空"（ISSUES §20）就是缺了这一条。
    """
    h, w = z.shape
    rows = (np.arange(h, dtype=np.float32)[:, None] + 0.5) / h
    above = rows < horizon_row(cam) - 0.005

    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gx = np.zeros_like(luma)
    gy = np.zeros_like(luma)
    gx[:, 1:-1] = np.abs(luma[:, 2:] - luma[:, :-2]) * 0.5
    gy[1:-1] = np.abs(luma[2:] - luma[:-2]) * 0.5
    grad = ndimage.uniform_filter(gx + gy, size=5)

    far_ref = float(np.percentile(z, 92))
    cand = above & (z > 0.55 * far_ref) & (grad < grad_max)
    # 形态学之前先按 edge 复制补边：直接对边界做 open 会把第一行/最后一行整行腐蚀掉，
    # 而"与顶边连通"这条判据正好要靠顶边 → 天空会被判成 0（实测踩过）。
    pad = np.pad(cand, 3, mode="edge")
    pad = ndimage.binary_closing(pad, np.ones((5, 5), bool))
    pad = ndimage.binary_opening(pad, np.ones((3, 3), bool))
    cand = pad[3:-3, 3:-3]

    lab, n = ndimage.label(cand)
    if n == 0:
        return np.zeros_like(cand)
    top = np.unique(lab[:max(2, h // 40)][lab[:max(2, h // 40)] > 0])
    if top.size == 0:                       # 顶边没有候选：退到最大连通块
        sizes = ndimage.sum(cand, lab, range(1, n + 1))
        top = np.array([int(np.argmax(sizes)) + 1])
    sky = np.isin(lab, top)
    return ndimage.binary_fill_holes(sky)


def foreground_mask(z: np.ndarray, sky: np.ndarray, near_max: float = 70.0) -> np.ndarray:
    """前景 = 明显比邻域背景近的像素（局部深度极小值）。

    这是"被遮挡区域"的定义来源：只有前景会挡住后面的东西，也只有在它背后
    才需要补出背景层。
    """
    zz = np.where(sky, 0.0, z).astype(np.float32)
    local_far = ndimage.maximum_filter(zz, size=25)
    fg = (zz > 0) & (zz < 0.80 * np.maximum(local_far, 1e-3)) & (zz < near_max)
    fg = ndimage.binary_opening(fg, np.ones((3, 3), bool))
    fg = ndimage.binary_dilation(fg, np.ones((5, 5), bool)) & (zz > 0)
    return fg


def ground_snap(z: np.ndarray, cam: common.Camera3, dirs: np.ndarray,
                tol: float = 0.40) -> tuple[np.ndarray, np.ndarray]:
    """把"确实落在地平面附近"的像素吸附到 y=0 平面。

    判据是**几何的**（射线与 y=0 的交点距离 vs 深度），不是画面行号 ——
    REVIEW.md D1 的结论：用行号判会把高耸主体整块拍平。
    吸附只沿射线做（z → t_plane），所以不改变该像素在原点的投影。
    """
    dy = dirs[..., 1].astype(np.float32)
    down = dy < -0.02
    t_plane = np.where(down, -cam.eye / np.minimum(dy, -1e-4), 0.0).astype(np.float32)
    ratio = z / np.maximum(t_plane, 1e-3)
    snap = down & (t_plane > 1.2) & (np.abs(np.log(np.maximum(ratio, 1e-3))) < tol)
    return np.where(snap, t_plane, z).astype(np.float32), snap


# --------------------------------------------------------------------------- 背景补全

def _down(a: np.ndarray) -> np.ndarray:
    h, w = a.shape
    if h % 2:
        a = np.pad(a, ((0, 1), (0, 0)), mode="edge")
    if w % 2:
        a = np.pad(a, ((0, 0), (0, 1)), mode="edge")
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean(axis=(1, 3))


def _up(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)[:size[0], :size[1]]


def push_pull_fill(val: np.ndarray, valid: np.ndarray, levels: int = 9,
                   coarse_weight: float = 0.35) -> np.ndarray:
    """把 valid 处的值平滑外推到整张图（经典 push-pull，O(N)）。

    用于"前景背后应该是什么"：从四周的背景往外长，得到一张连续、低频、
    没有硬边的深度/颜色场。相比最近邻填充，它不会把前景颜色糊到背景上。
    """
    num = np.where(valid, val, 0.0).astype(np.float32)
    den = valid.astype(np.float32)
    stack = []
    for _ in range(levels):
        stack.append((num, den))
        if min(num.shape) < 6:
            break
        num, den = _down(num), _down(den)
    if float(den.sum()) <= 1e-6:
        return np.zeros_like(val, dtype=np.float32)
    cur = num / np.maximum(den, 1e-6)
    for (n2, d2) in reversed(stack[:-1]):
        up = _up(cur, n2.shape)
        # 该层没有有效像素时直接吃上一层的结果；有的话按权重混。
        # 注意分母必须只统计"有值"的那部分，否则 d2=0 处会被除成 0
        # （实测症状：整张背景深度被填成 0 → exp(0)=1m → 背景层全部挤到眼前）。
        den_safe = np.maximum(d2 * (1.0 + coarse_weight), 1e-6)
        cur = np.where(d2 > 1e-6, (n2 + coarse_weight * up * d2) / den_safe, up)
    return cur.astype(np.float32)


def nearest_source(valid: np.ndarray) -> np.ndarray:
    """每个像素最近的 valid 像素的扁平索引（给背景层取"应该是什么颜色"）。"""
    idx = ndimage.distance_transform_edt(~valid, return_distances=False,
                                         return_indices=True)
    return (idx[0] * valid.shape[1] + idx[1]).astype(np.int64)


# --------------------------------------------------------------------------- 网格

def _dist(pos: np.ndarray, eye: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pos - eye, axis=-1)


def grid_mesh(pos: np.ndarray, valid: np.ndarray, uv: np.ndarray,
              dirs: np.ndarray, eye: np.ndarray, color: np.ndarray | None = None,
              cliff: float = 1.8, skirt: float = 0.30,
              skirt_mask: np.ndarray | None = None,
              safe_mask: np.ndarray | None = None,
              safe_uv: np.ndarray | None = None,
              floor: np.ndarray | None = None) -> dict:
    """规则像素网格 → 无洞、带厚度的曲面（断崖用双四边形，见模块文档）。"""
    h, w = valid.shape
    eye = np.asarray(eye, dtype=np.float32)
    vid = np.full((h, w), -1, dtype=np.int64)
    vid[valid] = np.arange(int(valid.sum()))

    keep = valid.reshape(-1)
    P = pos.reshape(-1, 3)[keep]
    UV = uv.reshape(-1, 2)[keep]
    C = None if color is None else color.reshape(-1, 4)[keep]
    dist = _dist(pos, eye)

    a = vid[:-1, :-1]
    b = vid[:-1, 1:]
    c = vid[1:, 1:]
    d = vid[1:, :-1]
    ok4 = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)

    dists = np.stack([dist[:-1, :-1], dist[:-1, 1:], dist[1:, 1:], dist[1:, :-1]], axis=-1)
    dmax = dists.max(axis=-1)
    dmin = dists.min(axis=-1)
    ratio = dmax / np.maximum(dmin, 1e-3)

    flat_cells = ok4 & (ratio <= cliff)
    split_cells = ok4 & (ratio > cliff)
    edge_fixed = 0

    # 绕序必须是"从相机看逆时针"（WebGL 默认正面 = CCW）。之前是 CW，
    # 所以只能全局关掉背面剔除 —— 后果是从背后/上方看，照片壳的背面糊满画面
    # （"漂浮的板子 + 竖向拉丝"就是这么来的）。绕序改对之后可以开剔除：
    # 站在照片背后看到的就是**地形**，而不是壳的背面。
    tri = np.concatenate([
        np.stack([a[flat_cells], c[flat_cells], b[flat_cells]], axis=1),
        np.stack([a[flat_cells], d[flat_cells], c[flat_cells]], axis=1),
    ], axis=0)

    # ---- 断崖格：近端/远端各一张四边形（顶点沿各自射线重定位，投影不变） ----
    if split_cells.any():
        ci, cj = np.nonzero(split_cells)
        ys = np.stack([ci, ci, ci + 1, ci + 1], axis=1)
        xs = np.stack([cj, cj + 1, cj + 1, cj], axis=1)
        cdir = dirs[ys, xs]
        cuv = uv[ys, xs].copy()
        # P0-2：断崖格四角若落在天空/无深度像素上，逐角插值会把天空色拖到
        # 山脊轮廓（1px 青边）。只把**被污染的那一个角**的 UV 换成最近的前景
        # 内部像素；不整格替换（整格取最近角会把 PSNR 从 48.2 打到 29.5dB）。
        edge_fixed = 0
        if safe_mask is not None and safe_uv is not None:
            bad = ~safe_mask[ys, xs]
            edge_fixed = int(bad.sum())
            if edge_fixed:
                cuv[bad] = safe_uv[ys[bad], xs[bad]]
        zn = dmin[split_cells][:, None, None]
        zf = dmax[split_cells][:, None, None]
        # 逐角保留各自的 UV。试过"整格取最近角的颜色"来消山脊上的青边，
        # 结果是天空那一侧被前景色盖住，原图区 PSNR 从 48.2 掉到 29.5dB ——
        # 断崖格两侧都要各自的颜色，宁可留一条 1px 的过渡带。
        Pn = (eye[None, None, :] + cdir * zn).reshape(-1, 3)
        Pf = (eye[None, None, :] + cdir * zf).reshape(-1, 3)

        base_n = P.shape[0]
        P = np.concatenate([P, Pn, Pf], axis=0)
        UV = np.concatenate([UV, cuv.reshape(-1, 2), cuv.reshape(-1, 2)], axis=0)
        if C is not None:
            ccol = color[ys, xs].reshape(-1, 4)
            C = np.concatenate([C, ccol, ccol], axis=0)

        m = ci.shape[0]
        idx_n = base_n + np.arange(m * 4).reshape(m, 4)
        idx_f = base_n + m * 4 + np.arange(m * 4).reshape(m, 4)
        tri = np.concatenate([
            tri,
            np.stack([idx_n[:, 0], idx_n[:, 2], idx_n[:, 1]], axis=1),
            np.stack([idx_n[:, 0], idx_n[:, 3], idx_n[:, 2]], axis=1),
            np.stack([idx_f[:, 0], idx_f[:, 2], idx_f[:, 1]], axis=1),
            np.stack([idx_f[:, 0], idx_f[:, 3], idx_f[:, 2]], axis=1),
        ], axis=0)

    # ---- 裙边：只给"面的外缘"（照片外框 / 天空边界 / 断崖边）加厚度 ----
    ext_pix = np.zeros((h, w), dtype=bool)
    pair_h = np.zeros((h, max(w - 1, 1)), dtype=bool)
    pair_v = np.zeros((max(h - 1, 1), w), dtype=bool)
    if skirt > 0:
        both_h = valid[:, :-1] & valid[:, 1:]
        both_v = valid[:-1, :] & valid[1:, :]
        edge = np.zeros((h, w), dtype=bool)
        diff_h = valid[:, :-1] != valid[:, 1:]
        diff_v = valid[:-1, :] != valid[1:, :]
        edge[:, :-1] |= diff_h
        edge[:, 1:] |= diff_h
        edge[:-1, :] |= diff_v
        edge[1:, :] |= diff_v
        edge[0, :] |= valid[0, :]
        edge[-1, :] |= valid[-1, :]
        edge[:, 0] |= valid[:, 0]
        edge[:, -1] |= valid[:, -1]
        if skirt_mask is not None:
            edge |= skirt_mask & valid
        ext_pix = valid & edge
        # 断崖格的边界也要封：否则从侧面看，近端/远端两张面之间是透的
        ext_pix[:-1, :-1] |= split_cells
        ext_pix[:-1, 1:] |= split_cells
        ext_pix[1:, :-1] |= split_cells
        ext_pix[1:, 1:] |= split_cells
        pair_h = both_h & ext_pix[:, :-1] & ext_pix[:, 1:]
        pair_v = both_v & ext_pix[:-1, :] & ext_pix[1:, :]

    n_front = P.shape[0]
    n_ext = int(ext_pix.sum())
    if n_ext:
        emap = np.full((h, w), -1, dtype=np.int64)
        emap[ext_pix] = n_front + np.arange(n_ext)
        pp = pos[ext_pix]
        dd = dirs[ext_pix]
        dist_e = dist[ext_pix]
        if floor is not None:
            # 边界**垂直向下**挤到地板：形成地形块的侧壁。沿视线向后挤只是一条薄鳍，
            # 从背后/上方看依然像纸片；向下封边之后从任何角度看都是同一个实体。
            pe = pp.copy()
            pe[:, 1] = np.minimum(floor[ext_pix], pp[:, 1] - 0.25)
            P = np.concatenate([P, pe], axis=0)
        else:
            P = np.concatenate([P, pp + dd * (dist_e * skirt)[:, None]], axis=0)
        UV = np.concatenate([UV, uv[ext_pix]], axis=0)
        if C is not None:
            C = np.concatenate([C, color[ext_pix]], axis=0)

        def quad(p1, p2, p3, p4):
            return np.concatenate([
                np.stack([p1, p2, p3], axis=1),
                np.stack([p1, p3, p4], axis=1),
            ], axis=0)

        if pair_h.any():
            i, j = np.nonzero(pair_h)
            tri = np.concatenate([tri, quad(vid[i, j], vid[i, j + 1],
                                           emap[i, j + 1], emap[i, j])], axis=0)
        if pair_v.any():
            i, j = np.nonzero(pair_v)
            tri = np.concatenate([tri, quad(vid[i, j], emap[i, j],
                                           emap[i + 1, j], vid[i + 1, j])], axis=0)

    nrm = _grid_normals(P, tri, eye)
    return {"pos": P.astype(np.float32), "nrm": nrm, "uv": UV.astype(np.float32),
            "color": None if C is None else C.astype(np.float32),
            "index": np.ascontiguousarray(tri.astype(np.uint32)),
            "split_cells": int(split_cells.sum()), "edge_fixed": edge_fixed}


def _grid_normals(P: np.ndarray, tri: np.ndarray, eye: np.ndarray) -> np.ndarray:
    """面法线按面积加权累加到顶点，再翻到朝向相机。"""
    n = np.zeros_like(P)
    if tri.size == 0:
        n[:, 1] = 1.0
        return n.astype(np.float32)
    v0, v1, v2 = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for k in range(3):
        np.add.at(n, tri[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(ln, 1e-9)
    view = eye[None, :] - P
    flip = np.sum(n * view, axis=1) < 0
    n[flip] *= -1.0
    bad = ln[:, 0] < 1e-8
    if bad.any():
        n[bad] = common.normalize(view[bad])
    return n.astype(np.float32)


# --------------------------------------------------------------------------- 主入口

def build(z: np.ndarray, rgb: np.ndarray, cam: common.Camera3, cfg: dict,
          bg_rgb: np.ndarray | None = None) -> dict:
    """视锥内 → 连续曲面 + 天空壳。返回 {"layers", "stats", "debug"}。"""
    step = int(cfg.get("step", 3))
    h, w = z.shape
    dirs = cam.rays()
    eye = cam.eye_pos

    sky = sky_mask(rgb, z, cam, grad_max=float(cfg.get("sky_grad", 0.045)))
    z_snap, snapped = ground_snap(z, cam, dirs, tol=float(cfg.get("ground_tol", 0.40)))

    # 背景层：把前景遮住的部分从背景侧补出来（侧移时才看得到，原点视角被前景压住）
    fg = foreground_mask(z_snap, sky, near_max=float(cfg.get("fg_near", 70.0)))
    valid_bg = (~fg) & (~sky) & np.isfinite(z_snap)
    z_bg = np.exp(push_pull_fill(np.log(np.maximum(z_snap, 0.2)), valid_bg)).astype(np.float32)
    # 背景层永远不许比前景更近：它只在"侧移后露出"时才该被看到。
    # 不加这一条，push-pull 在某些构图下会给出比前景更小的值 → 从原点看
    # 背景层压在前景上（实测 near+fill 从 48.5dB 掉到 30.6dB）。
    z_bg = np.maximum(z_bg, z_snap * 1.02).astype(np.float32)
    src = nearest_source(valid_bg)
    # 视锥内的规则 UV 网格：断崖角安全取色（uv_core）与背景外推（uv_src）都要用，
    # 必须在引用之前定义（曾因 P0-2 断崖角修复插队而触发 UnboundLocalError）。
    uu, vv = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
    uv_photo = np.stack([uu, vv], axis=-1).astype(np.float32)
    # P0-2：前景内部（腐蚀 1px）作为断崖角的安全取色源。
    # 只有"完全落在前景内部"的像素才允许把自己的 UV 用在断崖角上；
    # 边缘 1px（可能混入天空色）改用最近内部像素的 UV。
    core = (~sky) & np.isfinite(z_snap)
    if int(cfg.get("edge_erode", 1)) > 0:
        core = ndimage.binary_erosion(core, iterations=int(cfg.get("edge_erode", 1)))
    src_core = nearest_source(core)
    uv_core = uv_photo.reshape(-1, 2)[src_core].reshape(h, w, 2)

    def sub(a: np.ndarray) -> np.ndarray:
        return a[::step, ::step]

    uv_src = uv_photo.reshape(-1, 2)[src].reshape(h, w, 2)

    pos = eye[None, None, :] + dirs * z_snap[..., None]
    pos_bg = eye[None, None, :] + dirs * z_bg[..., None]

    uv_v = float(cfg.get("uv_v_scale", 1.0))
    uv_near = uv_photo.copy()
    uv_near[..., 1] *= uv_v
    uv_bg = uv_src.copy()
    uv_bg[..., 1] *= uv_v

    # 每像素地板高度（按方位取最小面再降 0.45m，限制在 -6m 以上）：
    # near 层边界沿它向下封边，照片壳变成封闭地形块。
    yy = (eye[None, None, 1] + dirs[..., 1] * z_snap).astype(np.float32)
    bins = 720
    azp = np.arctan2(dirs[..., 0], -dirs[..., 2])
    aip = np.clip(((azp + math.pi) / (2 * math.pi) * bins).astype(np.int64), 0, bins - 1)
    okp = (~sky) & np.isfinite(yy)
    fb = np.full(bins, np.inf, dtype=np.float32)
    if okp.any():
        np.minimum.at(fb, aip[okp], yy[okp])
    fb = np.where(np.isfinite(fb), fb, 0.0)
    floor_px = np.clip(fb[aip] - float(cfg.get("floor_drop", 0.45)),
                       float(cfg.get("floor_min", -6.0)), -0.15).astype(np.float32)

    layers = []
    near = grid_mesh(sub(pos), sub(~sky), sub(uv_near), sub(dirs), eye,
                     cliff=float(cfg.get("cliff", 1.8)),
                     skirt=float(cfg.get("skirt", 0.30)),
                     # solidify 默认**关**：垂直向下封边会在轮廓下方拉出一片幕布
                     # （实测原图区 PSNR 47.1 → 26.9dB）。只有"沿相机射线"的挤出
                     # 才是投影不变的，所以厚度靠 skirt 而不是靠垂直封底。
                     floor=sub(floor_px) if cfg.get("solidify", False) else None,
                     safe_mask=sub(core), safe_uv=sub(uv_core))
    near["name"] = "near"
    if near["index"].size:
        layers.append(near)

    # 背景层只铺"被前景遮住"的那一块：非前景区它与 near 共面，
    # 整片铺开会共面 z-fighting（v2 的 ISSUES #4 就是这么来的）。
    # 有扩散背景板时，fill 层改用**顶点色**（不走贴图）：那块颜色必须来自背景板，
    # 而整包 GLB 只有一张贴图，顶点色是唯一能带第二套颜色的通道（UV 指到白条）。
    if bg_rgb is not None:
        uv_bg = np.empty((h, w, 2), dtype=np.float32)
        uv_bg[..., 0] = float(cfg.get("white_u", 0.5))
        uv_bg[..., 1] = float(cfg.get("white_v", 0.9959))
        col_bg = np.concatenate([bg_rgb, np.ones((h, w, 1), np.float32)], axis=-1)
    else:
        col_bg = None
    bg = grid_mesh(sub(pos_bg), sub(fg & ~sky), sub(uv_bg), sub(dirs), eye,
                   color=None if col_bg is None else sub(col_bg),
                   cliff=float(cfg.get("cliff_bg", 6.0)),
                   skirt=float(cfg.get("skirt_bg", 0.15)))
    bg["name"] = "fill"
    if bg["index"].size:
        layers.append(bg)

    # 天空壳：推远到固定半径，忽略视差
    sky_r = float(cfg.get("sky_radius", 900.0))
    pos_sky = eye[None, None, :] + dirs * sky_r
    sky_layer = grid_mesh(sub(pos_sky), sub(sky), sub(uv_near), sub(dirs), eye,
                          cliff=1e9, skirt=0.0)
    # 名字必须与穹顶（backdrop）区分：照片天空壳只在**出生点**是对的，
    # 相机一移动/一飞起来就是一张悬在空中的照片卡片（实测仰视时像漂浮的画框）。
    # 单独成层之后前端可以按"离出生点的距离 / 是否飞行"把它淡掉。
    sky_layer["name"] = "skycap"
    if sky_layer["index"].size:
        layers.append(sky_layer)

    stats = {
        "sky_frac": float(sky.mean()),
        "fg_frac": float(fg.mean()),
        "ground_snap_frac": float(snapped.mean()),
        "z_near_p2": float(np.percentile(z_snap[~sky], 2)) if (~sky).any() else 0.0,
        "z_far_p98": float(np.percentile(z_snap[~sky], 98)) if (~sky).any() else 0.0,
        "verts": int(sum(l["pos"].shape[0] for l in layers)),
        "tris": int(sum(l["index"].size // 3 for l in layers)),
        "step": step,
        "edge_fixed": int(near.get("edge_fixed", 0)),
    }
    return {"layers": layers, "stats": stats,
            "debug": {"sky": sky, "fg": fg, "z_snap": z_snap, "z_bg": z_bg,
                      "snapped": snapped}}
