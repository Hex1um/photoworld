"""TSDF-lite：用单目深度 + SfM 位姿做体素融合，再用 marching cubes 出**带垂直面**的网格。

为什么需要它：高度场（fuse.py）把世界压成"一张起伏的地面"，墙、树干、台阶全变成一条边；
室内或有竖直结构体的场景会缺面。体素融合能表达任意朝向的表面。

"lite" 的含义（都是刻意的取舍）：
  * 不做真正的截断符号距离场积分（那要对每个体素沿射线采样，Python 里太慢），
    而是把每帧**反投影出的点**投进体素格，记录"命中次数 / 命中深度均值 / 颜色均值"，
    再用**占据率**当隐式场（0.5 等值面）跑 marching cubes；
  * 深度用 `Depth-Anything` 的单目相对视差，按 SfM 稀疏点做**每帧线性对齐**
    （`z = 1/(a·disp + b)`），所以尺度与 SfM 一致；
  * 网格只保留面积 ≥ 阈值的连通块（去掉飞点造成的小碎块）。

输出与 fuse.py 同构（pos/nrm/col/idx + extent/fill），可直接走同一个 GLB 写出器。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    pos: np.ndarray
    nrm: np.ndarray
    col: np.ndarray
    idx: np.ndarray
    extent: tuple
    fill: float


def _align_depth(disp: np.ndarray, pts_world: np.ndarray, K: np.ndarray, R: np.ndarray,
                 t: np.ndarray) -> tuple[float, float, int]:
    """用落在该帧里的 SfM 点，拟合 z = 1/(a·disp + b)。返回 (a,b,用到的点数)。"""
    h, w = disp.shape
    Xc = (R @ pts_world.T + t).T
    z = Xc[:, 2]
    ok = z > 1e-6
    if ok.sum() < 20:
        return 1.0, 0.0, int(ok.sum())
    uv = (K @ Xc[ok].T).T
    u = (uv[:, 0] / uv[:, 2]).astype(np.int64)
    v = (uv[:, 1] / uv[:, 2]).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, zz = u[inside], v[inside], z[ok][inside]
    if len(zz) < 20:
        return 1.0, 0.0, len(zz)
    d = disp[v, u].astype(np.float64)
    # 1/z 与 disp 线性相关：invz = a·disp + b
    A = np.stack([d, np.ones_like(d)], axis=1)
    sol, *_ = np.linalg.lstsq(A, 1.0 / np.maximum(zz, 1e-6), rcond=None)
    return float(sol[0]), float(sol[1]), int(len(zz))



def dense_points(frames, res, step: int = 3, max_points: int = 200000):
    """把每帧的单目深度按 SfM 位姿反投影成**稠密点云**（给 TSDF 和泼溅初始化共用）。

    为什么需要它：SfM 只给几百到几千个稀疏点，泼溅/体素都覆盖不了画面；
    单目深度能给出每帧 5~7 万个点，代价是深度有偏差（已按 SfM 点逐帧线性对齐）。
    """
    from v3 import depth as v3depth
    pts = res.pts
    out_p, out_c = [], []
    if pts is None or len(pts) < 100:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    span = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    for fi, f in enumerate(frames):
        p = res.poses[fi] if fi < len(res.poses) else None
        if p is None:
            continue
        R, t = p
        disp = v3depth.disparity(f.rgb)
        a, b, npts = _align_depth(disp, pts, res.cams[fi].K, R, t)
        if npts < 30 or not np.isfinite(a) or a <= 1e-4:
            continue
        h, w = disp.shape
        vv, uu = np.mgrid[0:h:step, 0:w:step]
        d = disp[vv, uu].astype(np.float64)
        z = 1.0 / np.maximum(a * d + b, 1e-6)
        ok = np.isfinite(z) & (z > 0.02) & (z < span * 20)
        if ok.sum() < 200:
            continue
        K = res.cams[fi].K
        ray = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1],
                        np.ones_like(uu, np.float64)], axis=-1)
        Xc = (ray * z[..., None])[ok]
        Xw = (R.T @ (Xc - t.ravel()).T).T
        rgb = f.rgb[vv[ok], uu[ok]].astype(np.float32) / 255.0
        out_p.append(Xw.astype(np.float32)); out_c.append(rgb)
    if not out_p:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    P = np.vstack(out_p); C = np.vstack(out_c)
    if len(P) > max_points:
        sel = np.random.default_rng(0).choice(len(P), max_points, replace=False)
        P, C = P[sel], C[sel]
    return P, C


def fuse_tsdf(frames, res, disp_fn=None, voxel: float | None = None,
              cells: int = 200, min_ratio: float = 0.02):
    """frames/res 来自 v4.ingest / v4.sfm。返回 Mesh 或 None（点太少时）。"""
    from v3 import depth as v3depth      # 复用 Depth-Anything 封装

    pts = res.pts
    if pts is None or len(pts) < 200:
        return None
    # 用 1/99 分位定范围：COLMAP 偶尔有离群点把 bbox 拉大 10 倍，
    # 体素就会被撑成 0.46（对象才 4 单位），出来的网格只有几百个顶点（实测）。
    lo_p = np.percentile(pts, 1, axis=0)
    hi_p = np.percentile(pts, 99, axis=0)
    span = float(np.max(hi_p - lo_p))
    vox = float(voxel) if voxel else max(1e-4, span / max(40, cells))

    lo = lo_p - vox * 4
    hi = hi_p + vox * 4
    dims = np.maximum(((hi - lo) / vox).astype(np.int64) + 1, 8)
    # 控制总量：8M 体素以内（200³ = 8M）
    while np.prod(dims) > 8_000_000:
        vox *= 1.25
        dims = np.maximum(((hi - lo) / vox).astype(np.int64) + 1, 8)
    print("      TSDF 体素 %.3f，网格 %s（%.1fM 体素）"
          % (vox, "×".join(map(str, dims)), np.prod(dims) / 1e6))

    hit = np.zeros(tuple(dims), np.float32)
    colsum = np.zeros(tuple(dims) + (3,), np.float32)
    for fi, f in enumerate(frames):
        p = res.poses[fi] if fi < len(res.poses) else None
        if p is None:
            continue
        R, t = p
        disp = disp_fn(f.rgb) if disp_fn else v3depth.disparity(f.rgb)
        a, b, npts = _align_depth(disp, pts, res.cams[fi].K, R, t)
        if npts < 30:
            print("        帧 %d：可用的 SfM 点太少（%d），跳过" % (fi, npts))
            continue
        if not np.isfinite(a) or a <= 1e-4:
            # a ≤ 0 说明"视差越大反而越远"，或者场景深度几乎没变化（拟合退化成噪声）。
            # 这种帧的深度点会把体素糊成一团，宁可不融合。
            print("        帧 %d：深度对齐退化（a=%.5f），跳过" % (fi, a))
            continue
        h, w = disp.shape
        step = 3
        vv, uu = np.mgrid[0:h:step, 0:w:step]
        d = disp[vv, uu].astype(np.float64)
        z = 1.0 / np.maximum(a * d + b, 1e-6)
        okz = np.isfinite(z) & (z > 0.02) & (z < span * 20)
        if okz.sum() < 100:
            continue
        # 反投影到相机系 → 世界
        ray = np.stack([(uu - res.cams[fi].K[0, 2]) / res.cams[fi].K[0, 0],
                        (vv - res.cams[fi].K[1, 2]) / res.cams[fi].K[1, 1],
                        np.ones_like(uu, np.float64)], axis=-1)
        Xc = ray * z[..., None]
        Xc = Xc[okz]
        Xw = (R.T @ (Xc - t.ravel()).T).T
        rgb = f.rgb[vv[okz], uu[okz]].astype(np.float32) / 255.0
        gi = np.floor((Xw - lo) / vox).astype(np.int64)
        keep = np.all((gi >= 0) & (gi < dims), axis=1)
        gi, rgb = gi[keep], rgb[keep]
        if len(gi) == 0:
            continue
        lin = np.ravel_multi_index(gi.T, dims)
        np.add.at(hit.reshape(-1), lin, 1.0)
        cf = colsum.reshape(-1, 3)
        np.add.at(cf, lin, rgb)
        colsum = cf.reshape(colsum.shape)
        print("        帧 %d：深度对齐 a=%.4f b=%.4f（%d 点）→ %d 个深度点进体素"
              % (fi, a, b, npts, len(gi)))

    occ = hit > 0
    if occ.sum() < 200:
        return None
    # 占据率当隐式场：多帧命中更可信；再轻微平滑，让 marching cubes 得到连续面
    from scipy import ndimage
    field = ndimage.gaussian_filter(occ.astype(np.float32), 0.9)
    if field.max() <= 0:
        return None
    thr = max(0.35, float(np.percentile(field[occ], 20)) * 0.9)
    try:
        from skimage import measure
    except Exception as e:
        raise SystemExit("需要 scikit-image 才能做 marching cubes：pip install scikit-image（%s）" % e)
    verts, faces, normals, _ = measure.marching_cubes(field, level=thr, spacing=(vox, vox, vox))
    verts = verts + lo[None, :]
    if len(verts) < 100:
        return None

    # 顶点颜色：就近取体素颜色均值。
    # v4.1.1：等值面顶点经常落在 hit=0 的空体素里，旧版除数是 1e-6 而分子是 0，
    # 于是那一整片面是纯黑（合成素材上的"大块黑面"）。对空体素改用**最近的有命中
    # 体素**的颜色，而不是默认黑色；scipy 已在上面导入。
    gi = np.clip(np.floor((verts - lo) / vox).astype(np.int64), 0, dims - 1)
    nearest = (gi[:, 0], gi[:, 1], gi[:, 2])
    empty = hit == 0
    if empty.any() and hit.any():
        # return_distances=False 时只返回 indices 一个值，不能解包成两个
        inds = ndimage.distance_transform_edt(
            empty, return_distances=False, return_indices=True)
        nearest = tuple(inds[k][gi[:, 0], gi[:, 1], gi[:, 2]] for k in range(3))
    hn = hit[nearest]
    vcol = colsum[nearest] / np.maximum(hn[:, None], 1.0)
    vcol = np.where(np.isfinite(vcol), vcol, 0.45)

    idx = faces.astype(np.uint32)
    col4 = np.concatenate([np.clip(vcol, 0, 1), np.ones((len(vcol), 1), np.float32)], axis=1)
    ext = (float(verts[:, 0].min()), float(verts[:, 0].max()),
           float(verts[:, 2].min()), float(verts[:, 2].max()))
    fill = float(occ.sum()) / float(np.prod(dims))
    return Mesh(verts.astype(np.float32), normals.astype(np.float32), col4.astype(np.float32),
                idx, ext, fill)



def _fit_viewer(pos: np.ndarray, target_span: float = 40.0):
    """把任意尺度的重建结果缩放/平移到查看器的习惯尺度。

    SfM 的尺度是任意单位（可能 0.5，也可能 200），而查看器的默认相机按"世界跨度
    几十个单位、眼高 1.6"来摆。不归一化的话，人眼会站在网格里面或离得极远，
    表现为"一片空白"（实测就是这么翻车的）。这里统一缩到水平跨度 ~40 单位、
    底部贴 y=0、水平居中。
    """
    mn, mx = pos.min(0), pos.max(0)
    span = max(float(mx[0] - mn[0]), float(mx[2] - mn[2]), 1e-6)
    s = target_span / span
    out = (pos - np.array([(mn[0] + mx[0]) / 2, mn[1], (mn[2] + mx[2]) / 2])) * s
    # 关键：查看器默认相机站在原点、眼高 1.6、朝 -z 看。
    # 重建结果是"一块场景"，必须整块放到相机前方合适的距离；否则要么站在模型内部
    # （背面被剔除 → 空白），要么贴到脸上（只看到几块巨大的面）。
    # 横向居中、竖向中心抬到眼高、纵向退到 1.4×跨度 —— 60° 视场下正好占满画面。
    out[:, 2] -= target_span * 1.4
    out[:, 1] += 1.6 - (out[:, 1].min() + out[:, 1].max()) / 2
    return out.astype(np.float32), float(s)


def to_glb(mesh: Mesh, path: str, name: str = "world-v4") -> dict:
    import io

    from PIL import Image

    from v3 import common as v3common
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="JPEG", quality=90)
    pos, _s = _fit_viewer(mesh.pos)
    uv = np.tile(np.array([[0.5, 0.5]], np.float32), (len(pos), 1))
    layer = {"name": "terrain", "pos": pos, "nrm": mesh.nrm, "uv": uv,
             "color": mesh.col, "index": mesh.idx}
    bbox = [float(pos[:, 0].min()), float(pos[:, 1].min()), float(pos[:, 2].min()),
            float(pos[:, 0].max()), float(pos[:, 1].max()), float(pos[:, 2].max())]
    extras = {
        "pipeline": "v4", "method": "tsdf",
        "v4": {"extent": [float(v) for v in mesh.extent], "fill": float(mesh.fill),
               "source": "multi-view SfM + monocular depth + marching cubes"},
        "bbox": bbox,
        "camera": {"hfov": 1.0472, "vfov": 0.6981, "w": 1024, "h": 640,
                   "pitch": 0.0, "eye_height": 1.6},
        "sun": [0.45, 0.78, 0.44], "amb": [0.55, 0.58, 0.62],
        "fog": [0.62, 0.66, 0.72], "white_uv": [0.5, 0.5],
    }
    return v3common.write_world_glb(path, [layer], buf.getvalue(), extras, name=name)
