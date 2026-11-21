"""把稀疏点云 / 深度点云拼成一个"世界"：体素降采样 → 笛卡尔高度场 → 网格（顶点色）。

为什么是高度场而不是直接网格化点云：
  * 网格化点云（泊松/滚球）在这个规模上又慢又不稳定，8GB 机器上不划算；
  * 我们的渲染器（v3 那套）本来就吃"地面 + 顶点色"，高度场是最省事且不会破面的表示；
  * 代价是说不了"垂直面"（墙、树干），所以 QC 里会明确报出**垂直覆盖率**，
    让用户知道这个世界是"俯视能看、平视会空"的那种。

坐标系：以点云的水平中心为原点，y 向上。尺度保持 SfM 的任意单位（qc 会报出
"1 单位 ≈ ?" 的相对尺度，不做绝对米制标定）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class World:
    pos: np.ndarray            # (N,3) float32
    nrm: np.ndarray            # (N,3)
    col: np.ndarray            # (N,4)
    idx: np.ndarray            # (M,3) uint32
    extent: tuple              # (xmin,xmax,zmin,zmax)
    fill: float                # 有数据格子占比


def voxel_downsample(pts: np.ndarray, cols: np.ndarray, voxel: float,
                     max_points: int = 400000):
    if len(pts) == 0:
        return pts, cols
    key = np.floor(pts / max(1e-6, voxel)).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    n = len(cnt)
    out = np.zeros((n, 3), np.float32)
    oc = np.zeros((n, 3), np.float32)
    for c in range(3):
        out[:, c] = np.bincount(inv, weights=pts[:, c], minlength=n) / cnt
        oc[:, c] = np.bincount(inv, weights=cols[:, c], minlength=n) / cnt
    if n > max_points:
        sel = np.random.default_rng(0).choice(n, max_points, replace=False)
        out, oc = out[sel], oc[sel]
    return out, oc


def to_world(pts: np.ndarray, cols: np.ndarray, cells: int = 220,
             smooth: int = 1, base: str = "median") -> World:
    """点云 → 高度场网格。cells 是网格分辨率（cells×cells 个格）。"""
    if len(pts) < 50:
        raise SystemExit("点太少（%d），无法成世界" % len(pts))
    cx, cz = np.median(pts[:, 0]), np.median(pts[:, 2])

    # v4.1.3：水平范围必须用分位数，不能用 min/max。SfM 的三角化离群点很多
    # （合成实测 20964 点里 x 有 -54..19、z 有 1..77，真实场景只有约 5 个单位），
    # 旧版 min/max 把 span 拉到 75.6，200×200 网格填充率仅 3.5%，于是已经 PASS
    # 的素材在建完后被判 FAIL 并删掉 GLB。这里按 2/98 分位裁掉离群点，
    # 再留 15% 边距；注意是**剔除**而不是 clip，否则离群点会被贴到网格边缘
    # 继续污染高度。高度方向也做同样裁剪（鸟/反光之类的尖峰）。
    qx = np.percentile(pts[:, 0], [2, 98])
    qz = np.percentile(pts[:, 2], [2, 98])
    qy = np.percentile(pts[:, 1], [2, 98])
    half = max(float(cx - qx[0]), float(qx[1] - cx),
               float(cz - qz[0]), float(qz[1] - cz), 1e-3) * 1.15
    keep = ((np.abs(pts[:, 0] - cx) <= half) & (np.abs(pts[:, 2] - cz) <= half)
            & (pts[:, 1] >= qy[0]) & (pts[:, 1] <= qy[1]))
    if int(keep.sum()) >= 50:
        pts = pts[keep]
        cols = cols[keep]
    p = pts - np.array([cx, 0.0, cz], np.float32)

    span = 2.0 * half
    gx = np.linspace(-half, half, cells)
    gz = np.linspace(-half, half, cells)
    gxx, gzz = np.meshgrid(gx, gz, indexing="ij")

    ix = np.clip(((p[:, 0] + half) / span * (cells - 1)).astype(np.int64), 0, cells - 1)
    iz = np.clip(((p[:, 2] + half) / span * (cells - 1)).astype(np.int64), 0, cells - 1)
    flat = ix * cells + iz
    ncell = cells * cells
    cnt = np.bincount(flat, minlength=ncell).astype(np.float32)
    has = cnt > 0
    wy = np.zeros(ncell, np.float32)
    wc = np.zeros((ncell, 3), np.float32)
    for c in range(3):
        wc[:, c] = np.bincount(flat, weights=cols[:, c], minlength=ncell)
    if base == "median":
        # 中位更抗离群（高树、鸟、噪声点），代价是要逐格排序
        wy[:] = 0.0
        order = np.argsort(flat, kind="stable")
        fs, ys = flat[order], p[order, 1]
        start = 0
        for k in range(len(fs)):
            if k + 1 == len(fs) or fs[k + 1] != fs[k]:
                wy[fs[k]] = float(np.median(ys[start:k + 1]))
                start = k + 1
    else:
        wy = np.bincount(flat, weights=p[:, 1], minlength=ncell)
    y = np.zeros(ncell, np.float32)
    col = np.ones((ncell, 3), np.float32) * 0.42
    if base == "median":
        # v4.1.1：median 分支算出的已经是"每格高度的中位数"，不能再除以点数。
        # 旧版在这里统一除 cnt，导致 60 点格子的高度被缩成 1/60（合成实测 10 → 0.167），
        # 所有 v4 默认世界的地形高度都不可信。mean 分支才需要除以 cnt。
        y[has] = wy[has]
    else:
        y[has] = wy[has] / np.maximum(cnt[has], 1)
    col[has] = wc[has] / np.maximum(cnt[has], 1)[:, None]

    # 空洞填充：反复用邻居均值补，避免大洞变成"黑洞"
    grid_y = y.reshape(cells, cells).copy()
    grid_c = col.reshape(cells, cells, 3).copy()
    mask = has.reshape(cells, cells)
    for _ in range(24):
        if mask.all():
            break
        ys = np.pad(grid_y, 1, mode="edge")
        # 只补空间两维：默认 np.pad(grid_c, 1) 会把通道维也补成 5 → 广播报错
        cs = np.pad(grid_c, ((1, 1), (1, 1), (0, 0)), mode="edge")
        ms = np.pad(mask.astype(np.float32), 1, mode="edge")
        acc = np.zeros_like(grid_y)
        accc = np.zeros_like(grid_c)
        wsum = np.zeros_like(grid_y)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                m = ms[1 + dy:1 + dy + cells, 1 + dx:1 + dx + cells]
                acc += ys[1 + dy:1 + dy + cells, 1 + dx:1 + dx + cells] * m
                accc += cs[1 + dy:1 + dy + cells, 1 + dx:1 + dx + cells] * m[..., None]
                wsum += m
        fillable = (~mask) & (wsum > 0)
        grid_y[fillable] = acc[fillable] / wsum[fillable]
        grid_c[fillable] = accc[fillable] / np.maximum(wsum[fillable], 1)[:, None]
        mask = mask | fillable
    if smooth > 0:
        for _ in range(smooth):
            k = np.array([0.25, 0.5, 0.25], np.float32)
            grid_y = (np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, grid_y)
                      + np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, grid_y)) * 0.5
    grid_y = grid_y.astype(np.float32)

    # 顶点 / 法线 / 颜色
    P = np.stack([gxx, grid_y, gzz], axis=-1).reshape(-1, 3).astype(np.float32)
    C = grid_c.reshape(-1, 3)
    # meshgrid 用 indexing="ij"：axis0 = x，axis1 = z。np.gradient 返回 (d/dx, d/dz)，
    # 旧版把两个分量命名反了，法线的 x/z 分量随之互换（斜面上的高光方向是错的）。
    gx_, gz_ = np.gradient(grid_y)
    N = np.stack([-gx_, np.ones_like(grid_y) * (span / cells), -gz_], axis=-1).reshape(-1, 3)
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-9)

    idx = []
    for i in range(cells - 1):
        for j in range(cells - 1):
            a = i * cells + j
            b = a + 1
            c = a + cells
            d = c + 1
            idx.append((a, c, b))
            idx.append((b, c, d))
    idx = np.array(idx, np.uint32)
    col4 = np.concatenate([C, np.ones((len(C), 1), np.float32)], axis=1)
    return World(P, N.astype(np.float32), col4.astype(np.float32), idx,
                 (-half, half, -half, half), float(has.mean()))



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


def to_glb(world: World, path: str, name: str = "world-v4") -> dict:
    """写成 GLB：一层 vertex-colored 三角网格 + 1×1 白贴图。"""
    import io

    from PIL import Image

    from v3 import common as v3common
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="JPEG", quality=90)
    # 顶点色层的 UV 指向贴图中心：贴图是纯白，所以颜色完全由 COLOR_0 决定
    pos, _s = _fit_viewer(world.pos)
    uv = np.tile(np.array([[0.5, 0.5]], np.float32), (len(pos), 1))
    layer = {"name": "terrain", "pos": pos, "nrm": world.nrm, "uv": uv,
             "color": world.col, "index": world.idx}
    extras = {
        "pipeline": "v4",
        # json 不接受 numpy 标量：extras 会被 GLB 写出器直接 json.dumps
        "v4": {"extent": [float(v) for v in world.extent], "fill": float(world.fill),
               "source": "multi-view SfM + heightfield"},
        "bbox": [float(pos[:, 0].min()), float(pos[:, 1].min()),
                 float(pos[:, 2].min()), float(pos[:, 0].max()),
                 float(pos[:, 1].max()), float(pos[:, 2].max())],
        "camera": {"hfov": 1.0472, "vfov": 0.6981, "w": 1024, "h": 640,
                   "pitch": 0.0, "eye_height": 1.6},
        "sun": [0.45, 0.78, 0.44], "amb": [0.55, 0.58, 0.62],
        "fog": [0.62, 0.66, 0.72], "white_uv": [0.5, 0.5],
    }
    return v3common.write_world_glb(path, [layer], buf.getvalue(), extras, name=name)
