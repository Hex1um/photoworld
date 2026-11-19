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
    p = pts - np.array([cx, 0.0, cz], np.float32)

    xmin, xmax = float(p[:, 0].min()), float(p[:, 0].max())
    zmin, zmax = float(p[:, 2].min()), float(p[:, 2].max())
    span = max(xmax - xmin, zmax - zmin, 1e-3)
    xmin, xmax = cx - span / 2, cx + span / 2
    zmin, zmax = cz - span / 2, cz + span / 2
    gx = np.linspace(xmin - cx, xmax - cx, cells)
    gz = np.linspace(zmin - cz, zmax - cz, cells)
    gxx, gzz = np.meshgrid(gx, gz, indexing="ij")

    ix = np.clip(((p[:, 0] - (xmin - cx)) / span * (cells - 1)).astype(np.int64), 0, cells - 1)
    iz = np.clip(((p[:, 2] - (zmin - cz)) / span * (cells - 1)).astype(np.int64), 0, cells - 1)
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
    gy, gx_ = np.gradient(grid_y)
    N = np.stack([-gx_, np.ones_like(grid_y) * (span / cells), -gy], axis=-1).reshape(-1, 3)
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
                 (xmin - cx, xmax - cx, zmin - cz, zmax - cz), float(has.mean()))



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
    # 关键：查看器默认相机站在原点、眼高 1.6、朝 -z 看。重建出来的是"一小块场景"，
    # 如果居中原点，人就站在模型**内部**（背面被剔除 → 一片空白，实测两次翻车）。
    # 所以把这块场景整体挪到相机前方，地面压到脚下：
    out[:, 2] -= target_span * 0.55
    out[:, 1] -= 1.6
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
