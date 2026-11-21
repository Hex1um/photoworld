"""Poisson 重建：把点云补成干净、封闭的实心网格。

⚠️ 实测结论（lego，14 万 MVS 点）：**在当前 MVS 法线下不如 TSDF 好用**。
   Poisson 对法线朝向极其敏感，COLMAP `stereo_fusion` 输出的法线只在每个参考视图内
   自洽、整体并**不是**全局一致朝向，于是泊松在正反之间来回，出来是一片"拉丝面条"。
   试过：直接用法线 / 自己估+一致定向 / 统计离群剔除 / depth 9~10，都不行。
   → 物体类素材请用 `--mesh tsdf`；要真上泊松，得先做**全局法线一致化**
   （Open3D 的 `orient_normals_consistent_tangent_plane` 在 360° 环绕上也常失败，
   业界做法是用相机位姿做可见性投票）。本模块保留为可选实验。


为什么需要（对比实测）：
  * 高度场（fuse.py）：只能表达"地面"，墙和树干变成一条边；
  * TSDF-lite（tsdf.py）：用占据率 + marching cubes，出来是**薄壳**，边缘毛、有洞；
  * Poisson：解一次泊松方程得到隐式表面，能把点云补成连续实心面 —— 物体、室内都适用。

代价与注意：
  * 必须有点云**法线**，且方向一致（否则会出现"外翻"或破洞）→ 先 `estimate_normals`
    再 `orient_normals_consistent_tangent_plane`；
  * `depth` 越大越细，显存/时间涨得快（8GB 上 9 是安全值，10 会吃满）；
  * Poisson 会"过度封闭"：给 `densities` 做低分位裁剪，去掉远离点云的面片。
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


def _voxel_down(P, C, N, vox):
    """numpy 体素降采样（同时平均法线与颜色）。返回降采样后的 P,C,N。"""
    key = np.floor(P / max(vox, 1e-6)).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    n = len(cnt)
    Pd = np.zeros((n, 3)); Cd = np.zeros((n, 3))
    Nd = np.zeros((n, 3)) if N is not None else None
    for c in range(3):
        Pd[:, c] = np.bincount(inv, weights=P[:, c], minlength=n) / cnt
        Cd[:, c] = np.bincount(inv, weights=C[:, c], minlength=n) / cnt
        if Nd is not None:
            Nd[:, c] = np.bincount(inv, weights=N[:, c], minlength=n) / cnt
    if Nd is not None:
        Nd = Nd / np.maximum(np.linalg.norm(Nd, axis=1, keepdims=True), 1e-9)
    return Pd, Cd, Nd


def reconstruct(pts: np.ndarray, cols: np.ndarray, depth: int = 9,
                density_pct: float = 5.0, verbose: bool = True,
                normals=None, voxel_div: float = 1200.0) -> "Mesh | None":
    """点云 → 泊松网格。失败返回 None（点太少/网格为空）。"""
    import open3d as o3d

    if pts is None or len(pts) < 3000:
        # 点太少时 Poisson 会造出球状团块，不如不做
        return None
    P = np.asarray(pts, np.float64)
    C = np.asarray(cols, np.float64)
    N = None if normals is None else np.asarray(normals, np.float64)
    span = float(np.max(P.max(0) - P.min(0)))
    Pd, Cd, Nd = _voxel_down(P, C, N, span / voxel_div)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Pd)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(Cd, 0, 1))
    if Nd is not None:
        # 优先用 COLMAP MVS 的**一致朝向**法线：泊松对法线极敏感，
        # 自己估再定向在 360° 环绕上很容易反掉一片（实测渲出来是一团糊）。
        pcd.normals = o3d.utility.Vector3dVector(Nd)
        if verbose:
            print("      使用 MVS 法线（%d 个点）" % len(Pd))
    else:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=span * 0.02, max_nn=30))
        try:
            pcd.orient_normals_consistent_tangent_plane(30)
        except Exception as e:
            if verbose:
                print("      法线定向失败（%s），退回按质心定向" % e)
            pcd.orient_normals_towards_camera_location(P.mean(0))

    # MVS 点云必然有外点（遮挡边、反光、远处误匹配）：先做统计离群剔除，
    # 否则泊松会把它们连成"面条"（实测：不剔除时渲出来一片拉丝）。
    try:
        pcd, _keep = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=1.6)
    except Exception:
        pass
    if verbose:
        print("      Poisson：%d 点（原始 %d，剔离群后），depth=%d" % (len(pcd.points), len(pts), depth))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=True)
    d = np.asarray(densities)
    if density_pct > 0 and len(d):
        mesh.remove_vertices_by_mask(d < np.percentile(d, density_pct))
    if len(mesh.vertices) == 0:
        return None

    # 顶点色：从降采样点云最近邻搬（Poisson 不保留颜色）
    kdt = o3d.geometry.KDTreeFlann(pcd)
    mpos = np.asarray(mesh.vertices)
    colors = np.zeros((len(mpos), 3), np.float64)
    src = np.asarray(pcd.colors)
    for i, p in enumerate(mpos):
        _, idx, _ = kdt.search_knn_vector_3d(p, 1)
        colors[i] = src[idx[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()

    pos = np.asarray(mesh.vertices, np.float32)
    nrm = np.asarray(mesh.vertex_normals, np.float32)
    idx = np.asarray(mesh.triangles, np.uint32)
    col = np.concatenate([colors.astype(np.float32),
                          np.ones((len(pos), 1), np.float32)], axis=1)
    ext = (float(pos[:, 0].min()), float(pos[:, 0].max()),
           float(pos[:, 2].min()), float(pos[:, 2].max()))
    fill = float(len(pos)) / max(1.0, len(pts))
    if verbose:
        print("      Poisson 完成：顶点 %d / 三角 %d" % (len(pos), len(idx)))
    return Mesh(pos, nrm, col, idx, ext, fill)


def to_glb(mesh: Mesh, path: str, name: str = "world-v4") -> dict:
    """写 GLB（与 fuse/tsdf 同构：一层顶点色网格 + 1×1 白贴图）。"""
    import io

    from PIL import Image

    from v3 import common as v3common
    from v4.fuse import _fit_viewer
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="JPEG", quality=90)
    pos, _s = _fit_viewer(mesh.pos)
    uv = np.tile(np.array([[0.5, 0.5]], np.float32), (len(pos), 1))
    layer = {"name": "terrain", "pos": pos, "nrm": mesh.nrm, "uv": uv,
             "color": mesh.col, "index": mesh.idx}
    bbox = [float(pos[:, 0].min()), float(pos[:, 1].min()), float(pos[:, 2].min()),
            float(pos[:, 0].max()), float(pos[:, 1].max()), float(pos[:, 2].max())]
    extras = {
        "pipeline": "v4", "method": "poisson",
        "v4": {"extent": [float(v) for v in mesh.extent], "fill": float(mesh.fill),
               "source": "multi-view MVS + poisson reconstruction"},
        "bbox": bbox,
        "camera": {"hfov": 1.0472, "vfov": 0.6981, "w": 1024, "h": 640,
                   "pitch": 0.0, "eye_height": 1.6},
        "sun": [0.45, 0.78, 0.44], "amb": [0.55, 0.58, 0.62],
        "fog": [0.62, 0.66, 0.72], "white_uv": [0.5, 0.5],
    }
    return v3common.write_world_glb(path, [layer], buf.getvalue(), extras, name=name)
