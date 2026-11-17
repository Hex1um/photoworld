"""视锥外 ②：全景 → 地形高度场 + 地物。

流程（每一步都有明确的"为什么"）
----------------------------------
1. 从全景切 **8 个 90° 透视视角**。为什么不直接对等距柱状图跑深度：
   深度网络是在透视图上训的，等距柱状图在两极和水平方向都被拉伸，
   同一条地平线在图上不是一条线 —— 标定会失准。切成透视视角后，
   每个视角都是"普通照片"，可以直接复用已验证的 `depth.calibrate`。
2. 每个视角独立反投影成点云（世界系与视锥内完全同源：同一个 eye、同一套旋转）。
3. 点云 → **极坐标高度场**（方位 720 × 半径 160，对数半径 3~600m）。
   极坐标而不是方格：视锥外的信息来自全景，天然是"按方位给"的，
   而且这样地形在视锥边界处与照片的方位一一对应，接缝好处理。
4. 高度场 → 网格；颜色从全景按方位/仰角采样（所以地貌颜色来自扩散模型，
   不是程序化调色）。
5. **地物**：按方位统计全景的"植被感"（绿/暗），在对应方位撒针叶树与石块，
   颜色同样取自全景。地物不做实例分割 —— 视锥外没有真值，实例化只增加成本。
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from . import common
from . import depth as v3depth

AZ_BINS = 720
R_BINS = 160
# R_MAX 试过 900（想让地形接到天空穹顶的下沿），结果更糟：600m 以外的
# 高度场是插值外推出来的，一铺就大面积陷下去 —— lowVoid 16% → 65%。
# 保持 600，穹顶下沿改由 sky_dome 的 lat 下探来兜（见那里）。
R_MIN, R_MAX = 3.0, 600.0


def _rot_y(d: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return np.stack([x * c + z * s, y, -x * s + z * c], axis=-1)


def dirs_to_pano_px(dirs: np.ndarray, pw: int, ph: int):
    d = np.asarray(dirs, dtype=np.float64)
    lon = np.arctan2(d[..., 0], -d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    x = (lon / (2 * math.pi) + 0.5) * pw - 0.5
    y = (0.5 - lat / math.pi) * ph - 0.5
    return x, y


def sample_pano(pano: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    ph, pw = pano.shape[:2]
    x, y = dirs_to_pano_px(dirs, pw, ph)
    xs = np.arange(pw + 1, dtype=np.float64) - 0.5
    ys = np.arange(ph, dtype=np.float64) - 0.5
    out = np.empty(dirs.shape[:-1] + (3,), dtype=np.float32)
    for c in range(3):
        # 水平方向环绕采样（map_coordinates 的 wrap 在 x 上不跨边界，手动补一列）
        ext = np.concatenate([pano[..., c], pano[:, :1, c]], axis=1)
        out[..., c] = ndimage.map_coordinates(ext, [np.clip(y, 0, ph - 1),
                                                    np.mod(x, pw)], order=1, mode="nearest")
    return np.clip(out, 0, 1)


def view_camera(pw: int, ph: int, yaw: float, fov_deg: float = 90.0,
                size: int = 448) -> common.Camera3:
    return common.Camera3(size, size, math.radians(fov_deg), pitch=0.0)


def render_views(pano: np.ndarray, n: int = 8, fov_deg: float = 90.0,
                 size: int = 448) -> list[dict]:
    """全景 → n 个透视视角（世界方向 + 图像）。"""
    views = []
    for k in range(n):
        yaw = 2 * math.pi * k / n
        cam = view_camera(pano.shape[1], pano.shape[0], yaw, fov_deg, size)
        dirs = _rot_y(cam.rays(), yaw)
        img = sample_pano(pano, dirs)
        views.append({"yaw": yaw, "cam": cam, "dirs": dirs, "img": img})
    return views


def point_cloud(views: list[dict], device: str = "cuda",
                long_side: int = 448) -> dict:
    """每个视角跑深度 + 标定 + 反投影，合并成一个世界点云。"""
    pts, cols, srcs = [], [], []
    metas = []
    for v in views:
        disp = v3depth.disparity(v["img"], device=device, long_side=long_side)
        res = v3depth.calibrate(disp, v["cam"])
        z = res["z"]
        eye = v["cam"].eye_pos[None, None, :]
        P = (eye + v["dirs"] * z[..., None]).reshape(-1, 3)
        C = v["img"].reshape(-1, 3)
        keep = np.isfinite(P).all(axis=1) & (z.reshape(-1) > 0.5)
        pts.append(P[keep])
        cols.append(C[keep])
        srcs.append(np.full(keep.sum(), v["yaw"], dtype=np.float32))
        metas.append({"yaw": round(math.degrees(v["yaw"]), 1),
                      "pitch_deg": round(math.degrees(res["cam"].pitch), 2),
                      "cal": res["cal"].get("mode_used")})
    return {"pos": np.concatenate(pts), "color": np.concatenate(cols),
            "yaw": np.concatenate(srcs), "views": metas}


def polar_heightfield(cloud: dict, pano: np.ndarray, eye: float,
                      near_edge: np.ndarray | None = None) -> dict:
    """点云 → 极坐标高度场（方位 × 对数半径），返回 y 与颜色。"""
    P = cloud["pos"]
    eye_height_ref = float(eye)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    r = np.hypot(x, z)
    az = np.arctan2(x, -z)

    # 只保留"可能是地面"的点。
    # 判据必须是**仰角**，不能只看 y 上限：远景的树/山在 300m 处 y 也能到 30m，
    # 用 y<45 过滤会把它们当地面，整片高度场被抬到 +35m（实测：地形整体浮在
    # 头顶，侧视下半屏全空）。地面点在相机附近仰角必为负、远处接近 0。
    elev = np.arctan2(y - eye_height_ref, r)
    ok = ((r > R_MIN) & (r < R_MAX) & (y > -6.0) & (y < 14.0)
          & (elev < 0.10) & (elev > -0.45))
    x, y, z, r, az = x[ok], y[ok], z[ok], r[ok], az[ok]

    az_i = np.clip(((az + math.pi) / (2 * math.pi) * AZ_BINS).astype(np.int64), 0, AZ_BINS - 1)
    lr = np.log(np.clip(r, R_MIN, R_MAX) / R_MIN) / math.log(R_MAX / R_MIN)
    r_i = np.clip((lr * (R_BINS - 1)).astype(np.int64), 0, R_BINS - 1)
    flat = az_i * R_BINS + r_i

    cnt = np.bincount(flat, minlength=AZ_BINS * R_BINS).astype(np.float32)
    sy = np.bincount(flat, weights=y, minlength=AZ_BINS * R_BINS).astype(np.float32)
    H = np.where(cnt > 0, sy / np.maximum(cnt, 1e-6), np.nan).reshape(AZ_BINS, R_BINS)

    # 半径方向插值补空（同一方位上，地形是连续的）
    for i in range(AZ_BINS):
        row = H[i]
        good = np.isfinite(row)
        if good.sum() < 2:
            H[i] = 0.0
            continue
        idx = np.arange(R_BINS)
        H[i] = np.interp(idx, idx[good], row[good])
    H = ndimage.median_filter(H, size=(3, 3), mode="wrap")
    H = ndimage.gaussian_filter1d(H, 1.0, axis=1, mode="nearest")

    r_grid = R_MIN * np.exp(np.log(R_MAX / R_MIN) * (np.arange(R_BINS) / (R_BINS - 1)))
    # 高度基准锚定到"离相机 30~90m 的环"上。为什么需要：
    # 全景里的视角是扩散生成的，没有 EXIF/相机高度，8 个视角各自标定后会带一个
    # 共同的低频偏差 —— 实测整片地形沉到 -8m（一个碗）。视锥外的地面必须与
    # 视锥内的 y=0 同高，否则走两步就是断崖。
    ref_band = (r_grid > 30) & (r_grid < 90)
    ref = float(np.nanmedian(H[:, ref_band])) if ref_band.any() else 0.0
    H = np.clip(H - ref, -2.5, 60.0).astype(np.float32)

    # 近处（<30m）强制压到 y=0，30~110m 过渡回实测高度场。
    # 为什么必须强制：全景是扩散生成的，8 个视角的深度在"脚下的地面"上彼此不一致，
    # 实测近场中位高度是 +10.6m —— 地形整体浮在头顶，平视时下半屏是空的。
    # 而视锥内的地面本来就被 ground_snap 压到 y=0，这里同样压平才能接上。
    w_near = np.clip((r_grid - 30.0) / 80.0, 0.0, 1.0)[None, :]
    w_near = w_near * w_near * (3 - 2 * w_near)
    H = (H * w_near).astype(np.float32)

    # 注意：这里**不能**再把 near_edge 以内的高度改成边界值。
    # 那是"填地板"时代的做法，它会把上面刚压平的近场又抬回实测高度
    # （实测近场中位 +10.6m 又回来了，平视时下半屏依旧空）。

    # 颜色：按方位/仰角从全景采样
    A, Rg = np.meshgrid(((np.arange(AZ_BINS) + 0.5) / AZ_BINS - 0.5) * 2 * math.pi,
                        r_grid, indexing="ij")
    dirs = np.stack([np.sin(A) * np.ones_like(Rg), np.zeros_like(Rg), -np.cos(A)], axis=-1)
    elev = np.arctan2(H - eye, np.maximum(Rg, 1e-3))
    dirs[..., 1] = np.sin(elev) / np.maximum(np.hypot(1.0, np.sin(elev)), 1e-6)
    dirs[..., 0] *= np.cos(elev)
    dirs[..., 2] *= np.cos(elev)
    col = sample_pano(pano, dirs)
    return {"H": H, "r": r_grid, "color": col}


def heightfield_mesh(field: dict, eye: float, near_edge: np.ndarray | None,
                     uv_white: tuple[float, float],
                     fov_half: float = math.pi / 3) -> dict:
    """极坐标高度场 → 三角网格（顶点色走白条 UV）。"""
    H, r_grid, col = field["H"], field["r"], field["color"]
    A = ((np.arange(AZ_BINS) + 0.5) / AZ_BINS - 0.5) * 2 * math.pi
    AA, RR = np.meshgrid(A, r_grid, indexing="ij")
    X = RR * np.sin(AA)
    Z = -RR * np.cos(AA)
    P = np.stack([X, H - 0.2, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    # 视锥内（照片覆盖的范围）**不铺地形**。
    # 试过在这里填一张 y=-0.2 的地板来消掉侧视露底，结果原点视角崩了：
    # 深度在近处常被高估（湖面给 100m），那个像素的真实表面落到 y=-3，
    # 地板（-0.4）反而更近，把湖面整片盖住 —— PSNR 50.5 → 10.1dB。
    # 所以这里只能挖洞；"侧视露底"靠 near 层自己的裙边（沿射线向后挤出厚度）去补。
    valid = np.ones((AZ_BINS, R_BINS), dtype=bool)
    if near_edge is not None:
        for i in range(AZ_BINS):
            if abs(A[i]) > fov_half * 1.06:
                continue
            e = near_edge[i]
            if np.isfinite(e):
                # 留 3m 重叠带避免边界处出现一条缝
                valid[i, r_grid < e + 3.0] = False
    H = np.where(np.isfinite(H), H, np.nan).astype(np.float32)

    a = np.arange(AZ_BINS)[:, None]
    b = (a + 1) % AZ_BINS
    i0 = np.arange(R_BINS - 1)[None, :]
    A0 = (a * R_BINS + i0)
    A1 = (b * R_BINS + i0)
    A2 = (b * R_BINS + i0 + 1)
    A3 = (a * R_BINS + i0 + 1)
    ok = valid[:, :-1] & valid[:, 1:] & valid[(np.arange(AZ_BINS) + 1) % AZ_BINS][:, :-1]
    ok = ok & np.isfinite(H[:, :-1]) & np.isfinite(H[:, 1:])
    ok &= (np.abs(H[:, :-1] - H[:, 1:]) < 40.0)          # 断崖不连面

    tri = np.concatenate([
        np.stack([A0[ok], A1[ok], A2[ok]], axis=1),
        np.stack([A0[ok], A2[ok], A3[ok]], axis=1),
    ], axis=0).astype(np.uint32)

    nrm = _normals(P, tri, eye)
    C = np.concatenate([col.reshape(-1, 3), np.ones((AZ_BINS * R_BINS, 1), np.float32)],
                       axis=1)
    UV = np.tile(np.array(uv_white, dtype=np.float32), (P.shape[0], 1))
    return {"name": "far", "pos": P, "nrm": nrm, "uv": UV, "color": C,
            "index": np.ascontiguousarray(tri)}


def _normals(P: np.ndarray, tri: np.ndarray, eye: float) -> np.ndarray:
    n = np.zeros_like(P)
    if tri.size:
        v0, v1, v2 = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        for k in range(3):
            np.add.at(n, tri[:, k], fn)
    n[:, 1] += 1e-3
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(ln, 1e-9)
    up = n[:, 1] < 0
    n[up] *= -1.0
    return n.astype(np.float32)


# --------------------------------------------------------------------------- 地物

def props(pano: np.ndarray, field: dict, near_edge: np.ndarray | None, eye: float,
          uv_white: tuple[float, float], seed: int = 0,
          density: float = 1.0, fov_half: float = math.pi / 3) -> dict | None:
    """按方位撒树/石：颜色取自全景，位置落在高度场上。

    不做实例分割：视锥外没有真值，实例化只增加成本；这里要的是
    "转过身能看到树林，而不是一片空坡"。
    """
    rng = np.random.default_rng(seed)
    H, r_grid, col = field["H"], field["r"], field["color"]
    A = ((np.arange(AZ_BINS) + 0.5) / AZ_BINS - 0.5) * 2 * math.pi

    # 每个方位的"植被感"：绿色占优 + 偏暗 → 有树
    green = (col[..., 1] > col[..., 0] * 1.02) & (col.mean(axis=-1) < 0.55)
    veg = green.mean(axis=1)
    dark = (col.mean(axis=-1) < 0.30).mean(axis=1)
    score = np.clip(0.6 * veg + 0.6 * dark, 0, 1)

    verts, cols, tris = [], [], []
    base = 0
    for i in range(AZ_BINS):
        # 照片视锥覆盖的方位不撒地物：那块已经有照片自己的内容（山/树/建筑），
        # 再撒三维树只会在原图视角里挡住照片 —— 实测 PSNR 40.6 → 33.2dB。
        if abs(float(A[i])) <= fov_half * 1.06:
            continue
        p = float(score[i]) * 0.42 * density
        if p <= 0.02:
            continue
        n = int(rng.poisson(p))
        if n <= 0:
            continue
        inside_fov = abs(float(A[i])) <= fov_half * 1.06
        e = float(near_edge[i]) if (inside_fov and near_edge is not None
                                    and np.isfinite(near_edge[i])) else 8.0
        lo = max(e + 6.0, 30.0)
        hi = max(lo * 1.2, 160.0)
        for _ in range(n):
            rr = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            j = min(max(int(np.searchsorted(r_grid, rr)), 1), R_BINS - 1)
            t = (rr - r_grid[j - 1]) / max(r_grid[j] - r_grid[j - 1], 1e-6)
            y = float(H[i, j - 1] * (1 - t) + H[i, j] * t)
            if not np.isfinite(y) or y > 30:
                continue
            az = float(A[i]) + float(rng.normal(0, 0.002))
            px, pz = rr * math.sin(az), -rr * math.cos(az)
            # 树高随距离略增（远处要能看见），近处不超过 12m
            hgt = float(np.clip(rng.uniform(3.0, 7.5) * (1.0 + rr / 400.0), 2.5, 11.0))
            rad = hgt * float(rng.uniform(0.16, 0.26))
            c = np.clip(col[i, j] * np.array([0.85, 0.95, 0.85], np.float32), 0, 1)
            v, f, cc = _cone(px, y, pz, rad, hgt, c, rng)
            verts.append(v); cols.append(cc); tris.append(f + base)
            base += v.shape[0]
            if rng.random() < 0.12:                     # 少量石块
                v2, f2, c2 = _rock(px + rng.normal(0, 1.4), y, pz + rng.normal(0, 1.4),
                                   rng.uniform(0.4, 1.1), c * 0.8, rng)
                verts.append(v2); cols.append(c2); tris.append(f2 + base)
                base += v2.shape[0]

    if not verts:
        return None
    P = np.concatenate(verts)
    F = np.concatenate(tris).astype(np.uint32)
    return {"name": "prop", "pos": P.astype(np.float32),
            "nrm": _normals(P, F, eye),
            "uv": np.tile(np.array(uv_white, dtype=np.float32), (P.shape[0], 1)),
            "color": np.concatenate(cols).astype(np.float32),
            "index": np.ascontiguousarray(F)}


def _cone(x: float, y: float, z: float, rad: float, hgt: float, col: np.ndarray,
          rng, sides: int = 6):
    ang = np.linspace(0, 2 * math.pi, sides, endpoint=False) + rng.uniform(0, 1)
    ring = np.stack([x + rad * np.cos(ang), np.full(sides, y), z + rad * np.sin(ang)], axis=1)
    apex = np.array([[x, y + hgt, z]])
    P = np.concatenate([ring, apex], axis=0).astype(np.float32)
    C = np.tile(np.concatenate([col, [1.0]]).astype(np.float32), (P.shape[0], 1))
    F = []
    for k in range(sides):
        F.append([k, (k + 1) % sides, sides])
    return P, np.array(F, dtype=np.uint32), C


def _rock(x: float, y: float, z: float, rad: float, col: np.ndarray, rng):
    n = 5
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    base = np.stack([x + rad * np.cos(ang), np.full(n, y), z + rad * np.sin(ang)], axis=1)
    top = np.array([[x + rng.normal(0, 0.15), y + rad * 0.8, z + rng.normal(0, 0.15)]])
    P = np.concatenate([base, top], axis=0).astype(np.float32)
    C = np.tile(np.concatenate([col, [1.0]]).astype(np.float32), (P.shape[0], 1))
    F = np.array([[k, (k + 1) % n, n] for k in range(n)], dtype=np.uint32)
    return P, F, C


# --------------------------------------------------------------------------- 天空

def sky_dome(pano: np.ndarray, radius: float = 920.0, az_bins: int = 192,
             lat_bins: int = 40, uv_white: tuple[float, float] = (0.5, 0.5),
             cam=None) -> dict:
    """360° 天空穹顶（顶点色）。

    只靠照片里的那片天空壳，转身 180° 就是黑的。穹顶用全景上采样出整片天空，
    半径取 920m（比照片天空壳的 900m 略远），两者不会 z-fighting，
    照片壳在前、穹顶兜住其余方向。
    """
    A = ((np.arange(az_bins) + 0.5) / az_bins - 0.5) * 2 * math.pi
    # 纬度必须**严格小于 90°**：超过极点会让顶点翻到对侧（cos(lat)<0），
    # 那一圈三角形会横跨整个天空，把照片天空整片盖掉 —— 实测 PSNR 46→32dB。
    # 纬度从 **-8°** 起（不是 0°）：地形只铺到 600m，600m 处的地平线比真地平线低
    # 0.16°，而穹顶从 0° 起 → 中间夹出一条空带，转身就是一道 void（实测 alpine
    # 的 B 圈 innerVoid 14%）。压到 -8° 让穹顶与地形外缘重叠，空带消失；
    # 穹顶在 920m、地形 ≤600m，深度测试保证地形永远在前，不会盖住地面。
    LAT = ((np.arange(lat_bins) + 0.5) / lat_bins * (math.pi * 0.5 + math.radians(8))
           - math.radians(8))
    AA, LL = np.meshgrid(A, LAT, indexing="ij")
    dirs = np.stack([np.sin(AA) * np.cos(LL), np.sin(LL), -np.cos(AA) * np.cos(LL)], axis=-1)
    P = (dirs * radius + np.array([0.0, 0.0, 0.0])).reshape(-1, 3).astype(np.float32)
    col = sample_pano(pano, dirs)
    C = np.concatenate([col.reshape(-1, 3), np.ones((P.shape[0], 1), np.float32)], axis=1)

    a = np.arange(az_bins)[:, None]
    b = (a + 1) % az_bins
    i0 = np.arange(lat_bins - 1)[None, :]
    A0 = a * lat_bins + i0
    A1 = b * lat_bins + i0
    A2 = b * lat_bins + i0 + 1
    A3 = a * lat_bins + i0 + 1
    tri = np.concatenate([
        np.stack([A0.reshape(-1), A1.reshape(-1), A2.reshape(-1)], axis=1),
        np.stack([A0.reshape(-1), A2.reshape(-1), A3.reshape(-1)], axis=1),
    ], axis=0)

    # 照片视锥覆盖的方向**不铺穹顶**：那块由照片天空壳（900m、UV=原图）负责。
    # 穹顶只在"照片之外"补天。不挖这个洞，穹顶会把照片边角的像素用扩散天空
    # 盖住 —— 原图区 PSNR 从 50.5dB 掉到 34dB（虽然肉眼看只是"更糊一点"）。
    if cam is not None:
        from . import pano as _pano
        d3 = dirs.reshape(-1, 3)
        u, v, inside = _pano.project_dirs(d3, cam)
        ins = inside.reshape(az_bins, lat_bins)
        t2 = ins.reshape(-1)
        tri = tri[~(t2[tri[:, 0]] & t2[tri[:, 1]] & t2[tri[:, 2]])]

    tri = tri.astype(np.uint32)
    # 法线朝内（相机在球心）
    nrm = (-P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-6)).astype(np.float32)
    return {"name": "backdrop", "pos": P, "nrm": nrm,
            "uv": np.tile(np.array(uv_white, dtype=np.float32), (P.shape[0], 1)),
            "color": C.astype(np.float32), "index": np.ascontiguousarray(tri)}


# --------------------------------------------------------------------------- 与照片对齐

def conform_to_photo(layer: dict, z_near: np.ndarray, sky: np.ndarray,
                     cam, margin: float = 1.06,
                     photo: np.ndarray | None = None) -> dict:
    """把视锥外的几何**压到照片表面之后**，并把落进天空的顶点剔掉。

    为什么必须做这一层（实测）：只按"方位半径"挖洞是不够的 —— 照片的足迹不是
    凸的，同一个方位上不同仰角的地面/山体半径差几十倍。coast 那张有 48% 是天空，
    远景地形在照片的方位范围内就从 r=15m 开始铺，直接盖住了照片下半屏
    （multiview 的 innerVoid 12%、void 18%）。

    做法（两种情形分开，因为"错"的方式不同）：
      * 落在**非天空**像素上：说明该方向照片已经给出了表面。地形要么在它后面，
        要么就是穿模 —— 沿该像素的射线把顶点推到 z_near×1.06（投影不变，
        所以原点视角看不到它，侧视时它正好贴着照片壳的下沿，不会留洞）。
      * 落在**天空**像素上：那个方向本来就不该有地形（是天空），直接剔掉整片顶点。
    """
    from . import verify as vfy

    P = np.asarray(layer["pos"], dtype=np.float64)
    idx = np.asarray(layer["index"], dtype=np.int64).reshape(-1, 3)
    scr, dist = vfy.project_cam(P, cam, cam.w, cam.h)
    ui = np.clip(scr[:, 0].astype(np.int64), 0, cam.w - 1)
    vi = np.clip(scr[:, 1].astype(np.int64), 0, cam.h - 1)
    inside = ((scr[:, 0] >= 0) & (scr[:, 0] < cam.w)
              & (scr[:, 1] >= 0) & (scr[:, 1] < cam.h) & (dist > 0.35))

    drop = inside & sky[vi, ui]
    ref = z_near[vi, ui].astype(np.float64)
    push = inside & ~sky[vi, ui] & (dist < ref * margin) & (ref > 0.5)

    keep = ~drop
    if keep.all() and not push.any():
        return layer

    # 顶点重排（丢掉的顶点不参与索引）
    remap = np.full(P.shape[0], -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    tri_ok = keep[idx[:, 0]] & keep[idx[:, 1]] & keep[idx[:, 2]]
    new_idx = remap[idx[tri_ok]]

    out = {"name": layer["name"],
           "pos": P[keep].astype(np.float32),
           "uv": np.asarray(layer["uv"])[keep].astype(np.float32),
           "nrm": np.asarray(layer["nrm"])[keep].astype(np.float32),
           "index": np.ascontiguousarray(new_idx.astype(np.uint32))}
    if layer.get("color") is not None:
        out["color"] = np.asarray(layer["color"])[keep].astype(np.float32)

    # 被推后的顶点：位置沿各自射线后移，颜色改成**它背后那个照片像素**的颜色。
    # 为什么改色：这些顶点侧移时会露出来，用全景色就是一块与照片不接的平板
    # （实测 desert 走两步就看到一片灰紫平面）。取照片颜色后，它读起来就是
    # "照片地面往外继续"，接缝也自然。
    pk = push[keep]
    if pk.any():
        eye = np.array([0.0, cam.eye, 0.0])
        d = out["pos"][pk] - eye
        ln = np.linalg.norm(d, axis=1, keepdims=True)
        target = (ref[keep][pk] * margin)[:, None]
        out["pos"][pk] = (eye + d / np.maximum(ln, 1e-9) * target).astype(np.float32)
        if photo is not None and out.get("color") is not None:
            ui2 = ui[keep][pk]
            vi2 = vi[keep][pk]
            c = photo[vi2, ui2]
            out["color"][pk, :3] = c
            out["color"][pk, 3] = 1.0

    out["culled"] = int((~keep).sum())
    out["pushed"] = int(pk.sum())
    return out
