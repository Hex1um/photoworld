"""统一着色：一套太阳/天光/接缝，把三层（视锥内 / 远景地形 / 地物）拉到同一个光照里。

为什么必须"集中"做
------------------
v2 的图层各自带着自己的光照假设：照片壳是"已经打光过的像素"（模式 4 直出），
程序化地面是顶点色 + 着色器里再算一遍光照，裙边又是另一种。三层交界处必然出现
"照片一条带 + 地面一块板"的割裂，32 机位质检里表现为一条亮线。

v3 的口径：
* **视锥内不动**（贴图就是原图，一个像素都不许改）——这是硬要求；
* 视锥外**预烘焙**：法线 + 太阳 + 天光，写进顶点色（渲染器 mode 4 只做 贴图×顶点色，
  不再二次打光，所以烘焙必须自带光照）；
* 光照**归一化到"水平地面 = 1.0"**，这样全景里地面的颜色原样保留，
  坡面才出现明暗 —— 否则整片地形会整体偏亮/偏暗，接缝处更明显；
* 接缝：按方位把远景地形在边界处的颜色拉向照片底缘的地面色，带宽 ~25m。
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from . import common


def sun_from_pano(pano: np.ndarray) -> dict:
    """太阳方位 = 全景里最亮的那块天空；天光 = 天空均值。"""
    h, w = pano.shape[:2]
    sky = pano[: max(8, h // 2)]
    luma = sky @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    k = max(3, int(0.02 * luma.size))
    idx = np.argpartition(luma.reshape(-1), -k)[-k:]
    ys, xs = np.unravel_index(idx, luma.shape)
    lat = (0.5 - (ys.mean() + 0.5) / h) * math.pi
    lon = ((xs.mean() + 0.5) / w - 0.5) * 2 * math.pi
    # 最亮处常在太阳附近，压低仰角避免"顶光"（风景照的实际光比顶光低）
    lat = float(np.clip(lat, math.radians(8), math.radians(55)))
    d = np.array([math.sin(lon) * math.cos(lat), math.sin(lat),
                  -math.cos(lon) * math.cos(lat)], dtype=np.float32)
    d /= max(float(np.linalg.norm(d)), 1e-6)

    sky_mean = sky.reshape(-1, 3).mean(axis=0)
    sky_lum = float(sky_mean @ np.array([0.299, 0.587, 0.114], np.float32))
    # 天光比太阳冷一点、弱一点；用亮度归一，避免整体过曝
    amb = np.clip(sky_mean / max(sky_lum, 1e-3) * 0.55, 0.12, 1.2).astype(np.float32)
    sun_col = np.clip(np.array([1.0, 0.96, 0.88], np.float32) * 0.55, 0.1, 1.0)
    return {"dir": d, "amb": amb, "sun": sun_col,
            "elev_deg": round(math.degrees(lat), 1),
            "az_deg": round(math.degrees(lon), 1)}


def bake_lighting(layer: dict, light: dict, strength: float = 0.85) -> dict:
    """把光照烘进顶点色（线性空间算，sRGB 出）。水平面保持原色。"""
    C = layer.get("color")
    if C is None:
        return layer
    n = layer["nrm"]
    l = np.asarray(light["dir"], dtype=np.float32)
    amb = np.asarray(light["amb"], dtype=np.float32)
    sun = np.asarray(light["sun"], dtype=np.float32)

    ndl = np.clip(n @ l, 0.0, 1.0)
    fill = 0.5 + 0.5 * n[:, 1]
    raw = amb[None, :] * (fill[:, None] * 0.9) + sun[None, :] * (ndl[:, None] * 1.05)
    # 归一化：法线朝上的平面，亮度 = 1（这样全景里的地面颜色原样保留）
    up = amb * 0.9 + sun * max(float(l[1]), 0.0) * 1.05
    k = 1.0 / np.maximum(up.mean(), 1e-3)
    shade = np.clip(raw * k, 0.0, 2.5)

    albedo = common.srgb_to_linear(np.clip(C[:, :3], 0, 1))
    lit = albedo * (1.0 + (shade - 1.0) * strength)
    out = C.copy()
    out[:, :3] = common.linear_to_srgb(np.clip(lit, 0, 1))
    layer = dict(layer)
    layer["color"] = out.astype(np.float32)
    return layer


def seam_blend(far: dict, near_edge: np.ndarray, edge_color: np.ndarray,
               band: float = 25.0) -> dict:
    """按方位把远景地形在视锥边界处拉向照片的地面色，消掉一圈色差。

    far 的顶点是极坐标网格：方位 = atan2(x, -z)，半径 = hypot(x, z)。
    near_edge[i] 是该方位上照片地面的半径；band 内做平滑过渡。
    """
    P = far["pos"]
    C = far["color"]
    if C is None or near_edge is None:
        return far
    x, z = P[:, 0], P[:, 2]
    r = np.hypot(x, z)
    az = np.arctan2(x, -z)
    bins = near_edge.shape[0]
    i = np.clip(((az + math.pi) / (2 * math.pi) * bins).astype(np.int64), 0, bins - 1)
    e = near_edge[i]
    ok = np.isfinite(e)
    t = np.clip(1.0 - (r - e) / band, 0.0, 1.0) * ok
    t = (t * t * (3 - 2 * t))[:, None]
    ec = edge_color[i]
    C = C.copy()
    C[:, :3] = C[:, :3] * (1 - t) + ec * t
    far = dict(far)
    far["color"] = C.astype(np.float32)
    return far


def edge_from_near(layers: list, bins: int = 720, ground_y: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """从视锥内的 near 层读出"每个方位上照片覆盖到多远"。

    半径取该方位上**所有** near 顶点的最大 r（不只是地面顶点）：
    如果只统计地面，山体方向（y 很大）会被漏掉，远景地形就会从山脚开始铺，
    结果从原点看过去，地形盖在山上 —— 实测 PSNR 从 41.8dB 掉到 21dB。
    视锥外的地形必须从"照片已经画到的地方"之外开始。
    """
    near = next((l for l in layers if l.get("name") == "near"), None)
    edge = np.zeros(bins, dtype=np.float32)
    if near is None:
        edge[:] = 60.0
        return edge, edge
    P = near["pos"]
    r = np.hypot(P[:, 0], P[:, 2])
    az = np.arctan2(P[:, 0], -P[:, 2])
    ok = (r > 2) & (r < 900)
    if ground_y is not None:
        ok &= np.abs(P[:, 1]) < ground_y
    if not ok.any():
        edge[:] = 60.0
        return edge, edge
    i = np.clip(((az[ok] + math.pi) / (2 * math.pi) * bins).astype(np.int64), 0, bins - 1)
    np.maximum.at(edge, i, r[ok].astype(np.float32))
    has = np.zeros(bins, dtype=bool)
    has[i] = True
    # 空方位（照片 FOV 之外的方位没有 near 顶点）：环形插值，取最近的已知值
    if has.sum() >= 2:
        idx = np.arange(bins)
        xi = np.concatenate([idx[has], idx[has] + bins])
        yi = np.concatenate([edge[has], edge[has]])
        edge = np.interp(idx, xi, yi).astype(np.float32)
    else:
        edge[:] = 60.0
    edge = np.clip(edge, 12.0, 500.0)
    return edge, edge


def ground_edge_color(rgb: np.ndarray, bins: int = 720) -> np.ndarray:
    """照片底缘按方位的地面色（给接缝用）。

    near 层走贴图没有顶点色，所以接缝的颜色基准从**照片本身**取：
    取底部若干行，按列方位重采样到 bins。
    """
    h, w = rgb.shape[:2]
    rows = rgb[int(h * 0.90):int(h * 0.995)]
    col = rows.reshape(-1, 3).mean(axis=0)
    strip = rows.mean(axis=0)                        # (w,3) 每列的地面色
    x = np.linspace(0, w - 1, bins)
    out = np.stack([np.interp(x, np.arange(w), strip[:, c]) for c in range(3)], axis=1)
    out = 0.65 * out + 0.35 * col[None, :]           # 略微拉向整体均值，避免逐列噪点
    return np.clip(out, 0, 1).astype(np.float32)
