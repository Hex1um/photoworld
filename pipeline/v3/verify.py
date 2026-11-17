"""独立验证：把 GLB 里真实存在的几何**真的光栅化一遍**，再与原图逐像素比。

为什么不能"用管线自己的深度图反投影一下就算通过"
------------------------------------------------
那样验证的是同一份数据的自洽性，不是"打包出去的世界能不能还原照片"。
v2 的 32 机位指标就是栽在这上面（ISSUES §25：世界塌成平地时 8 项指标几乎不动）。
所以这里走最笨但最诚实的路：三角形 → 屏幕 → z-buffer → 双线性采样贴图。

指标：
  * `psnr`      原点视角与原图的峰值信噪比（门控 >= 26dB，理想 >= 32dB）
  * `ssim`      结构相似度（门控 >= 0.85）
  * `coverage`  有几何覆盖的像素比例（天空由几何壳覆盖，所以应接近 1）
  * `reproj_px` 顶点重投影误差（中位/95 分位，像素）—— 相机内参/坐标系是否一致的硬证据
"""
from __future__ import annotations

import math

import numpy as np


def project_cam(P: np.ndarray, cam, out_w: int, out_h: int,
                yaw: float = 0.0, pitch: float | None = None,
                eye_y: float | None = None):
    """世界点 → 输出画布像素坐标 + 深度（沿 -Z）。

    yaw/pitch/eye_y 可覆盖，用于"同机位之外"的目检（侧后视、俯视）。
    旋转约定必须与 `js/mat.js::perspectiveFromCamera` 一致：
    forward = (sin yaw·cos pitch, sin pitch, -cos yaw·cos pitch)。
    """
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    ey = cam.eye if eye_y is None else eye_y
    pt = cam.pitch if pitch is None else pitch
    d = P - np.array([0.0, ey, 0.0])
    if abs(pt) > 1e-12:                      # 去俯仰
        c, s = math.cos(pt), math.sin(pt)
        x, y, z = d[:, 0], d[:, 1], d[:, 2]
        d = np.stack([x, y * c + z * s, -y * s + z * c], axis=1)
    if abs(yaw) > 1e-12:                     # 去偏航
        c, s = math.cos(yaw), math.sin(yaw)
        x, y, z = d[:, 0], d[:, 1], d[:, 2]
        d = np.stack([x * c + z * s, y, -x * s + z * c], axis=1)
    z = -d[:, 2]
    safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    xn = d[:, 0] / (safe * cam.tan_h)
    yn = d[:, 1] / (safe * cam.tan_v)
    px = (xn * 0.5 + 0.5) * out_w
    py = (0.5 - yn * 0.5) * out_h
    return np.stack([px, py], axis=1), z


def rasterize(layers: list, tex: np.ndarray, cam, out_w: int, out_h: int,
              near: float = 0.05, yaw: float = 0.0, pitch: float | None = None,
              eye_y: float | None = None) -> dict:
    """z-buffer 光栅化（双面）。返回 {"rgb", "depth", "coverage"}。"""
    th, tw = tex.shape[:2]
    zbuf = np.full((out_h, out_w), np.inf, dtype=np.float32)
    uvbuf = np.zeros((out_h, out_w, 2), dtype=np.float32)
    # 顶点色缓冲：渲染器的 mode 4 是「贴图 × 顶点色」，光栅化必须一起算。
    # 少了这一步，顶点色层（远景地形/地物/穹顶）会被画成纯白，
    # 侧视图全白、原点 PSNR 假性掉到 29dB（实测踩过）。
    colbuf = np.ones((out_h, out_w, 4), dtype=np.float32)
    hit = np.zeros((out_h, out_w), dtype=bool)

    for lay in layers:
        pos = np.asarray(lay["pos"], dtype=np.float64)
        uv = np.asarray(lay["uv"], dtype=np.float64)
        idx = np.asarray(lay["index"], dtype=np.int64).reshape(-1, 3)
        if idx.size == 0:
            continue
        scr, z = project_cam(pos, cam, out_w, out_h, yaw=yaw, pitch=pitch, eye_y=eye_y)
        col = lay.get("color")
        col = None if col is None else np.asarray(col, dtype=np.float64).reshape(-1, 4)
        # 逐三角形（Python 循环，只为验证，不追求实时）
        for t in idx:
            z0, z1, z2 = z[t[0]], z[t[1]], z[t[2]]
            if z0 <= near or z1 <= near or z2 <= near:
                continue
            x0, y0 = scr[t[0]]
            x1, y1 = scr[t[1]]
            x2, y2 = scr[t[2]]
            minx = max(int(math.floor(min(x0, x1, x2))), 0)
            maxx = min(int(math.ceil(max(x0, x1, x2))), out_w - 1)
            miny = max(int(math.floor(min(y0, y1, y2))), 0)
            maxy = min(int(math.ceil(max(y0, y1, y2))), out_h - 1)
            if minx > maxx or miny > maxy:
                continue
            xs = np.arange(minx, maxx + 1) + 0.5
            ys = np.arange(miny, maxy + 1) + 0.5
            X, Y = np.meshgrid(xs, ys)
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-12:
                continue
            w0 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / den
            w1 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / den
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not inside.any():
                continue
            iz = w0 / z0 + w1 / z1 + w2 / z2
            zz = np.where(np.abs(iz) > 1e-12, 1.0 / np.maximum(iz, 1e-12), np.inf)
            sub = zbuf[miny:maxy + 1, minx:maxx + 1]
            upd = inside & (zz < sub)
            if not upd.any():
                continue
            sub[upd] = zz[upd]
            u = (w0 * uv[t[0], 0] / z0 + w1 * uv[t[1], 0] / z1 + w2 * uv[t[2], 0] / z2) / iz
            v = (w0 * uv[t[0], 1] / z0 + w1 * uv[t[1], 1] / z1 + w2 * uv[t[2], 1] / z2) / iz
            usub = uvbuf[miny:maxy + 1, minx:maxx + 1]
            usub[upd, 0] = u[upd]
            usub[upd, 1] = v[upd]
            if col is not None:
                cbuf = colbuf[miny:maxy + 1, minx:maxx + 1]
                for c in range(4):
                    cc = (w0 * col[t[0], c] / z0 + w1 * col[t[1], c] / z1
                          + w2 * col[t[2], c] / z2) / iz
                    cbuf[upd, c] = cc[upd]
            hsub = hit[miny:maxy + 1, minx:maxx + 1]
            hsub |= upd

    # 双线性采样
    sx = np.clip(uvbuf[..., 0] * tw - 0.5, 0, tw - 1.001)
    sy = np.clip(uvbuf[..., 1] * th - 0.5, 0, th - 1.001)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    fx = (sx - x0)[..., None]
    fy = (sy - y0)[..., None]
    c00 = tex[y0, x0]
    c10 = tex[y0, np.minimum(x0 + 1, tw - 1)]
    c01 = tex[np.minimum(y0 + 1, th - 1), x0]
    c11 = tex[np.minimum(y0 + 1, th - 1), np.minimum(x0 + 1, tw - 1)]
    rgb = (c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy)
           + c01 * (1 - fx) * fy + c11 * fx * fy)
    rgb = rgb * colbuf[..., :3]
    rgb[~hit] = 0.0
    return {"rgb": rgb.astype(np.float32), "depth": zbuf, "coverage": hit}


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(a.shape[:2], dtype=bool)
    d = (a[mask].astype(np.float64) - b[mask].astype(np.float64))
    mse = float(np.mean(d * d))
    return 99.0 if mse <= 1e-12 else float(10 * math.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """灰度 SSIM（11×11 高斯窗，skimage 口径的简化实现）。"""
    from scipy import ndimage
    ga = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gb = b @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sig = 1.5
    mu_a = ndimage.gaussian_filter(ga, sig)
    mu_b = ndimage.gaussian_filter(gb, sig)
    saa = ndimage.gaussian_filter(ga * ga, sig) - mu_a ** 2
    sbb = ndimage.gaussian_filter(gb * gb, sig) - mu_b ** 2
    sab = ndimage.gaussian_filter(ga * gb, sig) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return float(np.mean(s))


def reprojection(layers: list, cam, vscale: float = 1.0) -> dict:
    """每个顶点投影回原图后，与它"应该"落到的像素差多少。

    这是坐标系/内参一致性的直接证据：如果这里不是亚像素级，
    说明射线、俯仰、纹理 UV 三者有一处对不上。
    """
    errs = []
    for lay in layers:
        pos = np.asarray(lay["pos"], dtype=np.float64)
        uv = np.asarray(lay["uv"], dtype=np.float64)
        if lay.get("name") in ("ground", "far", "prop", "fill"):
            continue
        scr, z = project_cam(pos, cam, cam.w, cam.h)
        ok = (z > 0.1) & (uv[:, 1] < 0.999)      # 白条上的顶点不参与
        if not ok.any():
            continue
        want = np.stack([uv[ok, 0] * cam.w,
                         uv[ok, 1] / max(vscale, 1e-6) * cam.h], axis=1)
        errs.append(np.abs(scr[ok] - want).max(axis=1))
    if not errs:
        return {"median_px": -1.0, "p95_px": -1.0}
    e = np.concatenate(errs)
    return {"median_px": float(np.median(e)), "p95_px": float(np.percentile(e, 95))}


def report(render: dict, photo: np.ndarray, cam) -> dict:
    m = render["coverage"]
    out = {
        "psnr": psnr(render["rgb"], photo, m) if m.any() else 0.0,
        "ssim": ssim(render["rgb"], photo),
        "coverage": float(m.mean()),
        "mean_render": float(render["rgb"][m].mean()) if m.any() else 0.0,
        "mean_photo": float(photo.mean()),
    }
    return out
