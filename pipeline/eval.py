"""质量验收：把重建结果"泼点"回渲染到原相机，和输入照片比对。

纯 numpy 向量化，不需要 GPU / 渲染器。

重要：单独用 `splat_render` 的 PSNR/SSIM **无法**约束尺度与视场 ——
生成与泼点用同一个相机时，全局尺度在 pos/fwd 的比值里抵消，FOV 误差
在 tan_h/tan_v 里自洽抵消，实测坐标整体 ×0.5、FOV 错 25% 仍然是
99.0dB / SSIM 1.0。它能发现的是 UV / 索引 / 贴图错位。

因此本模块额外提供两道独立参照：
  * `ground_scale_check`：用相机眼高这个绝对量卡住地面平面（尺度门）；
  * `cam_verify`：用独立来源的相机内参交叉泼点（视场门，需调用方提供）。
没有独立信息时不要声称"尺度错误一定掉分"。
"""
from __future__ import annotations

import math

import numpy as np


def splat_render(pos: np.ndarray, uv: np.ndarray, tex: np.ndarray,
                 cam, out_hw: tuple[int, int] | None = None) -> np.ndarray:
    """正交投影到原相机（z 已知，直接用相机模型反投影），返回渲染图。"""
    h, w = out_hw if out_hw else (tex.shape[0], tex.shape[1])
    fwd = np.maximum(-pos[:, 2], 1e-6)          # glTF 约定：视线朝 -Z
    xs = (pos[:, 0] / fwd) / cam.tan_h * 0.5 + 0.5
    ys = 0.5 - (pos[:, 1] / fwd) / cam.tan_v * 0.5
    px = np.clip((xs * w).astype(np.int32), 0, w - 1)
    py = np.clip((ys * h).astype(np.int32), 0, h - 1)

    order = np.argsort(-fwd)               # 远的先画
    px, py, z = px[order], py[order], fwd[order]

    th, tw = tex.shape[:2]
    tu = np.clip((uv[order, 0] * tw).astype(np.int32), 0, tw - 1)
    tv = np.clip((uv[order, 1] * th).astype(np.int32), 0, th - 1)
    col = tex[tv, tu]

    img = np.zeros((h, w, 3), dtype=np.float32)
    zbuf = np.full((h, w), np.inf, dtype=np.float32)
    # 从远到近覆盖
    for i in range(px.size):
        yy, xx = py[i], px[i]
        if z[i] < zbuf[yy, xx]:
            zbuf[yy, xx] = z[i]
            img[yy, xx] = col[i]
    return img


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """简化版 SSIM（灰度、11x11 均值滤波）。"""
    if a.ndim == 3:
        a = a.mean(axis=2)
    if b.ndim == 3:
        b = b.mean(axis=2)
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    def blur(x):
        k = 11
        pad = k // 2
        xp = np.pad(x, pad, mode="edge")
        cs = np.cumsum(np.cumsum(xp, axis=0), axis=1)
        cs = np.pad(cs, ((1, 0), (1, 0)))
        s = cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]
        return s / (k * k)

    mu1, mu2 = blur(a), blur(b)
    s1 = blur(a * a) - mu1 * mu1
    s2 = blur(b * b) - mu2 * mu2
    s12 = blur(a * b) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2)
    return float(np.mean(num / np.maximum(den, 1e-8)))


def ground_scale_check(pos, uv, eye: float = 1.7, v_min: float = 0.88,
                       v_max: float = 0.995, tol: float = 0.25) -> dict:
    """用眼高作绝对尺度参照：地面带顶点应满足 y ≈ -eye（相机系）。

    这是唯一不随全局缩放抵消的量：所有坐标 ×0.5 时 PSNR 纹丝不动，
    但地面 y 会从 -1.7 变成 -0.85，相对误差 0.5 直接超门限。
    """
    v = np.asarray(uv)[:, 1]
    y = np.asarray(pos)[:, 1]
    m = (v >= v_min) & (v <= v_max) & np.isfinite(y)
    n = int(m.sum())
    if n < 50:
        return {"n": n, "ok": False, "rel_error": None, "tol": tol,
                "reason": "地面带顶点不足 50 个，无法验收尺度"}
    med = float(np.median(y[m]))
    rel = abs(med + eye) / max(eye, 1e-6)
    return {
        "n": n, "ground_y_median": round(med, 3), "expected": round(-eye, 3),
        "rel_error": round(rel, 4), "tol": tol, "ok": bool(rel <= tol),
        "reason": "" if rel <= tol else "地面平面偏离眼高 %.1f%%（疑似整体尺度错误）" % (rel * 100),
    }


def ground_band_check(pos, uv, eye: float = 1.7, v_lo: float = 0.55,
                      v_hi: float = 0.86, tol: float = 0.20) -> dict:
    """T4（v1.22）：**非锚带**尺度检查。

    尺度门只看画面最底部（v≈0.9+），那一段正是标定锚点本身 —— 用它验收等于
    自己证明自己。这里改看中景行（v 0.55~0.86）：这些行的地面点没有参与标定，
    如果 p 偏了，它们的 y 会系统性地偏离 -eye。
    """
    v = np.asarray(uv)[:, 1]
    y = np.asarray(pos)[:, 1]
    m = (v >= v_lo) & (v <= v_hi) & np.isfinite(y)
    n = int(m.sum())
    if n < 50:
        return {"n": n, "ok": True, "rel_error": None, "tol": tol,
                "reason": ""}          # 中景没有地面（纯天空/水面）就不判失败
    med = float(np.median(y[m]))
    rel = abs(med + eye) / max(eye, 1e-6)
    return {"n": n, "band_y_median": round(med, 3), "expected": round(-eye, 3),
            "rel_error": round(rel, 4), "tol": tol, "ok": bool(rel <= tol),
            "reason": "" if rel <= tol else
                      "非锚带（中景）地面偏离眼高 %.1f%%（p 或尺度疑似偏置）" % (rel * 100)}


def report(pos, uv, tex, cam, ref, coverage_min=0.85, step=1,
           cam_verify=None, eye=None, cal=None, scale_tol=0.25,
           psnr_min=14.0) -> dict:
    """PSNR/SSIM + 覆盖率 + 尺度门 + 可选独立相机交叉门。

    eye 不为 None 时启用地面平面尺度门；cam_verify 不为 None 时用它
    （而不是生成用的 cam）再泼一次点，才能发现内参错误。cal 的 p 只做
    诊断告警，不直接判失败：双锚点默认 1500m 常给出 p≈2.3。
    """
    ref_f = ref.astype(np.float32) / 255.0 if ref.dtype == np.uint8 else ref.astype(np.float32)
    if step > 1:
        # 与网格采样点严格对齐：网格每隔 step 像素取一个顶点
        ref_f = ref_f[::step, ::step]
    h, w = ref_f.shape[:2]
    img = splat_render(pos, uv, tex, cam, (h, w))
    cov = float(np.mean(np.any(img > 1e-6, axis=2)))
    # 只在前景覆盖处算指标，避免黑洞拉低分数造成误判
    mask = np.any(img > 1e-6, axis=2)
    if mask.sum() < 100:
        return {"coverage": cov, "psnr": 0.0, "ssim": 0.0, "ok": False,
                "reason": "coverage too low"}
    p = psnr(img[mask], ref_f[mask])
    s = ssim(img * mask[..., None], ref_f * mask[..., None])
    reasons = []
    if cov < coverage_min:
        reasons.append("coverage below %.2f" % coverage_min)
    if p < psnr_min:
        reasons.append("psnr below %.1f" % psnr_min)
    ok = not reasons

    # T4：再报一个**全图口径**的 PSNR（旧实现只在覆盖区域内算，
    # 覆盖 40% 也能刷出高分）。
    p_full = psnr(img, ref_f)
    out = {"coverage": round(cov, 4), "psnr": round(p, 2), "ssim": round(s, 4),
           "psnr_full": round(p_full, 2)}

    if eye is not None:
        out["scale"] = ground_scale_check(pos, uv, eye, tol=scale_tol)
        if not out["scale"]["ok"]:
            ok = False
            reasons.append(out["scale"]["reason"])
        # T4：非锚带（中景）尺度检查，锚点自证的问题在这里被堵住
        out["scale_band"] = ground_band_check(pos, uv, eye)
        if not out["scale_band"]["ok"]:
            ok = False
            reasons.append(out["scale_band"]["reason"])

    if cam_verify is not None:
        img2 = splat_render(pos, uv, tex, cam_verify, (h, w))
        m2 = np.any(img2 > 1e-6, axis=2)
        cov2 = float(m2.mean())
        if m2.sum() >= 100:
            p2 = psnr(img2[m2], ref_f[m2])
            s2 = ssim(img2 * m2[..., None], ref_f * m2[..., None])
        else:
            p2, s2 = 0.0, 0.0
        # 除了绝对阈值，还要求交叉渲染的覆盖率不比自洽渲染低太多：
        # 内参不一致时顶点会缩到画面中间（实测 FOV 错 25% → 1.00 降到 0.59），
        # 但常数贴图的 PSNR 可能看不出变化，相对覆盖率更可靠。
        ratio = cov2 / max(cov, 1e-6)
        cross_ok = bool(cov2 >= coverage_min and p2 >= psnr_min and ratio >= 0.90)
        out["cross"] = {"coverage": round(cov2, 4), "psnr": round(p2, 2),
                        "ssim": round(s2, 4), "coverage_ratio": round(ratio, 4),
                        "ok": cross_ok}
        if not cross_ok:
            ok = False
            reasons.append("独立相机交叉校验未通过（coverage %.2f / 相对 %.2f / PSNR %.1fdB）"
                           % (cov2, ratio, p2))

    if cal is not None:
        pv = float(cal.get("p", 1.0))
        out["calibration"] = {"p": round(pv, 4), "limit": 2.0,
                              "warning": "" if pv <= 2.0 else
                              "p=%.2f > 2：深度 10%% 误差会放大成距离 %.1f%% 误差" %
                              (pv, (0.9 ** -pv - 1) * 100)}

    out["ok"] = bool(ok)
    out["reason"] = "" if ok else "; ".join(reasons)
    return out
