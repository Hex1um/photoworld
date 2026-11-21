"""重建质量报告：先给结论，再给世界。

指标都是"用户能理解 + 能据此改拍法"的那种：

| 指标 | 含义 | 差的时候该怎么办 |
|---|---|---|
| 定位率 | 有多少帧成功求出位姿 | 加重叠、别只原地转 |
| 内点中位 | 每对图 RANSAC 保留的匹配数 | 增加纹理/避免大面积纯色天空 |
| 视差中位 | 三角化的视线夹角（度） | **多走动**，纯旋转没有基线 |
| 重投影 RMSE | 三维点投回图像的平均误差（px） | 通常 <2px 才可信 |
| 点云规模 | 三角化出的三维点数 | 提高分辨率 / 增加帧 |
| 高度场填充率 | 网格里"有数据"的格子占比 | 环绕拍，不要只拍一面 |

判定（v4.1.2 收紧视差，v4.1.3 修正填充率门槛）：
  PASS  定位率 ≥ 0.7 且 视差 ≥ 1.0° 且 RMSE ≤ 3px 且 点 ≥ 100 且 填充率 ≥ 0.08 且 光度 ≤ 0.09
  WARN  定位率 ≥ 0.5 且 视差 ≥ 0.5° 且 RMSE ≤ 4.5px 且 点 ≥ 100、填充率 ≥ 0.03
  FAIL  定位率 < 0.4、RMSE > 6px、视差 < 0.5°、点 < 100、填充率 < 0.03 或其它

填充率门槛为什么是 0.08 而不是 0.5：fill = 有数据的格子数 / cells²，cells 默认 200，
也就是 4 万个格子。SfM 只有 ~1~2 万个稀疏点，数学上根本填不满一半；实测完整绕拍
的素材 fill≈0.10，只拍 18° 窄带的 fill≈0.018。旧门槛 0.2/0.5 会让**所有**合格素材
在建完后被判 FAIL 并删掉 GLB（端到端实测复现）。0.03/0.08 是按这组对照数据定的。
"""
from __future__ import annotations

import json
import os

import numpy as np


def summarize(res, n_frames: int, fill: float | None = None,
              photo_l1: float | None = None) -> dict:
    solvable = [p for p in (res.pairs or []) if p.inliers >= 50]
    inl = [p.inliers for p in solvable]
    # v4.1.2：视差中位必须把"可解但对数为 0"的配对算进去。旧版只统计 parallax > 0
    # 的配对，于是 15 组可解里只有 1 组有 5° 时中位也报 5°，把没有基线的素材捧成 PASS。
    par = [p.parallax for p in solvable]
    loc = res.localized / max(1, n_frames)
    rep = res.reproj_rmse
    npts = int(len(res.pts)) if res.pts is not None else 0
    med_par = float(np.median(par)) if par else 0.0
    med_inl = float(np.median(inl)) if inl else 0.0

    # 硬门槛：这些情况出不了可信世界，不能因为 RMSE 恰好很小就放行。
    # v4.1.1 的 WARN 只看了 loc/rep，纯旋转素材（4/7 定位、1.40px、0.25° 视差）
    # 被判 WARN 并继续建世界，与 docs/v4.md 声称的 FAIL 直接矛盾。
    hard_fail = (fill is not None and fill < 0.03) or med_par < 0.5 or npts < 100
    if hard_fail or loc < 0.4 or rep > 6.0:
        verdict = "FAIL"
    elif loc >= 0.7 and med_par >= 1.0 and rep <= 3.0 and (fill is None or fill >= 0.08):
        verdict = "PASS"
    elif loc >= 0.5 and med_par >= 0.5 and rep <= 4.5 and (fill is None or fill >= 0.03):
        verdict = "WARN"
    else:
        verdict = "FAIL"

    # 光度一致性（必须放在判定链**之后**，否则会被后面的 PASS 覆盖）
    # 这是唯一能发现"位姿整体错位"的指标：RMSE 只看点与位姿自洽，错位时照样很小。
    # 实测 lego 真素材：RMSE 0.86px / 26 帧全定位 / 视差 11°，但光度 0.148 → 世界是碎片。
    if photo_l1 is not None and photo_l1 == photo_l1:
        if photo_l1 > 0.13:
            verdict = "FAIL"
        elif photo_l1 > 0.09 and verdict == "PASS":
            verdict = "WARN"

    advice = []
    if npts < 100:
        advice.append("三角化出的三维点太少（%d）：主体纹理不足或重叠不够，请重拍" % npts)
    if loc < 0.7:
        advice.append("相邻照片的**重叠不够**：每张至少和上一张共享 1/3 画面，宁可多拍几张")
    if med_par < 1.0:
        advice.append("**视差不足**：要么原地转圈（没有基线），要么主体太远（位移相对太小）。"
                      "请靠近主体、并保持横向移动（相对主体的位移 ≥ 场景深度的 5%）")
    if med_inl < 80:
        advice.append("匹配太少：避免大片纯色（天空/白墙），可以把主体放在画面中间")
    if rep > 3.0:
        advice.append("重投影误差偏高：镜头畸变大或运动模糊，建议关防抖、慢一点走")
    if fill is not None and fill < 0.08:
        advice.append("高度场填充率低：**环绕拍**一圈（或至少 180°），不要只拍一面")

    return {
        "frames": n_frames,
        "localized": res.localized,
        "localized_ratio": round(loc, 3),
        "pairs_solvable": sum(1 for p in res.pairs if p.inliers >= 50),
        "pairs_total": len(res.pairs),
        "inliers_median": round(med_inl, 1),
        "parallax_median_deg": round(med_par, 2),
        "reproj_rmse_px": round(rep, 2),
        "points": int(len(res.pts)) if res.pts is not None else 0,
        "fill_ratio": None if fill is None else round(fill, 3),
        "photo_l1": None if photo_l1 is None else round(float(photo_l1), 4),
        "verdict": verdict,
        "advice": advice,
    }


def write(report: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "qc.json")
    def _py(v):
        # numpy 标量（float32/int64）不能直接 json：统一转原生类型
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    with open(p, "w", encoding="utf-8") as f:
        json.dump({k: _py(v) for k, v in report.items()}, f, ensure_ascii=False, indent=1)
    lines = ["# 重建质量报告", "",
             "**判定：%s**" % report["verdict"], "",
             "| 指标 | 值 |", "|---|---|",
             "| 帧数 | %d |" % report["frames"],
             "| 成功定位 | %d（%d%%） |" % (report["localized"],
                                          round(report["localized_ratio"] * 100)),
             "| 可解配对 | %d / %d |" % (report["pairs_solvable"], report["pairs_total"]),
             "| 内点中位 | %.0f |" % report["inliers_median"],
             "| 视差中位 | %.2f° |" % report["parallax_median_deg"],
             "| 重投影 RMSE | %.2f px |" % report["reproj_rmse_px"],
             "| 稀疏点 | %d |" % report["points"]]
    if report.get("fill_ratio") is not None:
        lines.append("| 高度场填充率 | %.0f%% |" % (report["fill_ratio"] * 100))
    if report["advice"]:
        lines += ["", "## 建议", ""] + ["- " + a for a in report["advice"]]
    md = os.path.join(out_dir, "qc.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return p


def photometric(frames, res, max_pts: int = 3000) -> float:
    """光度一致性：把三维点的颜色投回每一帧，和该帧真实像素比。

    为什么必须有这一项：重投影 RMSE 只能说明"点和位姿自洽"，**位姿整体错位时它照样很小**
    （实测 lego 素材：RMSE 1.21px、26/26 定位，但相机半径 10.3/13.0/8.5 完全不成圆，
    和真值差 120°+）。颜色对不上才说明几何真的错了。
    返回中位 L1 颜色误差（0..1，越小越好）。
    """
    if res.pts is None or len(res.pts) < 200 or res.colors is None:
        return float("nan")
    rng = np.random.default_rng(0)
    sel = rng.choice(len(res.pts), min(max_pts, len(res.pts)), replace=False)
    X = res.pts[sel]
    C = res.colors[sel]
    errs = []
    for fi, p in enumerate(res.poses):
        if p is None or fi >= len(frames):
            continue
        R, t = p
        K = res.cams[fi].K
        Xc = (R @ X.T + t).T
        z = Xc[:, 2]
        ok = z > 0.05
        if ok.sum() < 30:
            continue
        uv = (K @ Xc[ok].T).T
        u = uv[:, 0] / z[ok] + 0.0
        v = uv[:, 1] / z[ok] + 0.0
        h, w = frames[fi].rgb.shape[:2]
        m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if m.sum() < 30:
            continue
        px = frames[fi].rgb[v[m].astype(int), u[m].astype(int)].astype(np.float32) / 255.0
        errs.append(np.abs(px - C[ok][m]).mean(axis=1))
    if not errs:
        return float("nan")
    return float(np.median(np.concatenate(errs)))
