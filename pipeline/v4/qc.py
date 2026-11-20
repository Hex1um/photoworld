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

判定（v4.1.2 收紧，避免"RMSE 很小但根本没有基线"被放行）：
  PASS  定位率 ≥ 0.7 且 视差 ≥ 1.0° 且 RMSE ≤ 3px 且 点 ≥ 100 且 填充率 ≥ 0.5
  WARN  定位率 ≥ 0.5 且 视差 ≥ 0.5° 且 RMSE ≤ 4.5px 且 点 ≥ 100（会有洞/歪）
  FAIL  定位率 < 0.4、RMSE > 6px、视差 < 0.5°、点 < 100、填充率 < 0.2 或其它
"""
from __future__ import annotations

import json
import os

import numpy as np


def summarize(res, n_frames: int, fill: float | None = None) -> dict:
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
    hard_fail = (fill is not None and fill < 0.20) or med_par < 0.5 or npts < 100
    if hard_fail or loc < 0.4 or rep > 6.0:
        verdict = "FAIL"
    elif loc >= 0.7 and med_par >= 1.0 and rep <= 3.0 and (fill is None or fill >= 0.5):
        verdict = "PASS"
    elif loc >= 0.5 and med_par >= 0.5 and rep <= 4.5 and (fill is None or fill >= 0.35):
        verdict = "WARN"
    else:
        verdict = "FAIL"

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
    if fill is not None and fill < 0.35:
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
