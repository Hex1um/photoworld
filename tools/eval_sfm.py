#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用带真值的合成场景评估 v4 的 SfM：位姿误差、定位率、重投影。

真值（gt.json）与重建之间差一个**相似变换**（尺度不可观），所以先做 Umeyama 对齐，
再报旋转/平移误差 —— 这是"重建得怎么样"唯一可量化的说法。

    python tools/eval_sfm.py --data testdata/synth1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """求 s,R,t 使 s·R·src + t ≈ dst（行向量实现）。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src)
    U, d, Vt = np.linalg.svd(cov)
    S_ = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_[2, 2] = -1
    R = U @ S_ @ Vt
    s = (np.trace(np.diag(d) @ S_) / (S ** 2).sum() * len(S)) if with_scale else 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="testdata/synth1")
    ap.add_argument("--focal-ratio", type=float, default=1.15)
    ap.add_argument("--max-side", type=int, default=960)
    ap.add_argument("--min-gap", type=float, default=0.0)
    args = ap.parse_args()

    from v4 import ingest, sfm

    gt = json.load(open(os.path.join(args.data, "gt.json"), encoding="utf-8"))
    frames = ingest.load_inputs(args.data, max_side=args.max_side, min_gap=args.min_gap)
    # gt.json 与图片同目录，ingest 会把 json 过滤掉（只认图片扩展名）✓
    res = sfm.run(frames, focal_ratio=args.focal_ratio, verbose=True)

    gt_c = np.array([v["center"] for v in gt["views"]], np.float64)
    if len(frames) != len(gt_c):
        print("警告：帧数 %d 与真值 %d 不一致，按索引对齐前 %d 个"
              % (len(frames), len(gt_c), min(len(frames), len(gt_c))))
    n = min(len(frames), len(gt_c))
    est, idx = [], []
    for i in range(n):
        p = res.poses[i] if i < len(res.poses) else None
        if p is None:
            continue
        R, t = p
        est.append(-R.T @ t.ravel())
        idx.append(i)
    if len(idx) < 3:
        print("FAIL  可定位帧不足（%d 帧），无法评估" % len(idx))
        return 1
    est = np.array(est)
    dst = gt_c[idx]
    s, R, t = umeyama(est, dst)
    aligned = (s * (R @ est.T).T) + t
    err = np.linalg.norm(aligned - dst, axis=1)
    scale_ref = float(np.linalg.norm(dst - dst.mean(0), axis=1).mean())
    # 旋转误差
    rot_err = []
    for k, i in enumerate(idx):
        Rg = np.array(gt["views"][i]["R"], np.float64)
        Re = res.poses[i][0]
        dR = Rg @ Re.T
        ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        rot_err.append(ang)
    rot_err = np.array(rot_err)

    print("\n================ v4 重建评估（%s）================" % args.data)
    print("定位        %d/%d 帧" % (len(idx), n))
    print("稀疏点      %d" % (len(res.pts) if res.pts is not None else 0))
    print("重投影 RMSE %.2f px" % res.reproj_rmse)
    print("对齐尺度    %.3f（重建 1 单位 → 真实 %.3f 单位）" % (s, s))
    print("平移误差    中位 %.3f / 最大 %.3f（场景半径 %.3f，即 %.1f%% / %.1f%%）"
          % (np.median(err), err.max(), scale_ref,
             np.median(err) / scale_ref * 100, err.max() / scale_ref * 100))
    print("旋转误差    中位 %.2f° / 最大 %.2f°" % (np.median(rot_err), rot_err.max()))
    verdict = ("PASS" if np.median(err) / scale_ref < 0.05 and np.median(rot_err) < 3.0
               else "WARN" if np.median(err) / scale_ref < 0.15 else "FAIL")
    print("判定        %s" % verdict)
    return 0 if verdict != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
