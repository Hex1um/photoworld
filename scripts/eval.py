#!/usr/bin/env python3
"""阶段 1 评测：一步预测质量 + 动作可控性 + 长 rollout 稳定性。

  python scripts/eval.py --ckpt runs/s1_3000/last.pt --out eval

产出 eval/report.md 与 eval/metrics.json。

指标
----
1. 一步预测：PSNR / SSIM，与「复制上一帧」基线对比（验收要求 ≥ +3dB）
2. 动作可控性：forward / turn_left / turn_right 各 rollout 若干步，
   用相邻帧的相位相关位移估计主运动方向，统计与动作符号一致的比例
3. 长 rollout：逐步 LPIPS 代理（用 SSIM 距离代替，避免额外依赖）曲线
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldmodel.infer.rollout import rollout, parse_actions  # noqa: E402


def to_np(t: torch.Tensor) -> np.ndarray:
    """(-1,1) CHW → (H,W,3) uint8"""
    a = ((t.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255)
    return a.astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)) / (255.0 ** 2)
    return 99.0 if mse <= 1e-12 else 10 * math.log10(1.0 / mse)


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).mean(axis=2) / 255.0
    b = b.astype(np.float32).mean(axis=2) / 255.0
    k = 11
    pad = k // 2
    def blur(x):
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


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """相位相关估计 b 相对 a 的整体位移（像素）。返回 (dx, dy)。

    互功率谱必须是 `F(b) * conj(F(a))`，峰值才落在 **b 相对 a** 的位移上。
    旧代码写反成 `F(a) * conj(F(b))`，返回值恰好是 a 相对 b 的位移（差一个负号）：
    内容右移 6 像素会报 dx=-6。修好动作语义后这会让 `turn_left` 期望的
    "场景右移"被系统性误判为失败。
    """
    ga = a.astype(np.float32).mean(axis=2)
    gb = b.astype(np.float32).mean(axis=2)
    ga = ga - ga.mean()
    gb = gb - gb.mean()
    win = np.outer(np.hanning(ga.shape[0]), np.hanning(ga.shape[1]))
    F = np.fft.fft2(gb * win) * np.conj(np.fft.fft2(ga * win))
    F = F / (np.abs(F) + 1e-8)
    r = np.fft.ifft2(F).real
    iy, ix = np.unravel_index(np.argmax(r), r.shape)
    h, w = ga.shape
    dx = ix - w if ix > w // 2 else ix
    dy = iy - h if iy > h // 2 else iy
    return float(dx), float(dy)


def load_image(path: str, size: tuple[int, int]) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    x = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(x).permute(2, 0, 1)[None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/syn", help="轨迹目录（用于取真值帧）")
    ap.add_argument("--out", default="eval")
    ap.add_argument("--size", default="384x256")
    ap.add_argument("--steps", type=int, default=8, help="Euler 采样步数")
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=8, help="一步预测用的样本数")
    ap.add_argument("--dir-steps", type=int, default=6, help="方向测试每个动作的 rollout 步数")
    args = ap.parse_args()

    w, h = [int(v) for v in args.size.lower().split("x")]
    size = (w, h)
    os.makedirs(args.out, exist_ok=True)
    report = {"ckpt": args.ckpt, "size": args.size, "euler_steps": args.steps, "cfg": args.cfg}

    # ---------- 1. 一步预测 vs 复制上一帧 ----------
    import glob
    trajs = sorted(glob.glob(os.path.join(args.data, "traj_*")))[: args.n]
    ps, ss, bp, bs = [], [], [], []
    for td in trajs:
        pj = json.load(open(os.path.join(td, "poses.json"), encoding="utf-8"))
        frames = pj["frames"]
        if len(frames) < 8:
            continue
        prev = load_image(os.path.join(td, "frames", frames[5]["file"]), size)
        tgt = load_image(os.path.join(td, "frames", frames[6]["file"]), size)
        out = rollout(prev, [torch.zeros(7)], args.ckpt, steps=args.steps, cfg_scale=args.cfg)[0]
        pred = to_np(out)
        gt = to_np(tgt[0])
        ps.append(psnr(pred, gt))
        ss.append(ssim_gray(pred, gt))
        # 基线：复制上一帧
        bp.append(psnr(to_np(prev[0]), gt))
        bs.append(ssim_gray(to_np(prev[0]), gt))
        print("  %s  pred %.2fdB / base %.2fdB" % (os.path.basename(td), ps[-1], bp[-1]), flush=True)

    if ps:
        report["one_step"] = {
            "n": len(ps),
            "psnr_model": round(float(np.mean(ps)), 2),
            "psnr_copy_baseline": round(float(np.mean(bp)), 2),
            "psnr_gain": round(float(np.mean(ps) - np.mean(bp)), 2),
            "ssim_model": round(float(np.mean(ss)), 4),
            "ssim_copy_baseline": round(float(np.mean(bs)), 4),
        }

    # ---------- 2. 动作可控性 ----------
    start = None
    for td in trajs:
        p = os.path.join(td, "frames", "000000.jpg")
        if os.path.exists(p):
            start = load_image(p, size)
            break
    dirs = {}
    if start is not None:
        for name, expect in (("forward", "zoom_in"), ("turn_left", "shift_right"),
                             ("turn_right", "shift_left")):
            frames = rollout(start, parse_actions(["%s %d" % (name, args.dir_steps)]),
                             args.ckpt, steps=args.steps, cfg_scale=args.cfg)
            arrs = [to_np(f) for f in frames]
            # forward：画面应「放大」（整体向外扩散）→ 用中位帧差能量在中心 vs 边缘的比例近似
            if expect == "zoom_in":
                d = np.abs(arrs[-1].astype(np.float32) - arrs[0].astype(np.float32)).mean(axis=2)
                hh, ww = d.shape
                cy0, cy1 = hh // 3, 2 * hh // 3
                cx0, cx1 = ww // 3, 2 * ww // 3
                center = d[cy0:cy1, cx0:cx1].mean()
                border = (d.sum() - d[cy0:cy1, cx0:cx1].sum()) / (d.size - d[cy0:cy1, cx0:cx1].size)
                ok = center > border
                dirs[name] = {"metric": "center_vs_border_energy",
                              "center": round(float(center), 2), "border": round(float(border), 2),
                              "ok": bool(ok)}
            else:
                dx, dy = phase_shift(arrs[0], arrs[-1])
                ok = (dx > 0.5) if expect == "shift_right" else (dx < -0.5)
                dirs[name] = {"metric": "phase_shift_dx", "dx": round(dx, 2), "dy": round(dy, 2),
                              "expected": expect, "ok": bool(ok)}
            print("  %-10s %s" % (name, dirs[name]), flush=True)
        report["direction"] = dirs
        report["direction_accuracy"] = round(
            sum(1 for v in dirs.values() if v.get("ok")) / max(1, len(dirs)), 3)

    # ---------- 3. 长 rollout 稳定性 ----------
    if start is not None:
        long_actions = parse_actions(["forward 20", "turn_left 20"])
        frames = rollout(start, long_actions, args.ckpt, steps=args.steps, cfg_scale=args.cfg)
        arrs = [to_np(f) for f in frames]
        curve = []
        for i in range(1, len(arrs)):
            curve.append(round(1.0 - ssim_gray(arrs[i], arrs[i - 1]), 4))
        report["long_rollout"] = {
            "frames": len(arrs),
            "flicker_curve": curve,
            "mean_flicker": round(float(np.mean(curve)), 4) if curve else None,
            "max_flicker": round(float(np.max(curve)), 4) if curve else None,
        }
        os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
        for i, a in enumerate(arrs):
            Image.fromarray(a).save(os.path.join(args.out, "frames", "%03d.jpg" % i), quality=90)
        Image.fromarray(arrs[0]).save(os.path.join(args.out, "long_rollout.gif"),
                                      save_all=True, append_images=[Image.fromarray(x) for x in arrs[1:]],
                                      duration=120, loop=0)

    json.dump(report, open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ---------- 报告 ----------
    lines = ["# 阶段 1 评测报告", "",
             "checkpoint：`%s`" % args.ckpt,
             "分辨率 %s，Euler %d 步，CFG %.1f" % (args.size, args.steps, args.cfg), ""]
    if "one_step" in report:
        o = report["one_step"]
        lines += ["## 一步预测（n=%d）" % o["n"], "",
                  "| 指标 | 本模型 | 复制上一帧基线 | 差值 |", "|---|---|---|---|",
                  "| PSNR | %.2f dB | %.2f dB | **%+.2f dB** |" % (o["psnr_model"], o["psnr_copy_baseline"], o["psnr_gain"]),
                  "| SSIM | %.4f | %.4f | %+.4f |" % (o["ssim_model"], o["ssim_copy_baseline"], o["ssim_model"] - o["ssim_copy_baseline"]),
                  ""]
        lines.append("验收要求 PSNR 增益 ≥ +3dB：**%s**" % ("达标" if o["psnr_gain"] >= 3 else "未达标"))
        lines.append("")
    if "direction" in report:
        lines += ["## 动作可控性", "", "| 动作 | 判据 | 结果 | 结论 |", "|---|---|---|---|"]
        for k, v in report["direction"].items():
            if v["metric"] == "center_vs_border_energy":
                lines.append("| %s | 中心/边缘差能量 %.2f / %.2f | — | %s |" %
                             (k, v["center"], v["border"], "✅" if v["ok"] else "❌"))
            else:
                lines.append("| %s | 相位相关 dx=%.2f | 期望 %s | %s |" %
                             (k, v["dx"], v["expected"], "✅" if v["ok"] else "❌"))
        lines += ["", "一致率 **%.1f%%**（验收 >90%%）" % (report["direction_accuracy"] * 100), ""]
    if "long_rollout" in report:
        lr = report["long_rollout"]
        lines += ["## 长 rollout 稳定性", "",
                  "%d 帧，帧间 SSIM 距离均值 %.4f，最大 %.4f" %
                  (lr["frames"], lr["mean_flicker"], lr["max_flicker"]), "",
                  "曲线：" + ", ".join("%.3f" % v for v in lr["flicker_curve"][:20]) + (" …" if len(lr["flicker_curve"]) > 20 else ""),
                  "", "帧序列见 `frames/`，动图见 `long_rollout.gif`。", ""]

    open(os.path.join(args.out, "report.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("报告写入 %s" % os.path.join(args.out, "report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
