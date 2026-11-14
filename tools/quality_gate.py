#!/usr/bin/env python3
"""视觉质量门：REFERENCE | RENDER | OVERLAY + 对齐误差指标。

复刻 google/three-live 的 `three-live-cli from-image --reference`
与 img2threejs 的 quality-gated comparison sheet，但零依赖、离线可跑：
先把参考照片等比缩放/居中裁剪到渲染截图尺寸，只比较中央「照片覆盖区」
（避开 HUD 与四周程序化环境），输出灰度 MAE、去均值 MAE、边缘 IoU、
亮度比，以及一张三联对照图。

用法：
  python tools/quality_gate.py --shot ab/final_alpine.png \
      --ref worlds/alpine.jpg --out ab/quality_alpine.png

截图由 `node tools/shot.mjs --url ".../?w=alpine&instant=1" --out ...` 生成。
指标只用于跨版本 A/B（同样的照片、同样的机位），不同照片之间不可比。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image, ImageDraw


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0


def align_to(ref: np.ndarray, shot: np.ndarray) -> np.ndarray:
    """等比缩放 + 居中裁剪，使参考图与渲染截图同尺寸。"""
    h, w = shot.shape[:2]
    rh, rw = ref.shape[:2]
    s = max(w / rw, h / rh)
    nw, nh = int(round(rw * s)), int(round(rh * s))
    im = Image.fromarray((ref * 255).astype('uint8')).resize((nw, nh), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    x0, y0 = max(0, (nw - w) // 2), max(0, (nh - h) // 2)
    return arr[y0:y0 + h, x0:x0 + w]


def edge_mask(gray: np.ndarray, tau: float = 0.06) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.hypot(gx, gy) > tau


def to_img(arr: np.ndarray, width: int) -> Image.Image:
    h, w = arr.shape[:2]
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype('uint8'))
    return im.resize((width, int(round(h * width / w))), Image.LANCZOS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shot', required=True, help='渲染截图 PNG')
    ap.add_argument('--ref', required=True, help='原照片')
    ap.add_argument('--out', required=True, help='三联对照图输出路径')
    ap.add_argument('--width', type=int, default=480)
    # T10（v1.22）：质量门必须有阈值 + 非零退出码，否则"门"只是打印。
    #   阈值口径来自跨版本 A/B：gray_mae 是整体色差、mean_centered_mae 去掉
    #   整体色调偏移后看结构、edge_iou 看轮廓重合度。
    ap.add_argument('--max-mae', type=float, default=26.0, help='灰度 MAE 上限')
    ap.add_argument('--max-centered', type=float, default=18.0, help='去均值 MAE 上限')
    ap.add_argument('--min-iou', type=float, default=0.30, help='边缘 IoU 下限')
    ap.add_argument('--max-luma-ratio', type=float, default=1.45, help='亮度比上限')
    ap.add_argument('--allow-fail', action='store_true', help='只看数字，不因超阈值失败')
    args = ap.parse_args()

    shot = load_rgb(args.shot)
    ref = align_to(load_rgb(args.ref), shot)
    h, w = shot.shape[:2]
    # 中央照片覆盖区：排除 HUD 与四周程序化环境
    roi = (slice(int(h * 0.30), int(h * 0.92)), slice(int(w * 0.16), int(w * 0.84)))
    a, b = ref[roi], shot[roi]
    ga, gb = a.mean(axis=2), b.mean(axis=2)

    raw = float(np.abs(a - b).mean() * 255)
    centered = float(np.abs((ga - ga.mean()) - (gb - gb.mean())).mean() * 255)
    ea, eb = edge_mask(ga), edge_mask(gb)
    edge_iou = float((ea & eb).sum() / max(1, (ea | eb).sum()))
    luma_ratio = float(gb.mean() / max(1e-6, ga.mean()))

    a_img, b_img = to_img(ref, args.width), to_img(shot, args.width)
    ov = to_img((ref + shot) * 0.5, args.width)
    hh = a_img.height
    sheet = Image.new('RGB', (args.width * 3 + 24, hh + 30), (24, 26, 30))
    sheet.paste(a_img, (8, 26))
    sheet.paste(b_img, (args.width + 16, 26))
    sheet.paste(ov, (args.width * 2 + 24, 26))
    d = ImageDraw.Draw(sheet)
    for x, label in [(12, 'REFERENCE'), (args.width + 20, 'RENDER'),
                     (args.width * 2 + 28, 'OVERLAY 50/50')]:
        d.text((x, 8), label, fill=(225, 228, 235))
    sheet.save(args.out)

    fails = []
    if raw > args.max_mae:
        fails.append('gray_mae %.1f > %.1f' % (raw, args.max_mae))
    if centered > args.max_centered:
        fails.append('centered_mae %.1f > %.1f' % (centered, args.max_centered))
    if edge_iou < args.min_iou:
        fails.append('edge_iou %.3f < %.3f' % (edge_iou, args.min_iou))
    if luma_ratio > args.max_luma_ratio:
        fails.append('luma_ratio %.3f > %.3f' % (luma_ratio, args.max_luma_ratio))
    ok = not fails
    print(json.dumps({
        'shot': args.shot, 'ref': args.ref, 'sheet': args.out,
        'gray_mae': round(raw, 2), 'mean_centered_mae': round(centered, 2),
        'edge_iou': round(edge_iou, 4), 'luma_ratio': round(luma_ratio, 4),
        'ok': ok, 'fails': fails,
        'roi': '%d..%d x %d..%d' % (int(w * 0.16), int(w * 0.84),
                                   int(h * 0.30), int(h * 0.92))
    }, ensure_ascii=False, indent=2))
    if not ok and not args.allow_fail:
        print('质量门未通过：' + '；'.join(fails))
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
