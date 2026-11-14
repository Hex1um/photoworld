#!/usr/bin/env node
/**
 * 清晰度指标（纯函数、零依赖、不碰 DOM）。
 *
 * 为什么不用 multiview 原有的 detail（平均梯度）：它把「锐利的边缘」和
 * 「大片低对比度的雾/底色渐变」混成同一个数，v1.17 的实测里 0.0113 → 0.0114
 * 根本分不出糊片和雾。这里拆成三类互补量：
 *   hfEnergy      3×3 盒式模糊的残差均值 —— 局部高频总量，对贴图放大糊掉最敏感
 *   lapVar        拉普拉斯响应的方差（×1000）—— 边缘锐度，纯色渐变几乎不贡献
 *   localContrast 5×5 窗口亮度标准差的均值 —— 局部对比幅度，与高频量互相印证
 * 并分区统计：中央 ROI（照片几何主体）、远景带（地平线附近）、全图。
 * 天空与程序化地面是低频区，全图均值会被它们稀释，所以 A/B 优先看 centerHF
 * 与 centerLapVar。
 *
 *   import { clarityMetrics } from './clarity.mjs';
 *   const c = clarityMetrics(L, w, h);   // L = Float32Array，亮度 0..1
 */
'use strict';

/** 亮度数组 -> 3×3 盒式模糊（可分离，O(N)） */
function boxBlur3(L, w, h) {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      tmp[row + x] = (L[row + (x > 0 ? x - 1 : 0)] + L[row + x]
        + L[row + (x < w - 1 ? x + 1 : 0)]) / 3;
    }
  }
  for (let y = 0; y < h; y++) {
    const row = y * w;
    const up = (y > 0 ? y - 1 : 0) * w;
    const dn = (y < h - 1 ? y + 1 : 0) * w;
    for (let x = 0; x < w; x++) out[row + x] = (tmp[up + x] + tmp[row + x] + tmp[dn + x]) / 3;
  }
  return out;
}

/** 积分图求 5×5 局部标准差均值（只在中央 ROI 采样） */
function localContrast(L, w, h, x0, x1, y0, y1) {
  const sw = w + 1;
  const S = new Float64Array(sw * (h + 1));
  const S2 = new Float64Array(sw * (h + 1));
  for (let y = 0; y < h; y++) {
    let rs = 0, rs2 = 0;
    for (let x = 0; x < w; x++) {
      const v = L[y * w + x];
      rs += v; rs2 += v * v;
      const i = (y + 1) * sw + (x + 1);
      S[i] = S[y * sw + (x + 1)] + rs;
      S2[i] = S2[y * sw + (x + 1)] + rs2;
    }
  }
  const r = 2, step = 2;
  let sum = 0, n = 0;
  for (let y = Math.max(r, y0); y < Math.min(h - r, y1); y += step) {
    for (let x = Math.max(r, x0); x < Math.min(w - r, x1); x += step) {
      const xa = x - r, xb = x + r + 1, ya = y - r, yb = y + r + 1;
      const area = (xb - xa) * (yb - ya);
      const s = S[yb * sw + xb] - S[ya * sw + xb] - S[yb * sw + xa] + S[ya * sw + xa];
      const s2 = S2[yb * sw + xb] - S2[ya * sw + xb] - S2[yb * sw + xa] + S2[ya * sw + xa];
      const m = s / area;
      sum += Math.sqrt(Math.max(0, s2 / area - m * m));
      n++;
    }
  }
  return sum / Math.max(1, n);
}

const variance = (s, sq, n) => (n ? Math.max(0, sq / n - (s / n) ** 2) : 0);

export function clarityMetrics(L, w, h) {
  if (!L || w < 3 || h < 3 || L.length < w * h) {
    return { hfEnergy: 0, farHF: 0, centerHF: 0, lapVar: 0, centerLapVar: 0, localContrast: 0 };
  }
  const B = boxBlur3(L, w, h);
  const cx0 = Math.floor(w * 0.25), cx1 = Math.floor(w * 0.75);
  const cy0 = Math.floor(h * 0.25), cy1 = Math.floor(h * 0.75);
  const fy0 = Math.floor(h * 0.20), fy1 = Math.floor(h * 0.55);

  let hfSum = 0, hfN = 0;
  let farSum = 0, farN = 0;
  let cHfSum = 0, cN = 0;
  let lapS = 0, lapSq = 0, lapN = 0;
  let cLapS = 0, cLapSq = 0, cLapN = 0;

  for (let y = 1; y < h - 1; y += 2) {
    const inFar = y >= fy0 && y < fy1;
    const inCenterY = y >= cy0 && y < cy1;
    for (let x = 1; x < w - 1; x += 2) {
      const k = y * w + x;
      const hf = Math.abs(L[k] - B[k]);
      hfSum += hf; hfN++;
      if (inFar) { farSum += hf; farN++; }
      const inCenter = inCenterY && x >= cx0 && x < cx1;
      if (inCenter) { cHfSum += hf; cN++; }
      const lap = 4 * L[k] - L[k - 1] - L[k + 1] - L[k - w] - L[k + w];
      lapS += lap; lapSq += lap * lap; lapN++;
      if (inCenter) { cLapS += lap; cLapSq += lap * lap; cLapN++; }
    }
  }
  const lc = localContrast(L, w, h, cx0, cx1, cy0, cy1);
  return {
    hfEnergy: +(hfSum / Math.max(1, hfN)).toFixed(5),
    farHF: +(farSum / Math.max(1, farN)).toFixed(5),
    centerHF: +(cHfSum / Math.max(1, cN)).toFixed(5),
    lapVar: +(variance(lapS, lapSq, lapN) * 1000).toFixed(4),
    centerLapVar: +(variance(cLapS, cLapSq, cLapN) * 1000).toFixed(4),
    localContrast: +lc.toFixed(5)
  };
}

/** A/B 方向：true = 越大越好 */
export const HIGHER_IS_BETTER = {
  hfEnergy: true, farHF: true, centerHF: true,
  lapVar: true, centerLapVar: true, localContrast: true, detail: true,
  voidFrac: false, innerVoid: false, lowVoid: false, borderVoid: false,
  darkFrac: false, stripe: false
};

export const CLARITY_KEYS = ['hfEnergy', 'farHF', 'centerHF', 'lapVar', 'centerLapVar', 'localContrast'];

/** 采集条件不一致时指标不可比：返回不一致的字段列表 */
export function captureDiff(a, b) {
  const keys = ['w', 'h', 'dpr', 'gl', 'views', 'radius', 'center', 'pitch', 'settle', 'clarity'];
  if (!a || !b) return keys;
  return keys.filter((k) => JSON.stringify(a[k]) !== JSON.stringify(b[k]));
}
