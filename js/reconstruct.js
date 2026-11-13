/* 照片 -> 场景参数：天际线提取、雾色/天光取样、太阳方位、调色板
 *
 * 核心是 analyzePixels(rgba, W, H, name, options)：纯函数、不依赖 DOM。
 * 浏览器（analyze）与 Node 批量评测（tools/eval.mjs）跑同一份逻辑；
 * 所有阈值集中在 DEFAULTS，评测脚本可以扫描这些参数做数据驱动的调参。
 */
(function (global) {
  'use strict';

  const { clamp, lerp, mulberry32, hashString } = global.PW.util;

  const FOV_H_DEG = 62;
  const PANO_TILES = 6;
  const NCOL = 360;

  const DEFAULTS = {
    analyzeW: 420,
    skyRows: 0.035,
    // 以下阈值由 503 张网图批量评测得出（tools/eval.mjs，score 0.704 vs 0.645）
    colorThresh: 0.050,
    strongThresh: 0.130,
    gradThresh: 0.020,
    smoothRounds: 2,
    medianGuard: 0.08,     // 邻域中值容差（压掉孤立极高点）
    anchorPct: 0.72,
    anchorWindow: 0.13,
    anchorCompress: 0.22,
    horizonPct: 0.82,
    horizonLo: 0.34,
    horizonHi: 0.86,
    heightDiv: 0.18,
    heightMax: 1.12
  };

  function colorDist2(r1, g1, b1, r2, g2, b2) {
    const dr = r1 - r2, dg = g1 - g2, db = b1 - b2;
    return (dr * dr + dg * dg + db * db) / (255 * 255 * 3);
  }

  function lum(r, g, b) { return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255; }

  /* 全景 UV：以照片中心为世界正前方，向两侧镜像平铺 */
  function panoU(angle) {
    let t = ((angle - Math.PI) / (Math.PI * 2)) % 1;
    if (t < 0) t += 1;
    const q = 0.5 - t * PANO_TILES;
    let m = q % 2;
    if (m < 0) m += 2;
    return m <= 1 ? m : 2 - m;
  }

  /* ---------------- 纯像素分析 ---------------- */
  function analyzePixels(data, W, H, name, options) {
    const O = Object.assign({}, DEFAULTS, options || {});
    const px = (x, y) => {
      const i = (y * W + x) * 4;
      return [data[i], data[i + 1], data[i + 2]];
    };

    /* --- 1. 天空参考色 --- */
    let skyR = 0, skyG = 0, skyB = 0, skyN = 0;
    const topRows = Math.max(2, Math.floor(H * O.skyRows));
    for (let y = 0; y < topRows; y++) {
      for (let x = 0; x < W; x++) {
        const c = px(x, y);
        skyR += c[0]; skyG += c[1]; skyB += c[2]; skyN++;
      }
    }
    skyR /= skyN; skyG /= skyN; skyB /= skyN;

    /* --- 2. 逐列找地形上缘 --- */
    const rawTop = new Float32Array(NCOL);
    for (let i = 0; i < NCOL; i++) {
      const x = Math.min(W - 1, Math.round((i + 0.5) / NCOL * W));
      let found = -1;
      let prevL = lum(...px(x, Math.max(0, Math.floor(H * 0.015))));
      for (let y = Math.max(2, Math.floor(H * 0.02)); y < H; y++) {
        const c = px(x, y);
        const d = colorDist2(c[0], c[1], c[2], skyR, skyG, skyB);
        const l = lum(c[0], c[1], c[2]);
        const grad = Math.abs(l - prevL);
        prevL = l;
        if (d > O.colorThresh && (grad > O.gradThresh || d > O.strongThresh)) {
          let ok = 0;
          for (let k = 0; k < 3; k++) {
            const yy = Math.min(H - 1, y + k);
            const cc = px(x, yy);
            if (colorDist2(cc[0], cc[1], cc[2], skyR, skyG, skyB) > O.colorThresh * 0.73) ok++;
          }
          if (ok >= 3) { found = y; break; }
        }
      }
      rawTop[i] = found < 0 ? 1.0 : found / H;
    }

    /* --- 3. 环形中值 + 多轮均值平滑 --- */
    const mid = new Float32Array(NCOL);
    for (let i = 0; i < NCOL; i++) {
      const a = rawTop[(i - 1 + NCOL) % NCOL], b = rawTop[i], c = rawTop[(i + 1) % NCOL];
      mid[i] = Math.max(Math.min(a, b), Math.min(Math.max(a, b), c));
    }
    const top = new Float32Array(NCOL);
    for (let i = 0; i < NCOL; i++) {
      let s = 0;
      for (let k = -3; k <= 3; k++) s += mid[(i + k + NCOL) % NCOL];
      top[i] = s / 7;
    }
    for (let pass = 0; pass < O.smoothRounds; pass++) {
      const src = top.slice();
      for (let i = 0; i < NCOL; i++) {
        let s = 0;
        for (let k = -2; k <= 2; k++) s += src[(i + k + NCOL) % NCOL];
        top[i] = s / 5;
      }
    }

    /* 邻域一致性：孤立极高点（多半是云被误判成地形）压回邻域中值 */
    for (let pass = 0; pass < 2; pass++) {
      const src = top.slice();
      for (let i = 0; i < NCOL; i++) {
        const neigh = [];
        for (let k = -6; k <= 6; k++) neigh.push(src[(i + k + NCOL) % NCOL]);
        neigh.sort((a, b) => a - b);
        const med = neigh[Math.floor(neigh.length / 2)];
        if (src[i] < med - O.medianGuard) top[i] = med - O.medianGuard;
      }
    }

    /* --- 4. 锚点约束 --- */
    const sortedT = Array.from(top).sort((a, b) => a - b);
    const anchorTop = sortedT[Math.floor(NCOL * O.anchorPct)];
    for (let i = 0; i < NCOL; i++) {
      top[i] = Math.min(top[i], anchorTop - 0.005);
      const lim = anchorTop - O.anchorWindow;
      if (top[i] < lim) top[i] = lim + (top[i] - lim) * O.anchorCompress;
    }

    /* --- 5. 地平线与参考色 --- */
    const sorted = Array.from(top).sort((a, b) => a - b);
    const horizonRaw = sorted[Math.floor(NCOL * O.horizonPct)];
    const horizon = clamp(horizonRaw, O.horizonLo, O.horizonHi);

    const avgBand = (y0, y1) => {
      let r = 0, g = 0, b = 0, n = 0;
      const ya = Math.max(0, Math.floor(y0 * H)), yb = Math.min(H, Math.ceil(y1 * H));
      for (let y = ya; y < yb; y++) {
        for (let x = 0; x < W; x += 2) {
          const c = px(x, y);
          r += c[0]; g += c[1]; b += c[2]; n++;
        }
      }
      n = Math.max(1, n);
      return [r / n / 255, g / n / 255, b / n / 255];
    };

    const fog = avgBand(horizon - 0.03, horizon + 0.025);
    const ground = avgBand(horizon + 0.12, 1.0);
    const skyTop = avgBand(0, 0.05);
    const skyHorizon = avgBand(Math.max(0.02, horizon * 0.45), horizon * 0.92);

    /* --- 6. 太阳位置：天空区域最亮处 --- */
    let bestScore = -1, sunX = W * 0.5, sunY = H * horizon * 0.4;
    for (let y = Math.floor(H * 0.02); y < H * horizon * 0.95; y += 2) {
      for (let x = 0; x < W; x += 2) {
        const c = px(x, y);
        const l = lum(c[0], c[1], c[2]);
        const warmth = (c[0] - c[2]) / 255;
        const score = l + warmth * 0.35 - Math.abs(y / H - horizon * 0.35) * 0.25;
        if (score > bestScore) { bestScore = score; sunX = x; sunY = y; }
      }
    }
    const sunLum = lum(...px(Math.round(sunX), Math.round(sunY)));
    const sunU = clamp(sunX / W, 0.05, 0.95);
    const sunElev = clamp(0.12 + (1 - sunY / (H * horizon)) * 0.55, 0.12, 0.7);
    const sunAz = Math.PI + (0.5 - sunU) * (Math.PI * 2 / PANO_TILES);

    /* --- 7. 调色板 --- */
    const rng = mulberry32(hashString(String(name || 'world')));
    const palette = [];
    for (let k = 0; k < 10; k++) {
      for (let tries = 0; tries < 24; tries++) {
        const x = Math.floor(rng() * W);
        const y = Math.floor((0.55 + rng() * 0.44) * H);
        const c = px(x, y);
        const l = lum(c[0], c[1], c[2]);
        if (l > 0.16 && l < 0.86) { palette.push([c[0] / 255, c[1] / 255, c[2] / 255]); break; }
      }
    }
    if (!palette.length) palette.push([ground[0], ground[1], ground[2]]);

    /* --- 8. 场景倾向与整体亮度 --- */
    let green = 0, cold = 0, meanL = 0, sat = 0, cnt = 0;
    let fgGreen = 0, fgCnt = 0;
    for (let y = Math.floor(H * 0.5); y < H; y += 2) {
      for (let x = 0; x < W; x += 2) {
        const c = px(x, y);
        green += (c[1] - (c[0] + c[2]) / 2) / 255;
        cold += (c[2] - c[0]) / 255;
        meanL += lum(c[0], c[1], c[2]);
        const mx = Math.max(c[0], c[1], c[2]), mn = Math.min(c[0], c[1], c[2]);
        sat += (mx - mn) / 255;
        cnt++;
        if (y > H * 0.76) { fgGreen += (c[1] - (c[0] + c[2]) / 2) / 255; fgCnt++; }
      }
    }
    green /= cnt; cold /= cnt; meanL /= cnt; sat /= cnt;
    fgGreen /= Math.max(1, fgCnt);

    let allL = 0, allN = 0;
    for (let y = 0; y < H; y += 2) {
      for (let x = 0; x < W; x += 2) {
        const c = px(x, y);
        allL += lum(c[0], c[1], c[2]);
        allN++;
      }
    }
    const meanLum = allL / Math.max(1, allN);

    const isGreen = fgGreen > 0.028;
    const isCold = cold > 0.02 && meanL < 0.55;
    const isSnow = meanL > 0.55 && sat < 0.16 && cold > 0.01;
    const exposure = clamp(0.35 / Math.max(meanL, 0.12), 0.88, 1.4);

    /* --- 9. 天际线 -> 高度 --- */
    const heights = new Float32Array(NCOL);
    let satCount = 0;
    for (let i = 0; i < NCOL; i++) {
      const raw = clamp((horizon - top[i]) / O.heightDiv, 0, 3.0);
      const v = clamp(O.heightMax * Math.tanh(raw * 0.72) / Math.tanh(O.heightMax * 0.72), 0, O.heightMax);
      heights[i] = v;
      if (v > O.heightMax * 0.9) satCount++;
    }

    /* 逐列主色 */
    const colTint = new Float32Array(NCOL * 3);
    for (let i = 0; i < NCOL; i++) {
      const x = Math.min(W - 1, Math.round((i + 0.5) / NCOL * W));
      const y = clamp(Math.round(lerp(top[i], horizon, 0.45) * H), 0, H - 1);
      const c = px(x, y);
      colTint[i * 3] = c[0] / 255;
      colTint[i * 3 + 1] = c[1] / 255;
      colTint[i * 3 + 2] = c[2] / 255;
    }

    /* --- 10. 质量指标（评测用） --- */
    let skyMiss = 0, hSum = 0, hSq = 0, minTop = 1;
    for (let i = 0; i < NCOL; i++) {
      if (rawTop[i] >= 0.995) skyMiss++;
      hSum += heights[i]; hSq += heights[i] * heights[i];
      minTop = Math.min(minTop, top[i]);
    }
    const hMean = hSum / NCOL;
    const hStd = Math.sqrt(Math.max(0, hSq / NCOL - hMean * hMean));

    let fMean = 0, fSq = 0, fN = 0;
    for (let y = Math.floor(H * 0.78); y < H; y += 2) {
      for (let x = 0; x < W; x += 2) {
        const l = lum(...px(x, y));
        fMean += l; fSq += l * l; fN++;
      }
    }
    fMean /= Math.max(1, fN);
    const fStd = Math.sqrt(Math.max(0, fSq / Math.max(1, fN) - fMean * fMean));

    const metrics = {
      horizon: +horizon.toFixed(3),
      horizonClipped: (horizonRaw < O.horizonLo || horizonRaw > O.horizonHi) ? 1 : 0,
      skyMissRate: +(skyMiss / NCOL).toFixed(3),
      satRate: +(satCount / NCOL).toFixed(3),
      hMean: +hMean.toFixed(3),
      hStd: +hStd.toFixed(3),
      minTop: +minTop.toFixed(3),
      fgStd: +fStd.toFixed(3),
      meanLum: +meanLum.toFixed(3),
      sunLum: +sunLum.toFixed(3)
    };

    let score = 1;
    score -= Math.min(0.40, Math.abs(horizon - 0.52) * 0.9);
    score -= Math.min(0.35, metrics.satRate * 0.5);
    score -= Math.min(0.30, metrics.skyMissRate * 0.6);
    score -= Math.min(0.20, Math.max(0, 0.05 - metrics.hStd) * 3);
    score -= Math.min(0.20, Math.max(0, 0.06 - metrics.fgStd) * 2);
    metrics.score = +clamp(score, 0, 1).toFixed(3);

    return {
      name: name || 'my photo',
      W, H,
      skyRef: [skyR / 255, skyG / 255, skyB / 255],
      skyTop, skyHorizon,
      fog, ground, palette,
      horizon,
      top, heights, colTint,
      sunAz, sunElev, sunLum, sunU,
      isGreen: isGreen || isSnow, isCold, isSnow,
      brightness: meanL,
      meanLum,
      exposure,
      metrics,
      options: O
    };
  }

  /* ---------------- 浏览器入口 ---------------- */
  function analyze(img, name, options) {
    const W = (options && options.analyzeW) || DEFAULTS.analyzeW;
    const H = Math.max(120, Math.round(W * img.height / img.width));
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, W, H);
    let data;
    try {
      data = ctx.getImageData(0, 0, W, H).data;
    } catch (e) {
      throw new Error('无法读取图像像素（可能是跨域或本地文件限制），请通过网页访问。');
    }
    const a = analyzePixels(data, W, H, name, options);
    a.canvas = cv;
    a.pixels = { data, W, H };
    return a;
  }

  global.PW.reconstruct = {
    analyze, analyzePixels, panoU, DEFAULTS, PANO_TILES, FOV_H_DEG, NCOL
  };
})(window);
