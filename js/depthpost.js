/* 深度后处理（v2.15.6）：治「遮挡边界竖直条纹 / 竖面被拉长」+「横移露洞」。
 *
 * 背景（报告 P0-3 与长文《移动端几何修复方案》方案一）：
 *   Depth Anything V2 的深度在**遮挡边界上是平滑过渡的**（编码器步长 + 平滑先验），
 *   轮廓两侧 3~8 像素的深度是前景与背景的**混合值**。逐像素反投影后，
 *   这些像素在前景和背景之间连成一层「斜坡片」——岩壁是竖直面、相机近乎平视时，
 *   斜坡片正好投影成竖直条纹。这是深度图里的病，换点云/网格表示都躲不掉。
 *
 * 做法（三步，全部在 CPU 上一次性完成，漫游期零成本）：
 *   1) 联合边缘带：5×5 窗口的相对深度跨度 + 灰度梯度，标出"深度不连续带"；
 *   2) 重切分：带内像素沿深度梯度方向向两侧取样，按颜色归属到近侧或远侧，
 *      再把深度**压扁进 3% 厚的薄层**（保留带内次序 + 确定性抖动打断条纹）。
 *      判不动时按深度就近归属，**不丢点** —— 点云盖得住洞，网格（?splat=0）盖不住；
 *   3) 双层窄带：前景轮廓内侧 k 像素内额外发一层"远侧深度"的点，
 *      原始视角被近点遮住，相机横移后正好填补露出的背景（窄带版 LDI）。
 *
 * 与长文的差异（有意为之，均为本仓库的工程约束）：
 *   · 长文用「同侧带内深度排序」算 rank01，这里用 (z-zn)/(zf-zn) 的归一化位置
 *     代替 —— 单调性一致、保留次序，但 O(1) 不需要逐行排序（320×200 全图排序
 *     在低端手机上是毫秒级抖动的主要来源）。
 *   · 长文在归属判不动时丢点，这里改为按深度就近归属：网格模式丢点会直接留洞。
 *   · 输入是**米制深度**（调用方已用 metric(d) 换算），阈值用相对量，不受标定尺度影响。
 */
(function (global) {
  'use strict';

  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

  /* 确定性哈希：保证同一张图每次加载的抖动完全一致（否则重载后条纹位置会跳） */
  function hash01(i) {
    const x = Math.sin(i * 12.9898) * 43758.5453;
    return x - Math.floor(x);
  }

  /* 小数组中值（原地插入排序，n ≤ 16） */
  function median(buf, n) {
    for (let a = 1; a < n; a++) {
      const v = buf[a];
      let b = a - 1;
      while (b >= 0 && buf[b] > v) { buf[b + 1] = buf[b]; b--; }
      buf[b + 1] = v;
    }
    return buf[n >> 1];
  }

  /* ---------------- 1) 联合边缘带 ---------------- */
  function edgeBand(z, lum, alpha, W, H, opt) {
    opt = opt || {};
    const spanThr = opt.bandSpan === undefined ? 0.12 : opt.bandSpan;
    const gradThr = opt.bandGrad === undefined ? 8 : opt.bandGrad;   // 0..255 亮度差
    const N = W * H;
    const band = new Uint8Array(N);
    const gx = new Float32Array(N), gy = new Float32Array(N);
    const useLum = !!(lum && gradThr > 0);
    for (let y = 2; y < H - 2; y++) {
      for (let x = 2; x < W - 2; x++) {
        const i = y * W + x;
        if (alpha && !alpha[i]) continue;                 // 天空/边界像素不参与
        let zmin = z[i], zmax = z[i];
        for (let dy = -2; dy <= 2; dy++) {
          const row = (y + dy) * W;
          for (let dx = -2; dx <= 2; dx++) {
            const v = z[row + x + dx];
            if (v < zmin) zmin = v;
            if (v > zmax) zmax = v;
          }
        }
        const rel = (zmax - zmin) / Math.max(zmin, 1e-3);
        if (rel <= spanThr) continue;
        if (useLum) {
          const g = Math.max(
            Math.abs(lum[i + 1] - lum[i - 1]),
            Math.abs(lum[i + W] - lum[i - W])) * 255;
          if (g <= gradThr) continue;
        }
        band[i] = 1;
        gx[i] = z[i + 1] - z[i - 1];
        gy[i] = z[i + W] - z[i - W];
      }
    }
    /* 膨胀 1 像素：真正拉出"斜坡片"的是带外紧邻的那一圈像素（混合值的边缘），
       只标裸带会留下一圈细条纹。 */
    if (opt.dilate !== 0) {
      const src = band.slice();
      for (let y = 1; y < H - 1; y++) {
        for (let x = 1; x < W - 1; x++) {
          const i = y * W + x;
          if (src[i]) continue;
          if (src[i - 1] || src[i + 1] || src[i - W] || src[i + W]) {
            band[i] = 1;
            gx[i] = gx[src[i - 1] ? i - 1 : (src[i + 1] ? i + 1 : (src[i - W] ? i - W : i + W))];
            gy[i] = gy[src[i - 1] ? i - 1 : (src[i + 1] ? i + 1 : (src[i - W] ? i - W : i + W))];
          }
        }
      }
    }
    return { band, gx, gy };
  }

  /* ---------------- 2) 重切分：归属 + 压扁到薄层 ---------------- */
  function recut(z, lum, band, gx, gy, W, H, opt) {
    opt = opt || {};
    const probe = opt.probe === undefined ? 5 : opt.probe;
    const thin = opt.thin === undefined ? 0.03 : opt.thin;  // 薄层厚度（相对）
    const farLo = opt.farLo === undefined ? 0.92 : opt.farLo;
    const farHi = opt.farHi === undefined ? 1.12 : opt.farHi;
    const colThr = opt.colThr === undefined ? 0.031 : opt.colThr; // ≈8/255
    const N = W * H;
    const out = Float32Array.from(z);
    const side = new Int8Array(N);      // 1 = 近侧、2 = 远侧、0 = 未处理
    const zn = new Float32Array(N);
    const zf = new Float32Array(N);
    const zA = new Float32Array(probe), lA = new Float32Array(probe);
    const zB = new Float32Array(probe), lB = new Float32Array(probe);
    let recutN = 0, nearN = 0, farN = 0, ambN = 0;
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const i = y * W + x;
        if (!band[i]) continue;
        let dx = gx[i], dy = gy[i];
        const L = Math.hypot(dx, dy);
        if (!(L > 1e-9)) continue;                 // 梯度为零：无方向，维持原深度
        dx /= L; dy /= L;
        let na = 0, nb = 0;
        for (let s = 1; s <= probe * 3; s++) {
          if (na >= probe && nb >= probe) break;
          const px = Math.round(x + dx * s), py = Math.round(y + dy * s);
          if (px < 0 || px >= W || py < 0 || py >= H) break;
          const j = py * W + px;
          if (!band[j] && na < probe) { zA[na] = z[j]; lA[na] = lum ? lum[j] : 0; na++; }
          const qx = Math.round(x - dx * s), qy = Math.round(y - dy * s);
          if (qx < 0 || qx >= W || qy < 0 || qy >= H) break;
          const k2 = qy * W + qx;
          if (!band[k2] && nb < probe) { zB[nb] = z[k2]; lB[nb] = lum ? lum[k2] : 0; nb++; }
        }
        if (na < 2 || nb < 2) continue;            // 两侧取样不足：保持原样
        const zmA = median(zA, na), zmB = median(zB, nb);
        const lmA = lum ? median(lA, na) : 0, lmB = lum ? median(lB, nb) : 0;
        const aNear = zmA < zmB;
        const zN = aNear ? zmA : zmB;
        const zF = aNear ? zmB : zmA;
        if (!(zF > zN * 1.02)) continue;           // 两侧深度几乎相同：不是真断崖
        zn[i] = zN; zf[i] = zF;
        const frac = clamp((z[i] - zN) / (zF - zN), 0, 1);
        recutN++;
        let toNear;
        if (lum) {
          const dN = Math.abs(lum[i] - (aNear ? lmA : lmB));
          const dF = Math.abs(lum[i] - (aNear ? lmB : lmA));
          if (Math.abs(dN - dF) < colThr) { toNear = frac < 0.5; ambN++; }
          else toNear = dN < dF;
        } else {
          toNear = frac < 0.5;
          ambN++;
        }
        if (toNear) {
          side[i] = 1; nearN++;
          out[i] = zN * (1 + thin * frac) + hash01(i) * zN * 0.004;
        } else {
          side[i] = 2; farN++;
          out[i] = clamp(z[i], zF * farLo, zF * farHi) + hash01(i) * zF * 0.004;
        }
      }
    }
    return { depth: out, side, zn, zf, stats: { band: recutN, near: nearN, far: farN, ambiguous: ambN } };
  }

  /* ---------------- 3) 双层窄带：前景轮廓内侧补远深度点 ----------------
     side===1（近侧）且到最近的 side===2 像素 ≤ k 的点，额外发一层 zf 深度的点。
     这批点与前景点同 (u,v)、不同深度：原地不动时被近点挡住，
     横移 0.3m 后正好补住露出来的背景，而不是露出黑洞。 */
  function secondLayer(zf, side, W, H, opt) {
    opt = opt || {};
    const kMax = opt.k === undefined ? 8 : opt.k;
    const N = W * H;
    const INF = 1e9;
    const d = new Float32Array(N);
    for (let i = 0; i < N; i++) d[i] = side[i] === 2 ? 0 : INF;
    // 3-4 chamfer 两遍扫描
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        let v = d[i];
        if (x > 0) v = Math.min(v, d[i - 1] + 1);
        if (y > 0) {
          v = Math.min(v, d[i - W] + 1);
          if (x > 0) v = Math.min(v, d[i - W - 1] + 1.4);
          if (x < W - 1) v = Math.min(v, d[i - W + 1] + 1.4);
        }
        d[i] = v;
      }
    }
    for (let y = H - 1; y >= 0; y--) {
      for (let x = W - 1; x >= 0; x--) {
        const i = y * W + x;
        let v = d[i];
        if (x < W - 1) v = Math.min(v, d[i + 1] + 1);
        if (y < H - 1) {
          v = Math.min(v, d[i + W] + 1);
          if (x < W - 1) v = Math.min(v, d[i + W + 1] + 1.4);
          if (x > 0) v = Math.min(v, d[i + W - 1] + 1.4);
        }
        d[i] = v;
      }
    }
    const idx = [];
    const zs = [];
    for (let i = 0; i < N; i++) {
      if (side[i] !== 1) continue;
      if (!(zf[i] > 0)) continue;
      if (d[i] > kMax) continue;
      idx.push(i);
      // 离边界越远，深度额外退一点：形成斜裙而不是一层平板
      zs.push(zf[i] * (1 + 0.02 * Math.min(d[i], kMax) / kMax));
    }
    return { idx: Int32Array.from(idx), z: Float32Array.from(zs), count: idx.length };
  }

  /* 一步到位：调用方给米制深度 + 灰度 + 几何掩码，拿回新深度与双层点 */
  function process(input) {
    const { depth, lum, alpha, W, H } = input;
    const opt = input.opt || {};
    const eb = edgeBand(depth, lum, alpha, W, H, opt);
    const rc = recut(depth, lum, eb.band, eb.gx, eb.gy, W, H, opt);
    const l2 = secondLayer(rc.zf, rc.side, W, H, opt);
    return {
      depth: rc.depth, side: rc.side, zn: rc.zn, zf: rc.zf,
      layer2: l2,
      stats: Object.assign({}, rc.stats, { layer2: l2.count })
    };
  }

  global.PW = global.PW || {};
  global.PW.depthPost = { process, edgeBand, recut, secondLayer, hash01 };
})(typeof window !== 'undefined' ? window : globalThis);
