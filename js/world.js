/* 世界构建：由照片参数生成 天穹 / 远山 / 近岭 / 地形 / 岩石 / 植被 / 粒子 */
(function (global) {
  'use strict';

  const { clamp, lerp, smoothstep, mulberry32 } = global.PW.util;
  const { panoU } = global.PW.reconstruct;

  const NCOL = 360;
  const lin = (v) => Math.pow(clamp(v, 0, 1), 2.2);
  const lin3 = (c) => [lin(c[0]), lin(c[1]), lin(c[2])];

  function sampleHeights(analysis, u) {
    const f = clamp(u, 0, 0.9999) * NCOL;
    const i0 = Math.floor(f), i1 = (i0 + 1) % NCOL;
    return lerp(analysis.heights[i0], analysis.heights[i1], f - i0);
  }
  function sampleTop(analysis, u) {
    const f = clamp(u, 0, 0.9999) * NCOL;
    const i0 = Math.floor(f), i1 = (i0 + 1) % NCOL;
    return lerp(analysis.top[i0], analysis.top[i1], f - i0);
  }

  /* ---------------- 地形高度场 ---------------- */
  function makeHeightField(analysis, scale) {
    const amp = clamp(1.25 - analysis.brightness * 0.4, 0.55, 1.15) * (scale === undefined ? 1 : scale);
    return function heightAt(x, z) {
      const r = Math.sqrt(x * x + z * z);
      let h = 0;
      h += 0.95 * Math.sin(x * 0.031 + 1.3) * Math.cos(z * 0.027 - 0.7);
      h += 0.55 * Math.sin(x * 0.077 - 2.1) * Math.sin(z * 0.064 + 1.1);
      h += 0.26 * Math.sin(x * 0.19 + z * 0.13) * Math.cos(z * 0.17 - x * 0.11);
      h *= amp;
      h += smoothstep(150, 330, r) * 1.8 * (scale === undefined ? 1 : scale);
      h *= smoothstep(0, 13, r) * 0.86 + 0.14;
      return h;
    };
  }

  /* ---------------- 多边形工具 ---------------- */
  function buildCurtain(analysis, opt) {
    const NC = opt.columns || 420;
    const M = opt.rows || 7;
    const rng = mulberry32(opt.seed || 1);
    const baseY = opt.baseY === undefined ? -1.5 : opt.baseY;
    const maxH = opt.maxHeight;

    // 每列 buf
    const colH = new Float32Array(NC + 1);
    const colV0 = new Float32Array(NC + 1);
    const colV1 = new Float32Array(NC + 1);
    const colR = new Float32Array(NC + 1);
    const noiseAmp = opt.noise === undefined ? 0 : opt.noise;
    const varAmp = opt.variation === undefined ? 0.16 : opt.variation;
    const winAz = opt.windowAz;
    const winHalf = opt.windowHalf || Math.PI;

    // 低频噪声，用于扰动近岭轮廓
    const N = 9;
    const lowfreq = new Float32Array(N);
    for (let i = 0; i < N; i++) lowfreq[i] = rng() * 2 - 1;
    const lowNoise = (a) => {
      const f = (a / (Math.PI * 2)) * N;
      const i0 = Math.floor(f) % N, i1 = (i0 + 1) % N, t = f - Math.floor(f);
      const s = t * t * (3 - 2 * t);
      return lerp(lowfreq[i0], lowfreq[i1], s);
    };

    for (let i = 0; i <= NC; i++) {
      const a = (i / NC) * Math.PI * 2;
      const u = panoU(a);
      if (opt.fixedHeight) {
        colH[i] = opt.fixedHeight;
        colV0[i] = 0.0;
        colV1[i] = opt.vBottom !== undefined ? opt.vBottom : Math.max(0.02, sampleTop(analysis, u) * 0.92);
        colR[i] = opt.radius || 700;
        continue;
      }
      let hn = sampleHeights(analysis, u);
      if (opt.heightScale) hn *= opt.heightScale;
      // 连续的角度调制：让 360° 上的每一段山形都不一样，抑制镜像平铺的重复感
      if (varAmp) {
        hn *= 1 + varAmp * (Math.sin(a * 1.0 + 0.7) * 0.7 + Math.sin(a * 2.3 + 2.1) * 0.3);
      }
      if (noiseAmp) hn += lowNoise(a + (opt.phase || 0)) * noiseAmp;
      // 局部山段窗口（近岭只用一段，避免出现一圈墙）
      if (winAz !== undefined) {
        const d = Math.abs((((a - winAz) % (Math.PI * 2)) + Math.PI * 3) % (Math.PI * 2) - Math.PI);
        hn *= 1 - smoothstep(winHalf * 0.45, winHalf, d);
      }
      hn = Math.max(opt.minHn || 0, hn);

      const topV = sampleTop(analysis, u);
      colH[i] = hn * maxH;
      colV0[i] = Math.max(opt.vTopFade === undefined ? topV : lerp(topV, analysis.horizon, opt.vTopFade), 0.12);
      colV1[i] = analysis.horizon - 0.004;
      colR[i] = (opt.radius || 190) + (noiseAmp ? lowNoise(a + 2.1 + (opt.phase || 0)) * 9 : 0);
    }

    const pos = [], nrm = [], uv = [], col = [], idx = [];
    const sunAz = analysis.sunAz;

    for (let i = 0; i <= NC; i++) {
      const a = (i / NC) * Math.PI * 2;
      const sa = Math.sin(a), ca = Math.cos(a);
      const uq = panoU(a);
      const h = colH[i], r = colR[i];

      // 轮廓斜率 -> 法线的竖直分量，让山体有受光面
      const hPrev = colH[(i - 1 + NC) % NC], hNext = colH[(i + 1) % NC];
      const slope = (hNext - hPrev) / (2 * (Math.PI * 2 / NC) * Math.max(r, 1));
      let nx = -sa, ny = clamp(slope * 1.4, -1.2, 1.2), nz = -ca;
      const nl = Math.hypot(nx, ny, nz) || 1;
      nx /= nl; ny /= nl; nz /= nl;

      for (let j = 0; j <= M; j++) {
        const t = j / M;                       // 0 = 顶, 1 = 底
        const y = lerp(h, baseY, t);
        pos.push(sa * r, y, ca * r);
        nrm.push(nx, ny, nz);
        uv.push(uq, lerp(colV0[i], colV1[i], t));

        // 顶点色：整体色调 + 顶部略亮（雪线）+ 山体受太阳侧控制
        let shade = opt.tint === undefined ? 1 : opt.tint;
        if (opt.flat) {
          shade *= 0.99 + 0.04 * (1 - t);
        } else {
          // 山体体积感：上亮下暗（大气透视），侧面受光
          shade *= 0.52 + 0.56 * t;   // 顶部压暗成剪影，山脚抬亮融入雾
          const facing = clamp(-(sa * Math.sin(sunAz) + ca * Math.cos(sunAz)) * 0.5 + 0.5, 0, 1);
          shade *= 0.88 + 0.24 * facing;
          shade *= 1 - clamp(Math.abs(slope) * 0.30, 0, 0.18);
        }
        let alpha = 1;
        if (opt.fadeTop) alpha *= 1 - smoothstep(1 - opt.fadeTop, 1.0, 1 - t);
        col.push(shade, shade, shade, alpha);
      }
    }

    for (let i = 0; i < NC; i++) {
      for (let j = 0; j < M; j++) {
        const a0 = i * (M + 1) + j;
        const a1 = a0 + 1;
        const b0 = (i + 1) * (M + 1) + j;
        const b1 = b0 + 1;
        idx.push(a0, b0, a1, a1, b0, b1);
      }
    }

    return {
      pos: new Float32Array(pos),
      normal: new Float32Array(nrm),
      uv: new Float32Array(uv),
      color: new Float32Array(col),
      index: idx
    };
  }

  /* ---------------- 地面 ---------------- */
  const HALF_H = 31 * Math.PI / 180;   // 假定照片水平半视角

  /* 世界坐标 -> 照片 UV：把照片前景按透视关系反投影到地面平面上 */
  function groundUV(analysis, x, z) {
    const d = Math.max(0.7, Math.sqrt(x * x + z * z));
    const theta = Math.atan2(x, -z);
    let q = 0.5 + theta / (2 * HALF_H);
    let m = q % 2;
    if (m < 0) m += 2;
    const u = m <= 1 ? m : 2 - m;
    // 距离 -> 画面行：近处取照片底部前景，远处收拢到地平线
    const v = analysis.horizon + 0.62 / (1 + d * 0.055);
    return [u, clamp(v, analysis.horizon + 0.012, 1)];
  }

  function buildGround(analysis, heightAt) {
    const RINGS = 120, SEGS = 168, R = 340;
    const pos = [], nrm = [], uv = [], col = [], idx = [];
    const eps = 0.7;

    for (let i = 0; i <= RINGS; i++) {
      const rr = Math.pow(i / RINGS, 1.75) * R;
      for (let s = 0; s <= SEGS; s++) {
        const a = (s / SEGS) * Math.PI * 2;
        const x = Math.sin(a) * rr, z = Math.cos(a) * rr;
        const y = heightAt(x, z);
        const dx = (heightAt(x + eps, z) - heightAt(x - eps, z)) / (2 * eps);
        const dz = (heightAt(x, z + eps) - heightAt(x, z - eps)) / (2 * eps);
        const nl = Math.hypot(-dx, 1, -dz);
        pos.push(x, y, z);
        nrm.push(-dx / nl, 1 / nl, -dz / nl);
        uv.push(x / 14, z / 14);
        const shade = 0.99 + smoothstep(60, 0, rr) * 0.05;
        col.push(shade, shade, shade, 1);
      }
    }
    for (let i = 0; i < RINGS; i++) {
      for (let s = 0; s < SEGS; s++) {
        const a0 = i * (SEGS + 1) + s;
        const a1 = a0 + 1;
        const b0 = (i + 1) * (SEGS + 1) + s;
        const b1 = b0 + 1;
        idx.push(a0, b0, a1, a1, b0, b1);
      }
    }
    return {
      pos: new Float32Array(pos), normal: new Float32Array(nrm),
      uv: new Float32Array(uv), color: new Float32Array(col), index: idx
    };
  }

  /* ---------------- 岩石 ---------------- */
  function rockRadius(x, y, z) {
    return 1
      + 0.30 * Math.sin(x * 3.1 + y * 2.3) * Math.cos(z * 2.7 - y * 1.9)
      + 0.16 * Math.sin(x * 7.3 - z * 6.1 + 1.2);
  }

  function pushTri(pos, nrm, col, v, c, center) {
    // v: 3 个顶点, c: 颜色
    const [p0, p1, p2] = v;
    const ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2];
    const vx = p2[0] - p0[0], vy = p2[1] - p0[1], vz = p2[2] - p0[2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    if (center) {
      const ox = (p0[0] + p1[0] + p2[0]) / 3 - center[0];
      const oy = (p0[1] + p1[1] + p2[1]) / 3 - center[1];
      const oz = (p0[2] + p1[2] + p2[2]) / 3 - center[2];
      if (nx * ox + ny * oy + nz * oz < 0) { nx = -nx; ny = -ny; nz = -nz; }
    }
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l; ny /= l; nz /= l;
    for (const p of v) {
      pos.push(p[0], p[1], p[2]);
      nrm.push(nx, ny, nz);
      col.push(c[0], c[1], c[2], 1);
    }
  }

  function buildRocks(analysis, heightAt, rng, count) {
    const pos = [], nrm = [], col = [];
    const pal = analysis.palette;

    // 基础八面体，细分一次
    let faces = [];
    const V = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
    const F = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]];
    for (const f of F) faces.push([V[f[0]], V[f[1]], V[f[2]]]);
    const sub = [];
    for (const [a, b, c] of faces) {
      const ab = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
      const bc = [(b[0] + c[0]) / 2, (b[1] + c[1]) / 2, (b[2] + c[2]) / 2];
      const ca = [(c[0] + a[0]) / 2, (c[1] + a[1]) / 2, (c[2] + a[2]) / 2];
      sub.push([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]);
    }

    for (let n = 0; n < count; n++) {
      const ang = rng() * Math.PI * 2;
      const rad = 9 + Math.pow(rng(), 0.65) * 108;
      const x = Math.sin(ang) * rad, z = Math.cos(ang) * rad;
      const y = heightAt(x, z);
      const sc = 0.35 + Math.pow(rng(), 2.1) * 2.6;
      const sy = sc * (0.55 + rng() * 0.5);
      const rot = rng() * Math.PI * 2;
      const cs = Math.cos(rot), sn = Math.sin(rot);
      const base = pal[Math.floor(rng() * pal.length)];
      const dark = 0.42 + rng() * 0.42;
      const tint = [
        lin(lerp(base[0], 0.5, 0.34) * dark),
        lin(lerp(base[1], 0.5, 0.34) * dark),
        lin(lerp(base[2], 0.5, 0.34) * dark)
      ];
      const tris = [];
      for (const [a, b, c] of sub) {
        const tp = [a, b, c].map(p => {
          const r = rockRadius(p[0], p[1], p[2]);
          let px = p[0] * r * sc, py = p[1] * r * sy, pz = p[2] * r * sc;
          const rx = px * cs - pz * sn;
          const rz = px * sn + pz * cs;
          return [x + rx, y + py + sy * 0.22, z + rz];
        });
        tris.push(tp);
      }
      for (const t of tris) pushTri(pos, nrm, col, t, tint, [x, y + sy * 0.22, z]);
    }
    return {
      pos: new Float32Array(pos), normal: new Float32Array(nrm),
      color: new Float32Array(col),
      uv: new Float32Array((pos.length / 3) * 2)
    };
  }

  /* ---------------- 植被 ---------------- */
  function makeGrassTexture(rng) {
    const S = 256;
    const cv = document.createElement('canvas');
    cv.width = cv.height = S;
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, S, S);
    const blades = 13;
    for (let i = 0; i < blades; i++) {
      const x0 = S * 0.5 + (rng() - 0.5) * S * 0.82;
      const h = S * (0.30 + rng() * 0.64);
      const w = 3 + rng() * 5.5;
      const bend = (rng() - 0.5) * S * 0.36;
      const tone = Math.floor(186 + rng() * 60);
      const g = ctx.createLinearGradient(0, S, 0, S - h);
      g.addColorStop(0, `rgba(${Math.floor(tone * 0.62)},${Math.floor(tone * 0.66)},${Math.floor(tone * 0.6)},0.99)`);
      g.addColorStop(0.5, `rgba(${tone},${tone},${tone},0.94)`);
      g.addColorStop(0.94, `rgba(${tone},${tone},${tone},0.55)`);
      g.addColorStop(1, `rgba(${tone},${tone},${tone},0)`);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.moveTo(x0 - w, S);
      ctx.quadraticCurveTo(x0 + bend * 0.30, S - h * 0.58, x0 + bend, S - h);
      ctx.quadraticCurveTo(x0 + bend * 0.30 + w * 0.9, S - h * 0.52, x0 + w, S);
      ctx.closePath();
      ctx.fill();
    }
    return cv;
  }

  function buildGrass(analysis, heightAt, rng, count, radius, palette, greenTint) {
    const pos = [], nrm = [], uv = [], col = [], idx = [];
    const sunAz = analysis.sunAz;
    for (let n = 0; n < count; n++) {
      const ang = rng() * Math.PI * 2;
      const rad = 2.0 + Math.pow(rng(), 1.45) * radius;
      const x = Math.sin(ang) * rad, z = Math.cos(ang) * rad;
      const y = heightAt(x, z);
      const h = 0.20 + rng() * 0.48;
      const w = h * (0.85 + rng() * 0.75);
      const base = palette[Math.floor(rng() * palette.length)];
      const vib = 0.55 + rng() * 0.6;
      const tint = (greenTint
        ? [lerp(base[0], 0.24, 0.45) * vib, lerp(base[1], 0.36, 0.30) * vib, lerp(base[2], 0.20, 0.52) * vib]
        : [lerp(base[0], 0.34, 0.42) * vib, lerp(base[1], 0.31, 0.42) * vib, lerp(base[2], 0.26, 0.45) * vib]
      ).map(lin);
      // 太阳方向带来的轻微明暗差异
      const shade = 0.75 + 0.35 * clamp(-(Math.sin(ang) * Math.sin(sunAz) + Math.cos(ang) * Math.cos(sunAz)) * 0.5 + 0.5, 0, 1);
      const planes = 3;
      for (let p = 0; p < planes; p++) {
        const a = (p / planes) * Math.PI + rng() * 0.5;
        const dx = Math.cos(a) * w * 0.5, dz = Math.sin(a) * w * 0.5;
        const start = pos.length / 3;
        const skew = (rng() - 0.5) * 0.18;
        pos.push(x - dx, y, z - dz);
        pos.push(x + dx, y, z + dz);
        pos.push(x + dx + skew, y + h, z + dz + skew);
        pos.push(x - dx + skew, y + h, z - dz + skew);
        const nx = Math.sin(a), nz = Math.cos(a);
        for (let k = 0; k < 4; k++) nrm.push(nx * 0.4, 1.0, nz * 0.4);
        uv.push(0, 0, 1, 0, 1, 1, 0, 1);
        for (let k = 0; k < 4; k++) col.push(tint[0] * shade, tint[1] * shade, tint[2] * shade, 1);
        idx.push(start, start + 1, start + 2, start, start + 2, start + 3);
      }
    }
    return {
      pos: new Float32Array(pos), normal: new Float32Array(nrm),
      uv: new Float32Array(uv), color: new Float32Array(col), index: idx
    };
  }

  /* ---------------- 粒子 ---------------- */
  function buildParticles(analysis, rng, count) {
    const pos = new Float32Array(count * 3);
    const size = new Float32Array(count);
    const col = new Float32Array(count * 4);
    const vel = new Float32Array(count * 3);
    const snowy = analysis.isSnow;
    const pal = analysis.palette;
    for (let i = 0; i < count; i++) {
      const ang = rng() * Math.PI * 2;
      const rad = Math.pow(rng(), 0.6) * 85;
      pos[i * 3] = Math.sin(ang) * rad;
      pos[i * 3 + 1] = 0.4 + rng() * 16;
      pos[i * 3 + 2] = Math.cos(ang) * rad;
      size[i] = snowy ? 2.2 + rng() * 3.4 : 1.0 + rng() * 2.6;
      const c = pal[Math.floor(rng() * pal.length)];
      const b = snowy ? 0.95 : 0.55 + rng() * 0.5;
      col[i * 4] = clamp(c[0] * 0.6 + b * 0.45, 0, 1);
      col[i * 4 + 1] = clamp(c[1] * 0.6 + b * 0.45, 0, 1);
      col[i * 4 + 2] = clamp(c[2] * 0.6 + b * 0.5, 0, 1);
      col[i * 4 + 3] = snowy ? 0.55 + rng() * 0.4 : 0.22 + rng() * 0.35;
      vel[i * 3] = (rng() - 0.5) * 0.5;
      vel[i * 3 + 1] = snowy ? -(0.35 + rng() * 0.7) : (rng() - 0.5) * 0.16;
      vel[i * 3 + 2] = (rng() - 0.5) * 0.5;
    }
    return { pos, size, col, vel, count, snowy };
  }

  /* ---------------- 天空贴图（重度模糊，去掉竖向拉伸条纹） ---------------- */
  function makeSkyTexture(analysis) {
    let minTop = 1;
    for (let i = 0; i < NCOL; i++) minTop = Math.min(minTop, analysis.top[i]);
    const bandH = clamp(minTop * 0.96, 0.03, 0.7);
    const W = 384, H = 128;
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d');
    ctx.drawImage(analysis.canvas, 0, 0, analysis.W, Math.max(2, analysis.H * bandH), 0, 0, W, H);
    // 逐级缩小再放大，得到均匀柔化
    let tmp = cv;
    for (const s2 of [48, 16, 6]) {
      const t = document.createElement('canvas');
      t.width = s2; t.height = Math.max(3, Math.round(s2 * H / W));
      const tc = t.getContext('2d');
      tc.imageSmoothingEnabled = true;
      tc.drawImage(tmp, 0, 0, t.width, t.height);
      tmp = t;
    }
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(tmp, 0, 0, W, H);
    return cv;
  }

  /* ---------------- 天空贴图（重度模糊，去掉竖向拉伸条纹） ---------------- */
  function makeSkyTexture(analysis) {
    let minTop = 1;
    for (let i = 0; i < NCOL; i++) minTop = Math.min(minTop, analysis.top[i]);
    const bandH = clamp(minTop * 0.96, 0.03, 0.7);
    const W = 384, H = 128;
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d');
    ctx.drawImage(analysis.canvas, 0, 0, analysis.W, Math.max(2, analysis.H * bandH), 0, 0, W, H);
    // 逐级缩小再放大，得到均匀柔化
    let tmp = cv;
    for (const s2 of [48, 16, 6]) {
      const t = document.createElement('canvas');
      t.width = s2; t.height = Math.max(3, Math.round(s2 * H / W));
      const tc = t.getContext('2d');
      tc.imageSmoothingEnabled = true;
      tc.drawImage(tmp, 0, 0, t.width, t.height);
      tmp = t;
    }
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(tmp, 0, 0, W, H);
    return cv;
  }

  /* ---------------- 程序化地面贴图（纯噪声，不铺照片） ---------------- */
  function makeProceduralGroundTexture(analysis, seed) {
    const rng = mulberry32(seed >>> 0);
    const S = 256;
    const cv = document.createElement('canvas');
    cv.width = cv.height = S;
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(S, S);
    const d = img.data;
    const g = analysis.ground;
    const base = [g[0] * 255, g[1] * 255, g[2] * 255];
    // 三层值噪声
    const octaves = [[8, 0.5], [24, 0.28], [64, 0.14]];
    const grids = octaves.map(([n]) => {
      const a = new Float32Array(n * n);
      for (let i = 0; i < a.length; i++) a[i] = rng();
      return a;
    });
    const sampleGrid = (grid, n, x, y) => {
      const fx = (x / S) * n, fy = (y / S) * n;
      const x0 = Math.floor(fx) % n, y0 = Math.floor(fy) % n;
      const x1 = (x0 + 1) % n, y1 = (y0 + 1) % n;
      let tx = fx - Math.floor(fx), ty = fy - Math.floor(fy);
      tx = tx * tx * (3 - 2 * tx); ty = ty * ty * (3 - 2 * ty);
      const a = grid[y0 * n + x0], b = grid[y0 * n + x1];
      const c = grid[y1 * n + x0], e = grid[y1 * n + x1];
      return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + e * tx) * ty;
    };
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        let n = 0;
        for (let o = 0; o < octaves.length; o++) {
          n += (sampleGrid(grids[o], octaves[o][0], x, y) - 0.5) * octaves[o][1] * 2;
        }
        const grain = (rng() - 0.5) * 0.05;
        const k = 1 + n * 0.55 + grain;
        const i = (y * S + x) * 4;
        d[i] = clamp(base[0] * k, 0, 255);
        d[i + 1] = clamp(base[1] * k, 0, 255);
        d[i + 2] = clamp(base[2] * k, 0, 255);
        d[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return cv;
  }

  /* ---------------- 地面贴图 ---------------- */
  function makeGroundTexture(analysis, seed) {
    const rng = mulberry32(seed >>> 0);
    const S = 512;
    const cv = document.createElement('canvas');
    cv.width = cv.height = S;
    const ctx = cv.getContext('2d');
    const src = analysis.canvas;
    const sy = Math.floor(analysis.H * 0.66);
    const sx = Math.floor(analysis.W * 0.18);
    const sw = Math.floor(analysis.W * 0.64);
    ctx.drawImage(src, sx, sy, sw, analysis.H - sy, 0, 0, S, S);

    // 轻度模糊：抹掉过度拉伸的条纹，但保留一点细节
    const tmp = document.createElement('canvas');
    tmp.width = 64; tmp.height = 64;
    tmp.getContext('2d').drawImage(cv, 0, 0, 64, 64);
    ctx.globalAlpha = 0.42;
    ctx.drawImage(tmp, 0, 0, S, S);
    ctx.globalAlpha = 1;

    // 细节噪点 + 向地面基色靠拢
    const img = ctx.getImageData(0, 0, S, S);
    const d = img.data;
    const g = analysis.ground;
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = (y * S + x) * 4;
        const n1 = Math.sin(x * 0.9 + y * 1.7) * 0.5 + Math.sin(x * 3.1 - y * 2.3) * 0.5;
        const n2 = (rng() - 0.5) * 2;
        const grain = (n1 * 6 + n2 * 9);
        for (let c = 0; c < 3; c++) {
          let v = d[i + c];
          v = lerp(v, g[c] * 255, 0.18);
          v += grain;
          d[i + c] = clamp(v, 0, 255);
        }
      }
    }
    ctx.putImageData(img, 0, 0);
    return cv;
  }

  /* ---------------- 深度重建：照片 -> 有起伏的三维表面 ----------------
   * 每个像素按其估计深度放到三维空间里（透过窗口看到的那一层表面）：
   *   - 地平线以上：距离由深度决定，山体/树木获得真实的前后关系与视差
   *   - 地平线以下：按"平地 + 眼高"反解距离，得到透视正确的地面（不是平铺贴图）
   *   - 360° 由镜像平铺补全，接缝处融合深度，避免出现垂直断崖
   */
  const HFOV = (62 * Math.PI) / 180;

  function buildRelief(analysis, depth, opt) {
    opt = opt || {};
    const aspect = analysis.W / Math.max(1, analysis.H);
    const vfov = 2 * Math.atan(Math.tan(HFOV / 2) / aspect);
    const eye = 1.7;
    const FAR_R = 128;   // 远景固定圆柱半径
    const NC = opt.columns || 360;
    const NR = opt.rows || 200;
    const dsz = depth.size;
    const D = depth.data;

    /* 深度分布拉伸 */
    const hist = [];
    for (let y = 0; y < dsz; y += 3) for (let x = 0; x < dsz; x += 3) hist.push(D[y * dsz + x]);
    hist.sort((a, b) => a - b);
    const pLo = hist[Math.floor(hist.length * 0.02)] || 0;
    const pHi = hist[Math.floor(hist.length * 0.98)] || 1;
    const span = Math.max(1e-4, pHi - pLo);

    const rawAt = (u, v) => {
      const x = clamp(u, 0, 1) * (dsz - 1);
      const y = clamp(v, 0, 1) * (dsz - 1);
      const x0 = Math.floor(x), y0 = Math.floor(y);
      const x1 = Math.min(dsz - 1, x0 + 1), y1 = Math.min(dsz - 1, y0 + 1);
      const fx = x - x0, fy = y - y0;
      const a = D[y0 * dsz + x0], b = D[y0 * dsz + x1];
      const c = D[y1 * dsz + x0], d = D[y1 * dsz + x1];
      return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
    };
    const dn = (u, v) => clamp((rawAt(u, v) - pLo) / span, 0, 1);

    // 镜像接缝融合深度
    const seam = 0.08;
    const depthAt = (u, v) => {
      let d = dn(u, v);
      if (u < seam) d = lerp(d, dn(1 - u, v), 0.5 * (1 - u / seam));
      else if (u > 1 - seam) d = lerp(d, dn(1 - u, v), 0.5 * ((u - (1 - seam)) / seam));
      return d;
    };

    // 深度 -> 沿射线距离（近 2.5m，远 ~260m）
    const distOf = (d) => clamp(2.5 + 250 * Math.pow(1 - clamp(d, 0, 1), 1.7), 2.6, 280);

    /* 天空边缘：逐列地形上缘，用来在天空处渐隐 */
    const topAt = (u) => {
      const f = clamp(u, 0, 0.9999) * NCOL;
      const i0 = Math.floor(f), i1 = (i0 + 1) % NCOL;
      return lerp(analysis.top[i0], analysis.top[i1], f - i0);
    };

    /* 1) 用真正的射线参数化：每个顶点 = 方向 × 距离（世界坐标，固定不动） */
    const W1 = NC + 1;
    const n4 = W1 * (NR + 1);
    const gx = new Float32Array(n4), gy = new Float32Array(n4), gz = new Float32Array(n4);
    const gT = new Float32Array(n4);
    const gV = new Float32Array(NR + 1);
    const gU = new Float32Array(W1);
    const gSky = new Float32Array(W1);

    // 视锥掩码：照片只决定"初始视角能看到的东西"。
    // 视锥外不再造几何（不镜像、不外推），交给背景渐变 + 雾，避免拉伸与尖刺。
    const halfFov = HFOV * 0.5 * 1.06;
    const gVis = new Float32Array(W1);
    for (let i = 0; i <= NC; i++) {
      const a = (i / NC) * Math.PI * 2;
      gU[i] = panoU(a);
      gSky[i] = topAt(gU[i]);
      // a=π 是照片中心
      let d = a - Math.PI;
      while (d > Math.PI) d -= Math.PI * 2;
      while (d < -Math.PI) d += Math.PI * 2;
      const ad = Math.abs(d);
      // 边缘 12% 内渐隐，避免出现硬切边（像一张立起来的卡片）
      gVis[i] = ad <= halfFov * 0.88 ? 1 : (ad >= halfFov ? 0 : (halfFov - ad) / (halfFov * 0.12));
    }
    for (let j = 0; j <= NR; j++) gV[j] = j / NR;

    for (let j = 0; j <= NR; j++) {
      const v = gV[j];
      const phi = (0.5 - v) * vfov;
      const sp = Math.sin(phi), cp = Math.cos(phi);
      const tGround = sp < -0.002 ? eye / (-sp) : 1e9;
      const w = smoothstep(-0.03, 0.02, sp);      // 0=地面, 1=深度
      for (let i = 0; i <= NC; i++) {
        const a = (i / NC) * Math.PI * 2;
        const u = gU[i];
        const tDepth = distOf(depthAt(u, clamp(v, 0, 1)));
        let t;
        if (sp < -0.002) {
          // 地平线以下：严格落在 y=0 平面上（走起来才正确），不参与远景收敛
          t = tGround;
        } else {
          // 地平线以上：远景收成刚性圆柱，避免离开视点后剪切成长条
          t = tDepth;
          t = lerp(t, FAR_R, smoothstep(18, 46, t));
        }
        t = clamp(t, 2.4, 420);
        const k = j * W1 + i;
        // 垂直位移钳制：远处 y 不得超过合理山高，避免深度噪点拉出尖刺
        const yMax = 180 + 60 * (1 - Math.abs(sp));
        gy[k] = clamp(sp * t + eye, -3.0, yMax);
        gT[k] = t;
        gx[k] = Math.sin(a) * cp * t;
        gz[k] = Math.cos(a) * cp * t;
      }
    }

    /* 2) 前拉约束：相邻格距离不允许突变（深度断崖 -> 陡坡而不是"鳍"） */
    for (let pass = 0; pass < 3; pass++) {
      const src = gT.slice();
      for (let j = 0; j <= NR; j++) {
        for (let i = 0; i <= NC; i++) {
          const k = j * W1 + i;
          const iL = j * W1 + ((i - 1 + W1) % W1);
          const iR = j * W1 + ((i + 1) % W1);
          const jD = Math.max(0, j - 1) * W1 + i;
          const jU2 = Math.min(NR, j + 1) * W1 + i;
          const mn = Math.min(src[iL], src[iR], src[jD], src[jU2]);
          if (src[k] > mn * 1.25) {
            // 深度断崖：直接吸附到前景深度（标准 DIBR 补洞），
            // 桥接面片因此贴在物体前面被遮住，而不是横拉成条纹
            const t = mn;
            gT[k] = t;
            const a = (i / NC) * Math.PI * 2;
            const phi = (0.5 - gV[j]) * vfov;
            gx[k] = Math.sin(a) * Math.cos(phi) * t;
            gz[k] = Math.cos(a) * Math.cos(phi) * t;
            gy[k] = Math.sin(phi) * t + eye;
          }
        }
      }
    }

    /* 3) 顶点 / UV / 法线 / 顶点色 */
    const pos = [], uv = [], col = [], nrm = [], idx = [];
    for (let j = 0; j <= NR; j++) {
      for (let i = 0; i <= NC; i++) {
        const k = j * W1 + i;
        pos.push(gx[k], gy[k], gz[k]);
        uv.push(gU[i], gV[j]);
      }
    }

    const sunAz = analysis.sunAz, sunEl = analysis.sunElev;
    const sunV = [Math.sin(sunAz) * Math.cos(sunEl), Math.sin(sunEl), Math.cos(sunAz) * Math.cos(sunEl)];
    for (let j = 0; j <= NR; j++) {
      const v = gV[j];
      for (let i = 0; i <= NC; i++) {
        const k = j * W1 + i;
        const iL = j * W1 + ((i - 1 + W1) % W1);
        const iR = j * W1 + ((i + 1) % W1);
        const jD = Math.max(0, j - 1) * W1 + i;
        const jU2 = Math.min(NR, j + 1) * W1 + i;
        const ax = gx[iR] - gx[iL], ay = gy[iR] - gy[iL], az = gz[iR] - gz[iL];
        const bx = gx[jU2] - gx[jD], by = gy[jU2] - gy[jD], bz = gz[jU2] - gz[jD];
        let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
        const nl = Math.hypot(nx, ny, nz) || 1;
        nx /= nl; ny /= nl; nz /= nl;
        if (nx * gx[k] + nz * gz[k] > 0) { nx = -nx; ny = -ny; nz = -nz; }
        const ndl = Math.max(0, nx * sunV[0] + ny * sunV[1] + nz * sunV[2]);
        // 体积感来自法线：越背离太阳越暗，但要保留照片本身的明暗
        const shade = clamp(0.80 + 0.34 * ndl + 0.06 * ny, 0.55, 1.16);
        // 天空处渐隐，交给程序化天穹
        const skyV = gSky[i];
        // 硬边剪影：只在天际线"上方"渐隐，下方必须实心，否则会出现亮缝
        const aSky = smoothstep(skyV - 0.012, skyV, v);
        // 画面下缘几行渐隐，交给程序化地面兜底（避免近处被放大成黑带）
        const aBot = 1 - smoothstep(0.985, 1.0, v);
        const alpha = Math.min(aSky, aBot) * gVis[i];
        col.push(shade, shade, shade, alpha);
        nrm.push(nx, ny, nz);
      }
    }

    /* 4) 索引：跳过完全在天空里的格子 */
    for (let j = 0; j < NR; j++) {
      const v0 = gV[j], v1 = gV[j + 1];
      for (let i = 0; i < NC; i++) {
        if (gVis[i] < 0.5 || gVis[(i + 1) % W1] < 0.5) continue;   // 视锥外不建面
        const s0 = gSky[i], s1 = gSky[(i + 1) % W1];
        if (v0 < Math.min(s0, s1) - 0.05 && v1 < Math.min(s0, s1) - 0.05) continue;
        const a0 = j * W1 + i;
        const a1 = a0 + 1;
        const b0 = (j + 1) * W1 + i;
        const b1 = b0 + 1;
        idx.push(a0, b0, a1, a1, b0, b1);
      }
    }

    return {
      pos: new Float32Array(pos),
      normal: new Float32Array(nrm),
      uv: new Float32Array(uv),
      color: new Float32Array(col),
      index: idx,
      vfov,
      tris: idx.length / 3
    };
  }

  /* 程序化地面：只在重建表面没有覆盖到的地方兜底（脚下、视锥外），不用照片平铺 */
  function buildFallbackGround(analysis, rng) {
    const RINGS = 40, SEGS = 96, R = 90;
    const pos = [], nrm = [], uv = [], col = [], idx = [];
    const g = analysis.ground;
    for (let i = 0; i <= RINGS; i++) {
      const rr = Math.pow(i / RINGS, 1.5) * R;
      for (let s = 0; s <= SEGS; s++) {
        const a = (s / SEGS) * Math.PI * 2;
        const x = Math.sin(a) * rr, z = Math.cos(a) * rr;
        pos.push(x, -0.15, z);
        nrm.push(0, 1, 0);
        uv.push(x / 9, z / 9);
        col.push(1, 1, 1, 1);
      }
    }
    for (let i = 0; i < RINGS; i++) {
      for (let s = 0; s < SEGS; s++) {
        const a0 = i * (SEGS + 1) + s, a1 = a0 + 1;
        const b0 = (i + 1) * (SEGS + 1) + s, b1 = b0 + 1;
        idx.push(a0, b0, a1, a1, b0, b1);
      }
    }
    return {
      pos: new Float32Array(pos), normal: new Float32Array(nrm),
      uv: new Float32Array(uv), color: new Float32Array(col), index: idx
    };
  }

  /* ---------------- 组装世界 ---------------- */
  function build(gl, analysis, opts) {
    opts = opts || {};
    const seed = opts.seed || 1;
    const rng = mulberry32(seed);
    const depthMap = opts.depth || null;
    const heightAt = makeHeightField(analysis, depthMap ? 0.12 : 1);
    const PWGL = global.PW.GL;

    const relief = depthMap ? buildRelief(analysis, depthMap, { seed }) : null;

    const farRidge = relief ? null : buildCurtain(analysis, {
      columns: 460, rows: 8, radius: 195, maxHeight: 56,
      seed: seed + 11, tint: 1.0, baseY: -3, noise: 0.04, variation: 0.10, minHn: 0.10
    });
    const nearRidge = relief ? null : buildCurtain(analysis, {
      columns: 380, rows: 6, radius: 118, maxHeight: 56, heightScale: 0.22,
      seed: seed + 29, tint: 0.88, baseY: -7, noise: 0.12, phase: 0.9,
      vTopFade: 0.52, variation: 0.25,
      windowAz: analysis.sunAz + 1.15, windowHalf: 0.85
    });
    // 天穹只取"绝对天空"那段：照片里最高的地形之上
    let skyBottom = 1;
    for (let i = 0; i < NCOL; i++) skyBottom = Math.min(skyBottom, analysis.top[i]);
    skyBottom = clamp(skyBottom - 0.01, 0.015, 0.6);

    const skyCurtain = relief ? null : buildCurtain(analysis, {
      columns: 420, rows: 6, radius: 760, maxHeight: 700,
      seed: seed + 5, tint: 1.0, baseY: 0, fadeTop: 0.62,
      fixedHeight: 700, flat: true
    });

    const ground = relief ? buildFallbackGround(analysis, rng) : buildGround(analysis, heightAt);
    // 深度重建模式下，照片表面本身已经是有纵深的细节，不再叠加程序化植被
    const rocks = relief ? null : buildRocks(analysis, heightAt, rng, analysis.isSnow ? 70 : 120);
    const grass = (!relief && analysis.isGreen)
      ? buildGrass(analysis, heightAt, rng, 5200, 105, analysis.palette, true) : null;
    const shrub = (!relief && !analysis.isGreen && analysis.brightness > 0.42)
      ? buildGrass(analysis, heightAt, rng, 520, 100, analysis.palette, false) : null;
    const particles = buildParticles(analysis, rng, analysis.isSnow ? 1100 : 700);

    const tex = {
      photo: PWGL.createTexture(gl, analysis.canvas, { wrap: 'clamp', clampV: true, aniso: 8, flipY: false }),
      ground: PWGL.createTexture(gl, relief ? makeProceduralGroundTexture(analysis, seed + 3) : makeGroundTexture(analysis, seed + 3), { wrap: 'mirror', aniso: 8 }),
      sky: PWGL.createTexture(gl, makeSkyTexture(analysis), { wrap: 'clamp', clampV: true, flipY: false }),
      grass: PWGL.createTexture(gl, makeGrassTexture(mulberry32(seed + 77)), { wrap: 'clamp', clampV: true })
    };

    const m = {
      relief: relief ? PWGL.createMesh(gl, relief) : null,
      farRidge: farRidge ? PWGL.createMesh(gl, farRidge) : null,
      nearRidge: nearRidge ? PWGL.createMesh(gl, nearRidge) : null,
      sky: skyCurtain ? PWGL.createMesh(gl, skyCurtain) : null,
      ground: ground ? PWGL.createMesh(gl, ground) : null,
      rocks: rocks ? PWGL.createMesh(gl, rocks) : null,
      grass: grass ? PWGL.createMesh(gl, grass) : null,
      shrub: shrub ? PWGL.createMesh(gl, shrub) : null
    };

    const worldInfo = {
      mode: relief ? 'depth' : 'legacy',
      eyeHeight: 1.7,
      walkRadius: relief ? 12 : 116
    };

    /* 光照与雾 */
    const sunDir = [
      Math.sin(analysis.sunAz) * Math.cos(analysis.sunElev),
      Math.sin(analysis.sunElev),
      Math.cos(analysis.sunAz) * Math.cos(analysis.sunElev)
    ];
    const sunPower = clamp((analysis.sunLum - 0.35) / 0.55, 0.12, 1.0);
    const warm = clamp(0.45 + (analysis.skyHorizon[0] - analysis.skyHorizon[2]) * 1.6, 0.3, 1.0);

    const fogS = analysis.fog;
    const fog = fogS;
    const ambient = [
      lerp(fog[0], analysis.skyTop[0], 0.45) * 1.06 + 0.04,
      lerp(fog[1], analysis.skyTop[1], 0.45) * 1.06 + 0.04,
      lerp(fog[2], analysis.skyTop[2], 0.45) * 1.06 + 0.05
    ];
    const sunColor = [
      (0.90 + warm * 0.35) * (0.45 + sunPower * 0.62),
      (0.84 + warm * 0.20) * (0.45 + sunPower * 0.58),
      (0.74 + (1 - warm) * 0.28) * (0.45 + sunPower * 0.52)
    ];

    /* 曝光反解：让重建画面的平均亮度对齐原片 */
    const aces = (x) => Math.min(1, Math.max(0, (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)));
    const ambientL = lin3(ambient);
    const sunL = lin3(sunColor).map((v) => v * 1.30);
    const fogL = lin3(fogS);
    const skyTopL = lin3(analysis.skyTop);
    const skyHorL = lin3(analysis.skyHorizon);

    const lightMean = 0.42 * ((ambientL[0] + ambientL[1] + ambientL[2]) / 3)
                    + 0.50 * ((sunL[0] + sunL[1] + sunL[2]) / 3) * 0.72;
    const albedoMean = lin(clamp(analysis.meanLum || analysis.brightness, 0.06, 0.95));
    const targetLum = Math.pow(clamp((analysis.meanLum || analysis.brightness) * 1.02, 0.08, 0.95), 2.2);
    let lo = 0.05, hi = 3.0;
    for (let i = 0; i < 26; i++) {
      const mid = (lo + hi) / 2;
      if (aces(albedoMean * lightMean * mid) < targetLum) lo = mid; else hi = mid;
    }
    const exposure = clamp((lo + hi) / 2 * 0.52, 0.08, 2.4);

    const world = {
      name: analysis.name,
      analysis,
      heightAt,
      sunDir,
      fogColor: fogL,
      skyTop: skyTopL,
      skyHorizon: skyHorL,
      ambient: ambientL,
      sunColor: sunL,
      sunPower,
      exposure,
      targetLum: clamp(analysis.meanLum || analysis.brightness || 0.4, 0.06, 0.9),
      groundTint: lin3(analysis.ground).map((v) => v * 0.85),
      lights: { sunDir, sunColor: sunL, ambient: ambientL, fogColor: fogL, fogDensity: 0.0040 },
      meshes: m,
      tex,
      particles,
      seed,
      spawn: { x: 0, z: 0 },
      mode: worldInfo.mode,
      eyeHeight: worldInfo.eyeHeight,
      walkRadius: worldInfo.walkRadius,
      reliefMeta: relief ? { vfov: relief.vfov } : null
    };

    return world;
  }

  global.PW.world = { build, makeHeightField, panoU };
})(window);
