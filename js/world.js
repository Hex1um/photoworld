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

  /* ---------------- 天空贴图（重度模糊，去掉竖向拉伸条纹） ----------------
     注意：这里曾有两份逐字相同的函数声明，后者静默覆盖前者。保留唯一一份，
     tools/test_regression.mjs 会断言定义次数，防止复制粘贴再次引入。 */
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

  /* ---------------- 低频「世界代理」：不可见区域的补全层 ----------------
   * 读 github.com/topics/scene-generation 与 google/three-live 后的共识：
   * VistaDream 的 coarse 3D scaffold、HunyuanWorld 的 panorama world proxy、
   * WorldGen 的 background inpainting、3d-photo-inpainting 的 LDI 补洞，
   * 都在解决同一件事 —— 单张照片只覆盖视锥内，物体背后/视锥外必须有东西。
   * 纯前端零依赖做不到 diffusion inpainting，这里用「多级降采样 + 边缘混环境色」
   * 得到一张没有物体细节的低频色带，作为零依赖的 coarse scaffold 替身：
   * 断崖跳面留下的洞、以及转身后视锥外的区域，看到的都是它而不是空白雾。 */
  function makeWorldBackdropTexture(analysis) {
    const W = 512, H = 256;
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d');
    ctx.drawImage(analysis.canvas, 0, 0, analysis.W, analysis.H, 0, 0, W, H);
    let tmp = cv;
    for (const s of [64, 24, 10]) {
      const t = document.createElement('canvas');
      t.width = s; t.height = Math.max(4, Math.round(s * H / W));
      const tc = t.getContext('2d');
      tc.imageSmoothingEnabled = true;
      tc.drawImage(tmp, 0, 0, t.width, t.height);
      tmp = t;
    }
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(tmp, 0, 0, W, H);
    // 顶/底缘向天光 / 雾色靠拢，避免与程序化天穹、地面之间出现硬横切线
    const img = ctx.getImageData(0, 0, W, H);
    const d = img.data;
    const fog = analysis.fog, sky = analysis.skyTop;
    for (let y = 0; y < H; y++) {
      const v = y / (H - 1);
      const kTop = Math.pow(1 - clamp(v / 0.30, 0, 1), 2) * 0.55;
      const kBot = Math.pow(clamp((v - 0.70) / 0.30, 0, 1), 2) * 0.72;
      const k = Math.max(kTop, kBot);
      const tgt = kBot > kTop ? fog : sky;
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        for (let c = 0; c < 3; c++) d[i + c] = lerp(d[i + c], tgt[c] * 255, k);
      }
    }
    ctx.putImageData(img, 0, 0);
    return cv;
  }

  /* 360° 连续低频远景幕：只提供「照片没覆盖到的地方」的连续背景，不参与近景视差；
     渲染时跟随相机平移（与 GLB 资产模式 backdrop 层同一策略）。
     R 取 780m（> 所有近景几何的 700m 上限，< 透视 far 900m），深度测试保证
     它永远在近景之后；高度按逐列天际线起伏，不是一堵等高的墙。 */
  function buildWorldBackdrop(analysis, opt) {
    opt = opt || {};
    const SEG = 180, ROWS = 5, R = opt.radius || 780;
    const pos = [], uv = [], col = [], nrm = [], idx = [];
    const hor = analysis.horizon;
    for (let i = 0; i <= SEG; i++) {
      const a = (i / SEG) * Math.PI * 2;
      const u = panoU(a);
      const topV = sampleTop(analysis, u);
      const h = clamp((hor - topV) * 220 + 24, 18, 260);
      const sa = Math.sin(a), ca = Math.cos(a);
      for (let j = 0; j <= ROWS; j++) {
        const t = j / ROWS;
        pos.push(sa * R, lerp(h, -8, t), ca * R);
        nrm.push(-sa, 0.12, -ca);
        uv.push(u, lerp(clamp(topV * 0.72, 0, 1), clamp(hor + 0.14, 0, 1), t));
        const shade = 0.88 + 0.12 * t;
        col.push(shade, shade, shade, 1);
      }
    }
    for (let i = 0; i < SEG; i++) {
      for (let j = 0; j < ROWS; j++) {
        const a0 = i * (ROWS + 1) + j, a1 = a0 + 1;
        const b0 = (i + 1) * (ROWS + 1) + j, b1 = b0 + 1;
        idx.push(a0, b0, a1, a1, b0, b1);
      }
    }
    return {
      pos: new Float32Array(pos), normal: new Float32Array(nrm),
      uv: new Float32Array(uv), color: new Float32Array(col), index: idx
    };
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
  /* v1.11.0：解决「整张照片被压成一张平铺贴片」的三处结构性根因
   *   1) 旧 distOf 是经验曲线，且 46m 外一律 lerp 到 128m 固定圆柱 —— 大部分
   *      可见像素落在同一个半径上，纵深被抹平，看起来就是贴在背景上的照片；
   *   2) 地平线以下一律 gy=0、t 按平地反解，深度图完全不参与 —— 半张照片
   *      是一张铺在地上的平面贴图；
   *   3) 地面 heightAt 在深度模式被乘 0.12，兜底地面又是常量平面。
   * 改法（思路取自 GitHub 上的同类方案，见 README「参考方案」）：
   *   - 双锚点物理标定（仓库 pipeline/geom.py 同一口径）替代经验曲线；
   *   - 超过 BACKDROP_T 的四边形拆成跟随相机的远景层（GLB 资产模式已验证
   *     这是消掉「沙盘感」的关键），近景保留真实视差；
   *   - 地面用「相对同行基线更近的深度」挤出近景物体，脚下渐隐、幅度钳制；
   *   - 竖直夸张 + 法线/AO 增强，让起伏真的看得出来（DepthWizard / PBRCELAIN
   *     里的 Z 夸张与局部均衡思路）。
   */
  const HORIZON_FAR = 520;       // 地平线远端锚点（米）
  const BACKDROP_T = 160;        // 超过该距离归入无视差远景层（米）
  const GROUND_LIFT_MAX = 1.4;   // 近景挤出最大高度（米）
  const VERT_GAIN = 1.10;        // 竖直夸张：补偿单目深度对中景的压缩

  /* 两锚点标定：pipeline/geom.py::calibrate_scale / metric_depth 的浏览器等价实现。
     d 是归一化视差（1 = 最近）。近端锚点来自地面平面几何 z=eye/(-sinφ)，
     远端锚点是声明式先验 HORIZON_FAR；解 z = A·(d+r0)^(-p)。
     标定失败时返回 null，调用方回退到旧经验曲线。 */
  function calibrateScale(depthAt, gV, vfov, eye) {
    const nearVals = [], farVals = [];
    let zSum = 0, zN = 0;
    for (let j = 0; j < gV.length; j++) {
      const v = gV[j];
      const sp = Math.sin((0.5 - v) * vfov);
      for (let i = 0; i <= 24; i++) {
        const d = depthAt(i / 24, clamp(v, 0, 1));
        if (!Number.isFinite(d)) continue;
        if (v >= 0.90 && v <= 0.98 && sp < -0.10) {
          nearVals.push(d); zSum += eye / (-sp); zN++;
        }
        if (v >= 0.42 && v <= 0.52) farVals.push(d);
      }
    }
    const q = (a, f) => {
      if (!a.length) return 0;
      a.sort((x, y) => x - y);
      return a[Math.min(a.length - 1, Math.floor(a.length * f))];
    };
    const rNear = Math.max(0.08, q(nearVals, 0.5));
    const rFar = Math.min(rNear * 0.85, Math.max(0.005, q(farVals, 0.20)));
    const zNear = zN ? Math.max(2.0, zSum / zN) : eye / 0.30;
    if (!nearVals.length || !farVals.length || rNear <= rFar * 1.02) return null;
    let p = clamp(Math.log(HORIZON_FAR / zNear) / Math.log(rNear / rFar), 0.6, 3.2);
    const A = zNear * Math.pow(rNear, p);
    const r0 = Math.pow(A / HORIZON_FAR, 1 / p);
    const Aeff = A * Math.pow(1 + r0 / rNear, p);
    const metric = (d) => clamp(Aeff * Math.pow(Math.max(0, d) + r0, -p), 0.6, HORIZON_FAR * 1.2);
    return { p, r0, Aeff, zNear, rNear, rFar, metric };
  }

  function buildRelief(analysis, depth, opt) {
    opt = opt || {};
    const aspect = analysis.W / Math.max(1, analysis.H);
    const vfov = 2 * Math.atan(Math.tan(HFOV / 2) / aspect);
    const eye = 1.7;
    const NC = opt.columns || 360;
    const NR = opt.rows || 200;
    // 深度图与照片同比例（见 js/depth.js 的等比 padding + 裁回），不再是方形拉伸图
    const dw = depth.width || depth.size;
    const dh = depth.height || depth.size;
    const D = depth.data;

    /* 深度分布拉伸 */
    const hist = [];
    for (let y = 0; y < dh; y += 3) for (let x = 0; x < dw; x += 3) hist.push(D[y * dw + x]);
    hist.sort((a, b) => a - b);
    const pLo = hist[Math.floor(hist.length * 0.02)] || 0;
    const pHi = hist[Math.floor(hist.length * 0.98)] || 1;
    const span = Math.max(1e-4, pHi - pLo);

    const rawAt = (u, v) => {
      const x = clamp(u, 0, 1) * (dw - 1);
      const y = clamp(v, 0, 1) * (dh - 1);
      const x0 = Math.floor(x), y0 = Math.floor(y);
      const x1 = Math.min(dw - 1, x0 + 1), y1 = Math.min(dh - 1, y0 + 1);
      const fx = x - x0, fy = y - y0;
      const a = D[y0 * dw + x0], b = D[y0 * dw + x1];
      const c = D[y1 * dw + x0], d = D[y1 * dw + x1];
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

    // 深度 -> 沿射线距离：在 gV 填充完成后按双锚点标定结果定义（见下方），
    // 旧的 2.5 + 250·(1-d)^1.7 经验曲线仅作为 ?v1=1 / 标定退化时的回退。

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
      // 边缘 25% 内渐隐，配合顶点 aEdge 把照片纹理融进雾色：
      // 纯放宽渐隐会露出镜像拉伸条纹（实测 38% 时左右一大片灰弧），
      // 但融色把它们盖住后，边界就既没有硬切也没有条纹。
      gVis[i] = ad <= halfFov * 0.75 ? 1 : (ad >= halfFov ? 0 : (halfFov - ad) / (halfFov * 0.25));
    }
    for (let j = 0; j <= NR; j++) gV[j] = j / NR;

    /* 距离标定：默认双锚点物理标定；?v1=1 或标定退化时回退旧经验曲线。 */
    const cal = opt.legacy ? null : calibrateScale(depthAt, gV, vfov, eye);
    const legacyCurve = (d) => clamp(2.5 + 250 * Math.pow(1 - clamp(d, 0, 1), 1.7), 2.6, 280);
    const distOf = cal ? cal.metric : legacyCurve;

    /* 逐行地面基线 + 邻域平滑的「近景超出量」：
       单目深度对平坦地面沿行单调递增；比同排中位明显更近的像素就是物体/凸起。
       平滑是为了不让深度噪点把地面打成锯齿（PBRCELAIN 的 local equalization 思路）。 */
    const gD = new Float32Array(n4);
    const gE = new Float32Array(n4);
    const gEs = new Float32Array(n4);
    const rowBase = new Float32Array(NR + 1);
    for (let j = 0; j <= NR; j++) {
      for (let i = 0; i < W1; i++) gD[j * W1 + i] = depthAt(gU[i], clamp(gV[j], 0, 1));
      const sorted = Float32Array.from(gD.subarray(j * W1, (j + 1) * W1));
      sorted.sort();
      const base = sorted[Math.floor(W1 / 2)];
      rowBase[j] = base;
      for (let i = 0; i < W1; i++) gE[j * W1 + i] = gD[j * W1 + i] - base;
    }
    // 5×5 低通：Depth Anything 在平坦水面/地面上有逐行细噪声，
    // 若直接拿去挤地面会形成一圈圈「波纹条纹」，比平铺还难看。
    for (let j = 0; j <= NR; j++) {
      for (let i = 0; i < W1; i++) {
        let sum = 0, n = 0;
        for (let dj = -2; dj <= 2; dj++) {
          const jj = Math.min(NR, Math.max(0, j + dj));
          for (let di = -2; di <= 2; di++) {
            const ii = ((i + di) % W1 + W1) % W1;
            sum += gE[jj * W1 + ii]; n++;
          }
        }
        gEs[j * W1 + i] = sum / n;
      }
    }

    /* 近景挤出量先算成一张 (方位 × 屏行) 的场，再做行方向低通。
       为什么必须沿行抹平：等 v 线在透视里是一圈圈同心弧，任何逐行差异都会被
       照片纹理放大成弧形条纹（实测第一版挤出就是这样毁掉湖面的）。
       另外避开镜像接缝 u≈0/1，那里的深度是左右混合出来的，不能拿来抬几何。 */
    const gLift = new Float32Array(n4);
    const gLiftS = new Float32Array(n4);
    for (let j = 0; j <= NR; j++) {
      const v = gV[j];
      const wv = smoothstep(0.50, 0.70, v) * (1 - smoothstep(0.88, 1.0, v));
      for (let i = 0; i < W1; i++) {
        const u = gU[i];
        const wu = smoothstep(0.02, 0.18, u) * (1 - smoothstep(0.82, 0.98, u));
        const ex = Math.max(0, gEs[j * W1 + i] - 0.10);
        gLift[j * W1 + i] = clamp(ex * 0.22 * 18, 0, GROUND_LIFT_MAX) * wv * wu;
      }
    }
    // 行方向 9 行低通。曾试过「横向 9 列 + 纵向 21 行」把挤出做成大块
    // （MIDI-3D 按实例建模的近似），但视锥边缘出现规则的点状竖条纹，
    // A/B 截图后回退；这里的 9 行低通是实测最稳的一组。
    for (let j = 0; j <= NR; j++) {
      for (let i = 0; i < W1; i++) {
        let s = 0, n = 0;
        for (let dj = -4; dj <= 4; dj++) {
          const jj = Math.min(NR, Math.max(0, j + dj));
          s += gLift[jj * W1 + i]; n++;
        }
        gLiftS[j * W1 + i] = s / n;
      }
    }

    const vg = Number.isFinite(opt.vg) ? clamp(opt.vg, 0.6, 2.2) : VERT_GAIN;
    for (let j = 0; j <= NR; j++) {
      const v = gV[j];
      const phi = (0.5 - v) * vfov;
      const sp = Math.sin(phi), cp = Math.cos(phi);
      const tGround = sp < -0.002 ? eye / (-sp) : 1e9;
      for (let i = 0; i <= NC; i++) {
        const a = (i / NC) * Math.PI * 2;
        const k = j * W1 + i;
        const tDepth = distOf(gD[k]);
        let t, y;
        if (sp < -0.002) {
          // 地面：水平距离仍按「平地 + 眼高」反解（透视与行走平面一致），
          // 但深度不再被丢弃 —— 明显比同排基线更近的连续区域被抬成近景物体。
          // gLiftS 已做死区 + 5×5 低通 + 9 行低通，平坦水面保持完全水平。
          t = tGround;
          y = (opt.flat || opt.legacy) ? 0 : gLiftS[k];
        } else {
          // 地平线以上：不再收成 128m 刚性圆柱，按标定距离放点，获得真实纵深；
          // 过远的四边形会在索引阶段拆到 backdrop 层（跟随相机、无视差）。
          t = tDepth;
          y = clamp(eye + sp * t * vg, -3.0, 520);
        }
        t = clamp(t, 2.4, 700);
        gy[k] = y;
        gT[k] = t;
        gx[k] = Math.sin(a) * cp * t;
        gz[k] = Math.cos(a) * cp * t;
      }
    }

    /* 2) 前拉约束：断崖处单轮单侧吸附，地面行豁免
     *
     * 旧实现是 3 轮四邻域 min 腐蚀，有两个系统性副作用：
     *  - 地面行 t = eye/(-sinφ)，靠近地平线时相邻行比值轻易 > 1.25，
     *    即使深度图完全平滑也会被吸附；吸附后按当前行 φ 重算 gy，
     *    地面离开 y=0（实测 1083 个顶点、最大 1.64m，是"走动尖刺"同源伪影）；
     *  - 三轮腐蚀等于对整张深度图做形态学腐蚀，断崖周边被拉成台阶。
     * 现在：地面行既不参与吸附，也不作为吸附目标；其余区域只做一轮。
     */
    const groundRow = new Uint8Array(NR + 1);
    for (let j = 0; j <= NR; j++) {
      groundRow[j] = Math.sin((0.5 - gV[j]) * vfov) < -0.002 ? 1 : 0;
    }
    {
      const src = gT.slice();
      // 物理标定后距离动态范围从几米到几百米，固定 1.25 倍会把连续地形
      // 误判成断崖、大量吸附成台阶（画面出现横向条纹与撕裂），故提高阈值。
      const pullRatio = cal ? 1.9 : 1.25;
      for (let j = 0; j <= NR; j++) {
        if (groundRow[j]) continue;                 // 地面行豁免前拉吸附
        for (let i = 0; i <= NC; i++) {
          const k = j * W1 + i;
          const iL = j * W1 + ((i - 1 + W1) % W1);
          const iR = j * W1 + ((i + 1) % W1);
          const jD = Math.max(0, j - 1) * W1 + i;
          const jU2 = Math.min(NR, j + 1) * W1 + i;
          // 垂直邻居若是地面行则忽略，避免把天空行拉到近地面距离
          const mn = Math.min(
            src[iL], src[iR],
            groundRow[Math.max(0, j - 1)] ? Infinity : src[jD],
            groundRow[Math.min(NR, j + 1)] ? Infinity : src[jU2]
          );
          if (Number.isFinite(mn) && src[k] > mn * pullRatio) {
            // 深度断崖：吸附到前景深度（标准 DIBR 补洞），桥接面片被遮住
            const t = mn;
            gT[k] = t;
            const a = (i / NC) * Math.PI * 2;
            const phi = (0.5 - gV[j]) * vfov;
            gx[k] = Math.sin(a) * Math.cos(phi) * t;
            gz[k] = Math.cos(a) * Math.cos(phi) * t;
            gy[k] = clamp(eye + Math.sin(phi) * t * vg, -3.0, 520);
          }
        }
      }
    }

    /* 3) 顶点 / UV / 法线 / 顶点色 */
    const pos = [], uv = [], col = [], nrm = [], idx = [];
    const edge = new Float32Array(n4 * 2);
    for (let j = 0; j <= NR; j++) {
      for (let i = 0; i <= NC; i++) {
        const k = j * W1 + i;
        pos.push(gx[k], gy[k], gz[k]);
        uv.push(gU[i], gV[j]);
        // 侧向融雾 / 底缘融地面色：让照片边界长进环境里，而不是一张硬切卡片
        edge[k * 2] = clamp((1 - gVis[i]) * 2.0, 0, 1) * 0.98;
        edge[k * 2 + 1] = clamp(smoothstep(0.80, 1.0, gV[j]) * 1.0, 0, 1.0);
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
        // 体积感来自法线：拉开受光/背光差，并在深度断崖处加一圈接触阴影（AO），
        // 让挤出的近景物体有「侧面」而不是一张和地面同色的斜坡。
        const gradD = Math.abs(gEs[iR] - gEs[iL]) + Math.abs(gEs[jU2] - gEs[jD]);
        const ao = 1 - 0.26 * clamp(gradD * 7, 0, 1);
        const shade = clamp((0.64 + 0.56 * ndl + 0.10 * ny) * ao
                            * (1 - 0.10 * clamp(-ny, 0, 1)), 0.40, 1.40);
        // 天空处渐隐，交给程序化天穹
        const skyV = gSky[i];
        // 硬边剪影：只在天际线"上方"渐隐，下方必须实心，否则会出现亮缝
        const aSky = smoothstep(skyV - 0.012, skyV, v);
        // 侧向与底缘不再靠 alpha 消失（那会露出硬切边，放宽渐隐又会露出
        // 拉伸条纹），改由顶点 edge 系数把颜色融进雾色 / 地面色；
        // 只有天际线以上的剪影保持透明，交给程序化天穹。
        const alpha = aSky;
        col.push(shade, shade, shade, alpha);
        nrm.push(nx, ny, nz);
      }
    }

    /* 4) 索引：跳过完全在天空里的格子；把远景四边形拆到 backdrop 层。
     * 远景本就不该有视差（GLB 资产模式的结论）：它由 renderer 带着相机平移，
     * 近景保持世界坐标。这样中景不再被「压到同一个圆柱半径」，
     * 走动时也不会出现整片照片跟着人一起滑的沙盘感。 */
    const idxBack = [];
    const bgIdx = [];
    let skipped = 0;
    const useLdi = !opt.legacy && opt.ldi === true;
    for (let j = 0; j < NR; j++) {
      const v0 = gV[j], v1 = gV[j + 1];
      for (let i = 0; i < NC; i++) {
        // 渐隐带前半照常建面（到 gVis=0.5，此时已几乎全融进雾色），
        // 视锥外不建面 —— 那里只有镜像拉伸的重复纹理，建了就是条纹。
        if (gVis[i] < 0.5 || gVis[(i + 1) % W1] < 0.5) continue;
        const s0 = gSky[i], s1 = gSky[(i + 1) % W1];
        if (v0 < Math.min(s0, s1) - 0.05 && v1 < Math.min(s0, s1) - 0.05) continue;
        const a0 = j * W1 + i;
        const a1 = a0 + 1;
        const b0 = (j + 1) * W1 + i;
        const b1 = b0 + 1;
        const t0 = gT[a0], t1 = gT[a1], t2 = gT[b0], t3 = gT[b1];
        const tMin = Math.min(t0, t1, t2, t3), tMax = Math.max(t0, t1, t2, t3);
        /* 断崖跳面（LDI / VistaDream / pipeline/geom.py 的共同判据）：
           前景物体与背景被强行连成一张面，就是「拉伸的鳍 / 纸片感」的来源。
           只在近中景、且确实跨越深度不连续处断开四边形，洞里由低频世界幕补齐。 */
        if (useLdi && !groundRow[j] && tMin < 90 && tMax / Math.max(tMin, 0.6) > 1.9) {
          const grad = Math.max(
            Math.abs(gEs[a0] - gEs[a1]), Math.abs(gEs[b0] - gEs[b1]),
            Math.abs(gEs[a0] - gEs[b0]), Math.abs(gEs[a1] - gEs[b1]));
          // 洞不是丢掉，而是交给背景补洞层（只在洞里画，与前景不重叠，
          // 否则两层半透明曲面叠在一起会产生波纹干涉条纹）。
          if (grad > 0.05) { bgIdx.push(a0, b0, a1, a1, b0, b1); skipped++; continue; }
        }
        const far = !opt.legacy && !groundRow[j] && !groundRow[j + 1] && tMin > BACKDROP_T;
        const target = far ? idxBack : idx;
        target.push(a0, b0, a1, a1, b0, b1);
      }
    }

    const posF = new Float32Array(pos);
    const nrmF = new Float32Array(nrm);
    const uvF = new Float32Array(uv);
    const colF = new Float32Array(col);
    const backdrop = idxBack.length
      ? { pos: posF, normal: nrmF, uv: uvF, color: colF, edge, index: idxBack, tris: idxBack.length / 3 }
      : null;

    /* 5) 背景补洞层（LDI 轻量版，对应 3d-photo-inpainting / VistaDream 的
       「遮挡区域必须有内容」）：用每行的地面基线深度替代前景深度，得到一张
       「去掉近景物体后的连续曲面」，整体推远 25% 保证 depth test 下永远在
       前景之后。断崖跳面留下的洞由它填，贴图是低频模糊照片（零依赖的
       RGB-D inpainting 替身），不会出现 360° 世界幕那种竖直色带与尖峰。 */
    let bg = null;
    if (useLdi) {
      const bpos = new Float32Array(n4 * 3);
      const bcol = new Float32Array(n4 * 4);
      const bidx = bgIdx.slice();
      for (let j = 0; j <= NR; j++) {
        const v = gV[j];
        const phi = (0.5 - v) * vfov;
        const sp = Math.sin(phi), cp = Math.cos(phi);
        const tBg = clamp(distOf(rowBase[j]) * 1.25 + 6, 2.6, 700);
        for (let i = 0; i < W1; i++) {
          const a = (i / NC) * Math.PI * 2;
          const k = j * W1 + i;
          bpos[k * 3] = Math.sin(a) * cp * tBg;
          bpos[k * 3 + 1] = (sp < -0.002)
            ? gLiftS[k] * 0.35 : clamp(eye + sp * tBg * vg, -3.0, 520);
          bpos[k * 3 + 2] = Math.cos(a) * cp * tBg;
          const shade = colF[k * 4] * 0.86;
          bcol[k * 4] = shade; bcol[k * 4 + 1] = shade; bcol[k * 4 + 2] = shade;
          bcol[k * 4 + 3] = colF[k * 4 + 3];
        }
      }
      bg = bidx.length
        ? { pos: bpos, normal: nrmF, uv: uvF, color: bcol, edge, index: bidx, tris: bidx.length / 3 }
        : null;
    }
    return {
      pos: posF,
      normal: nrmF,
      uv: uvF,
      color: colF,
      edge,
      index: idx,
      backdrop,
      bg,
      skipped,
      vfov,
      calibration: cal ? { p: cal.p, zNear: cal.zNear, rNear: cal.rNear, rFar: cal.rFar } : null,
      /* 行走高度场：把挤出的近景物体当作可站立表面，避免玩家直接穿过几何。
         按 (角度, 距离) 反查地面行并做 3×3 均值平滑，幅度收 0.8×，
         防止站上物体时眼高突跳。 */
      heightField: function (x, z) {
        const r = Math.hypot(x, z);
        if (r < 3.0 || r > 120) return 0;
        const v = 0.5 + Math.asin(clamp(eye / r, 0, 0.999)) / vfov;
        if (v <= 0.5 || v >= 0.995) return 0;
        const a = ((Math.atan2(x, z) % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
        const fi = a / (Math.PI * 2) * NC;
        const fj = Math.round(v * NR);
        let sum = 0, n = 0;
        for (let dj = -1; dj <= 1; dj++) {
          const j = Math.min(NR, Math.max(0, fj + dj));
          if (!groundRow[j]) continue;
          for (let di = -1; di <= 1; di++) {
            const i = ((Math.round(fi) + di) % W1 + W1) % W1;
            sum += gy[j * W1 + i]; n++;
          }
        }
        return n ? clamp(sum / n * 0.8, -0.1, 1.8) : 0;
      },
      tris: idx.length / 3
    };
  }

  /* ---------------- 直角坐标重建（v1.13，默认路径） ----------------
   * 放弃「视锥 / 球面参数化」：把屏幕行当仰角、把深度当半径、视锥外用镜像外推，
   * 这套做法在边界必然产生尖刺与平铺贴片（v1.11 的融色只是把症状盖住）。
   * 改成 video-to-3D 方案同一口径（HunyuanWorld-Voyager 的 RGB-D → 世界点云、
   * WorldGen 的 RGBD→mesh、VistaDream 的 coarse scaffold + 遮挡补全）：
   *   1) 每个像素按相机内参反投影成世界坐标 P = eye + dir(u,v) · z(u,v)；
   *   2) 只保留有限直角区域 x∈[-A,A]、z∈[-B,0]、y∈[-2,Y]；
   *   3) 按屏幕邻接关系连网格，深度断崖处跳过四边形（不连成拉伸鳍）；
   *   4) 3×3 中值压掉单像素尖刺，法线 3×3 平滑。
   * 没有视锥掩码、没有镜像平铺、没有球面半径 —— 几何就是世界坐标本身。
   */
  function buildCartesianWorld(analysis, depth, opt) {
    opt = opt || {};
    const GW = opt.gw || 320, GH = opt.gh || 200;
    const A = opt.areaX || 55;           // 直角区域半宽（米）
    const B = opt.areaZ || 100;          // 相机前方深度（米）——有限扇环的半径
    const edgeRatio = opt.edgeRatio || 2.6;   // 真实深度噪声大，1.8 会在湖面打出一片碎片洞
    const yGain = opt.yGain || 1.35;     // 只作用于地平线以上：把山体/树木的高度立起来
    const prog = opt.onProgress || function () {};
    const eye = 1.7;
    const aspect = analysis.W / Math.max(1, analysis.H);
    const tanH = Math.tan(HFOV / 2);
    const tanV = tanH / aspect;
    const vfov = 2 * Math.atan(tanV);
    /* 相机俯仰角：从地平线位置反解。照片是俯拍（楼梯/桌面）时地平线在上方，
       若仍按水平相机反投影，地面会被卷成弧形（鱼眼）——这是室内照片崩掉的根因。 */
    const pitch = Math.atan(clamp((0.5 - analysis.horizon) * 2 * tanV, -0.9, 0.9));
    const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
    /* 室内 / 俯拍检测：地平线被顶到画面上方 42% 以外（或压到 72% 以下）。
       户外风景的地平线基本在 0.45~0.60。室内走短基线测距、不夸张高度、
       不裁天空，否则会被户外双锚点拉到几十米、卷成弧形。 */
    const indoor = analysis.horizon < 0.42 || analysis.horizon > 0.72;

    const dw = depth.width || depth.size;
    const dh = depth.height || depth.size;
    const D = depth.data;

    /* 逐列天际线：用**未平滑**的原始轮廓做 alpha 判定，树冠/悬崖的锯齿才不会被
       平滑成一条水平直线（蓝调雪林的树梢被整排切掉就是这个原因）。
       原始轮廓先做 3 列中值，去掉单列噪点。 */
    const rawTop = analysis.topRaw || analysis.top;
    const topS = new Float32Array(NCOL);
    for (let i = 0; i < NCOL; i++) {
      // 5 列中值：只保留轮廓的大起伏，去掉树与树之间的单列跳变
      // （3 列时树梢会被拉成一排竖直冰柱）。
      const w = [];
      for (let k = -2; k <= 2; k++) w.push(rawTop[(i + k + NCOL) % NCOL]);
      w.sort((a, b) => a - b);
      topS[i] = w[2];
    }
    const topAt = (u) => {
      const f = clamp(u, 0, 0.9999) * NCOL;
      const i0 = Math.floor(f), i1 = (i0 + 1) % NCOL;
      return lerp(topS[i0], topS[i1], f - i0);
    };

    /* 2/98 分位归一化（与 depth.js / pipeline 的约定一致） */
    const hist = [];
    for (let y = 0; y < dh; y += 3) for (let x = 0; x < dw; x += 3) hist.push(D[y * dw + x]);
    hist.sort((a, b) => a - b);
    const pLo = hist[Math.floor(hist.length * 0.02)] || 0;
    const pHi = hist[Math.floor(hist.length * 0.98)] || 1;
    const span = Math.max(1e-4, pHi - pLo);
    const dnAt = (u, v) => {
      const x = clamp(u, 0, 1) * (dw - 1);
      const y = clamp(v, 0, 1) * (dh - 1);
      const x0 = Math.floor(x), y0 = Math.floor(y);
      const x1 = Math.min(dw - 1, x0 + 1), y1 = Math.min(dh - 1, y0 + 1);
      const fx = x - x0, fy = y - y0;
      const a = D[y0 * dw + x0], b = D[y0 * dw + x1];
      const c = D[y1 * dw + x0], d = D[y1 * dw + x1];
      const raw = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
      return clamp((raw - pLo) / span, 0, 1);
    };

    /* 网格深度 + 3×3 中值：Depth Anything 的单像素离群值就是「尖刺」的来源 */
    const gD = new Float32Array(GW * GH);
    const gM = new Float32Array(GW * GH);
    for (let j = 0; j < GH; j++) {
      for (let i = 0; i < GW; i++) gD[j * GW + i] = dnAt(i / (GW - 1), j / (GH - 1));
    }
    // 5×5 中值：3×3 对真实 ONNX 深度不够，湖面/云边会残留成片的单像素跳变，
    // 反投影后就是「漂浮碎片」。25 元素插入排序，40960 顶点约 10ms。
    const win = new Float32Array(25);
    for (let j = 0; j < GH; j++) {
      for (let i = 0; i < GW; i++) {
        let n = 0;
        for (let dj = -2; dj <= 2; dj++) {
          const jj = clamp(j + dj, 0, GH - 1);
          for (let di = -2; di <= 2; di++) {
            const ii = clamp(i + di, 0, GW - 1);
            win[n++] = gD[jj * GW + ii];
          }
        }
        for (let a = 1; a < 25; a++) {
          const v = win[a]; let b = a - 1;
          while (b >= 0 && win[b] > v) { win[b + 1] = win[b]; b--; }
          win[b + 1] = v;
        }
        gM[j * GW + i] = win[12];
      }
    }
    prog(0.30);
    /* 地平线以下（地面/水面）再做一次各向异性低通：单目深度会把水面倒影当成
       另一层几何，反投影后就是一片「漂浮碎片」。横向 7 列 + 纵向 5 行的均值
       保留大尺度起伏，抹掉高频碎片；地平线以上（山体/物体）保留原深度。 */
    const gS = new Float32Array(GW * GH);
    for (let j = 0; j < GH; j++) {
      const v = j / (GH - 1);
      const isGround = v > analysis.horizon + 0.005;
      for (let i = 0; i < GW; i++) {
        if (!isGround) { gS[j * GW + i] = gM[j * GW + i]; continue; }
        // 3×3（不是 7×5）：室内楼梯/台阶的踏步细节必须保留，
        // 过强的地面低通会把整段楼梯抹成一个弧形大斜面。
        let s = 0, n = 0;
        for (let dj = -1; dj <= 1; dj++) {
          const jj = clamp(j + dj, 0, GH - 1);
          for (let di = -1; di <= 1; di++) {
            const ii = clamp(i + di, 0, GW - 1);
            s += gM[jj * GW + ii]; n++;
          }
        }
        gS[j * GW + i] = s / n;
      }
    }
    /* 地面单调性（v1.14）：俯视 / 下楼梯时，越往下（v 越大）必须越近（d 越大）。
       单目深度在台阶、雪地、水面上常出现局部反号，反投影后就是
       「向下的楼梯被识别成略微向上」。用行中位的单调包络修正整体趋势，
       局部台阶细节保留 45%，方向一定正确。 */
    const rowD = new Float32Array(GH);
    for (let j = 0; j < GH; j++) {
      let s = 0;
      for (let i = 0; i < GW; i++) s += gS[j * GW + i];
      rowD[j] = s / GW;
    }
    for (let j = 1; j < GH; j++) rowD[j] = Math.max(rowD[j], rowD[j - 1]);
    for (let j = 0; j < GH; j++) {
      const v = j / (GH - 1);
      if (v <= analysis.horizon + 0.005) continue;
      for (let i = 0; i < GW; i++) {
        const k = j * GW + i;
        gS[k] = clamp(rowD[j] + (gS[k] - rowD[j]) * 0.75, 0, 1);
      }
    }
    prog(0.55);

    const dSample = (u, v) => {
      const x = clamp(u, 0, 1) * (GW - 1);
      const y = clamp(v, 0, 1) * (GH - 1);
      const x0 = Math.floor(x), y0 = Math.floor(y);
      const x1 = Math.min(GW - 1, x0 + 1), y1 = Math.min(GH - 1, y0 + 1);
      const fx = x - x0, fy = y - y0;
      const a = gS[y0 * GW + x0], b = gS[y0 * GW + x1];
      const c = gS[y1 * GW + x0], d = gS[y1 * GW + x1];
      return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
    };

    /* 双锚点标定沿用 v1.11 的成果：深度 → 米，不再拍脑袋 */
    const gV = new Float32Array(GH);
    for (let j = 0; j < GH; j++) gV[j] = j / (GH - 1);
    const cal = calibrateScale(dSample, gV, vfov, eye);
    /* 室内（无天空）不能用户外双锚点：地面带只有一两米，外推到 520m 会把
       1m 的台阶放到 5m、3m 的走廊放到 50m，整段楼梯被拉成弧形。
       室内改用短基线线性映射（0.6~18m），等 hasSky 算出来后延迟生效。 */
    const metric = (d) => {
      if (indoor) return 0.6 + 18 * (1 - clamp(d, 0, 1));
      return cal ? cal.metric(d)
        : clamp(2.5 + 250 * Math.pow(1 - clamp(d, 0, 1), 1.7), 2.6, 280);
    };

    /* 天空深度基线：顶部 8% 行的中位视差。天空 = 深度最小（最远）的那一片，
       用深度判定天空（而不是逐列天际线）边缘是平滑的，不会在树冠上留下锯齿。 */
    const skyVals = [];
    const skyRows = Math.max(1, Math.floor(GH * 0.08));
    for (let j = 0; j < skyRows; j++) {
      for (let i = 0; i < GW; i += 3) skyVals.push(gS[j * GW + i]);
    }
    skyVals.sort((a, b) => a - b);
    const skyD = skyVals.length ? skyVals[Math.floor(skyVals.length * 0.5)] : 0.1;
    // 有没有天空：底部 20% 行与顶部 8% 行的中位深度差。室内（楼梯/房间）
    // 顶部是天花板、底部是地面，差很小 → 不裁任何像素，否则会把半张照片切掉。
    let botSum = 0, botN = 0;
    for (let j = Math.floor(GH * 0.8); j < GH; j++) {
      for (let i = 0; i < GW; i += 3) { botSum += gS[j * GW + i]; botN++; }
    }
    const hasSky = botN ? (botSum / botN) - skyD > 0.12 : false;

    /* 1) 反投影：世界坐标点（相机朝 -Z，yaw=0，眼高 1.7m） */
    const pos = new Float32Array(GW * GH * 3);
    const uv = new Float32Array(GW * GH * 2);
    const col = new Float32Array(GW * GH * 4);
    const edge = new Float32Array(GW * GH * 2);
    const gT = new Float32Array(GW * GH);
    for (let j = 0; j < GH; j++) {
      const v = j / (GH - 1);
      for (let i = 0; i < GW; i++) {
        const u = i / (GW - 1);
        const k = j * GW + i;
        // 天空也压到同一个有限半径 B（不能放到 1.6B，否则会被区域裁剪整片切掉，
        // 天际线处出现一条水平分界线）。
        const d0 = dSample(u, v);
        const t = clamp(metric(d0), 0.6, B);
        const dx = (u - 0.5) * 2 * tanH;
        const dy0 = (0.5 - v) * 2 * tanV;
        // 绕相机 X 轴加俯仰：dy0 = tan(pitch) 的那一行正好落在地平线
        const dy = dy0 * cosP - sinP;
        const dz = -dy0 * sinP - cosP;
        const l = Math.hypot(dx, dy, dz) || 1;
        pos[k * 3] = (dx / l) * t;
        // 地平线以上做竖直夸张：单目深度会把山体/树压扁，照片里的仰角
        // 乘上距离才是真实高度；地面（dy<0）不夸张，避免脚下被拉低。
        // 室内（无天空）不做竖直夸张：那会把楼梯/天花板拉成弧形大曲面
        pos[k * 3 + 1] = eye + (dy / l) * t * ((!indoor && hasSky && dy > 0) ? yGain : 1);
        pos[k * 3 + 2] = (dz / l) * t;
        uv[k * 2] = u;
        uv[k * 2 + 1] = v;
        gT[k] = t;
        /* alpha：天际线以上透明（交给天穹），四周 3.5% 渐隐（边界是照片梯形
           的投影，不是视锥卡片，所以不需要镜像/融色） */
        /* 天空判定用**深度**而不是逐列天际线：
           比天空基线明显更近的像素才保留（地面/山体/树木/建筑），
           天空 alpha=0 交给程序化天穹。深度图是连续的，所以树冠边缘
           不会出现锯齿状青边或一条水平切线。 */
        const aSky = (indoor || !hasSky) ? 1 : smoothstep(skyD + 0.03, skyD + 0.14, d0);
        const aU = smoothstep(0, 0.02, u) * (1 - smoothstep(0.98, 1, u));
        const aV = smoothstep(0, 0.015, v) * (1 - smoothstep(0.985, 1, v));
        col[k * 4 + 3] = aSky * aU * aV;
        col[k * 4] = 1; col[k * 4 + 1] = 1; col[k * 4 + 2] = 1;
        // 有限区域的边界就是照片投影的梯形，alpha 硬边比"融色灰带"干净；
        // edge 保持 0，不参与 mode 4 的环境色混合。
        edge[k * 2] = 0;
        edge[k * 2 + 1] = 0;
      }
    }

    /* 1b) 竖直尖刺抑制：树梢/细杆/电线这类 1~2 像素宽的远景，深度稍有误差
       就会被反投影成几米长的"冰柱"。这里对 y 做 3×3 中值，只把偏离邻域中值
       超过 1.6m 的顶点拉回 —— 山脊/悬崖的连续起伏仍然保留。 */
    const yWin = new Float32Array(9);
    for (let j = 0; j < GH; j++) {
      for (let i = 0; i < GW; i++) {
        let n = 0;
        for (let dj = -1; dj <= 1; dj++) {
          const jj = clamp(j + dj, 0, GH - 1);
          for (let di = -1; di <= 1; di++) {
            const ii = clamp(i + di, 0, GW - 1);
            yWin[n++] = pos[(jj * GW + ii) * 3 + 1];
          }
        }
        for (let a = 1; a < 9; a++) {
          const v = yWin[a]; let b = a - 1;
          while (b >= 0 && yWin[b] > v) { yWin[b + 1] = yWin[b]; b--; }
          yWin[b + 1] = v;
        }
        const k = j * GW + i;
        pos[k * 3 + 1] = clamp(pos[k * 3 + 1], yWin[4] - 1.6, yWin[4] + 1.6);
      }
    }

    /* 2) 法线：世界坐标邻居坐标差分叉积 + 3×3 平滑（尖刺的第二道防线） */
    const nrm = new Float32Array(GW * GH * 3);
    for (let j = 0; j < GH; j++) {
      for (let i = 0; i < GW; i++) {
        const k = j * GW + i;
        const iL = j * GW + Math.max(0, i - 1), iR = j * GW + Math.min(GW - 1, i + 1);
        const jD = Math.max(0, j - 1) * GW + i, jU = Math.min(GH - 1, j + 1) * GW + i;
        const ax = pos[iR * 3] - pos[iL * 3], ay = pos[iR * 3 + 1] - pos[iL * 3 + 1], az = pos[iR * 3 + 2] - pos[iL * 3 + 2];
        const bx = pos[jU * 3] - pos[jD * 3], by = pos[jU * 3 + 1] - pos[jD * 3 + 1], bz = pos[jU * 3 + 2] - pos[jD * 3 + 2];
        let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
        const nl = Math.hypot(nx, ny, nz);
        if (nl > 1e-8) { nx /= nl; ny /= nl; nz /= nl; }
        else { nx = 0; ny = 1; nz = 0; }
        // 地形法线统一朝上（索引顺序决定叉积朝 -Z，地面会变成背光）
        if (ny < 0) { nx = -nx; ny = -ny; nz = -nz; }
        nrm[k * 3] = nx; nrm[k * 3 + 1] = ny; nrm[k * 3 + 2] = nz;
      }
    }
    const nrmS = new Float32Array(nrm.length);
    for (let j = 0; j < GH; j++) {
      for (let i = 0; i < GW; i++) {
        let sx = 0, sy = 0, sz = 0;
        for (let dj = -1; dj <= 1; dj++) {
          const jj = clamp(j + dj, 0, GH - 1);
          for (let di = -1; di <= 1; di++) {
            const ii = clamp(i + di, 0, GW - 1);
            const kk = jj * GW + ii;
            sx += nrm[kk * 3]; sy += nrm[kk * 3 + 1]; sz += nrm[kk * 3 + 2];
          }
        }
        const l = Math.hypot(sx, sy, sz) || 1;
        const k = j * GW + i;
        nrmS[k * 3] = sx / l; nrmS[k * 3 + 1] = sy / l; nrmS[k * 3 + 2] = sz / l;
      }
    }
    prog(0.78);

    /* 3) 顶点色：预烘焙太阳光照 + 深度断崖接触阴影 */
    const sunAz = analysis.sunAz, sunEl = analysis.sunElev;
    const sunV = [Math.sin(sunAz) * Math.cos(sunEl), Math.sin(sunEl), Math.cos(sunAz) * Math.cos(sunEl)];
    for (let j = 0; j < GH; j++) {
      for (let i = 0; i < GW; i++) {
        const k = j * GW + i;
        const iL = j * GW + Math.max(0, i - 1), iR = j * GW + Math.min(GW - 1, i + 1);
        const jD = Math.max(0, j - 1) * GW + i, jU = Math.min(GH - 1, j + 1) * GW + i;
        const ndl = Math.max(0, nrmS[k * 3] * sunV[0] + nrmS[k * 3 + 1] * sunV[1] + nrmS[k * 3 + 2] * sunV[2]);
        const gx = (gT[iR] - gT[iL]) / Math.max(0.5, gT[k] * 0.1);
        const gy = (gT[jU] - gT[jD]) / Math.max(0.5, gT[k] * 0.1);
        const ao = 1 - 0.24 * clamp(Math.hypot(gx, gy) * 1.6, 0, 1);
        const shade = clamp((0.64 + 0.56 * ndl + 0.08 * nrmS[k * 3 + 1]) * ao, 0.40, 1.38);
        col[k * 4] = shade; col[k * 4 + 1] = shade; col[k * 4 + 2] = shade;
      }
    }

    /* 4) 索引：alpha 剔除 + 有限直角区域裁剪 + 断崖跳面。
       区域裁剪是「有限直角坐标世界」的边界：照片投影成梯形后自然收边，
       不再需要视锥掩码或镜像平铺。 */
    const idx = [];
    let skipped = 0, culledAlpha = 0, culledArea = 0;
    for (let j = 0; j < GH - 1; j++) {
      for (let i = 0; i < GW - 1; i++) {
        const a0 = j * GW + i, a1 = a0 + 1;
        const b0 = (j + 1) * GW + i, b1 = b0 + 1;
        const alpha = Math.max(col[a0 * 4 + 3], col[a1 * 4 + 3], col[b0 * 4 + 3], col[b1 * 4 + 3]);
        if (alpha < 0.06) { culledAlpha++; continue; }
        let inside = true;
        for (const k of [a0, a1, b0, b1]) {
          if (Math.abs(pos[k * 3]) > A || pos[k * 3 + 2] < -B - 0.5 || pos[k * 3 + 2] > 1.5) { inside = false; break; }
        }
        if (!inside) { culledArea++; continue; }
        const tMax = Math.max(gT[a0], gT[a1], gT[b0], gT[b1]);
        const tMin = Math.min(gT[a0], gT[a1], gT[b0], gT[b1]);
        if (tMax / Math.max(tMin, 0.6) > edgeRatio) { skipped++; continue; }
        idx.push(a0, b0, a1, a1, b0, b1);
      }
    }

    /* 5) 行走高度场：把反投影顶点烘进 1m 的 XZ 网格（取格内最高点），
       玩家站在真实几何上，而不是被强制 y=0 的平面。 */
    const cell = 1.0;
    const hw = Math.ceil((2 * A) / cell) + 1, hh = Math.ceil(B / cell) + 1;
    const hgrid = new Float32Array(hw * hh).fill(-1e9);
    for (let k = 0; k < GW * GH; k++) {
      const x = pos[k * 3], y = pos[k * 3 + 1], z = pos[k * 3 + 2];
      if (Math.abs(x) > A || z < -B || z > 0) continue;
      const gi = clamp(Math.round((x + A) / cell), 0, hw - 1);
      const gj = clamp(Math.round((-z) / cell), 0, hh - 1);
      const idx2 = gj * hw + gi;
      if (y > hgrid[idx2]) hgrid[idx2] = y;
    }
    // 空格用 3×3 邻域均值补两轮，避免走到网格缝里掉下去
    for (let pass = 0; pass < 2; pass++) {
      const src = hgrid.slice();
      for (let gj = 0; gj < hh; gj++) {
        for (let gi = 0; gi < hw; gi++) {
          const k = gj * hw + gi;
          if (src[k] > -1e8) continue;
          let s = 0, n = 0;
          for (let dj = -1; dj <= 1; dj++) {
            for (let di = -1; di <= 1; di++) {
              const jj = gj + dj, ii = gi + di;
              if (jj < 0 || jj >= hh || ii < 0 || ii >= hw) continue;
              const v = src[jj * hw + ii];
              if (v > -1e8) { s += v; n++; }
            }
          }
          if (n) hgrid[k] = s / n;
        }
      }
    }
    function heightField(x, z) {
      const fx = clamp((x + A) / cell, 0, hw - 1.001);
      const fz = clamp((-z) / cell, 0, hh - 1.001);
      const i0 = Math.floor(fx), j0 = Math.floor(fz);
      const tx = fx - i0, tz = fz - j0;
      const a = hgrid[j0 * hw + i0], b = hgrid[j0 * hw + i0 + 1];
      const c = hgrid[(j0 + 1) * hw + i0], d = hgrid[(j0 + 1) * hw + i0 + 1];
      const v = (a * (1 - tx) + b * tx) * (1 - tz) + (c * (1 - tx) + d * tx) * tz;
      return (v > -1e8 && Number.isFinite(v)) ? clamp(v, -0.4, 40) : 0;
    }

    prog(1.0);
    const quadTotal = Math.max(1, (GW - 1) * (GH - 1));
    return {
      pos, normal: nrmS, uv, color: col, edge, index: idx,
      vfov, tris: idx.length / 3, skipped,
      calibration: cal ? { p: cal.p, zNear: cal.zNear, rNear: cal.rNear, rFar: cal.rFar } : null,
      heightField,
      bbox: { A, B },
      grid: { GW, GH },
      /* 信息保留率诊断：哪一步丢了多少面，直接决定「图片丢信息」的排查方向 */
      stats: {
        kept: idx.length / 6, total: quadTotal,
        keepRate: (idx.length / 6) / quadTotal,
        culledAlpha, culledArea, skipped
      }
    };
  }

  /* ---------------- 侧面 / 背面（v1.14） ----------------
   * 单张照片只有正面。这里用「低分辨率镜像副本 + 绕 Y 轴旋转」把同一片
   * 几何铺到左右和背后：±90° 是侧面、180° 是背面，UV 做水平镜像。
   * 这不是生成式补全（没有 inpainting 模型），但转身/走远时不再是空白雾，
   * 而是连续的镜像景深；配合雾和降亮度，视觉上像「四周都有岸线」。
   * 参考 WorldGen / Matrix-3D 的 360° 可探索目标，但只做几何外推这一层。 */
  function buildSurround(relief, opt) {
    opt = opt || {};
    const step = opt.step || 2;
    const GW = relief.grid.GW, GH = relief.grid.GH;
    const cols = [], rows = [];
    for (let i = 0; i < GW; i += step) cols.push(i);
    for (let j = 0; j < GH; j += step) rows.push(j);
    const w = cols.length, h = rows.length;
    const pos = [], uv = [], col = [], nrm = [], idx = [], edge = [];
    // 5 份副本 + 主网格 = 每 60° 一个方向，6×62° 刚好覆盖 360°，
    // 转身不会在方向之间留出大片空白雾。
    for (let q = 1; q < 6; q++) {
      const angle = (q * Math.PI) / 3;
      const ca = Math.cos(angle), sa = Math.sin(angle);
      const base = pos.length / 3;
      for (let jj = 0; jj < h; jj++) {
        const j = rows[jj];
        for (let ii = 0; ii < w; ii++) {
          const i = cols[ii], k = j * GW + i;
          const x = relief.pos[k * 3], y = relief.pos[k * 3 + 1], z = relief.pos[k * 3 + 2];
          pos.push(x * ca + z * sa, y, -x * sa + z * ca);
          nrm.push(relief.normal[k * 3], relief.normal[k * 3 + 1], relief.normal[k * 3 + 2]);
          uv.push(1 - relief.uv[k * 2], relief.uv[k * 2 + 1]);
          // 副本左右各 12% 渐隐：六份 60° 副本在接缝处重叠过渡，
          // 否则会看到一条条竖直硬边（像几块屏幕拼在一起）。
          const uu = relief.uv[k * 2];
          const a = relief.color[k * 4 + 3]
            * smoothstep(0, 0.12, uu) * (1 - smoothstep(0.88, 1, uu));
          const shade = relief.color[k * 4] * 0.60;   // 侧/背面压暗，弱化镜像重复感
          col.push(shade, shade, shade, a);
          edge.push(0, 0);
        }
      }
      for (let jj = 0; jj < h - 1; jj++) {
        for (let ii = 0; ii < w - 1; ii++) {
          const a0 = base + jj * w + ii, a1 = a0 + 1;
          const b0 = base + (jj + 1) * w + ii, b1 = b0 + 1;
          idx.push(a0, b0, a1, a1, b0, b1);
        }
      }
    }
    return {
      pos: new Float32Array(pos), normal: new Float32Array(nrm),
      uv: new Float32Array(uv), color: new Float32Array(col),
      edge: new Float32Array(edge), index: idx, tris: idx.length / 3
    };
  }

  /* 程序化地平环：视锥外的延续（不是照片内容，是"推测"出的远景雾带）
   * 实验特性，默认关闭（?ring=1）。做法上只保证"不突兀"：
   * 顶边 alpha 渐隐到 0、颜色从地平线色渐变到天顶色，否则会出现一条硬边灰带。 */
  function buildHorizonRing(analysis, opt) {
    opt = opt || {};
    const SEG = 160, R = opt.radius || 280;
    const hor = analysis.fog, top = analysis.skyTop;   // 底部对齐天穹底色，避免地平线处出现横切线
    const pos = [], col = [], uv = [], nrm = [], idx = [];
    const rows = [-2.0, 6.0, 22.0, 48.0];
    const tints = [0.74, 0.88, 1.0, 0.96];
    const alphas = [1.0, 0.72, 0.34, 0.0];        // 越往上越透明，融进程序化天穹
    for (let r = 0; r < rows.length; r++) {
      for (let i = 0; i <= SEG; i++) {
        const a = (i / SEG) * Math.PI * 2;
        pos.push(Math.sin(a) * R, rows[r], Math.cos(a) * R);
        nrm.push(-Math.sin(a), 0.25, -Math.cos(a));
        uv.push(i / SEG, r / (rows.length - 1));
        const t = tints[r];
        const mix = r / (rows.length - 1);
        col.push(lerp(hor[0], top[0], mix) * t, lerp(hor[1], top[1], mix) * t,
                 lerp(hor[2], top[2], mix) * t, alphas[r]);
      }
    }
    for (let r = 0; r < rows.length - 1; r++) {
      for (let i = 0; i < SEG; i++) {
        const a0 = r * (SEG + 1) + i, a1 = a0 + 1;
        const b0 = (r + 1) * (SEG + 1) + i, b1 = b0 + 1;
        idx.push(a0, b0, a1, a1, b0, b1);
      }
    }
    return { pos: new Float32Array(pos), normal: new Float32Array(nrm),
             uv: new Float32Array(uv), color: new Float32Array(col), index: idx };
  }

  /* 程序化地面：只在重建表面没有覆盖到的地方兜底（脚下、视锥外），不用照片平铺 */
  /* 兜底地面（v1.14）：从「半径 90m 的无限圆环」改成「覆盖有限直角区域的矩形」，
     边缘 8 格 alpha 渐隐 —— 这就是「有限长扇环」的地面边界：走到边上，
     世界收进雾里，而不是一直延伸下去。 */
  function buildFallbackGround(analysis, rng, opt) {
    opt = opt || {};
    const A = opt.A || 55, B = opt.B || 100;
    const NX = 56, NZ = 56;
    const pos = [], nrm = [], uv = [], col = [], idx = [];
    for (let j = 0; j <= NZ; j++) {
      const z = -(j / NZ) * B;
      for (let i = 0; i <= NX; i++) {
        const x = -A + (i / NX) * 2 * A;
        pos.push(x, -0.05, z);
        nrm.push(0, 1, 0);
        uv.push(x / 9, z / 9);
        const edge = clamp(Math.min(i, NX - i, j, NZ - j) / 8, 0, 1);
        col.push(1, 1, 1, edge);
      }
    }
    for (let j = 0; j < NZ; j++) {
      for (let i = 0; i < NX; i++) {
        const a0 = j * (NX + 1) + i, a1 = a0 + 1;
        const b0 = (j + 1) * (NX + 1) + i, b1 = b0 + 1;
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
    const PWGL = global.PW.GL;

    /* v1.13：默认走直角坐标反投影重建（buildCartesianWorld）。
       ?cart=0 或 ?v1=1 才回到旧的视锥/球面参数化路径，仅作 A/B 与复现。 */
    const useCartesian = !!(depthMap && !(opts && opts.legacy) && !(opts && opts.cartesian === false));
    const relief = depthMap
      ? (useCartesian
          ? buildCartesianWorld(analysis, depthMap, {
              areaX: opts && opts.areaX, areaZ: opts && opts.areaZ,
              gw: opts && opts.gw, gh: opts && opts.gh,
              yGain: opts && opts.yGain,
              onProgress: opts && opts.onProgress
            })
          : buildRelief(analysis, depthMap, {
              seed,
              legacy: !!(opts && opts.legacy),
              flat: !!(opts && opts.flat),
              ldi: !!(opts && opts.ldi),
              vg: opts && opts.vg
            }))
      : null;
    // 深度模式下玩家站在「挤出后的地面」上，不再是被乘 0.12 的程序化噪声
    const heightAt = relief ? relief.heightField : makeHeightField(analysis, 1);

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
    // 深度模式不叠天空幕：照片天空幕会在天际线处留一条水平分界线，
    // 反而把树冠/山脊切成两半（蓝调雪林 A/B 实测），天空交给程序化天穹。
    const skyCurtain = relief ? null : buildCurtain(analysis, {
      columns: 420, rows: 6, radius: 760, maxHeight: 700,
      seed: seed + 5, tint: 1.0, baseY: 0, fadeTop: 0.62,
      fixedHeight: 700, flat: true
    });

    // 360° 低频世界幕：补上照片没覆盖到的方向，也兜住断崖跳面留下的洞。
    // 默认关闭（?proxy=1 开启）：A/B 截图里它会在卡片边缘露出竖直色带与
    // 重复天际线尖峰，比 v1.11 的融色边界更突兀，故先作为实验特性保留。
    const worldBackdrop = (relief && opts && opts.proxy) ? buildWorldBackdrop(analysis) : null;
    // 兜底地面比重建区域大一倍：玩家可以走出重建区（移动不限制），
    // 但脚下仍然有地面，不会一步踏进纯雾。
    const ground = relief
      ? buildFallbackGround(analysis, rng, {
          A: (relief.bbox ? relief.bbox.A : 55) * 2,
          B: (relief.bbox ? relief.bbox.B : 100) * 2
        })
      : buildGround(analysis, heightAt);
    const horizonRing = (relief && opts && opts.ring) ? buildHorizonRing(analysis) : null;
    // 深度重建模式下，照片表面本身已经是有纵深的细节，不再叠加程序化植被
    const rocks = relief ? null : buildRocks(analysis, heightAt, rng, analysis.isSnow ? 70 : 120);
    const grass = (!relief && analysis.isGreen)
      ? buildGrass(analysis, heightAt, rng, 5200, 105, analysis.palette, true) : null;
    const shrub = (!relief && !analysis.isGreen && analysis.brightness > 0.42)
      ? buildGrass(analysis, heightAt, rng, 520, 100, analysis.palette, false) : null;
    const particles = buildParticles(analysis, rng, analysis.isSnow ? 1100 : 700);

    const tex = {
      photo: PWGL.createTexture(gl, analysis.canvas, { wrap: 'clamp', clampV: true, aniso: 8, flipY: false }),
      // 区域外地面用程序化色斑（A/B：改用照片底部延展会在边界外出现大面积
      // 照片纹理重复，比纯色更糟，已回退）。
      ground: PWGL.createTexture(gl, relief ? makeProceduralGroundTexture(analysis, seed + 3) : makeGroundTexture(analysis, seed + 3), { wrap: 'mirror', aniso: 8 }),
      sky: PWGL.createTexture(gl, makeSkyTexture(analysis), { wrap: 'clamp', clampV: true, flipY: false }),
      worldBackdrop: worldBackdrop ? PWGL.createTexture(gl, makeWorldBackdropTexture(analysis), { wrap: 'mirror', aniso: 4 }) : null,
      backdrop: (relief && relief.bg) ? PWGL.createTexture(gl, makeWorldBackdropTexture(analysis), { wrap: 'mirror', aniso: 4 }) : null,
      grass: PWGL.createTexture(gl, makeGrassTexture(mulberry32(seed + 77)), { wrap: 'clamp', clampV: true })
    };

    const m = {
      relief: relief ? PWGL.createMesh(gl, relief) : null,
      reliefSurround: (relief && relief.grid) ? PWGL.createMesh(gl, buildSurround(relief, { step: 3 })) : null,
      reliefBg: (relief && relief.bg) ? PWGL.createMesh(gl, relief.bg) : null,
      reliefBackdrop: (relief && relief.backdrop) ? PWGL.createMesh(gl, relief.backdrop) : null,
      worldBackdrop: worldBackdrop ? PWGL.createMesh(gl, worldBackdrop) : null,
      farRidge: farRidge ? PWGL.createMesh(gl, farRidge) : null,
      nearRidge: nearRidge ? PWGL.createMesh(gl, nearRidge) : null,
      sky: skyCurtain ? PWGL.createMesh(gl, skyCurtain) : null,
      ground: ground ? PWGL.createMesh(gl, ground) : null,
      horizon: horizonRing ? PWGL.createMesh(gl, horizonRing) : null,
      rocks: rocks ? PWGL.createMesh(gl, rocks) : null,
      grass: grass ? PWGL.createMesh(gl, grass) : null,
      shrub: shrub ? PWGL.createMesh(gl, shrub) : null
    };

    const worldInfo = {
      mode: relief ? 'depth' : 'legacy',
      eyeHeight: 1.7,
      // v1.14：深度重建不再限制移动半径（0 = 不限制），边界交给几何与雾
      walkRadius: relief ? 0 : 116
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
