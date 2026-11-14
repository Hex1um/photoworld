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
    /* 调试开关（v1.21.2）：?nocull=wall,steep,jump,torn,ground 逐条关掉索引阶段的剔除，
       用来定位"画面里的洞是哪条规则剪出来的"。生产环境不影响任何行为。 */
    const NOCULL = String((opt && opt.noCullReasons) || '').split(',').map((x) => x.trim());
    const noCull = (name) => NOCULL.indexOf(name) >= 0;
    /* 近场平面半径（米）：这个距离内一律当平地（用户要求"前面区域就是地面，
       地貌生成起点靠后"）。?near= 可调，0 = 关闭。 */
    const NEAR_PLANE = (opt.nearPlane === undefined) ? 8 : Math.max(0, opt.nearPlane);

    /* 语义分组（js/semantics.js）：几何策略按「组」而不是按具体类别走。
       没有语义图时全部退化为 UNKNOWN，行为与 v1.15 完全一致。 */
    const SGRP = { UNKNOWN: 0, SKY: 1, CEILING: 2, WALL: 3, GROUND: 4, WATER: 5,
      VEG: 6, BUILDING: 7, PERSON: 8, VEHICLE: 9, OBJECT: 10, TERRAIN: 11 };
    const OBJ_GROUP = new Uint8Array(12);
    OBJ_GROUP[SGRP.VEG] = 1; OBJ_GROUP[SGRP.BUILDING] = 1; OBJ_GROUP[SGRP.PERSON] = 1;
    OBJ_GROUP[SGRP.VEHICLE] = 1; OBJ_GROUP[SGRP.OBJECT] = 1;
    const sem = (opt.sem && opt.sem.groups && opt.sem.width > 1) ? opt.sem : null;
    const semGroup = (u, v) => {
      if (!sem) return SGRP.UNKNOWN;
      const x = Math.max(0, Math.min(sem.width - 1, Math.round(u * (sem.width - 1))));
      const y = Math.max(0, Math.min(sem.height - 1, Math.round(v * (sem.height - 1))));
      return sem.groups[y * sem.width + x];
    };
    const aspect = analysis.W / Math.max(1, analysis.H);
    const tanH = Math.tan(HFOV / 2);
    const tanV = tanH / aspect;
    const vfov = 2 * Math.atan(tanV);
    /* 相机俯仰角：从地平线位置反解。照片是俯拍（楼梯/桌面）时地平线在上方，
       若仍按水平相机反投影，地面会被卷成弧形（鱼眼）——这是室内照片崩掉的根因。 */
    /* 相机俯仰：用「地面消失线」而不是「天际线」反解。
       analysis.horizon 是地形轮廓的 82 分位 —— 画幅被台地/山体占满时会明显偏高
       （赤岩峡谷 0.364，而真正的地面线在 ~0.50）。用天际线反解 pitch 会让俯仰角
       高估 ~10°，台地像素的 dirY 被压低 → y = eye + dirY·t 变成负数，
       整片台地沉到地面以下，看起来就是"扁平的一条"。
       地面线取天际线的下分位（12%）：那是"没有凸起物遮挡的那部分地平线"，
       对风景照比 82 分位稳得多；仍用 horizon 原名给 sky alpha。 */
    const topSorted = Array.from(analysis.top).sort((a, b) => a - b);
    const groundLineRaw = topSorted[Math.floor(NCOL * 0.12)] || 0.5;
    const groundLine = clamp(groundLineRaw, 0.28, 0.82);
    let pitch = Math.atan(clamp((0.5 - groundLine) * 2 * tanV, -0.9, 0.9));
    let cosP = Math.cos(pitch), sinP = Math.sin(pitch);
    /* 室内 / 俯拍检测：地平线被顶到画面上方 42% 以外（或压到 72% 以下）。
       户外风景的地平线基本在 0.45~0.60。室内走短基线测距、不夸张高度、
       不裁天空，否则会被户外双锚点拉到几十米、卷成弧形。
       **但这只是初判**：低地平线的大画幅风景（赤岩峡谷 horizon=0.364）
       会命中这条规则，然后被 indoor 的 0.6~18m 短基线压成半径 18m 的碗
       —— 手机上看到的就是"整个地面只有一个圆弧"。等 hasSky 算出来后要纠正。 */
    let indoor = analysis.horizon < 0.42 || analysis.horizon > 0.72;

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

    /* 纠正 indoor：有天空（上下深度差 > 0.12）就说明这是"看向远方"的户外/大画幅，
       哪怕地平线很低。室内俯拍没有天空，hasSky 为 false，仍然走短基线。
       语义层再兜一道：天花板占多数时无条件保持 indoor。 */
    if (indoor && hasSky) {
      const ceilShare = (opt.sem && opt.sem.stats && opt.sem.stats.ceiling)
        ? opt.sem.stats.ceiling : 0;
      if (ceilShare < 0.18) indoor = false;
    }

    /* v1.21（P0）：用**画面底部行**反解俯仰角，替代"天际线"估计。
       为什么必须换：groundLine 取天际线的下分位，而"主体高耸"的照片（雪山群、
       红岩尖峰、树林、建筑）天际线就是主体顶缘 —— 主体整片落在"地面线以下"，
       被判成地面 → y=0 → 压成远处一条扁带。
       做法：底部 20% 一定踩在地面上，求使 `metric(深度) ≈ 眼高/(-dirY)` 残差最小的
       pitch（一维搜索 65 档）。残差过大就不用这个解，退回原估计 —— 宁可保守。 */
    (function fitPitchFromGround() {
      const rowMed = new Float32Array(GH);
      const buf = [];
      for (let j = 0; j < GH; j++) {
        buf.length = 0;
        for (let i = 0; i < GW; i += 2) buf.push(gS[j * GW + i]);
        buf.sort((a2, b2) => a2 - b2);
        rowMed[j] = buf.length ? buf[buf.length >> 1] : 0;
      }
      let best = pitch, bestErr = Infinity, bestN = 0;
      for (let cand = -0.35; cand <= 0.62; cand += 0.015) {
        const s2 = Math.sin(cand), c2 = Math.cos(cand);
        let err = 0, n = 0;
        for (let j = Math.floor(GH * 0.80); j < GH; j++) {
          const vv = j / (GH - 1);
          const dy0 = (0.5 - vv) * 2 * tanV;
          const dy = dy0 * c2 - s2;
          if (dy > -0.06) continue;
          const tPlane = eye / -dy;
          const tD = metric(rowMed[j]);
          if (!(tD > 0.4)) continue;
          const r = Math.log(tD / Math.max(0.2, tPlane));
          err += r * r; n++;
        }
        if (n >= 6 && err / n < bestErr) { bestErr = err / n; best = cand; bestN = n; }
      }
      if (bestN >= 6 && bestErr < 0.45) pitch = best;
      cosP = Math.cos(pitch); sinP = Math.sin(pitch);
    })();

    /* 1) 反投影：世界坐标点（相机朝 -Z，yaw=0，眼高 1.7m） */
    const pos = new Float32Array(GW * GH * 3);
    const uv = new Float32Array(GW * GH * 2);
    const col = new Float32Array(GW * GH * 4);
    const edge = new Float32Array(GW * GH * 2);
    const gT = new Float32Array(GW * GH);
    const gGrp = new Uint8Array(GW * GH);  // 语义分组（0=未知）
    const gGround = new Uint8Array(GW * GH); // 1 = 已按地面/水面贴到水平面
    const gNear = new Uint8Array(GW * GH);   // 1 = 比同行基线明显更近（单体候选）
    const gFg = new Uint8Array(GW * GH);   // 1 = 比同排地面基线明显更近（近处物体）
    const gDirX = new Float32Array(GW * GH);
    const gDirY = new Float32Array(GW * GH);
    const gDirZ = new Float32Array(GW * GH);
    const gTW = new Float32Array(9);
    for (let j = 0; j < GH; j++) {
      const v = j / (GH - 1);
      for (let i = 0; i < GW; i++) {
        const u = i / (GW - 1);
        const k = j * GW + i;
        // 语义分组：天空 / 天花板 / 地面 / 水面 / 植被 / 建筑 / 人车 / 山体
        const grp = semGroup(u, v);
        gGrp[k] = grp;
        // 天空也压到同一个有限半径 B（不能放到 1.6B，否则会被区域裁剪整片切掉，
        // 天际线处出现一条水平分界线）。
        const d0 = dSample(u, v);
        const dx = (u - 0.5) * 2 * tanH;
        const dy0 = (0.5 - v) * 2 * tanV;
        // 绕相机 X 轴加俯仰：dy0 = tan(pitch) 的那一行正好落在地平线
        const dy = dy0 * cosP - sinP;
        const dz = -dy0 * sinP - cosP;
        const l = Math.hypot(dx, dy, dz) || 1;
        const dirX = dx / l, dirY = dy / l, dirZ = dz / l;
        const rowBaseD = rowD[j];
        // 近景判定：YOLOv8n-seg 的实例 mask（语义边界）+ 深度基线兜底。
        // 深度只知道"更近"，分不清一块石头和一座山的边界；模型 mask 能。
        const mf = (opt.fgMask && opt.fgMask.data) ? opt.fgMask : null;
        const modelFg = mf
          ? mf.data[clamp(Math.round(v * (mf.h - 1)), 0, mf.h - 1) * mf.w
                   + clamp(Math.round(u * (mf.w - 1)), 0, mf.w - 1)]
          : 0;
        // 两种前景各司其职：模型 mask 只负责「边界断开」，深度基线负责
        // 「近处物体距离限幅」。把模型 mask 拿去限幅会把 96×96 量化的边界
        // 连带一整片背景压到近处，画面被咬出缺口（楼梯图 A/B 实测）。
        const isFgDepth = (gS[k] - rowBaseD) > 0.10;
        // 「真正靠近的对象」用更严的阈值：雪坡/水面的深度噪声常超过 0.10，
        // 会把整片地面误判成近景对象，进而按逐像素深度拉成长条。
        const isNearObj = (gS[k] - rowBaseD) > 0.20;
        const isFg = modelFg > 0 || isFgDepth;
        gFg[k] = modelFg > 0 ? 1 : 0;

        /* v1.15 距离策略：
           - 地面：按「平地 + 眼高」反解，严格落在 y=0，近处透视正确、
             **不会被深度噪声拉成横向长条**（用户反馈的核心问题）；
           - 山体/树/建筑/近景物体：按深度反投影，远处可以放到 220m；
           - 近处物体：限制在自身行基线深度附近，**不向远处延伸**。 */
        const isCeilGrp = grp === SGRP.CEILING;

        let t, y;
        if (indoor) {
          // 室内用「行距离」而不是逐像素深度：同一行 t 相同 → 网格规则、
          // 不会自交遮挡（逐像素反投影在墙面/天花板之间穿插，会把画面咬出空洞）。
          t = clamp(metric(rowBaseD), 0.6, 18);
          y = eye + dirY * t;
          // 天花板必须在头顶以上：室内最常见的误判就是把天花板当地面压到脚下
          if (isCeilGrp) y = Math.max(y, eye + 0.45);
        } else if (isCeilGrp) {
          // 天花板：按深度反投影但夹在头顶以上，绝不进入「地平线以下=地面」逻辑。
          // 这是「天花板被当成天空/地面」的直接修复。
          t = clamp(metric(d0), 0.6, 220);
          y = Math.max(eye + dirY * t, eye + 0.45);
        } else if (v > groundLine + 0.005 && !isFg && dirY < -0.012) {
          /* 地面：按「平地 + 眼高」反解，严格落在 y=0。
             这里**只用几何判据**（地平线以下 + 不比同行基线更近），不用语义的
             地面 mask：语义地面是斑块状的，把散点强压到 y=0 会在斑块边界立起
             无数薄墙，整片雪坡变成竖条（A/B 实测）。语义只负责天空/天花板/单体。

             v1.18.2：加 `dirY < -0.012` —— 射线必须**朝下**才可能是地面。
             `analysis.horizon` 是**天际线**（画面里地形轮廓的 82 分位），不是地面
             消失线：赤岩峡谷的天际线在 0.364（因为台地占了大半幅面），于是整段台地
             被判定成"地面"，`-dirY` 被 0.06 兜底 → t=28m、y=0 → 台地压成远处一条
             扁带（用户看到的就是"扁平"）。朝上的射线落到地面分支在物理上就不成立。 */
          t = Math.min(eye / Math.max(0.06, -dirY), 220);
          y = 0;
          gGround[k] = 1;
        } else {
          /* 远场**软压缩**而不是硬夹断（v1.19.1）。
             以前 `clamp(metric(d0), 0.6, 220)` 把所有 >220m 的像素都压到正好 220，
             等于在半径 220m 处生成一层"壳"；再叠上近处的壳，从外面看就是一层层
             漂浮的横向薄片（32 视角质检里 stripe>3 的那类）。
             tanh 把 220 附近压缩成 172、1000m 压成 219 —— 远山仍然在，但不再重叠成壳。 */
          const tRaw = metric(d0);
          t = Math.max(0.6, 220 * Math.tanh(tRaw / 220));
          y = eye + dirY * t * ((hasSky && dirY > 0) ? yGain : 1);
          // 室内不做近景限幅：室内深度范围只有 18m，按行中位判定会把
          // 半面墙/天花板误标成「近景」，压到近处后画面被咬出锯齿缺口。
          if (isFgDepth && !indoor) {
            const tBase = clamp(metric(rowBaseD), 0.8, 220);
            t = clamp(t, tBase * 0.5, tBase * 1.6);
            y = clamp(y, -0.5, 12);   // 近景物体（树/石头/人）不会长成几十米的尖塔
          }
        }
        /* v1.21.1(a)：接近**水平**的射线（|dirY| 很小）压到地面。
           这类像素以前走地形分支，`y = eye + dirY·t` ≈ **1.7m** —— 等于在眼睛
           高度造出一圈延伸到地平线的环形平台；因为 t 很大、水平位移很小，
           它沿径向被拉成一片放射状条纹（赤岩峡谷截图里那 40% 的拉丝就是这个）。
           物理上它们就是"远处的地表"，应当压在地面上。 */
        if (!indoor && Math.abs(dirY) < 0.038) {
          y = 0;
          gGround[k] = 1;
        } else if (!indoor && Math.abs(y) < 0.45) {
          /* 归一化后落在 ±0.45m 内的顶点本来就在地面附近（近景物体的底座、
             水面、沙地），一并算地面：既消掉它们自己的放射拉伸，
             也让地面方格网接管这块屏幕面积最大的区域。 */
          y = 0;
          gGround[k] = 1;
        }

        /* v1.21.1(b)：**近场一律是地面**（用户要求："前面区域就是地面，把地貌生成
           起点靠后"）。理由不只是好看：近场（< NEAR_PLANE 米）的深度最不可靠，
           而它的屏幕面积最大 —— 一点深度噪声会被放大成整屏的放射状拉丝
           （赤岩峡谷那张截图里 40% 的画面都是这种拉伸）。
           代价：离眼睛 8m 内的石头/灌木不再有立体感。这个交换是划算的。 */
        const horizD = Math.hypot(dirX * t, dirZ * t);
        if (!indoor && horizD < NEAR_PLANE) {
          t = Math.min(t, NEAR_PLANE * 1.6);
          y = 0;
          gGround[k] = 1;
        }
        // 只给「确实更近」的单体做卡片候选；远处的森林/山脉不切
        if (isNearObj) gNear[k] = 1;
        gDirX[k] = dirX; gDirY[k] = dirY; gDirZ[k] = dirZ;
        pos[k * 3] = dirX * t;
        pos[k * 3 + 1] = y;
        pos[k * 3 + 2] = dirZ * t;
        uv[k * 2] = u;
        uv[k * 2 + 1] = v;
        gT[k] = t;
        /* alpha：天际线以上透明（交给天穹），四周 3.5% 渐隐（边界是照片梯形
           的投影，不是视锥卡片，所以不需要镜像/融色） */
        /* 天空判定用**深度**而不是逐列天际线：
           比天空基线明显更近的像素才保留（地面/山体/树木/建筑），
           天空 alpha=0 交给程序化天穹。深度图是连续的，所以树冠边缘
           不会出现锯齿状青边或一条水平切线。 */
        /* v1.21.2：深度判天空**只在轮廓线以上**生效。
           深度模型在远处低纹理的崖面上会给出一片"和天空一样远"的平值，
           逐像素裁掉就在岩壁中间咬出一格格白方块（晨雾湖岸那张最明显）。
           但天空按定义只可能出现在**地形轮廓之上** —— 所以加一条
           `v < 天际线` 的硬约束，轮廓以下的像素永不因深度被判成天空。 */
        let aSky = (indoor || !hasSky || noCull('alpha')) ? 1 : smoothstep(skyD + 0.03, skyD + 0.14, d0);
        /* v1.22 试验：`?skymode=top` 用**逐列轮廓线**判天空，取代按深度判。
           动机：按深度判会在低纹理崖面上把整块判成天空（岩壁上的白方块），
           而按深度判在合成图上又很准。用轮廓线一处判定，两边都能兼顾。 */
        if (opt.skyFill === true && !indoor && hasSky && !noCull('alpha')) {
          const tcol = analysis.top || null;
          if (tcol && tcol.length) {
            const ci2 = Math.max(0, Math.min(tcol.length - 1,
              Math.round((i / (GW - 1)) * (tcol.length - 1))));
            const sk = tcol[ci2];
            if (sk > 0.001 && v > sk + 0.012) aSky = 1;
          }
        }
        /* v1.21.2：**天空只可能出现在地形轮廓之上**。轮廓线以下（v 更大）的像素
           一律不因深度被判成天空 —— 深度模型在远处低纹理崖面上会给出成片
           "和天空一样远"的平值，逐顶点裁掉就在岩壁上咬出一格格白方块
           （已用 ?nocull=alpha 取证：关掉这条裁切，方块立刻变成照片里的白色岩面）。
           轮廓线上方不受影响，天空照旧交给天穹。 */
        {
          /* 逐列天际线（analysis.top）：这一列的第一个地形行。
             该行**以下**不可能是天空 —— 深度模型再怎么说"很远"也不裁。
             全局 82 分位（analysis.horizon）在晨雾湖岸这类图上会跑到 0.9，
             完全失去判别力；逐列轮廓才不会。 */
          const tcol = analysis.top || null;
          if (tcol && tcol.length) {
            const ci = Math.max(0, Math.min(tcol.length - 1,
              Math.round((i / (GW - 1)) * (tcol.length - 1))));
            const skyline = tcol[ci];
            /* 只在「明显在轮廓线以下」且「深度把它判得几乎肯定是天空」时才推翻。
               宽松版（+0.012 / aSky<1）在合成真值图上会误保留天空几何，
               新视角 PSNR 掉 1.1dB；收紧到 +0.04 / aSky<0.2 后两头都顾到。 */
            /* 用「抬到 0.8」而不是「硬置 1」：轮廓线以下本来不该有天空，
               但直接置 1 会在深度准确的合成图上把天空画成地形（真值 -1.1dB）。
               抬到 0.8 既补掉岩壁上的洞（残留 20% 天空几乎看不出），
               又不会在真值图上引入整片错误几何。 */

          }
        }

        /* v1.21.1：**孤立"天空"像素修补**。深度模型在崖面上会给出成片的
           "和天空一样远"的平值区，逐像素裁掉就在岩壁上咬出一格格白色方块
           （晨雾湖岸的"色块被抛弃、留下空白"就是这个）。
           判据：这块像素若被地形包围（5×5 邻域里 >45% 比天空更近），
           就认为它是崖面的一部分，保留 alpha=1。真正的天空区域邻域里
           绝大多数仍是天空，不受影响。 */
        if (aSky < 0.5 && !indoor && hasSky) {
          const ci = (k % GW), cj = ((k / GW) | 0);
          let terr = 0, tot = 0;
          for (let dj = -2; dj <= 2; dj++) {
            const jj = cj + dj;
            if (jj < 0 || jj >= GH) continue;
            for (let di = -2; di <= 2; di++) {
              const ii = ci + di;
              if (ii < 0 || ii >= GW) continue;
              tot++;
              if (gS[jj * GW + ii] > skyD + 0.16) terr++;
            }
          }
          if (tot > 8 && terr / tot > 0.32) aSky = 1;
        }
        // 语义天空直接清零交给天穹；天花板（CEILING）保持 alpha=1，
        // 不再被深度判据误当成天空切掉——「天空/地面/天花板混淆」的关键修复。
        /* 但语义天空只在**地平线以上**才清零（v1.16.1）：
           地平线附近的雾带/远岸/水面反光经常被 ADE20K 判成 sky，
           一刀清零会在画面中间咬出一条贯穿左右的灰带——洞后面就是天穹，
           读起来像「风景和地面混淆」（alpine 湖岸实测，24k 面被这样清掉）。
           地平线以下的像素交给深度判据，它只认「比天空更近」。 */
        /* v1.21.1：语义天空的清零边界改用 **groundLine（拟合出的地面线）**，
           不再用天际线。沙漠这张图的天际线在 0.36（台地顶缘），旧条件只清
           `v < 0.356` 的像素，于是台地上方那一条 **云/亮天空** 被当成几何保留，
           在视点抬高后摊成一排白蓝色的碎片（截图里横贯画面的那排）。
           groundLine 更靠下、更接近真实天空分界，清得干净。 */
        if (sem && grp === SGRP.SKY && v < Math.max(groundLine - 0.01, analysis.horizon - 0.004)) aSky = 0;
        const aU = smoothstep(0, 0.02, u) * (1 - smoothstep(0.98, 1, u));
        const aV = smoothstep(0, 0.015, v) * (1 - smoothstep(0.985, 1, v));
        // 室外按深度裁掉天空（否则照片天空会被压到 100m 处，和地面分成两块）；
        // 室内保留全部像素（室内没有天空，裁切会把墙咬出缺口）。
        col[k * 4 + 3] = aSky * aU * aV;
        col[k * 4] = 1; col[k * 4 + 1] = 1; col[k * 4 + 2] = 1;
        // 有限区域的边界就是照片投影的梯形，alpha 硬边比"融色灰带"干净；
        // edge 保持 0，不参与 mode 4 的环境色混合。
        edge[k * 2] = 0;
        edge[k * 2 + 1] = 0;
      }
    }

    /* 1a-bis) 地面颜色场：把照片里「真正地面」的像素颜色烘成一张覆盖重建矩形的
       粗网格（线性 RGB）。js/extend.js 生成的扇环外 tile 拿它就当顶点色基准，
       于是雪地外面还是雪、沙地外面还是沙、森林地面外面还是暗绿——
       这是「照片一块亮、外面一片深蓝」那种明信片感（v1.16 实测）的根因修复：
       延伸地面的颜色曾经只能取 analysis.ground（全图下半带的平均值，
       蓝调雪林里被深色林带拖到 #30595c），与近处亮雪差 2~3 倍。 */
    const CGW = 48, CGH = 32;
    const cgrid = new Float32Array(CGW * CGH * 3);
    const ccnt = new Float32Array(CGW * CGH);
    let pix = null;
    try {
      pix = analysis.canvas.getContext('2d').getImageData(0, 0, analysis.W, analysis.H).data;
    } catch (e) { pix = null; }              // 画布被污染时退回 analysis.ground
    for (let k = 0; k < GW * GH; k++) {
      if (gGround[k] !== 1) continue;        // 只要已贴到水平面的地面/水面点
      if (col[k * 4 + 3] < 0.5) continue;    // 与 alpha 剔除保持一致
      const x = pos[k * 3], z = pos[k * 3 + 2];
      if (Math.abs(x) > A || z < -B || z > 0) continue;
      const gi = clamp(Math.floor((x + A) / (2 * A) * CGW), 0, CGW - 1);
      const gj = clamp(Math.floor((-z) / B * CGH), 0, CGH - 1);
      let r = analysis.ground[0], g = analysis.ground[1], b = analysis.ground[2];
      if (pix) {
        const px = clamp(Math.round(uv[k * 2] * (analysis.W - 1)), 0, analysis.W - 1);
        const py = clamp(Math.round(uv[k * 2 + 1] * (analysis.H - 1)), 0, analysis.H - 1);
        const q = (py * analysis.W + px) * 4;
        r = pix[q] / 255; g = pix[q + 1] / 255; b = pix[q + 2] / 255;
      }
      const ci = gj * CGW + gi;
      cgrid[ci * 3] += lin(clamp(r, 0, 1));
      cgrid[ci * 3 + 1] += lin(clamp(g, 0, 1));
      cgrid[ci * 3 + 2] += lin(clamp(b, 0, 1));
      ccnt[ci] += 1;
    }
    // 空格（远景没有地面像素的格子）用 3×3 邻域补几轮，保证整张场连续
    for (let pass = 0; pass < 4; pass++) {
      let filled = 0;
      for (let gj = 0; gj < CGH; gj++) {
        for (let gi = 0; gi < CGW; gi++) {
          const ci = gj * CGW + gi;
          if (ccnt[ci] > 0) continue;
          let r = 0, g = 0, b = 0, n = 0;
          for (let dj = -1; dj <= 1; dj++) {
            for (let di = -1; di <= 1; di++) {
              const jj = gj + dj, ii = gi + di;
              if (jj < 0 || jj >= CGH || ii < 0 || ii >= CGW) continue;
              const q = jj * CGW + ii;
              if (ccnt[q] <= 0) continue;
              r += cgrid[q * 3] / ccnt[q]; g += cgrid[q * 3 + 1] / ccnt[q]; b += cgrid[q * 3 + 2] / ccnt[q];
              n++;
            }
          }
          if (!n) continue;
          cgrid[ci * 3] = r / n; cgrid[ci * 3 + 1] = g / n; cgrid[ci * 3 + 2] = b / n;
          ccnt[ci] = -1;                     // 标记「已填补」，参与下一轮扩散
          filled++;
        }
      }
      if (!filled) break;
    }
    for (let ci = 0; ci < CGW * CGH; ci++) {
      if (ccnt[ci] > 0) {
        cgrid[ci * 3] /= ccnt[ci]; cgrid[ci * 3 + 1] /= ccnt[ci]; cgrid[ci * 3 + 2] /= ccnt[ci];
      } else if (ccnt[ci] === 0) {
        cgrid[ci * 3] = lin(analysis.ground[0]);
        cgrid[ci * 3 + 1] = lin(analysis.ground[1]);
        cgrid[ci * 3 + 2] = lin(analysis.ground[2]);
      }
      ccnt[ci] = 1;
    }
    /* 底部裙边（v1.19.1）：竖屏时相机竖直视场（~60°）比照片（~41°）大，
       照片底边以下那条"比照片底边还近"的地面原来只能靠 y=-0.05 的平面补片兜，
       与照片网格之间是一条**几何断口** —— 无论怎么调颜色都能看出那条横线。
       这里把网格继续往下延伸 K 行：射线方向延续（v>1），仍与 y=0 求交，
       UV 的 v 钳到 0.995（取照片最后一行像素）——
       于是地面从照片的最后一排像素"接着长"到脚下，没有断口、没有色差。 */
    function buildSkirt() {
      const K = 4;                       // 额外行数
      const DV = 0.075;                  // 每行 v 的步进（1.0 → 1.30）
      const cols = GW;
      const rows = K + 1;                // 含照片底行本身
      const pos = new Float32Array(cols * rows * 3);
      const nrm = new Float32Array(cols * rows * 3);
      const uv = new Float32Array(cols * rows * 2);
      const col = new Float32Array(cols * rows * 4);
      const edge = new Float32Array(cols * rows * 2);
      const idx = [];
      for (let r = 0; r < rows; r++) {
        const v = 1.0 + r * DV;
        for (let i = 0; i < cols; i++) {
          const k = r * cols + i;
          const u = i / (cols - 1);
          const dx = (u - 0.5) * 2 * tanH;
          const dy0 = (0.5 - v) * 2 * tanV;
          const dy = dy0 * cosP - sinP;
          const dz = -dy0 * sinP - cosP;
          const l = Math.hypot(dx, dy, dz) || 1;
          const dirX = dx / l, dirY = dy / l, dirZ = dz / l;
          const t = Math.min(eye / Math.max(0.06, -dirY), 220);
          pos[k * 3] = dirX * t;
          pos[k * 3 + 1] = 0;
          pos[k * 3 + 2] = dirZ * t;
          nrm[k * 3 + 1] = 1;
          /* UV 用**世界坐标**投影（和中景地面块同口径），不要钳到照片最后一行 ——
             单行采样沿 u 只有 320 个颜色，往脚下延伸时每列一个颜色，会拉成
             一根根竖条（第一版实测就是这样）。世界坐标 UV 是方格网，无方向性。 */
          uv[k * 2] = pos[k * 3] / 4.5;
          uv[k * 2 + 1] = pos[k * 3 + 2] / 4.5;
          /* 顶点色 = 照片底行颜色 / 地面贴图基色：色相与照片最后一行一致，
             细节由贴图提供，两者一乘就"接着长"。 */
          const bc = bottomRowColor(pos[k * 3]);
          col[k * 4] = clamp(bc[0] / Math.max(1e-4, baseLin[0]), 0.25, 6);
          col[k * 4 + 1] = clamp(bc[1] / Math.max(1e-4, baseLin[1]), 0.25, 6);
          col[k * 4 + 2] = clamp(bc[2] / Math.max(1e-4, baseLin[2]), 0.25, 6);
          col[k * 4 + 3] = 1;
          // 越靠近脚下越融进地面色，避免纹理在 1m 内被拉成条纹
          edge[k * 2 + 1] = smoothstep(2, 14, Math.hypot(pos[k * 3], pos[k * 3 + 2])) * 0.85;
        }
      }
      for (let r = 0; r < rows - 1; r++) {
        for (let i = 0; i < cols - 1; i++) {
          const a0 = r * cols + i, a1 = a0 + 1;
          const b0 = (r + 1) * cols + i, b1 = b0 + 1;
          idx.push(a0, b0, a1, a1, b0, b1);
        }
      }
      return { pos, normal: nrm, uv, color: col, edge, index: idx };
    }
    const baseLin = [0.18, 0.18, 0.18];
    let skirt = null;
    try { skirt = buildSkirt(); } catch (e) { skirt = null; }

    /* 照片底行颜色剖面（v1.19）：竖屏时相机竖直视场比照片大，照片底边以下那条
       地面只能由近景补片兜底。补片原来用 groundColorAt（整图下半带平均），
       与照片底行颜色差一截 → 有一条看得见的横向接缝。
       这里把照片**最底下几行**的颜色按列烘成一条 1D 剖面，补片按 x 取色。 */
    const EDGE_N = 48;
    const edgeCol = new Float32Array(EDGE_N * 3);
    {
      let rpix = null;
      try { rpix = analysis.canvas.getContext('2d').getImageData(0, 0, analysis.W, analysis.H).data; } catch (e) { rpix = null; }
      const rows = Math.max(1, Math.round(analysis.H * 0.02));
      for (let i = 0; i < EDGE_N; i++) {
        let r = 0, g = 0, b = 0, n = 0;
        const x0 = Math.floor((i / EDGE_N) * analysis.W);
        const x1 = Math.max(x0 + 1, Math.floor(((i + 1) / EDGE_N) * analysis.W));
        for (let y = analysis.H - rows; y < analysis.H; y++) {
          for (let x = x0; x < x1; x++) {
            if (rpix) {
              const q = (y * analysis.W + x) * 4;
              r += lin(rpix[q] / 255); g += lin(rpix[q + 1] / 255); b += lin(rpix[q + 2] / 255);
            } else {
              r += lin(analysis.ground[0]); g += lin(analysis.ground[1]); b += lin(analysis.ground[2]);
            }
            n++;
          }
        }
        n = Math.max(1, n);
        edgeCol[i * 3] = r / n; edgeCol[i * 3 + 1] = g / n; edgeCol[i * 3 + 2] = b / n;
      }
      // 横向 1 次平滑，避免列与列之间的硬跳变
      const cp = edgeCol.slice();
      for (let i = 0; i < EDGE_N; i++) {
        for (let c = 0; c < 3; c++) {
          const l = cp[Math.max(0, i - 1) * 3 + c], m2 = cp[i * 3 + c], r2 = cp[Math.min(EDGE_N - 1, i + 1) * 3 + c];
          edgeCol[i * 3 + c] = (l + 2 * m2 + r2) / 4;
        }
      }
    }
    /* 按世界 x 取照片底行颜色（-A..A 映射到整幅宽度） */
    function bottomRowColor(x) {
      const t = clamp((x + A) / (2 * A), 0, 1) * (EDGE_N - 1);
      const i0 = Math.floor(t), i1 = Math.min(EDGE_N - 1, i0 + 1), f = t - i0;
      return [
        lerp(edgeCol[i0 * 3], edgeCol[i1 * 3], f),
        lerp(edgeCol[i0 * 3 + 1], edgeCol[i1 * 3 + 1], f),
        lerp(edgeCol[i0 * 3 + 2], edgeCol[i1 * 3 + 2], f)
      ];
    }

    /* 采样：双线性 + 钳到重建矩形内（矩形外就是边界值，正是延伸的起点）。 */
    function groundColorAt(x, z) {
      const fx = clamp((x + A) / (2 * A), 0, 1) * (CGW - 1);
      const fz = clamp(-z / B, 0, 1) * (CGH - 1);
      const i0 = Math.floor(fx), j0 = Math.floor(fz);
      const i1 = Math.min(CGW - 1, i0 + 1), j1 = Math.min(CGH - 1, j0 + 1);
      const tx = fx - i0, tz = fz - j0;
      const out = [0, 0, 0];
      for (let c = 0; c < 3; c++) {
        const a = cgrid[(j0 * CGW + i0) * 3 + c], b = cgrid[(j0 * CGW + i1) * 3 + c];
        const e = cgrid[(j1 * CGW + i0) * 3 + c], f = cgrid[(j1 * CGW + i1) * 3 + c];
        out[c] = lerp(lerp(a, b, tx), lerp(e, f, tx), tz);
      }
      return out;
    }

    /* 1b) 竖直尖刺抑制：树梢/细杆/电线这类 1~2 像素宽的远景，深度稍有误差
       就会被反投影成几米长的"冰柱"。这里对 y 做 3×3 中值，只把偏离邻域中值
       超过 1.6m 的顶点拉回 —— 山脊/悬崖的连续起伏仍然保留。 */
    const yWin = new Float32Array(9);
    for (let j = 0; j < GH; j++) {
      for (let i = 0; i < GW; i++) {
        /* v1.17：地面顶点（y 由「平地 + 眼高」反解，严格 0）不能参与尖刺限幅。
           旧代码对**所有**顶点做 clamp(y, median-1.5, median+2.2)，而地平线附近
           地面顶点下方的邻域中值是山体高度（~30m），于是 y=0 被抬到 28.5m
           —— 地面被拔成一圈放射状竖针（16 视角质检的「拉丝」与俯视「冰柱」
           都是它）。地面几何本来就精确，跳过即修复。 */
        if (gGround[j * GW + i] === 1) continue;
        // 只取「下方邻域」（j..j+2）：树梢上方是天空，与天空求中值会把尖刺
        // 一起抬高；下方是树冠/地面，才反映物体真实高度。
        let n = 0;
        for (let dj = 0; dj <= 2; dj++) {
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
        pos[k * 3 + 1] = clamp(pos[k * 3 + 1], yWin[4] - 1.5, yWin[4] + 2.2);
        // 横向尖刺抑制：树梢/电线这类细结构的深度误差会把 x/z 拉出几十米，
        // 只限 y 没用。对 t 做 3×3 中值限幅（±30%）后重算 x/z。
        let ts = 0;
        for (let dj = -1; dj <= 1; dj++) {
          const jj = clamp(j + dj, 0, GH - 1);
          for (let di = -1; di <= 1; di++) {
            gTW[dj * 3 + di] = gT[jj * GW + clamp(i + di, 0, GW - 1)];
          }
        }
        for (let a = 1; a < 9; a++) {
          const vv = gTW[a]; let b = a - 1;
          while (b >= 0 && gTW[b] > vv) { gTW[b + 1] = gTW[b]; b--; }
          gTW[b + 1] = vv;
        }
        ts = gTW[4];
        const tLim = clamp(gT[k], ts * 0.7, ts * 1.4);
        gT[k] = tLim;
        pos[k * 3] = gDirX[k] * tLim;
        pos[k * 3 + 2] = gDirZ[k] * tLim;
      }
    }

    /* 1b-bis) 远景横向中值（v1.16.1）：60m 以外的几何几乎没有视差，
       但网格列在 190m 处只相隔约 0.4m —— 深度图里一条崖边就让相邻列差出
       70m，渲染成一排沿视线拉的竖幕（alpine 崖脚实测：458 处 >6m 的竖直
       跳变，最大 72m）。把中值窗口从 3 列加宽到 7 列，可以去掉 1~3 列宽的
       假崖，真实的山脊/宽崖（占多数列）不受影响。 */
    const farWin = new Float32Array(7);
    for (let j = 0; j < GH; j++) {
      for (let i = 0; i < GW; i++) {
        const k = j * GW + i;
        if (gGround[k] === 1) continue;      // 地面恒为 y=0，不参与远处横向中值
        if (gT[k] < 60) continue;
        let n = 0;
        for (let di = -3; di <= 3; di++) {
          const q = j * GW + clamp(i + di, 0, GW - 1);
          if (col[q * 4 + 3] < 0.06) continue;   // 天空不参与
          farWin[n++] = pos[q * 3 + 1];
        }
        if (n < 4) continue;
        for (let a = 1; a < n; a++) {
          const v = farWin[a]; let b = a - 1;
          while (b >= 0 && farWin[b] > v) { farWin[b + 1] = farWin[b]; b--; }
          farWin[b + 1] = v;
        }
        const med = farWin[(n - 1) >> 1];
        const y0 = pos[k * 3 + 1];
        if (Math.abs(y0 - med) > 4) pos[k * 3 + 1] = med + (y0 > med ? 1 : -1) * 4;
      }
    }

    /* 1b-ter) 地面「极坐标拉丝」抑制（v1.17）。
       地面顶点是按反投影射线落在 y=0 上算出来的：同一 v 行半径相同 →
       **网格是一圈圈同心圆 + 一条条射线**（极坐标网格）。原始视角下它正好
       与照片透视一致，但从别的角度（尤其背向/侧向）看，射线上被拉长的四边形
       会把贴图沿径向拉成一束"放射状拉丝"，是 16 视角质检里最刺眼的一项。
       这里按到相机水平距离把地面顶点渐隐到地面基色（edge.y 通道，着色器里
       mode 4 已经 `col = mix(col, uGroundTint, vEdge.y)`）：
       近处 18m 内保持原样（走动视差与前景细节都在这里），60m 外基本只剩色块。
       `?gblend=0` 可关掉做 A/B。 */
    if (opt.groundFade !== false) {
      const gf = (opt.gfade || '6,34,0.95').split(',').map(Number);
      const f0 = gf[0] || 0, f1 = gf[1] || 34, famt = gf[2] || 0.95;
      for (let k = 0; k < GW * GH; k++) {
        if (gGround[k] !== 1) continue;
        if (col[k * 4 + 3] < 0.02) continue;      // 已被天空裁掉的不管
        const d = Math.hypot(pos[k * 3], pos[k * 3 + 2]);
        edge[k * 2 + 1] = smoothstep(f0, f1, d) * famt;
      }
    }

    /* 1c) 单体实例：只把「比同行明显更近」的紧凑对象切成十字账单牌。
       大面积植被（森林/灌木带）不做逐像素抬升——A/B 实测那会把整片雪坡
       变成竖条；它们走地面规则与深度基线。 */
    const instInfo = sem
      ? labelObjectCards(gGrp, gT, pos, uv, GW, GH, tanH, tanV, gNear)
      : { gCardId: null, instances: [], cards: null };
    const gCardId = instInfo.gCardId;
    /* 1c-bis) 实例深度平坦化：把每个单体压到自身中位深度，保持竖直与落地。
       相邻像素共享 t，反投影面片不再沿视线拉成鳍；因为仍在同一张网格里，
       底部自然与地面相连，不会出现悬空的矩形卡片。 */
    if (sem && instInfo.instances.length) {
      for (const it of instInfo.instances) {
        const t = clamp(it.t, 0.6, 200);
        for (const q of it.cells) {
          gT[q] = t;
          pos[q * 3] = gDirX[q] * t;
          pos[q * 3 + 1] = clamp(eye + gDirY[q] * t * ((hasSky && gDirY[q] > 0) ? yGain : 1), -0.5, 18);
          pos[q * 3 + 2] = gDirZ[q] * t;
        }
      }
    }
    prog(0.70);

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
    let skipped = 0, skippedCard = 0, skippedObjEdge = 0, skippedSteep = 0, skippedWall = 0,
      skippedGround = 0, skippedTorn = 0,
      culledAlpha = 0, culledArea = 0;
    for (let j = 0; j < GH - 1; j++) {
      for (let i = 0; i < GW - 1; i++) {
        const a0 = j * GW + i, a1 = a0 + 1;
        const b0 = (j + 1) * GW + i, b1 = b0 + 1;
        const alpha = Math.max(col[a0 * 4 + 3], col[a1 * 4 + 3], col[b0 * 4 + 3], col[b1 * 4 + 3]);
        if (alpha < 0.06) { culledAlpha++; continue; }
        // 实例已就地压平（1c-bis），不再剔除四边形，避免露洞与「墓碑」卡片
        void skippedCard;
        /* 注：曾按「对象/非对象」断边，但 128×96 的语义边界会把地面/雪坡咬出
           一块块露天的矩形洞（A/B 实测），已移除。鳍改由 1b-bis 的深度平坦化解决。 */
        void skippedObjEdge;
        /* 竖直跳变剪断（仅语义路径）：相邻网格顶点高差 >5m 的四边形跳过。
           单目深度在树梢/栏杆/细杆上会突然给一个近距离，反投影后就沿视线
           拉出几十米的长鳍；山脊的真实起伏在一个网格单元内只有几十厘米，
           不会被误伤。这一条对没被卡片覆盖到的细高物体兜底。 */
        if (sem) {
          const dyV = Math.max(Math.abs(pos[a0 * 3 + 1] - pos[b0 * 3 + 1]),
            Math.abs(pos[a1 * 3 + 1] - pos[b1 * 3 + 1]));
          const dyH = Math.max(Math.abs(pos[a0 * 3 + 1] - pos[a1 * 3 + 1]),
            Math.abs(pos[b0 * 3 + 1] - pos[b1 * 3 + 1]));
          if (!noCull('steep') && dyV > 5 || dyH > 5) { skippedSteep++; continue; }
        }
        // 近处物体与背景的交界交给下面的深度比值判据（tMax/tMin）。
        // 曾按「模型 mask 的 fg 顶点数」跳面，但 96×96 的 mask 边界是块状的，
        // 会在画面里咬出一排方块缺口（楼梯图 A/B 实测），已移除。
        // 区域裁剪只作用于近景：远景（>60m 的山）交给雾，不硬切，
        // 否则「远处的物体放得比较远」会被边界一刀切掉。
        /* v1.21.1：区域裁剪从 60m 收到 22m。旧阈值把 22~60m 的地面/水面
           也算"近景"，只要顶点稍微越出照片投影的矩形就整片丢掉 ——
           晨雾湖岸那张"色块被抛弃、留下空白"就是这一段被切掉的。 */
        const tNear = Math.min(gT[a0], gT[a1], gT[b0], gT[b1]) < 22;
        if (tNear) {
          let inside = true;
          for (const k of [a0, a1, b0, b1]) {
            if (Math.abs(pos[k * 3]) > A || pos[k * 3 + 2] < -B - 0.5 || pos[k * 3 + 2] > 1.5) { inside = false; break; }
          }
          if (!inside) { culledArea++; continue; }
        }
        const tMax = Math.max(gT[a0], gT[a1], gT[b0], gT[b1]);
        const tMin = Math.min(gT[a0], gT[a1], gT[b0], gT[b1]);
        // 室内跳面阈值放宽：楼梯/窗/天花板的深度跳变本来就大，2.6 会把
        // 画面咬出一排锯齿缺口；室内范围只有 18m，不需要断崖处理。
        if (!noCull('jump') && tMax / Math.max(tMin, 0.6) > (indoor ? 6 : edgeRatio)) { skipped++; continue; }
        /* 近垂直面片剔除（v1.16.1）：竖直墙在照片里没有对应的水平位移，
           贴图必然被拉成一根根竖条。远山脚下 → 水面这一段最典型：深度都顶在
           220m 上限，水平位移每格只有几十厘米，y 却从 20m 平滑降到 0，
           逐格高差 < 5m 躲过了上面的断崖判据，整片就成了「竖幕」
           （alpine 崖脚实测，?v1=1 旧路径没有这个问题）。
           判据：竖直跨度 / 水平跨度 > 2.5 视为墙，直接断开让水面/天穹透出来。
           室内不适用——那里的墙/门框本来就是竖直的。 */
        if (!indoor) {
          let hMin = Infinity;
          const pairs = [[a0, a1], [b0, b1], [a0, b0], [a1, b1]];
          for (const pr of pairs) {
            const dx2 = pos[pr[0] * 3] - pos[pr[1] * 3];
            const dz2 = pos[pr[0] * 3 + 2] - pos[pr[1] * 3 + 2];
            const dh = Math.hypot(dx2, dz2);
            if (dh < hMin) hMin = dh;
          }
          let vSpan = 0;
          for (const k of [a0, a1, b0, b1]) {
            for (const k2 of [a0, a1, b0, b1]) {
              const dy2 = Math.abs(pos[k * 3 + 1] - pos[k2 * 3 + 1]);
              if (dy2 > vSpan) vSpan = dy2;
            }
          }
          /* 只剪「高墙」（> 3m）。v1.16.1 用 1m 门槛去掉 alpine 崖脚的竖直拉花，
             代价是把**近景石头/灌木的侧面**也一起剪掉了 —— 那块几何一没，
             物体就变成悬空的片（手机竖屏的"悬空感"）。3m 以下保留：那点高度
             读作物体自身的侧壁，不构成横贯画面的竖幕。 */
          /* v1.21.2：**只剪远处的竖幕**（tMin > 60m）。
             竖幕是远场"深度顶到上限、水平位移却极小"造成的贴图拉伸；
             而中近处的真实崖面（晨雾湖岸那块岩壁，40~120m）竖向跨度本来就大，
             旧规则连它一起剪，在岩壁上咬出一格格白色方块。
             距离条件把两者分开：远处该剪的照剪，近处崖面保留。 */
          if (!noCull('wall') && vSpan > 3.0 && vSpan > hMin * 2.5 && tMin > 60) { skippedWall++; continue; }
          /* v1.19.1：法线朝下的四边形一律丢掉。
             单层薄壳在照片「天际线附近」会连出一批**底面朝上的长条**：
             四角深度差异大、竖直跨度却不到 3m，躲过了竖墙筛，转身看过去就是
             一层层漂浮的薄片（32 视角质检里 stripe 3.1 的那类）。
             地面的法线永远朝上（y=+1），所以 ny < 0 只可能是这种翻面几何。 */
          const qax = pos[b0 * 3] - pos[a0 * 3], qay = pos[b0 * 3 + 1] - pos[a0 * 3 + 1], qaz = pos[b0 * 3 + 2] - pos[a0 * 3 + 2];
          const qbx = pos[a1 * 3] - pos[a0 * 3], qby = pos[a1 * 3 + 1] - pos[a0 * 3 + 1], qbz = pos[a1 * 3 + 2] - pos[a0 * 3 + 2];
          const qny = qaz * qbx - qax * qbz;          // cross(pa,pb) 的 y 分量
          const qlen = Math.hypot(qax, qay, qaz) * Math.hypot(qbx, qby, qbz) || 1;
          if (qny / qlen < -0.12) { skippedWall++; continue; }
          /* 更狠的一条：**只保留「从照片相机位置看得见」的面**。
             整块重建是"从原始视点看到的一层壳"，从别的角度看到的背面本来就是
             编出来的内容 —— 转身时那些横贯画面的漂浮薄片就是它们的背面。
             判据：四边形中心相对照片相机（0, 1.7, 0）的方向与面法线同向 → 是背面。 */
          const nqx = qay * qbz - qaz * qby;
          const nqy = qaz * qbx - qax * qbz;
          const nqz = qax * qby - qay * qbx;
          const nql = Math.hypot(nqx, nqy, nqz) || 1;
          const ccx = (pos[a0 * 3] + pos[a1 * 3] + pos[b0 * 3] + pos[b1 * 3]) * 0.25;
          const ccy = (pos[a0 * 3 + 1] + pos[a1 * 3 + 1] + pos[b0 * 3 + 1] + pos[b1 * 3 + 1]) * 0.25 - eye;
          const ccz = (pos[a0 * 3 + 2] + pos[a1 * 3 + 2] + pos[b0 * 3 + 2] + pos[b1 * 3 + 2]) * 0.25;
          const ccl = Math.hypot(ccx, ccy, ccz) || 1;
          if ((nqx * ccx + nqy * ccy + nqz * ccz) / (nql * ccl) > 0.15) { skippedWall++; continue; }
        }
        /* 四角都是地面的四边形交给**世界坐标方格网**（见下），
           不要留在极坐标扇形里 —— 扇形在近处是放射状长条，贴图沿径向被拉成
           一丝丝条纹（手机上贴着地面看特别明显）。混合四边形（地面↔物体交界）
           仍然保留，避免交界处出现缝。 */
        if (opt.groundGrid !== false
          && gGround[a0] && gGround[a1] && gGround[b0] && gGround[b1]) {
          skippedGround++; continue;
        }
        /* 混合四边形（一部分角被归为地面、一部分不是）如果**跨得很深**，
           会从地面一路拉到一个高处的顶点，摊成一片白蓝色的撕裂碎片
           （赤岩峡谷截图里横贯画面的那排）。这类直接丢掉：缺口由地面方格网
           与天穹填上，比留一片撕裂的贴图干净。 */
        if (opt.groundGrid !== false) {
          const gsum = gGround[a0] + gGround[a1] + gGround[b0] + gGround[b1];
          if (gsum > 0 && gsum < 4) {
            const tHi = Math.max(gT[a0], gT[a1], gT[b0], gT[b1]);
            if (!noCull('torn') && tHi > 22) { skippedTorn++; continue; }
          }
        }
        idx.push(a0, b0, a1, a1, b0, b1);
      }
    }

    /* 4b) 世界坐标地面方格网（v1.20）
       地面像素的 y 恒为 0（射线与 y=0 求交），所以几何上它就是一张平面；
       问题只在**拓扑**：极坐标扇形在近处是放射状长条，UV 沿径向被拉长。
       这里改成规则方格：每个格子 1m 出头，UV 直接取「该点反投影到照片的 uv」
       （从地面顶点烘一张 uv 场，双线性采样），于是贴图在世界里是均匀的、
       从任何角度看都不会有放射状拉丝。 */
    const baseLinG = [
      Math.max(1e-4, Math.pow(clamp(analysis.ground[0], 0, 1), 2.2)),
      Math.max(1e-4, Math.pow(clamp(analysis.ground[1], 0, 1), 2.2)),
      Math.max(1e-4, Math.pow(clamp(analysis.ground[2], 0, 1), 2.2))
    ];
    let groundGrid = null;
    if (opt.groundGrid !== false) {
      const GS = 1.25;                        // 格子边长（米）
      const NX = Math.ceil((2 * A) / GS) + 1;
      const NZ = Math.ceil(B / GS) + 1;
      /* 1) UV 用**解析反投影**，不要 sprinkla 取整：
         地面点 (x, 0, z) 相对照片相机 (0, eye, 0) 的方向 dir，
         转到相机空间得到 dy0 = dy·cosP − dz·sinP，再按针孔投影反解
            v = 0.5 − dy0/(2·tanV)，u = 0.5 + dirX/(2·tanH)
         这样 uv 在网格上是连续光滑的；按格内平均烘场会在近处退化成
         人字纹（每格跨越很大的 uv 变化）。
         1b) 起雾/边界淡出仍按到相机的水平距离。 */
      function uvOfGround(x, z) {
        const l = Math.hypot(x, eye, z) || 1;
        const dx = x / l, dy = -eye / l, dz = z / l;
        const dy0 = dy * cosP - dz * sinP;
        return [0.5 + dx / (2 * Math.max(0.05, tanH)),
          clamp(0.5 - dy0 / (2 * Math.max(0.05, tanV)), 0.0, 0.995)];
      }
      // 3) 建网格
      const gpos = new Float32Array(NX * NZ * 3);
      const gnrm = new Float32Array(NX * NZ * 3);
      const guv = new Float32Array(NX * NZ * 2);
      const gcol = new Float32Array(NX * NZ * 4);
      const gedge = new Float32Array(NX * NZ * 2);
      const gidx = [];
      for (let gj = 0; gj < NZ; gj++) {
        for (let gi = 0; gi < NX; gi++) {
          const id = gj * NX + gi;
          const x = -A + gi * GS;
          const z = -gj * GS;
          gpos[id * 3] = x; gpos[id * 3 + 1] = -0.005; gpos[id * 3 + 2] = z;
          gnrm[id * 3 + 1] = 1;
          /* UV 用世界坐标平铺（4.5m 一次），**不要**用照片投影 uv：
             照片投影在视锥外会饱和（v→1 被 clamp），整片采样同一行 → 条纹；
             而且掠射角下的各向异性放大在手机 GPU 上更明显。
             照片地面的"颜色"由顶点色带入（groundColorAt/base），细节交给平铺贴图。 */
          guv[id * 2] = x / 4.5;
          guv[id * 2 + 1] = z / 4.5;
          {
            /* 顶点色 = 照片地面色 / 地面贴图基色（与延伸块的算法同口径）。
               注意这里不能用 `world.*`：buildCartesianWorld 里还没有 world 对象，
               第一版写了 world.baseLin 直接 ReferenceError → 整个世界建不出来。 */
            const gc = groundColorAt(x, z);
            gcol[id * 4] = clamp(gc[0] / Math.max(1e-4, baseLinG[0]), 0.25, 6);
            gcol[id * 4 + 1] = clamp(gc[1] / Math.max(1e-4, baseLinG[1]), 0.25, 6);
            gcol[id * 4 + 2] = clamp(gc[2] / Math.max(1e-4, baseLinG[2]), 0.25, 6);
          }
          gcol[id * 4 + 3] = 1;
          gedge[id * 2 + 1] = smoothstep(8, 34, Math.hypot(x, z)) * 0.92;
        }
      }
      for (let gj = 0; gj < NZ - 1; gj++) {
        for (let gi = 0; gi < NX - 1; gi++) {
          const a0 = gj * NX + gi, a1 = a0 + 1;
          const b0 = (gj + 1) * NX + gi, b1 = b0 + 1;
          gidx.push(a0, b0, a1, a1, b0, b1);
        }
      }
      groundGrid = {
        pos: gpos, normal: gnrm, uv: guv, color: gcol, edge: gedge, index: gidx
      };
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
      vfov, tris: idx.length / 3, skipped, groundGrid,
      calibration: cal ? { p: cal.p, zNear: cal.zNear, rNear: cal.rNear, rFar: cal.rFar } : null,
      heightField,
      bbox: { A, B },
      grid: { GW, GH },
      /* 单体补全：十字账单牌 + 实例元信息（供后续 InstantMesh/TripoSR 替换） */
      cards: instInfo.cards,
      instances: instInfo.instances,
      /* 照片地面颜色场（线性 RGB）采样器：js/extend.js 给扇环外 tile 上色用 */
      groundColorAt,
      /* 照片最底几行的按列颜色：竖屏近景补片用它接上照片底边，消掉横向接缝 */
      bottomRowColor,
      /* 底部裙边网格（照片底行向下连续延伸）：消掉照片底边的几何断口 */
      skirt,
      semGroups: sem ? sem.groups : null,
      semSize: sem ? [sem.width, sem.height] : null,
      /* 信息保留率诊断：哪一步丢了多少面，直接决定「图片丢信息」的排查方向 */
      stats: {
        kept: idx.length / 6, total: quadTotal,
        keepRate: (idx.length / 6) / quadTotal,
        culledAlpha, culledArea, skipped, skippedCard, skippedObjEdge, skippedSteep, skippedWall,
        skippedGround, skippedTorn,
        objectCards: instInfo.instances.length
      },
      /* 质检用：把顶点位置暴露出来（?dbgpos=1），tools/probe.mjs 可以据此
         统计「相邻网格竖直跳变」发生在画面的哪个位置，定位竖条/鳍的成因。 */
      dbgPos: (opt && opt.dbgPos)
        ? (function () { const a = new Float32Array(GW * GH); for (let q = 0; q < a.length; q++) a[q] = col[q * 4 + 3]; return { pos, uv, gT, GW, GH, gGrp, alpha: a, idx }; })()
        : null,
      meta: {
        indoor, hasSky, skyD: +skyD.toFixed(3), culledAlpha, culledArea, skipped,
        skippedCard, skippedObjEdge, skippedSteep, skippedWall,
        objectCards: instInfo.instances.length,
        sem: sem ? 'on' : 'off'
      }
    };
  }

  /* ---------------- 单体实例 → 不拉伸（v1.16） ----------------
   * 问题：树/房屋被反投影成连续网格时，树冠（远）与地面（近）被连成一个四边形，
   * 沿视线方向拉成几十米的「鳍」。示例图里的针叶树就是这样变成三角尖刺的。
   * 做法：
   *   1) 语义对象（植被/建筑/人/车/器物）且「比同行基线明显更近」的像素，
   *      按 4 邻域连通域切成实例；
   *   2) 实例面积过小（噪点）或过大（森林/山脉整体）不处理，避免把地貌切碎；
   *   3) 每个实例压到**自身中位深度**（保持竖直、仍与地面相连），
   *      相邻像素共享深度，面片不再沿视线拉长。
   * 真三维补全走 js/object3d.js + 服务端 InstantMesh/TripoSR（可选）。
   */
  function labelObjectCards(gGrp, gT, pos, uv, GW, GH, tanH, tanV, gNear) {
    const N = GW * GH;
    const OBJ = new Uint8Array(12);
    OBJ[6] = OBJ[7] = OBJ[8] = OBJ[9] = OBJ[10] = 1;   // VEG/BUILDING/PERSON/VEHICLE/OBJECT
    // 真正要卡片的像素：语义是对象 **且** 比同行基线明显更近。
    // 不加后半个条件，整片森林/雪坡都会被当成一个巨型对象切出来。
    const isCard = (k) => OBJ[gGrp[k]] && (!gNear || gNear[k]);
    const gCardId = new Int32Array(N).fill(-1);
    const insts = [];
    const seen = new Uint8Array(N);
    const stack = [];
    const minArea = Math.max(16, Math.round(N * 0.0006));
    const maxArea = Math.round(N * 0.30);   // 超过 30% 画面的当作地貌/森林，不切卡片
    const MAX_INST = 120;

    for (let seed = 0; seed < N; seed++) {
      if (seen[seed] || !isCard(seed)) continue;
      const cells = [];
      let minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9, ymin = 1e9;
      let sx = 0, sy = 0, sz = 0;
      stack.length = 0; stack.push(seed); seen[seed] = 1;
      while (stack.length) {
        const q = stack.pop();
        cells.push(q);
        const qi = q % GW, qj = (q / GW) | 0;
        const u = qi / (GW - 1), v = qj / (GH - 1);
        if (u < minx) minx = u; if (u > maxx) maxx = u;
        if (v < miny) miny = v; if (v > maxy) maxy = v;
        const y = pos[q * 3 + 1];
        if (y < ymin) ymin = y;
        sx += pos[q * 3]; sy += y; sz += pos[q * 3 + 2];
        const nb = [qi > 0 ? q - 1 : -1, qi < GW - 1 ? q + 1 : -1,
          qj > 0 ? q - GW : -1, qj < GH - 1 ? q + GW : -1];
        for (const n of nb) {
          if (n < 0 || seen[n] || !isCard(n)) continue;
          seen[n] = 1; stack.push(n);
        }
        if (cells.length > maxArea * 1.6) break;   // 超大实例提前收手
      }
      const area = cells.length;
      if (area < minArea || area > maxArea) continue;
      const bw = maxx - minx, bh = maxy - miny;
      if (bw < 0.02 || bh < 0.02 || bw > 0.9 || bh > 0.95) continue;
      const id = insts.length;
      if (id >= MAX_INST) break;
      const ts = [];
      for (const q of cells) ts.push(gT[q]);
      ts.sort((a, b) => a - b);
      const tMed = ts[ts.length >> 1];
      if (!(tMed > 0.3 && tMed < 240)) continue;
      for (const q of cells) gCardId[q] = id;
      insts.push({
        id, area, uv: [minx, miny, maxx, maxy], t: tMed,
        cx: sx / area, cz: sz / area, ymin, group: gGrp[seed], cells
      });
    }

    /* 不再生成独立的矩形卡片：A/B 实测矩形裁片会在场景里变成「墓碑」
       （裁片带背景、且与地面脱开）。改为把实例压到自身中位深度后**就地保留**，
       由 buildCartesianWorld 统一重算顶点，见 1c-bis。 */
    return { gCardId, instances: insts, cards: null };
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
              nearPlane: opts && opts.nearPlane,
              noCullReasons: opts && opts.noCullReasons,
              skyFill: opts && opts.skyFill,
              fgMask: opts && opts.fgMask,
              sem: opts && opts.sem,
              dbgPos: opts && opts.dbgPos,
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
      // v1.20.3：主照片贴图改用全分辨率 photoCanvas（2048 上限），不再用 420px
      // 分析画布。其余程序化纹理（sky / ground / backdrop）仍取 analysis.canvas，
      // 它们本来就是低频色带，且与语义采样口径绑定，不能跟着换源。
      photo: PWGL.createTexture(gl, analysis.photoCanvas || analysis.canvas, { wrap: 'clamp', clampV: true, aniso: 8, flipY: false }),
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
      // 世界坐标地面方格网（替代极坐标扇形的地面部分，消掉放射状拉丝）
      groundGrid: (relief && relief.groundGrid) ? PWGL.createMesh(gl, relief.groundGrid) : null,
      // 照片底行向下的连续裙边（消掉照片底边的几何断口）
      skinSkirt: (relief && relief.skirt) ? PWGL.createMesh(gl, relief.skirt) : null,
      // 单体十字账单牌（v1.16）：替代被拉成鳍的对象四边形
      objectCards: (relief && relief.cards) ? PWGL.createMesh(gl, relief.cards) : null,
      // v1.15：默认只生成正面一个扇环，其他地方留空（用户明确要求）。
      // 镜像副本保留为实验（?surround=1），默认不建，避免假内容。
      reliefSurround: (relief && relief.grid && opts && opts.surround)
        ? PWGL.createMesh(gl, buildSurround(relief, { step: 3 })) : null,
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
      // v1.19：所有模式都不限制移动（0 = 不限制）。旧 fallback 的 116m 半径
      // 就是用户遇到的「空气墙」。
      walkRadius: 0,
      /* v1.17：重建表面是单层薄壳，从背面看就是糊成一片的拉花。
         默认关闭背面（?cull=0 保留旧双面），转到侧后方时看到的是
         360° 低频世界幕 + 雾，而不是「纸片背面」。 */
      reliefCull: !(opts && opts.noCull)
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
      /* 延伸地面用的照片颜色场（线性 RGB，覆盖整个重建矩形）。
         extend.js 用它给扇环外的 tile 上色，接缝处颜色连续。 */
      groundColorAt: (relief && relief.groundColorAt) ? relief.groundColorAt : null,
      /* 质检用顶点数据（?dbgpos=1）：排查竖条/鳍时直接读网格 */
      dbgPos: (relief && relief.dbgPos) ? relief.dbgPos : null,
      groundColorMeta: (relief && relief.bbox)
        ? { x0: -relief.bbox.A, z0: -relief.bbox.B, spanX: 2 * relief.bbox.A, spanZ: relief.bbox.B }
        : { x0: -55, z0: -100, spanX: 110, spanZ: 100 },
      /* 程序化地面贴图的基色（线性）：tile 的顶点色是「目标色 / 基色」的增益 */
      groundBaseLin: lin3(analysis.ground),
      /* v1.21.1：雾密度 0.0040 → 0.0026。0.004 在 150m 处就把台地糊掉一半，
         用户对赤岩峡谷的评价是"雾气很浓厚"。晴天照片（天空亮、对比高）
         再额外减半，见下面的 hazeFactor。 */
      lights: { sunDir, sunColor: sunL, ambient: ambientL, fogColor: fogL, fogDensity: 0.0026 },
      meshes: m,
      // 扇环延伸生成的动态 tile（js/extend.js 往里塞，渲染器逐块绘制）
      extraMeshes: [],
      extendMeta: null,
      tex,
      particles,
      seed,
      // 照片重建设定覆盖的直角区域（延伸器据此判断扇环边界）
      bbox: (relief && relief.bbox) ? relief.bbox : { A: 55, B: 100 },
      spawn: { x: 0, z: 0 },
      mode: worldInfo.mode,
      eyeHeight: worldInfo.eyeHeight,
      walkRadius: worldInfo.walkRadius,
      reliefMeta: relief ? Object.assign({ vfov: relief.vfov }, relief.meta || {}) : null,
      // 单体实例（bbox / 中位深度 / 世界中心），供 js/object3d.js 升级为真网格
      objectInstances: (relief && relief.instances) ? relief.instances : [],
      _cardsData: (relief && relief.cards) ? relief.cards : null,
      /* 用真网格替换部分账单牌：把已升级实例的顶点从卡片网格里剔除并重建。
         ids = 已升级为 3D mesh 的实例 id 列表（由 js/object3d.js 调用）。 */
      hideObjectCards: function (ids) {
        if (!relief || !relief.cards || !m.objectCards) return;
        const ex = new Set(ids || []);
        const keep = (relief.instances || []).filter(function (it) { return !ex.has(it.id); });
        if (!keep.length) { m.objectCards = null; return; }
        const src = relief.cards;
        let nv = 0;
        for (let i = 0; i < keep.length; i++) nv += keep[i].vCount;
        const pos = new Float32Array(nv * 3), nrm = new Float32Array(nv * 3);
        const uv = new Float32Array(nv * 2), col = new Float32Array(nv * 4);
        const index = [];
        let vo = 0;
        for (const it of keep) {
          pos.set(src.pos.subarray(it.vStart * 3, (it.vStart + it.vCount) * 3), vo * 3);
          nrm.set(src.normal.subarray(it.vStart * 3, (it.vStart + it.vCount) * 3), vo * 3);
          uv.set(src.uv.subarray(it.vStart * 2, (it.vStart + it.vCount) * 2), vo * 2);
          col.set(src.color.subarray(it.vStart * 4, (it.vStart + it.vCount) * 4), vo * 4);
          for (let k = 0; k < it.iCount; k++) index.push(src.index[it.iStart + k] - it.vStart + vo);
          vo += it.vCount;
        }
        m.objectCards = PWGL.createMesh(gl, { pos, normal: nrm, uv, color: col, index });
      }
    };

    return world;
  }

  global.PW.world = { build, makeHeightField, panoU };
})(window);
