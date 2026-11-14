/* 有限扇环 → 无限地貌延伸（v1.16）
 *
 * 目标：照片重建只覆盖「相机前方一个梯形/扇环」。玩家往前走、往侧走时，
 * 扇环外应依照已有地貌自动延伸，而不是一步踏进雾里（参考 oasis 式无限世界）。
 *
 * 实现分两层：
 *   1) 程序化兜底（默认，零依赖）：全局连续 fBm 高度场 + 与重建边界的平滑混合。
 *      因为是同一个全局函数，相邻 tile 在共享坐标上天然无缝，不需要做接缝求解。
 *   2) 服务端生成（可选）：POST /api/extend 把「种子 + tile 坐标 + 边界高度采样」
 *      发给本地 GPU 服务（Terrain Diffusion / 扩散 inpainting），返回同尺寸高度图。
 *      服务不可用时静默回退到第 1 层，绝阻塞主流程。
 *
 * 行走：extender 会接管 world.heightAt，扇环外也站在真实地形上。
 */
(function (global) {
  'use strict';

  const { clamp, lerp, smoothstep } = global.PW.util;

  /* ---------------- 确定性噪声 ---------------- */
  function hash2(ix, iy, seed) {
    let h = (ix * 374761393 + iy * 668265263 + seed * 2246822519) | 0;
    h = (h ^ (h >>> 13)) * 1274126177;
    h = h ^ (h >>> 16);
    return (h >>> 0) / 4294967295;
  }
  function vnoise(x, y, seed) {
    const ix = Math.floor(x), iy = Math.floor(y);
    const fx = x - ix, fy = y - iy;
    const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
    const a = hash2(ix, iy, seed), b = hash2(ix + 1, iy, seed);
    const c = hash2(ix, iy + 1, seed), d = hash2(ix + 1, iy + 1, seed);
    return lerp(lerp(a, b, ux), lerp(c, d, ux), uy);
  }
  function fbm(x, y, seed, oct) {
    let s = 0, amp = 0.5, f = 1, norm = 0;
    for (let i = 0; i < (oct || 5); i++) {
      s += amp * vnoise(x * f, y * f, seed + i * 131);
      norm += amp; amp *= 0.5; f *= 2.03;
    }
    return s / norm;                       // 0..1
  }

  function createExtender(opts) {
    opts = opts || {};
    const gl = opts.gl;
    const world = opts.world;
    const seed = opts.seed || 1;
    const TS = opts.tile || 32;            // tile 边长（米）
    const RES = opts.res || 17;            // 每边顶点数（16 段）
    const RADIUS = opts.radius || 5;       // 以玩家为中心保留的 tile 半径
    const SERVER = (opts.server || '').replace(/\/$/, '');
    const MAX_TILES = opts.maxTiles || (2 * RADIUS + 3) * (2 * RADIUS + 3);

    const bb = world.bbox || { A: 55, B: 100 };
    const A = bb.A, B = bb.B;
    const baseHeight = world.heightAt.bind(world);

    const insideRect = (x, z) => (x >= -A && x <= A && z >= -B && z <= 0);

    /* 基础地貌的边界高度：把点钳到重建矩形内再取高度，作为延伸的起点 */
    function edgeHeight(x, z) {
      const cx = clamp(x, -A + 0.01, A - 0.01);
      const cz = clamp(z, -B + 0.01, -0.01);
      return baseHeight(cx, cz);
    }

    /* 扇环外高度：全局连续噪声 + 边界平滑过渡。
       dist 是到重建矩形的距离；前 14m 内主要继承边界高度，之后逐渐交给噪声。 */
    function outsideHeight(x, z) {
      const eh = edgeHeight(x, z);
      let dist;
      if (insideRect(x, z)) dist = 0;
      const dx = Math.max(-A - x, 0, x - A);
      const dz = Math.max(-B - z, 0, z - 0);
      dist = Math.hypot(dx, dz);
      const ramp = smoothstep(0, 14, dist);
      // 起伏幅度随距离增长，远处允许更夸张的地貌；基线保持在边界高度附近
      const amp = 1.2 + 7.5 * smoothstep(0, 90, dist);
      const n = fbm(x * 0.021, z * 0.021, seed, 5) - 0.5;
      const n2 = fbm(x * 0.079, z * 0.079, seed + 17, 4) - 0.5;
      const h = eh + ramp * (n * amp + n2 * amp * 0.28);
      // 越远越不往下掉，避免走出边界后出现断崖
      return h;
    }

    function heightAt(x, z) {
      if (insideRect(x, z)) return baseHeight(x, z);
      return outsideHeight(x, z);
    }

    /* 接管行走高度：扇环外不再 clamp 到边缘 */
    world.heightAt = heightAt;

    /* 区域内底板（v1.16.1）：重建矩形在断崖/竖幕处被剪开时，洞后面是天穹的
       灰色，远看像一条横贯画面的灰带（alpine 湖岸实测）。这里铺一张
       y=-0.06 的底板，颜色取照片地面色，画在重建表面**之前**——
       有几何的地方被挡住，洞后面是地面色，读作「雾里的远岸」而不是天空。 */
    function buildFloorMesh() {
      const NX = 49, NZ = 33;
      const pos = new Float32Array(NX * NZ * 3);
      const nrm = new Float32Array(NX * NZ * 3);
      const uv = new Float32Array(NX * NZ * 2);
      const col = new Float32Array(NX * NZ * 4);
      const idx = [];
      const base = (world.groundBaseLin && world.groundBaseLin.length === 3)
        ? world.groundBaseLin.map((v) => Math.max(1e-4, v)) : [0.18, 0.18, 0.18];
      const colorAt = world.groundColorAt || null;
      for (let j = 0; j < NZ; j++) {
        for (let i = 0; i < NX; i++) {
          const k = j * NX + i;
          const x = -A + (i / (NX - 1)) * 2 * A;
          const z = (j / (NZ - 1)) * B;          // z: 0 → -B
          pos[k * 3] = x; pos[k * 3 + 1] = -0.06; pos[k * 3 + 2] = -z;
          nrm[k * 3 + 1] = 1;
          // 底板只在「洞」里可见，UV 放大 + 高 mipBias 让纹理糊成一层柔和色块，
          // 避免掠射角下被拉成一条条竖纹
          uv[k * 2] = x / 40; uv[k * 2 + 1] = z / 40;
          let r = 1, g = 1, b = 1;
          if (colorAt) {
            const c = colorAt(x, -z);
            /* 往雾色混一半：这段只在洞（远景断崖/远岸）里露出，混合之后
               读起来像雾里的岸线，而不是一块生硬的地面色。 */
            const fc = world.fogColor || [0, 0, 0];
            r = clamp((c[0] * 0.45 + fc[0] * 0.55) / base[0], 0.25, 6);
            g = clamp((c[1] * 0.45 + fc[1] * 0.55) / base[1], 0.25, 6);
            b = clamp((c[2] * 0.45 + fc[2] * 0.55) / base[2], 0.25, 6);
          }
          col[k * 4] = r; col[k * 4 + 1] = g; col[k * 4 + 2] = b; col[k * 4 + 3] = 1;
        }
      }
      for (let j = 0; j < NZ - 1; j++) {
        for (let i = 0; i < NX - 1; i++) {
          const a0 = j * NX + i, a1 = a0 + 1;
          const b0 = (j + 1) * NX + i, b1 = b0 + 1;
          idx.push(a0, b0, a1, a1, b0, b1);
        }
      }
      return global.PW.GL.createMesh(gl, { pos, normal: nrm, uv, color: col, index: idx });
    }
    try {
      if (world.meshes) world.meshes.floor = buildFloorMesh();
    } catch (e) { /* 底板失败不影响主流程 */ }


    const tiles = new Map();
    let lastUpdate = -1e9;
    let pending = 0;

    function tileKey(tx, tz) { return tx + ',' + tz; }

    function buildTileMesh(tx, tz, heights) {
      const x0 = tx * TS, z0 = tz * TS;
      const pos = new Float32Array(RES * RES * 3);
      const nrm = new Float32Array(RES * RES * 3);
      const uv = new Float32Array(RES * RES * 2);
      const col = new Float32Array(RES * RES * 4);
      const idx = [];
      const step = TS / (RES - 1);
      const tint = world.groundTint || [0.32, 0.32, 0.30];
      /* 顶点色 = 照片地面色 / 程序化地面贴图基色（线性空间的增益）。
         贴图保留细节，颜色回到照片本身：蓝调雪林外延是亮的雪蓝，
         沙漠外延是砂黄，不会再出现「照片一块亮、外面一片深蓝」的明信片感。 */
      const base = (world.groundBaseLin && world.groundBaseLin.length === 3)
        ? world.groundBaseLin.map((v) => Math.max(1e-4, v))
        : [0.18, 0.18, 0.18];
      const colorAt = world.groundColorAt || null;
      for (let j = 0; j < RES; j++) {
        for (let i = 0; i < RES; i++) {
          const k = j * RES + i;
          const x = x0 + i * step, z = z0 + j * step;
          const h = heights ? heights[k] : heightAt(x, z);
          pos[k * 3] = x; pos[k * 3 + 1] = h; pos[k * 3 + 2] = z;
          uv[k * 2] = x / 9; uv[k * 2 + 1] = z / 9;
          // 逐顶点噪声：同一片地面不要太均匀，也不要强到能看出 tile 网格
          const nz = vnoise(x * 0.05, z * 0.05, seed + 5) - 0.5;
          const shade = 1 + nz * 0.26;
          let r = 1, g = 1, b = 1;
          if (colorAt) {
            const c = colorAt(x, z);            // 线性 RGB（矩形外自动取边界值）
            // 距离越远越回到贴图基色，避免远处变成一望无际的纯色底
            const ddx = Math.max(-A - x, 0, x - A);
            const ddz = Math.max(-B - z, 0, z - 0);
            const kFar = smoothstep(30, 150, Math.hypot(ddx, ddz));
            for (let q = 0; q < 3; q++) {
              const gain = clamp(c[q] / base[q], 0.25, 6);
              const v = lerp(gain, 1, kFar * 0.75) * shade;
              if (q === 0) r = v; else if (q === 1) g = v; else b = v;
            }
          } else {
            r = g = b = shade * (0.85 + tint[0] * 0.15);
          }
          col[k * 4] = r; col[k * 4 + 1] = g; col[k * 4 + 2] = b; col[k * 4 + 3] = 1;
        }
      }
      for (let j = 0; j < RES; j++) {
        for (let i = 0; i < RES; i++) {
          const k = j * RES + i;
          const x = x0 + i * step, z = z0 + j * step;
          const hL = heightAt(x - step * 0.5, z), hR = heightAt(x + step * 0.5, z);
          const hD = heightAt(x, z - step * 0.5), hU = heightAt(x, z + step * 0.5);
          let nx = hL - hR, ny = 2 * step, nz = hD - hU;
          const l = Math.hypot(nx, ny, nz) || 1;
          nrm[k * 3] = nx / l; nrm[k * 3 + 1] = ny / l; nrm[k * 3 + 2] = nz / l;
        }
      }
      for (let j = 0; j < RES - 1; j++) {
        for (let i = 0; i < RES - 1; i++) {
          const a0 = j * RES + i, a1 = a0 + 1;
          const b0 = (j + 1) * RES + i, b1 = b0 + 1;
          /* 只保留「四边形中心落在重建矩形之外」的面。
             跨边界的 tile 会整块生成，如果不裁，靠近扇环的那一格（32m 深）
             会盖在照片几何上——alpine 湖面那道又厚又平的灰带就是这么来的。 */
          const cx = (x0 + (i + 0.5) * step), cz = (z0 + (j + 0.5) * step);
          if (cx >= -A && cx <= A && cz >= -B && cz <= 0) continue;
          idx.push(a0, b0, a1, a1, b0, b1);
        }
      }
      const data = { pos, normal: nrm, uv, color: col, index: idx };
      return global.PW.GL.createMesh(gl, data);
    }

    /* 服务端延伸：把 tile 边界高度作为条件发过去，拿回同尺寸高度图 */
    async function fetchServerHeights(tx, tz) {
      if (!SERVER || typeof fetch === 'undefined') return null;
      const x0 = tx * TS, z0 = tz * TS;
      const edges = { left: [], right: [], top: [], bottom: [] };
      for (let i = 0; i < RES; i++) {
        const t = i / (RES - 1);
        edges.left.push(heightAt(x0, z0 + t * TS));
        edges.right.push(heightAt(x0 + TS, z0 + t * TS));
        edges.top.push(heightAt(x0 + t * TS, z0));
        edges.bottom.push(heightAt(x0 + t * TS, z0 + TS));
      }
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), 2500);
      try {
        const r = await fetch(SERVER + '/api/extend', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ seed, tx, tz, ts: TS, res: RES, edges }),
          signal: ctrl.signal
        });
        if (!r.ok) return null;
        const j = await r.json();
        if (!j || !Array.isArray(j.heights) || j.heights.length !== RES * RES) return null;
        return Float32Array.from(j.heights);
      } catch (e) {
        return null;                        // 服务端不可用 → 静默回退程序化
      } finally {
        clearTimeout(to);
      }
    }

    async function ensureTile(tx, tz) {
      const key = tileKey(tx, tz);
      if (tiles.has(key)) return;
      // 先在 Map 里占位，避免同一帧重复排队
      tiles.set(key, { pending: true, mesh: null, tx, tz });
      pending++;
      let heights = null;
      if (SERVER) heights = await fetchServerHeights(tx, tz);
      const mesh = buildTileMesh(tx, tz, heights);
      const rec = tiles.get(key);
      rec.pending = false;
      rec.mesh = mesh;
      rec.server = !!heights;
      pending--;
      if (!world.extraMeshes) world.extraMeshes = [];
      world.extraMeshes.push({ key, mesh, tex: world.tex.ground, fog: 0.34 });
      // 超出上限时删掉最早插入的一块（tiles / extraMeshes 都按插入顺序）
      if (world.extraMeshes.length > MAX_TILES) {
        const first = world.extraMeshes.shift();
        if (first) {
          tiles.delete(first.key);
          // 显式删除 GL buffer，避免长时间行走后显存持续增长
          try {
            const mm = first.mesh;
            for (const a of ['aPos', 'aNormal', 'aUV', 'aColor', 'aEdge']) {
              if (mm[a] && mm[a].buf) gl.deleteBuffer(mm[a].buf);
            }
            if (mm.index && mm.index.buf) gl.deleteBuffer(mm.index.buf);
          } catch (e) { /* ignore */ }
        }
      }
    }

    /* 每帧调用：节流 + 只补玩家周围缺失的 tile */
    function update(px, pz, now) {
      world._px = px; world._pz = pz;
      const t = now === undefined ? (global.performance ? global.performance.now() : Date.now()) : now;
      if (t - lastUpdate < 260) return 0;     // 260ms 节流，避免每帧重算 tile 列表
      lastUpdate = t;
      const cx = Math.round(px / TS), cz = Math.round(pz / TS);
      let queued = 0;
      for (let dz = -RADIUS; dz <= RADIUS; dz++) {
        for (let dx = -RADIUS; dx <= RADIUS; dx++) {
          const tx = cx + dx, tz = cz + dz;
          const key = tileKey(tx, tz);
          if (tiles.has(key)) continue;
          // 完全落在重建矩形内的 tile 不生成（那里已有照片几何）
          const x0 = tx * TS, z0 = tz * TS, x1 = x0 + TS, z1 = z0 + TS;
          const fullyInside = x0 >= -A && x1 <= A && z0 >= -B && z1 <= 0;
          if (fullyInside) continue;
          ensureTile(tx, tz);
          queued++;
          if (queued >= 3) return queued;     // 每轮最多补 3 块，避免卡帧
        }
      }
      return queued;
    }

    function stats() {
      return { tiles: tiles.size, meshes: (world.extraMeshes || []).length, pending, server: !!SERVER };
    }

    function dispose() {
      world.extraMeshes = [];
      tiles.clear();
    }

    return { update, heightAt, stats, dispose, tileSize: TS };
  }

  global.PW = global.PW || {};
  global.PW.createExtender = createExtender;
  global.PW.extendInternals = { fbm, vnoise, hash2 };
})(window);
