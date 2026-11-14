/* 有限扇环 → 视锥驱动的实时地貌生成（v1.18）
 *
 * 背景：照片重建只覆盖相机前方一块矩形（x∈[-A,A]、z∈[-B,0]）。矩形外的世界
 * 有三种做法，本项目逐版淘汰到第 3 种：
 *   1) v1.14 镜像副本（buildSurround）：把照片几何绕 Y 轴复制 5 份铺满 360°。
 *      对称重复一眼可辨，转身看到的是自己刚看过的山（默认已关）。
 *   2) v1.16 全局 fBm tile：连续噪声 + 边界硬约束，走出去地形不断，但**永远同一套地貌**，
 *      且不区分朝向 —— 身后的 tile 也在算，白烧 CPU。
 *   3) 本版：**视锥驱动 + 可调主题**。只生成相机视锥覆盖的格子（转身时才补身后），
 *      每个扇区有自己的「主题」（起伏幅度/基线/频率/色相），主题与照片的贴合度由
 *      `混乱度` 控制、世界是否会被重掷由 `记忆力` 控制。
 *      这是 Oasis（Decart 的实时生成世界）那份「按视野流式生成 + 记忆衰减」的行为骨架，
 *      差别在于我们用零依赖的程序化地形，而不是扩散模型。
 *
 * 参数语义（与 UI 一致）：
 *   混乱度 chaos  0 = 整个世界按照片这一个主题生成（相邻扇区完全一致，边界无缝）；
 *                 1 = 每个扇区的地貌与之前的、与邻居的**完全无关**（主题独立采样）。
 *   记忆力 memory 1 = 生成后基本不再改变；0 = 眼前（视锥内）保持不变，视锥外的扇区
 *                 按秒级概率重掷（epoch+1），转身时看到的是新地貌 —— Oasis 式体验。
 *
 * 行走：heightAt 是解析函数（不是查已生成的 tile），所以玩家可以无限走，
 * 脚下永远有地面，且与渲染的地形同源（同一 sector + epoch）。
 */
(function (global) {
  'use strict';

  const { clamp, lerp, smoothstep } = global.PW.util;
  const TAU = Math.PI * 2;

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
    const RADIUS = opts.radius || 5;       // 关闭实时模式时以玩家为中心的固定半径
    const SERVER = (opts.server || '').replace(/\/$/, '');
    /* 上限按「12 格扫描里视锥那一片」估：25×25 的 40% 约 250 块，再留余量。
       旧的 289 在扫描半径拉到 12 之后会被顶满，导致刚生成的块被立刻回收
       （表现为走两步地形就"塌"一层）。 */
    const MAX_TILES = opts.maxTiles || 560;

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

    /* ---------------- 主题：扇区 × 纪元 ----------------
       扇区按「相对照片相机（世界原点）」的方位角切 24 份（每 15°）。
       用世界原点的方位角而不是玩家位置：玩家走动时，同一块地仍属于同一扇区，
       主题不会因为人挪了两米就换一套。 */
    const NS = opts.sectors || 24;
    const epochs = new Int32Array(NS);
    const live = {
      enabled: !!opts.live,
      chaos: typeof opts.chaos === 'number' ? clamp(opts.chaos, 0, 1) : 0.35,
      memory: typeof opts.memory === 'number' ? clamp(opts.memory, 0, 1) : 0.7
    };
    let rerolls = 0;

    function sectorOf(x, z) {
      const a = Math.atan2(x, -z);          // 0 = 照片正前方（-Z），±π = 背后
      const s = Math.floor(((a + Math.PI) / TAU) * NS);
      return (s < 0 ? 0 : (s >= NS ? NS - 1 : s)) | 0;
    }

    /* 图集训练出的生物群系原型（tools/train_live_params.mjs，1268 张语料）。
       有它之后，混乱度=1 时扇区拿到的是「雪原 / 沙漠 / 森林 / 岩滩」这类真实地貌的
       配色与起伏统计，而不是随机色相（第一版用 sin(hue) 当增益，出来一片荧光绿）。 */
    const PROTOS = (global.PW && global.PW.liveParams && global.PW.liveParams.prototypes) || null;

    function prototypeFor(sector, epoch) {
      if (!PROTOS || !PROTOS.length) return null;
      const r = hash2(sector + 3, epoch + 11, seed + 707);
      return PROTOS[Math.min(PROTOS.length - 1, Math.floor(r * PROTOS.length))];
    }

    /* 每个 (扇区, 纪元) 一套主题：原型提供配色/起伏/频率，hash 提供形状种子。
       chaos=1 时完全由原型决定（与照片、与邻居都无关）；chaos=0 时退化为照片主题。 */
    function themeFor(sector, epoch) {
      const r1 = hash2(sector + 1, epoch + 1, seed + 101);
      const r2 = hash2(sector + 7, epoch + 3, seed + 202);
      const c = live.chaos;
      const p = prototypeFor(sector, epoch);
      const ampP = p ? p.amp : 1.1;
      const roughP = p ? p.rough : 1.0;
      return {
        amp: lerp(1, ampP, c),
        bias: (r2 - 0.5) * 7.0 * c,         // 基线高度偏移 ±3.5m（随混乱度生效）
        rough: lerp(1, roughP, c),
        proto: p,
        seedOff: 1 + Math.floor(r1 * 997)   // 噪声种子偏移 → 地形形状本身就是新的
      };
    }

    /* 关闭实时模式时的旧路径（v1.16 行为，逐字节保持，回归测试依赖它） */
    function outsideHeight(x, z) {
      const eh = edgeHeight(x, z);
      const dx = Math.max(-A - x, 0, x - A);
      const dz = Math.max(-B - z, 0, z - 0);
      const dist = Math.hypot(dx, dz);
      const ramp = smoothstep(0, 14, dist);
      const amp = 1.2 + 7.5 * smoothstep(0, 90, dist);
      const n = fbm(x * 0.021, z * 0.021, seed, 5) - 0.5;
      const n2 = fbm(x * 0.079, z * 0.079, seed + 17, 4) - 0.5;
      return eh + ramp * (n * amp + n2 * amp * 0.28);
    }

    /* 主题化高度：共享主题（照片地貌）与独立主题按 chaos 线性混合。
       chaos=0 时逐值等于 outsideHeight（上面那条分支保证位相同）；
       chaos=1 时完全由本扇区本纪元的主题决定 —— 与邻居、与上一纪元都无关。 */
    function themedHeight(x, z, sector, epoch) {
      const chaos = live.chaos;
      const eh = edgeHeight(x, z);
      const dx = Math.max(-A - x, 0, x - A);
      const dz = Math.max(-B - z, 0, z - 0);
      const dist = Math.hypot(dx, dz);
      const ramp = smoothstep(0, 14, dist);
      const ampBase = 1.2 + 7.5 * smoothstep(0, 90, dist);

      const ns1 = fbm(x * 0.021, z * 0.021, seed, 5) - 0.5;
      const ns2 = fbm(x * 0.079, z * 0.079, seed + 17, 4) - 0.5;
      const hShared = eh + ramp * (ns1 * ampBase + ns2 * ampBase * 0.28);
      if (chaos <= 0.001) return hShared;

      const th = themeFor(sector, epoch);
      const f = 0.021 * th.rough;
      const ni1 = fbm(x * f, z * f, seed + th.seedOff, 5) - 0.5;
      const ni2 = fbm(x * f * 3.76, z * f * 3.76, seed + th.seedOff + 31, 4) - 0.5;
      const ampI = ampBase * th.amp;
      // 基线偏移在贴近矩形的前 4~40m 内渐入，保证照片边界处不出现断墙
      const hIndep = eh + th.bias * smoothstep(4, 40, dist)
        + ramp * (ni1 * ampI + ni2 * ampI * 0.28);
      return lerp(hShared, hIndep, chaos);
    }

    function heightAt(x, z) {
      if (insideRect(x, z)) return baseHeight(x, z);
      if (!live.enabled) return outsideHeight(x, z);
      const s = sectorOf(x, z);
      return themedHeight(x, z, s, epochs[s]);
    }

    /* 接管行走高度：玩家可以一直走，扇环外站在生成的地形上（不限制移动范围） */
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
          uv[k * 2] = x / 40; uv[k * 2 + 1] = z / 40;
          let r = 1, g = 1, b = 1;
          if (colorAt) {
            const c = colorAt(x, -z);
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
    /* 近处地面补片（v1.18.1）：竖屏时相机竖直视场（~60°）比照片自己的视场（~41°）
       大一截，照片几何下方会留出一条"比照片底边还近"的地面 —— 那条只能用底板兜，
       但底板 UV 放大到 40m + 高 mipBias，是一块没有细节的灰板（手机截图里最显眼）。
       这里再铺一张**带贴图细节**的近景补片（uv 4.5m 一次，mipBias 0.2），
       同样画在重建表面之前：有几何处被完全遮住，只在缺口/近处露出来。 */
    function buildNearGroundMesh() {
      const NX = 41, NZ = 41;
      const pos = new Float32Array(NX * NZ * 3);
      const nrm = new Float32Array(NX * NZ * 3);
      const uv = new Float32Array(NX * NZ * 2);
      const col = new Float32Array(NX * NZ * 4);
      const idx = [];
      const base = (world.groundBaseLin && world.groundBaseLin.length === 3)
        ? world.groundBaseLin.map((v) => Math.max(1e-4, v)) : [0.18, 0.18, 0.18];
      const colorAt = world.groundColorAt || null;
      const W2 = 40;                       // 补片半宽（米）：覆盖玩家周围的近景
      for (let j = 0; j < NZ; j++) {
        for (let i = 0; i < NX; i++) {
          const k = j * NX + i;
          const x = -W2 + (i / (NX - 1)) * 2 * W2;
          const z = -W2 + (j / (NZ - 1)) * 2 * W2;
          pos[k * 3] = x; pos[k * 3 + 1] = -0.05; pos[k * 3 + 2] = z;
          nrm[k * 3 + 1] = 1;
          uv[k * 2] = x / 4.5; uv[k * 2 + 1] = z / 4.5;
          /* 颜色来源优先级：照片底行剖面（最贴边）→ 地面色场 → 单色。
             底行剖面随世界 x 变化、与照片最下方像素同源，因此补片和照片底边之间
             不会再有那条横向色差接缝。 */
          let c = null;
          if (world.bottomRowColor) c = world.bottomRowColor(x);
          else if (colorAt) c = colorAt(x, z);
          let r = 1, g = 1, b = 1;
          if (c) {
            for (let q = 0; q < 3; q++) {
              const gain = clamp(c[q] / base[q], 0.25, 6);
              if (q === 0) r = gain; else if (q === 1) g = gain; else b = gain;
            }
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
      if (world.meshes) {
        world.meshes.floor = buildFloorMesh();
        world.meshes.nearGround = buildNearGroundMesh();
      }
    } catch (e) { /* 底板失败不影响主流程 */ }

    const tiles = new Map();
    let lastUpdate = -1e9;
    let pending = 0;
    let lastYaw = 0;
    let lastFov = (62 * Math.PI) / 180;

    function tileKey(tx, tz) { return tx + ',' + tz; }

    /* 原型色 → 叠加在贴图上的三通道增益（线性空间）。
       增益的口径与 tile 顶点色一致：`贴图 × 增益 ≈ 目标色`。
       chaos=0 → 恒 1（照片地面色原样延伸）；chaos=1 → 整块变成该生物群系的颜色。 */
    function tintOf(sector, epoch) {
      const th = themeFor(sector, epoch);
      const base = (world.groundBaseLin && world.groundBaseLin.length === 3)
        ? world.groundBaseLin : [0.18, 0.18, 0.18];
      const suit = th.proto ? th.proto.tint : null;
      const out = [1, 1, 1];
      for (let q = 0; q < 3; q++) {
        const target = suit ? clamp(suit[q] / Math.max(1e-4, base[q]), 0.2, 6) : 1;
        out[q] = lerp(1, target, live.chaos);
      }
      return out;
    }

    function buildTileMesh(tx, tz, heights, epochOfSector) {
      const x0 = tx * TS, z0 = tz * TS;
      const pos = new Float32Array(RES * RES * 3);
      const nrm = new Float32Array(RES * RES * 3);
      const uv = new Float32Array(RES * RES * 2);
      const col = new Float32Array(RES * RES * 4);
      const idx = [];
      const step = TS / (RES - 1);
      const tint = world.groundTint || [0.32, 0.32, 0.30];
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
          // 贴图密度：9m 一次重复在 40m 外已经糊成一片纯色，
          // 收到 4.5m 一次，远景还能看出地表明暗细节
          uv[k * 2] = x / 4.5; uv[k * 2 + 1] = z / 4.5;
          /* 双尺度明暗：低频（~40m）给大片地形分出层次，高频（~8m）压掉
             "一整块纯色板"的观感。第一版只有 0.26 的单层噪声，远看像塑料。 */
          const nz = vnoise(x * 0.05, z * 0.05, seed + 5) - 0.5;
          const nz2 = vnoise(x * 0.011, z * 0.011, seed + 29) - 0.5;
          const shade = 1 + nz * 0.30 + nz2 * 0.42;
          let r = 1, g = 1, b = 1;
          if (colorAt) {
            const c = colorAt(x, z);            // 线性 RGB（矩形外自动取边界值）
            const ddx = Math.max(-A - x, 0, x - A);
            const ddz = Math.max(-B - z, 0, z - 0);
            const kFar = smoothstep(30, 150, Math.hypot(ddx, ddz));
            const gin = live.enabled ? tintOf(sectorOf(x, z), epochOfSector(x, z)) : [1, 1, 1];
            for (let q = 0; q < 3; q++) {
              const gain = clamp(c[q] / base[q], 0.25, 6);
              const v = lerp(gain, 1, kFar * 0.75) * gin[q] * shade;
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
          const nx = hL - hR, ny = 2 * step, nz2 = hD - hU;
          const l = Math.hypot(nx, ny, nz2) || 1;
          nrm[k * 3] = nx / l; nrm[k * 3 + 1] = ny / l; nrm[k * 3 + 2] = nz2 / l;
        }
      }
      /* 陡坡/平地分开：陡坡（坡度 > ~40°）走顶点色，不做贴图 —— 竖直面上的
         UV 沿坡长几乎不变，贴图会被拉成一条条竖丝（悬崖"瀑布"那种假象）。
         平地保留贴图细节，两者共用同一份顶点缓冲，只是索引不同。 */
      const idxFlat = [], idxSteep = [];
      for (let j = 0; j < RES - 1; j++) {
        for (let i = 0; i < RES - 1; i++) {
          const a0 = j * RES + i, a1 = a0 + 1;
          const b0 = (j + 1) * RES + i, b1 = b0 + 1;
          /* 只保留「四边形中心落在重建矩形之外」的面。
             跨边界的 tile 会整块生成，如果不裁，靠近扇环的那一格（32m 深）
             会盖在照片几何上——alpine 湖面那道又厚又平的灰带就是这么来的。 */
          const cx = (x0 + (i + 0.5) * step), cz = (z0 + (j + 0.5) * step);
          if (cx >= -A && cx <= A && cz >= -B && cz <= 0) continue;
          const y0 = pos[a0 * 3 + 1], y1 = pos[a1 * 3 + 1];
          const y2 = pos[b0 * 3 + 1], y3 = pos[b1 * 3 + 1];
          const slope = Math.max(
            Math.abs(y1 - y0), Math.abs(y2 - y0),
            Math.abs(y3 - y2), Math.abs(y3 - y1)) / step;
          if (slope > 0.85) idxSteep.push(a0, b0, a1, a1, b0, b1);
          else idxFlat.push(a0, b0, a1, a1, b0, b1);
        }
      }
      return {
        flat: global.PW.GL.createMesh(gl, { pos, normal: nrm, uv, color: col, index: idxFlat }),
        steep: idxSteep.length
          ? global.PW.GL.createMesh(gl, { pos, normal: nrm, uv, color: col, index: idxSteep })
          : null,
        steepTris: idxSteep.length / 3
      };
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

    /* 从 extraMeshes 里摘掉某个 tile 的两条 draw（平 + 陡） */
    function dropTileMeshes(key) {
      const list = world.extraMeshes || [];
      for (const k of [key, key + ':steep']) {
        const at = list.findIndex((m) => String(m.key) === k);
        if (at >= 0) {
          disposeMesh(list[at].mesh);
          list.splice(at, 1);
        }
      }
    }

    function disposeMesh(mm) {
      if (!mm) return;
      try {
        for (const a of ['aPos', 'aNormal', 'aUV', 'aColor', 'aEdge']) {
          if (mm[a] && mm[a].buf) gl.deleteBuffer(mm[a].buf);
        }
        if (mm.index && mm.index.buf) gl.deleteBuffer(mm.index.buf);
      } catch (e) { /* ignore */ }
    }

    async function ensureTile(tx, tz) {
      const key = tileKey(tx, tz);
      if (tiles.has(key)) return;
      // 先在 Map 里占位，避免同一帧重复排队
      const epochSnapshot = live.enabled
        ? { f: (x, z) => epochs[sectorOf(x, z)] }
        : null;
      tiles.set(key, { pending: true, mesh: null, tx, tz });
      pending++;
      let heights = null;
      if (SERVER) heights = await fetchServerHeights(tx, tz);
      const parts = buildTileMesh(tx, tz, heights, epochSnapshot ? epochSnapshot.f : null);
      const rec = tiles.get(key);
      if (!rec) { pending--; return; }        // 生成期间被重掷删掉了
      rec.pending = false;
      rec.mesh = parts.flat;
      rec.steep = parts.steep;
      rec.server = !!heights;
      rec.steepTris = parts.steepTris;
      pending--;
      if (!world.extraMeshes) world.extraMeshes = [];
      world.extraMeshes.push({ key, mesh: parts.flat, tex: world.tex.ground, fog: 0.22, mode: 4 });
      if (parts.steep) {
        world.extraMeshes.push({ key: key + ':steep', mesh: parts.steep, tex: null, fog: 0.20, mode: 1 });
      }
      // 上限按 tile 数算（一个 tile 最多两条 draw）
      const tileCount = Math.ceil(world.extraMeshes.length / 2);
      if (tileCount > MAX_TILES) {
        const first = world.extraMeshes.shift();
        if (first) {
          const base = String(first.key).replace(/:steep$/, '');
          tiles.delete(base);
          // 同一 tile 的陡坡网格可能紧跟在后面，一并摘掉
          if (world.extraMeshes.length && String(world.extraMeshes[0].key) === base + ':steep') {
            const s2 = world.extraMeshes.shift();
            disposeMesh(s2.mesh);
          }
          disposeMesh(first.mesh);
        }
      }
    }

    /* 视锥判定：返回 { inFov, dist }。fov 是水平视场（rad）。 */
    function fovTest(wx, wz, px, pz, fx, fz, fov, margin) {
      const vx = wx - px, vz = wz - pz;
      const d = Math.hypot(vx, vz);
      const dot = d > 1e-4 ? (vx * fx + vz * fz) / d : 1;
      const ang = Math.acos(clamp(dot, -1, 1));
      return { inFov: ang < fov * 0.5 + margin, dist: d };
    }

    /* 记忆衰减：视锥外的扇区按 (1-memory)^2 的概率每秒重掷一次。
       直接删 tile 而不是就地重建 —— 它们本来就在视野外，等转回来时用新纪元
       重新生成，既省 CPU（不重建看不见的几何），也正是「身后景象迅速转变」的来源。 */
    function decayMemory(px, pz, fx, fz, fov, dt) {
      if (!live.enabled || live.memory >= 0.995) return 0;
      const rate = Math.pow(1 - live.memory, 2) * 1.8;   // 次/秒/扇区
      if (rate <= 0.001) return 0;
      let changed = 0;
      for (let s = 0; s < NS; s++) {
        if (changed >= 3) break;                         // 每拍最多重掷 3 个扇区
        // 用扇区中线上 70m 处的一点判断"玩家现在看不看得到这片区域"
        const ang = ((s + 0.5) / NS) * TAU - Math.PI;
        const wx = Math.sin(ang) * 70, wz = -Math.cos(ang) * 70;
        const t = fovTest(wx, wz, px, pz, fx, fz, fov, 0.35);
        if (t.inFov) continue;                           // 眼前不动
        if (Math.random() > rate * dt) continue;
        epochs[s] = (epochs[s] + 1) | 0;
        rerolls++;
        changed++;
        // 删掉该扇区内离玩家足够远的 tile（脚底下那块永不删）
        for (const [key, rec] of tiles) {
          if (rec.pending) continue;
          const cx = rec.tx * TS + TS / 2, cz = rec.tz * TS + TS / 2;
          if (sectorOf(cx, cz) !== s) continue;
          if (Math.hypot(cx - px, cz - pz) < TS * 1.8) continue;
          tiles.delete(key);
          dropTileMeshes(key);
        }
      }
      return changed;
    }

    /* 每帧调用：节流 + 只补视锥内缺失的 tile（转身时才补身后）。
       view = { yaw, fov }：相机朝向与水平视场，缺省时沿用上一次。 */
    function update(px, pz, now, view) {
      world._px = px; world._pz = pz;
      const t = now === undefined ? (global.performance ? global.performance.now() : Date.now()) : now;
      if (view && typeof view.yaw === 'number') lastYaw = view.yaw;
      if (view && typeof view.fov === 'number' && view.fov > 0.2) lastFov = view.fov;
      if (t - lastUpdate < (live.enabled ? 160 : 260)) return 0;
      const dt = clamp((t - lastUpdate) / 1000, 0.05, 1.2);
      lastUpdate = t;

      const fx = -Math.sin(lastYaw), fz = -Math.cos(lastYaw);
      const cx = Math.round(px / TS), cz = Math.round(pz / TS);
      /* 视野内要生成到雾的尽头：12 格 = 384m。以前的 9 格在雾还没吃掉之前就断供，
         正前方会出现一条「突然没有地形」的边界，走起来像空气墙。
         非实时模式也放宽到 8 格（256m）。 */
      const scan = live.enabled ? Math.max(RADIUS, 12) : Math.max(RADIUS, 8);

      const cand = [];
      for (let dz = -scan; dz <= scan; dz++) {
        for (let dx = -scan; dx <= scan; dx++) {
          const tx = cx + dx, tz = cz + dz;
          const key = tileKey(tx, tz);
          if (tiles.has(key)) continue;
          const x0 = tx * TS, z0 = tz * TS, x1 = x0 + TS, z1 = z0 + TS;
          if (x0 >= -A && x1 <= A && z0 >= -B && z1 <= 0) continue;   // 照片几何覆盖区
          const wx = x0 + TS / 2, wz = z0 + TS / 2;
          const ft = fovTest(wx, wz, px, pz, fx, fz, lastFov, live.enabled ? 0.30 : Math.PI);
          const near = ft.dist <= TS * 1.6;
          // 实时模式：视锥外不预生成（转身/走近时才补），省下的算力给视野内
          if (live.enabled && !ft.inFov && !near) continue;
          cand.push({ tx, tz, dist: ft.dist, inFov: ft.inFov ? 1 : 0 });
        }
      }
      // 视锥内优先，其次由近到远：玩家往前走时，正前方先补齐
      cand.sort((a, b) => (b.inFov - a.inFov) || (a.dist - b.dist));

      // 追赶预算：视野内缺口越大，单拍补得越多（每块 ~1-3ms，8 块仍在一帧预算内）
      const missing = cand.filter((c) => c.inFov).length;
      const budget = live.enabled ? (missing > 8 ? 8 : 4) : 3;
      let queued = 0;
      for (const c of cand) {
        ensureTile(c.tx, c.tz);
        if (++queued >= budget) break;
      }

      decayMemory(px, pz, fx, fz, lastFov, dt);
      return queued;
    }

    /* 运行时改参数。chaos 改变会改变高度场：把视锥外的 tile 全部作废，
       让它们按新主题重新生成（眼前的不动，避免脚下的地形突然抬高）。 */
    function setLive(p) {
      p = p || {};
      const prevChaos = live.chaos;
      if (typeof p.enabled === 'boolean') live.enabled = p.enabled;
      if (typeof p.chaos === 'number') live.chaos = clamp(p.chaos, 0, 1);
      if (typeof p.memory === 'number') live.memory = clamp(p.memory, 0, 1);
      const chaosChanged = Math.abs(live.chaos - prevChaos) > 0.01;
      if (chaosChanged || (p.enabled === false)) {
        const px = world._px || 0, pz = world._pz || 0;
        for (const [key, rec] of tiles) {
          const cx = rec.tx * TS + TS / 2, cz = rec.tz * TS + TS / 2;
          if (Math.hypot(cx - px, cz - pz) < TS * 1.8) continue;   // 脚下保留
          tiles.delete(key);
          dropTileMeshes(key);
        }
      }
      return getLive();
    }

    function getLive() {
      return {
        enabled: live.enabled, chaos: live.chaos, memory: live.memory,
        sectors: NS, epochs: Array.from(epochs), rerolls,
        tiles: tiles.size, meshes: (world.extraMeshes || []).length
      };
    }

    function stats() {
      return {
        tiles: tiles.size, meshes: (world.extraMeshes || []).length, pending,
        server: !!SERVER, live: live.enabled, chaos: live.chaos, memory: live.memory,
        rerolls
      };
    }

    function dispose() {
      for (const rec of tiles.values()) { disposeMesh(rec.mesh); disposeMesh(rec.steep); }
      world.extraMeshes = [];
      tiles.clear();
    }

    return {
      update, heightAt, stats, dispose, setLive, getLive,
      tileSize: TS, sectors: NS
    };
  }

  global.PW = global.PW || {};
  global.PW.createExtender = createExtender;
  global.PW.extendInternals = { fbm, vnoise, hash2 };
})(window);
