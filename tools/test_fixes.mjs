#!/usr/bin/env node
/**
 * 修复回归（v2.15.6）：报告 P0/P1 逐条断言 + 深度后处理单元测试。
 *
 *   node tools/test_fixes.mjs
 *
 * 纪律（报告 §4.1）：每个修复必须带**可运行断言**；"已修复"必须能指到断言。
 * 所以这里不写"看截图"式的软检查，全部是真的会 fail 的硬断言。
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = (f) => fs.readFileSync(path.join(ROOT, f), 'utf8');
let pass = 0;
const ok = (msg) => { console.log('PASS  ' + msg); pass++; };

/* ================= 1) 静态断言：确认修复没被回退 ================= */

// P0-5 buildSkirt TDZ：调用点必须晚于 edgeCol 声明，且失败不能静默
{
  const src = read('js/world.js');
  const iEdge = src.indexOf('const edgeCol = new Float32Array(EDGE_N * 3);');
  const iCall = src.indexOf('skirt = buildSkirt();');
  assert.ok(iEdge > 0, '找不到 edgeCol 声明');
  assert.ok(iCall > 0, '找不到 buildSkirt 调用');
  assert.ok(iCall > iEdge,
    'P0-5 回归：buildSkirt() 调用仍在 edgeCol 声明之前（TDZ → 裙边永久失效）');
  assert.ok(!/catch\s*\(\s*e\s*\)\s*\{\s*skirt\s*=\s*null;\s*\}/.test(src),
    'P0-5 回归：buildSkirt 的 catch 仍在静默吞异常');
  ok('P0-5  地面裙边不再命中 TDZ，失败会告警');
}

// P0-8 yGain 重复键
{
  const src = read('js/app.js');
  const n = (src.match(/^\s*yGain[:,]/gm) || []).length;
  assert.equal(n, 1, 'P0-8 回归：opts 里 yGain 键出现 ' + n + ' 次（重复键会静默覆盖，?yg= 再次失效）');
  ok('P0-8  ?yg= / ?ygain= 别名只有一个键，不再互相覆盖');
}

// P0-9 tex.ground 恢复
{
  const src = read('js/app.js');
  assert.ok(/texGround:\s*world\.tex \? world\.tex\.ground : null/.test(src),
    'P0-9 回归：飞行时没有把 tex.ground 存进 _stash');
  assert.ok(/world\._stash\.texGround\)\s*world\.tex\.ground = world\._stash\.texGround/.test(src),
    'P0-9 回归：落地时没有恢复 tex.ground（地面会变黑白板）');
  ok('P0-9  飞行/落地之间 tex.ground 可恢复');
}

// P0-7 飞行时门控真点云（不是我修了 snow 粒子）
{
  const src = read('js/renderer.js');
  assert.ok(/world\.showPlan\s*\)\s*\{\s*\/\*[\s\S]{0,400}?\}\s*else if \(world\.splatMode/.test(src),
    'P0-7 回归：splat 层没有 showPlan 门控（空中仍会看到放射尖刺）');
  assert.ok(/points2Mesh && !skip\('shell'\) && !world\.showPlan/.test(src),
    'P0-7 回归：points2 点云没有 showPlan 门控');
  ok('P0-7  俯视/飞行时 splat 与 points2 都不再绘制');
}

// P0-6 ldi.py 点云分支不得引用未赋值的 seg
{
  const src = read('pipeline/v2/ldi.py');
  const iPoints = src.indexOf('if use_points:');
  const iSegAssign = src.indexOf('seg = seg_labels[idx_seg[0], idx_seg[1]]');
  const iSegUse = src.indexOf('_g2.shape != seg.shape');
  assert.ok(iPoints > 0 && iSegAssign > 0, 'ldi.py 结构变了，请更新断言');
  assert.ok(iSegUse === -1 || iSegUse > iSegAssign,
    'P0-6 回归：点云分支仍在 seg 赋值之前引用 seg.shape（UnboundLocalError）');
  ok('P0-6  点云路径的地面层不再抛 UnboundLocalError');
}

// 标定退化：**钳位 + 告警**，不能 return null（v2.15.14 事故）
{
  const src = read('js/world.js');
  assert.ok(/rFarClamped = Math\.min\(rNear \* 0\.85/.test(src),
    '回归：标定的 rFar 钳位被去掉了（它是深度→米曲线的承重件，去掉会把世界压平）');
  assert.ok(!/rFarRaw >= rNear \* 0\.85[\s\S]{0,200}?return null/.test(src),
    '回归：标定退化又改成 return null（v2.15.6 就是这么把山塌成平地的）');
  assert.ok(/rFarRaw >= rNear \* 0\.85/.test(src), '缺少退化告警判据');
  ok('标定退化 = 钳位 + 告警（不再 return null）');
}

// P1-4 行基线必须是真中值
{
  const src = read('js/world.js');
  assert.ok(!/rowD\[j\] = s \/ GW;/.test(src),
    'P1-4 回归：行基线又变回均值');
  assert.ok(/rowD\[j\] = buf\[\(GW - 1\) >> 1\]/.test(src),
    'P1-4 回归：行基线不是排序后的中值');
  ok('P1-4  行基线是真中值');
}

// P1-7 语义映射：桌/柜/箱/冰箱/桶不得划进地形或水面
{
  const src = read('js/semantics.js');
  const mTerrain = src.match(/mark\(GROUP\.TERRAIN, \[([^\]]*)\]\)/);
  const mWater = src.match(/mark\(GROUP\.WATER, \[([^\]]*)\]\)/);
  const mObj = src.match(/mark\(GROUP\.OBJECT, \[([\s\S]*?)\]\)/);
  assert.ok(mTerrain && mWater && mObj, 'semantics.js 的 mark() 结构变了');
  const list = (s) => s.split(',').map((v) => parseInt(v.trim(), 10)).filter(Number.isFinite);
  const terr = list(mTerrain[1]), wat = list(mWater[1]), obj = list(mObj[1]);
  for (const id of [33, 41, 44, 50]) {
    assert.ok(!terr.includes(id), 'P1-7 回归：类别 ' + id + ' 又回到 TERRAIN（室内桌柜会被压平）');
    assert.ok(obj.includes(id), 'P1-7 回归：类别 ' + id + ' 不在 OBJECT');
  }
  assert.ok(!wat.includes(111), 'P1-7 回归：111 barrel 又回到 WATER');
  assert.ok(obj.includes(111), 'P1-7 回归：111 barrel 不在 OBJECT');
  ok('P1-7  desk/box/chest/refrigerator/barrel 归到 OBJECT');
}

// 报告 §4.1 纪律：新增代码不得裸 catch 吞异常
{
  const src = read('js/world.js') + read('js/depthpost.js');
  const bad = src.match(/catch\s*\(\s*\w+\s*\)\s*\{\s*\}/g) || [];
  assert.equal(bad.length, 0, 'world.js / depthpost.js 有 ' + bad.length + ' 处空 catch');
  ok('纪律  新增路径没有空 catch');
}

/* ================= 2) 深度后处理单元测试 ================= */
const ctx = (() => {
  const s = { console };
  s.window = s; s.global = s;
  vm.createContext(s);
  vm.runInContext(read('js/depthpost.js'), s, { filename: 'depthpost.js' });
  return s;
})();
const DP = ctx.PW.depthPost;
assert.ok(DP && typeof DP.process === 'function', 'depthpost.js 未挂到 PW.depthPost');

const W = 160, Hgt = 100;
function cliffScene() {
  // 背景 60m、中间一块 8m 的岩壁（竖直断崖），底部地面从 8m 渐变到 60m
  const z = new Float32Array(W * Hgt);
  const lum = new Float32Array(W * Hgt);
  for (let y = 0; y < Hgt; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const inRock = x >= 55 && x <= 105 && y >= 20 && y <= 75;
      z[i] = inRock ? 8 : 60;
      lum[i] = inRock ? 0.25 : 0.85;
    }
  }
  // 岩壁边缘 6px 的"混合值"——正是真实深度图在遮挡边界上的表现
  for (let y = 20; y <= 75; y++) {
    for (let d = 1; d <= 6; d++) {
      const ramp = 8 + (44 * d) / 7;
      for (const x of [55 - d, 105 + d]) {
        if (x < 0 || x >= W) continue;
        z[y * W + x] = ramp;
        lum[y * W + x] = 0.25 + 0.6 * (d / 7);
      }
    }
  }
  return { z, lum };
}

{
  const { z, lum } = cliffScene();
  const alpha = new Uint8Array(W * Hgt).fill(1);
  const res = DP.process({ depth: z, lum, alpha, W, H: Hgt, opt: {} });
  assert.ok(res.stats.band > 100, '边缘带没检出：' + res.stats.band + ' px');
  assert.ok(res.stats.near > 0 && res.stats.far > 0,
    '近侧/远侧归属失败：near=' + res.stats.near + ' far=' + res.stats.far);

  // 核心判据：混合斜坡必须被"压扁"——带内不再出现 8m 与 60m 之间的一串中间值
  let mid = 0;
  for (let i = 0; i < W * Hgt; i++) {
    if (!res.side[i]) continue;
    if (res.depth[i] > 12 && res.depth[i] < 52) mid++;
  }
  assert.equal(mid, 0, '重切分后带内仍残留 ' + mid + ' 个中间深度（斜坡片没被压扁）');

  // 近侧像素必须真的落在 [zn, 1.03·zn]
  let badNear = 0;
  for (let i = 0; i < W * Hgt; i++) {
    if (res.side[i] !== 1) continue;
    const v = res.depth[i];
    if (!(v >= 7.9 && v <= 8.35)) badNear++;
  }
  assert.equal(badNear, 0, '近侧像素没有压进薄层：' + badNear + ' 个越界');
  ok('方案一  边界带 ' + res.stats.band + ' px 重切分（近 ' + res.stats.near
    + ' / 远 ' + res.stats.far + '），中间深度清零');

  // 双层窄带：轮廓内侧必须发得出来，且深度必须比近侧远
  assert.ok(res.layer2.count > 0, '第二层点数为 0（横移会露黑洞）');
  let badLayer = 0;
  for (let q = 0; q < res.layer2.count; q++) {
    const i = res.layer2.idx[q];
    if (!(res.layer2.z[q] > res.depth[i] * 2)) badLayer++;
  }
  assert.equal(badLayer, 0, '第二层点有 ' + badLayer + ' 个不比前景远');
  ok('方案一  第二层窄带 ' + res.layer2.count + ' 点，深度均显著远于前景');
}

// 一致性：同一输入两次运行结果必须逐元素相同（抖动是确定性哈希）
{
  const { z, lum } = cliffScene();
  const alpha = new Uint8Array(W * Hgt).fill(1);
  const a = DP.process({ depth: z, lum, alpha, W, H: Hgt, opt: {} });
  const b = DP.process({ depth: z, lum, alpha, W, H: Hgt, opt: {} });
  let diff = 0;
  for (let i = 0; i < a.depth.length; i++) if (a.depth[i] !== b.depth[i]) diff++;
  assert.equal(diff, 0, '重切分不是确定性的：' + diff + ' 个像素两次结果不同');
  ok('方案一  重切分确定性（同图重载条纹位置不跳）');
}

// 退化输入：平坦深度不得被"切"出假边缘
{
  const z = new Float32Array(W * Hgt).fill(30);
  const lum = new Float32Array(W * Hgt).fill(0.5);
  const alpha = new Uint8Array(W * Hgt).fill(1);
  const res = DP.process({ depth: z, lum, alpha, W, H: Hgt, opt: {} });
  assert.equal(res.stats.band, 0, '平坦深度被判出 ' + res.stats.band + ' 个边缘像素');
  assert.equal(res.layer2.count, 0, '平坦深度生成了第二层点');
  ok('方案一  平坦深度零副作用');
}

/* ================= 3) 端到端：buildCartesianWorld 能跑通且裙边不再为空 ================= */
{
  const ctx2d = {
    fillRect() {}, drawImage() {}, putImageData() {}, fillStyle: '',
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    getImageData: (x, y, w, h) => {
      const d = new Uint8ClampedArray(w * h * 4);
      for (let i = 0; i < w * h; i++) {
        d[i * 4] = 120; d[i * 4 + 1] = 130; d[i * 4 + 2] = 140; d[i * 4 + 3] = 255;
      }
      return { data: d };
    },
  };
  const s = {
    console,
    document: { currentScript: { src: 'http://localhost/js/world.js' }, createElement: () => ({ getContext: () => ctx2d, width: 0, height: 0 }) },
    location: { href: 'http://localhost/' }, performance: { now: () => 0 },
    setTimeout, clearTimeout,
  };
  s.window = s; s.global = s;
  vm.createContext(s);
  for (const f of ['js/mat.js', 'js/reconstruct.js', 'js/depthpost.js']) {
    vm.runInContext(read(f), s, { filename: f });
  }
  s.PW.GL = { createTexture: () => ({}), createMesh: () => ({}) };
  const src = read('js/world.js').replace(
    'global.PW.world = { build, makeHeightField, panoU };',
    'global.PW.world = { build, makeHeightField, panoU, buildCartesianWorld };');
  vm.runInContext(src, s, { filename: 'world.js' });

  const NC = 360;
  const top = new Float32Array(NC);
  for (let i = 0; i < NC; i++) top[i] = 0.35 + 0.05 * Math.sin(i * 0.05);
  const A2 = {
    name: 'e2e', W: 640, H: 400, top, topRaw: top,
    heights: new Float32Array(NC).fill(0.5), horizon: 0.42,
    sunAz: 3.1, sunElev: 0.4, sunLum: 0.7, isSnow: false, isGreen: true,
    brightness: 0.5, meanLum: 0.45, palette: [[0.5, 0.5, 0.5]],
    ground: [0.4, 0.4, 0.4], skyTop: [0.4, 0.6, 0.9],
    skyHorizon: [0.7, 0.7, 0.7], fog: [0.7, 0.7, 0.7],
    canvas: { width: 640, height: 400, getContext: () => ctx2d },
  };
  // 深度：中部一块近景岩壁 + 下半渐变地面
  const DW = 518, DD = new Float32Array(DW * DW);
  for (let y = 0; y < DW; y++) {
    for (let x = 0; x < DW; x++) {
      const inRock = x > 170 && x < 300 && y > 90 && y < 300;
      DD[y * DW + x] = inRock ? 0.72 : (y / DW) * 0.6 + 0.05;
    }
  }
  const r = s.PW.world.buildCartesianWorld(A2, { data: DD, width: DW, height: DW }, {
    sem: null, fgMask: null, gw: 320, gh: 200,
  });
  assert.ok(r && r.pos && r.pos.length > 0, 'buildCartesianWorld 没有产出顶点');
  assert.ok(r.tris > 0, '没有生成三角面');
  assert.ok(r.skirt && r.skirt.pos && r.skirt.pos.length > 0,
    'P0-5：地面裙边仍为空（TDZ 没修好）');
  assert.ok(r.postStats && r.postStats.band > 0,
    '深度后处理没有在端到端路径生效（postStats=' + JSON.stringify(r.postStats) + '）');
  assert.ok(r.layer2 && r.layer2.count > 0, '端到端路径没有产出第二层点');
  /* v2.15.14 回归：标定**必须**钳位而不是返回 null。
     v2.15.6 曾把"远锚不够远"改成 `return null`，结果整条深度→米曲线换成经验
     兜底（近端全部夹到 2.6m）→ 山体塌成平地，而 32 机位指标几乎不动。
     这里用"远近深度几乎相同"的合成图触发那条判据，断言仍拿到了标定参数。 */
  {
    const flatD = new Float32Array(DW * DW).fill(0.5);
    const rf = s.PW.world.buildCartesianWorld(A2, { data: flatD, width: DW, height: DW }, {
      sem: null, fgMask: null, gw: 160, gh: 100,
    });
    assert.ok(rf && rf.calibration, '标定退化时不能返回 null（会把整个世界的纵深压平）');
    assert.ok(rf.calibration.p > 0.5 && rf.calibration.p <= 3.2,
      '标定指数越界：' + (rf.calibration && rf.calibration.p));
  }
  ok('端到端  buildCartesianWorld：' + r.tris + ' 三角、裙边 '
    + (r.skirt.pos.length / 3) + ' 顶点、后处理带 ' + r.postStats.band
    + ' px、第二层 ' + r.layer2.count + ' 点');
}

console.log('\n全部通过：' + pass + ' 项');
