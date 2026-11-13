#!/usr/bin/env node
/**
 * 无浏览器回归：前端几何 + 深度后处理。
 *
 *   node tools/test_regression.mjs
 *   node tools/test_regression.mjs --depth /path/to/real.f32  # 附带真实深度图
 *
 * 覆盖过去两次事故：
 *  1) build() 深度模式曾因 `opt.ring`（未定义变量）直接抛 ReferenceError，
 *     模型加载成功反而进不去世界 —— 静态断言 + 沙箱加载；
 *  2) 前拉约束的三轮腐蚀把地面顶点吸离 y=0（实测 1083 个、最大 1.64m）——
 *     对平滑深度 / 断崖深度 / 可选真实深度逐顶点验证。
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = (f) => fs.readFileSync(path.join(ROOT, f), 'utf8');
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, v, i, a) => {
  if (v.startsWith('--')) acc.push([v.slice(2), a[i + 1]]);
  return acc;
}, []));

function sandbox() {
  const ctx2d = {
    fillRect() {}, drawImage() {}, putImageData() {}, fillStyle: '',
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    getImageData: (x, y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
  };
  const s = {
    console, URL, setTimeout, clearTimeout, fetch: () => Promise.reject(new Error('no network')),
    document: {
      currentScript: { src: 'http://localhost/js/depth.js' },
      createElement: () => ({ width: 0, height: 0, getContext: () => ctx2d }),
    },
    location: { href: 'http://localhost/' },
    performance: { now: () => 0 },
  };
  s.window = s;
  s.global = s;
  vm.createContext(s);
  return s;
}

const worldSrc = read('js/world.js');

// ---- 1) P0：深度模式主路径不得再出现裸的 opt.ring ----
assert.ok(!/\bopt\.ring\b/.test(worldSrc), 'world.js 仍有未定义的 opt.ring');
assert.ok(/opts\s*&&\s*opts\.ring/.test(worldSrc), 'ring 开关必须读 opts');
console.log('PASS  P0  build() 的 ring 开关读 opts');

// ---- 2) 加载真实模块，导出 buildRelief 与内部统计 ----
const NCOL = 360, NR = 200;
const analysis = {
  name: 'regression', W: 1536, H: 968,
  top: new Float32Array(NCOL).fill(0.42),
  heights: new Float32Array(NCOL).fill(0.5),
  horizon: 0.52, sunAz: 3.14, sunElev: 0.4, sunLum: 0.7,
  isSnow: false, isGreen: true, brightness: 0.5, meanLum: 0.45,
  palette: [[0.5, 0.5, 0.5]], ground: [0.4, 0.4, 0.4],
  skyTop: [0.4, 0.6, 0.9], skyHorizon: [0.7, 0.7, 0.7], fog: [0.7, 0.7, 0.7],
};
const injected = worldSrc
  .replace('/* 2) 前拉约束：', 'const gT_before = gT.slice();\n    /* 2) 前拉约束：')
  .replace('/* 3) 顶点 / UV / 法线 / 顶点色 */',
    'global.__stats = { before: gT_before, after: gT.slice(), gy: gy.slice(), gV, W1, NR, vfov };\n'
    + '    /* 3) 顶点 / UV / 法线 / 顶点色 */')
  .replace('global.PW.world = { build, makeHeightField, panoU };',
    'global.PW.world = { build, buildRelief, makeHeightField, panoU };');
const ctx = sandbox();
for (const f of ['mat.js', 'reconstruct.js']) {
  vm.runInContext(read('js/' + f), ctx, { filename: f });
}
ctx.PW = ctx.PW || {};
ctx.PW.GL = { createTexture: () => ({}), createMesh: () => ({}) };
vm.runInContext(injected, ctx, { filename: 'world.js' });
assert.equal(typeof ctx.PW.world.buildRelief, 'function', 'buildRelief 未导出到沙箱');

/* v1.11.0：地面不再被硬钉在 y=0（那是「照片平铺在地上」的根因之一），
   改为有上限的近景挤出；同时新增「纵深是否被压平」的量化判据：
   旧实现把 46m 外全部 lerp 到 128m 固定圆柱，p90/p10 会趋近 1、
   128m 顶点占比接近 100%。 */
function reliefStats(dm, opt) {
  const r = ctx.PW.world.buildRelief(analysis, dm, opt || {});
  const s = ctx.__stats;
  let groundN = 0, maxAbsY = 0, pulledGround = 0, pulledSky = 0;
  let cyl = 0, skyN = 0;
  const ts = [];
  for (let j = 0; j <= s.NR; j++) {
    const sp = Math.sin((0.5 - s.gV[j]) * s.vfov);
    for (let i = 0; i < s.W1; i++) {
      const k = j * s.W1 + i;
      const pulled = Math.abs(s.after[k] - s.before[k]) > 1e-3;
      if (sp < -0.002) {
        groundN++;
        maxAbsY = Math.max(maxAbsY, Math.abs(s.gy[k]));
        if (pulled) pulledGround++;
      } else {
        skyN++;
        ts.push(s.after[k]);
        if (Math.abs(s.after[k] - 128) < 1) cyl++;
        if (pulled) pulledSky++;
      }
    }
  }
  ts.sort((a, b) => a - b);
  return {
    r, groundN, maxAbsY, pulledGround, pulledSky,
    p10: ts[Math.floor(ts.length * 0.1)], p90: ts[Math.floor(ts.length * 0.9)],
    cylRatio: cyl / Math.max(1, skyN)
  };
}

function checkRelief(tag, dm) {
  const s = reliefStats(dm, {});
  const r = s.r;
  assert.ok(r.tris > 0, tag + '：没有生成三角面');
  assert.equal(s.pulledGround, 0, tag + '：地面顶点仍被前拉吸附');
  // 平坦水面/地面必须保持水平（旧版满屏波纹条纹就是这里没做死区+低通）；
  // 带近景断崖的照片则必须真的挤出物体，否则「平铺在地面上」没被解决。
  if (tag.indexOf('断崖') >= 0) {
    assert.ok(s.maxAbsY >= 0.10, tag + '：近景挤出未生效（地面仍是死平的）');
  } else {
    assert.ok(s.maxAbsY <= 0.05, tag + '：平坦地面产生了波纹起伏 ' + s.maxAbsY + 'm');
  }
  assert.ok(s.maxAbsY <= 1.5, tag + '：地面起伏超出上限 ' + s.maxAbsY + 'm');
  assert.ok(r.calibration && r.calibration.p >= 0.6 && r.calibration.p <= 3.2,
    tag + '：双锚点物理标定未生效');
  // 天空上缘很高时可见区域可能全部落在 70m 内，此时远景层为空是正确行为；
  // 分层逻辑本身由下面的「深天空」用例单独验证。
  assert.ok(!r.backdrop || r.backdrop.tris > 0, tag + '：远景层三角数异常');
  const spread = s.p90 / Math.max(1e-3, s.p10);
  assert.ok(spread > 4, tag + '：纵深仍被压平（p90/p10=' + spread.toFixed(2) + '）');
  assert.ok(s.cylRatio < 0.10,
    tag + '：仍有 ' + (s.cylRatio * 100).toFixed(1) + '% 顶点被压在 128m 圆柱上');
  console.log(`PASS  P1  几何 [${tag}]：地面 ${s.groundN} 顶点 max|y|=${s.maxAbsY.toFixed(3)}m，`
    + `纵深 p10=${s.p10.toFixed(1)}m p90=${s.p90.toFixed(1)}m (×${spread.toFixed(1)})，`
    + `128m 圆柱占比 ${(s.cylRatio * 100).toFixed(2)}%，远层 ${r.backdrop ? r.backdrop.tris : 0} 三角，`
    + `吸附 地面 ${s.pulledGround} / 非地面 ${s.pulledSky}`);
  return { pulledSky: s.pulledSky };
}

const SIZE = 518;
function synth(cliff) {
  const d = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      let v = y / SIZE * 0.9 + 0.05 * Math.sin(x * 0.03);
      if (cliff && x > 180 && x < 320 && y > 120 && y < 300) v += 0.55;  // 一块近景
      d[y * SIZE + x] = v;
    }
  }
  return { data: d, width: SIZE, height: SIZE, size: SIZE };
}
checkRelief('平滑深度', synth(false));
const cliff = checkRelief('带断崖深度', synth(true));
assert.ok(cliff.pulledSky > 0, '断崖处应仍有单侧补洞，否则几何会拉出"鳍"');

// ---- 2a) 天际线更高的照片：远景区可见，必须真的拆出跟随相机的 backdrop 层 ----
{
  const deep = Object.assign({}, analysis, { top: new Float32Array(NCOL).fill(0.02) });
  const rr = ctx.PW.world.buildRelief(deep, synth(false), {});
  assert.ok(rr.backdrop && rr.backdrop.tris > 0, '远景可见时应拆出 backdrop 层');
  assert.ok(rr.tris > 0, '近景层不应被拆空');
  console.log(`PASS  P1  近/远分层：近景 ${rr.tris} 三角 / 远景 ${rr.backdrop.tris} 三角`);
}

// ---- 2b) ?v1=1 必须能完整复现旧路径（地面死平 / 无远层 / 无物理标定）----
{
  const s = reliefStats(synth(false), { legacy: true });
  assert.ok(s.maxAbsY < 1e-3, 'legacy 地面应严格 y=0，实测 ' + s.maxAbsY + 'm');
  assert.equal(s.r.backdrop, null, 'legacy 不应拆远景层');
  assert.equal(s.r.calibration, null, 'legacy 不应走物理标定');
  console.log('PASS  P1  ?v1=1 旧路径可复现（地面 y=0 / 无远层 / 无标定）');
}

if (args.depth && fs.existsSync(args.depth)) {
  const buf = fs.readFileSync(args.depth);
  const d = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
  checkRelief('真实 ONNX 深度', { data: d, width: SIZE, height: SIZE, size: SIZE });
}

// ---- 3) 深度预处理：等比 padding 规划与后处理 ----
const ds = sandbox();
ds.PW = {};
vm.runInContext(read('js/depth.js'), ds, { filename: 'depth.js' });
const { squarePlan, postprocess } = ds.PW.depth.__internals;
assert.equal(typeof postprocess, 'function', 'depth.js 未暴露 __internals');

const plan = squarePlan(1536, 968, SIZE);
// vm 里的对象来自另一个 realm，不能用 deepStrictEqual 比原型
assert.equal([plan.ox, plan.oy, plan.nw, plan.nh].join(','), '0,96,518,326');
console.log('PASS  P1  等比 padding：1536x968 → 518x326 居中（不再拉伸）');

const makeRaw = (sign) => {
  const raw = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const v = (sign > 0 ? y / SIZE : 1 - y / SIZE) * 10 + 0.05 * Math.sin(x * 0.04);
      raw[y * SIZE + x] = v;
    }
  }
  return raw;
};
for (const sign of [1, -1]) {
  const pp = postprocess(makeRaw(sign), [1, 1, SIZE, SIZE], plan);
  assert.equal(pp.width, 518);
  assert.equal(pp.height, 326);
  assert.equal(pp.data.length, 518 * 326);
  assert.ok(pp.corr > 0.9, '输出必须统一成「越往下越近」');
  let lo = Infinity, hi = -Infinity;
  for (const v of pp.data) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  assert.ok(lo >= 0 && hi <= 1 && hi - lo > 0.9, '分位归一化应覆盖 0..1');
  console.log(`PASS  P1  后处理 [极性 ${sign > 0 ? '正常' : '反转'}]：`
    + `inverted=${pp.inverted} flipped=${pp.flipped} corr=${pp.corr.toFixed(3)}`);
}
const outlier = makeRaw(1);
outlier[0] = 1e6;                         // 单个离群值不应再压扁整张深度图
const ppOut = postprocess(outlier, [1, 1, SIZE, SIZE], plan);
const sorted = Float32Array.from(ppOut.data).sort();
assert.ok(sorted[Math.floor(sorted.length * 0.5)] > 0.3,
  '全图 min-max 回归：离群值把中位深度压到 ' + sorted[Math.floor(sorted.length * 0.5)]);
console.log('PASS  P1  2/98 分位截断对离群值稳健，中位深度 '
  + sorted[Math.floor(sorted.length * 0.5)].toFixed(3));

// ---- 4) P3：重复定义 / 死代码 / asset 相机一致性 ----
const appSrc = read('js/app.js');
const depthSrc = read('js/depth.js');
const geomSrc = read('pipeline/geom.py');

const skyDefs = (worldSrc.match(/function makeSkyTexture\(/g) || []).length;
assert.equal(skyDefs, 1,
  'makeSkyTexture 仍有 ' + skyDefs + ' 份定义，后者会静默覆盖前者');
assert.ok(!/const w = smoothstep\(-0\.03, 0\.02, sp\)/.test(worldSrc),
  'buildRelief 里仍留有算完不用的死变量 w');
assert.ok(!/skyBottom/.test(worldSrc), 'build() 里仍留有算完不用的 skyBottom');
assert.ok(!/^\s*idx = \[\]/m.test(geomSrc), 'geom.build_mesh 里仍留有死变量 idx');
assert.ok(!/def estimate_pitch|def _fit_scale/.test(geomSrc),
  'geom.py 里仍有主流程不调用的历史标定函数');
assert.ok(!/eye: \[0, 1\.7, 0\]/.test(appSrc),
  '曝光预热相机仍写死 y=1.7，与 asset 模式烘出的地面高度不一致');
assert.ok(/groundY \+ world\.eyeHeight/.test(appSrc), '曝光预热应使用真实眼位');
assert.ok(!/if \(x < -half \|\| x > half \|\| z < -half \|\| z > half\) return 0;/.test(appSrc),
  '高度图 at() 越界仍直接返回 0');
assert.ok(/CACHE\.get\(key\)/.test(depthSrc), 'estimate 未接深度图缓存');
console.log('PASS  P3  重复函数 / 死代码已清理，曝光相机与高度图边界一致');

// ---- 5) P3：深度图缓存真的省掉第二次推理（mock ONNX session） ----
const cs = sandbox();
cs.PW = {};
cs.fetch = () => Promise.resolve({ ok: true, arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)) });
let runs = 0;
cs.ort = {
  env: { wasm: {} },
  Tensor: function (type, data, dims) { this.type = type; this.data = data; this.dims = dims; },
  InferenceSession: {
    create: async () => ({
      inputNames: ['input'], outputNames: ['output'],
      run: async () => {
        runs++;
        return { output: { data: new Float32Array(SIZE * SIZE), dims: [1, 1, SIZE, SIZE] } };
      }
    })
  }
};
vm.runInContext(read('js/depth.js'), cs, { filename: 'depth.js' });
const fakeImg = { naturalWidth: 1536, naturalHeight: 968, currentSrc: 'http://localhost/worlds/alpine.jpg' };
const r1 = await cs.PW.depth.estimate(fakeImg);
const r2 = await cs.PW.depth.estimate(fakeImg);
assert.equal(runs, 1, '同一张图重复进入世界又跑了一次推理（runs=' + runs + '）');
assert.equal(r1.cached, false, '首次推理不应标记 cached');
assert.equal(r2.cached, true, '第二次应命中深度缓存');
assert.equal(r2.width, r1.width);
assert.equal(r2.height, r1.height);
assert.equal(cs.PW.depth.cacheStats().size, 1, '缓存里应只有一张深度图');
cs.PW.depth.clearCache();
assert.equal(cs.PW.depth.cacheStats().size, 0, 'clearCache 未清空');
await cs.PW.depth.estimate(fakeImg);
assert.equal(runs, 2, '清空缓存后应重新推理');
const other = { naturalWidth: 800, naturalHeight: 600, currentSrc: fakeImg.currentSrc };
assert.notEqual(cs.PW.depth.__internals.contentKey(fakeImg),
  cs.PW.depth.__internals.contentKey(other), '尺寸不同必须落在不同的缓存 key');
console.log('PASS  P3  深度图缓存：重复进入免推理（runs 1→2），清空后重算，尺寸进入 key');

console.log('\n全部回归通过');
