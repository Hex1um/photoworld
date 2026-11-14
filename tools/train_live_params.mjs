#!/usr/bin/env node
/**
 * 图集训练：从语料学「生物群系原型」，产出 js/live_params.js
 *
 * 为什么需要：混乱度=1 时每个扇区的地貌必须与照片无关，但**不能是随机色相**
 * （第一版直接拿 sin(hue) 当通道增益，出来是一片荧光绿，观感像贴了滤色片）。
 * 正确做法是从真实语料里学几套「地面色 + 起伏强度 + 粗糙度 + 基线偏移」的原型，
 * 生成时按扇区/纪元哈希挑一套 —— 于是"无关"的新地貌仍然像地球上的某个地方。
 *
 * 方法（零依赖，k-means）：
 *   特征  [ground_r, ground_g, ground_b（线性空间）, 起伏强度 prox, 粗糙度 prox]
 *   聚类  k=6，按特征方差归一化后做球面 k-means，30 轮
 *   输出  每个原型的中心色、起伏倍率、粗糙倍率、基线偏移，以及训练/验证误差
 *
 *   node tools/train_live_params.mjs --corpus corpus/_norm --k 6 --out js/live_params.js
 */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import jpeg from 'jpeg-js';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? argv[i + 1] : d; };
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
const CORPUS = path.join(ROOT, arg('corpus', 'corpus/_norm'));
const K = parseInt(arg('k', '6'), 10);
const OUT = path.join(ROOT, arg('out', 'js/live_params.js'));
const VAL_RATIO = parseFloat(arg('val', '0.2'));

function loadPipeline() {
  const sandbox = { window: {}, console, Math, Date, Array, Object, Float32Array, URLSearchParams };
  sandbox.window.window = sandbox.window;
  vm.createContext(sandbox);
  for (const f of ['js/mat.js', 'js/reconstruct.js']) {
    vm.runInContext(fs.readFileSync(path.join(ROOT, f), 'utf8'), sandbox, { filename: f });
  }
  return sandbox.window.PW.reconstruct;
}

function decode(file, targetW) {
  const img = jpeg.decode(fs.readFileSync(file), { useTArray: true, maxMemoryUsageInMB: 1024 });
  const { width: sw, height: sh, data } = img;
  const W = Math.min(targetW, sw);
  const H = Math.max(120, Math.round(W * sh / sw));
  const out = new Uint8Array(W * H * 4);
  const fx = sw / W, fy = sh / H;
  for (let y = 0; y < H; y++) {
    const y0 = Math.floor(y * fy), y1 = Math.min(sh, Math.max(y0 + 1, Math.floor((y + 1) * fy)));
    for (let x = 0; x < W; x++) {
      const x0 = Math.floor(x * fx), x1 = Math.min(sw, Math.max(x0 + 1, Math.floor((x + 1) * fx)));
      let r = 0, g = 0, b = 0, n = 0;
      for (let sy = y0; sy < y1; sy++) {
        let idx = (sy * sw + x0) * 4;
        for (let sx = x0; sx < x1; sx++, idx += 4) { r += data[idx]; g += data[idx + 1]; b += data[idx + 2]; n++; }
      }
      const o = (y * W + x) * 4;
      out[o] = r / n; out[o + 1] = g / n; out[o + 2] = b / n; out[o + 3] = 255;
    }
  }
  return { data: out, W, H };
}

const lin = (v) => Math.pow(Math.max(0, Math.min(1, v)), 2.2);

function features(R, px, res) {
  const { data, W, H } = px;
  const a = res.analyzePixels(data, W, H, 'x', {});
  const g = a.ground;                        // sRGB 0..1
  // 起伏强度：用「前景细节方差 / 高度方差」这类分析量做代理（都和地貌起伏正相关）
  const amp = Math.max(0.15, Math.min(1.6, 0.45 + a.metrics.fgStd * 3.2 + a.metrics.hStd * 0.55));
  const rough = Math.max(0.55, Math.min(1.6, 0.7 + a.metrics.fgStd * 4.0));
  return {
    v: [lin(g[0]), lin(g[1]), lin(g[2]), amp, rough],
    meta: { lum: a.metrics.meanLum, hStd: a.metrics.hStd, fgStd: a.metrics.fgStd, green: a.isGreen ? 1 : 0, snow: a.isSnow ? 1 : 0 }
  };
}

/* ---------------- 读语料 ---------------- */
const manifest = JSON.parse(fs.readFileSync(path.join(CORPUS, 'index.json'), 'utf8'));
const R = loadPipeline();
const rows = [];
for (const item of manifest) {
  const file = path.join(CORPUS, item.file);
  if (!fs.existsSync(file)) continue;
  try {
    const px = decode(file, 420);
    rows.push(features(R, px, R));
  } catch (e) { /* 跳过坏图 */ }
}
if (rows.length < 50) { console.error('语料太少：' + rows.length); process.exit(1); }
console.log(`语料 ${rows.length} 张，特征维 ${rows[0].v.length}，k=${K}`);

/* ---------------- 划分 train / val ---------------- */
const idx = rows.map((_, i) => i);
let seed = 20261114;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
const nVal = Math.max(1, Math.round(rows.length * VAL_RATIO));
const valIdx = new Set(idx.slice(0, nVal));
const train = rows.filter((_, i) => !valIdx.has(i));
const val = rows.filter((_, i) => valIdx.has(i));
console.log(`训练 ${train.length} / 验证 ${val.length}`);

/* ---------------- 标准差归一化 + k-means ---------------- */
const D = rows[0].v.length;
const mean = new Array(D).fill(0);
for (const r of train) for (let d = 0; d < D; d++) mean[d] += r.v[d] / train.length;
const std = new Array(D).fill(0);
for (const r of train) for (let d = 0; d < D; d++) std[d] += (r.v[d] - mean[d]) ** 2 / train.length;
for (let d = 0; d < D; d++) std[d] = Math.max(1e-4, Math.sqrt(std[d]));
const norm = (v) => v.map((x, d) => (x - mean[d]) / std[d]);

// k-means++ 初始化
const centers = [];
centers.push(norm(train[Math.floor(rnd() * train.length)].v));
while (centers.length < K) {
  const dists = train.map((r) => {
    const nv = norm(r.v);
    let best = Infinity;
    for (const c of centers) {
      let s = 0; for (let d = 0; d < D; d++) s += (nv[d] - c[d]) ** 2;
      best = Math.min(best, s);
    }
    return best;
  });
  const total = dists.reduce((a, b) => a + b, 0);
  let t = rnd() * total, pick = 0;
  for (let i = 0; i < dists.length; i++) { t -= dists[i]; if (t <= 0) { pick = i; break; } }
  centers.push(norm(train[pick].v));
}

let assign = new Array(train.length).fill(0);
for (let iter = 0; iter < 30; iter++) {
  let moved = 0;
  train.forEach((r, i) => {
    const nv = norm(r.v);
    let best = 0, bestD = Infinity;
    centers.forEach((c, ci) => {
      let s = 0; for (let d = 0; d < D; d++) s += (nv[d] - c[d]) ** 2;
      if (s < bestD) { bestD = s; best = ci; }
    });
    if (assign[i] !== best) { assign[i] = best; moved++; }
  });
  const sums = centers.map(() => new Array(D).fill(0));
  const cnt = new Array(K).fill(0);
  train.forEach((r, i) => { const nv = norm(r.v); cnt[assign[i]]++; for (let d = 0; d < D; d++) sums[assign[i]][d] += nv[d]; });
  for (let ci = 0; ci < K; ci++) {
    if (!cnt[ci]) continue;
    centers[ci] = sums[ci].map((s) => s / cnt[ci]);
  }
  if (iter > 3 && moved === 0) { console.log(`k-means 收敛于第 ${iter} 轮`); break; }
}

/* ---------------- 原型还原到物理量 ---------------- */
const protos = [];
for (let ci = 0; ci < K; ci++) {
  const members = train.filter((_, i) => assign[i] === ci);
  if (!members.length) continue;
  const m = new Array(D).fill(0);
  for (const r of members) for (let d = 0; d < D; d++) m[d] += r.v[d] / members.length;
  protos.push({
    tint: [+m[0].toFixed(4), +m[1].toFixed(4), +m[2].toFixed(4)],
    amp: +Math.max(0.5, Math.min(2.0, m[3])).toFixed(3),
    rough: +Math.max(0.55, Math.min(1.6, m[4])).toFixed(3),
    weight: +(members.length / train.length).toFixed(4),
    n: members.length
  });
}

/* ---------------- 验证：把每张图归到最近原型，颜色误差 ---------------- */
function nearest(v) {
  let best = 0, bestD = Infinity;
  protos.forEach((p, i) => {
    const d = (lin2(v[0]) - p.tint[0]) ** 2 + (lin2(v[1]) - p.tint[1]) ** 2 + (lin2(v[2]) - p.tint[2]) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  });
  return { i: best, d: bestD };
}
const lin2 = (x) => x;   // 特征里已经是线性值

let sumD = 0, sumAbs = 0;
for (const r of val) {
  const nr = nearest(r.v);
  sumD += Math.sqrt(nr.d);
  sumAbs += (Math.abs(r.v[0] - protos[nr.i].tint[0]) + Math.abs(r.v[1] - protos[nr.i].tint[1])
    + Math.abs(r.v[2] - protos[nr.i].tint[2])) / 3;
}
const rmse = Math.sqrt(sumD / val.length);
const mae = sumAbs / val.length;

/* 相对误差：与「直接用全局均值当唯一原型」的基线比 */
const gm = [0, 1, 2].map((d) => val.reduce((s, r) => s + r.v[d], 0) / val.length);
let baseAbs = 0;
for (const r of val) baseAbs += (Math.abs(r.v[0] - gm[0]) + Math.abs(r.v[1] - gm[1]) + Math.abs(r.v[2] - gm[2])) / 3;
baseAbs /= val.length;

const report = {
  generatedAt: new Date().toISOString(),
  corpus: path.basename(CORPUS), samples: rows.length,
  train: train.length, val: val.length, k: protos.length,
  valColorRmse: +rmse.toFixed(4),
  valColorMae: +mae.toFixed(4),
  baselineMae: +baseAbs.toFixed(4),
  improvement: +(1 - mae / Math.max(1e-6, baseAbs)).toFixed(3)
};
console.log('训练报告：' + JSON.stringify(report));

/* ---------------- 落地成前端可直接用的模块 ---------------- */
const body = `/* 生物群系原型（由 tools/train_live_params.mjs 从图集训练，勿手改）
 *
 * 训练集：${report.corpus} · ${report.samples} 张（train ${report.train} / val ${report.val}）
 * 验证：颜色 MAE ${report.valColorMae}（无原型基线 ${report.baselineMae}，改善 ${(report.improvement * 100).toFixed(1)}%）
 *
 * 用途：实时变化模式里「混乱度」把当前扇区往某套原型推，
 *      于是新生成的地貌与照片无关，但仍然像真实世界的某个生物群系。
 * 字段：tint 线性 RGB 地面色 / amp 起伏倍率 / rough 频率倍率 / weight 语料占比
 */
(function (global) {
  'use strict';
  var DATA = ${JSON.stringify({
    version: '1.18.0',
    trainedAt: report.generatedAt,
    report,
    prototypes: protos
  }, null, 2).split('\n').join('\n  ')};
  global.PW = global.PW || {};
  global.PW.liveParams = DATA;
})(window);
`;
fs.writeFileSync(OUT, body);
fs.writeFileSync(path.join(ROOT, 'reports/live_params_report.json'), JSON.stringify({ report, prototypes: protos }, null, 1));
console.log('原型：');
for (const p of protos) console.log(`  tint=[${p.tint.join(', ')}] amp=${p.amp} rough=${p.rough} n=${p.n}`);
console.log('写入 ' + path.relative(ROOT, OUT));
