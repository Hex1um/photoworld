#!/usr/bin/env node
/**
 * PHOTOWORLD 批量评测 / 数据驱动调参
 *
 * 用 Node 解码语料 -> 调与浏览器完全相同的 analyzePixels -> 统计质量指标。
 * 通过 --sets 传入多组参数覆盖，横向对比哪组参数在 500+ 张网图上更稳。
 *
 *   node tools/eval.mjs --corpus corpus --out reports/eval.json
 *   node tools/eval.mjs --sets baseline,strict,loose
 */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import jpeg from 'jpeg-js';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');

/* ---------- 载入浏览器端代码（同一份实现） ---------- */
function loadPipeline() {
  const sandbox = { window: {}, console, Math, Date, Array, Object, Float32Array, URLSearchParams };
  sandbox.window.window = sandbox.window;
  vm.createContext(sandbox);
  for (const f of ['js/mat.js', 'js/reconstruct.js']) {
    vm.runInContext(fs.readFileSync(path.join(ROOT, f), 'utf8'), sandbox, { filename: f });
  }
  return sandbox.window.PW.reconstruct;
}

/* ---------- 解码 + 盒式缩放 ---------- */
function decodeToRGBA(file, targetW) {
  const raw = fs.readFileSync(file);
  const img = jpeg.decode(raw, { useTArray: true, maxMemoryUsageInMB: 1024 });
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
        for (let sx = x0; sx < x1; sx++, idx += 4) {
          r += data[idx]; g += data[idx + 1]; b += data[idx + 2]; n++;
        }
      }
      const o = (y * W + x) * 4;
      out[o] = r / n; out[o + 1] = g / n; out[o + 2] = b / n; out[o + 3] = 255;
    }
  }
  return { data: out, W, H };
}

/* ---------- 参数集 ---------- */
const SETS = {
  baseline: {},
  strict: { colorThresh: 0.050, strongThresh: 0.130, gradThresh: 0.020, medianGuard: 0.08 },
  loose: { colorThresh: 0.022, strongThresh: 0.070, gradThresh: 0.009, medianGuard: 0.12 },
  smooth: { smoothRounds: 4, medianGuard: 0.07, anchorCompress: 0.16 },
  softanchor: { anchorPct: 0.66, anchorWindow: 0.17, anchorCompress: 0.35 },
  tall: { heightDiv: 0.15, heightMax: 1.20 },
  short: { heightDiv: 0.22, heightMax: 1.0 }
};

function arg(name, def) {
  const i = process.argv.indexOf('--' + name);
  return i > 0 ? process.argv[i + 1] : def;
}

const corpusDir = path.join(ROOT, arg('corpus', 'corpus'));
const outPath = path.join(ROOT, arg('out', 'reports/eval.json'));
const setNameList = arg('sets', 'baseline').split(',');
const limit = parseInt(arg('limit', '0'), 10);
const targets = setNameList.map((n) => ({ name: n, opt: SETS[n] || {} }));

// 原始语料用 manifest.json（harvest 产物）；归一化目录只有 index.json（normalize 产物）。
// 两者字段都含 file/cat，直接按目录挑一个，省掉「再去复制一份清单」的麻烦。
const manifestPath = ['manifest.json', 'index.json']
  .map((f) => path.join(corpusDir, f))
  .find((p) => fs.existsSync(p));
if (!manifestPath) {
  console.error('缺少 manifest.json / index.json：先跑 tools/harvest.py 或 tools/normalize.py');
  process.exit(1);
}
let manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
if (limit > 0) manifest = manifest.slice(0, limit);

const R = loadPipeline();
const rows = [];
let n = 0;

for (const item of manifest) {
  const file = path.join(corpusDir, item.file);
  if (!fs.existsSync(file)) continue;
  let px;
  try {
    px = decodeToRGBA(file, 420);
  } catch (e) {
    continue;
  }
  const rec = { file: item.file, cat: item.cat || 'unknown', w: px.W, h: px.H, m: {} };
  for (const t of targets) {
    try {
      const a = R.analyzePixels(px.data, px.W, px.H, item.file, t.opt);
      rec.m[t.name] = a.metrics;
    } catch (e) {
      rec.m[t.name] = null;
    }
  }
  rows.push(rec);
  n++;
  if (n % 50 === 0) process.stdout.write(`  已分析 ${n}\r`);
}
console.log(`\n分析完成：${rows.length} 张`);

/* ---------- 聚合 ---------- */
function agg(list, setName) {
  const ms = list.map((r) => r.m[setName]).filter(Boolean);
  if (!ms.length) return null;
  const mean = (k) => ms.reduce((s, m) => s + m[k], 0) / ms.length;
  const pct = (k, pred) => ms.filter((m) => pred(m[k])).length / ms.length;
  return {
    n: ms.length,
    score: +mean('score').toFixed(4),
    horizonClipped: +pct('horizonClipped', (v) => v === 1).toFixed(4),
    skyMissHigh: +pct('skyMissRate', (v) => v > 0.15).toFixed(4),
    satHigh: +pct('satRate', (v) => v > 0.25).toFixed(4),
    flat: +pct('hStd', (v) => v < 0.05).toFixed(4),
    hStd: +mean('hStd').toFixed(4),
    hMean: +mean('hMean').toFixed(4),
    fgStd: +mean('fgStd').toFixed(4)
  };
}

const report = { generatedAt: new Date().toISOString(), corpusSize: rows.length, sets: {}, byCat: {}, rows };
for (const t of targets) {
  report.sets[t.name] = { opt: t.opt, all: agg(rows, t.name) };
  const cats = [...new Set(rows.map((r) => r.cat))];
  const per = {};
  for (const c of cats) {
    const sub = rows.filter((r) => r.cat === c);
    per[c] = agg(sub, t.name);
  }
  report.byCat[t.name] = per;
}

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(report, null, 1), 'utf8');

/* ---------- 打印 ---------- */
const pad = (s, w) => String(s).padEnd(w);
console.log('\n参数集对比（% 越低越好，score 越高越好）');
console.log(pad('set', 12) + pad('score', 8) + pad('地平线截断', 12) + pad('天空失效', 10) + pad('高度饱和', 10) + pad('过平', 8) + pad('hStd', 8));
for (const t of targets) {
  const a = report.sets[t.name].all;
  if (!a) continue;
  console.log(pad(t.name, 12) + pad(a.score, 8) + pad((a.horizonClipped * 100).toFixed(1), 12) +
    pad((a.skyMissHigh * 100).toFixed(1), 10) + pad((a.satHigh * 100).toFixed(1), 10) +
    pad((a.flat * 100).toFixed(1), 8) + pad(a.hStd, 8));
}

const best = targets.map((t) => ({ t: t.name, s: report.sets[t.name].all.score }))
  .sort((a, b) => b.s - a.s)[0];
console.log('\n最佳参数集：' + best.t + '  score=' + best.s);
console.log('明细写入 ' + path.relative(ROOT, outPath));
