#!/usr/bin/env node
/**
 * 多视角质检结果的汇总器：按圈（A/B/C）拆指标，列出最差的图与最差视角。
 *
 *   node tools/mv_report.mjs reports/all32.json
 *   node tools/mv_report.mjs reports/all32.json --top 8 --ring C
 */
import fs from 'node:fs';

const argv = process.argv.slice(2);
const file = argv[0] || 'reports/all32.json';
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? argv[i + 1] : d; };
const TOP = parseInt(arg('top', '8'), 10);
const RING = arg('ring', '');

const d = JSON.parse(fs.readFileSync(file, 'utf8'));
const ok = (d.results || []).filter((r) => r.ok);
const mean = (a) => a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;

const keys = ['voidFrac', 'innerVoid', 'lowVoid', 'borderVoid', 'darkFrac', 'stripe', 'detail'];
const rings = [...new Set(ok.flatMap((r) => r.views.map((v) => v.ring || 'A')))].sort();

console.log(`样本 ${ok.length} 张 · 每张 ${ok[0] ? ok[0].views.length : 0} 视角 · 圈 ${rings.join('/')}`);
console.log('圈   ' + keys.map((k) => k.padEnd(11)).join(''));
for (const ring of rings) {
  const vs = ok.flatMap((r) => r.views.filter((v) => (v.ring || 'A') === ring));
  console.log(ring.padEnd(4) + keys.map((k) => mean(vs.map((v) => v[k])).toFixed(4).padEnd(11)).join(''));
}

/* 单视角最差：按指标排序 */
const flat = [];
for (const r of ok) {
  for (const v of r.views) {
    if (RING && (v.ring || 'A') !== RING) continue;
    flat.push({ image: r.image, ring: v.ring, a: v.a, ...v });
  }
}
const bad = (k) => flat.slice().sort((x, y) => y[k] - x[k]).slice(0, TOP);
for (const k of ['voidFrac', 'lowVoid', 'innerVoid', 'stripe', 'darkFrac']) {
  console.log(`\n最差 ${k}：`);
  for (const v of bad(k)) {
    console.log(`  ${v[k].toFixed(4)}  ${String(v.image).split('/').pop()}  ${v.ring}${v.a}°  ${v.file.split('/').pop()}`);
  }
}
const worstStripe = bad('stripe');
console.log('\n竖幕/拉丝最重的文件：');
for (const v of worstStripe.slice(0, 5)) console.log('  ' + v.file);
