#!/usr/bin/env node
/**
 * 无浏览器自测：证明新增清晰度指标确实能分辨「糊片」和「雾/渐变」。
 *   node tools/test_clarity.mjs
 */
import assert from 'node:assert/strict';
import { clarityMetrics, captureDiff } from './clarity.mjs';

const w = 128, h = 96;
const sharp = new Float32Array(w * h);
let seed = 12345;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
for (let y = 0; y < h; y++) {
  for (let x = 0; x < w; x++) {
    const check = ((x >> 2) + (y >> 2)) % 2;
    sharp[y * w + x] = Math.min(1, Math.max(0, 0.25 + check * 0.5 + (rnd() - 0.5) * 0.18));
  }
}
function blur3(src) {
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0, n = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy >= 0 && yy < h && xx >= 0 && xx < w) { s += src[yy * w + xx]; n++; }
    }
    out[y * w + x] = s / n;
  }
  return out;
}
let soft = sharp;
for (let i = 0; i < 4; i++) soft = blur3(soft);
// 纯线性渐变：模拟「大片雾 / 均匀底色」，梯度不小但没有任何高频
const fog = new Float32Array(w * h);
for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) fog[y * w + x] = 0.2 + 0.5 * (x / w);

const A = clarityMetrics(sharp, w, h);
const B = clarityMetrics(soft, w, h);
const C = clarityMetrics(fog, w, h);

console.log('清晰图  ', JSON.stringify(A));
console.log('模糊图  ', JSON.stringify(B));
console.log('渐变/雾 ', JSON.stringify(C));

for (const k of ['hfEnergy', 'centerHF', 'lapVar', 'centerLapVar', 'localContrast']) {
  assert(Number.isFinite(A[k]) && A[k] >= 0, k + ' 应为非负有限数');
}
assert(A.hfEnergy > B.hfEnergy * 2, `hfEnergy 未拉开差距：${A.hfEnergy} vs ${B.hfEnergy}`);
assert(A.centerLapVar > B.centerLapVar * 2, `centerLapVar 未拉开差距：${A.centerLapVar} vs ${B.centerLapVar}`);
assert(C.lapVar < 0.05, `纯渐变不该有拉普拉斯方差：${C.lapVar}`);
assert(C.centerLapVar < 0.05, `纯渐变不该有中央拉普拉斯方差：${C.centerLapVar}`);
assert(A.localContrast > C.localContrast, '清晰图局部对比度应高于纯渐变');

const cap = { w: 960, h: 600, dpr: 1, gl: 'hw', views: 32, radius: 11, center: [0, -26], pitch: -0.1, settle: 260, clarity: 1 };
assert.deepEqual(captureDiff(cap, { ...cap }), [], '相同采集条件应判为一致');
assert.deepEqual(captureDiff(cap, { ...cap, w: 1440 }), ['w'], '分辨率不同必须拒绝对比');
assert.deepEqual(captureDiff(cap, { ...cap, gl: 'sw' }), ['gl'], '光栅后端不同必须拒绝对比');

console.log('✓ 6/6 组断言通过：清晰度指标可分辨糊片与雾，A/B 采集条件校验生效');
