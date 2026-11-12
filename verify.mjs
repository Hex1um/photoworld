import fs from 'fs';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const INFER = require('./infer.js');

const raw = JSON.parse(fs.readFileSync(new URL('./train/weights.json', import.meta.url), 'utf8'));
const W = {};
for (const k in raw.w) {
  const buf = Buffer.from(raw.w[k], 'base64');
  W[k] = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}
const rd = (p, n) => {
  const b = fs.readFileSync(new URL(p, import.meta.url));
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};
const inp = rd('./train/in.f32');
const eh = rd('./train/out_h.f32');
const ec = rd('./train/out_c.f32');

const t0 = Date.now();
const r = INFER.infer(inp, W);
const ms = Date.now() - t0;

let dh = 0, dc = 0, mh = 0, mc = 0;
for (let i = 0; i < eh.length; i++) {
  const d = Math.abs(r.h.d[i] - eh[i]);
  dh += d; if (d > mh) mh = d;
}
for (let i = 0; i < ec.length; i++) {
  const d = Math.abs(r.c.d[i] - ec[i]);
  dc += d; if (d > mc) mc = d;
}
console.log('js infer ' + ms + 'ms   h: mean ' + (dh / eh.length).toFixed(6) + ' max ' + mh.toFixed(6) +
  '   c: mean ' + (dc / ec.length).toFixed(6) + ' max ' + mc.toFixed(6));
console.log('js h std', Math.sqrt(r.h.d.reduce((s, v, _, a) => s + (v - a.reduce((x, y) => x + y, 0) / a.length) ** 2, 0) / r.h.d.length).toFixed(4));
