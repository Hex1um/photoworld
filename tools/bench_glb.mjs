#!/usr/bin/env node
/**
 * 真值基准：用 GLB 场景当"摄影棚"，拍一张照片 → 当输入喂进重建 → 与真值逐视角对比。
 *
 * 流程（每个 pose 组一次）：
 *   1. 打开 `?glb=<glb>&instant=1`，把相机放到 POSE[0] → 截图 = 照片（真值 A）
 *   2. 把照片存成 `corpus/bench/<glb>-a.jpg`（同源可访问）
 *   3. 打开 `?img=…/<glb>-a.jpg&instant=1` → 截图 = 重建在 A 视角的结果（自洽检查）
 *   4. 对每个 hold-out 视角 B：真值页面 __setCam(B) 截图；重建页面 __setCam(B) 截图
 *   5. 计算 PSNR / 结构相似 / 我们的 7 项指标，输出 JSON
 *
 * 这条链路的意义：重建的几何是"编"的，**只有和真 3D 比才知道编得准不准**。
 * 单图 PSNR 高只说明"从原视点看像"，novel-view PSNR 才是几何保真度。
 *
 *   node tools/bench_glb.mjs --glb assets/world-alpine.glb \
 *     --poses "0,0,0,-0.02|5,-8,0.5,-0.02|-6,-14,-0.6,0.05|0,-22,3.14,-0.02"
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import jpeg from 'jpeg-js';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? argv[i + 1] : d; };
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
const BASE = arg('base', 'http://127.0.0.1:8099');
const GLB = arg('glb', 'assets/world-alpine.glb');
const POSES = arg('poses', '0,0,0,-0.02|5,-8,0.5,-0.02|-6,-14,-0.6,0.05|0,-22,3.14,-0.02')
  .split('|').map((s) => s.split(',').map(Number));
const W = parseInt(arg('w', '640'), 10);
const H = parseInt(arg('h', '420'), 10);
const OUT = arg('out', 'reports/bench_glb.json');
const SHOTS = arg('shots', 'shots/bench');
const WAIT = parseInt(arg('wait', '13000'), 10);
const Q = (arg('q', '') || '').trim();   // 追加查询串，例如 --q "sem=0"

fs.mkdirSync(SHOTS, { recursive: true });
fs.mkdirSync(path.join(ROOT, 'corpus/bench'), { recursive: true });
const name = path.basename(GLB).replace(/\.[a-z0-9]+$/i, '');

/* ---------------- CDP ---------------- */
function connect(port) {
  return (async () => {
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    let ws;
    for (let i = 0; i < 120; i++) {
      try {
        const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
        const p = list.find((t) => t.type === 'page');
        if (p && p.webSocketDebuggerUrl) { ws = p.webSocketDebuggerUrl; break; }
      } catch (e) { /* 未就绪 */ }
      await sleep(250);
    }
    const sock = new WebSocket(ws);
    let id = 0;
    const pend = new Map();
    sock.addEventListener('message', (e) => {
      const m = JSON.parse(e.data);
      if (m.id && pend.has(m.id)) { pend.get(m.id)(m.result); pend.delete(m.id); }
    });
    await new Promise((r) => sock.addEventListener('open', r));
    const send = (method, params = {}) => {
      const mid = ++id;
      sock.send(JSON.stringify({ id: mid, method, params }));
      return new Promise((r) => pend.set(mid, r));
    };
    const ev = async (expr) => {
      const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
      return r && r.result ? r.result.value : undefined;
    };
    return { send, ev, close: () => sock.close(), sleep };
  })();
}

let PORT = 9500;
async function launch() {
  PORT += 7;
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'pwbench-'));
  const proc = spawn('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', [
    '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
    '--use-gl=angle', '--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist',
    '--no-first-run', `--window-size=${W},${H}`, 'about:blank'
  ], { stdio: 'ignore' });
  const c = await connect(PORT);
  await c.send('Page.enable');
  await c.send('Runtime.enable');
  await c.send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  c.kill = () => spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { stdio: 'ignore' });
  return c;
}

const ready = `(document.body.dataset||{}).ready === '1'`;
async function open(c, url, extraCheck) {
  await c.send('Page.navigate', { url });
  for (let i = 0; i < Math.ceil(WAIT / 250); i++) {
    await c.sleep(250);
    if (await c.ev(ready)) break;
  }
  if (extraCheck) await c.ev(extraCheck);
  await c.sleep(600);
}

async function shot(c, file) {
  const s = await c.send('Page.captureScreenshot', { format: 'jpeg', quality: 92 });
  const buf = Buffer.from(s.data, 'base64');
  fs.writeFileSync(file, buf);
  return jpeg.decode(buf, { useTArray: true, maxMemoryUsageInMB: 512 });
}

/* ---------------- 指标 ---------------- */
function psnr(a, b) {
  const n = Math.min(a.data.length, b.data.length);
  let se = 0;
  for (let i = 0; i < n; i += 4) {
    for (let k = 0; k < 3; k++) {
      const d = a.data[i + k] - b.data[i + k];
      se += d * d;
    }
  }
  const mse = se / (n / 4 * 3);
  return mse <= 1e-6 ? 99 : 10 * Math.log10((255 * 255) / mse);
}

function viewMetrics(img) {
  const { width: w, height: h, data } = img;
  let voidN = 0, innerN = 0, innerV = 0, lowN = 0, lowV = 0, dark = 0;
  const hist = new Uint32Array(16 * 16 * 16);
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    hist[(data[o] >> 4) * 256 + (data[o + 1] >> 4) * 16 + (data[o + 2] >> 4)]++;
  }
  let bq = 0, bn = -1;
  for (let q = 0; q < hist.length; q++) if (hist[q] > bn) { bn = hist[q]; bq = q; }
  const mr = ((bq >> 8) & 15) * 16 + 8, mg = ((bq >> 4) & 15) * 16 + 8, mb = (bq & 15) * 16 + 8;
  let vy = 0, vx = 0, det = 0, cells = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4;
      const r = data[o], g = data[o + 1], b = data[o + 2];
      const l = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
      if (l < 0.05) dark++;
      const isVoid = Math.abs(r - mr) <= 4 && Math.abs(g - mg) <= 4 && Math.abs(b - mb) <= 4;
      if (isVoid) voidN++;
      if (x >= w * 0.25 && x < w * 0.75 && y >= h * 0.25 && y < h * 0.75) { innerN++; if (isVoid) innerV++; }
      if (y >= h * 0.6) { lowN++; if (isVoid) lowV++; }
      if (y > 0 && x > 0 && ((x + y) & 1) === 0) {
        const oL = (y * w + x - 1) * 4, oU = ((y - 1) * w + x) * 4;
        const lL = (0.2126 * data[oL] + 0.7152 * data[oL + 1] + 0.0722 * data[oL + 2]) / 255;
        const lU = (0.2126 * data[oU] + 0.7152 * data[oU + 1] + 0.0722 * data[oU + 2]) / 255;
        vy += Math.abs(l - lU); vx += Math.abs(l - lL);
        det += Math.abs(l - lU) + Math.abs(l - lL);
        cells++;
      }
    }
  }
  return {
    voidFrac: +(voidN / (w * h)).toFixed(4),
    innerVoid: +(innerN ? innerV / innerN : 0).toFixed(4),
    lowVoid: +(lowN ? lowV / lowN : 0).toFixed(4),
    darkFrac: +(dark / (w * h)).toFixed(4),
    stripe: +(vy / Math.max(1e-6, vx)).toFixed(3),
    detail: +(det / Math.max(1, cells)).toFixed(4)
  };
}

/* ---------------- 主流程 ---------------- */
const camExpr = (p) => `window.__setCam ? window.__setCam(${p[0]}, ${p[1]}, ${p[2]}, ${p[3]}) : false`;

const truth = await launch();
const rec = await launch();
const rows = [];
try {
  // 1) 真值 A（作照片）
  const qs = Q ? '&' + Q : '';
  await open(truth, `${BASE}/?glb=${encodeURIComponent(GLB)}&instant=1${qs}`);
  const photoFile = path.join(ROOT, 'corpus/bench', name + '-a.jpg');
  const photoImg = await shot(truth, photoFile);
  console.log('照片（真值 A）已存：' + path.relative(ROOT, photoFile));

  // 2) 重建（用刚才那张照片）
  await open(rec, `${BASE}/?img=${encodeURIComponent('corpus/bench/' + name + '-a.jpg')}&instant=1${qs}`);
  rows.push({ pose: 'A (输入视角)', truthFile: path.relative(ROOT, photoFile), recon: null });
  const recA = await shot(rec, path.join(SHOTS, `${name}_A_rec.jpg`));
  const tA = await shot(truth, path.join(SHOTS, `${name}_A_truth.jpg`));
  rows[0].psnr = +psnr(recA, tA).toFixed(2);
  rows[0].metrics = viewMetrics(recA);
  console.log(`A 自洽：PSNR ${rows[0].psnr} dB`);

  // 3) hold-out 视角
  for (let i = 1; i < POSES.length; i++) {
    const p = POSES[i];
    await truth.ev(camExpr(p));
    await truth.sleep(700);
    const tImg = await shot(truth, path.join(SHOTS, `${name}_P${i}_truth.jpg`));
    await rec.ev(camExpr(p));
    await rec.sleep(900);
    const rImg = await shot(rec, path.join(SHOTS, `${name}_P${i}_rec.jpg`));
    const r = {
      pose: p.join(','), psnr: +psnr(rImg, tImg).toFixed(2),
      truth: viewMetrics(tImg), recon: viewMetrics(rImg),
      files: [`${name}_P${i}_truth.jpg`, `${name}_P${i}_rec.jpg`]
    };
    rows.push(r);
    console.log(`  P${i} [${p.join(',')}]  PSNR ${r.psnr} dB  `
      + `stripe 真值 ${r.truth.stripe} → 重建 ${r.recon.stripe}`);
  }
} finally {
  truth.kill(); rec.kill();
}

const novel = rows.filter((r) => r.psnr !== undefined && !String(r.pose).startsWith('A'));
const mean = novel.length ? novel.reduce((s, r) => s + r.psnr, 0) / novel.length : 0;
const report = {
  glb: GLB, size: [W, H], generatedAt: new Date().toISOString(),
  inputViewPsnr: rows[0] && rows[0].psnr,
  novelViewPsnr: +mean.toFixed(2),
  rows
};
fs.mkdirSync(path.dirname(path.join(ROOT, OUT)), { recursive: true });
fs.writeFileSync(path.join(ROOT, OUT), JSON.stringify(report, null, 1));
console.log(`\n输入视角 PSNR ${report.inputViewPsnr} dB · 新视角平均 PSNR ${report.novelViewPsnr} dB`);
console.log('明细 → ' + OUT);
process.exit(0);
