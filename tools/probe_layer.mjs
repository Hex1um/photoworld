#!/usr/bin/env node
/**
 * 逐层定位：把 world.meshes 里的某一层置空后截图，看画面里哪一层在冒碎片。
 *
 *   node tools/probe_layer.mjs --url "http://127.0.0.1:8099/?img=...&instant=1&x=..&z=..&yaw=.." \
 *     --outdir shots/layer
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? argv[i + 1] : d; };
const URL_ = arg('url', 'http://127.0.0.1:8099/?w=alpine&instant=1');
const OUTDIR = arg('outdir', 'shots/layer');
const W = parseInt(arg('w', '480'), 10);
const H = parseInt(arg('h', '1040'), 10);
const LAYERS = (arg('layers', 'reliefBackdrop,reliefBg,skinSkirt,relief,worldBackdrop') || '').split(',');
const PORT = 9700 + Math.floor(Math.random() * 150);
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'pwl-'));

fs.mkdirSync(OUTDIR, { recursive: true });
const proc = spawn('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  '--use-gl=angle', '--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist',
  '--no-first-run', `--window-size=${W},${H}`, 'about:blank'
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ws;
for (let i = 0; i < 120; i++) {
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    const p = list.find((t) => t.type === 'page');
    if (p && p.webSocketDebuggerUrl) { ws = p.webSocketDebuggerUrl; break; }
  } catch (e) { /* */ }
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
const send = (m, p = {}) => {
  const mid = ++id;
  sock.send(JSON.stringify({ id: mid, method: m, params: p }));
  return new Promise((r) => pend.set(mid, r));
};
const ev = async (x) => {
  const r = await send('Runtime.evaluate', { expression: x, returnByValue: true, awaitPromise: true });
  return r && r.result ? r.result.value : undefined;
};

await send('Page.enable');
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: URL_ });
await sleep(14000);

const names = await ev(`JSON.stringify(Object.keys((window.__pw&&window.__pw.world&&window.__pw.world.meshes)||{}))`);
console.log('可用层：' + names);

for (const layer of LAYERS) {
  await ev(`(function(){var m=window.__pw.world.meshes; if(m) window.__pw.saved_${layer}=m.${layer}||null, m.${layer}=null; return 1})()`);
  await sleep(700);
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  const f = path.join(OUTDIR, `hide_${layer}.png`);
  fs.writeFileSync(f, Buffer.from(shot.data, 'base64'));
  console.log('  ' + f);
  await ev(`(function(){var m=window.__pw.world.meshes; if(m) m.${layer}=window.__pw.saved_${layer}||null; return 1})()`);
  await sleep(300);
}
const shot = await send('Page.captureScreenshot', { format: 'png' });
fs.writeFileSync(path.join(OUTDIR, 'all_on.png'), Buffer.from(shot.data, 'base64'));
sock.close();
spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { stdio: 'ignore' });
process.exit(0);
