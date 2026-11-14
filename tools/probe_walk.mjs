#!/usr/bin/env node
/**
 * 移动边界体检：把玩家推到远处，检查是否有「空气墙」（位置被夹住 / 远处没有地形）。
 *
 *   node tools/probe_walk.mjs --url "http://127.0.0.1:8099/?w=desert&instant=1" \
 *     --to -200 --out shots/wall.png
 *
 * 输出：mode / walkRadius / 目标点的 heightAt / tile 数量 / 实际坐标（判断有没有被夹）
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? argv[i + 1] : d; };
const URL_ = arg('url', 'http://127.0.0.1:8099/?w=desert&instant=1');
const TO = parseFloat(arg('to', '-200'));
const OUT = arg('out', '');
const W = parseInt(arg('w', '480'), 10);
const H = parseInt(arg('h', '1040'), 10);
const PORT = 9600 + Math.floor(Math.random() * 200);
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'pwwalk-'));

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
  if (!r) return 'NO RESULT';
  if (r.exceptionDetails) return 'EXC ' + JSON.stringify(r.exceptionDetails.exception && r.exceptionDetails.exception.description || r.exceptionDetails.text).slice(0, 300);
  return r.result ? r.result.value : 'NO VALUE';
};

await send('Page.enable');
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: URL_ });
await sleep(14000);

console.log('before :', await ev(`(function(){try{
  var w=window.__pw&&window.__pw.world, p=window.__pw&&window.__pw.player;
  return JSON.stringify({hasPw:!!window.__pw, mode:w&&w.mode, walkRadius:w&&w.walkRadius,
    bbox:w&&w.bbox, y:p&&+p.state.y.toFixed(2),
    err:(document.body.dataset||{}).err||'', errs:(window.__errs||[]).slice(0,3)});
}catch(e){return 'ERR '+e.message}})()`));

await ev(`(function(){try{
  var p=window.__pw.player, w=window.__pw.world;
  p.state.z=${TO}; p.state.y=w.heightAt(0,${TO})+1.7; return 1;
}catch(e){return 0}})()`);
await sleep(3000);

console.log('after  :', await ev(`(function(){try{
  var p=window.__pw.player, w=window.__pw.world, ext=window.__pw.extender;
  return JSON.stringify({x:+p.state.x.toFixed(1), z:+p.state.z.toFixed(1), y:+p.state.y.toFixed(2),
    h:+w.heightAt(0,${TO}).toFixed(2), tiles:(w.extraMeshes||[]).length, ext: ext&&ext.stats?ext.stats():null});
}catch(e){return 'ERR '+e.message}})()`));

if (OUT) {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(OUT, Buffer.from(shot.data, 'base64'));
  console.log('截图：' + OUT);
}
sock.close();
spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { stdio: 'ignore' });
process.exit(0);
