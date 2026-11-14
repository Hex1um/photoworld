#!/usr/bin/env node
/**
 * 深度模型加载体检：限速 + 采样载入文案，回答「界面会不会卡在某个百分比」。
 *
 *   node tools/probe_depth.mjs --url "http://127.0.0.1:8099/?w=alpine&instant=1" \
 *     --mbps 1.6 --latency 150 --out reports/depth_probe.json
 *
 * 判定：
 *   samples   载入文案出现过多少个不同取值（旧实现在 27MB 下载期间只有 2 个）
 *   maxStall  同一文案的最长停留时间（旧实现 = 整段下载耗时）
 *   ready     是否最终进入世界
 *
 * 第二次跑同一 profile 会命中 Cache Storage，用来验证「缓存后秒开」。
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? argv[i + 1] : d; };

const URL_ = arg('url', 'http://127.0.0.1:8099/?w=alpine&instant=1');
const MBPS = parseFloat(arg('mbps', '1.6'));
const LAT = parseInt(arg('latency', '150'), 10);
const WAIT = parseInt(arg('wait', '180'), 10);
const OUT = arg('out', 'reports/depth_probe.json');
const PROFILE = arg('profile', '');            // 复用一个 profile 可测缓存命中
const PORT = 9350 + Math.floor(Math.random() * 120);
const EDGE = arg('edge', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe');

const profile = PROFILE || fs.mkdtempSync(path.join(os.tmpdir(), 'pwdepth-'));
fs.mkdirSync(path.dirname(OUT), { recursive: true });

const proc = spawn(EDGE, [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  '--use-gl=angle', '--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist',
  '--no-first-run', '--no-default-browser-check', '--window-size=900,600', 'about:blank'
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function wsUrl() {
  for (let i = 0; i < 120; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const p = list.find((t) => t.type === 'page');
      if (p && p.webSocketDebuggerUrl) return p.webSocketDebuggerUrl;
    } catch (e) { /* 未就绪 */ }
    await sleep(250);
  }
  throw new Error('CDP 未就绪');
}

function connect(url) {
  return new Promise((res, rej) => {
    const ws = new WebSocket(url);
    let id = 0;
    const pend = new Map();
    ws.addEventListener('open', () => res({ ws, send }));
    ws.addEventListener('error', () => rej(new Error('ws error')));
    ws.addEventListener('message', (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && pend.has(m.id)) {
        const { resolve, reject } = pend.get(m.id);
        pend.delete(m.id);
        if (m.error) reject(new Error(JSON.stringify(m.error)));
        else resolve(m.result);
      }
    });
    function send(method, params) {
      const mid = ++id;
      ws.send(JSON.stringify({ id: mid, method, params: params || {} }));
      return new Promise((resolve, reject) => pend.set(mid, { resolve, reject }));
    }
  });
}

const evaluate = async (send, expr, awaitPromise = false) => {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise });
  return r.result ? r.result.value : undefined;
};

const timeline = [];
let ready = false;
let err = '';
try {
  const { ws, send } = await connect(await wsUrl());
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Network.enable');
  if (MBPS > 0) {
    await send('Network.emulateNetworkConditions', {
      offline: false,
      latency: LAT,
      downloadThroughput: MBPS * 1024 * 1024 / 8,
      uploadThroughput: MBPS * 1024 * 1024 / 8 * 0.4,
      connectionType: 'cellular3g'
    });
  }
  await send('Page.navigate', { url: URL_ });
  const t0 = Date.now();
  let last = null;
  while (Date.now() - t0 < WAIT * 1000) {
    const st = await evaluate(send, `JSON.stringify({
      step: (document.getElementById('loaderSteps')||{}).textContent || '',
      bar: (document.getElementById('loaderBar')||{}).style ? (document.getElementById('loaderBar')||{}).style.width : '',
      ready: (document.body.dataset||{}).ready || '',
      err: (document.body.dataset||{}).err || ''
    })`);
    let s = {};
    try { s = JSON.parse(st || '{}'); } catch (e) { s = {}; }
    const label = (s.step || '').trim();
    const t = Date.now() - t0;
    if (label && (!last || last.label !== label)) {
      timeline.push({ t, label, bar: s.bar });
      last = { label, t };
    }
    if (s.ready === '1') { ready = true; timeline.push({ t, label: '[ready]', bar: '' }); break; }
    if (s.err) { err = s.err; break; }
    await sleep(250);
  }
  ws.close();
} catch (e) {
  err = e.message;
} finally {
  try { spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { stdio: 'ignore' }); }
  catch (e) { try { proc.kill(); } catch (e2) { /* ignore */ } }
}

/* 同一文案的最长停留 */
let maxStall = 0, stallLabel = '';
for (let i = 0; i < timeline.length; i++) {
  const dt = (i + 1 < timeline.length ? timeline[i + 1].t : timeline[timeline.length - 1].t) - timeline[i].t;
  if (dt > maxStall) { maxStall = dt; stallLabel = timeline[i].label; }
}
const distinct = new Set(timeline.map((x) => x.label)).size;
const result = {
  url: URL_, mbps: MBPS, latency: LAT, profile,
  ready, err, samples: distinct, maxStallMs: maxStall, stallLabel,
  totalMs: timeline.length ? timeline[timeline.length - 1].t : 0,
  timeline
};
fs.writeFileSync(OUT, JSON.stringify(result, null, 1));

console.log(`ready=${ready}  文案取样=${distinct} 个  总耗时=${(result.totalMs / 1000).toFixed(1)}s`);
console.log(`最长停留 ${(maxStall / 1000).toFixed(1)}s → 「${stallLabel}」`);
for (const x of timeline) console.log(`  ${String((x.t / 1000).toFixed(1)).padStart(6)}s  ${x.label}`);
if (err) { console.error('错误：' + err); process.exitCode = 1; }
process.exitCode = ready ? 0 : 1;
