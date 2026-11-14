#!/usr/bin/env node
/**
 * PHOTOWORLD 冒烟测试：把「页面能不能进世界」变成一条命令。
 *
 *   node tools/smoke.mjs --base http://127.0.0.1:8137
 *   node tools/smoke.mjs --shots /tmp/smoke          # 顺便每例存一张截图
 *   node tools/smoke.mjs --only alpine               # 只跑名字里含 alpine 的
 *
 * 判据（硬）：body[data-ready] === '1' 且 window.__errs 为空、无 data-err。
 * 任何一例失败 → 退出码 1，可直接挂到提交前钩子或 CI。
 *
 * 之所以要有它：前端一行改名（opt→opts）就能让主路径 100% 进不去世界，
 * 而这种问题在无头浏览器里 20 秒就能抓到。
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, v, i, a) => {
  if (v.startsWith('--')) acc.push([v.slice(2), a[i + 1]]);
  return acc;
}, []));

const BASE = (args.base || 'http://127.0.0.1:8099').replace(/\/$/, '');
const WAIT = parseInt(args.wait || '90', 10);
const SHOTS = args.shots || '';
const ONLY = args.only || '';
const EDGE = args.edge || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const PORT = 9500 + Math.floor(Math.random() * 300);
const W = 1280, H = 800;

const CASES = [
  { name: 'alpine', url: '/?w=alpine&instant=1' },
  { name: 'desert', url: '/?w=desert&instant=1' },
  { name: 'asset-glb', url: '/?glb=assets/world-alpine.glb&instant=1' },
  { name: 'ring-on', url: '/?w=alpine&ring=1&instant=1' },
  { name: 'skip-sky', url: '/?w=alpine&skip=sky&instant=1' }
].filter((c) => !ONLY || c.name.includes(ONLY));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'pwsmoke-'));
const proc = spawn(EDGE, ['--headless=new', `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`, '--use-gl=angle', '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader', '--no-proxy-server', '--hide-scrollbars', '--no-first-run',
  `--window-size=${W},${H}`, 'about:blank'], { stdio: 'ignore' });

async function wsUrl() {
  for (let i = 0; i < 80; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/list`);
      const t = (await r.json()).find((x) => x.type === 'page');
      if (t) return t.webSocketDebuggerUrl;
    } catch (e) { /* not up */ }
    await sleep(250);
  }
  throw new Error('CDP 未就绪');
}

function connect(url) {
  return new Promise((res, rej) => {
    const sock = new WebSocket(url);
    let id = 0; const pend = new Map();
    sock.addEventListener('open', () => res({ sock, send }));
    sock.addEventListener('error', () => rej(new Error('ws error')));
    sock.addEventListener('message', (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && pend.has(m.id)) {
        const { resolve, reject } = pend.get(m.id); pend.delete(m.id);
        if (m.error) reject(new Error(JSON.stringify(m.error))); else resolve(m.result);
      }
    });
    function send(method, params) {
      const mid = ++id;
      sock.send(JSON.stringify({ id: mid, method, params: params || {} }));
      return new Promise((resolve, reject) => pend.set(mid, { resolve, reject }));
    }
  });
}

const rows = [];

try {
  if (SHOTS) fs.mkdirSync(SHOTS, { recursive: true });
  const { sock, send } = await connect(await wsUrl());
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });

  for (const c of CASES) {
    const t0 = Date.now();
    await send('Page.navigate', { url: BASE + c.url });
    let st = {};
    while (Date.now() - t0 < WAIT * 1000) {
      const r = await send('Runtime.evaluate', {
        expression: `JSON.stringify(Object.assign({}, document.body && document.body.dataset, {errs: (window.__errs||[]).join(';')}))`,
        returnByValue: true
      });
      try { st = JSON.parse(r.result.value || '{}'); } catch (e) { st = {}; }
      if (st.ready === '1' || st.err) break;
      await sleep(300);
    }
    const ms = Date.now() - t0;
    if (SHOTS) {
      try {
        const shot = await send('Page.captureScreenshot', { format: 'png' });
        fs.writeFileSync(path.join(SHOTS, c.name + '.png'), Buffer.from(shot.data, 'base64'));
      } catch (e) { /* 截图失败不影响判据 */ }
    }
    const ok = st.ready === '1' && !st.err && !(st.errs || '').trim();
    rows.push({ name: c.name, ok, ms, mode: st.mode || '-', err: st.err || st.errs || '' });
  }
  sock.close();
} catch (e) {
  console.error('冒烟失败：' + e.message);
  rows.push({ name: '(harness)', ok: false, ms: 0, mode: '-', err: e.message });
} finally {
  try { spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { stdio: 'ignore' }); } catch (e) { /* ignore */ }
}

const pad = (s, n) => String(s).padEnd(n);
console.log('\n' + pad('用例', 14) + pad('结果', 6) + pad('耗时', 9) + pad('mode', 10) + '错误');
for (const r of rows) {
  console.log(pad(r.name, 14) + pad(r.ok ? 'PASS' : 'FAIL', 6) + pad((r.ms / 1000).toFixed(1) + 's', 9) +
    pad(r.mode, 10) + r.err);
}
const bad = rows.filter((r) => !r.ok).length;
console.log(`\n${rows.length - bad}/${rows.length} 通过` + (bad ? `，${bad} 例失败` : ''));
process.exitCode = bad ? 1 : 0;
