#!/usr/bin/env node
/**
 * 真实等待式截图：启动 Edge(headless) -> CDP 连接 -> 轮询页面就绪标记 -> 截图
 * 用法: node tools/shot.mjs --url "http://127.0.0.1:8099/?w=alpine" --out shot.png [--wait 90]
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, v, i, a) => {
    if (v.startsWith('--')) acc.push([v.slice(2), a[i + 1]]);
    return acc;
  }, [])
);
const URL_ = args.url || 'http://127.0.0.1:8099/';
const OUT = args.out || 'shot.png';
const WAIT = parseInt(args.wait || '90', 10);
const W = parseInt(args.w || '1440', 10);
const H = parseInt(args.h || '900', 10);
const PORT = 9333 + Math.floor(Math.random() * 200);

const EDGE = args.edge || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const profile = args.profile || fs.mkdtempSync(path.join(os.tmpdir(), 'pwshot-'));

const proc = spawn(EDGE, [
  '--headless=new',
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--hide-scrollbars', '--no-first-run', '--no-default-browser-check',
  `--window-size=${W},${H}`,
  'about:blank'
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getWsUrl() {
  for (let i = 0; i < 60; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/list`);
      const list = await r.json();
      const page = list.find((t) => t.type === 'page');
      if (page && page.webSocketDebuggerUrl) return page.webSocketDebuggerUrl;
    } catch (e) { /* not up yet */ }
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
    ws.addEventListener('error', (e) => rej(new Error('ws error')));
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && pend.has(msg.id)) {
        const { resolve, reject } = pend.get(msg.id);
        pend.delete(msg.id);
        if (msg.error) reject(new Error(JSON.stringify(msg.error)));
        else resolve(msg.result);
      }
    });
    function send(method, params) {
      const mid = ++id;
      ws.send(JSON.stringify({ id: mid, method, params: params || {} }));
      return new Promise((resolve, reject) => pend.set(mid, { resolve, reject }));
    }
  });
}

try {
  const wsUrl = await getWsUrl();
  const { ws, send } = await connect(wsUrl);
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url: URL_ });

  const t0 = Date.now();
  let state = '';
  while (Date.now() - t0 < WAIT * 1000) {
    const r = await send('Runtime.evaluate', {
      expression: `JSON.stringify(Object.assign({}, document.body.dataset, { errs: (window.__errs || []).join(';') }))`,
      returnByValue: true
    });
    state = r.result.value || '{}';
    let st = {};
    try { st = JSON.parse(state); } catch (e) { st = {}; }
    if (st.ready === '1' || st.ready === '0') break;
    await sleep(500);
  }
  console.log('state=' + state);

  await sleep(600);
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(OUT, Buffer.from(shot.data, 'base64'));
  console.log('截图保存 ' + OUT + '  ' + ((Date.now() - t0) / 1000).toFixed(1) + 's');
  ws.close();
} catch (e) {
  console.error('失败：' + e.message);
  process.exitCode = 1;
} finally {
  try {
    spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { stdio: 'ignore' });
  } catch (e) {
    try { proc.kill(); } catch (e2) {}
  }
}
