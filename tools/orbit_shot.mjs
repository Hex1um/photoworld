#!/usr/bin/env node
/**
 * 轨道 / 侧拍质检：绕世界中心拍 N 个角度 + 一张俯视侧拍，
 * 用来检查「单体补全是否拉伸、扇环延伸是否接缝、侧后方是否有洞」。
 *
 * 用法：
 *   node tools/orbit_shot.mjs --url "http://127.0.0.1:8099/?w=alpine&instant=1" \
 *     --out shots/alpine --angles 8 --radius 14 --center 0,-30 --pitch -0.18
 *
 * 依赖：本机 Edge（或 --edge 指定 chrome），页面需已实现 window.__setCam。
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
const URL_ = args.url || 'http://127.0.0.1:8099/?instant=1';
const OUTDIR = args.out || 'shots/orbit';
const ANGLES = parseInt(args.angles || '8', 10);
const RADIUS = parseFloat(args.radius || '14');
const CENTER = (args.center || '0,-30').split(',').map(Number);
const PITCH = parseFloat(args.pitch === undefined ? '-0.18' : args.pitch);
const WAIT = parseInt(args.wait || '120', 10);
const W = parseInt(args.w || '1280', 10);
const H = parseInt(args.h || '800', 10);
const PORT = 9500 + Math.floor(Math.random() * 300);

const EDGE = args.edge || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'pworbit-'));
fs.mkdirSync(OUTDIR, { recursive: true });

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
  for (let i = 0; i < 80; i++) {
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
    ws.addEventListener('error', () => rej(new Error('ws error')));
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

async function shot(send, file) {
  const r = await send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(file, Buffer.from(r.data, 'base64'));
}

try {
  const { ws, send } = await connect(await getWsUrl());
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url: URL_ });

  const t0 = Date.now();
  let ready = false, state = '{}';
  while (Date.now() - t0 < WAIT * 1000) {
    const r = await send('Runtime.evaluate', {
      expression: `JSON.stringify(Object.assign({}, document.body.dataset, { errs: (window.__errs || []).join(';') }))`,
      returnByValue: true
    });
    state = r.result.value || '{}';
    let st = {};
    try { st = JSON.parse(state); } catch (e) { st = {}; }
    if (st.ready === '1') { ready = true; break; }
    if (st.err) break;
    await sleep(500);
  }
  console.log('state=' + state);
  if (!ready) throw new Error('世界未就绪（见上方 state）');

  const cx = Number.isFinite(CENTER[0]) ? CENTER[0] : 0;
  const cz = Number.isFinite(CENTER[1]) ? CENTER[1] : -30;
  const prefix = args.prefix || 'orbit';

  for (let i = 0; i < ANGLES; i++) {
    const a = (i / ANGLES) * Math.PI * 2;
    const x = cx + Math.sin(a) * RADIUS;
    const z = cz + Math.cos(a) * RADIUS;
    const yaw = -a;                                  // 朝向中心
    const ok = await send('Runtime.evaluate', {
      expression: `window.__setCam ? window.__setCam(${x}, ${z}, ${yaw}, ${PITCH}) : false`,
      returnByValue: true
    });
    if (!ok.result.value) throw new Error('页面缺少 window.__setCam，请确认 app.js 已更新');
    await sleep(320);                                 // 等一帧 + 延伸 tile 追加
    const file = path.join(OUTDIR, `${prefix}_${String(i).padStart(2, '0')}.png`);
    await shot(send, file);
    console.log('  ' + file + `  (${(a * 180 / Math.PI).toFixed(0)}°)`);
  }

  // 额外一张俯视侧拍：检查地面/延伸区是否有洞、单体是否在地面上
  // 鸟瞰：把相机抬到 26m 再俯视（第 5 个参数是眼高覆盖），
  // 只看地面会漏掉「洞/悬浮单体」，从上方看最直观
  const TOP_H = parseFloat(args.topheight || '26');
  await send('Runtime.evaluate', {
    expression: `window.__setCam(${cx}, ${cz - RADIUS * 0.6}, 0, -1.0, ${TOP_H})`,
    returnByValue: true
  });
  await sleep(320);
  await shot(send, path.join(OUTDIR, `${prefix}_top.png`));
  console.log('  ' + path.join(OUTDIR, `${prefix}_top.png`) + '  (俯视)');

  ws.close();
} catch (e) {
  console.error('失败：' + e.message);
  process.exitCode = 1;
} finally {
  try { spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { stdio: 'ignore' }); }
  catch (e) { try { proc.kill(); } catch (e2) { /* ignore */ } }
}
