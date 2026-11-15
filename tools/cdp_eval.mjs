#!/usr/bin/env node
/**
 * 通用 CDP 探针：启动真实 GPU 加速的 Edge(headless) 打开一个页面，执行任意表达式并打印 JSON。
 *
 *   node tools/cdp_eval.mjs --url "http://127.0.0.1:8199/probe.html" --expr "JSON.stringify(navigator.gpu?1:0)"
 *   node tools/cdp_eval.mjs --url "..." --exprfile tools/gpu_probe.js [--wait 60] [--hw]
 *
 * 与 tools/shot.mjs 的区别：默认**不开** swiftshader（--hw=1 用硬件 D3D12 + 打开 unsafe-webgpu），
 * 用来验证 WebGPU / 硬件光栅路径；截图仍走 shot.mjs。
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
const URL_ = args.url || 'about:blank';
const WAIT = parseInt(args.wait || '30', 10);
const PORT = 9700 + Math.floor(Math.random() * 200);
const EDGE = args.edge || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'pwcdp-'));
const expr = args.exprfile ? fs.readFileSync(args.exprfile, 'utf8') : (args.expr || 'document.title');

const flags = [
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  '--disk-cache-dir=/dev/null', '--disk-cache-size=1', '--disable-application-cache',
  '--hide-scrollbars', '--no-first-run', '--no-default-browser-check',
  '--window-size=1280,800'
];
if (!args.headful) flags.push('--headless=new');
if (args.hw) {
  // 硬件路径：不要 swiftshader，打开 WebGPU（headless 下新版 Edge 需要显式旗标）。
  // force_high_performance_gpu 必须加：笔记本上 headless 默认抓 Intel 核显，
  // 不加强制就测不出独显的真实性能（实测 adapter vendor 会变成 intel gen-12lp）。
  flags.push('--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU',
    '--force_high_performance_gpu', '--enable-gpu-rasterization');
} else {
  flags.push('--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader');
}
flags.push(URL_);
if (args.flags) flags.splice(flags.length - 1, 0, ...String(args.flags).split('|').filter(Boolean));

const proc = spawn(EDGE, flags, { stdio: 'ignore' });
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

try {
  const { ws, send } = await connect(await getWsUrl());
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Page.navigate', { url: URL_ });
  await sleep(1200);

  const r = await send('Runtime.evaluate', {
    expression: `(async () => { try { return JSON.stringify(await (async () => { return ${expr}; })()); } catch (e) { return JSON.stringify({ error: String(e && e.stack || e) }); } })()`,
    awaitPromise: true,
    returnByValue: true,
    timeout: WAIT * 1000
  });
  const out = (r.result && r.result.value) || '(空)';
  console.log(out);
  // --shot <path>：表达式执行完之后存一张截图（成套的"设置状态→看一眼"很有用）
  if (args.shot) {
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(args.shot, Buffer.from(shot.data, 'base64'));
    console.log('截图 → ' + args.shot);
  }
  ws.close();
} catch (e) {
  console.error('失败：' + e.message);
  process.exitCode = 1;
} finally {
  try { spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { stdio: 'ignore' }); } catch (e) { try { proc.kill(); } catch (e2) {} }
}
