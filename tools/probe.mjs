#!/usr/bin/env node
/**
 * 运行时探针：加载页面 -> 等就绪 -> 执行任意表达式 -> 打印 JSON。
 * 用于排查「延伸 tile 是否生成」「语义分组统计」「相机位置」等运行时状态。
 *
 *   node tools/probe.mjs --url "http://127.0.0.1:8099/?w=winter&instant=1" \
 *        --expr "JSON.stringify(window.__pw.extender && window.__pw.extender.stats())"
 *
 * 额外参数：
 *   --wait  秒（默认 90，等 ready 标记）
 *   --settle 毫秒（ready 之后再等多久才求值，默认 3000，给异步 tile 生成留时间）
 *   --w/--h 视口尺寸
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
// --exprfile 用于放不下的多行诊断脚本（例如 tools/diag_jumps.js）
const EXPR = args.exprfile
  ? fs.readFileSync(args.exprfile, 'utf8')
  : (args.expr || 'JSON.stringify({ ready: document.body.dataset.ready })');
const WAIT = parseInt(args.wait || '90', 10);
const SETTLE = parseInt(args.settle || '3000', 10);
const W = parseInt(args.w || '1440', 10);
const H = parseInt(args.h || '900', 10);
const PORT = 9600 + Math.floor(Math.random() * 200);
const EDGE = args.edge || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const profile = args.profile || fs.mkdtempSync(path.join(os.tmpdir(), 'pwprobe-'));

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
  const wsUrl = await getWsUrl();
  const { ws, send } = await connect(wsUrl);
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });

  // 收集页面里的错误，诊断时最有用
  const errs = [];
  const origHandler = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.method === 'Runtime.exceptionThrown') {
      const d = msg.params.exceptionDetails;
      errs.push((d.exception && d.exception.description) || d.text || 'unknown');
    }
    if (msg.method === 'Runtime.consoleAPICalled' && msg.params.type === 'error') {
      errs.push(msg.params.args.map((a) => a.value || a.description || '').join(' '));
    }
  };
  ws.addEventListener('message', origHandler);

  await send('Page.navigate', { url: URL_ });

  const t0 = Date.now();
  let st = {};
  while (Date.now() - t0 < WAIT * 1000) {
    const r = await send('Runtime.evaluate', {
      expression: 'JSON.stringify(document.body.dataset)',
      returnByValue: true
    });
    try { st = JSON.parse(r.result.value || '{}'); } catch (e) { st = {}; }
    if (st.ready === '1' || st.ready === '0') break;
    await sleep(500);
  }
  console.log('dataset=' + JSON.stringify(st));
  await sleep(SETTLE);

  const r2 = await send('Runtime.evaluate', {
    expression: EXPR, returnByValue: true, awaitPromise: true
  });
  if (r2.exceptionDetails) {
    console.error('表达式抛错：' + (r2.exceptionDetails.exception && r2.exceptionDetails.exception.description));
    process.exitCode = 1;
  } else {
    console.log('result=' + (typeof r2.result.value === 'string' ? r2.result.value : JSON.stringify(r2.result.value)));
  }
  // 可选截图：表达式执行完（比如摆好相机）后拍一张，用来把「运行时状态」和「画面」对上
  if (args.out) {
    await sleep(parseInt(args.shotdelay || '900', 10));
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    fs.mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
    fs.writeFileSync(args.out, Buffer.from(shot.data, 'base64'));
    console.log('截图 ' + args.out);
  }
  if (errs.length) console.error('page-errors:\n  ' + errs.slice(0, 12).join('\n  '));
  console.log('elapsed=' + ((Date.now() - t0) / 1000).toFixed(1) + 's');
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
