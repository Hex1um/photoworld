#!/usr/bin/env node
/**
 * A/B 多机位截图台：一次启动浏览器，在同一页面里连续 enterWorld 多张图，每图拍多个机位。
 *
 *   node tools/abshots.mjs --base http://127.0.0.1:8199 --list testlist.txt --out C:/tmp/ab/before --hw
 *   node tools/abshots.mjs --base ... --list ... --out ... --only 00_grass
 *
 * 为什么不用 tools/shot.mjs 循环：每次 shot 都要新 profile → 重新下载 45MB 模型 + WASM 初始化，
 * 20s/张全花在冷启动上；这里复用同一次会话，第二张起只需推理时间。
 *
 * 机位默认三个，回答三个不同的问题：
 *   front  正视：和在画廊里看到的一致，看构图/天空/曝光
 *   side   右移 8m 不转视角：暴露遮挡区（LDI 该填补的地方）——"幕布/拉丝"在这里最明显
 *   back   后退 3m 转身 180°：看延伸/回望有没有空洞
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
const BASE = (args.base || 'http://127.0.0.1:8199').replace(/\/$/, '');
const OUT = args.out || 'C:/tmp/ab/out';
const LIST = args.list;
const ONLY = args.only || '';
const PER_IMAGE_TIMEOUT = parseInt(args.wait || '120', 10) * 1000;
const W = parseInt(args.w || '1280', 10);
const H = parseInt(args.h || '800', 10);
const VIEWS = (args.views || 'front,side,back').split(',');
// 追加到每个 URL 后面的查询串（例如 ext=1）；注意别直接塞进 --base
const APPEND = args.append ? ('&' + String(args.append).replace(/^&/, '')) : '';
const PORT = 9500 + Math.floor(Math.random() * 300);
const EDGE = args.edge || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';

const POSE = {
  front: { x: 0, z: 0, yaw: 0, pitch: -0.02 },
  side: { x: 8, z: 0, yaw: 0, pitch: -0.02 },
  side16: { x: 16, z: 0, yaw: 0, pitch: -0.02 },
  back: { x: 0, z: -3, yaw: Math.PI, pitch: -0.02 },
  up: { x: 0, z: 0, yaw: 0, pitch: 0.28 }
};

if (!LIST) { console.error('需要 --list'); process.exit(2); }
const images = fs.readFileSync(LIST, 'utf8').split('\n').map((s) => s.trim()).filter(Boolean)
  .map((line) => {
    const [src, name] = line.split('|');
    return { src, name: name || path.basename(src).replace(/\.[a-z]+$/i, '') };
  })
  .filter((it) => !ONLY || it.name.includes(ONLY));

fs.mkdirSync(OUT, { recursive: true });

const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'pwsess-'));
const flags = [
  '--headless=new',
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  '--disk-cache-dir=/dev/null', '--disk-cache-size=1', '--disable-application-cache',
  '--hide-scrollbars', '--no-first-run', '--no-default-browser-check',
  `--window-size=${W},${H}`
];
if (args.hw) {
  flags.push('--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--force_high_performance_gpu');
} else {
  flags.push('--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader');
}
flags.push('about:blank');

const proc = spawn(EDGE, flags, { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getWsUrl() {
  for (let i = 0; i < 100; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = list.find((t) => t.type === 'page');
      if (page && page.webSocketDebuggerUrl) return page.webSocketDebuggerUrl;
    } catch (e) { /* not up */ }
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

async function evaluate(send, expression, awaitPromise = true) {
  const r = await send('Runtime.evaluate', { expression, awaitPromise, returnByValue: true, timeout: PER_IMAGE_TIMEOUT });
  if (r.exceptionDetails) throw new Error('JS 异常: ' + JSON.stringify(r.exceptionDetails.text || r.exceptionDetails));
  return r.result ? r.result.value : undefined;
}

const rows = [];
try {
  const { ws, send } = await connect(await getWsUrl());
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url: `${BASE}/?instant=1${APPEND}&cb=${Date.now()}` });

  // 等 __pw 就绪
  for (let i = 0; i < 120; i++) {
    const ok = await evaluate(send, '!!(window.__pw && window.__pw.enterWorld)', false);
    if (ok) break;
    if (i === 119) throw new Error('__pw 未就绪');
    await sleep(250);
  }
  // 清掉落地页动画层，避免它盖住画面
  await evaluate(send, `(() => { const l=document.getElementById('landing'); if(l){l.style.display='none';} return 1; })()`, false);

  const GLB_MODE = args.mode === 'glb';
  for (const it of images) {
    const t0 = Date.now();
    if (GLB_MODE) {
      // 资产模式：每次重新导航（GLB 加载走 URL 参数，没有页内 API）
      await send('Page.navigate', { url: `${BASE}/?glb=${encodeURIComponent(it.src)}&instant=1${APPEND}&cb=${Date.now()}` });
      await sleep(300);
      for (let i = 0; i < 200; i++) {
        const ok = await evaluate(send, '!!(window.__pw && window.__pw.world)', false);
        if (ok) break;
        await sleep(250);
      }
    } else {
      await evaluate(send, `(() => {
        document.body.dataset.ready = '0';
        document.body.dataset.mode = '';
        window.__errs = [];
        window.__pw.enterWorld({ name: ${JSON.stringify(it.name)}, sub: '', src: ${JSON.stringify(it.src)}, thumb: ${JSON.stringify(it.src)} });
        return 1;
      })()`, false);
    }

    let st = {};
    const deadline = Date.now() + PER_IMAGE_TIMEOUT;
    while (Date.now() < deadline) {
      const raw = await evaluate(send, `JSON.stringify(Object.assign({}, document.body.dataset, { errs: (window.__errs||[]).join(';') }))`, false);
      try { st = JSON.parse(raw || '{}'); } catch (e) { st = {}; }
      if (st.ready === '1') break;
      if (st.errs) break;
      await sleep(400);
    }
    const loadMs = Date.now() - t0;

    for (const v of VIEWS) {
      const p = POSE[v] || POSE.front;
      await evaluate(send, `(() => {
        const pl = window.__pw.player;
        if (!pl) return 0;
        pl.state.x = ${p.x}; pl.state.z = ${p.z};
        pl.state.yaw = ${p.yaw}; pl.state.pitch = ${p.pitch};
        pl.state.vx = 0; pl.state.vz = 0; pl.state.bob = 0;
        return 1;
      })()`, false);
      await sleep(500);
      const shot = await send('Page.captureScreenshot', { format: 'png' });
      fs.writeFileSync(path.join(OUT, `${it.name}__${v}.png`), Buffer.from(shot.data, 'base64'));
    }
    rows.push({ name: it.name, mode: st.mode || '-', loadS: (loadMs / 1000).toFixed(1), errs: st.errs || '', tex: st.texW ? `${st.texW}x${st.texH}` : '-' });
    console.log(`${it.name.padEnd(22)} mode=${(st.mode || '-').padEnd(7)} ${(loadMs / 1000).toFixed(1)}s ${st.errs ? 'ERR ' + st.errs : ''}`);
  }
  ws.close();
} catch (e) {
  console.error('失败：' + e.message);
  process.exitCode = 1;
} finally {
  try { spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { stdio: 'ignore' }); } catch (e) { try { proc.kill(); } catch (e2) {} }
  if (rows.length) {
    fs.writeFileSync(path.join(OUT, '_summary.json'), JSON.stringify(rows, null, 1));
    console.log('\n共 ' + rows.length + ' 张，截图 ' + rows.length * VIEWS.length + ' 张 → ' + OUT);
  }
}
