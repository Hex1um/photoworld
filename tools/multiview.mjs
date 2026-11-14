#!/usr/bin/env node
/**
 * 多视角批量质检（语料 → 16 方位 + 俯视）
 *
 * 把语料图一张张喂进**在线重建链路**（?img= 同源直读），绕世界中心取 16 个方位
 * 各拍一张，再补一张俯视侧拍，同时对截图算四项客观指标：
 *   voidFrac   与画面主色（clear color）几乎一致的像素占比 —— 洞 / 露底 / 天穹灰边
 *   innerVoid  中央 50% ROI 的空底色占比 —— 画面中部的贯穿灰带
 *   lowVoid    下 40%（地面）的空底色占比 —— 脚下的空洞 / 没铺到的地面
 *   borderVoid 外圈 6% 的空底色占比 —— 照片没铺满视口
 *   darkFrac   luma < 0.05 的像素占比 —— 塌黑
 *   stripe     列方向竖直能量 / 行方向水平能量 —— 竖幕、拉花、冰柱
 *   detail     平均梯度 —— 贴图是否被放大糊掉（越低越糊）
 *
 * PNG 用 Node 内置 zlib 自解（CDP 截图是 8bit RGBA、无隔行），不引第三方依赖。
 *
 *   node tools/multiview.mjs --base http://127.0.0.1:8099 \
 *     --list corpus/sample.txt --views 16 --out shots/mv --json reports/mv.json
 *
 * 列表文件每行一个相对 base 的图片路径（如 corpus/_norm/aa.jpg），也支持 # 注释。
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import zlib from 'node:zlib';

/* ---------------- 参数 ---------------- */
const argv = process.argv.slice(2);
const arg = (name, def) => {
  const i = argv.indexOf('--' + name);
  return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1] : def;
};
const BASE = arg('base', 'http://127.0.0.1:8099').replace(/\/$/, '');
const LIST = arg('list', '');
const CORPUS = arg('corpus', '');       // 直接给目录时取目录下所有 jpg（与 --list 二选一）
const VIEWS = parseInt(arg('views', '16'), 10);
const OUTDIR = arg('out', 'shots/mv');
const JSONOUT = arg('json', 'reports/multiview.json');
const RADIUS = parseFloat(arg('radius', '11'));
const CENTER = (arg('center', '0,-26')).split(',').map(Number);
const PITCH = parseFloat(arg('pitch', '-0.10'));
const WAIT = parseInt(arg('wait', '100'), 10);       // 单图最长等待（秒）
const SETTLE = parseInt(arg('settle', '260'), 10);   // 每视角稳定毫秒
const W = parseInt(arg('w', '960'), 10);
const H = parseInt(arg('h', '600'), 10);
const LIMIT = parseInt(arg('limit', '0'), 10);
const EXTRA = arg('query', '');                      // 追加到 URL 的查询串（A/B 用）
const EDGE = arg('edge', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe');
/* 光栅后端：hw = D3D11 真 GPU（默认）；sw = SwiftShader 软件光栅。
   两者观感差别很大（软件光栅在掠射角会拉出放射状条纹），
   指标只能在同一后端之间比较，所以默认走 hw。 */
const GLMODE = arg('gl', 'hw');
const PORT = 9600 + Math.floor(Math.random() * 300);

/* ---------------- 名单 ---------------- */
let images = [];
if (LIST) {
  images = fs.readFileSync(LIST, 'utf8').split(/\r?\n/)
    .map((s) => s.trim()).filter((s) => s && !s.startsWith('#'));
} else if (CORPUS) {
  images = fs.readdirSync(CORPUS).filter((f) => /\.jpe?g$/i.test(f))
    .sort().map((f) => CORPUS.replace(/\\/g, '/').replace(/\/$/, '') + '/' + f);
}
if (!images.length) {
  console.error('没有输入图片：用 --list 或 --corpus 指定');
  process.exit(2);
}
if (LIMIT > 0) images = images.slice(0, LIMIT);
fs.mkdirSync(OUTDIR, { recursive: true });
fs.mkdirSync(path.dirname(JSONOUT), { recursive: true });

/* ---------------- PNG 解码（CDP 截图：8bit RGBA，无隔行） ---------------- */
function decodePNG(buf) {
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error('不是 PNG');
  let off = 8, w = 0, h = 0, bitDepth = 8, colorType = 6, interlace = 0;
  const idat = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off);
    const type = buf.toString('ascii', off + 4, off + 8);
    const data = buf.subarray(off + 8, off + 8 + len);
    if (type === 'IHDR') {
      w = data.readUInt32BE(0); h = data.readUInt32BE(4);
      bitDepth = data[8]; colorType = data[9]; interlace = data[12];
    } else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    off += 12 + len;
  }
  if (bitDepth !== 8 || interlace !== 0) throw new Error('只支持 8bit 无隔行 PNG');
  const bpp = colorType === 6 ? 4 : colorType === 2 ? 3 : colorType === 0 ? 1 : 0;
  if (!bpp) throw new Error('不支持的 colorType ' + colorType);
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const stride = w * bpp;
  const out = Buffer.alloc(h * stride);
  let prev = Buffer.alloc(stride);
  for (let y = 0; y < h; y++) {
    const ft = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const cur = out.subarray(y * stride, (y + 1) * stride);
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? cur[x - bpp] : 0;
      const b = prev[x];
      const c = x >= bpp ? prev[x - bpp] : 0;
      let v = line[x];
      if (ft === 1) v += a;
      else if (ft === 2) v += b;
      else if (ft === 3) v += (a + b) >> 1;
      else if (ft === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
      }
      cur[x] = v & 0xff;
    }
    prev = cur;
  }
  return { w, h, bpp, data: out };
}

/* ---------------- 指标 ---------------- */
const luma = (r, g, b) => (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;

function metrics(png) {
  const { w, h, bpp, data } = png;
  const L = new Float32Array(w * h);
  const total = w * h;

  /* 空底色检测不依赖「雾色有没有传进来」：把画面量化成 16 级立方体直方图，
     取最大桶的中心当作 clear color。天穹/空洞是均匀底色，纹理不是。
     旧版拿 body[data-fog] 比色，夜景（雾色接近黑）会把整屏都判成雾。 */
  const hist = new Uint32Array(16 * 16 * 16);
  for (let i = 0; i < total; i++) {
    const o = i * bpp;
    const q = (data[o] >> 4) * 256 + (data[o + 1] >> 4) * 16 + (data[o + 2] >> 4);
    hist[q]++;
  }
  let bestQ = 0, bestN = -1;
  for (let q = 0; q < hist.length; q++) if (hist[q] > bestN) { bestN = hist[q]; bestQ = q; }
  const mr = ((bestQ >> 8) & 15) * 16 + 8, mg = ((bestQ >> 4) & 15) * 16 + 8, mb = (bestQ & 15) * 16 + 8;

  let voidN = 0, darkN = 0, innerN = 0, innerVoid = 0, lowN = 0, lowVoid = 0, borderN = 0, borderVoid = 0;
  const x0 = Math.floor(w * 0.25), x1 = Math.floor(w * 0.75);
  const y0 = Math.floor(h * 0.25), y1 = Math.floor(h * 0.75);
  const bx = Math.floor(w * 0.06), by = Math.floor(h * 0.06);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * bpp;
      const r = data[o], g = data[o + 1], b = data[o + 2];
      const l = luma(r, g, b);
      L[y * w + x] = l;
      if (l < 0.05) darkN++;
      const isVoid = Math.abs(r - mr) <= 4 && Math.abs(g - mg) <= 4 && Math.abs(b - mb) <= 4;
      if (isVoid) voidN++;
      const inMid = x >= x0 && x < x1 && y >= y0 && y < y1;
      if (inMid) { innerN++; if (isVoid) innerVoid++; }
      if (y >= h * 0.6) { lowN++; if (isVoid) lowVoid++; }
      if (x < bx || x >= w - bx || y < by || y >= h - by) { borderN++; if (isVoid) borderVoid++; }
    }
  }
  // 竖直能量（沿 y 差分）/ 水平能量（沿 x 差分）：竖幕 / 拉花的直接指纹
  let vy = 0, vx = 0, detail = 0, cells = 0;
  for (let y = 1; y < h; y += 2) {
    for (let x = 1; x < w; x += 2) {
      const k = y * w + x;
      vy += Math.abs(L[k] - L[k - w]);
      vx += Math.abs(L[k] - L[k - 1]);
      detail += Math.abs(L[k] - L[k - 1]) + Math.abs(L[k] - L[k - w]);
      cells++;
    }
  }
  return {
    voidFrac: +(voidN / total).toFixed(4),
    innerVoid: +(innerN ? innerVoid / innerN : 0).toFixed(4),
    lowVoid: +(lowN ? lowVoid / lowN : 0).toFixed(4),
    borderVoid: +(borderN ? borderVoid / borderN : 0).toFixed(4),
    darkFrac: +(darkN / total).toFixed(4),
    stripe: +(vy / Math.max(1e-6, vx)).toFixed(3),
    detail: +(detail / Math.max(1, cells)).toFixed(4)
  };
}

/* ---------------- CDP ---------------- */
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'pwmv-'));
const GLFLAGS = GLMODE === 'sw'
  ? ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']
  : ['--use-gl=angle', '--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist'];
const proc = spawn(EDGE, [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  ...GLFLAGS,
  '--hide-scrollbars', '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
  `--window-size=${W},${H}`, 'about:blank'
], { stdio: 'ignore' });

async function wsUrl() {
  for (let i = 0; i < 120; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/list`);
      const list = await r.json();
      const page = list.find((t) => t.type === 'page');
      if (page && page.webSocketDebuggerUrl) return page.webSocketDebuggerUrl;
    } catch (e) { /* 还没起来 */ }
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

const evalIn = async (send, expression) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true });
  return r.result ? r.result.value : undefined;
};

/* ---------------- 主流程 ---------------- */
const results = [];
let exitCode = 0;
try {
  const { ws, send } = await connect(await wsUrl());
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride',
    { width: W, height: H, deviceScaleFactor: 1, mobile: false });

  const cx = Number.isFinite(CENTER[0]) ? CENTER[0] : 0;
  const cz = Number.isFinite(CENTER[1]) ? CENTER[1] : -26;
  const t0 = Date.now();

  for (let n = 0; n < images.length; n++) {
    const rel = images[n];
    const url = `${BASE}/?img=${encodeURIComponent(rel)}&instant=1&name=`
      + encodeURIComponent(path.basename(rel).replace(/\.[a-z0-9]+$/i, ''))
      + (EXTRA ? '&' + EXTRA : '');
    const rec = { image: rel, views: [], ok: false };
    await send('Page.navigate', { url });

    const ts = Date.now();
    let state = {};
    while (Date.now() - ts < WAIT * 1000) {
      const s = await evalIn(send, `JSON.stringify(Object.assign({}, document.body.dataset, { errs: (window.__errs || []).join(';') }))`);
      try { state = JSON.parse(s || '{}'); } catch (e) { state = {}; }
      if (state.ready === '1') break;
      if (state.err) break;
      await sleep(400);
    }
    if (state.ready !== '1') {
      rec.error = state.err || 'ready 超时';
      rec.errs = state.errs || '';
      results.push(rec);
      console.log(`✗ ${rel}  ${rec.error}`);
      exitCode = 1;
      continue;
    }
    rec.meta = {
      mode: state.mode, horizon: state.horizon, expo: state.expo,
      sem: state.sem, objects: state.objects || '', version: state.version || ''
    };
    const dbg = state.dbg ? JSON.parse(state.dbg) : null;
    if (dbg) rec.relief = dbg;

    for (let i = 0; i < VIEWS; i++) {
      const a = (i / VIEWS) * Math.PI * 2;
      const x = cx + Math.sin(a) * RADIUS;
      const z = cz + Math.cos(a) * RADIUS;
      const ok = await evalIn(send, `window.__setCam ? window.__setCam(${x.toFixed(2)}, ${z.toFixed(2)}, ${(-a).toFixed(4)}, ${PITCH}) : false`);
      if (!ok) throw new Error('页面缺少 window.__setCam');
      await sleep(SETTLE);
      const shot = await send('Page.captureScreenshot', { format: 'png' });
      const buf = Buffer.from(shot.data, 'base64');
      const file = path.join(OUTDIR, `${path.basename(rel).replace(/\.[a-z0-9]+$/i, '')}_${String(i).padStart(2, '0')}.png`);
      fs.writeFileSync(file, buf);
      const m = metrics(decodePNG(buf));
      rec.views.push({ i, a: +(a * 180 / Math.PI).toFixed(0), ...m, file: file.replace(/\\/g, '/') });
    }
    // 俯视：检查地面有没有洞、单体是否浮空
    await evalIn(send, `window.__setCam(${cx}, ${(cz - RADIUS * 0.55).toFixed(2)}, 0, -1.0, 24)`);
    await sleep(SETTLE);
    const top = Buffer.from((await send('Page.captureScreenshot', { format: 'png' })).data, 'base64');
    const topFile = path.join(OUTDIR, `${path.basename(rel).replace(/\.[a-z0-9]+$/i, '')}_top.png`);
    fs.writeFileSync(topFile, top);
    rec.top = { ...metrics(decodePNG(top)), file: topFile.replace(/\\/g, '/') };

    rec.ok = true;
    // 聚合：均值 + 最差视角（改进前/后对比看这两个数）
    const mean = (k) => +(rec.views.reduce((s, v) => s + v[k], 0) / rec.views.length).toFixed(4);
    rec.agg = {
      voidFrac: mean('voidFrac'), darkFrac: mean('darkFrac'),
      innerVoid: mean('innerVoid'), lowVoid: mean('lowVoid'),
      borderVoid: mean('borderVoid'), stripe: mean('stripe'), detail: mean('detail'),
      worstInner: +Math.max(...rec.views.map((v) => v.innerVoid)).toFixed(4),
      worstLow: +Math.max(...rec.views.map((v) => v.lowVoid)).toFixed(4),
      worstStripe: +Math.max(...rec.views.map((v) => v.stripe)).toFixed(3)
    };
    results.push(rec);
    const el = ((Date.now() - t0) / 1000).toFixed(0);
    console.log(`✓ ${path.basename(rel)}  void ${rec.agg.voidFrac} inner ${rec.agg.innerVoid} `
      + `low ${rec.agg.lowVoid} stripe ${rec.agg.stripe} detail ${rec.agg.detail}  `
      + `[${n + 1}/${images.length}, ${el}s]`);
    fs.writeFileSync(JSONOUT, JSON.stringify({ base: BASE, views: VIEWS, results }, null, 1));
  }
  ws.close();
} catch (e) {
  console.error('失败：' + e.message);
  exitCode = 1;
} finally {
  try { spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { stdio: 'ignore' }); }
  catch (e) { try { proc.kill(); } catch (e2) { /* ignore */ } }
}

/* ---------------- 汇总 ---------------- */
const ok = results.filter((r) => r.ok);
if (ok.length) {
  const avg = (k) => +(ok.reduce((s, r) => s + r.agg[k], 0) / ok.length).toFixed(4);
  const worst = ok.map((r) => ({ image: r.image, ...r.agg }))
    .sort((a, b) => b.worstLow + b.worstInner - (a.worstLow + a.worstInner)).slice(0, 10);
  const summary = {
    images: results.length, ok: ok.length, failed: results.length - ok.length,
    views: VIEWS,
    avg: {
      voidFrac: avg('voidFrac'), innerVoid: avg('innerVoid'), lowVoid: avg('lowVoid'),
      borderVoid: avg('borderVoid'), stripe: avg('stripe'),
      darkFrac: avg('darkFrac'), detail: avg('detail')
    },
    worstInner: worst
  };
  fs.writeFileSync(JSONOUT, JSON.stringify({ base: BASE, views: VIEWS, summary, results }, null, 1));
  console.log('汇总 ' + JSON.stringify(summary.avg) + '  失败 ' + summary.failed);
  console.log('最差 inner/low void：' + worst.slice(0, 5).map((w) => path.basename(w.image) + ' ' + w.worstInner + '/' + w.worstLow).join(' | '));
  console.log('明细 → ' + JSONOUT + '，截图 → ' + OUTDIR);
}
process.exitCode = exitCode;
