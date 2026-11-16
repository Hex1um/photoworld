#!/usr/bin/env node
/**
 * 多视角批量质检（语料 → 16 方位 + 俯视）
 *
 * 把语料图一张张喂进**在线重建链路**（?img= 同源直读），绕世界中心取 16 个方位
 * 各拍一张，再补一张俯视侧拍，同时对截图算两组客观指标：
 *   voidFrac   与画面主色（clear color）几乎一致的像素占比 —— 洞 / 露底 / 天穹灰边
 *   innerVoid  中央 50% ROI 的空底色占比 —— 画面中部的贯穿灰带
 *   lowVoid    下 40%（地面）的空底色占比 —— 脚下的空洞 / 没铺到的地面
 *   borderVoid 外圈 6% 的空底色占比 —— 照片没铺满视口
 *   darkFrac   luma < 0.05 的像素占比 —— 塌黑
 *   stripe     列方向竖直能量 / 行方向水平能量 —— 竖幕、拉花、冰柱
 *   detail     平均梯度 —— 保留兼容，但它分不出「糊片」和「雾」，A/B 别只看它
 *   hfEnergy / farHF / centerHF / lapVar / centerLapVar / localContrast
 *              清晰度六项，见 tools/clarity.mjs（越大越清晰；优先看 centerHF、
 *              centerLapVar，全图指标会被天空与程序化地面稀释）
 *
 * PNG 用 Node 内置 zlib 自解（CDP 截图是 8bit RGBA、无隔行），不引第三方依赖。
 *
 *   node tools/multiview.mjs --base http://127.0.0.1:8099 \
 *     --list corpus/sample.txt --views 16 --out shots/mv --json reports/mv.json \
 *     --baseline reports/clarity_baseline.json
 *
 * --baseline 第一次跑写入基线，之后同参数再跑即输出逐指标 A/B 表；
 * 采集条件（分辨率 / dpr / 光栅后端 / 视角数 / 稳定时长）不一致会拒绝对比。
 * 列表文件每行一个相对 base 的图片路径（如 corpus/_norm/aa.jpg），也支持 # 注释。
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import zlib from 'node:zlib';
import { clarityMetrics, CLARITY_KEYS, HIGHER_IS_BETTER, captureDiff } from './clarity.mjs';

/* ---------------- 参数 ---------------- */
const argv = process.argv.slice(2);
const arg = (name, def) => {
  const i = argv.indexOf('--' + name);
  return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1] : def;
};
const BASE = arg('base', 'http://127.0.0.1:8099').replace(/\/$/, '');
const LIST = arg('list', '');
const CORPUS = arg('corpus', '');       // 直接给目录时取目录下所有 jpg（与 --list 二选一）
/* GLB 模式（v2 资产）：--glb-list 每行一个相对 base 的 world.glb 路径。
   复用同一套 32 视角套餐与指标 —— 这样"在线链路 vs v2 网格 vs v2 点云"三种表示
   可以在**完全相同的机位与指标口径**下比较。 */
const GLBLIST = arg('glb-list', '');
const VIEWS = parseInt(arg('views', '16'), 10);
const OUTDIR = arg('out', 'shots/mv');
const JSONOUT = arg('json', 'reports/multiview.json');
const RADIUS = parseFloat(arg('radius', '11'));
/* 32 视角套餐：三圈不同「视角 + 坐标」
   A 圈 12 个方位（r=10，平视）      ：照片区域内的日常观看视角
   B 圈 12 个方位（r=26，微仰）      ：照片区域内的远景 / 地平线衔接
   C 圈 8 个方位（r=75）             ：**区域外**（|x|>55 或 z>0 或 z<-100），
                                      检查视锥实时生成的地貌；一半平视、一半 14m 俯视。
   之前的 C 圈放在 r=18（仍在照片矩形内），32 张全落在照片几何上，
   等于没测到生成区 —— 细看指标才发现。 */
/* v2 专用 32 机位：**拍摄点附近、朝场景看**。
   为什么不能沿用 A/B/C 三圈：那套是给在线链路的"无限地貌"设计的，r=10~75 绕世界
   中心转 —— 对 v2 的单视锥重建（内容都在原点前方 2~70m）来说，相机直接落进了
   几何内部，看到的是薄壳内壁，测出来的指标没有意义（实测两边都是满屏拉丝）。

   这套的分布：8 个水平偏移 × 4 个纵深，全部朝前（-Z），带 ±60° 的偏航与少量俯仰。
   它回答的是真实使用中的问题：站在重建区域里左右移动、转身看，画面站不站得住。 */
const PLAN_V2 = (() => {
  const plan = [];
  const xs = [-6, -2, 2, 6];
  const zs = [0, -1.5, -3, -5];
  for (let zi = 0; zi < zs.length; zi++) {
    for (let xi = 0; xi < xs.length; xi++) {
      // 每个位置配 2 个偏航（0 与 ±25°），凑满 32
      for (const yaw of (zi % 2 === 0 ? [0, -0.44] : [0.44, 0])) {
        plan.push({ ring: 'V', x: xs[xi], z: zs[zi], yaw, pitch: -0.04 + zi * 0.02, eye: null });
      }
    }
  }
  return plan;
})();

const RINGS_32 = (() => {
  const plan = [];
  for (let i = 0; i < 12; i++) plan.push({ ring: 'A', a: (i / 12) * Math.PI * 2, r: 10, pitch: -0.10, eye: null });
  for (let i = 0; i < 12; i++) plan.push({ ring: 'B', a: (i / 12) * Math.PI * 2, r: 26, pitch: 0.05, eye: null });
  for (let i = 0; i < 8; i++) {
    const elevated = i % 2 === 1;
    plan.push({
      ring: 'C', a: (i / 8) * Math.PI * 2, r: 75,
      pitch: elevated ? -0.35 : -0.02,
      eye: elevated ? 14 : null
    });
  }
  return plan;
})();
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
const BASELINE = arg('baseline', '');
/* 采集条件随报告一起落盘：A/B 时若这些字段不一致，指标没有可比性，
   直接拒绝出表，避免把「换了后端/分辨率」误读成清晰度变化。 */
const CAPTURE = {
  w: W, h: H, dpr: 1, gl: GLMODE, views: VIEWS,
  radius: RADIUS, center: CENTER, pitch: PITCH, settle: SETTLE,
  query: EXTRA, clarity: 1
};
const PORT = 9600 + Math.floor(Math.random() * 300);

/* ---------------- 名单 ---------------- */
const IS_GLB = !!GLBLIST;
let images = [];
if (GLBLIST) {
  images = fs.readFileSync(GLBLIST, 'utf8').split(/\r?\n/)
    .map((s) => s.trim()).filter((s) => s && !s.startsWith('#'));
} else if (LIST) {
  images = fs.readFileSync(LIST, 'utf8').split(/\r?\n/)
    .map((s) => s.trim()).filter((s) => s && !s.startsWith('#'));
} else if (CORPUS) {
  images = fs.readdirSync(CORPUS).filter((f) => /\.jpe?g$/i.test(f))
    .sort().map((f) => CORPUS.replace(/\\/g, '/').replace(/\/$/, '') + '/' + f);
}
if (!images.length) {
  console.error('没有输入图片：用 --list / --corpus / --glb-list 指定');
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

// 空洞判据：以渲染背景色（clearColor = world.fogColor）为参照的容差。
// 三处使用（整幅像素 / 1/4 掩码 / 最长横段）必须共用这一个值：
// 改前掩码是 ±8、另两处 ±6，导致块数类指标比占比类多算边界像素、口径不齐。
const VOID_TOL = 6;

function metrics(png, refBg = null) {
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
      // **空洞 = 看到背景**。参照色优先用渲染的 clearColor（= world.fogColor），
      // 而不是"画面主色"。否则水面/雪原这种大面积单色的场景会被整片误判成洞
      // （实测 lake 从侧后方看就是一片平坦湖面，被算成 voidFrac 0.43）。
      const isVoid = refBg
        ? (Math.abs(r - refBg[0]) <= VOID_TOL && Math.abs(g - refBg[1]) <= VOID_TOL && Math.abs(b - refBg[2]) <= VOID_TOL)
        : (Math.abs(r - mr) <= 4 && Math.abs(g - mg) <= 4 && Math.abs(b - mb) <= 4);
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
  /* ---- 新判据：块数而非只算占比 ---- */
  // 1/4 分辨率的掩码：空白块 / 拉丝块
  const sw = Math.max(8, w >> 2), sh = Math.max(6, h >> 2);
  const vmask = new Uint8Array(sw * sh);
  const smask = new Uint8Array(sw * sh);
  const scale = w / sw;
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const sx = Math.min(w - 1, Math.floor((x + 0.5) * scale));
      const sy = Math.min(h - 1, Math.floor((y + 0.5) * scale));
      const o = (sy * w + sx) * bpp;
      const r = data[o], g = data[o + 1], b = data[o + 2];
      vmask[y * sw + x] = (refBg
        ? (Math.abs(r - refBg[0]) <= VOID_TOL && Math.abs(g - refBg[1]) <= VOID_TOL && Math.abs(b - refBg[2]) <= VOID_TOL)
        : (Math.abs(r - mr) <= 6 && Math.abs(g - mg) <= 6 && Math.abs(b - mb) <= 6)) ? 1 : 0;
      // 拉丝像素：竖直差分明显大于水平差分（竖条/拉花），且不是噪点
      if (sy > 0 && sx > 0) {
        const k = sy * w + sx;
        const dv = Math.abs(L[k] - L[k - w]);
        const dh = Math.abs(L[k] - L[k - 1]);
        smask[y * sw + x] = (dv > 0.055 && dv > 2.2 * dh) ? 1 : 0;
      }
    }
  }
  const vb = blobStats(vmask, sw, sh, 3);
  const sb = blobStats(smask, sw, sh, 3);
  // 最长横向空白段（贯穿式的"缝"最难受）：逐行找最长连续空白
  let runMax = 0;
  for (let y = 0; y < h; y += 2) {
    let run = 0;
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * bpp;
      const isVoid = refBg
        ? (Math.abs(data[o] - refBg[0]) <= VOID_TOL && Math.abs(data[o + 1] - refBg[1]) <= VOID_TOL
           && Math.abs(data[o + 2] - refBg[2]) <= VOID_TOL)
        : (Math.abs(data[o] - mr) <= 4 && Math.abs(data[o + 1] - mg) <= 4
           && Math.abs(data[o + 2] - mb) <= 4);
      if (isVoid) { run++; if (run > runMax) runMax = run; } else run = 0;
    }
  }

  return {
    voidBlobs: vb.count,
    // **单块**空白最大占比（防"一块就占半屏"）：必须取 blobStats 的 maxArea。
    // 曾用 totalArea（= 空白总覆盖率，与 voidFrac 重复），"大洞"判据形同虚设。
    voidBlobMax: +(vb.maxArea / Math.max(1, sw * sh)).toFixed(4),
    voidRunMax: +(runMax / Math.max(1, w)).toFixed(4),
    streakBlobs: sb.count,
    streakBlobFrac: +(sb.totalArea / Math.max(1, sw * sh)).toFixed(4),
    voidFrac: +(voidN / total).toFixed(4),
    innerVoid: +(innerN ? innerVoid / innerN : 0).toFixed(4),
    lowVoid: +(lowN ? lowVoid / lowN : 0).toFixed(4),
    borderVoid: +(borderN ? borderVoid / borderN : 0).toFixed(4),
    darkFrac: +(darkN / total).toFixed(4),
    stripe: +(vy / Math.max(1e-6, vx)).toFixed(3),
    detail: +(detail / Math.max(1, cells)).toFixed(4),
    ...clarityMetrics(L, w, h)
  };
}


/* ---------------- 连通域统计（新判据用） ----------------
   原本只有"占比"类指标（voidFrac / stripe），没有"块数"。
   可一个画面里 3% 的空白如果散成 200 个小点，观感远好过 1 块 3% 的大洞；
   拉丝也一样 —— 一条贯穿的竖条比满屏细纹更难受。所以补上"块数/最大块"。
   为了快，连通域在 1/4 分辨率的掩码上做（观感尺度足够）。 */
function blobStats(mask, w, h, minArea) {
  const seen = new Uint8Array(w * h);
  const stack = new Int32Array(w * h);
  let count = 0, maxArea = 0, totalArea = 0;
  for (let i = 0; i < w * h; i++) {
    if (!mask[i] || seen[i]) continue;
    let sp = 0, area = 0;
    stack[sp++] = i; seen[i] = 1;
    while (sp > 0) {
      const k = stack[--sp];
      area++;
      const x = k % w, y = (k / w) | 0;
      if (x > 0 && mask[k - 1] && !seen[k - 1]) { seen[k - 1] = 1; stack[sp++] = k - 1; }
      if (x < w - 1 && mask[k + 1] && !seen[k + 1]) { seen[k + 1] = 1; stack[sp++] = k + 1; }
      if (y > 0 && mask[k - w] && !seen[k - w]) { seen[k - w] = 1; stack[sp++] = k - w; }
      if (y < h - 1 && mask[k + w] && !seen[k + w]) { seen[k + w] = 1; stack[sp++] = k + w; }
    }
    if (area >= minArea) { count++; totalArea += area; if (area > maxArea) maxArea = area; }
  }
  return { count, maxArea, totalArea };
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
    // GLB 模式：每个条目是一个 world.glb，走资产加载；机位/指标口径完全一致
    const url = IS_GLB
      ? `${BASE}/?glb=${encodeURIComponent(rel)}&instant=1&cb=${Date.now()}` + (EXTRA ? '&' + EXTRA : '')
      : `${BASE}/?img=${encodeURIComponent(rel)}&instant=1&name=`
        + encodeURIComponent(path.basename(rel).replace(/\.[a-z0-9]+$/i, ''))
        + (EXTRA ? '&' + EXTRA : '');
    const rec = { image: rel, views: [], ok: false };
    let refBg = null;      // 背景色（0~255），空洞判据用它作参照
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
    if (state.ready === '1') {
      // clearColor 就是 world.fogColor（renderer 里直接 gl.clearColor(fogColor)），
      // 取出来当"空洞"的参照色。取不到就退回旧的主色判定。
      try {
        const c = JSON.parse(await evalIn(send,
          `JSON.stringify((window.__pw && window.__pw.world && window.__pw.world.fogColor) || null)`));
        if (Array.isArray(c) && c.length === 3) {
          refBg = c.map((v) => Math.max(0, Math.min(255, Math.round(Number(v) * 255))));
          rec.refBg = refBg;
        }
      } catch (e) { /* 忽略，退回主色 */ }
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

    const plan = (VIEWS === 32)
      ? (IS_GLB ? PLAN_V2 : RINGS_32)
      : Array.from({ length: VIEWS }, (_, i) => ({
      ring: 'A', a: (i / VIEWS) * Math.PI * 2, r: RADIUS, pitch: PITCH, eye: null
    }));
    for (let i = 0; i < plan.length; i++) {
      const v = plan[i];
      const a = v.a === undefined ? 0 : v.a;
      const x = (v.x !== undefined) ? v.x : cx + Math.sin(a) * v.r;
      const z = (v.z !== undefined) ? v.z : cz + Math.cos(a) * v.r;
      const yawArg = (v.yaw !== undefined) ? v.yaw : (-a);
      const eyeArg = (v.eye === null || v.eye === undefined) ? 'undefined' : String(v.eye);
      const ok = await evalIn(send, `window.__setCam ? window.__setCam(${x.toFixed(2)}, ${z.toFixed(2)}, ${yawArg.toFixed(4)}, ${v.pitch}, ${eyeArg}) : false`);
      if (!ok) throw new Error('页面缺少 window.__setCam');
      await sleep(SETTLE);
      const shot = await send('Page.captureScreenshot', { format: 'png' });
      const buf = Buffer.from(shot.data, 'base64');
      const tag = IS_GLB
        ? (path.basename(path.dirname(rel)) + '_' + path.basename(path.dirname(path.dirname(rel))))
        : path.basename(rel).replace(/\.[a-z0-9]+$/i, '');
      const file = path.join(OUTDIR, `${tag}_${v.ring}${String(i).padStart(2, '0')}.png`);
      fs.writeFileSync(file, buf);
      const m = metrics(decodePNG(buf), refBg);
      rec.views.push({ i, ring: v.ring, r: v.r, a: +(a * 180 / Math.PI).toFixed(0), ...m, file: file.replace(/\\/g, '/') });
    }
    // 俯视：检查地面有没有洞、单体是否浮空（32 视角套餐的 C 圈已含俯视，跳过）
    if (VIEWS !== 32) await evalIn(send, `window.__setCam(${cx}, ${(cz - RADIUS * 0.55).toFixed(2)}, 0, -1.0, 24)`);
    if (VIEWS !== 32) {
      await sleep(SETTLE);
      const top = Buffer.from((await send('Page.captureScreenshot', { format: 'png' })).data, 'base64');
      const topFile = path.join(OUTDIR, `${path.basename(rel).replace(/\.[a-z0-9]+$/i, '')}_top.png`);
      fs.writeFileSync(topFile, top);
      // 俯视也传 refBg：此前主视角用背景色、俯视用画面主色，同一份报告里口径不齐。
      rec.top = { ...metrics(decodePNG(top), refBg), file: topFile.replace(/\\/g, '/') };
    }

    rec.ok = true;
    // 聚合：均值 + 最差视角（改进前/后对比看这两个数）
    const mean = (k) => +(rec.views.reduce((s, v) => s + v[k], 0) / rec.views.length).toFixed(4);
    rec.agg = {
      voidFrac: mean('voidFrac'), darkFrac: mean('darkFrac'),
      innerVoid: mean('innerVoid'), lowVoid: mean('lowVoid'),
      borderVoid: mean('borderVoid'), stripe: mean('stripe'), detail: mean('detail'),
      ...Object.fromEntries(CLARITY_KEYS.map((k) => [k, mean(k)])),
      worstInner: +Math.max(...rec.views.map((v) => v.innerVoid)).toFixed(4),
      worstLow: +Math.max(...rec.views.map((v) => v.lowVoid)).toFixed(4),
      worstStripe: +Math.max(...rec.views.map((v) => v.stripe)).toFixed(3),
      // 块数类：看"碎成多少块"，而不只是总占比
      voidBlobs: Math.max(...rec.views.map((v) => v.voidBlobs || 0)),
      streakBlobs: Math.max(...rec.views.map((v) => v.streakBlobs || 0)),
      voidRunMax: +Math.max(...rec.views.map((v) => v.voidRunMax || 0)).toFixed(4),
      voidBlobMax: +Math.max(...rec.views.map((v) => v.voidBlobMax || 0)).toFixed(4),
      streakBlobFrac: +Math.max(...rec.views.map((v) => v.streakBlobFrac || 0)).toFixed(4)
    };
    results.push(rec);
    const el = ((Date.now() - t0) / 1000).toFixed(0);
    console.log(`✓ ${path.basename(rel)}  void ${rec.agg.voidFrac} inner ${rec.agg.innerVoid} `
      + `low ${rec.agg.lowVoid} stripe ${rec.agg.stripe} vB ${rec.agg.voidBlobs} sB ${rec.agg.streakBlobs} `
      + `run ${rec.agg.voidRunMax} detail ${rec.agg.detail} `
      + `hf ${rec.agg.centerHF} lap ${rec.agg.centerLapVar}  `
      + `[${n + 1}/${images.length}, ${el}s]`);
    fs.writeFileSync(JSONOUT, JSON.stringify({ base: BASE, views: VIEWS, capture: CAPTURE, results }, null, 1));
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
      darkFrac: avg('darkFrac'), detail: avg('detail'),
      ...Object.fromEntries(CLARITY_KEYS.map((k) => [k, avg(k)]))
    },
    worstInner: worst
  };
  fs.writeFileSync(JSONOUT, JSON.stringify({ base: BASE, views: VIEWS, capture: CAPTURE, summary, results }, null, 1));
  console.log('汇总 ' + JSON.stringify(summary.avg) + '  失败 ' + summary.failed);
  console.log('最差 inner/low void：' + worst.slice(0, 5).map((w) => path.basename(w.image) + ' ' + w.worstInner + '/' + w.worstLow).join(' | '));
  console.log('明细 → ' + JSONOUT + '，截图 → ' + OUTDIR);
}
/* ---------------- 清晰度基线 / A-B 对比 ---------------- */
/* 第一次带 --baseline 跑 = 建立基线；之后同参数再跑 = 逐指标出 A/B 表。
   刻意只用一个参数：基线文件和对比文件是同一个，避免两份报告用错版本。 */
if (BASELINE && fs.existsSync(JSONOUT)) {
  const cur = JSON.parse(fs.readFileSync(JSONOUT, 'utf8'));
  if (!fs.existsSync(BASELINE)) {
    fs.mkdirSync(path.dirname(BASELINE), { recursive: true });
    fs.writeFileSync(BASELINE, JSON.stringify(cur, null, 1));
    if (exitCode) console.warn('注意：本次有失败样本，基线不完整，建议修复后重跑');
    console.log(`基线已建立 → ${BASELINE}（同参数再跑一次即为 A/B）`);
  } else {
    const base = JSON.parse(fs.readFileSync(BASELINE, 'utf8'));
    const diff = captureDiff(base.capture, cur.capture);
    if (diff.length) {
      console.error(`采集条件不一致，拒绝对比：${diff.join(', ')}`);
      console.error('  基线 ' + JSON.stringify(base.capture));
      console.error('  本次 ' + JSON.stringify(cur.capture));
      exitCode = 2;   // 末尾统一写 process.exitCode，避免被成功码覆盖
    } else {
      const keys = [...CLARITY_KEYS, 'detail', 'voidFrac', 'innerVoid', 'lowVoid', 'borderVoid', 'darkFrac', 'stripe'];
      const mapOf = (d) => new Map((d.results || []).filter((r) => r.ok && r.agg).map((r) => [r.image, r.agg]));
      const bm = mapOf(base), cm = mapOf(cur);
      const shared = [...cm.keys()].filter((k) => bm.has(k));
      console.log(`\nA/B 对比：基线 ${path.basename(BASELINE)} · 共同样本 ${shared.length} 张 · 采集条件一致`);
      console.log('指标'.padEnd(16) + '基线'.padEnd(11) + '当前'.padEnd(11) + '变化'.padEnd(11) + '改善/退化');
      for (const k of keys) {
        let bs = 0, cs = 0, betterN = 0, worseN = 0;
        for (const img of shared) {
          const b = bm.get(img)[k], c = cm.get(img)[k];
          if (!Number.isFinite(b) || !Number.isFinite(c)) continue;
          bs += b; cs += c;
          const hi = HIGHER_IS_BETTER[k] !== false;
          if (hi ? c > b : c < b) betterN++;
          else if (hi ? c < b : c > b) worseN++;
        }
        const n = shared.length || 1;
        const pct = bs ? ((cs - bs) / bs) * 100 : 0;
        console.log(k.padEnd(16) + (bs / n).toFixed(5).padEnd(11) + (cs / n).toFixed(5).padEnd(11)
          + ((pct >= 0 ? '+' : '') + pct.toFixed(1) + '%').padEnd(11) + `${betterN}/${worseN}`);
      }
    }
  }
}
process.exitCode = exitCode;
