#!/usr/bin/env node
/**
 * GitHub 图片采集：从公开仓库的 raw 直链拉取风景/壁纸 JPEG，用于图集训练。
 *
 * 为什么不是 Pixabay：pixabay.com 对 curl 返回 403，headless 浏览器拿到的是
 * 「请稍候…」JS 挑战页，站点抓取这条路在本机走不通（图搜引擎也索引不到
 * cdn.pixabay.com 直链）。GitHub 是本机唯一稳定可达的源，因此改为从仓库取图：
 * 公开仓库里的壁纸集多数是实拍风景，且同样受其仓库 LICENSE 约束（见 manifest 记录）。
 *
 * 过滤（宁可少收，不要脏数据）：
 *   - JPEG 真解码通过，宽 ≥ 900，横向构图（宽高比 1.15~2.6）
 *   - 文件 ≥ 45KB（排除图标/缩略图）
 *   - 天空启发式：顶部 12% 的亮度方差小、且整体比底部亮 —— 风景照的常见构图；
 *     动漫壁纸/纯色块/室内图大多过不了这条
 *
 *   node tools/harvest_github.mjs --target 900 --out corpus
 *   node tools/harvest_github.mjs --repos a/b,c/d --target 200 --dry
 */
import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import jpeg from 'jpeg-js';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? argv[i + 1] : d; };
const has = (n) => argv.includes('--' + n);

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
const OUT = path.join(ROOT, arg('out', 'corpus'));
const TARGET = parseInt(arg('target', '600'), 10);
const PER_REPO = parseInt(arg('per-repo', '220'), 10);
const MIN_W = parseInt(arg('min-w', '900'), 10);
const MIN_BYTES = parseInt(arg('min-bytes', '45000'), 10);
const DRY = has('dry');
const WORKERS = parseInt(arg('workers', '6'), 10);

/* 默认仓库池：壁纸/风景集，实测每个 100~250 张真实 JPEG。
   按「风景占比」粗排，先跑概率高的。 */
const DEFAULT_REPOS = [
  'Doble-2/scenery-wallpapers',
  'Joao2Pereira1/Wallpapers',
  'na-ive/wallpapers',
  'malwarepad/wallpapers',
  'ItsTerm1n4l/Wallpapers-old-archive',
  'Nirupam-Ghosh2004/VisualVault',
  'IcePocket/Wallpapers',
  'OntologicalShock/nord-stalker-wallpapers'
];

const REPOS = (arg('repos', '') || '').split(',').map((s) => s.trim()).filter(Boolean).length
  ? arg('repos', '').split(',').map((s) => s.trim()).filter(Boolean)
  : DEFAULT_REPOS;

function token() {
  if (process.env.GITHUB_TOKEN) return process.env.GITHUB_TOKEN;
  try {
    const out = execFileSync('git', ['credential', 'fill'], {
      input: 'protocol=https\nhost=github.com\n\n', encoding: 'utf8'
    });
    const m = out.match(/password=(.+)/);
    return m ? m[1].trim() : '';
  } catch (e) { return ''; }
}
const TOK = token();

function api(p) {
  const args = ['-k', '-s', '-m', '40', '-H', 'Accept: application/vnd.github+json'];
  if (TOK) args.push('-H', 'Authorization: Bearer ' + TOK);
  args.push('https://api.github.com' + p);
  const out = execFileSync('curl', args, { maxBuffer: 64 * 1024 * 1024 });
  try { return JSON.parse(out.toString('utf8')); } catch (e) { return null; }
}

/* 真异步下载：上一版用 execFileSync，虽然开了 8 个 worker，但同步调用把事件循环
   堵死，实际是串行（0.18 张/s，收 800 张要一小时）。改用 undici/https。
   本机代理是自签证书，因此对该请求单独关闭证书校验（不动全局 env）。 */
import https from 'node:https';
function fetchRaw(repo, file) {
  const url = 'https://raw.githubusercontent.com/' + repo + '/HEAD/'
    + file.split('/').map(encodeURIComponent).join('/');
  return new Promise((resolve) => {
    const req = https.get(url, {
      timeout: 60000,
      headers: { 'User-Agent': 'photoworld-trainer' },
      rejectUnauthorized: false
    }, (res) => {
      if (res.statusCode !== 200) { res.resume(); return resolve(null); }
      const chunks = [];
      let size = 0;
      res.on('data', (c) => {
        size += c.length;
        if (size > 40 * 1024 * 1024) { req.destroy(); resolve(null); return; }
        chunks.push(c);
      });
      res.on('end', () => resolve(Buffer.concat(chunks)));
    });
    req.on('error', () => resolve(null));
    req.on('timeout', () => { req.destroy(); resolve(null); });
  });
}

/* 天空启发式：顶部 12% 与底部 25% 的亮度差 + 顶部方差。 */
function looksLikeLandscape(img) {
  const { width: W, height: H, data } = img;
  const rows = Math.max(2, Math.floor(H * 0.12));
  const rowsB = Math.max(2, Math.floor(H * 0.25));
  let topL = 0, topN = 0, botL = 0, botN = 0;
  const tops = [];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < W; x += 4) {
      const i = (y * W + x) * 4;
      const l = (0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]) / 255;
      topL += l; topN++; tops.push(l);
    }
  }
  for (let y = H - rowsB; y < H; y++) {
    for (let x = 0; x < W; x += 4) {
      const i = (y * W + x) * 4;
      botL += (0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]) / 255;
      botN++;
    }
  }
  topL /= Math.max(1, topN); botL /= Math.max(1, botN);
  const mean = topL;
  let varr = 0;
  for (const l of tops) varr += (l - mean) * (l - mean);
  varr /= Math.max(1, tops.length);
  // 天空在上：顶部更亮或至少不比底部暗很多；顶部本身比较"平"（云/雾/蓝天都平）
  return (topL > botL * 0.92) && varr < 0.06;
}

function verify(buf) {
  if (!buf || buf.length < MIN_BYTES) return null;
  let img;
  try { img = jpeg.decode(buf, { useTArray: true, maxMemoryUsageInMB: 512 }); } catch (e) { return null; }
  if (!img || !img.width) return null;
  const { width: w, height: h } = img;
  if (w < MIN_W || h < 400) return null;
  const ar = w / h;
  if (ar < 1.15 || ar > 2.6) return null;
  if (!looksLikeLandscape(img)) return null;
  return { w, h, digest: crypto.createHash('md5').update(buf).digest('hex').slice(0, 16) };
}

/* ---------------- 主流程 ---------------- */
fs.mkdirSync(OUT, { recursive: true });
const manifestPath = path.join(OUT, 'manifest.json');
const manifest = fs.existsSync(manifestPath) ? JSON.parse(fs.readFileSync(manifestPath, 'utf8')) : [];
const seen = new Set(manifest.map((m) => m.file));

let added = 0, tried = 0, rejected = 0;
const t0 = Date.now();
const queue = [];

for (const repo of REPOS) {
  if (added >= TARGET) break;
  const tree = api('/repos/' + repo + '/git/trees/HEAD?recursive=1');
  if (!tree || !tree.tree) { console.log(`✗ ${repo}  无法读取目录树`); continue; }
  const files = tree.tree
    .filter((x) => x.type === 'blob' && /\.jpe?g$/i.test(x.path) && (x.size || 0) >= MIN_BYTES)
    .sort(() => Math.random() - 0.5)
    .slice(0, PER_REPO);
  console.log(`· ${repo}  候选 ${files.length} 张`);
  for (const f of files) queue.push({ repo, path: f.path });
}

console.log(`共 ${queue.length} 个候选，目标再收 ${TARGET} 张`);

/* 并发下载：单张 raw 大约 1~3s，串行收 800 张要 40 分钟，6 并发足够且不激进。
   计数与写 manifest 都在主线程（Node 单线程），不需要锁。 */
ML: {
  let next = 0;
  const worker = async () => {
    for (;;) {
      if (added >= TARGET) return;
      const i = next++;
      if (i >= queue.length) return;
      const item = queue[i];
      tried++;
      const buf = await fetchRaw(item.repo, item.path);
      const info = verify(buf);
      if (!info) { rejected++; continue; }
      if (seen.has(info.digest + '.jpg')) continue;
      if (!DRY) {
        fs.writeFileSync(path.join(OUT, info.digest + '.jpg'), buf);
        manifest.push({
          file: info.digest + '.jpg', cat: 'landscape', query: 'github:' + item.repo,
          src: 'https://github.com/' + item.repo + '/blob/HEAD/' + item.path,
          repo: item.repo, path: item.path,
          license: 'public repo (见源仓库 LICENSE)', w: info.w, h: info.h
        });
        seen.add(info.digest + '.jpg');
      }
      added++;
      if (added % 20 === 0) {
        if (!DRY) fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 1));
        const sp = (Date.now() - t0) / 1000;
        console.log(`  已收 ${added} / ${TARGET}（尝试 ${tried}，拒绝 ${rejected}）${sp.toFixed(0)}s `
          + `≈ ${(added / sp).toFixed(1)} 张/s`);
      }
    }
  };
  await Promise.all(Array.from({ length: WORKERS }, worker));
}

if (!DRY) fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 1));
console.log(`完成：新增 ${added} 张（尝试 ${tried}，过滤 ${rejected}）· 清单 ${manifest.length} 张 → ${path.relative(ROOT, manifestPath)}`);
