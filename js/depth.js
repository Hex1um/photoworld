/* 单目深度估计：Depth-Anything-V2-Small (ONNX, 自托管, 浏览器内 CPU 推理)
 *
 * 输出归一化深度图（0=最远, 1=最近），用于把照片还原成有真实起伏的三维表面。
 * 模型加载失败时上层会回退到旧的"天际线分层"方案。
 */
(function (global) {
  'use strict';

  // 以本脚本所在目录推算站点根，兼容子目录部署
  const SELF = (document.currentScript && document.currentScript.src) || location.href;
  const BASE = new URL('../', SELF).href;
  const ORT_JS = BASE + 'vendor/ort/ort.min.js';
  const ORT_DIR = BASE + 'vendor/ort/';
  const MODEL = BASE + 'models/model_quantized.onnx';
  const SIZE = 518;   // DPT 输入尺寸（14 的倍数）

  const S = { session: null, loading: null, failed: null, ort: null, listeners: [], stage: null };

  /* 同一张照片重复进入世界时复用深度图，省掉一次约 5s 的 CPU 推理。
     key 用「尺寸 + 16×16 缩略图内容指纹」而不是纯 URL：CDN 同名换图、
     data URL 重建都会让纯 URL 缓存误命中。只做内存 LRU（最多 4 张）；
     跨刷新持久化要 IndexedDB，会带来配额与异步复杂度，静态站点收益不确定，
     留作后续 A/B，不在这版擅自引入。 */
  const CACHE = new Map();
  const CACHE_MAX = 4;

  function contentKey(img) {
    const iw = img.naturalWidth || img.width;
    const ih = img.naturalHeight || img.height;
    const cv = document.createElement('canvas');
    cv.width = cv.height = 16;
    let fp = 2166136261;
    try {
      const c = cv.getContext('2d', { willReadFrequently: true });
      c.drawImage(img, 0, 0, 16, 16);
      const p = c.getImageData(0, 0, 16, 16).data;
      for (let i = 0; i < p.length; i += 4) {
        fp = Math.imul(fp ^ p[i], 16777619);
        fp = Math.imul(fp ^ p[i + 1], 16777619);
        fp = Math.imul(fp ^ p[i + 2], 16777619);
      }
    } catch (e) {
      // 画布被跨域图片污染时退回 URL 长度；这张图本来也进不了推理
      fp = String(img.currentSrc || img.src || '').length;
    }
    return iw + 'x' + ih + ':' + (fp >>> 0).toString(16);
  }

  function clearCache() { CACHE.clear(); }
  function cacheStats() { return { size: CACHE.size, max: CACHE_MAX }; }

  /* 等比缩放 + 黑底 padding 到正方形（与 pipeline/depth.py::_pad_to_square 一致）。
     旧实现 drawImage(img, 0, 0, SIZE, SIZE) 会把横构图直接压扁，违反离线
     pipeline/README.md「绝不拉伸」的约定，也让两条链路的几何不可比。 */
  function squarePlan(w, h, size) {
    const s = size / Math.max(w, h);
    const nw = Math.max(1, Math.round(w * s));
    const nh = Math.max(1, Math.round(h * s));
    return { ox: Math.floor((size - nw) / 2), oy: Math.floor((size - nh) / 2), nw, nh };
  }

  function quantile(sorted, q) {
    const pos = (sorted.length - 1) * q;
    const i = Math.floor(pos), f = pos - i;
    return sorted[i] * (1 - f) + sorted[Math.min(sorted.length - 1, i + 1)] * f;
  }

  /* 模型输出 → 裁掉 padding → 归一化深度，逐项对齐离线后处理：
       1) 顶/底 1/8 带均值判极性（反了就取负）
       2) 2/98 分位截断归一化（替代对离群值敏感的全图 min-max）
       3) 行均值与行号相关系数兜底，统一成「越大越近」
     离线会再 BICUBIC resize 回原图尺寸；前端按归一化 uv 采样，
     保持裁剪后的 cw×ch 即可（各向同性缩放等价，少一次插值）。 */
  function postprocess(raw, dims, crop) {
    const oh = (dims && dims.length >= 2) ? dims[dims.length - 2] : SIZE;
    const ow = (dims && dims.length >= 2) ? dims[dims.length - 1] : SIZE;
    const sx = ow / SIZE, sy = oh / SIZE;
    const cx = Math.max(0, Math.round(crop.ox * sx));
    const cy = Math.max(0, Math.round(crop.oy * sy));
    const cw = Math.max(1, Math.min(ow - cx, Math.round(crop.nw * sx)));
    const ch = Math.max(1, Math.min(oh - cy, Math.round(crop.nh * sy)));
    const d = new Float32Array(cw * ch);
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) d[y * cw + x] = raw[(cy + y) * ow + (cx + x)];
    }

    const band = Math.max(1, Math.floor(ch / 8));
    let top = 0, bot = 0;
    for (let y = 0; y < band; y++) for (let x = 0; x < cw; x++) top += d[y * cw + x];
    for (let y = ch - band; y < ch; y++) for (let x = 0; x < cw; x++) bot += d[y * cw + x];
    const inverted = (top / (band * cw)) > (bot / (band * cw));
    if (inverted) for (let i = 0; i < d.length; i++) d[i] = -d[i];

    const sorted = Float32Array.from(d);
    sorted.sort();                                  // TypedArray 默认按数值升序
    const lo = quantile(sorted, 0.02), hi = quantile(sorted, 0.98);
    const span = Math.max(1e-6, hi - lo);
    for (let i = 0; i < d.length; i++) d[i] = Math.min(1, Math.max(0, (d[i] - lo) / span));

    const rows = new Float64Array(ch);
    for (let y = 0; y < ch; y++) {
      let s = 0;
      for (let x = 0; x < cw; x++) s += d[y * cw + x];
      rows[y] = s / cw;
    }
    let mr = 0, mi = 0;
    for (let y = 0; y < ch; y++) { mr += rows[y]; mi += y; }
    mr /= ch; mi /= ch;
    let num = 0, vr = 0, vi = 0;
    for (let y = 0; y < ch; y++) {
      const a = rows[y] - mr, b = y - mi;
      num += a * b; vr += a * a; vi += b * b;
    }
    const corr = (vr > 1e-12 && vi > 1e-12) ? num / Math.sqrt(vr * vi) : 0;
    const flipped = corr < 0;
    if (flipped) for (let i = 0; i < d.length; i++) d[i] = 1 - d[i];

    return { data: d, width: cw, height: ch, inverted, flipped, corr, lo, hi };
  }

  function loadScript(src) {
    if (global.ort) return Promise.resolve();
    return new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = src;
      s.onload = () => res();
      s.onerror = () => rej(new Error('运行时加载失败'));
      document.head.appendChild(s);
    });
  }

  /* 模型字节：流式下载 + 真实进度 + 浏览器缓存。
   *
   * 旧实现 `fetch(url).then(r => r.arrayBuffer())` 只有「拿到响应头」和「下载结束」
   * 两个事件，中间 27MB 全程没有回调 —— 界面就卡在 15% 不动（手机 4G 上能卡十几秒
   * 到一分钟），用户以为死了。现在按 chunk 读，进度 = 已下载字节 / content-length。
   *
   * 另外两级缓存：
   *   1) Cache Storage（跨会话持久）：第二次进世界直接命中，不再走网络；
   *   2) HTTP 缓存由 CDN 自己管，这里不干预（GitHub Pages 带 ETag）。
   * 没有 caches API（file:// 或旧浏览器）时静默退化为直接下载。
   *
   * 卡死保护：20 秒没有任何新字节就 abort，让上层按「模型不可用」回退到天际线方案，
   * 而不是让载入界面永远停在某一个百分比。
   */
  const MODEL_CACHE = 'pw-depth-model-v1';
  const MB = 1048576;

  async function fromCacheStorage() {
    try {
      if (!global.caches) return null;
      const c = await global.caches.open(MODEL_CACHE);
      const hit = await c.match(MODEL);
      if (!hit) return null;
      return await hit.arrayBuffer();
    } catch (e) {
      return null;                       // 隐私模式 / 配额满：当作没缓存
    }
  }

  async function toCacheStorage(buf) {
    try {
      if (!global.caches) return;
      const c = await global.caches.open(MODEL_CACHE);
      await c.put(MODEL, new Response(buf, {
        headers: {
          'Content-Type': 'application/octet-stream',
          'Content-Length': String(buf.byteLength)
        }
      }));
    } catch (e) { /* 存不下不影响本次使用 */ }
  }

  async function fetchBuffer(onProgress) {
    const report = (p, label) => { if (onProgress) onProgress(p, label); };

    const cached = await fromCacheStorage();
    if (cached && cached.byteLength > 1e6) {
      report(0.9, '深度模型已在本地缓存（' + (cached.byteLength / MB).toFixed(1) + 'MB）');
      return cached;
    }

    report(0.02, '连接模型服务器…');
    /* 卡死保护依赖 AbortController + setInterval。Node 回归沙箱里没有
       AbortController（也不该为了一个 UI 计时器去 polyfill），这里按能力降级：
       没有就用不带 signal 的 fetch，保证纯逻辑回归能跑。 */
    const canAbort = typeof AbortController === 'function' && typeof setInterval === 'function';
    const ctrl = canAbort ? new AbortController() : null;
    let lastByte = performance.now();
    const guard = canAbort ? setInterval(() => {
      if (performance.now() - lastByte > 20000) ctrl.abort();
    }, 1000) : null;

    try {
      const r = await fetch(MODEL, ctrl ? { signal: ctrl.signal } : undefined);
      if (!r.ok) throw new Error('模型下载失败 ' + r.status);
      const total = (r.headers && typeof r.headers.get === 'function')
        ? (Number(r.headers.get('content-length')) || 0) : 0;
      if (!r.body || !r.body.getReader) {
        // 极老浏览器没有流式 body：退回一次性读取，至少给个"下载中"的标签
        report(0.05, '下载深度模型（浏览器不支持流式进度）…');
        const buf = await r.arrayBuffer();
        report(0.9, '深度模型下载完成（' + (buf.byteLength / MB).toFixed(1) + 'MB）');
        return buf;
      }

      const reader = r.body.getReader();
      const chunks = [];
      let got = 0, lastPct = -1;
      const t0 = performance.now();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        got += value.length;
        lastByte = performance.now();
        const p = total ? Math.min(1, got / total) : Math.min(0.85, got / (27 * MB));
        if (p - lastPct >= 0.01 || p === 1) {
          lastPct = p;
          const secs = Math.max(0.2, (performance.now() - t0) / 1000);
          const speed = (got / MB / secs).toFixed(1);
          report(0.02 + p * 0.86, total
            ? '下载深度模型 ' + Math.round(p * 100) + '%（'
              + (got / MB).toFixed(1) + ' / ' + (total / MB).toFixed(1) + 'MB · ' + speed + 'MB/s）'
            : '下载深度模型 ' + (got / MB).toFixed(1) + 'MB（' + speed + 'MB/s）');
        }
      }

      const buf = new Uint8Array(got);
      let off = 0;
      for (const c of chunks) { buf.set(c, off); off += c.length; }
      report(0.9, '深度模型下载完成（' + (got / MB).toFixed(1) + 'MB），写入本地缓存');
      toCacheStorage(buf.buffer);
      return buf.buffer;
    } finally {
      if (guard) clearInterval(guard);
    }
  }

  /* 进度广播：落地页会预取模型，用户此时点进世界 -> 两个调用方共用同一个 init。
     没有广播的话，第二个调用方拿不到任何回调，界面又会「卡在某个百分比」。
     `S.stage` 记住最后一次进度，新订阅者立刻收到当前状态。 */
  function attach(onProgress) {
    if (!onProgress) return;
    S.listeners.push(onProgress);
    if (S.stage) { try { onProgress(S.stage.p, S.stage.label); } catch (e) {} }
  }
  function broadcast(p, label) {
    S.stage = { p, label };
    for (const f of S.listeners) { try { f(p, label); } catch (e) { /* 单个订阅者异常不影响加载 */ } }
  }

  async function init(onProgress) {
    if (S.session) {
      if (onProgress) onProgress(1, '深度模型已就绪');
      return S.session;
    }
    if (S.failed) throw S.failed;
    if (S.loading) { attach(onProgress); return S.loading; }

    attach(onProgress);
    S.loading = (async () => {
      broadcast(0, '加载推理运行时…');
      await loadScript(ORT_JS);
      const ort = global.ort;
      if (!ort) throw new Error('onnxruntime-web 未就绪');
      ort.env.wasm.wasmPaths = ORT_DIR;
      ort.env.wasm.numThreads = 1;   // 静态站点没有 COOP/COEP，只能单线程
      ort.env.wasm.simd = true;
      ort.env.logLevel = 'error';
      S.ort = ort;

      const buf = await fetchBuffer(broadcast);
      // 建会话 = 解析 ONNX 图 + 编译 WASM 内核，弱机上要几秒到十几秒。
      // 这一段以前完全静默，界面停在「下载 100%」不动，同样像卡死。
      broadcast(0.92, '编译 WASM 内核（解析 ONNX 计算图）…');
      const t0 = performance.now();
      const session = await ort.InferenceSession.create(buf, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      S.session = session;
      broadcast(1, '深度模型就绪（' + ((performance.now() - t0) / 1000).toFixed(1) + 's）');
      return session;
    })();

    try {
      const out = await S.loading;
      S.listeners.length = 0;
      return out;
    } catch (e) {
      S.failed = e;
      S.loading = null;
      S.listeners.length = 0;
      throw e;
    }
  }

  /* 图片 -> 归一化深度（width×height，0..1，1 为最近；比例与原图一致） */
  async function estimate(img, onProgress) {
    const key = contentKey(img);
    const hit = CACHE.get(key);
    // data 是只读约定：上层 buildRelief / drawDepthPreview 都只采样不写入
    if (hit) return Object.assign({}, hit, { cached: true, ms: 0 });

    const session = await init(onProgress);
    const ort = S.ort;

    const iw = img.naturalWidth || img.width;
    const ih = img.naturalHeight || img.height;
    const crop = squarePlan(iw, ih, SIZE);

    const cv = document.createElement('canvas');
    cv.width = SIZE; cv.height = SIZE;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#000';                       // padding 必须是纯黑，与离线一致
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.drawImage(img, crop.ox, crop.oy, crop.nw, crop.nh);
    const rgba = ctx.getImageData(0, 0, SIZE, SIZE).data;

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const n = SIZE * SIZE;
    const input = new Float32Array(n * 3);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      input[i] = (rgba[p] / 255 - mean[0]) / std[0];
      input[n + i] = (rgba[p + 1] / 255 - mean[1]) / std[1];
      input[2 * n + i] = (rgba[p + 2] / 255 - mean[2]) / std[2];
    }

    const feeds = {};
    feeds[session.inputNames[0]] = new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]);
    const t0 = performance.now();
    const out = await session.run(feeds);
    const dt = performance.now() - t0;

    const o = out[session.outputNames[0]];
    const pp = postprocess(o.data, o.dims || [1, 1, SIZE, SIZE], crop);
    const result = {
      data: pp.data, width: pp.width, height: pp.height,
      size: pp.width,                 // 兼容旧字段；采样请用 width/height
      ms: Math.round(dt), inverted: pp.inverted, flippedByCorr: pp.flipped,
      corr: pp.corr
    };
    CACHE.set(key, result);
    if (CACHE.size > CACHE_MAX) CACHE.delete(CACHE.keys().next().value);
    return Object.assign({}, result, { cached: false });
  }

  global.PW.depth = {
    init, estimate, clearCache, cacheStats,
    // 暴露纯函数供 tools/test_regression.mjs 做前后端一致性与缓存回归
    __internals: { squarePlan, postprocess, quantile, contentKey },
    get ready() { return !!S.session; },
    get failed() { return S.failed; }
  };
})(window);
