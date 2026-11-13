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

  const S = { session: null, loading: null, failed: null, ort: null };

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

  function fetchBuffer(url, onProgress) {
    return fetch(url).then((r) => {
      if (!r.ok) throw new Error('模型下载失败 ' + r.status);
      if (onProgress) onProgress(0.15);
      return r.arrayBuffer();
    }).then((b) => {
      if (onProgress) onProgress(1);
      return b;
    });
  }

  async function init(onProgress) {
    if (S.session) return S.session;
    if (S.failed) throw S.failed;
    if (S.loading) return S.loading;

    S.loading = (async () => {
      await loadScript(ORT_JS);
      const ort = global.ort;
      if (!ort) throw new Error('onnxruntime-web 未就绪');
      ort.env.wasm.wasmPaths = ORT_DIR;
      ort.env.wasm.numThreads = 1;   // 静态站点没有 COOP/COEP，只能单线程
      ort.env.wasm.simd = true;
      ort.env.logLevel = 'error';
      S.ort = ort;

      const buf = await fetchBuffer(MODEL, onProgress);
      const session = await ort.InferenceSession.create(buf, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      S.session = session;
      return session;
    })();

    try {
      return await S.loading;
    } catch (e) {
      S.failed = e;
      S.loading = null;
      throw e;
    }
  }

  /* 图片 -> 归一化深度（Float32Array SIZE*SIZE，0..1，1 为最近） */
  async function estimate(img, onProgress) {
    const session = await init(onProgress);
    const ort = S.ort;

    const cv = document.createElement('canvas');
    cv.width = SIZE; cv.height = SIZE;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, SIZE, SIZE);
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
    const d = o.data;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < d.length; i++) {
      const v = d[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const span = Math.max(1e-6, hi - lo);
    const depth = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) depth[i] = (d[i] - lo) / span;

    // 自动判定极性：天空应在远端。若顶部比底部更"近"则翻转。
    let topSum = 0, botSum = 0, cnt = 0;
    for (let y = 0; y < 40; y++) {
      for (let x = 0; x < SIZE; x += 4) { topSum += depth[y * SIZE + x]; cnt++; }
    }
    const nTop = cnt;
    let botCnt = 0;
    for (let y = SIZE - 40; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x += 4) { botSum += depth[y * SIZE + x]; botCnt++; }
    }
    const invert = (topSum / nTop) > (botSum / botCnt);
    if (invert) {
      for (let i = 0; i < depth.length; i++) depth[i] = 1 - depth[i];
    }

    return { data: depth, size: SIZE, ms: Math.round(dt), inverted: invert };
  }

  global.PW.depth = {
    init, estimate,
    get ready() { return !!S.session; },
    get failed() { return S.failed; }
  };
})(window);
