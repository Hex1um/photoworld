/* 近景物体识别：YOLOv8n-seg (COCO 80 类, 14MB, ONNX, 浏览器内 CPU 推理)
 *
 * 模型来源（真实可用，已下载到 models/yolov8n-seg.onnx）：
 *   HuggingFace EdgeFirst/yolov8-seg · onnx/yolov8n-seg-fp32.onnx
 *   https://hf-mirror.com/EdgeFirst/yolov8-seg/resolve/main/onnx/yolov8n-seg-fp32.onnx
 *   输入 images [1,3,640,640] float32，输出 output0 [1,116,8400]（4 box + 80 cls + 32 mask coeff）
 *   与 output1 [1,32,160,160]（mask protos）。COCO val2017 Box mAP@0.5 = 51.43%。
 *
 * 用途：给「近处物体不要延伸到远处」提供**语义边界**——
 *   深度只能在数值上判断"更近"，分不清"一块石头"和"一座山"的边界；
 *   YOLO 的实例 mask 直接给出物体轮廓，buildCartesianWorld 用它在边界断开网格，
 *   并把近景物体的距离限制在自身深度附近。
 * 失败不阻塞主流程：没有模型时回退到「比同排深度基线更近」的纯几何判定。
 */
(function (global) {
  'use strict';

  const SELF = (document.currentScript && document.currentScript.src) || location.href;
  const BASE = new URL('../', SELF).href;
  const MODEL = BASE + 'models/yolov8n-seg.onnx';
  const SIZE = 640, FW = 96, FH = 96;
  const CONF = 0.28, IOU = 0.45;

  const COCO = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
  ];

  const S = { session: null, loading: null, failed: null, ort: null };
  const CACHE = new Map();

  function loadScript(src) {
    if (global.ort) return Promise.resolve();
    return new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = src;
      s.onload = () => res();
      s.onerror = () => rej(new Error('ort-web 加载失败'));
      document.head.appendChild(s);
    });
  }

  function contentKey(img) {
    // 与 depth.js 同一套指纹：尺寸 + 16×16 缩略图
    const iw = img.naturalWidth || img.width, ih = img.naturalHeight || img.height;
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
    } catch (e) { fp = String(img.currentSrc || img.src || '').length; }
    return iw + 'x' + ih + ':' + (fp >>> 0).toString(16);
  }

  async function init(onProgress) {
    if (S.session) return S.session;
    if (S.failed) throw S.failed;
    if (S.loading) return S.loading;
    S.loading = (async () => {
      await loadScript(BASE + 'vendor/ort/ort.min.js');
      const ort = global.ort;
      if (!ort) throw new Error('onnxruntime-web 未就绪');
      ort.env.wasm.wasmPaths = BASE + 'vendor/ort/';
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
      ort.env.logLevel = 'error';
      S.ort = ort;
      const r = await fetch(MODEL);
      if (!r.ok) throw new Error('YOLO 模型下载失败 ' + r.status);
      if (onProgress) onProgress(0.5);
      const buf = await r.arrayBuffer();
      const session = await ort.InferenceSession.create(buf, {
        executionProviders: ['wasm'], graphOptimizationLevel: 'all'
      });
      if (onProgress) onProgress(1);
      S.session = session;
      return session;
    })();
    try { return await S.loading; }
    catch (e) { S.failed = e; S.loading = null; throw e; }
  }

  function letterbox(img) {
    const iw = img.naturalWidth || img.width;
    const ih = img.naturalHeight || img.height;
    const sc = SIZE / Math.max(iw, ih);
    const nw = Math.max(1, Math.round(iw * sc)), nh = Math.max(1, Math.round(ih * sc));
    const ox = Math.floor((SIZE - nw) / 2), oy = Math.floor((SIZE - nh) / 2);
    const cv = document.createElement('canvas');
    cv.width = cv.height = SIZE;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = 'rgb(114,114,114)';
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.drawImage(img, ox, oy, nw, nh);
    const rgba = ctx.getImageData(0, 0, SIZE, SIZE).data;
    const n = SIZE * SIZE;
    const data = new Float32Array(n * 3);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      data[i] = rgba[p] / 255;
      data[n + i] = rgba[p + 1] / 255;
      data[2 * n + i] = rgba[p + 2] / 255;
    }
    return { data, ox, oy, nw, nh, iw, ih };
  }

  function iou(a, b) {
    const x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y);
    const x1 = Math.min(a.x + a.w, b.x + b.w), y1 = Math.min(a.y + a.h, b.y + b.h);
    const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
    return inter / Math.max(1e-6, a.w * a.h + b.w * b.h - inter);
  }

  async function estimate(img, onProgress) {
    const key = contentKey(img);
    const hit = CACHE.get(key);
    if (hit) return Object.assign({}, hit, { cached: true, ms: 0 });
    const session = await init(onProgress);
    const ort = S.ort;
    const lb = letterbox(img);
    const t0 = performance.now();
    const out = await session.run({
      images: new ort.Tensor('float32', lb.data, [1, 3, SIZE, SIZE])
    });
    const dt = performance.now() - t0;

    const o0 = out.output0.data;          // [116,8400]
    const o1 = out.output1.data;          // [32,160,160]
    const N = 8400, M = 32, PS = 160, PS2 = PS * PS;

    // 1) 解码：每个 anchor 取最大类别分
    const cands = [];
    for (let a = 0; a < N; a++) {
      let best = 0, cls = -1;
      for (let c = 0; c < 80; c++) {
        const s = o0[(4 + c) * N + a];
        if (s > best) { best = s; cls = c; }
      }
      if (best < CONF) continue;
      const cx = o0[a], cy = o0[N + a], w = o0[2 * N + a], h = o0[3 * N + a];
      const coeff = new Float32Array(M);
      for (let m = 0; m < M; m++) coeff[m] = o0[(84 + m) * N + a];
      cands.push({ x: cx - w / 2, y: cy - h / 2, w, h, cx, cy, score: best, cls, coeff });
    }
    cands.sort((p, q) => q.score - p.score);

    // 2) NMS
    const keep = [];
    for (const c of cands) {
      let ok = true;
      for (const k of keep) if (iou(c, k) > IOU) { ok = false; break; }
      if (ok) { keep.push(c); if (keep.length >= 24) break; }
    }

    // 3) mask：sigmoid(coeff · proto) > 0.5，并集合成 96×96 前景网格
    const fg = new Uint8Array(FW * FH);
    const buf = new Float32Array(PS2);
    const instances = [];
    for (const c of keep) {
      buf.fill(0);
      for (let m = 0; m < M; m++) {
        const co = c.coeff[m];
        if (Math.abs(co) < 1e-6) continue;
        const base = m * PS2;
        for (let p = 0; p < PS2; p++) buf[p] += co * o1[base + p];
      }
      let filled = 0;
      for (let p = 0; p < PS2; p++) {
        const s = 1 / (1 + Math.exp(-buf[p]));
        if (s <= 0.5) continue;
        filled++;
        const px = (p % PS) * 4, py = ((p / PS) | 0) * 4;
        const u = (px - lb.ox) / lb.nw;
        const v = (py - lb.oy) / lb.nh;
        if (u < 0 || u > 1 || v < 0 || v > 1) continue;
        const gx = Math.max(0, Math.min(FW - 1, Math.round(u * (FW - 1))));
        const gy = Math.max(0, Math.min(FH - 1, Math.round(v * (FH - 1))));
        fg[gy * FW + gx] = 1;
      }
      instances.push({
        label: COCO[c.cls] || ('class' + c.cls),
        score: +c.score.toFixed(3),
        // 归一化到原图坐标，UI 用
        box: [
          +Math.max(0, (c.x - lb.ox) / lb.nw).toFixed(3),
          +Math.max(0, (c.y - lb.oy) / lb.nh).toFixed(3),
          +Math.min(1, c.w / lb.nw).toFixed(3),
          +Math.min(1, c.h / lb.nh).toFixed(3)
        ],
        coverage: +(filled / PS2).toFixed(3)
      });
    }

    const result = {
      instances,
      labels: instances.map((i) => i.label),
      fgMask: { data: fg, w: FW, h: FH },
      ms: Math.round(dt),
      size: { w: lb.iw, h: lb.ih }
    };
    CACHE.set(key, result);
    if (CACHE.size > 4) CACHE.delete(CACHE.keys().next().value);
    return Object.assign({}, result, { cached: false });
  }

  function clearCache() { CACHE.clear(); }

  global.PW = global.PW || {};
  global.PW.objects = {
    init, estimate, clearCache, COCO,
    __internals: { letterbox, iou, contentKey },
    get ready() { return !!S.session; },
    get failed() { return S.failed; }
  };
})(window);
