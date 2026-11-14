/* 语义分割：SegFormer-B0 · ADE20K 150 类（浏览器内 ONNX 推理）
 *
 * 为什么换掉 COCO YOLOv8n-seg：
 *   COCO 80 类里**没有 tree / sky / ground / ceiling / water / mountain**，
 *   草地、雪坡、水面、天花板在几何上无法区分，这正是「树被拉成鳍、天空和地面混在一起」
 *   的根因。ADE20K 恰好覆盖这些户外/室内类别，能给出逐像素的类别图。
 *
 * 模型：Xenova/segformer-b0-finetuned-ade-512-512（ONNX 量化版约 15MB）
 *   输入 pixel_values [1,3,512,512]（ImageNet 归一化）
 *   输出 logits [1,150,128,128]，argmax 得到每个 patch 的类别
 *   下载（镜像）：tools/fetch_models.py
 *
 * 设计原则：**没有模型也要能跑**。
 *   模型缺失 / 推理失败时退化为启发式（深度 + 地平线 + YOLO 标签），
 *   仍然能给出 sky / ground / ceiling 三个关键分组，只是没有 tree/building 细分。
 */
(function (global) {
  'use strict';

  const SELF = (document.currentScript && document.currentScript.src) || location.href;
  const BASE = new URL('../', SELF).href;
  const MODEL = BASE + 'models/segformer_b0_ade20k.onnx';
  const SIZE = 512;

  /* 分组编号：几何策略按组走，不按具体类别走 */
  const GROUP = {
    UNKNOWN: 0, SKY: 1, CEILING: 2, WALL: 3, GROUND: 4, WATER: 5,
    VEG: 6, BUILDING: 7, PERSON: 8, VEHICLE: 9, OBJECT: 10, TERRAIN: 11
  };
  const GROUP_NAMES = ['unknown', 'sky', 'ceiling', 'wall', 'ground', 'water',
    'vegetation', 'building', 'person', 'vehicle', 'object', 'terrain'];

  /* ADE20K / SceneParse150 标准类别表（索引即模型输出通道） */
  const ADE20K = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
    'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door',
    'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water',
    'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
    'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
    'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard',
    'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace',
    'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case',
    'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge',
    'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
    'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer',
    'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel',
    'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth',
    'television receiver', 'airplane', 'dirt track', 'apparel', 'pole',
    'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster',
    'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer',
    'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
    'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step',
    'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle',
    'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
    'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
    'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
    'clock', 'flag'
  ];

  /* 类别 → 分组。只登记对几何有意义的类别，其余保持 UNKNOWN（不特殊处理）。 */
  const CLASS_GROUP = new Uint8Array(150);
  const mark = (group, ids) => { for (const i of ids) CLASS_GROUP[i] = group; };
  mark(GROUP.WALL, [0, 8, 27, 43, 58, 63, 74, 95, 100, 124, 130, 132, 144]);
  mark(GROUP.BUILDING, [1, 25, 48, 84, 79, 86]);
  mark(GROUP.SKY, [2]);
  mark(GROUP.GROUND, [3, 6, 9, 11, 13, 29, 46, 52, 53, 54, 59, 94, 121]);
  mark(GROUP.CEILING, [5]);
  mark(GROUP.VEG, [4, 17, 66, 72]);
  mark(GROUP.TERRAIN, [16, 68, 33, 34, 41, 42, 44, 50]);
  mark(GROUP.WATER, [21, 26, 60, 104, 109, 111, 113, 128]);
  mark(GROUP.PERSON, [12]);
  mark(GROUP.VEHICLE, [20, 76, 80, 82, 90, 102, 103, 116, 127]);
  mark(GROUP.OBJECT, [7, 10, 14, 15, 18, 19, 22, 23, 24, 28, 30, 31, 32,
    35, 36, 37, 38, 39, 40, 45, 47, 49, 51, 55, 56, 57, 61, 62, 64, 65,
    67, 69, 70, 71, 73, 75, 77, 78, 81, 83, 85, 87, 88, 89, 91, 92, 93,
    96, 97, 98, 99, 101, 105, 106, 107, 108, 110, 112, 114, 115, 117, 118,
    119, 120, 122, 123, 125, 126, 129, 131, 133, 134, 135, 136, 137, 138,
    139, 140, 141, 142, 143, 145, 146, 147, 148, 149]);

  const OBJECT_GROUPS = new Uint8Array(12);
  OBJECT_GROUPS[GROUP.VEG] = 1;
  OBJECT_GROUPS[GROUP.BUILDING] = 1;
  OBJECT_GROUPS[GROUP.PERSON] = 1;
  OBJECT_GROUPS[GROUP.VEHICLE] = 1;
  OBJECT_GROUPS[GROUP.OBJECT] = 1;

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
    } catch (e) { fp = String(img.currentSrc || img.src || '').length; }
    return 'sem:' + iw + 'x' + ih + ':' + (fp >>> 0).toString(16);
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
      if (!r.ok) throw new Error('语义模型缺失（' + r.status + '），可运行 tools/fetch_models.py 下载');
      if (onProgress) onProgress(0.4);
      const buf = await r.arrayBuffer();
      if (onProgress) onProgress(0.7);
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

  /* 图片 → (1,3,512,512) ImageNet 归一化。直接拉伸到正方形：
     SegFormer 对纵横比不敏感，而按 uv 直接映射回网格能省掉 padding 反算。 */
  function preprocess(img) {
    const iw = img.naturalWidth || img.width;
    const ih = img.naturalHeight || img.height;
    const cv = document.createElement('canvas');
    cv.width = SIZE; cv.height = SIZE;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, iw, ih, 0, 0, SIZE, SIZE);
    const rgba = ctx.getImageData(0, 0, SIZE, SIZE).data;
    const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
    const n = SIZE * SIZE;
    const data = new Float32Array(n * 3);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      data[i] = (rgba[p] / 255 - mean[0]) / std[0];
      data[n + i] = (rgba[p + 1] / 255 - mean[1]) / std[1];
      data[2 * n + i] = (rgba[p + 2] / 255 - mean[2]) / std[2];
    }
    return { data, iw, ih };
  }

  function argmaxChannels(logits, C, HW) {
    const out = new Uint8Array(HW);
    for (let p = 0; p < HW; p++) {
      let best = -1e9, cls = 0;
      for (let c = 0; c < C; c++) {
        const v = logits[c * HW + p];
        if (v > best) { best = v; cls = c; }
      }
      out[p] = cls;
    }
    return out;
  }

  /* 模型输出 → 低分辨率类别图（默认 128×96，与深度网格同量级） */
  function decode(logits, dims, gw, gh) {
    const C = dims[dims.length - 3] || 150;
    const LH = dims[dims.length - 2] || 128;
    const LW = dims[dims.length - 1] || 128;
    const flat = argmaxChannels(logits, C, LH * LW);
    const classes = new Uint8Array(gw * gh);
    const groups = new Uint8Array(gw * gh);
    for (let j = 0; j < gh; j++) {
      const sy = Math.min(LH - 1, Math.round((j / Math.max(1, gh - 1)) * (LH - 1)));
      for (let i = 0; i < gw; i++) {
        const sx = Math.min(LW - 1, Math.round((i / Math.max(1, gw - 1)) * (LW - 1)));
        const c = flat[sy * LW + sx];
        const k = j * gw + i;
        classes[k] = c;
        groups[k] = CLASS_GROUP[c] || GROUP.UNKNOWN;
      }
    }
    return { classes, groups };
  }

  /* 启发式回退：没有语义模型时，用深度 + 地平线 + YOLO 框给出关键分组。
     只能分 sky / ground / ceiling，无法分 tree / building。 */
  function heuristic(depth, analysis, objInfo, gw, gh) {
    const groups = new Uint8Array(gw * gh);
    const classes = new Uint8Array(gw * gh).fill(255);
    const dw = depth && (depth.width || depth.size) || 1;
    const dh = depth && (depth.height || depth.size) || 1;
    const D = depth && depth.data;
    const horizon = analysis && typeof analysis.horizon === 'number' ? analysis.horizon : 0.5;
    const indoor = horizon < 0.42 || horizon > 0.72;
    for (let j = 0; j < gh; j++) {
      const v = j / Math.max(1, gh - 1);
      for (let i = 0; i < gw; i++) {
        const u = i / Math.max(1, gw - 1);
        let d = 0;
        if (D) {
          const sx = Math.min(dw - 1, Math.round(u * (dw - 1)));
          const sy = Math.min(dh - 1, Math.round(v * (dh - 1)));
          d = D[sy * dw + sx];
        }
        let g;
        if (v < horizon - 0.02 && d < 0.5) g = indoor ? GROUP.CEILING : GROUP.SKY;
        else if (v > horizon + 0.01) g = GROUP.GROUND;
        else g = GROUP.TERRAIN;
        groups[j * gw + i] = g;
      }
    }
    // YOLO 标签（COCO）落到对应分组，至少能把人/车/背包从地形里分出来
    const map = { person: GROUP.PERSON, car: GROUP.VEHICLE, bus: GROUP.VEHICLE, truck: GROUP.VEHICLE,
      bicycle: GROUP.VEHICLE, motorcycle: GROUP.VEHICLE, boat: GROUP.VEHICLE,
      potted_plant: GROUP.VEG, 'potted plant': GROUP.VEG, backpack: GROUP.OBJECT };
    if (objInfo && objInfo.instances) {
      for (const ins of objInfo.instances) {
        const g = map[ins.label];
        if (g === undefined) continue;
        const [bx, by, bw, bh] = ins.box || [0, 0, 0, 0];
        for (let j = 0; j < gh; j++) {
          const v = j / Math.max(1, gh - 1);
          if (v < by || v > by + bh) continue;
          for (let i = 0; i < gw; i++) {
            const u = i / Math.max(1, gw - 1);
            if (u < bx || u > bx + bw) continue;
            groups[j * gw + i] = g;
          }
        }
      }
    }
    return { classes, groups, source: 'heuristic' };
  }

  /**
   * ctx = { img, depth, analysis, objInfo, gw, gh }
   * 始终 resolve，不抛错：模型不可用时返回启发式结果。
   */
  async function estimate(ctx, onProgress) {
    ctx = ctx || {};
    const img = ctx.img;
    const gw = ctx.gw || 128, gh = ctx.gh || 96;
    if (img) {
      const key = contentKey(img) + ':' + gw + 'x' + gh;
      const hit = CACHE.get(key);
      if (hit) return Object.assign({}, hit, { cached: true, ms: 0 });
      /* v1.26 L2：刷新页面后跳过 SegFormer 会话与推理。
         ns 带版本号，模型 / decode 分组表变更必须升 ns，避免命中旧语义图。 */
      const idb = global.PW && global.PW.inferenceCache;
      if (idb) {
        const rec = await idb.get('sem-v1', key);
        if (rec && rec.groups && rec.classes) {
          CACHE.set(key, rec);
          if (CACHE.size > 4) CACHE.delete(CACHE.keys().next().value);
          return Object.assign({}, rec, { cached: true, persisted: true, ms: 0 });
        }
      }
    }
    let result = null;
    if (img) {
      try {
        const session = await init(onProgress);
        const ort = S.ort;
        const pp = preprocess(img);
        const feeds = {};
        feeds[session.inputNames[0]] = new ort.Tensor('float32', pp.data, [1, 3, SIZE, SIZE]);
        const t0 = (global.performance || Date).now();
        const out = await session.run(feeds);
        const o = out[session.outputNames[0]];
        const dt = (global.performance || Date).now() - t0;
        const dec = decode(o.data, o.dims || [1, 150, 128, 128], gw, gh);
        result = {
          classes: dec.classes, groups: dec.groups,
          width: gw, height: gh,
          source: 'segformer', ms: Math.round(dt)
        };
      } catch (e) {
        console.warn('语义模型不可用，使用启发式分组：', e && e.message);
      }
    }
    if (!result) {
      const h = heuristic(ctx.depth, ctx.analysis, ctx.objInfo, gw, gh);
      result = { classes: h.classes, groups: h.groups, width: gw, height: gh, source: h.source, ms: 0 };
    }
    // 统计各组占比，写进 body.dataset 方便排查
    const counts = new Array(12).fill(0);
    for (let i = 0; i < result.groups.length; i++) counts[result.groups[i]]++;
    result.stats = {};
    for (let g = 1; g < 12; g++) result.stats[GROUP_NAMES[g]] = +(counts[g] / result.groups.length).toFixed(3);
    if (img) {
      const key = contentKey(img) + ':' + gw + 'x' + gh;
      CACHE.set(key, result);
      if (CACHE.size > 4) CACHE.delete(CACHE.keys().next().value);
      /* 只持久化模型推理成功的结果：启发式回退依赖 depth / analysis / objInfo，
         参数一变就应该重算，缓存它反而会锁住旧结论。 */
      const idb = global.PW && global.PW.inferenceCache;
      if (idb && result.source === 'segformer') idb.put('sem-v1', key, result);
    }
    return Object.assign({}, result, { cached: false });
  }

  function clearCache() { CACHE.clear(); }

  /* 取某一分组在 uv 处是否成立（world.js 用） */
  function groupAt(sem, u, v) {
    if (!sem || !sem.groups) return 0;
    const x = Math.max(0, Math.min(sem.width - 1, Math.round(u * (sem.width - 1))));
    const y = Math.max(0, Math.min(sem.height - 1, Math.round(v * (sem.height - 1))));
    return sem.groups[y * sem.width + x];
  }

  function isObjectGroup(g) { return !!OBJECT_GROUPS[g]; }

  global.PW = global.PW || {};
  global.PW.semantics = {
    init, estimate, clearCache, groupAt, isObjectGroup,
    GROUP, GROUP_NAMES, ADE20K, CLASS_GROUP, OBJECT_GROUPS,
    __internals: { preprocess, decode, heuristic, contentKey },
    get ready() { return !!S.session; },
    get failed() { return S.failed; }
  };
})(window);
