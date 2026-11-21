/* 应用层：画廊 / 载入 / 主循环 / HUD */
(function () {
  'use strict';

  const { hashString, clamp } = window.PW.util;

  /* 版本号单一来源：index.html 的顶栏徽标与页脚都由这里写入，
     测试脚本读 body[data-version]，避免两处手改不同步。 */
  const VERSION = '4.3.2';
  /* F6（v2.15.12）：`__PW_BUILD__` 与 VERSION 必须一致，否则 index.html 的
     "过期缓存自检"会每次打开都强制整页重载一次并打 data-stale 标记。
     bump_cache.py 现在两者一起改，这条断言用来在控制台第一时间暴露漂移。 */
  if (typeof console !== 'undefined' && console.assert) {
    console.assert(window.__PW_BUILD__ === VERSION,
      '[app] 版本漂移：__PW_BUILD__=' + window.__PW_BUILD__ + ' VERSION=' + VERSION);
  }

  /* v2 服务的地址：显式 ?api= 优先，其次是自动发现（本机常见端口）。
     自动发现写在启动流程里，不需要使用者改 URL —— 这是"更快捷地启用 v2"的关键。 */
  let apiBase = '';
  /* 自动发现的 Promise：?w= / ?img= 会在 DOMContentLoaded 后 120ms 就进世界，
     而自动探测要几百毫秒。早先没有这个等待，结果是"服务明明在，却走了在线链路"
     —— 表现为徽标写着 v2 就绪、画面却是 518² 的观感。 */
  let capsReady = null;

  const WORLDS = [
    { id: 'alpine', name: '晨雾湖岸', sub: '高山湖泊 · 破晓', src: 'worlds/alpine.jpg', thumb: 'worlds/alpine_thumb.jpg', badge: '晨雾' },
    { id: 'desert', name: '赤岩峡谷', sub: '犹他荒漠 · 黄金时刻', src: 'worlds/desert.jpg', thumb: 'worlds/desert_thumb.jpg', badge: '峡谷' },
    { id: 'coast', name: '风暴海岸', sub: '爱尔兰海崖 · 阴雨', src: 'worlds/coast.jpg', thumb: 'worlds/coast_thumb.jpg', badge: '海崖' },
    { id: 'winter', name: '蓝调雪林', sub: '针叶林 · 蓝调时刻', src: 'worlds/winter.jpg', thumb: 'worlds/winter_thumb.jpg', badge: '雪林' }
  ];

  const el = (id) => document.getElementById(id);
  const canvas = el('gl');
  const landing = el('landing');
  const stage = el('stage');
  const loader = el('loader');
  const pause = el('pause');

  let renderer = null;
  let world = null;
  let player = null;
  let current = null;
  let objectUrls = [];
  /* F12（v2.15.12）：进入世界的**加载令牌**。快速连点两张卡片（或画廊 + 上传）会
     交叠跑两个 enterWorld/enterAsset，它们在多个 await 处交错，后者会 dispose 掉
     前者刚建好的 player、world 字段互相覆盖、进度条互踩。
     每个 await 之后比对令牌，过期就静默退出。 */
  let loadToken = 0;
  let running = false;
  let lastT = 0;
  let time = 0;
  let fade = 0;
  let fadeStart = 0;
  let fadeFrames = 0;
  let expoCount = 0;
  let fpsAcc = 0, fpsN = 0;            // F21：HUD 帧率统计
  let frameIdx = 0;
  let instant = false;
  let intro = null;
  let introPhotoUrl = null;
  const ambience = window.PW.createAmbience();

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  /* ---------------- 开屏字标 ---------------- */
  function initIntro() {
    const cv = el('introCanvas');
    if (!cv || !window.PW.createIntro) return;
    intro = window.PW.createIntro(cv, el('introBg'));
    setIntroPhoto(WORLDS[0].thumb);
    if (!instant) intro.autoPlay();
  }

  function setIntroPhoto(src) {
    if (!intro || !src || src === introPhotoUrl) return;
    introPhotoUrl = src;
    const img = new Image();
    img.onload = () => intro.setPhoto(img);
    img.src = src;
  }

  /* ---------------- 画廊 ---------------- */
  function buildGallery() {
    const g = el('gallery');
    g.innerHTML = '';
    WORLDS.forEach((w) => {
      const card = document.createElement('div');
      card.className = 'card';
      card.dataset.id = w.id;
      card.innerHTML =
        `<img src="${w.thumb}" alt="${w.name}" loading="lazy" draggable="false">` +
        `<div class="badge">${w.badge}</div>` +
        `<div class="cap"><b>${w.name}</b><span>${w.sub}</span></div>`;
      card.addEventListener('mouseenter', () => setIntroPhoto(w.thumb));
      card.addEventListener('click', () => startWorld(w, card));
      g.appendChild(card);
    });

    const up = document.createElement('div');
    up.className = 'card upload';
    up.innerHTML =
      `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round">` +
      `<path d="M12 5v14M5 12h14"/></svg>` +
      `<div class="t">用你的照片</div>` +
      `<div class="s">横向风景照效果最好</div>`;
    up.addEventListener('click', () => el('fileInput').click());
    g.appendChild(up);
  }

  /* ---------------- 载入 ---------------- */
  let lastProgress = 0;
  function setProgress(p, text) {
    p = Math.max(p, lastProgress);   // 单调递增：避免阶段回退造成"卡住"的错觉
    lastProgress = p;
    el('loaderBar').style.width = p + '%';
    if (text) el('loaderSteps').textContent = text;
  }

  function loadImage(src) {
    return new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res(img);
      img.onerror = () => rej(new Error('图片读取失败'));
      img.src = src;
    });
  }

  function ensureRenderer() {
    if (renderer) return renderer;
    renderer = window.PW.createRenderer(canvas);
    return renderer;
  }

  /* 点卡片：先播开屏动画（字标按该照片的地形立起来），中途开始重建 */
  /* F7（v2.15.12）：world 级 GL 资源回收。
     以前每次换图/换世界都只是把 `world = w` 覆盖掉，旧 world 的几十~几百个 buffer
     与 5~7 张纹理（2048² 照片纹理 + 三份点云）永远留在 GPU 上 —— 全仓库没有任何
     gl.deleteBuffer / deleteTexture（只有 extend.js 管自己的 tile）。
     连续切换两三次就能把移动端 512MB 配额打穿，表现是 webglcontextlost、整屏黑。
     这里在替换旧 world 之前调用；任何一层出问题都不阻断新世界进入。 */
  /* F20（v2.15.12）：上传照片的 blob URL 只增不减 —— 每上传一张就永久占住整份
     原始文件。切走/换图时把**不再被 current 引用**的都 revoke 掉（当前世界的
     src/thumb 还要用来重进、显示缩略图，不能回收）。 */
  function pruneObjectUrls() {
    if (!objectUrls.length) return;
    const keep = new Set();
    if (current) { if (current.src) keep.add(current.src); if (current.thumb) keep.add(current.thumb); }
    const left = [];
    for (const u of objectUrls) {
      if (keep.has(u)) { left.push(u); continue; }
      try { URL.revokeObjectURL(u); } catch (e) { /* ignore */ }
    }
    objectUrls = left;
  }

  function disposeWorld(w) {
    if (!w || !renderer || !renderer.gl) return;
    const gl = renderer.gl;
    const killMesh = (mesh) => {
      if (!mesh) return;
      try {
        for (const k of ['aPos', 'aNormal', 'aUV', 'aColor', 'aEdge', 'aSize', 'aCount']) {
          if (mesh[k] && mesh[k].buf) gl.deleteBuffer(mesh[k].buf);
        }
        if (mesh.index && mesh.index.buf) gl.deleteBuffer(mesh.index.buf);
      } catch (e) { /* 单个 mesh 出问题不影响其他 */ }
    };
    try {
      for (const key in (w.meshes || {})) killMesh(w.meshes[key]);
      for (const it of (w.extraMeshes || [])) killMesh(it && it.mesh);
      killMesh(w.pointsMesh);
      killMesh(w.points2Mesh);
      if (w.skyBg && w.skyBg.shell) killMesh(w.skyBg.shell);
      if (w.extender && typeof w.extender.dispose === 'function') w.extender.dispose();
      for (const k in (w.tex || {})) {
        if (w.tex[k]) gl.deleteTexture(w.tex[k]);
      }
    } catch (e) {
      console.warn('[app] 旧世界资源释放失败（不影响新世界）：', e && e.message);
    }
  }

  function startWorld(def, card) {
    if (intro && !instant) {
      const img = card && card.querySelector('img');
      if (img && img.complete) intro.setPhoto(img);
      let started = false;
      const go = () => { if (!started) { started = true; enterWorld(def); } };
      intro.play(go).then(go);
      setTimeout(go, 2200);
    } else {
      enterWorld(def);
    }
  }

  /* 深度图可视化：让"照片 → 深度 → 三维"的过程看得见 */
  function drawDepthPreview(d) {
    const cv = el('loaderDepth');
    if (!cv || !d || !d.data) return;
    const ctx = cv.getContext('2d');
    const DW = d.width || d.size, DH = d.height || d.size;
    const img = ctx.createImageData(cv.width, cv.height);
    for (let y = 0; y < cv.height; y++) {
      for (let x = 0; x < cv.width; x++) {
        const sx = Math.floor(x / cv.width * DW), sy = Math.floor(y / cv.height * DH);
        const v = Math.max(0, Math.min(1, d.data[sy * DW + sx]));
        const g = Math.round(v * 255);
        const o = (y * cv.width + x) * 4;
        img.data[o] = g * 0.55 + 40; img.data[o + 1] = g * 0.85; img.data[o + 2] = g;
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    cv.style.opacity = '1';
    const lbl = el('loaderDepthLabel');
    if (lbl) lbl.textContent = '深度图（亮=近 / 暗=远）';
  }

  /* v2 链路：把原图 POST 给本地 GPU 服务，拿回分层 GLB 直接进资产模式。
   *
   * 为什么值得加这一跳：浏览器内 518² 小模型 + 一张连续薄片的老链路，在实测里
   * 会让山体变成幕布、近景塌成地毯（见 REVIEW.md / docs/AB_v2.md）。
   * 5070 上跑 ViT-L 高分辨率深度 + LDI 建网格，一张图冷启动 ~14s、命中缓存 ~0s，
   * 产物是 10MB 左右的 GLB —— 观感是照片级的。
   *
   * 触发条件：URL 带 ?api=http://127.0.0.1:8100（与地貌延伸共用同一个服务地址），
   * 且没有显式 ?v2=0。服务不在或失败时**静默回退**在线链路，不影响原有用法。 */
  async function tryReconstructV2(def) {
    const q = new URLSearchParams(location.search);
    if (q.get('v2') === '0' || !def || !def.src) return false;
    if (!apiBase && capsReady) {
      // 最多等 4s：宁可慢一点，也不要因为"探测还没回来"就降级到在线链路
      await Promise.race([capsReady, sleep(4000)]);
    }
    if (!apiBase) return false;
    const base = apiBase.replace(/\/+$/, '');
    try {
      loader.classList.remove('hidden', 'done', 'burst');
      el('loaderTitle').textContent = '正在用 GPU 重建场景';
      el('loaderDepthLabel').textContent = '';
      setProgress(6, '上传照片到本地重建服务…');
      const srcResp = await fetch(def.src, { cache: 'no-store' });
      if (!srcResp.ok) return false;
      const photo = await srcResp.blob();
      const fd = new FormData();
      fd.append('image', photo, 'photo.jpg');
      setProgress(18, 'GPU 深度估计 + LDI 建网格（首次约 15s）…');
      const t0 = performance.now();
      const resp = await fetch(base + '/api/reconstruct', { method: 'POST', body: fd });
      if (!resp.ok) {
        console.warn('[v2] 重建服务不可用', resp.status, await resp.text().catch(() => ''));
        return false;
      }
      const glb = await resp.blob();
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      setProgress(88, '重建完成（' + secs + 's），装载世界…');
      const url = URL.createObjectURL(glb);
      try {
        await enterAsset(url);
      } finally {
        /* F20（v2.15.12）：GLB 的 blob URL 用完必须 revoke —— enterAsset 已经把
           字节读进 GL 与内存，URL 再留着只是占着整份 GLB（几 MB~几十 MB）。 */
        URL.revokeObjectURL(url);
      }
      setHudPipe('v2', 'V2');
      return true;
    } catch (e) {
      console.warn('[v2] 失败，回退在线链路', e);
      return false;
    }
  }

  async function enterWorld(def) {
    const token = ++loadToken;
    if (await tryReconstructV2(def)) return;
    if (token !== loadToken) return;             // 期间又点了别的卡片
    if (running) stopLoop();
    loader.classList.remove('hidden', 'done', 'burst');
    pause.classList.add('hidden');
    el('loaderImg').src = def.thumb || def.src;
    el('loaderTitle').textContent = '正在重建场景';
    lastProgress = 0;
    el('loaderDepthLabel').textContent = '';
    el('loaderDepth').style.opacity = '0';
    setProgress(3, '读取照片…');

    try {
      const img = await loadImage(def.src);
      if (token !== loadToken) return;
      setProgress(8, '照片解码完成');
      await sleep(80);
      setProgress(12, '分析天空 / 地平线…');
      await sleep(60);
      // 照片贴图分辨率 A/B：?tex=0 回到旧的 420px 分析画布；?tex=4096 抬高上限。
      // 不传时走 DEFAULTS.displayMaxEdge（2048），仍受 GL 的 MAX_TEXTURE_SIZE 保护。
      const texArg = new URLSearchParams(location.search).get('tex');
      // 只接受纯十进制整数：parseInt('1e3')=1、parseInt('abc')=NaN||0 会把
      // 非法输入静默变成 1×1 贴图或 420px 旧画布，质检还会误判为链路跑通。
      // 非法值一律走默认（DEFAULTS.displayMaxEdge），只有 ?tex=0 才关闭高清链路。
      const texOpts = (texArg === null || !/^\d+$/.test(texArg)) ? undefined : {
        displayMaxEdge: Math.min(8192, parseInt(texArg, 10))
      };
      const analysis = window.PW.reconstruct.analyze(img, def.name, texOpts);
      // 运行时可验证：质检探针读 body.dataset.tex* 即可确认贴图链路与分辨率
      if (document.body) {
        document.body.dataset.texW = String(analysis.photoCanvas.width);
        document.body.dataset.texH = String(analysis.photoCanvas.height);
      }
      setProgress(20, '提取天际线与太阳方位');
      await sleep(60);
      setProgress(24, '采样雾色 / 地面调色板');
      await sleep(50);
      setProgress(28, '初始化重建器…');
      await sleep(40);

      ensureRenderer();

      // 用预训练深度模型把照片还原成三维表面（失败则回退到分层方案）
      let depthMap = null;
      try {
        setProgress(32, '加载深度模型（27MB，仅首次）…');
        /* v1.26：不再无条件 await init()。
           estimate() 内部只在 L1/L2 都未命中时才调 init(onProgress)；
           旧写法即使深度图命中 IndexedDB，也照样下载 27MB 模型 + 编译 WASM 内核，
           把持久化的收益吃掉一大半。进度回调交给 estimate 传递，冷启动文案不变。 */
        const onDepthProgress = (p, label) => {
          setProgress(32 + p * 32, label || ('下载深度模型 ' + Math.round(p * 100) + '%（27MB）'));
        };
        depthMap = await Promise.race([
          window.PW.depth.estimate(img, onDepthProgress),
          new Promise((_, rej) => setTimeout(() => rej(new Error('深度估计超时')), 60000))
        ]);
        setProgress(72, (depthMap && (depthMap.cached || depthMap.persisted))
          ? (depthMap.persisted ? '深度图命中本地缓存（跳过推理）' : '深度图命中内存缓存')
          : '深度图后处理（分位归一化 / 极性）');
        drawDepthPreview(depthMap);
      } catch (e) {
        console.warn('深度模型不可用，回退到天际线方案：', e && e.message);
      }

      /* 近景物体识别（YOLOv8n-seg，COCO 80 类）：给「近处物体不要延伸到远处」
         提供语义边界。失败不影响主流程，回退到纯深度分层。 */
      let objInfo = null;
      if (depthMap && window.PW.objects) {
        try {
          setProgress(73, '识别近景物体（YOLOv8n-seg · COCO 80 类）…');
          objInfo = await Promise.race([
            window.PW.objects.estimate(img, (p) => {
              if (p < 1) setProgress(73 + p, '下载识别模型 ' + Math.round(p * 100) + '%（14MB）');
            }),
            new Promise((_, rej) => setTimeout(() => rej(new Error('物体识别超时')), 25000))
          ]);
          setProgress(74, objInfo.labels.length
            ? ('识别到：' + objInfo.labels.slice(0, 4).join(' / '))
            : '未识别到显著近景物体');
        } catch (e) {
          console.warn('物体识别不可用，回退纯几何分层：', e && e.message);
        }
      }

      /* 语义分组（SegFormer ADE20K 150 类）：把 sky / ground / ceiling / tree /
         building / person / vehicle 分开。没有模型时自动退化为启发式，不阻塞主流程。 */
      let sem = null;
      // ?sem=0 关闭语义分层，回到 v1.15 的纯深度路径（A/B 用）
      if (depthMap && window.PW.semantics &&
          new URLSearchParams(location.search).get('sem') !== '0') {
        try {
          setProgress(74, '语义分组：天空 / 地面 / 天花板 / 植被 / 建筑…');
          sem = await Promise.race([
            window.PW.semantics.estimate({ img, depth: depthMap, analysis, objInfo }),
            new Promise((_, rej) => setTimeout(() => rej(new Error('语义推理超时')), 20000))
          ]);
          setProgress(74, '语义分组完成（' + sem.source + '）');
        } catch (e) {
          console.warn('语义分组不可用，退化为纯深度分层：', e && e.message);
        }
      }

      setProgress(75, depthMap ? '反投影：像素 → 世界坐标…' : '构建分层地形…');
      await sleep(40);
      const params = new URLSearchParams(location.search);
      /* 「第三种方案」网页版（简化 LDI，?migan=1）：
         MI-GAN 把照片放在中心、补出画面外一圈，作为世界背景幕。
         原版 3d-photo-inpainting 是 PyTorch 三模型 + MiDaS（2~3 分钟/图），
         浏览器进不来；MI-GAN 29MB、512²、CPU 0.57s（实测）可以。 */
      if (params.get('migan') === '1' && window.PW.inpaint) {
        try {
          setProgress(75.5, 'AI 补全画面外（MI-GAN 29MB，仅首次）…');
          const bg = await Promise.race([
            window.PW.inpaint.expand(img, { pad: 0.18 }, (p) => {
              if (p < 1) setProgress(75.5, '下载补全模型 ' + Math.round(p * 100) + '%（29MB）');
            }),
            new Promise((_, rej) => setTimeout(() => rej(new Error('补全超时')), 45000))
          ]);
          analysis.bgCanvas = bg;
          setProgress(76, '补全完成：背景使用 AI 外扩图');
        } catch (e) {
          console.warn('MI-GAN 补全不可用：', e && e.message);
        }
      }
      /* 简化 LDI 的第二半（?ldi=2）：把前景物体擦掉，让 MI-GAN 补出
         「它背后的背景」，再用 buildHolePoints 垫一层稍远的点。
         侧移时前景移开，背后有真实补全内容。 */
      if (params.get('ldi') === '2' && objInfo && objInfo.fgMask && window.PW.inpaint) {
        try {
          setProgress(76, 'AI 补前景背后（LDI，MI-GAN）…');
          const mc = document.createElement('canvas');
          mc.width = objInfo.fgMask.w;
          mc.height = objInfo.fgMask.h;
          const mc2 = mc.getContext('2d', { willReadFrequently: true });
          mc2.fillStyle = '#ffffff';            // 白 = 保留
          mc2.fillRect(0, 0, mc.width, mc.height);
          const md = mc2.getImageData(0, 0, mc.width, mc.height);
          for (let q = 0; q < objInfo.fgMask.w * objInfo.fgMask.h; q++) {
            if (objInfo.fgMask.data[q]) {       // 前景 = 黑 = 要补
              md.data[q * 4] = 0; md.data[q * 4 + 1] = 0; md.data[q * 4 + 2] = 0;
            }
          }
          mc2.putImageData(md, 0, 0);
          analysis.bgPhotoCanvas = await Promise.race([
            window.PW.inpaint.fill(img, mc),
            new Promise((_, rej) => setTimeout(() => rej(new Error('LDI 补全超时')), 45000))
          ]);
          setProgress(77, 'LDI 补全完成：前景背后已补齐');
        } catch (e) {
          console.warn('LDI 补全不可用：', e && e.message);
        }
      }
      // 竖直夸张：?yg= 与 ?ygain= 是同一参数的历史别名。旧代码在 opts 字面量里
      // 写了两次 yGain 键，后者覆盖前者，?yg= 永久失效；这里统一成一处解析。
      const yGainRaw = params.has('ygain') ? params.get('ygain') : params.get('yg');
      const yGainVal = (yGainRaw === null) ? undefined : parseFloat(yGainRaw);
      const yGain = Number.isFinite(yGainVal)
        ? Math.max(0.6, Math.min(4, yGainVal || 1.35)) : undefined;
      const w = window.PW.world.build(renderer.gl, analysis, {
        seed: hashString(def.id + '|' + def.name),
        depth: depthMap,
        // 视锥外的程序化地平雾带是实验特性，默认关闭，用 ?ring=1 打开
        ring: params.get('ring') === '1',
        // 重建 A/B：?v1=1 旧经验曲线、?flat=1 关闭近景挤出、?vg=1.35 调竖直夸张
        legacy: params.get('v1') === '1',
        flat: params.get('flat') === '1',
        cartesian: params.get('cart') !== '0', // ?cart=0 回到旧的视锥/球面参数化（A/B）
        ldi: params.get('ldi') === '1',      // ?ldi=1 实验：断崖跳面 + 背景补洞（A/B 未达标，默认关）
        proxy: params.get('proxy') === '1', // ?proxy=1 实验：360° 低频世界幕（默认关）
        vg: params.has('vg') ? parseFloat(params.get('vg')) : undefined,
        yGain,
        fgMask: objInfo && objInfo.fgMask,
        // v1.24.7：点云/surfel 渲染**默认开**。A/B 结论：自然风景（峡谷/雪山/
        // 湖岸）明显优于三角网格（无竖条幕、无薄片、地面有照片颗粒），
        // 建筑/玻璃这类深度失效的题材仍然差。?splat=0 回到网格。
        splat: params.get('splat') !== '0',
        splatScale: params.has('splats') ? parseFloat(params.get('splats')) : undefined,
        // 语义分组：天空/地面/天花板/植被/建筑/人车，决定几何策略与单体卡片
        sem: sem,
        // 质检：?dbgpos=1 时把 relief 顶点位置暴露到 world.dbgPos（排查竖条/鳍）
        dbgPos: params.get('dbgpos') === '1',
        // 重建表面背面剔除：默认开（v1.17），?cull=0 回到旧的双面渲染做 A/B
        noCull: params.get('cull') === '0',
        // 地面远处渐隐到地面基色（压极坐标拉丝）：默认开，?gblend=0 做 A/B
        groundFade: params.get('gblend') !== '0',
        // 近场平面半径：?near=12 放宽，?near=0 关闭（A/B）
        nearPlane: params.has('near') ? Math.max(0, Math.min(40, parseFloat(params.get('near')) || 0)) : undefined,
        // 调试：?nocull=wall,steep,jump,torn 逐条关掉剔除，定位"洞的来源"
        noCullReasons: params.get('nocull') || '',
        /* 逐列轮廓线补天（默认开）：轮廓线以下的像素不因深度被判成天空，
           修掉低纹理崖面上那一片白色方块。?skyfill=0 回到纯深度判定
           （真值基准高 1.05dB，但真实照片会有洞）。 */
        skyFill: params.get('skyfill') !== '0',
        /* 平坦地面顶点法线早退（v1.26，默认开，严格数学等价）：
           ?fastnorm=0 关掉逐元素 A/B。 */
        fastNorm: params.get('fastnorm') !== '0',
        // 竖直夸张（只作用于地平线以上）：?ygain= / ?yg= 做 A/B
        // 注意：这里**不能再写一次 yGain 键** —— 对象字面量重复键会静默覆盖，
        // 旧代码就是这么让 ?yg= 永久失效的（见上方 yGainRaw 的解析）。
        /* 深度后处理（v2.15.6）：遮挡边界重切分 + 双层窄带（见 js/depthpost.js）。
           **默认只在网格路径开启**（?splat=0）：它的机理是消除"混合深度连成的斜坡片"，
           而斜坡片只有三角形连接才会形成；点云不连接，实测指标与观感都中性。
           ?dpost=1 强制开（点云路径也能看第二层点），?dpost=0 强制关。 */
        noDepthPost: params.get('dpost') === '0'
          || (params.get('splat') !== '0' && params.get('dpost') !== '1'),
        dpostThin: params.has('dpostthin') ? parseFloat(params.get('dpostthin')) : undefined,
        dpostK: params.has('dpostk') ? parseFloat(params.get('dpostk')) : undefined,
        dpostSpan: params.has('dpostspan') ? parseFloat(params.get('dpostspan')) : undefined,
        // 照片投影 UV 地面层（默认关，?groundphoto=1 打开；掠射角会拉丝）
        // v1.24.6：照片 UV 地面**默认关**。v1.24.5 的 desert A/B 里细节更好，
        // 但 32 视角语料质检暴露了代价：雪地/水面/建筑这类低纹理题材，
        // 照片 UV 在掠射角被拉成流线（edd4e65e 雪地 lowVoid 0.41）。
        // ?groundphoto=1 可按题材手动开。
        // v1.26.5：照片 UV 地面恢复默认开。v1.24.6 关掉它是怕雪地拉丝，
        // 但关掉后地面只剩程序化色板（24 张批量里最差几张的根因）。
        // 现在用 mipBias 2.2 取低频照片色，兼顾色调与抗拉丝。
        groundPhoto: params.get('groundphoto') !== '0',
        // 地面拓扑：默认世界坐标方格网；?gfan=1 回到旧的极坐标扇形地面（A/B）
        groundGrid: params.get('gfan') !== '1',
        // 地面渐隐区间可调（A/B 用）：?gfade=起始m,结束m,强度
        gfade: params.get('gfade') || undefined,
        /* 壳层地面像素按距离交棒给平铺网格（v2.15.18）：
           地面顶点的 UV 是照片 UV，沿径向被拉成放射状拉丝（18 张语料每张都有）。
           11m 内保留照片细节，30m 外完全交给世界坐标平铺网格。
           `?galpha=0` 关掉做 A/B。 */
        gndAlpha: (params.get('galpha') === '0' ? false : (params.get('galpha') || undefined)),
        /* 地面点云改用世界坐标平铺 UV（v2.15.18，消放射状拉丝）：默认开，
           `?gpts=0` 回到照片 UV 做 A/B。 */
        groundPtsTile: params.get('gpts') === '1',
        /* 断口补片（v2.15.20）：被剔除的四边形压扁后填回洞里，默认开。
           `?fill=0` 回到"直接丢掉"的旧行为做 A/B。 */
        gapFill: params.get('fill') === '1',
        // 真实分阶段回调：中值 / 反投影 / 法线 / 索引 / 高度场各推进一格
        onProgress: (p) => setProgress(75 + p * 12,
          p < 0.35 ? '深度去噪：5×5 中值…'
            : p < 0.60 ? '反投影 + 地面单调化…'
              : p < 0.85 ? '法线 / 光照烘焙…'
                : '索引裁剪 + 行走高度场…')
      });
      setProgress(88, '铺设地表与植被…');
      await sleep(50);

      // 视觉闭环：tools/quality_gate.py 实测首帧中央区域比原片暗约 19%
      // (luma_ratio ≈ 0.81)。?expo= 用来对 targetLum 做对齐 A/B；
      // 探针口径是全屏、质量门口径是中央 ROI，所以补偿系数要实测而非拍脑袋。
      const expoAlign = parseFloat(params.get('expo'));
      if (Number.isFinite(expoAlign) && expoAlign > 0.5 && expoAlign < 2) {
        w.targetLum = clamp(w.targetLum * expoAlign, 0.05, 0.95);
      }
      w.pointsMesh = renderer.makePointMesh(w.particles);
      setProgress(94, '校准空气与雾…');
      await sleep(60);

      if (token !== loadToken) return;           // 期间又点了别的卡片：丢弃这次构建
      if (player) player.dispose();
      disposeWorld(world);                       // F7：先回收旧 world 的 GL 资源
      world = w;
      current = def;
      pruneObjectUrls();
      player = window.PW.createPlayer(canvas, world);
      if (window.__spawn) player.reset(window.__spawn.x, window.__spawn.z, window.__spawn.yaw);
      else if (def.start) player.reset(def.start.x || 0, def.start.z || 0, def.start.yaw || 0);
      if (window.__spawn && typeof window.__spawn.pitch === 'number') player.state.pitch = window.__spawn.pitch;

      /* 扇环外地貌延伸（v1.16）：玩家前进时按 tile 依已有地貌生成。
         ?ext=0 关闭；?api=http://127.0.0.1:8100 指向本地 GPU 生成服务（否则纯程序化）。 */
      if (params.get('ext') !== '0' && window.PW.createExtender) {
        try {
          const ext = window.PW.createExtender({
            gl: renderer.gl, world,
            seed: hashString(def.id + '|extend'),
            server: params.get('api') || '',
            radius: params.has('extr') ? Math.max(2, Math.min(8, parseInt(params.get('extr'), 10) || 5)) : 5,
            // 实时变化：视锥驱动生成 + 主题/记忆参数（v1.18）
            live: liveState.on, chaos: liveState.chaos, memory: liveState.memory
          });
          world.extender = ext;
          /* 重要：extender 接管 heightAt 之后必须把玩家重新落地一次。
             出生点可能在矩形外（?x=0&z=49 这类），玩家 reset 时 heightAt 还是
             照片高度场（矩形外恒 0），接管后真实地形可能有十几米 ——
             不重落就会"站在地下"，看到的是被背面剔除后的空洞。 */
          if (player) {
            // reset() 无条件把 pitch 置为 -0.02；先存后恢复，否则 ?pitch=
            // 指定的出生俯仰角会在 extender 接管 heightAt 后的重新落地时被抹掉。
            const spawnPitch = player.state.pitch;
            player.reset(player.state.x, player.state.z, player.state.yaw);
            player.state.pitch = spawnPitch;
          }
          ext.update(player.state.x, player.state.z, undefined,
            { yaw: player.state.yaw, fov: fovRad() });
        } catch (e) {
          console.warn('地貌延伸初始化失败：', e && e.message);
        }
      }

      /* 单体真网格升级（可选）：?obj3d=1 且 ?api= 指向本地 GPU 服务时，
         把近景显著单体的裁片交给 InstantMesh/TripoSR，替换账单牌。 */
      if (params.get('obj3d') === '1' && params.get('api') && window.PW.object3d) {
        window.PW.object3d.upgrade(world, img, {
          server: params.get('api'), gl: renderer.gl, max: 4
        }).then((r) => {
          if (r && r.upgraded) console.log('单体真网格已替换 ' + r.upgraded + ' 个');
        }).catch((e) => console.warn('单体补全跳过：', e && e.message));
      }

      // 入景前先做一次曝光收敛：避免进入后画面亮度跳变
      try {
        const vf = fovRad();
        let lastMeas = -1;
        for (let i = 0; i < 10; i++) {
          const m = renderer.render(world, {
            eye: [player.state.x, player.state.y, player.state.z],
            yaw: player.state.yaw,
            pitch: player.state.pitch,
            fov: vf
          }, { time: 0, fade: 1, measure: true });
          if (!(m > 0)) break;
          lastMeas = m;
          const ratio = (world.targetLum || 0.35) / Math.max(m, 0.02);
          if (Math.abs(ratio - 1) < 0.02) break;
          world.exposure = clamp(world.exposure * Math.pow(ratio, 0.7), 0.08, 3.6);
        }
        if (instant) document.title = 'PW meas=' + lastMeas.toFixed(3) + ' target=' + world.targetLum.toFixed(3) + ' expo=' + world.exposure.toFixed(3);
      } catch (e) { if (instant) document.title = 'PW err ' + e.message; }

      // 曝光预热只保留一份：这里曾有两份逐字相同的 10 次循环，
      // 第二份在第一份已收敛后又白跑最多 10 帧全屏渲染（首屏进入白白变慢）。
      el('hudName').textContent = def.name;
      el('hudSub').textContent = (def.sub || 'photoworld') + ' · v' + VERSION;
      // 在线链路（浏览器内 518² 小模型）与 v2 观感差一个代际，必须标出来
      setHudPipe('online', '在线');
      el('keysHint').classList.remove('fade');
      setTimeout(() => el('keysHint').classList.add('fade'), 5200);

      setProgress(100, '进入世界');
      await sleep(140);

      landing.classList.add('leaving');
      setTimeout(() => { landing.style.display = 'none'; }, 820);
      stage.classList.add('visible');
      stage.setAttribute('aria-hidden', 'false');

      loader.classList.add('burst');
      await sleep(160);
      loader.classList.add('done');
      setTimeout(() => loader.classList.add('hidden'), 900);

      fadeStart = performance.now();
      fadeFrames = 0;
      expoCount = 0;
      frameIdx = 0;
      fade = instant ? 1 : 0.02;
      lastT = performance.now();
      startLoop();
      if (document.body) {
        document.body.dataset.ready = '1';
        document.body.dataset.mode = world.mode;
        document.body.dataset.pipe = 'v2';
        document.body.dataset.expo = (world.exposure || 0).toFixed(3);
        if (objInfo) document.body.dataset.objects = objInfo.labels.slice(0, 8).join(',');
        if (sem) {
          document.body.dataset.sem = sem.source;
          document.body.dataset.semstats = JSON.stringify(sem.stats || {});
        }
        if (world.reliefMeta) document.body.dataset.dbg = JSON.stringify(world.reliefMeta);
        document.body.dataset.horizon = (world.analysis.horizon || 0).toFixed(3);
        document.body.dataset.fog = world.fogColor.map((v) => v.toFixed(2)).join(',');
      }
    } catch (err) {
      console.error(err);
      document.title = 'ERR ' + (err && err.message ? err.message : err);
      if (document.body) document.body.dataset.err = String(err && err.message ? err.message : err).slice(0, 200);
      el('loaderTitle').textContent = '重建失败';
      setProgress(100, String(err.message || err));
      setTimeout(() => loader.classList.add('done'), 1400);
    }
  }

  function backToGallery() {
    stopLoop();
    if (document.pointerLockElement) document.exitPointerLock();
    pause.classList.add('hidden');
    if (player) player.enabled = true;
    stage.classList.remove('visible');
    stage.setAttribute('aria-hidden', 'true');
    landing.style.display = 'flex';
    requestAnimationFrame(() => landing.classList.remove('leaving'));
    if (intro && !instant) { intro.setPhoto(null); intro.autoPlay(); }
  }

  /* ---------------- 资产模式（离线管线产物） ---------------- */
  function colorsFromImage(img) {
    const c = document.createElement('canvas');
    c.width = 64; c.height = 64;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(img, 0, 0, 64, 64);
    let top = [0, 0, 0], bot = [0, 0, 0], all = 0;
    try {
      const d = g.getImageData(0, 0, 64, 64).data;
      const acc = (y0, y1) => {
        const s = [0, 0, 0]; let n = 0;
        for (let y = y0; y < y1; y++) for (let x = 0; x < 64; x++) {
          const i = (y * 64 + x) * 4;
          s[0] += d[i]; s[1] += d[i + 1]; s[2] += d[i + 2]; n++;
        }
        return s.map((v) => v / n / 255);
      };
      top = acc(0, 14); bot = acc(50, 64);
      for (let i = 0; i < d.length; i += 4) {
        all += (0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]) / 255;
      }
      all /= (d.length / 4);
    } catch (e) { /* 跨域时用默认色 */ }
    return { top, bot, lum: all || 0.45 };
  }

  /* 1×1 白纹理：顶点色资产（无贴图 GLB）用它占位，材质分支走 vC.rgb */
  function whiteTex(gl) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([255, 255, 255, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  /* 多层合并：HunyuanWorld 会吐出 backdrop + 若干前景层，渲染器只认「近景 / 远景」两条通道，
   * 所以这里把同名属性逐层拼接（索引加偏移），让任意层数都能进同一条渲染路径。
   * 一律返回 TypedArray —— WebGL 的 bufferData 不接受普通 Array。 */
  function mergeLayers(list) {
    const anyN = list.some((l) => l.nrm && l.nrm.length);
    const anyU = list.some((l) => l.uv && l.uv.length);
    const anyC = list.some((l) => l.color && l.color.length);
    /* F13（v2.15.12）：GLB 里可能有没有 indices 的 primitive（index:null）。
       旧写法 `ni += l.index.length` 直接 TypeError → "资产加载失败"。
       这里给无索引层补线性索引（0..n-1），保持"一律索引绘制"的单一路径。 */
    let nv = 0, ni = 0;
    for (const l of list) {
      nv += l.pos.length / 3;
      ni += l.index && l.index.length ? l.index.length : (l.pos.length / 3);
    }
    const pos = new Float32Array(nv * 3);
    const nrm = anyN ? new Float32Array(nv * 3) : null;
    const uv = anyU ? new Float32Array(nv * 2) : null;
    const color = anyC ? new Float32Array(nv * 4) : null;
    const index = new Uint32Array(ni);
    let vo = 0, io = 0;
    for (const l of list) {
      const n = l.pos.length / 3;
      pos.set(l.pos, vo * 3);
      if (nrm) {
        if (l.nrm && l.nrm.length) nrm.set(l.nrm, vo * 3);
        else for (let i = 0; i < n; i++) nrm[(vo + i) * 3 + 1] = 1;
      }
      if (uv && l.uv && l.uv.length) uv.set(l.uv, vo * 2);
      if (color) {
        if (l.color && l.color.length) color.set(l.color, vo * 4);
        else for (let i = 0; i < n; i++) color.fill(1, (vo + i) * 4, (vo + i) * 4 + 4);
      }
      if (l.index && l.index.length) {
        for (let i = 0; i < l.index.length; i++) index[io + i] = l.index[i] + vo;
        io += l.index.length;
      } else {
        for (let i = 0; i < n; i++) index[io + i] = vo + i;
        io += n;
      }
      vo += n;
    }
    return { pos, nrm, uv, color, index, vertices: nv };
  }

  /* 从网格顶点烘一张 2D 高度图，供资产模式贴地（离线管线产出的世界没有 world.json 也要能走）
   * 每格取「中位数」而不是最低/平均：最低会被埋在地下的孤立尖刺带跑，平均会被树冠抬高，
   * 中位数对两者都稳。 */
  function buildHeightGrid(layers, radius, cell) {
    const N = Math.max(8, Math.round((radius * 2) / cell) + 1);
    const half = radius;
    const buckets = new Array(N * N);
    for (const l of layers) {
      const p = l.pos;
      for (let i = 0; i < p.length; i += 3) {
        const x = p[i], y = p[i + 1], z = p[i + 2];
        if (x < -half || x > half || z < -half || z > half) continue;
        const k = Math.round((z + half) / cell) * N + Math.round((x + half) / cell);
        if (k < 0 || k >= buckets.length) continue;
        (buckets[k] || (buckets[k] = [])).push(y);
      }
    }
    const best = new Float32Array(N * N);
    best.fill(NaN);
    for (let k = 0; k < buckets.length; k++) {
      const b = buckets[k];
      if (!b) continue;
      b.sort((a, c) => a - c);
      best[k] = b[b.length >> 1];
    }
    // 空洞用 3×3 邻居补两轮，再整体平滑一轮，避免格子边界出现台阶
    const tmp = new Float32Array(N * N);
    for (let pass = 0; pass < 2; pass++) {
      tmp.set(best);
      for (let gz = 0; gz < N; gz++) {
        for (let gx = 0; gx < N; gx++) {
          const k = gz * N + gx;
          if (best[k] === best[k]) continue;
          let s = 0, n = 0;
          for (let dz = -1; dz <= 1; dz++) {
            for (let dx = -1; dx <= 1; dx++) {
              const z2 = gz + dz, x2 = gx + dx;
              if (z2 < 0 || z2 >= N || x2 < 0 || x2 >= N) continue;
              const v = tmp[z2 * N + x2];
              if (v === v) { s += v; n++; }
            }
          }
          if (n) best[k] = s / n;
        }
      }
    }
    for (let i = 0; i < best.length; i++) if (best[i] !== best[i]) best[i] = 0;
    const sm = new Float32Array(N * N);
    for (let gz = 0; gz < N; gz++) {
      for (let gx = 0; gx < N; gx++) {
        let s = 0, n = 0;
        for (let dz = -1; dz <= 1; dz++) {
          for (let dx = -1; dx <= 1; dx++) {
            const z2 = gz + dz, x2 = gx + dx;
            if (z2 < 0 || z2 >= N || x2 < 0 || x2 >= N) continue;
            s += best[z2 * N + x2]; n++;
          }
        }
        sm[gz * N + gx] = s / n;
      }
    }
    function at(x, z) {
      // 越界钳到边缘格：旧实现直接 return 0，而 GLB 相机系地面约在 y=-1.7，
      // 玩家一旦走出网格就会从 -1.7 弹到 0（凭空抬高 1.7m）。
      const cx = Math.min(half - 1e-3, Math.max(-half, x));
      const cz = Math.min(half - 1e-3, Math.max(-half, z));
      const fx = (cx + half) / cell, fz = (cz + half) / cell;
      const x0 = Math.min(N - 2, Math.max(0, Math.floor(fx)));
      const z0 = Math.min(N - 2, Math.max(0, Math.floor(fz)));
      const tx = fx - x0, tz = fz - z0;
      const v00 = sm[z0 * N + x0], v10 = sm[z0 * N + x0 + 1];
      const v01 = sm[(z0 + 1) * N + x0], v11 = sm[(z0 + 1) * N + x0 + 1];
      return (v00 * (1 - tx) + v10 * tx) * (1 - tz) + (v01 * (1 - tx) + v11 * tx) * tz;
    }
    return { at, size: N, cell };
  }

  async function enterAsset(url) {
    const token = ++loadToken;
    loader.classList.remove('hidden', 'done', 'burst');
    el('loaderImg').style.opacity = '0';
    el('loaderTitle').textContent = '加载三维世界';
    lastProgress = 0;
    setProgress(4, '下载资产…');
    try {
      ensureRenderer();
      const asset = await PW.loadGLB(url, (p) => setProgress(4 + p * 82, '下载资产 ' + Math.round(p * 100) + '%'));
      setProgress(90, '上传显卡…');
      await sleep(40);

      const col = asset.img ? colorsFromImage(asset.img)
        : { top: [0.34, 0.44, 0.62], bot: [0.55, 0.56, 0.52], lum: 0.45 };
      const vertexColor = !asset.img;
      /* v3 资产（pipeline/v3）：视锥内是照片贴图 + 双四边形连续曲面，视锥外是
         扩散生成的地形/地物/360° 天空穹顶。它自带真实几何，所以两件事必须区别对待：
           · 不再接程序化延伸器（已有地形会与它打架，且会在照片区铺一块灰板）；
           · 出生俯仰角 = 拍摄俯仰角（照片光轴常常不是水平的）。 */
      const v3meta = (asset.extras && asset.extras.v3) || null;
      const gl = renderer.gl;
      const toMesh = (l) => ({
        aPos: buf(gl, l.pos, 3),
        aNormal: l.nrm ? buf(gl, l.nrm, 3) : null,
        aUV: l.uv ? buf(gl, l.uv, 2) : null,
        aColor: l.color ? buf(gl, l.color, 4) : null,
        index: { buf: idxBuf(gl, l.index), count: l.index.length },
        count: l.index.length,
        mode: gl.TRIANGLES
      });
      // 点云层（v2.1）：GLB 里 mode=0 的层。用于"细枝/树叶"这类网格必然剪碎的
      // 内容 —— 没有"连接"就没有拉丝与碎片。点云层单独走 points2 通道。
      const pointLayers = asset.layers.filter((l) => l.points);
      const triLayers = asset.layers.filter((l) => !l.points);
      let pointsMesh = null;
      if (pointLayers.length) {
        let n = 0;
        for (const l of pointLayers) n += l.pos.length / 3;
        const pos = new Float32Array(n * 3);
        const col = new Float32Array(n * 4);
        const siz = new Float32Array(n);
        let o = 0;
        for (const l of pointLayers) {
          pos.set(l.pos, o * 3);
          if (l.color) col.set(l.color, o * 4);
          else for (let i = 0; i < l.pos.length / 3; i++) {
            col[(o + i) * 4] = 0.8; col[(o + i) * 4 + 1] = 0.8;
            col[(o + i) * 4 + 2] = 0.8; col[(o + i) * 4 + 3] = 1;
          }
          if (l.size) siz.set(l.size, o);
          o += l.pos.length / 3;
        }
        pointsMesh = {
          aPos: buf(gl, pos, 3),
          aColor: buf(gl, col, 4),
          aSize: buf(gl, siz, 1),
          count: n
        };
      }
      // 分层规则：名为 backdrop 的层做无视差远景；其余层合并成近景（真视差）
      const farLayers = triLayers.filter((l) => l.name === 'backdrop');
      /* v3 的"照片天空壳"单独一层（skycap）：它只在出生点与原图对齐，相机一动就是
         一张悬空的照片卡片。单独拿出来，才能按距离/飞行把它淡掉，而不是连穹顶一起淡。 */
      const capLayers = triLayers.filter((l) => l.name === 'skycap');
      /* 俯视顶面（v2.6）：从点云生成的高度图，只在**视点高于地形最高点**时才画。
         它不能并进近景层 —— 近景是"朝前的壳"，顶面是一张水平的高度图，
         混在一起会让平视时的轮廓变糊。 */
      const topLayers = triLayers.filter((l) => l.name === 'top');
      /* 挖洞版地面（v2.15.1）：地面视角用它，照片足迹留给近景壳。 */
      const topHoleLayers = triLayers.filter((l) => l.name === 'topHole');
      /* 地面裙边单独一层（v2.7.1）：它带顶点色、没有照片纹理，
         并进近景层会被套上照片贴图，整片偏色（实测偏暗红）。 */
      const groundLayers = triLayers.filter((l) => l.name === 'ground');
      /* 封闭体（v2.12）：每个近景物体是"正面+镜像背面+封边"的实体，
         必须**双面绘制**（背面在，绕到后面才不会看到洞）。 */
      const solidLayers = triLayers.filter((l) => l.name === 'solid');
      const solidMesh = solidLayers.length ? toMesh(mergeLayers(solidLayers)) : null;
      /* F3（v2.15.12）：名单里必须排除 topHole。
         漏了它 → 同一份几何既进 nearMesh（mode 4 照片贴图）又进 topHoleMesh
         （mode 1 无贴图 + 白顶点色）→ 共面 z-fighting，mode 1 那份赢下深度测试时
         就是"侧视一块无纹理浅灰面"（ISSUES #4 的真正根因，不是 solid）。 */
      /* v3 把"视锥外的地形/地物"单独成一层：
         · 它有自己的生命（有起伏、顶点色），不该跟着照片壳一起淡出/剔除；
         · 靠它才能在飞行时把照片壳淡掉之后，世界仍然是连续的地面。 */
      const terrainLayers = triLayers.filter((l) => l.name === 'far' || l.name === 'prop');
      const nearLayers = triLayers.filter((l) => l.name !== 'backdrop'
        && l.name !== 'skycap' && l.name !== 'far' && l.name !== 'prop'
        && l.name !== 'top' && l.name !== 'topHole'
        && l.name !== 'ground' && l.name !== 'solid');
      const groundMesh = groundLayers.length ? toMesh(mergeLayers(groundLayers)) : null;
      const topHoleMesh = topHoleLayers.length ? toMesh(mergeLayers(topHoleLayers)) : null;
      const farMesh = farLayers.length ? toMesh(mergeLayers(farLayers)) : null;
      const capMesh = capLayers.length ? toMesh(mergeLayers(capLayers)) : null;
      const terrainMesh = terrainLayers.length ? toMesh(mergeLayers(terrainLayers)) : null;
      const nearMesh = nearLayers.length ? toMesh(mergeLayers(nearLayers)) : null;
      const topMesh = topLayers.length ? toMesh(mergeLayers(topLayers)) : null;
      /* topMaxY 的统计已随 F18 一起删除（顶面是常驻绘制，高度阈值不存在了）。 */
      const tex = asset.img
        ? PW.GL.createTexture(gl, asset.img, { wrap: 'clamp', clampV: true, aniso: 8, flipY: false })
        : whiteTex(gl);

      // 贴地：从网格顶点里烘一张高度图（每格取最低点再平滑），行走时才不会漂在半空。
      // 取"最低"而不是"平均"，是因为分层世界里树/石/山都在地面之上，地面才是能站的那层。
      // ?dbg=layers：每层画成纯色（红=近景 蓝=远景 绿=裙边），用来定位 artifact 归属
      const dbgLayers = new URLSearchParams(location.search).get('dbg') === 'layers';

      const grid = buildHeightGrid(asset.layers, 90, 2.0);
      // 注意顺序：isWall 必须先于 groundY 求值（写成后面会导致 TDZ 报错，
      // 表现为"资产加载失败"、所有世界一片黑 —— 实测踩过）。
      const isWall = asset.layers.length === 1 && asset.layers[0].name === 'wall';
      const groundY = isWall ? 0 : grid.at(0, 0);

      /* 照片墙世界（非风景照的退回形态）：只有一面四边形，没有地面网格。
         必须把视点钉在原点 —— 否则 buildHeightGrid 会从四个角点算出一个离谱的
         地面高度，玩家视线被抬到墙的上方，屏幕上方露出一大片灰
         （实测：云海那张自下而上只看到半张照片）。 */
      const w = {
        name: '离线重建',
        mode: 'asset',
        vertexColor,
        heightAt: isWall ? (() => 0) : ((x, z) => grid.at(x, z)),
        eyeHeight: isWall ? 0 : 1.7,
        walkRadius: isWall ? 0 : 55,
        dbgLayers,
        fogColor: col.bot,
        skyTop: col.top,
        skyHorizon: col.bot,
        ambient: [1, 1, 1],
        sunColor: [0.9, 0.9, 0.9],
        sunPower: 0.4,
        exposure: 1.0,
        /* 曝光目标 = 原片平均亮度 × 0.95（上限 0.62）。
           原来 × 0.82 且上限 0.55 → 亮照片被压暗约 18%，
           站在原点看明显比原片暗（实测 0.495 vs 原片 0.582）。
           注意：这个值只影响**自动曝光的目标**，改光照参数是无效的
           （expо 循环会把任何光照变化归一化掉）。 */
        targetLum: Math.min(0.62, Math.max(0.28, col.lum * 0.95)),
        sunDir: [0.3, 0.5, -0.8],
        lights: { fogDensity: 0.0016 },
        tex: { photo: tex, ground: null, sky: null, grass: null },
        /* ?hide=top,solid,... ：把 GLB 的某一层整层摘掉，用来定位"这条artifact/
           断口是哪一层画的"。v2 链路（服务端 LDI → 6 层 GLB）里层名是
           near / backdrop / solid / ground / top / topHole，
           逐层摘是唯一能给出确定结论的取证方式（指标对这些不敏感）。 */
        meshes: (() => {
          const m = { relief: nearMesh, backdrop: farMesh, skyCap: capMesh,
            terrain: terrainMesh, farRidge: null, nearRidge: null,
            sky: null, ground: groundMesh, solid: solidMesh, rocks: null, grass: null,
            shrub: null, top: topMesh, topHole: topHoleMesh };
          const hide = (new URLSearchParams(location.search).get('hide') || '')
            .split(',').map((t) => t.trim()).filter(Boolean);
          const map = { near: 'relief', backdrop: 'backdrop', solid: 'solid',
            ground: 'ground', top: 'top', topHole: 'topHole', skycap: 'skyCap' };
          for (const h of hide) {
            const k = map[h] || h;
            if (k in m) m[k] = null;
          }
          if (hide.length) console.info('[app] 已隐藏层：' + hide.join(','));
          return m;
        })(),
        /* F18（v2.15.12）：这里原来还有 `topShowAbove: topMaxY + 1.0`，标注"视点
           高于地形最高点才显示顶面"—— 但渲染端从 v2.13 起是**常驻绘制**，这个字段
           从来没人读。留着会误导排查（按字段语义推断会得出错误结论），已删除。
           `topMaxY` 仍保留给下面的调试信息用。 */
        particles: null,
        pointsMesh: null,
        points2Mesh: pointsMesh,
        points2Px: (asset.camera && asset.camera.pointPx) || 2.6,
        analysis: { horizon: 0.5, meanLum: col.lum }
      };

      /* 相机内参与几何必须同源（v2）：GLB 里写的是**照片真实的** hfov/vfov，
         而 fovRad() 的默认分支是按 66° 推的。不接这根线，整幅重建会被放大并错位，
         画面底部会露出被拉长的边缘几何。
         做法：把照片尺寸塞进 analysis（fovRad 用它还原照片宽高比）、把 vfov 塞进
         reliefMeta —— fovRad 已有的分支就会算出"水平与照片一致、竖直按画布比例裁切"
         的正确取景。 */
      if (asset.camera && asset.camera.vfov > 0.1) {
        w.reliefMeta = { vfov: asset.camera.vfov, hfov: asset.camera.hfov };
        w.analysis = {
          horizon: 0.5, meanLum: col.lum,
          W: asset.camera.w || 4, H: asset.camera.h || 3
        };
        w.photoMeta = asset.camera;
      }

      if (token !== loadToken) return;            // 期间又点了别的卡片
      if (player) player.dispose();
      disposeWorld(world);                        // F7：回收旧 world 的 GL 资源
      world = w;
      current = { name: '离线重建', sub: 'asset · world.glb' };
      pruneObjectUrls();
      player = PW.createPlayer(canvas, world);
      /* 资产模式的视场必须**正好等于**资产的视场：在线链路用 fovMul=1.06 多看一点
         周边没关系（视锥外还有程序化几何），但 GLB 的几何正好止于照片视锥边界，
         多看 6% 就会在左右两侧露出被拉长的贴图边缘（实测的竖条）。 */
      player.state.fovMul = 1.0;
      if (window.__spawn) player.reset(window.__spawn.x, window.__spawn.z, window.__spawn.yaw);
      else player.reset(0, 0, 0);
      // 与 enterWorld 同一口径：reset() 会把 pitch 置为 -0.02，
      // ?pitch= 指定的出生俯仰角在离线 GLB 路径也要还原。
      if (window.__spawn && typeof window.__spawn.pitch === 'number') player.state.pitch = window.__spawn.pitch;
      /* v3：没有显式 ?pitch= 时用**拍摄俯仰角**出生。世界是按真实俯仰建的，
         站在原点用 pitch=0（reset 的默认 -0.02）看，地平线会整条错位。 */
      else if (v3meta && asset.camera && typeof asset.camera.pitch === 'number') {
        player.state.pitch = asset.camera.pitch;
      }
      if (v3meta) {
        if (document.body) document.body.dataset.pipe = 'v3';
        if (v3meta.bbox) {
          /* 包围盒要**外扩 8m**：v3 的世界全部在相机前方（z ≤ -1），
             而玩家出生在 z=0 —— 不扩的话出生点就算"在壳外 0.96m"，
             照片壳按 shellAlpha 立刻被淡成 0，整屏只剩天空壳，发灰看不清
             （实测：probe 出 shellAlpha=0，这是最容易被当成"雾太重"的坑）。 */
          const PAD = 8;
          w.groundColorMeta = {
            x0: v3meta.bbox.x0 - PAD, z0: v3meta.bbox.z0 - PAD,
            spanX: Math.max(1, v3meta.bbox.x1 - v3meta.bbox.x0 + PAD * 2),
            spanZ: Math.max(1, v3meta.bbox.z1 - v3meta.bbox.z0 + PAD * 2)
          };
        }
        // 天空壳/穹顶不加雾：它们本身就是远景，加雾会整屏发灰
        w.backdropFog = 0.02;
        /* 近景层绕序已在 v3 管线里改成"从相机看逆时针"，所以**开**背面剔除：
           站在照片背后/上方时看到的是地形，而不是照片壳的背面（那才是"漂浮板子"的来源）。 */
        w.reliefCull = true;
        /* 飞行时不摘场景：v3 的几何是真三维表面，仰视时视锥内那块地面只存在于
           near 层里，按 v2 的做法跳过就会"一起飞全白"。 */
        /* 关掉程序化天空的"太阳光晕"：那是 pow(sd,20)*0.16 的一团大白雾，
           直径接近画面一半 —— 用户反复指的"视线中心大圆圈"其实是它，
           不是穹顶、也不是天空壳（v3 的天空内容来自全景，本来就不需要这层光晕）。 */
        w.sunPower = 0;
        /* 暗角对天空很敏感（见 renderer 里的注释）：v3 收到 0.10 */
        w.vignette = 0.10;
        w.planKeepsScene = true;
        /* 飞行上限：单视图重建的头顶没有几何，飞太高只会看到壳的破口 */
        w.flyMaxRel = 12;
        /* 活动半径：单视图重建只在拍摄点附近成立。走开 5m 之后无论怎么渲染都不对
           （对比图第 3/4/5 格全是化的）——所以干脆把"走"限制成小范围环视，
           真正的观察手段是**抬头/转身/起飞**。 */
        w.moveRadius = 1.5;
        /* 投影纹理参数：拍照时的相机位姿（GLB 坐标），着色器据此把世界坐标反解成 UV。
           照片在贴图里占 v ∈ [0, vScale]，下面 8px 是顶点色层用的白条。 */
        /* 地形细节：把扩散全景当贴图按方向采样（见 glutil 的 mode 7）。
           约定与 GLB 同目录、同名：world-v3-x.glb → world-v3-x.pano.jpg。
           没有这张图就静默退回顶点色，不影响进入世界。 */
        if (url) {
          const panoUrl = String(url).replace(/\.glb($|\?)/i, '.pano.jpg$1');
          if (panoUrl !== url) {
            const im = new Image();
            im.onload = () => {
              /* 顺手把程序化天空的上下色取自全景本身：v3 改用程序化天空之后，
                 天空色如果还来自"照片分析"就会和地形/全景不是一个色调
                 （抬头是一块中性灰，低头是暖色湖面）。这里取全景的顶行与地平线行。 */
              try {
                const cv = document.createElement('canvas');
                cv.width = 4; cv.height = 8;
                const cx = cv.getContext('2d');
                cx.drawImage(im, 0, 0, 4, 8);
                const px = cx.getImageData(0, 0, 4, 8).data;
                const at = (row) => {
                  let r = 0, g = 0, b = 0;
                  for (let i = 0; i < 4; i++) {
                    const o = (row * 4 + i) * 4;
                    r += px[o]; g += px[o + 1]; b += px[o + 2];
                  }
                  return [r / 4 / 255, g / 4 / 255, b / 4 / 255];
                };
                w.skyTop = at(0);
                w.skyHorizon = at(4);
              } catch (e) { /* 跨域画布会抛，忽略 */ }
              try {
                // U 方向必须 REPEAT（360° 环绕），V 方向 CLAMP（天顶/天底）
                w.tex.pano = PW.GL.createTexture(gl, im, {
                  wrap: 'repeat', clampV: true, aniso: 4, flipY: false
                });
                w.panoUrl = panoUrl;
              } catch (e) { console.warn('全景贴图创建失败：', e && e.message); }
            };
            im.onerror = () => console.warn('没有全景贴图（' + panoUrl + '），地形用顶点色');
            im.src = panoUrl;
          }
        }
        /* 不再绘制"照片天空壳"：它在出生点与原图一致，但抬头看时它的边缘
           相对穹顶就是一圈可见的**大圆盘**（对比图第 2 格）。v3 的天空统一交给
           穹顶（纯竖直渐变 + 全景方位中值），整片天空不再有边界。 */
        /* 天空统一交给程序化 sky pass（竖直渐变，逐像素、不可能有边界）。
           穹顶那 7.5k 顶点是"按纬度取全景中值"来的，只要全景里有一点方位变化，
           抬头看就是一圈圆盘；程序化渐变没有这个问题，也少一次绘制。 */
        if (capMesh) w.meshes.skyCap = null;
        /* 穹顶**不删**：改成用全景贴图当天空球（mode 8）。之前删掉它是因为
           顶点色的穹顶会出一圈圆盘；用贴图 + 方向采样就没有这个问题，
           而且天空终于有云和天光了（原来抬头是一片死灰）。 */
        /* 天空由 sky pass 全屏采样全景，**不再画穹顶几何**（网格边界 = 那个圆盘） */
        w.skyDome = 'pano';
        w.meshes.backdrop = null;
        if (asset.camera) {
          const c3 = asset.camera;
          const wu = (v3meta.white_uv && v3meta.white_uv[1]) || 1.0;
          w.projCam = {
            on: 1,
            eye: [0, (typeof c3.eye_height === 'number' ? c3.eye_height : 1.7), 0],
            tanH: Math.tan(c3.hfov * 0.5),
            tanV: Math.tan(c3.vfov * 0.5),
            pitch: c3.pitch || 0,
            vMax: Math.max(0.5, 2 * wu - 1) - 0.002
          };
        }
        setHudPipe('v3', 'v3 · 扩散地形');
      }

      /* 调试钩子：资产世界的运行时对象（质检脚本用）。
         与 window.__setCam / window.__spawn 同一套路，不改运行时行为。 */
      if (typeof window !== 'undefined') window.__pwWorld = w;

      el('hudName').textContent = '离线重建世界';
      if (!el('hudPipe').textContent) setHudPipe('asset', '资产');

      /* 视锥外的程序化延伸（v2.5）：资产世界此前只到网格边界，外面是雾。
         接上 extend.js 之后，走出去能看到继续生成的地形，并且"实时变化"
         （混乱度/记忆力）在这里也生效 —— 与在线链路同一套参数。
         全部包在 try 里：延伸器依赖若干世界字段，缺一个都不该把资产世界弄坏。 */
      /* 程序化延伸**默认开启**（?ext=0 关闭）：要求是"看到的地方都要补满" ——
         照片只覆盖一个视锥，其余方向必须是继续生成的地形，不能是雾或空底。
         适配层把资产世界的事实喂给它（近景层的 bbox、照片地面色、
         顶点色 tile、纵向基准），四处契约冲突已在 v2.4.3/2.4.4 修完。 */
      const wantExt = new URLSearchParams(location.search).get('ext') !== '0' && !v3meta;
      if (wantExt && window.PW.createExtender && !isWall) {
        try {
          w.extraMeshes = [];
          /* ---- 资产世界 → extend.js 的适配层 ----
             三个字段必须按"资产世界的事实"给，不能沿用在线链路的常数：
               1) bbox：必须是**网格实际覆盖的矩形**。写小了 → insideRect 判错 →
                  tile 生成在照片区内部，把照片盖成一片暗地面（实测就是这么翻车的）。
               2) groundBaseLin：必须是照片里的**地面色**（线性空间），不是天空底部的颜色，
                  否则延伸地形会偏色成灰蓝。
               3) flatMode=1：平面 tile 用顶点色，不用照片贴图（UV 是平铺坐标，会采错）。 */
          /* 只用**近景层**算照片区域：backdrop 层本身横跨 ±437m（蓝天/远山），
             把它算进去会让 A/B 变成 800m —— tile 就只会在 800m 外生成，
             而底板（±A × B 的一大片）会盖住整个近景（实测：三视图全黑）。 */
          let maxFwd = 0, maxSide = 0;
          for (const l of asset.layers) {
            if (!l.pos || l.points) continue;
            if (l.name === 'backdrop' || l.name === 'ground' || l.name === 'scatter') continue;
            for (let i = 0; i < l.pos.length; i += 3) {
              const x = l.pos[i], z = -l.pos[i + 2];
              if (z > maxFwd) maxFwd = z;
              if (Math.abs(x) > maxSide) maxSide = Math.abs(x);
            }
          }
          if (maxFwd < 10) maxFwd = 60;
          const B = clamp(maxFwd * 1.04, 25, 400);
          const A = clamp(maxSide * 1.04, 25, 600);
          w.bbox = { A, B };

          /* ---- 延伸地形的地面配色（这一段踩过三次坑，按结论写） ----
             延伸器的顶点色是 col = (colorAt*0.45 + fog*0.55) / groundBaseLin
             （见 extend.js 的 buildTileMesh）。两个要求：

             ① base 必须与 fog **同量级**。取照片地面线性色（0.033）而 fog 是 0.19 时，
                比值 3.6 → 整片地形刷成浅灰（背后那条"白带"）。
             ② colorAt 必须接，且返回值 ≈ base。不接时顶点色恒为 1，渲染成白色。

             所以：base = 雾色均值×1.05，colorAt 返回 base 加一点低频扰动。
             比值 ≈ 0.45 + 0.55×(fog/base) ≈ 1，延伸地形与雾自然衔接。 */
          let gcol = col.bot;
          try {
            if (asset.img) {
              // 取照片下方 30% 里**70 分位亮度**的像素：平均值会混进大片阴影（偏黑），
              // 最大值会撞上岩石高光（偏砖红），70 分位接近"日照下的地面"。
              const cv = document.createElement('canvas');
              cv.width = 24; cv.height = 24;
              const c2 = cv.getContext('2d');
              c2.drawImage(asset.img, 0, Math.floor(asset.img.height * 0.70),
                           asset.img.width, Math.ceil(asset.img.height * 0.30), 0, 0, 24, 24);
              const d = c2.getImageData(0, 0, 24, 24).data;
              const arr = [];
              for (let i = 0; i < d.length; i += 4) {
                arr.push([0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2],
                          d[i], d[i + 1], d[i + 2]]);
              }
              arr.sort((a, b2) => a[0] - b2[0]);
              const px = arr[Math.min(arr.length - 1, Math.floor(arr.length * 0.70))];
              gcol = [px[1] / 255, px[2] / 255, px[3] / 255];
            }
          } catch (e) { /* 取不到就用底色 */ }

          const fogBase = Math.max(0.02, (col.bot[0] + col.bot[1] + col.bot[2]) / 3);
          w.groundBaseLin = [fogBase * 1.05, fogBase * 1.0, fogBase * 0.96];
          w.groundTint = gcol;
          const gb = w.groundBaseLin;
          w.groundColorAt = function (x, z) {
            const v = 1 + 0.09 * Math.sin(x * 0.018 + 1.7) * Math.cos(z * 0.021 - 0.4);
            return [gb[0] * v, gb[1] * v, gb[2] * v];
          };
          // 资产世界的地面**没有照片贴图可铺**（网格自带 UV），底板走顶点色
          w.floorMode = 1;
          w.floorTex = null;

          const ext = window.PW.createExtender({
            gl, world: w,
            seed: hashString((current && current.name || 'asset') + '|ext'),
            server: apiBase || '',
            radius: 5, flatMode: 1,
            // 资产世界的 y=0 在视平线，地面在 groundY：延伸几何要整体下移这么多
            yBase: groundY,
            live: liveState.on, chaos: liveState.chaos, memory: liveState.memory
          });
          w.extender = ext;
          /* 延伸器会给"洞"铺一块兜底底板（在线链路需要，因为它的地面是程序化的）。
             资产世界本身有 GLB 地面几何，这块底板只会变成一大片灰板挡在眼前
             （实测：下半屏被一块灰平面盖住，顶点色是 1×光照所以发白）。 */
          w.meshes.floor = null;
          w.meshes.nearGround = null;
          setHudPipe('v2', 'V2+延伸');
        } catch (e) {
          console.warn('[ext] 资产世界接入延伸器失败，保持仅有网格边界', e);
        }
      }
      el('hudSub').textContent = pointsMesh
        ? (pointsMesh.count.toLocaleString() + ' 点 · ' + (asset.bytes / 1048576).toFixed(1) + 'MB · 点云')
        : (asset.triangles.toLocaleString() + ' 三角 · ' + (asset.bytes / 1048576).toFixed(1) + 'MB · ' + asset.layers.length + ' 层');
      setProgress(100, '完成');

      landing.classList.add('leaving');
      setTimeout(() => { landing.style.display = 'none'; }, 820);
      stage.classList.add('visible');
      stage.setAttribute('aria-hidden', 'false');
      loader.classList.add('burst');
      await sleep(150);
      loader.classList.add('done');
      setTimeout(() => loader.classList.add('hidden'), 900);

      // 曝光预热：让画面均值对齐贴图亮度
      try {
        const a0 = Math.max(0.4, canvas.clientWidth / Math.max(1, canvas.clientHeight));
        const vf = clamp(2 * Math.atan(Math.tan(33 * Math.PI / 180) / a0), 0.55, 1.42);
        for (let i = 0; i < 8; i++) {
          // 预热相机必须和玩家实际眼位一致：asset 模式的地面在 GLB 相机系里
          // 不是 y=0，写死 [0,1.7,0] 会让曝光按另一个视角去测量。
          const m = renderer.render(world, { eye: [0, groundY + world.eyeHeight, 0], yaw: 0, pitch: 0, fov: vf },
            { time: 0, fade: 1, measure: true });
          if (!(m > 0)) break;
          const ratio = (world.targetLum || 0.42) / Math.max(m, 0.02);
          if (Math.abs(ratio - 1) < 0.03) break;
          world.exposure = clamp(world.exposure * Math.pow(ratio, 0.7), 0.08, 3.6);
        }
      } catch (e) { /* 预热失败不阻塞 */ }

      fadeStart = performance.now();
      fadeFrames = 0; expoCount = 0; frameIdx = 0;
      fade = instant ? 1 : 0.02;
      lastT = performance.now();
      startLoop();
      if (document.body) document.body.dataset.ready = '1';
      if (document.body) document.body.dataset.mode = 'asset';
      /* 站立高度（给渲染器判断"是否在地面附近"用） */
      w.groundY = groundY;
      if (document.body) document.body.dataset.ground = groundY.toFixed(2);
    } catch (e) {
      console.error(e);
      el('loaderTitle').textContent = '资产加载失败';
      setProgress(100, String(e.message || e));
      if (document.body) document.body.dataset.err = String(e.message || e).slice(0, 200);
    }
  }

  function buf(gl, data, size) {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return { buf: b, size: size };
  }
  function idxBuf(gl, data) {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return b;
  }

  /* ---------------- 主循环 ---------------- */

  /* 相机垂直 FOV（v1.16.1）：
     重建是按「照片自己的视场」把像素反投影回三维的。如果渲染相机比这个视场更宽，
     照片几何就只占屏幕中间一块，四周全是天穹的灰色——这就是用户反馈的
     「一张明信片浮在空中」。这里让相机视场与重建视场对齐（取覆盖而非留边，
     宁可裁掉照片边缘几个百分点，也不要露出灰边）。 */
  function fovRad() {
    const aspect = Math.max(0.4, canvas.clientWidth / Math.max(1, canvas.clientHeight));
    const mul = player ? player.state.fovMul : 1;
    const rm = world && world.reliefMeta;
    if (rm && rm.vfov > 0.1 && world.analysis) {
      const tanV = Math.tan(rm.vfov / 2);
      const pa = world.analysis.W / Math.max(1, world.analysis.H);
      /* v1.18.1：按**水平**视场取景，而不是"竖直/水平都要盖住"。
         旧写法 need = max(tanV, tanV·pa/aspect) 在竖屏（aspect≈0.46）算出
         need=1.30 → 竖直 81°，比照片的 41.5° 大一倍：上下各多出 20° 没有几何，
         于是手机上是"照片条悬在大片地面/天空中间"，边缘还露出照片网格的断片。
         现在把水平视场夹在 30°~78°，竖直由宽高比推出：
           · 竖屏：水平 30° → 竖直 60°（只多 9°，天空和地面都能兜住）
           · 横屏：水平照旧由照片宽高比推出，与 v1.16.1 一致
         水平视场小于照片视场时是"裁掉照片左右几个百分点"，不会露边。 */
      const hfovPhoto = 2 * Math.atan(tanV * pa);
      const hfov = clamp(hfovPhoto, 30 * Math.PI / 180, 78 * Math.PI / 180);
      const vfov = 2 * Math.atan(Math.tan(hfov / 2) / Math.max(0.4, aspect));
      return clamp(vfov * mul, 0.35, 1.4);
    }
    const hfov = (66 * Math.PI / 180) * mul;
    return clamp(2 * Math.atan(Math.tan(hfov / 2) / aspect), 0.55, 1.42);
  }

  function updateParticles(dt) {
    const p = world.particles;
    if (!p) return;
    const pos = p.pos;
    const n = p.count;
    const px = player.state.x, pz = player.state.z;
    for (let i = 0; i < n; i++) {
      const i3 = i * 3;
      pos[i3] += p.vel[i3] * dt;
      pos[i3 + 2] += p.vel[i3 + 2] * dt;
      pos[i3 + 1] += p.vel[i3 + 1] * dt;

      if (pos[i3 + 1] < -0.6) {
        pos[i3 + 1] = 13 + Math.random() * 4;
        const a = Math.random() * Math.PI * 2, r = Math.random() * 46;
        pos[i3] = px + Math.sin(a) * r;
        pos[i3 + 2] = pz + Math.cos(a) * r;
      }
      const dx = pos[i3] - px, dz = pos[i3 + 2] - pz;
      if (dx * dx + dz * dz > 3200) {
        const a = Math.random() * Math.PI * 2, r = 6 + Math.random() * 44;
        pos[i3] = px + Math.sin(a) * r;
        pos[i3 + 2] = pz + Math.cos(a) * r;
        pos[i3 + 1] = 0.5 + Math.random() * 14;
      }
    }
    const gl = renderer.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, world.pointsMesh.aPos.buf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, pos);
  }

  const CARD = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  let hudTick = 0;

  function updateHUD(dt) {
    hudTick += dt;
    if (hudTick < 0.1) return;
    hudTick = 0;

    const s = player.state;
    let deg = (s.yaw * 180 / Math.PI) % 360;
    if (deg < 0) deg += 360;
    const strip = el('compassStrip');
    if (strip) {
      const step = 26 / 15;
      strip.style.transform = `translateX(${-((360 + deg) / 15) * 26}px)`;
    }
    el('hudCoords').textContent = s.x.toFixed(1) + ' / ' + s.z.toFixed(1);
  }

  /* 照片薄壳淡出开关（v2.15.11）。默认开，`?shellfade=0` 关掉做 A/B。
     判据是"相机是否还在照片覆盖矩形内"：走出矩形就说明你已经在照片壳的背面/
     侧后方，单层壳在那里只会摊成放射长刺；而程序化地面 + 世界幕接管反而干净。 */
  const SHELL_FADE = (typeof location !== 'undefined'
    && new URLSearchParams(location.search).get('shellfade') !== '0');

  function frame(now) {
    if (!running) return;
    const dt = Math.min((now - lastT) / 1000 || 0.016, 0.05);
    lastT = now;
    time += dt;

    player.update(dt);
    /* 俯视模式跟着**飞行状态**走：顶面含远景后最高点几百米，按高度阈值永远切不过去。
       飞行 = 想看全局，直接用它当开关最稳。
       进入/退出时**显式**摘掉/挂回地面裙边与延伸 tile —— 只靠渲染器里的
       `!world.showPlan` 判断实测没生效（那道 V 形台阶就是裙边），
       这里直接把 mesh 从世界上取下来，不依赖渲染器内部条件。 */
    /* v3：照片天空壳随"离出生点的水平距离"和"是否飞行"淡出（见 renderer 的 skyCap） */
    if (world.meshes.skyCap) {
      const d = Math.hypot(player.state.x, player.state.z);
      const t = Math.min(1, Math.max(0, (d - 1.0) / 4.0));
      world.skyCapAlpha = (1 - t) * (player.state.flying ? 0 : 1);
    /* 陡俯视时照片壳没有意义：它的"底面"是拍摄时看不到的那一侧，俯视会露出一片
       被拉长的斜面（对比图"低头看地"那格）。俯角超过 ~20° 就只留全景天空球。 */
    world.hideShellLookDown = (!player.state.flying && player.state.pitch < -0.35);
    }
    /* 飞行时把"照片壳"淡掉：它是从拍摄点投影出来的一层，从上方看就是一堆漂浮的
       碎片 + 竖向拉丝。12m 起淡、20m 淡完；剩下的地形层是连续的地面。 */
    /* 当前眼高交给渲染器：地面高度附近不画地形（见 renderer 里的说明） */
    world.eyeY = player.state.y;
    if (world.meshes.terrain) {
      // 用**绝对高度**而不是"离地高度"：heightAt 会把外层地形的起伏算进去，
      // 在上坡时"离地 12m"其实已经飞了 20m，阈值形同虚设（实测没淡掉）。
      /* 照片壳的可见范围：出生点附近 3m 内是 1（与原图逐像素一致），
         走到 10m 外淡完，飞行 8~14m 淡完 —— 之后再留着它只会露出"位移曲面被拉伸"
         的行拉丝（单视图重建的固有问题）。淡出之后看到的是同一片地方的全景地形，
         不是另一个世界，也不是空白。 */
      /* 淡出窗口要和飞行上限对齐：上限 12m，如果 14m 才淡完，玩家停在上限时永远是
         半透明叠加（照片壳与地形叠成"双影"）。这里 6m 起淡、12m 淡完。 */
      const dShell = Math.hypot(player.state.x, player.state.z);
      /* 水平窗口收紧到 1.5→4.5m：站在出生点旁边 5m 时壳还半透明的话，
         看到的是"位移曲面被斜着看"的一片糊（对比图第 3/4/5 格）。
         淡完之后用同一张全景铺的地形，观感与照片基本一致（第 6~9 格）。 */
      /* 活动范围锁在出生点 1.5m 内（见 player 的 world.moveRadius），
         所以壳几乎不会因为距离而淡出；这里只保留飞行淡出。 */
      const tD = Math.min(1, Math.max(0, (dShell - 2.0) / 6.0));
      const tH = Math.min(1, Math.max(0, (player.state.y - 6.0) / 6.0));
      const t = Math.max(tD, tH);
      world.shellHide = t * t * (3 - 2 * t);
    }
    /* v3 资产**不切俯视 LOD**：v2 那套"飞行就只画顶面 + tile"是因为它的世界是
       一层朝前的薄壳，从上面看必然是一堆竖条。v3 有真三维地形，切过去反而把
       near/far 全摘掉 —— 实测一起飞整屏只剩清屏色（"全白"）。
       这里让 plan 恒为 false，飞行就只是把相机抬高。 */
    const planNow = !!player.state.flying && !v3meta;
    if (planNow !== world.showPlan) {
      world.showPlan = planNow;
      if (planNow) {
        world._stash = {
          ground: world.meshes.ground,
          extra: world.extraMeshes,
          // tex.ground 必须一并存下来：它是 GL 纹理句柄，置 null 后渲染器
          // drawMesh(tex: world.tex.ground) 拿到 null → 降级成纯顶点色，
          // 落地后整片地面变黑白板（旧代码只存 mesh 不存 tex，永久丢失）。
          texGround: world.tex ? world.tex.ground : null,
        };
        world.meshes.ground = null;
        world.extraMeshes = [];
        if (world.tex) world.tex.ground = null;
      } else if (world._stash) {
        world.meshes.ground = world._stash.ground;
        world.extraMeshes = world._stash.extra || [];
        if (world.tex && world._stash.texGround) world.tex.ground = world._stash.texGround;
        world._stash = null;
      }
    }
    /* 照片薄壳淡出（v2.15.9）：走出照片覆盖矩形（x∈[-55,55]、z∈[-100,0]）后淡出。
       原因：单层壳在矩形外被**侧视/背视**时，只会把纹理沿视线摊成一片放射状长刺
       （32 机位 C 圈 r=75 的实测截图），而那里本来就该由程序化地面 + 世界幕接管 ——
       同一机位 `?skip=shell` 的画面反而干净。
       出界 2m 起淡、18m 外全透明；`?shellfade=0` 可关掉做 A/B。 */
    if (SHELL_FADE) {
      const bb = world.groundColorMeta || { x0: -55, z0: -100, spanX: 110, spanZ: 100 };
      const x0 = bb.x0, x1 = bb.x0 + bb.spanX, z0 = bb.z0, z1 = bb.z0 + bb.spanZ;
      const px = player.state.x, pz = player.state.z;
      const dOut = Math.max(0, x0 - px, px - x1, z0 - pz, pz - z1);
      /* 判据（v2.15.11 修正）：不是"离矩形多远"，而是**在壳的哪一侧**。
         C 圈（r=75）落在 z=-101，已经在照片壳的**背面**：那里每条射线看到的都是
         壳的背面/极掠射面，单层壳必然摊成放射长刺（`?skip=shell` 干净得多）。
         所以只要出界就淡出，0.6m 过渡（≈跑动 0.2s），不做 16m 长坡 —— 长坡在
         出界 1m 处几乎不生效，等于没修。 */
      const t = Math.min(1, Math.max(0, dOut / 0.6));
      world.shellAlpha = 1 - (t * t * (3 - 2 * t));       // smoothstep(2, 18, dOut) 的补
      if (typeof window !== 'undefined' && window.PW) {
        window.PW.__shellAlpha = { a: +world.shellAlpha.toFixed(3), dOut: +dOut.toFixed(1) };
      }
    } else {
      world.shellAlpha = 1;
    }

    if (world.extender) {
      world.extender.update(player.state.x, player.state.z, now,
        { yaw: player.state.yaw, fov: fovRad() });
    }
    updateParticles(dt);
    /* F21（v2.15.12）：`#hudFps` 一直是死 UI（没有任何代码写过它）。
       这里接上真实帧率：每 0.5s 更新一次，避免每帧写 DOM。 */
    fpsAcc += dt; fpsN++;
    if (fpsAcc >= 0.5) {
      const fEl = el('hudFps');
      if (fEl) fEl.textContent = Math.round(fpsN / fpsAcc) + ' fps';
      fpsAcc = 0; fpsN = 0;
    }
    fadeFrames++;
    if (instant) {
      fade = 1;
    } else {
      const ft = Math.min(1, Math.max((now - fadeStart) / 1100, fadeFrames / 42));
      fade = 0.06 + 0.94 * (ft * ft * (3 - 2 * ft));
    }

    const s = player.state;
    frameIdx++;
    // 自动曝光：在入景后几十帧内用实际画面亮度对齐原片
    const needMeasure = expoCount < 8 && fade > 0.55;
    const measured = renderer.render(world, {
      eye: [s.x, s.y, s.z],
      yaw: s.yaw,
      pitch: s.pitch,
      fov: fovRad()
    }, { time, fade, measure: needMeasure });

    if (needMeasure && measured >= 0) {
      expoCount++;
      const ratio = (world.targetLum || 0.35) / Math.max(measured, 0.02);
      if (Math.abs(ratio - 1) > 0.05) {
        world.exposure = clamp(world.exposure * Math.pow(ratio, 0.55), 0.10, 3.4);
      }
    }

    updateHUD(dt);
    requestAnimationFrame(frame);
  }

  function startLoop() {
    if (running) return;
    running = true;
    lastT = performance.now();
    requestAnimationFrame(frame);
  }
  function stopLoop() { running = false; }

  /* ---------------- 交互 ---------------- */
  function buildCompass() {
    const strip = el('compassStrip');
    strip.innerHTML = '';
    for (let h = -360; h <= 720; h += 15) {
      const d = document.createElement('div');
      const norm = ((h % 360) + 360) % 360;
      const isCard = norm % 45 === 0;
      d.className = 'tick' + (isCard ? ' card' : '');
      d.textContent = isCard ? CARD[(norm / 45) % 8] : '·';
      strip.appendChild(d);
    }
  }

  function togglePause(show) {
    const isOpen = !pause.classList.contains('hidden');
    const want = show === undefined ? !isOpen : show;
    if (want) {
      pause.classList.remove('hidden');
      if (player) player.enabled = false;
      if (document.pointerLockElement) document.exitPointerLock();
    } else {
      pause.classList.add('hidden');
      if (player) { player.enabled = true; player.requestLock(); }
    }
  }

  function wireUI() {
    el('btnAbout').addEventListener('click', () => {
      el('modal').classList.remove('hidden');
      renderCapReport();
    });
    el('btnCloseModal').addEventListener('click', () => el('modal').classList.add('hidden'));

    // 开屏 CTA：随便走走 = 随机进一个内置世界；用我的照片 = 同一个上传入口
    const btnStart = el('btnStart');
    if (btnStart) btnStart.addEventListener('click', () => {
      const pick = WORLDS[Math.floor(Math.random() * WORLDS.length)];
      const card = document.querySelector('.card[data-id="' + pick.id + '"]');
      startWorld(pick, card);
    });
    const btnUp = el('btnUploadTop');
    if (btnUp) btnUp.addEventListener('click', () => el('fileInput').click());

    canvas.addEventListener('click', () => {
      if (player && !document.pointerLockElement) player.requestLock();
    });

    el('fileInput').addEventListener('change', (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      const url = URL.createObjectURL(f);
      objectUrls.push(url);
      const name = f.name.replace(/\.[^.]+$/, '') || '我的照片';
      enterWorld({ id: 'up' + Date.now(), name, sub: 'your photo · local', src: url, thumb: url });
      e.target.value = '';
    });

    wireLive();
    document.querySelectorAll('.actions [data-act]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const act = btn.dataset.act;
        if (act === 'gallery') backToGallery();
        // 'live' 不在通用分支里处理：wireLive() 已给 #btnLive 绑了「展开/收起面板」，
        // 这里再翻一次 liveState.on 会让同一次点击把实时生成反向关掉。
        if (act === 'upload') el('fileInput').click();
        if (act === 'sound') {
          const on = ambience.toggle();
          btn.classList.toggle('on', on);
        }
        if (act === 'full') {
          if (!document.fullscreenElement) document.documentElement.requestFullscreen && document.documentElement.requestFullscreen();
          else document.exitFullscreen && document.exitFullscreen();
        }
      });
    });

    el('btnResume').addEventListener('click', () => togglePause(false));
    el('btnNewPhoto').addEventListener('click', () => el('fileInput').click());
    el('btnBackGallery').addEventListener('click', backToGallery);

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (!stage.classList.contains('visible')) return;
        if (!el('modal').classList.contains('hidden')) { el('modal').classList.add('hidden'); return; }
        togglePause();
      }
      if (e.key.toLowerCase() === 'h' && stage.classList.contains('visible')) {
        el('hud').classList.toggle('hide');
      }
    });

    // 点击画廊外部不做事；阻止拖动图片
    document.addEventListener('dragstart', (e) => e.preventDefault());
  }

  /* ---------------- 版本号 ---------------- */
  function applyVersion() {
    const v = 'v' + VERSION;
    const chip = el('verBadge');
    if (chip) chip.textContent = v;
    const fv = el('footVer');
    if (fv) fv.textContent = v;
    if (document.body) document.body.dataset.version = VERSION;
  }

  /* 落地页空闲时预取深度模型：用户真正点「用我的照片」时模型已在本机，
     少掉整段 27MB 等待。省流量模式与 2G 不预取（用户没主动要，别烧流量）。 */
  function prefetchDepthModel() {
    const nav = window.navigator || {};
    const c = nav.connection || {};
    if (c.saveData || c.effectiveType === 'slow-2g' || c.effectiveType === '2g') return;
    if (!window.PW || !window.PW.depth) return;
    const run = () => {
      // v1.26：本次会话若已命中 L1/L2 深度缓存，就不需要 ONNX 会话。
      // 旧逻辑无条件预取，会把「同图重开跳过推理」省下的时间又花在
      // 27MB 模型加载 + WASM 内核编译上。用户之后换一张新图时，
      // estimate() 会自己 init()，只是少了一次提前下载。
      const st = window.PW.depth.cacheStats ? window.PW.depth.cacheStats() : {};
      if (st.idbHits > 0 || st.memHits > 0) return;
      window.PW.depth.init(() => {}).catch(() => {});
    };
    if (window.requestIdleCallback) window.requestIdleCallback(run, { timeout: 5000 });
    else setTimeout(run, 2600);
  }

  /* ---------------- 实时变化（视锥生成 + 混乱度 / 记忆力） ---------------- */
  const LIVE_KEY = 'pw.live';

  function loadLiveState() {
    /* v1.19：默认开启。以前默认关着，扇环外只有一套全局 fBm —— 转身看到的是
       同一套地貌，用户读作「重复生成」。现在默认按扇区主题生成（混乱度 0.45），
       按钮只是用来关掉/调参，而不是「开启才生成」。 */
    // 用户指定默认：实时变化开、混乱度 0.1（整片世界同一个主题）、记忆力 1.0（生成后基本不变）
    let st = { on: true, chaos: 0.1, memory: 1.0 };
    try {
      const saved = JSON.parse(window.localStorage.getItem(LIVE_KEY) || 'null');
      if (saved && typeof saved === 'object') {
        st.on = !!saved.on;
        if (typeof saved.chaos === 'number') st.chaos = clamp(saved.chaos, 0, 1);
        if (typeof saved.memory === 'number') st.memory = clamp(saved.memory, 0, 1);
      }
    } catch (e) { /* 隐私模式没有 localStorage */ }
    const q = new URLSearchParams(location.search);
    if (q.has('live')) st.on = q.get('live') !== '0';
    if (q.has('chaos')) st.chaos = clamp(parseFloat(q.get('chaos')) || 0, 0, 1);
    if (q.has('memory')) st.memory = clamp(parseFloat(q.get('memory')) || 0, 0, 1);
    return st;
  }

  let liveState = { on: true, chaos: 0.1, memory: 1.0 };

  function saveLiveState() {
    try { window.localStorage.setItem(LIVE_KEY, JSON.stringify(liveState)); } catch (e) { /* ignore */ }
  }

  /* 面板是否展开与「实时生成是否开启」分开：默认生成开着、面板收起。
     否则竖屏上一进世界就被面板挡住半个屏幕（实测 480×1040 挡掉 40%）。 */
  let livePanelOpen = false;

  /* 把状态同步到 extender / DOM / body.dataset（测试用） */
  function applyLiveState(push) {
    const ext = world && world.extender;
    // extender 的字段叫 enabled，不是 liveState.on。早期直接传 liveState
    // 会让「关掉实时变化」只改 HUD 与 dataset，视锥外仍继续生成。
    if (push && ext && ext.setLive) ext.setLive({
      enabled: !!liveState.on,
      chaos: liveState.chaos,
      memory: liveState.memory
    });
    const btn = el('btnLive');
    if (btn) btn.classList.toggle('on', livePanelOpen);
    const panel = el('livePanel');
    if (panel) panel.classList.toggle('hidden', !livePanelOpen);
    const cEl = el('liveChaos'), mEl = el('liveMemory');
    if (cEl) cEl.value = String(liveState.chaos);
    if (mEl) mEl.value = String(liveState.memory);
    const onOff = el('liveOn');
    if (onOff) onOff.checked = !!liveState.on;
    const cv = el('liveChaosVal'), mv = el('liveMemoryVal');
    if (cv) cv.textContent = liveState.chaos.toFixed(2);
    if (mv) mv.textContent = liveState.memory.toFixed(2);
    if (document.body) {
      document.body.dataset.live = liveState.on ? '1' : '0';
      document.body.dataset.chaos = liveState.chaos.toFixed(2);
      document.body.dataset.memory = liveState.memory.toFixed(2);
    }
  }

  function wireLive() {
    const btn = el('btnLive');
    if (btn) {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        livePanelOpen = !livePanelOpen;
        if (livePanelOpen && !liveState.on) liveState.on = true;   // 打开面板视为启用
        saveLiveState();
        applyLiveState(true);
      });
    }
    const close = el('btnLiveClose');
    if (close) close.addEventListener('click', (e) => {
      e.stopPropagation();
      livePanelOpen = false;      // 只收面板，不关生成
      applyLiveState(false);
    });
    const onOff = el('liveOn');
    if (onOff) {
      onOff.addEventListener('change', () => {
        liveState.on = !!onOff.checked;
        saveLiveState();
        applyLiveState(true);
      });
    }
    const bind = (id, key) => {
      const input = el(id);
      if (!input) return;
      input.addEventListener('input', () => {
        liveState[key] = clamp(parseFloat(input.value) || 0, 0, 1);
        saveLiveState();
        applyLiveState(true);
      });
      // 触屏拖动时别让指针锁定/摇杆抢事件
      input.addEventListener('pointerdown', (e) => e.stopPropagation());
      input.addEventListener('touchstart', (e) => e.stopPropagation(), { passive: true });
    };
    bind('liveChaos', 'chaos');
    bind('liveMemory', 'memory');
    applyLiveState(false);
    // 训练规模直接写在面板上：这版的地貌原型来自图集聚类，用户能一眼看到用了多少张
    const lp = window.PW && window.PW.liveParams;
    const meta = el('liveMeta');
    if (meta) {
      if (lp && lp.report) {
        const r = lp.report;
        meta.textContent = '地貌原型 · 图集 ' + r.samples + ' 张（训练 ' + r.train + ' / 验证 '
          + r.val + '）· ' + r.k + ' 类 · 颜色 MAE ' + r.valColorMae;
      } else {
        meta.textContent = '地貌原型未载入（js/live_params.js 缺失），退回 hash 主题';
      }
    }
  }

  /* ---------------- 启动 ---------------- */
  /* 运行环境探测 → 顶栏徽标。
     只做"读"：拿不到就照常进世界，绝不因为探测失败而卡住。 */
  async function initCaps() {
    const badge = el('pipeBadge');
    const capsApi = window.PW && window.PW.caps;
    if (!badge || !capsApi) return;
    const q = new URLSearchParams(location.search);
    const explicit = q.get('api');
    badge.textContent = '检测中…';
    try {
      const webgpuP = capsApi.probeWebGPU();
      // 没显式指定就自动找本机服务（含上次成功的地址）
      const found = await capsApi.discoverService(explicit);
      const service = found.found || { url: explicit || '', ok: false, reason: '未发现本机服务' };
      const webgpu = await webgpuP;
      const v2ok = service.ok && service.reconstruct && service.reconstruct.ok;
      /* v3 是离线烘焙链路：它不需要本机 GPU 服务，只要有预烘资产就"就绪"。
         ?pipe=v3（启动器默认带上）时优先报 v3，别再把徽标写成 v2 —— 用户看到
         "v2 就绪"会以为脚本启动的是旧链路（实测就是这么反馈的）。 */
      const wantV3 = q.get('pipe') === 'v3';
      if (v2ok) apiBase = service.url;
      const v = capsApi.verdict({ webgpu, service });
      if (wantV3) {
        v.level = 'v3';
        v.label = 'v3 就绪 · 离线资产';
        v.detail = 'v3 管线：离线烘焙的 world.glb（不需要本机 GPU 服务）'
          + (v2ok ? '\n本机 v2 服务也在（' + service.url + '），仅作对比用' : '');
      } else if (v2ok && !explicit) { v.label = 'v2 就绪 · 自动'; }
      window.__pwCaps = { webgpu, service, verdict: v };
      // 页脚也要写清楚链路与模型：只写"无服务器"在启用 v2 时是错的（本机服务≠云端）
      const foot = el('footPipe');
      const stat = el('footStat');
      if (foot) {
        foot.textContent = v.level === 'v3'
          ? 'v3 管线 · 离线烘焙资产（照片不出本机）'
          : (v.level === 'v2'
            ? 'v2 管线 · 本机 GPU 服务（照片不出本机）'
            : '在线链路 · 浏览器内推理（照片不上传）');
      }
      if (stat) {
        stat.textContent = v.level === 'v3'
          ? '逐像素反投影 · 双四边形连续曲面 · 扩散外推视锥外'
          : (v.level === 'v2'
            ? 'Depth-Anything-V2-Large · 长边 1540 · LDI 分层'
            : 'Depth-Anything-V2-Small · 518×518');
      }
      badge.dataset.level = v.level;
      badge.textContent = v.label;
      badge.title = v.detail
        + (v.level === 'v2' ? '' : '\n点开查看运行环境自检（WebGPU / 服务 / 建议）');
      if (document.body) document.body.dataset.pipeline = v.level;

      /* CTA 下面那行：启用与否都说清楚，并且给一个"下一步"而不是只报状态。
         没检测到服务时不要只写"不可用"—— 写清楚一条命令就能开。 */
      const note = el('v2Note');
      if (note) {
        if (v2ok) {
          note.dataset.level = 'v2';
          note.innerHTML = 'GPU 加速已开启（v2 管线）· <a href="?v2=0">改用在线链路</a>';
        } else {
          note.dataset.level = 'off';
          note.innerHTML = '想要照片级画质：起本地服务后刷新即可自动启用 · '
            + '<a href="#" id="v2How">怎么做</a>';
          const how = el('v2How');
          if (how) {
            how.addEventListener('click', (ev) => {
              ev.preventDefault();
              el('modal').classList.remove('hidden');
              renderCapReport();
            });
          }
        }
      }
    } catch (e) {
      badge.textContent = '检测失败';
      badge.dataset.level = 'v1';
    }
  }

  /* 把探测结果写进「工作原理」弹窗：使用者点开就知道自己现在跑哪条链路、
     以及为什么没跑上更好的那条。 */
  function renderCapReport() {
    const c = window.__pwCaps;
    const pipe = el('capPipe'), gpu = el('capGpu'), svc = el('capSvc');
    const advice = el('capAdvice');
    if (!c) {
      if (pipe) pipe.textContent = '探测中…（稍后再点一次）';
      return;
    }
    const v = c.verdict;
    if (pipe) {
      pipe.textContent = v.label + ' —— ' + v.detail;
      pipe.dataset.ok = v.level === 'v2' ? '1' : (v.level === 'v2-partial' ? '2' : '0');
    }
    if (gpu) {
      const g = c.webgpu;
      gpu.textContent = g.ok
        ? (g.vendor || '?') + ' / ' + (g.architecture || '?')
          + (g.device ? ' · ' + g.device : '') + ' · ' + g.features.length + ' 项特性'
        : ('不可用 —— ' + g.reason);
      gpu.dataset.ok = g.ok ? (/intel/i.test(g.vendor) ? '2' : '1') : '0';
    }
    if (svc) {
      const s = c.service;
      svc.textContent = s.ok
        ? (s.url + ' · GPU: ' + (s.gpu || '-') + ' · v2: '
           + (s.reconstruct && s.reconstruct.ok ? '可用' : (s.reason || '不可用')))
        : (s.reason || '未连接');
      svc.dataset.ok = (s.ok && s.reconstruct && s.reconstruct.ok) ? '1' : (s.ok ? '2' : '0');
    }
    if (advice) {
      if (v.level === 'v2') {
        advice.textContent = '你现在走的是 v2 链路（GPU 高分辨率深度 + LDI 分层），画质最好。';
      } else if (v.level === 'v2-partial') {
        advice.textContent = '服务在，但 v2 不可用：' + (c.service.reason || '')
          + '。按 docs/WebGPU.md 的「路线一」补齐依赖后即可启用。';
      } else if (c.webgpu.ok && /intel/i.test(c.webgpu.vendor)) {
        advice.textContent = '浏览器把 WebGPU 交给了核显（' + c.webgpu.vendor + '）。'
          + '核显上跑 GPU 推理不如直接起本地服务；想走 v2：在 Windows「设置 → 系统 → 显示 → 显卡」'
          + '里把 msedge.exe 设为「高性能」，再按 docs/WebGPU.md 起服务。';
      } else {
        advice.innerHTML = '当前是浏览器内在线链路（518² 小模型）。想要 v2 画质：'
          + '<a href="start_v2.bat" download>下载一键启动脚本</a> 双击运行即可，'
          + '或看 <a href="gpu.html" target="_blank" rel="noopener">运行环境自检</a>。';
      }
    }
  }

  function setHudPipe(level, text) {
    const e = el('hudPipe');
    if (!e) return;
    e.dataset.level = level;
    e.textContent = text;
  }

  function boot() {
    applyVersion();
    capsReady = initCaps();
    liveState = loadLiveState();
    buildGallery();
    buildCompass();
    prefetchDepthModel();
    initIntro();
    wireUI();

    const q = new URLSearchParams(location.search);
    instant = q.get('instant') === '1';
    const sx = parseFloat(q.get('x') || '0'), sz = parseFloat(q.get('z') || '0');
    const syaw = parseFloat(q.get('yaw') || '0');
    const spitch = parseFloat(q.get('pitch') || '0');
    window.__spawn = {
      x: isNaN(sx) ? 0 : sx, z: isNaN(sz) ? 0 : sz,
      yaw: isNaN(syaw) ? 0 : syaw, pitch: isNaN(spitch) ? 0 : spitch
    };
    // 侧拍 / 轨道截图用：直接设定相机（tools/orbit_shot.mjs 调用）
    /* 质检用相机：x, z, yaw, pitch，可选 eyeY 覆盖眼高（看鸟瞰/侧拍用）。
       tools/orbit_shot.mjs 与 tools/probe.mjs 都靠它把相机摆到任意角度。 */
    window.__setCam = (x, z, yaw, pitch, eyeY) => {
      if (!player || !world) return false;
      player.state.x = x; player.state.z = z;
      player.state.yaw = yaw || 0; player.state.pitch = pitch || 0;
      const groundY = world.heightAt(x, z);
      const useY = (typeof eyeY === 'number' && isFinite(eyeY)) ? eyeY : null;
      player.state.eyeOverride = useY;
      player.state.y = groundY + (useY === null ? (world.eyeHeight || 1.7) : useY);
      return true;
    };
    if (instant) document.body.classList.add('instant');
    const glbArg = q.get('glb');
    // 与 ?img= 同样只接受同源相对路径：否则会向任意外部地址发起 fetch。
    const glbUrl = (glbArg && glbArg.indexOf('//') < 0 && glbArg.indexOf('data:') !== 0)
      ? glbArg : null;
    if (glbArg && !glbUrl) console.warn('?glb= 只接受同源相对路径，已忽略：' + glbArg);
    if (glbUrl) { setTimeout(() => enterAsset(glbUrl), 100); }
    /* ?img=<同源路径> 直接进世界：批量质检（tools/multiview.mjs）用它把语料图
       一张张喂进同一条在线链路，不经过画廊/上传。只接受同源相对路径，
       远程地址会被画布污染（getImageData 抛错），且无 CORS 头时本来也读不到像素。 */
    const imgUrl = q.get('img');
    /* G4（v1.24.1）：这里要记录**流程是否真的启动了**，而不是"参数在不在"。
       旧写法下面用 `!imgUrl` 判断是否进 `?w=` 分支，于是 `?img=%ZZ&w=alpine`
       （畸形编码被忽略、什么都没启动）会把 `?w=alpine` 一并挡掉 —— 页面空白。
       现在只有真的 schedule 了 enterWorld 才算占用。 */
    let imgStarted = false;
    if (imgUrl && !glbUrl) {
      // ?img=%ZZ 这类畸形编码会让 decodeURIComponent 抛 URIError 并中断 boot()
      // （__pw 不赋值、?w= 分支不执行）。降级为忽略该参数，保住其余启动流程。
      let src = null;
      try { src = decodeURIComponent(imgUrl); }
      catch (e) { console.warn('?img= 含非法百分号编码，已忽略：' + imgUrl); }
      const sameOrigin = !!src && src.indexOf('//') < 0 && src.indexOf('data:') !== 0;
      if (sameOrigin) {
        const nm = q.get('name') || src.split('/').pop().replace(/\.[a-z0-9]+$/i, '');
        imgStarted = true;
        /* P0-1：?pipe=v3 让 ?img= 走离线 v3 GLB，而不是服务端 v2 六层资产。
           约定路径 out/v3/worlds/<name>/world.glb（与 tools/v3_sweep.py / start_v3.bat
           的烘焙目录一致）。没有预烘资产时自动回退在线 v2，绝不把页面卡死。 */
        if (q.get('pipe') === 'v3') {
          const v3glb = 'out/v3/worlds/' + encodeURIComponent(nm) + '/world.glb';
          fetch(v3glb, { method: 'HEAD' }).then((r) => {
            if (r.ok) enterAsset(v3glb);
            else {
              console.warn('?pipe=v3 找不到 ' + v3glb + '，回退 v2 在线路径');
              enterWorld({ id: 'corpus:' + nm, name: nm, sub: '语料 · 批量质检', src: src });
            }
          }).catch(() => {
            console.warn('?pipe=v3 探测失败，回退 v2 在线路径');
            enterWorld({ id: 'corpus:' + nm, name: nm, sub: '语料 · 批量质检', src: src });
          });
        } else {
          setTimeout(() => enterWorld({ id: 'corpus:' + nm, name: nm, sub: '语料 · 批量质检', src: src }), 120);
        }
      } else if (src) {
        console.warn('?img= 只接受同源相对路径，已忽略：' + src);
      }
    }
    // 一条 URL 只启动一个 enter 流程：旧代码 ?glb=&w= / ?img=&w= 会各自
    // setTimeout 并交叠创建 world/player 与 rAF 链，先建的 GL 资源直接泄漏。
    const pollV3 = q.get('pipe') === 'v3';
    const wid = q.get('w');
    if (wid && !glbUrl && !imgStarted) {
      const def = WORLDS.find((w) => w.id === wid);
      if (def) {
        /* ?pipe=v3：内置世界优先走**预烘的 v3 资产**（assets/world-v3-<id>.glb），
           没有就退回 v2 在线链路。这样"用启动器打开 → 点世界"进去的就是 v3。 */
        if (pollV3 && def.id) {
          const v3glb = 'assets/world-v3-' + def.id + '.glb';
          fetch(v3glb, { method: 'HEAD' }).then((r) => {
            if (r.ok) enterAsset(v3glb);
            else {
              console.warn('?pipe=v3 没有 ' + v3glb + '，回退 v2 在线路径');
              enterWorld(def);
            }
          }).catch(() => enterWorld(def));
        } else {
          setTimeout(() => enterWorld(def), 120);
        }
      }
    }
    window.__pw = {
      enterWorld, WORLDS, version: VERSION,
      get live() { return Object.assign({}, liveState, world && world.extender ? { ext: world.extender.getLive() } : {}); },
      setLive(p) {
        if (typeof p.chaos === 'number') liveState.chaos = clamp(p.chaos, 0, 1);
        if (typeof p.memory === 'number') liveState.memory = clamp(p.memory, 0, 1);
        if (typeof p.on === 'boolean') liveState.on = p.on;
        saveLiveState(); applyLiveState(true);
        return liveState;
      },
      get world() { return world; },
      get renderer() { return renderer; },
      get player() { return player; },
      get extender() { return world && world.extender; }
    };
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
