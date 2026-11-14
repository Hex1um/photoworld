/* 应用层：画廊 / 载入 / 主循环 / HUD */
(function () {
  'use strict';

  const { hashString, clamp } = window.PW.util;

  /* 版本号单一来源：index.html 的顶栏徽标与页脚都由这里写入，
     测试脚本读 body[data-version]，避免两处手改不同步。 */
  const VERSION = '1.20.1';

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
  let running = false;
  let lastT = 0;
  let time = 0;
  let fade = 0;
  let fadeStart = 0;
  let fadeFrames = 0;
  let expoCount = 0;
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

  async function enterWorld(def) {
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
      setProgress(8, '照片解码完成');
      await sleep(80);
      setProgress(12, '分析天空 / 地平线…');
      await sleep(60);
      const analysis = window.PW.reconstruct.analyze(img, def.name);
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
        // 流式下载：p 是真实字节进度，label 由 depth.js 给出（含 MB/s 与阶段名），
        // 没有 label 时才退回旧文案。卡在「下载深度模型 15%」就是旧实现的症状。
        await window.PW.depth.init((p, label) => {
          setProgress(32 + p * 32, label || ('下载深度模型 ' + Math.round(p * 100) + '%（27MB）'));
        });
        setProgress(65, '模型就绪，准备推理');
        await sleep(60);
        setProgress(67, '推理：估计每个像素的距离…');
        await sleep(30);
        depthMap = await Promise.race([
          window.PW.depth.estimate(img),
          new Promise((_, rej) => setTimeout(() => rej(new Error('深度估计超时')), 60000))
        ]);
        setProgress(72, '深度图后处理（分位归一化 / 极性）');
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
        yGain: params.has('yg') ? parseFloat(params.get('yg')) : undefined,
        fgMask: objInfo && objInfo.fgMask,
        // 语义分组：天空/地面/天花板/植被/建筑/人车，决定几何策略与单体卡片
        sem: sem,
        // 质检：?dbgpos=1 时把 relief 顶点位置暴露到 world.dbgPos（排查竖条/鳍）
        dbgPos: params.get('dbgpos') === '1',
        // 重建表面背面剔除：默认开（v1.17），?cull=0 回到旧的双面渲染做 A/B
        noCull: params.get('cull') === '0',
        // 地面远处渐隐到地面基色（压极坐标拉丝）：默认开，?gblend=0 做 A/B
        groundFade: params.get('gblend') !== '0',
        // 地面拓扑：默认世界坐标方格网；?gfan=1 回到旧的极坐标扇形地面（A/B）
        groundGrid: params.get('gfan') !== '1',
        // 地面渐隐区间可调（A/B 用）：?gfade=起始m,结束m,强度
        gfade: params.get('gfade') || undefined,
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

      if (player) player.dispose();
      world = w;
      current = def;
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
          if (player) player.reset(player.state.x, player.state.z, player.state.yaw);
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
    let nv = 0, ni = 0;
    for (const l of list) { nv += l.pos.length / 3; ni += l.index.length; }
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
      for (let i = 0; i < l.index.length; i++) index[io + i] = l.index[i] + vo;
      io += l.index.length;
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
      // 分层规则：名为 backdrop 的层做无视差远景；其余层合并成近景（真视差）
      const farLayers = asset.layers.filter((l) => l.name === 'backdrop');
      const nearLayers = asset.layers.filter((l) => l.name !== 'backdrop');
      const farMesh = farLayers.length ? toMesh(mergeLayers(farLayers)) : null;
      const nearMesh = nearLayers.length ? toMesh(mergeLayers(nearLayers)) : null;
      const tex = asset.img
        ? PW.GL.createTexture(gl, asset.img, { wrap: 'clamp', clampV: true, aniso: 8, flipY: false })
        : whiteTex(gl);

      // 贴地：从网格顶点里烘一张高度图（每格取最低点再平滑），行走时才不会漂在半空。
      // 取"最低"而不是"平均"，是因为分层世界里树/石/山都在地面之上，地面才是能站的那层。
      const grid = buildHeightGrid(asset.layers, 90, 2.0);
      const groundY = grid.at(0, 0);

      const w = {
        name: '离线重建',
        mode: 'asset',
        vertexColor,
        heightAt: (x, z) => grid.at(x, z),
        eyeHeight: 1.7,
        walkRadius: 55,
        fogColor: col.bot,
        skyTop: col.top,
        skyHorizon: col.bot,
        ambient: [1, 1, 1],
        sunColor: [0.9, 0.9, 0.9],
        sunPower: 0.4,
        exposure: 1.0,
        targetLum: Math.min(0.55, Math.max(0.28, col.lum * 0.82)),
        sunDir: [0.3, 0.5, -0.8],
        lights: { fogDensity: 0.0016 },
        tex: { photo: tex, ground: null, sky: null, grass: null },
        meshes: { relief: nearMesh, backdrop: farMesh, farRidge: null, nearRidge: null, sky: null, ground: null, rocks: null, grass: null, shrub: null },
        particles: null,
        pointsMesh: null,
        analysis: { horizon: 0.5, meanLum: col.lum }
      };

      if (player) player.dispose();
      world = w;
      current = { name: '离线重建', sub: 'asset · world.glb' };
      player = PW.createPlayer(canvas, world);
      if (window.__spawn) player.reset(window.__spawn.x, window.__spawn.z, window.__spawn.yaw);
      else player.reset(0, 0, 0);

      el('hudName').textContent = '离线重建世界';
      el('hudSub').textContent = asset.triangles.toLocaleString() + ' 三角 · ' + (asset.bytes / 1048576).toFixed(1) + 'MB · ' + asset.layers.length + ' 层';
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

  function frame(now) {
    if (!running) return;
    const dt = Math.min((now - lastT) / 1000 || 0.016, 0.05);
    lastT = now;
    time += dt;

    player.update(dt);
    if (world.extender) {
      world.extender.update(player.state.x, player.state.z, now,
        { yaw: player.state.yaw, fov: fovRad() });
    }
    updateParticles(dt);
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
    el('btnAbout').addEventListener('click', () => el('modal').classList.remove('hidden'));
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
        if (act === 'live') { liveState.on = !liveState.on; saveLiveState(); applyLiveState(true); }
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
    const run = () => { window.PW.depth.init(() => {}).catch(() => {}); };
    if (window.requestIdleCallback) window.requestIdleCallback(run, { timeout: 5000 });
    else setTimeout(run, 2600);
  }

  /* ---------------- 实时变化（视锥生成 + 混乱度 / 记忆力） ---------------- */
  const LIVE_KEY = 'pw.live';

  function loadLiveState() {
    /* v1.19：默认开启。以前默认关着，扇环外只有一套全局 fBm —— 转身看到的是
       同一套地貌，用户读作「重复生成」。现在默认按扇区主题生成（混乱度 0.45），
       按钮只是用来关掉/调参，而不是「开启才生成」。 */
    let st = { on: true, chaos: 0.45, memory: 0.85 };
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

  let liveState = { on: true, chaos: 0.45, memory: 0.85 };

  function saveLiveState() {
    try { window.localStorage.setItem(LIVE_KEY, JSON.stringify(liveState)); } catch (e) { /* ignore */ }
  }

  /* 面板是否展开与「实时生成是否开启」分开：默认生成开着、面板收起。
     否则竖屏上一进世界就被面板挡住半个屏幕（实测 480×1040 挡掉 40%）。 */
  let livePanelOpen = false;

  /* 把状态同步到 extender / DOM / body.dataset（测试用） */
  function applyLiveState(push) {
    const ext = world && world.extender;
    if (push && ext && ext.setLive) ext.setLive(liveState);
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
  function boot() {
    applyVersion();
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
    const glbUrl = q.get('glb');
    if (glbUrl) { setTimeout(() => enterAsset(glbUrl), 100); }
    /* ?img=<同源路径> 直接进世界：批量质检（tools/multiview.mjs）用它把语料图
       一张张喂进同一条在线链路，不经过画廊/上传。只接受同源相对路径，
       远程地址会被画布污染（getImageData 抛错），且无 CORS 头时本来也读不到像素。 */
    const imgUrl = q.get('img');
    if (imgUrl && !glbUrl) {
      const src = decodeURIComponent(imgUrl);
      const sameOrigin = src.indexOf('//') < 0 && src.indexOf('data:') !== 0;
      if (sameOrigin) {
        const nm = q.get('name') || src.split('/').pop().replace(/\.[a-z0-9]+$/i, '');
        setTimeout(() => enterWorld({ id: 'corpus:' + nm, name: nm, sub: '语料 · 批量质检', src: src }), 120);
      } else {
        console.warn('?img= 只接受同源相对路径，已忽略：' + src);
      }
    }
    const wid = q.get('w');
    if (wid) {
      const def = WORLDS.find((w) => w.id === wid);
      if (def) setTimeout(() => enterWorld(def), 120);
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
