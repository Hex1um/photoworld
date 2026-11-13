/* 应用层：画廊 / 载入 / 主循环 / HUD */
(function () {
  'use strict';

  const { hashString, clamp } = window.PW.util;

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
    intro = window.PW.createIntro(cv);
    setIntroPhoto(WORLDS[0].thumb);
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
  function setProgress(p, text) {
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

  async function enterWorld(def) {
    if (running) stopLoop();
    loader.classList.remove('hidden', 'done', 'burst');
    pause.classList.add('hidden');
    el('loaderImg').src = def.thumb || def.src;
    el('loaderTitle').textContent = '正在重建场景';
    setProgress(6, '读取照片…');

    try {
      const img = await loadImage(def.src);
      setProgress(24, '分析天际线与光位…');
      await sleep(60);

      const analysis = window.PW.reconstruct.analyze(img, def.name);
      setProgress(52, '构建分层地形…');
      await sleep(50);

      ensureRenderer();

      // 用预训练深度模型把照片还原成三维表面（失败则回退到分层方案）
      let depthMap = null;
      try {
        setProgress(48, '加载深度模型（27MB，仅首次）…');
        await window.PW.depth.init((p) => {
          setProgress(48 + p * 16, '加载深度模型 ' + Math.round(p * 100) + '%');
        });
        setProgress(66, '估计场景深度…');
        await sleep(30);
        depthMap = await Promise.race([
          window.PW.depth.estimate(img),
          new Promise((_, rej) => setTimeout(() => rej(new Error('深度估计超时')), 60000))
        ]);
      } catch (e) {
        console.warn('深度模型不可用，回退到天际线方案：', e && e.message);
      }

      setProgress(74, depthMap ? '重建三维表面…' : '构建分层地形…');
      await sleep(30);
      const w = window.PW.world.build(renderer.gl, analysis, {
        seed: hashString(def.id + '|' + def.name),
        depth: depthMap
      });
      setProgress(84, '铺设地表与植被…');
      await sleep(50);

      w.pointsMesh = renderer.makePointMesh(w.particles);
      setProgress(92, '校准空气与雾…');
      await sleep(60);

      if (player) player.dispose();
      world = w;
      current = def;
      player = window.PW.createPlayer(canvas, world);
      if (window.__spawn) player.reset(window.__spawn.x, window.__spawn.z, window.__spawn.yaw);
      else if (def.start) player.reset(def.start.x || 0, def.start.z || 0, def.start.yaw || 0);

      // 入景前先做一次曝光收敛：避免进入后画面亮度跳变
      try {
        const aspect0 = Math.max(0.4, canvas.clientWidth / Math.max(1, canvas.clientHeight));
        const hf = (66 * Math.PI / 180) * player.state.fovMul;
        const vf = clamp(2 * Math.atan(Math.tan(hf / 2) / aspect0), 0.55, 1.42);
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

      // 入景前先做一次曝光收敛：避免进入后画面亮度跳变
      try {
        const aspect0 = Math.max(0.4, canvas.clientWidth / Math.max(1, canvas.clientHeight));
        const hf = (66 * Math.PI / 180) * player.state.fovMul;
        const vf = clamp(2 * Math.atan(Math.tan(hf / 2) / aspect0), 0.55, 1.42);
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

      el('hudName').textContent = def.name;
      el('hudSub').textContent = (def.sub || 'photoworld');
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
  }

  /* ---------------- 主循环 ---------------- */
  function fovRad() {
    const aspect = Math.max(0.4, canvas.clientWidth / Math.max(1, canvas.clientHeight));
    const hfov = (66 * Math.PI / 180) * (player ? player.state.fovMul : 1);
    const v = 2 * Math.atan(Math.tan(hfov / 2) / aspect);
    return clamp(v, 0.55, 1.42);
  }

  function updateParticles(dt) {
    const p = world.particles;
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

    document.querySelectorAll('.actions [data-act]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const act = btn.dataset.act;
        if (act === 'gallery') backToGallery();
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

  /* ---------------- 启动 ---------------- */
  function boot() {
    buildGallery();
    buildCompass();
    initIntro();
    wireUI();

    const q = new URLSearchParams(location.search);
    instant = q.get('instant') === '1';
    const sx = parseFloat(q.get('x') || '0'), sz = parseFloat(q.get('z') || '0');
    const syaw = parseFloat(q.get('yaw') || '0');
    window.__spawn = { x: isNaN(sx) ? 0 : sx, z: isNaN(sz) ? 0 : sz, yaw: isNaN(syaw) ? 0 : syaw };
    if (instant) document.body.classList.add('instant');
    const wid = q.get('w');
    if (wid) {
      const def = WORLDS.find((w) => w.id === wid);
      if (def) setTimeout(() => enterWorld(def), 120);
    }
    window.__pw = { enterWorld, WORLDS, get world() { return world; }, get renderer() { return renderer; } };
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
