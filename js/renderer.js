/* 渲染器：FBO + 天穹 / 分层地形 / 植被 / 粒子 / 后期 */
(function (global) {
  'use strict';

  const { M4 } = global.PW;
  const PWGL = global.PW.GL;

  const SKIP = (typeof location !== 'undefined' ? (new URLSearchParams(location.search).get('skip') || '') : '').split(',');
  const skip = (k) => SKIP.indexOf(k) >= 0;


  function createRenderer(canvas) {
    const gl = canvas.getContext('webgl2', {
      antialias: false, alpha: false, depth: true, stencil: false,
      powerPreference: 'high-performance', preserveDrawingBuffer: false
    });
    if (!gl) throw new Error('当前浏览器不支持 WebGL2，请使用较新的 Chrome / Edge / Safari。');

    const progs = {
      main: PWGL.createProgram(gl, PWGL.VS_MAIN, PWGL.FS_MAIN),
      sky: PWGL.createProgram(gl, PWGL.VS_FULL, PWGL.FS_SKY),
      post: PWGL.createProgram(gl, PWGL.VS_FULL, PWGL.FS_POST),
      points: PWGL.createProgram(gl, PWGL.VS_POINTS, PWGL.FS_POINTS)
    };

    // 全屏四边形
    const quadBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    const state = {
      fbo: null, colorTex: null, depthBuf: null,
      probe: null, probeTex: null, probeW: 64, probeH: 36,
      width: 1, height: 1,
      dpr: Math.min(window.devicePixelRatio || 1, 1.75)
    };

    function ensureFBO(w, h) {
      if (state.fbo && state.width === w && state.height === h) return;
      if (state.fbo) {
        gl.deleteFramebuffer(state.fbo);
        gl.deleteTexture(state.colorTex);
        gl.deleteRenderbuffer(state.depthBuf);
      }
      state.width = w; state.height = h;
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

      const depth = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);

      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      state.fbo = fbo; state.colorTex = tex; state.depthBuf = depth;
    }

    function ensureProbe() {
      if (state.probe) return;
      const w = state.probeW, h = state.probeH;
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      state.probe = fbo; state.probeTex = tex;
    }

    let probeBuf = null;

    function resize() {
      const dpr = state.dpr;
      const w = Math.max(2, Math.floor(canvas.clientWidth * dpr));
      const h = Math.max(2, Math.floor(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w; canvas.height = h;
        ensureFBO(w, h);
      }
      ensureFBO(canvas.width, canvas.height);
      return { w: canvas.width, h: canvas.height };
    }

    /* ---------- 粒子网格 ---------- */
    function makePointMesh(p) {
      const mk = (arr, size) => {
        const b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
        return { buf: b, size };
      };
      return {
        aPos: mk(p.pos, 3),
        aSize: mk(p.size, 1),
        aColor: mk(p.col, 4),
        count: p.count
      };
    }

    const vp = M4.create();
    const invVP = M4.create();

    function drawSkyBackground(world, eye, aspect) {
      const p = progs.sky;
      gl.useProgram(p.prog);
      gl.disable(gl.DEPTH_TEST);
      gl.depthMask(false);
      gl.disable(gl.BLEND);
      const loc = gl.getAttribLocation(p.prog, 'aClip');
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      PWGL.setUniforms(gl, p, {
        uInvVP: invVP,
        uEye: eye,
        uTop: world.skyTop,
        uHorizon: world.skyHorizon,
        uBottom: world.fogColor,
        uSunDir: world.sunDir,
        uSunColor: world.sunColor,
        uSunPower: world.sunPower
      });
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.depthMask(true);
      gl.enable(gl.DEPTH_TEST);
    }

    function drawMesh(world, mesh, opt) {
      const p = progs.main;
      gl.useProgram(p.prog);
      PWGL.bindMesh(gl, p, mesh);
      PWGL.setUniforms(gl, p, {
        uVP: vp,
        uEye: camera.eye,
        uBackdrop: opt.backdrop || [0, 0],
        uSunDir: world.sunDir,
        uSunColor: opt.sunCol || world.sunColor,
        uSkyAmb: opt.skyAmb || world.ambient,
        uFogColor: world.fogColor,
        uFogDensity: world.lights.fogDensity,
        uFogAmount: opt.fog === undefined ? 1 : opt.fog,
        uMode: opt.mode || 0,
        uMipBias: opt.mipBias || 0.0,
        uGroundTint: world.groundTint || [0.3, 0.3, 0.3],
        uAlphaTest: opt.alphaTest || 0.3,
        uExposure: opt.exposure === undefined ? 1 : opt.exposure
      });
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, opt.tex || null);
      const tl = gl.getUniformLocation(p.prog, 'uTex');
      if (tl) gl.uniform1i(tl, 0);
      gl.drawElements(mesh.mode || gl.TRIANGLES, mesh.count, gl.UNSIGNED_INT, 0);
    }

    const camera = { eye: [0, 2, 0], yaw: 0, pitch: 0, fov: 66 * Math.PI / 180, aspect: 1 };

    function render(world, cam, opts) {
      opts = opts || {};
      camera.eye = cam.eye; camera.yaw = cam.yaw; camera.pitch = cam.pitch;
      camera.fov = cam.fov || camera.fov;
      const { w, h } = resize();
      const aspect = w / h;
      camera.aspect = aspect;

      M4.perspectiveFromCamera(vp, cam.eye, cam.yaw, cam.pitch, camera.fov, aspect, 0.25, 900);
      M4.invert(invVP, vp);

      gl.bindFramebuffer(gl.FRAMEBUFFER, state.fbo);
      gl.viewport(0, 0, w, h);
      gl.clearColor(world.fogColor[0], world.fogColor[1], world.fogColor[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.BACK);

      drawSkyBackground(world, cam.eye, aspect);

      // 360° 低频世界幕：补上照片没覆盖到的方向与断崖跳面留下的洞
      // （coarse scaffold / world proxy 思路）。跟随相机平移，远景不该有视差。
      if (world.meshes.worldBackdrop && !skip('bg')) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(true);
        gl.disable(gl.CULL_FACE);
        drawMesh(world, world.meshes.worldBackdrop, {
          tex: world.tex.worldBackdrop, mode: 4, fog: 0.55, mipBias: 1.2,
          backdrop: [cam.eye[0], cam.eye[2]]
        });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }

      // 深度重建的远景层：>70m 的四边形，跟随相机平移（无视差），
      // 与 GLB 资产模式的 backdrop 同一策略，避免走动时整片照片跟着人滑。
      // 先画远层、再画近层，深度测试处理遮挡。
      if (world.meshes.reliefBackdrop && !skip('shell')) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(true);
        gl.disable(gl.CULL_FACE);
        drawMesh(world, world.meshes.reliefBackdrop, {
          tex: world.tex.photo, mode: world.vertexColor ? 1 : 4, fog: 0.26, mipBias: 0,
          backdrop: [cam.eye[0], cam.eye[2]]
        });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }

      // 背景补洞层：断崖跳面留下的洞由它填（低频模糊照片 = 零依赖 inpainting 替身），
      // 先于前景渲染，depth test 保证只在洞里露出来。
      if (world.meshes.reliefBg && !skip('shell')) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(true);
        gl.disable(gl.CULL_FACE);
        drawMesh(world, world.meshes.reliefBg, {
          tex: world.tex.backdrop, mode: 4, fog: 0.30, mipBias: 1.0
        });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }

      // 深度重建的三维表面（照片即场景）
      if (world.meshes.relief && !skip('shell')) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(true);
        gl.disable(gl.CULL_FACE);
        drawMesh(world, world.meshes.relief, {
          tex: world.tex.photo, mode: world.vertexColor ? 1 : 4, fog: 0.26, mipBias: 0,
          backdrop: [0, 0]
        });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }

      // 视锥外的程序化延续：地平雾带（跟随相机，纯粹是"推测"的远景）
      if (world.meshes.horizon && !skip('ring')) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.disable(gl.CULL_FACE);
        drawMesh(world, world.meshes.horizon, {
          tex: null, mode: 1, fog: 0.5, mipBias: 0,
          backdrop: [cam.eye[0], cam.eye[2]]
        });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }

      // 远景层：无视差背景（跟随相机平移），避免"沙雕沙盘"感
      if (world.meshes.backdrop) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.disable(gl.CULL_FACE);
        drawMesh(world, world.meshes.backdrop, {
          tex: world.tex.photo, mode: world.vertexColor ? 1 : 4, fog: 0.30, mipBias: 0,
          backdrop: [cam.eye[0], cam.eye[2]]
        });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }

      // 天穹（照片天空）
      if (world.meshes.sky) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      gl.disable(gl.CULL_FACE);
      if (!skip('sky')) drawMesh(world, world.meshes.sky, { tex: world.tex.sky, mode: 3, fog: 0, exposure: 1, mipBias: 0.4 });
      gl.depthMask(true);
      gl.enable(gl.CULL_FACE);
      gl.disable(gl.BLEND);
      }

      // 远山
      gl.disable(gl.CULL_FACE);
      if (world.meshes.farRidge && !skip('far')) drawMesh(world, world.meshes.farRidge, { tex: world.tex.photo, mode: 4, fog: 0.32, exposure: 1.0, mipBias: 1.9 });
      // 近岭
      if (world.meshes.nearRidge && !skip('near')) drawMesh(world, world.meshes.nearRidge, { tex: world.tex.photo, mode: 4, fog: 0.34, exposure: 0.94, mipBias: 2.2 });
      gl.enable(gl.CULL_FACE);

      // 地面
      if (world.meshes.ground && !skip('ground')) drawMesh(world, world.meshes.ground, { tex: world.tex.ground, mode: world.mode === 'depth' ? 4 : 6, fog: 0.35, mipBias: -0.1 });

      // 岩石
      gl.disable(gl.CULL_FACE);
      if (world.meshes.rocks) drawMesh(world, world.meshes.rocks, { tex: null, mode: 1, fog: 1 });
      gl.enable(gl.CULL_FACE);

      // 植被
      if (world.meshes.grass) {
        gl.disable(gl.CULL_FACE);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        drawMesh(world, world.meshes.grass, { tex: world.tex.grass, mode: 2, fog: 1, alphaTest: 0.34 });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }
      if (world.meshes.shrub) {
        gl.disable(gl.CULL_FACE);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        drawMesh(world, world.meshes.shrub, { tex: world.tex.grass, mode: 2, fog: 1, alphaTest: 0.42 });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }

      // 粒子
      if (world.pointsMesh) {
        const p = progs.points;
        gl.useProgram(p.prog);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
        gl.depthMask(false);
        const ap = gl.getAttribLocation(p.prog, 'aPos');
        const as = gl.getAttribLocation(p.prog, 'aSize');
        const ac = gl.getAttribLocation(p.prog, 'aColor');
        gl.bindBuffer(gl.ARRAY_BUFFER, world.pointsMesh.aPos.buf);
        gl.enableVertexAttribArray(ap);
        gl.vertexAttribPointer(ap, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, world.pointsMesh.aSize.buf);
        gl.enableVertexAttribArray(as);
        gl.vertexAttribPointer(as, 1, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, world.pointsMesh.aColor.buf);
        gl.enableVertexAttribArray(ac);
        gl.vertexAttribPointer(ac, 4, gl.FLOAT, false, 0, 0);
        PWGL.setUniforms(gl, p, {
          uVP: vp, uEye: cam.eye, uScale: h * 0.055,
          uFogColor: world.fogColor, uFogDensity: world.lights.fogDensity
        });
        gl.drawArrays(gl.POINTS, 0, world.pointsMesh.count);
        gl.depthMask(true);
        gl.disable(gl.BLEND);
      }

      // 后期
      gl.disable(gl.DEPTH_TEST);
      const pp = progs.post;
      gl.useProgram(pp.prog);
      const loc = gl.getAttribLocation(pp.prog, 'aClip');
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, state.colorTex);
      PWGL.setUniforms(gl, pp, {
        uScene: 0, uRes: [w, h], uTime: opts.time || 0,
        uFade: opts.fade === undefined ? 1 : opts.fade,
        uExposure: world.exposure || 1.0,
        uVignette: opts.vignette === undefined ? 0.42 : opts.vignette,
        uGrain: 0.006, uAberr: 0, uAA: 1
      });

      let measured = -1;
      if (opts.measure) {
        ensureProbe();
        gl.bindFramebuffer(gl.FRAMEBUFFER, state.probe);
        gl.viewport(0, 0, state.probeW, state.probeH);
        PWGL.setUniforms(gl, pp, { uRes: [w, h], uAA: 0 });
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        const n = state.probeW * state.probeH * 4;
        if (!probeBuf || probeBuf.length !== n) probeBuf = new Uint8Array(n);
        gl.readPixels(0, 0, state.probeW, state.probeH, gl.RGBA, gl.UNSIGNED_BYTE, probeBuf);
        let sum = 0;
        for (let i = 0; i < n; i += 4) {
          sum += (0.299 * probeBuf[i] + 0.587 * probeBuf[i + 1] + 0.114 * probeBuf[i + 2]) / 255;
        }
        measured = sum / (n / 4);
      }

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, w, h);
      PWGL.setUniforms(gl, pp, { uRes: [w, h], uAA: 1, uExposure: world.exposure || 1.0 });
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      return measured;
    }

    return {
      gl, render, resize, makePointMesh, camera,
      get size() { return { w: state.width, h: state.height }; }
    };
  }

  global.PW.createRenderer = createRenderer;
})(window);
