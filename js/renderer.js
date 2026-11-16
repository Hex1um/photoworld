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
      points: PWGL.createProgram(gl, PWGL.VS_POINTS, PWGL.FS_POINTS),
      splat: PWGL.createProgram(gl, PWGL.VS_SPLAT, PWGL.FS_SPLAT),
      points2: PWGL.createProgram(gl, PWGL.VS_POINTS2, PWGL.FS_POINTS2)
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
        uDebugCol: opt.debugCol || [1, 0, 1],
        uMipBias: opt.mipBias || 0.0,
        uGroundTint: world.groundTint || [0.3, 0.3, 0.3],
        uAlphaTest: opt.alphaTest || 0.3,
        uExposure: opt.exposure === undefined ? 1 : opt.exposure
      });
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, opt.tex || null);
      const tl = gl.getUniformLocation(p.prog, 'uTex');
      if (tl) gl.uniform1i(tl, 0);
      // 无索引网格（例如深度模型失败回退时的 rocks）必须走 drawArrays；
      // 否则 ELEMENT_ARRAY_BUFFER 绑定为 null 还调 drawElements 会每帧报
      // INVALID_OPERATION，几何整个不显示。
      if (mesh.index) gl.drawElements(mesh.mode || gl.TRIANGLES, mesh.count, gl.UNSIGNED_INT, 0);
      else gl.drawArrays(mesh.mode || gl.TRIANGLES, 0, mesh.count);
    }

    /* 点云 / surfel 绘制（v1.24.7）：照片像素反投影成点，每点带自身 UV 与顶点色。
       没有三角形连接，所以不存在「沿射线拉成长条/竖幕」这种 mesh 病。 */
    function drawSplat(world, mesh, opt) {
      if (!mesh || !mesh.aPos) return;
      const p = progs.splat;
      gl.useProgram(p.prog);
      PWGL.bindMesh(gl, p, mesh);
      PWGL.setUniforms(gl, p, {
        uVP: vp,
        uEye: camera.eye,
        uScale: opt.scale || 30,
        uTex: 0,
        uFogColor: world.fogColor,
        uFogDensity: world.lights.fogDensity,
        uFogAmt: opt.fog === undefined ? 0.5 : opt.fog,
        uMipBias: opt.mip || 0
      });
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, opt.tex || null);
      const tl = gl.getUniformLocation(p.prog, 'uTex');
      if (tl) gl.uniform1i(tl, 0);
      gl.drawArrays(gl.POINTS, 0, mesh.count);
    }

    /* 资产点云（v2.1）：world.points2Mesh = { aPos, aColor, count }，点尺寸由
       world.points2Px 给（屏幕像素，与分辨率无关）。用于"细枝/树叶"这类
       网格必然剪碎的内容。 */
    function drawPoints2(world, mesh, opt) {
      if (!mesh || !mesh.aPos) return;
      const p = progs.points2;
      gl.useProgram(p.prog);
      const ap = gl.getAttribLocation(p.prog, 'aPos');
      const ac = gl.getAttribLocation(p.prog, 'aColor');
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.aPos.buf);
      gl.enableVertexAttribArray(ap);
      gl.vertexAttribPointer(ap, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.aColor.buf);
      gl.enableVertexAttribArray(ac);
      gl.vertexAttribPointer(ac, 4, gl.FLOAT, false, 0, 0);
      // 每点的原始距离（没有就统一给 0，着色器里回退成恒定像素尺寸）
      const asz = gl.getAttribLocation(p.prog, 'aSize');
      if (asz >= 0) {
        if (mesh.aSize) {
          gl.bindBuffer(gl.ARRAY_BUFFER, mesh.aSize.buf);
          gl.enableVertexAttribArray(asz);
          gl.vertexAttribPointer(asz, 1, gl.FLOAT, false, 0, 0);
        } else {
          gl.disableVertexAttribArray(asz);
          gl.vertexAttrib1f(asz, 0);
        }
      }
      PWGL.setUniforms(gl, p, {
        uVP: vp, uEye: camera.eye,
        // 点尺寸按画布高度缩放：点是"图像采样点"，屏幕尺寸应随分辨率等比放大，
        // 否则高分辨率屏上会漏缝、低分辨率屏上会糊成一片。
        uPx: (opt.px || 3.2) * ((gl.drawingBufferHeight || 800) / 800),
        uFogColor: world.fogColor,
        uFogDensity: world.lights.fogDensity,
        uFogAmt: opt.fog === undefined ? 0.3 : opt.fog
      });
      gl.drawArrays(gl.POINTS, 0, mesh.count);
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
      /* 点云模式下相机脚下常常没有几何（照片投影范围外），底部会露出清屏色
         —— 手机竖屏最下方那条灰带。用地面色清屏，它就读成「远处地面」。 */
      if (world.splatMode && world.groundTint) {
        const g = world.groundTint;
        gl.clearColor(g[0] * 0.85, g[1] * 0.85, g[2] * 0.85, 1);
      } else {
        gl.clearColor(world.fogColor[0], world.fogColor[1], world.fogColor[2], 1);
      }
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
      if (world.meshes.reliefBackdrop && !skip('shell') && !skip('backdrop')) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(true);
        /* v1.19.1：远景层也是照片衍生的**单面薄壳**。以前这里 disable(CULL_FACE)，
           转身面对它背面时，整片远景会被摊成一层层漂浮的横向薄片（32 视角质检里
           「断片/拉丝」最重的一类，stripe≈3.1）。剔除背面后它只在正对照片时出现。 */
        gl.enable(gl.CULL_FACE);
        drawMesh(world, world.meshes.reliefBackdrop, {
          tex: world.tex.photo, mode: world.vertexColor ? 1 : 4, fog: 0.26, mipBias: 0,
          backdrop: [cam.eye[0], cam.eye[2]]
        });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }

      // 侧面 / 背面：低分辨率镜像副本，±90° 与 180° 各一份（v1.14）
      if (world.meshes.reliefSurround && !skip('shell') && !skip('surround')) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(true);
        gl.disable(gl.CULL_FACE);
        drawMesh(world, world.meshes.reliefSurround, {
          tex: world.tex.photo, mode: world.vertexColor ? 1 : 4, fog: 0.65, mipBias: 1.4
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
        gl.enable(gl.CULL_FACE);      // 同理：补洞层也是单面薄壳
        drawMesh(world, world.meshes.reliefBg, {
          tex: world.tex.backdrop, mode: 4, fog: 0.30, mipBias: 1.0
        });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }

      /* 区域内底板（v1.16.1）：铺在重建表面**之前**、y=-0.06。
         重建网格在断崖/竖幕处被剪开时后面不再是天穹灰色，而是照片地面色。
         正常几何会把它完全遮住，所以只在「洞」里可见，等于一层兜底。 */
      if (world.meshes.floor && !skip('floor')) {
        gl.disable(gl.CULL_FACE);
        drawMesh(world, world.meshes.floor, {
          // mipBias 2.6 会把底板糊成一块灰（手机端画面最下方那条灰带），收到 1.0。
          // tex/mode 允许世界覆盖：资产世界的底板必须走**顶点色**（mode 1），
          // 因为底板的 UV 是 x/40 这种平铺坐标，拿去采样照片贴图会取到天空 →
          // 一道蓝色拉丝横插进画面（实测）。在线链路保持原样（贴图 × 顶点色）。
          tex: (world.floorTex === undefined ? world.tex.ground : world.floorTex),
          mode: (world.floorMode === undefined ? 4 : world.floorMode),
          fog: 0.42, mipBias: 1.0
        });
        /* 近景补片：竖向视场比照片大时，照片底边以下那条地面靠它提供细节，
           否则是一块糊平的灰板（底板 UV 放大 40m + mipBias 2.6）。 */
        if (world.meshes.nearGround) {
          drawMesh(world, world.meshes.nearGround, {
            tex: (world.floorTex === undefined ? world.tex.ground : world.floorTex),
            mode: (world.floorMode === undefined ? 4 : world.floorMode),
            fog: 0.40, mipBias: 0.2
          });
        }
        gl.enable(gl.CULL_FACE);
      }

      /* 地面裙边（v2.7.1）：顶点色、不吃贴图。它铺在照片区域之外，
         给"视锥外"一块有颜色的地面；用照片贴图会整片偏色（UV 只指向一个像素）。 */
      if (world.meshes.ground && !world.showPlan) {
        gl.disable(gl.CULL_FACE);
        /* fog 给满（1.0）：400m 外那条边缘如果只吃到 55% 的雾，会和天空的
           horizon 色差出一条明显的白脊（实测背视图远处那条亮线）。
           满雾之后远端正好等于 fogColor，与 skyHorizon 同源，接缝自然消失。 */
        drawMesh(world, world.meshes.ground,
          world.dbgLayers ? { mode: 9, debugCol: [0.2, 0.9, 0.3] }
            : { tex: null, mode: 1, fog: 1.0, mipBias: 0 });
        gl.enable(gl.CULL_FACE);
      }

      /* 近景封闭体（v2.12）：每个物体的正面+镜像背面+封边。
         必须关掉背面剔除 —— 绕到物体后面时，看到的是它自己的背面，
         而不是一个洞（这正是"物体不该只朝相机一张薄膜"的最小实现）。 */
      if (world.meshes.solid) {
        gl.disable(gl.CULL_FACE);
        /* 恒用 mode 4（贴图 × 顶点色）：实体层没有自己的顶点色，
           merge 时会填 1.0，所以效果就是"贴该物体自己的照片像素"。
           用 mode 1 会变成纯白板（实测侧视图里那几块浅灰片）。 */
        drawMesh(world, world.meshes.solid, {
          tex: world.tex.photo, mode: 4, fog: 0.26, mipBias: 0
        });
        gl.enable(gl.CULL_FACE);
      }

      /* 俯视顶面（v2.6）：点云生成的"正俯视高度图"，只在视点高于地形最高点时显示。
         作用是把"从上方看到的单层壳背面（竖直拉丝 + 白洞）"盖住 ——
         高空中看到的是同一片地貌的俯视图，纹理仍是照片自己的。 */
      /* 显示条件：视点必须**明显高于脚下的地面**（+6m），而不只是"高于顶面最高点"。
         平地的顶面高度接近 0，早先的判据（顶面最高点+1m）在站姿下也成立，
         于是这片水平顶面在地平线处露成一条亮带（实测：背后那条白带就是它，
         去掉裙边、去掉顶面后就消失 —— 靠这个对照才定位到）。 */
      const topLift = world.heightAt ? cam.eye[1] - world.heightAt(cam.eye[0], cam.eye[2]) : 1e9;
      if (world.meshes.top && (world.showPlan || topLift > 6.0)) {
        gl.disable(gl.CULL_FACE);
        /* mipBias 给到 2.6：单视图重建**没有俯视纹理**，顶面 UV 是从相邻像素借来的，
           贴着看会是一圈圈放射状拉伸。高空看地表本来也该是低频的，
           干脆重度模糊 —— 读作"远处地表"而不是"糊掉的照片"。 */
        /* 顶点色（mode 1），不再采样照片贴图：那个 UV 是从邻格借的，
           贴着看是一圈圈放射状拉伸。 */
        /* 俯视时地面法线朝上、而这张照片太阳很低（ndl≈0）→ 不补光是一团暗褐；
           补太狠（1.7）整屏发白。1.3 是实测两端之间的值。
           fog 1.0 让网格边界正好等于雾色，边缘不再是一条硬边。 */
        /* skyAmb 现在给 0.8：法线已经按高度场梯度算过，坡面自带明暗，
           不再需要靠补光去"提亮"——1.3 会整片发白（这就是"起飞地面变全白"）。
           fog 0.55 而不是 1.0：1.0 把远处压成一块均匀灰，反而失去层次。 */
        drawMesh(world, world.meshes.top, {
          tex: null, mode: 1, fog: 0.55, mipBias: 0,
          skyAmb: [0.8, 0.78, 0.74]
        });
        gl.enable(gl.CULL_FACE);
      }

      /* 世界坐标地面方格网（v1.20）：先画地面，再画照片表面。
         地面的几何与照片平面完全重合（都是 y=0），只是拓扑换成规则方格，
         贴图 UV 取自反投影场 —— 从任何角度看都没有放射状拉丝。 */
      /* 点云模式下不再画地面方格网：手机低角度那张「放射长条」就是它的
         近处三角形沿视线拉长、从点云缝隙里透出来的。地面交给点云自己覆盖。 */
      if (world.meshes.groundGrid && !skip('ground') && (!world.splatMode || world.snowGround)) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(true);
        gl.disable(gl.CULL_FACE);
        drawMesh(world, world.meshes.groundGrid, {
          tex: world.tex.ground, mode: 4, fog: 0.26, mipBias: 0.2
        });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }

      /* 照片 UV 地面（v1.26.5 起独立绘制）：点云模式下也必须画。
         点云地面是 step2 的稀疏点，只靠它 + floor 底板，整片地面会退成
         「一整块程序化色板」——24 张批量里最差那几张（粉紫峡谷平板、
         全绿草原平板）就是这个。mipBias 2.2 = 只取照片的低频颜色，
         既保住草/砂/水的地域色调，又不产生掠射拉丝。 */
      if (world.meshes.groundGridPhoto && !skip('ground') && !world.snowGround) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(true);
        gl.disable(gl.CULL_FACE);
        drawMesh(world, world.meshes.groundGridPhoto, {
          tex: world.tex.photo, mode: 4, fog: 0.26, mipBias: 2.2, backdrop: [0, 0]
        });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }

      /* 资产点云（v2.1）：GLB 里带了 POINTS 层时走这里。
         **必须放在 relief 分支之外** —— 点云世界没有 relief 网格，
         塞进那个 else-if 里会整块被跳过（实测表现：只画出天空，点全都不见）。
         网格在"细枝/树叶"这类几像素跨 30 倍深度的内容上必然被剪碎；
         点云没有"连接"，既不拉丝也不会被剪碎。 */
      if (world.points2Mesh && !skip('shell')) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(true);
        gl.disable(gl.CULL_FACE);
        drawPoints2(world, world.points2Mesh, { px: world.points2Px || 3.2 });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }

      // 深度重建的三维表面（照片即场景）
      if (world.meshes.flat && !skip('shell')) {
        /* 夜景 / 大面积水面：深度是噪声，不硬重建，贴回「照片墙」。 */
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(true);
        gl.disable(gl.CULL_FACE);
        drawMesh(world, world.meshes.flat, {
          tex: world.tex.photo, mode: 4, fog: 0.12, mipBias: 0
        });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      } else if (world.meshes.relief && !skip('shell')) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(true);
        /* v1.17：默认剔除背面。单层薄壳的双面渲染在侧后方是「糊纸片」，
           读起来比雾还假；剔除后露出低频世界幕 + 雾，观感连续。
           混合与剔除可以并存（alpha 只用于天空渐隐），不影响正面边缘。
           ?cull=0 可回到 v1.16 的双面行为做 A/B。 */
        if (world.splatMode && world.meshes.splat) {
          /* LDI 补洞层（?ldi=2）：先用 MI-GAN 补出的背景点垫在后面，
             再画原照片点云 —— 侧移时前景移开，背后不是洞。 */
          if (world.meshes.splatHole && world.tex.bg) {
            drawSplat(world, world.meshes.splatHole, {
              tex: world.tex.bg, fog: 0.5, scale: 34
            });
          }
          /* 点云路径（?splat=1）：三角形网格在掠射角会拉丝、单层壳侧面是竖幕，
             点云没有「连接」，天然不会出现这两类病。 */
          if (world.meshes.splatGround && world.meshes.splatBody) {
            /* 层化：地面用 mip 2.2 的低频照片色（掠射不拉丝），主体保持原图清晰。 */
            drawSplat(world, world.meshes.splatGround, {
              tex: world.tex.photo, fog: 0.5, scale: world.splatScale || 30, mip: 2.2
            });
            drawSplat(world, world.meshes.splatBody, {
              tex: world.tex.photo, fog: 0.5, scale: world.splatScale || 30, mip: 0
            });
          } else {
            drawSplat(world, world.meshes.splat, {
              tex: world.tex.photo, fog: 0.5, scale: world.splatScale || 30
            });
          }
        } else {
          const cullRelief = world.reliefCull !== false;
          if (cullRelief) { gl.enable(gl.CULL_FACE); gl.cullFace(gl.BACK); }
          else gl.disable(gl.CULL_FACE);
          /* 高空时不画照片壳（v2.7.2）：它是一层"朝前的薄膜"，从上方看到的是
             竖直拉长的崖壁穿出顶面，表现为一片黑红竖条（实测 150m 俯视）。
             超过 6m 就切成"只画顶面 + tile"的俯视模式 —— 这是显式的 LOD 切换，
             不是几何修补：单视图重建本来就没有从上看的样子。 */
          if (world.showPlan) {
            // 高空：跳过照片壳（顶面与 tile 已经铺满）
          } else {
            drawMesh(world, world.meshes.relief,
              world.dbgLayers
                ? { mode: 9, debugCol: [0.95, 0.2, 0.2] }
                : {
                  tex: world.tex.photo, mode: world.vertexColor ? 1 : 4, fog: 0.26, mipBias: 0,
                  backdrop: [0, 0]
                });
          }
        }
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }

      /* 照片底行向下的裙边（v1.19.1）：几何上延续照片网格，UV 钳到最后一行。
         竖屏相机竖直视场比照片大时，它能接住"照片底边以下"那条地面，
         不再露出 y=-0.05 平面的断口与色差。画在 relief 之后，深度测试负责遮挡。 */
      if (world.meshes.skinSkirt && !skip('shell')) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(true);
        gl.disable(gl.CULL_FACE);
        drawMesh(world, world.meshes.skinSkirt, {
          tex: world.tex.ground, mode: 4, fog: 0.30, mipBias: 0.2, backdrop: [0, 0]
        });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }

      // 单体十字账单牌（v1.16）：照片贴图，双面渲染，替代被拉成鳍的对象四边形
      if (world.meshes.objectCards && !skip('cards')) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(true);
        gl.disable(gl.CULL_FACE);
        drawMesh(world, world.meshes.objectCards, {
          tex: world.tex.photo, mode: 4, fog: 0.26, mipBias: 0
        });
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      }

      // 扇环延伸 tile（js/extend.js）：真实几何，随玩家前进逐块生成
      if (world.extraMeshes && world.extraMeshes.length && !world.showPlan) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.disable(gl.CULL_FACE);
        for (const t of world.extraMeshes) {
          if (!t.mesh) continue;
          /* mode 由 tile 自己决定（v1.18）：陡坡用 mode 1（顶点色，无贴图），
             平地用 mode 4（贴图 × 顶点色）。陡坡贴图在掠射/竖直面上会被拉成
             一条条竖丝 —— 36 视角质检里最刺眼的一项，改用顶点色后消失。 */
          drawMesh(world, t.mesh, {
            tex: t.tex || world.tex.ground, mode: t.mode === undefined ? 4 : t.mode,
            fog: t.fog === undefined ? 0.32 : t.fog, mipBias: t.mode === 1 ? 0 : 0.4
          });
        }
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
      if (world.meshes.backdrop && !world.showPlan) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.disable(gl.CULL_FACE);
        drawMesh(world, world.meshes.backdrop,
          world.dbgLayers
            ? { mode: 9, debugCol: [0.15, 0.35, 0.95] }
            : {
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
      if (world.meshes.ground && !skip('ground')) {
        // 有限扇环的边界：顶点 alpha 在区域边缘渐隐，融进雾里
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        drawMesh(world, world.meshes.ground, { tex: world.tex.ground, mode: world.mode === 'depth' ? 4 : 6, fog: 0.35, mipBias: -0.1 });
        gl.disable(gl.BLEND);
      }

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
