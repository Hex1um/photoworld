/* WebGL2 基础设施：程序编译、纹理、网格上传、着色器源码 */
(function (global) {
  'use strict';

  const VS_MAIN = `#version 300 es
precision highp float;
in vec3 aPos;
in vec3 aNormal;
in vec2 aUV;
in vec4 aColor;
in vec2 aEdge;      // x: 视锥侧向融雾  y: 底边融入地面色
uniform mat4 uVP;
uniform vec3 uEye;
uniform vec2 uBackdrop;   // 远景跟随相机的水平位移（近景不跟随，保证走动真实）
out vec3 vN;
out vec2 vUV;
out vec4 vC;
out float vDist;
out vec3 vWorld;
out vec2 vEdge;
void main() {
  vec3 wp = aPos;
  // 18m 内严格世界坐标（走动真实）；42m 外整体跟随相机（远景视差可忽略，避免穿帮）
  wp.xz += uBackdrop * smoothstep(18.0, 42.0, length(aPos.xz));
  gl_Position = uVP * vec4(wp, 1.0);
  vWorld = wp;
  vN = aNormal;
  vUV = aUV;
  vC = aColor;
  vEdge = aEdge;
  vDist = length(wp - uEye);
}`;

  const FS_MAIN = `#version 300 es
precision highp float;
in vec3 vN;
in vec2 vUV;
in vec4 vC;
in float vDist;
in vec3 vWorld;
in vec2 vEdge;
uniform sampler2D uTex;
uniform vec3 uEye;
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform vec3 uSkyAmb;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform float uFogAmount;
uniform float uMode;
uniform float uAlphaTest;
uniform float uExposure;
uniform float uMipBias;
uniform vec3 uGroundTint;
out vec4 fragColor;

vec3 srgb2lin(vec3 c) {
  return pow(max(c, vec3(0.0)), vec3(2.2));
}

float hash21(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash21(i), b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0)), d = hash21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

void main() {
  int mode = int(uMode + 0.5);
  vec3 col;
  float alpha = vC.a;

  if (mode == 3) {
    // 天空穹顶：贴图直出，顶端用顶点 alpha 融进渐变
    col = srgb2lin(texture(uTex, vUV, uMipBias).rgb) * vC.rgb;
    if (alpha < 0.004) discard;
    fragColor = vec4(col * uExposure, alpha);
    return;
  }

  vec3 n = normalize(vN);
  float ndl = max(dot(n, uSunDir), 0.0);
  float fill = 0.5 + 0.5 * n.y;

  if (mode == 2) {
    // 植被：alpha test + 半兰伯特
    vec4 t = texture(uTex, vUV, uMipBias);
    if (t.a < uAlphaTest) discard;
    vec3 albedo = srgb2lin(t.rgb) * vC.rgb;
    col = albedo * (uSkyAmb * 0.85 + uSunColor * (0.4 + 0.6 * max(uSunDir.y, 0.0)));
  } else if (mode == 1) {
    col = vC.rgb * (uSkyAmb * fill * 0.9 + uSunColor * (ndl * 0.95 + 0.08));
  } else if (mode == 6) {
    // 地表：双尺度采样 + 低频色斑，抑制平铺感
    vec3 a1 = srgb2lin(texture(uTex, vUV, uMipBias).rgb);
    vec3 a2 = srgb2lin(texture(uTex, vUV * 0.41 + vec2(0.37, 0.21), uMipBias).rgb);
    vec3 albedo = mix(a1, a2, 0.34) * vC.rgb;
    float nz = vnoise(vUV * 0.16) * 0.6 + vnoise(vUV * 0.47) * 0.4;
    float fine = vnoise(vUV * 7.0) * 0.45 + vnoise(vUV * 23.0) * 0.55;
    float fineFade = clamp(1.0 - vDist / 45.0, 0.0, 1.0);
    albedo *= 0.88 * (0.93 + 0.14 * nz) * (1.0 + (fine - 0.5) * 0.22 * fineFade);
    float k = smoothstep(55.0, 170.0, vDist);
    albedo = mix(albedo, uGroundTint, k * 0.80);
    col = albedo * (uSkyAmb * fill + uSunColor * ndl * 1.02 + 0.02);
    // 掠射角天光反射：远处地表带上天色，像水面/湿地
    vec3 V = normalize(uEye - vWorld);
    float fres = pow(1.0 - clamp(dot(n, V), 0.0, 1.0), 4.5);
    col = mix(col, uSkyAmb * 1.05 + uFogColor * 0.30, fres * 0.30 * (1.0 - k));
  } else if (mode == 5) {
    // 反投影地面：照片前景直接铺在地形上，远处收敛到地面基色
    vec3 albedo = srgb2lin(texture(uTex, vUV, uMipBias).rgb) * vC.rgb;
    float k = smoothstep(34.0, 110.0, vDist);
    albedo = mix(albedo, uGroundTint, k * 0.82);
    col = albedo * (uSkyAmb * fill + uSunColor * ndl * 1.02 + 0.02);
  } else if (mode == 4) {
    // 预烘焙光照（远山 / 近岭 / 深度表面）：贴图 × 顶点色
    col = srgb2lin(texture(uTex, vUV, uMipBias).rgb) * vC.rgb;
    // 视锥边缘与画面底缘把照片纹理融进环境色，消掉「插在水面上的卡片」
    // 那种四周硬切边；不做这一步，任何放宽渐隐的尝试都会露出拉伸条纹。
    // 上半屏融天光、下半屏融地面色，否则深色场景侧边会出现一条浅灰「雾墙」
    vec3 envCol = mix(uSkyAmb, uGroundTint, smoothstep(0.42, 0.62, vUV.y));
    col = mix(col, envCol, vEdge.x);
    col = mix(col, uGroundTint, vEdge.y);
  } else {
    vec3 albedo = srgb2lin(texture(uTex, vUV, uMipBias).rgb) * vC.rgb;
    col = albedo * (uSkyAmb * fill + uSunColor * ndl * 1.02 + 0.02);
  }

  float f = 1.0 - exp(-pow(vDist * uFogDensity, 2.2));
  f = clamp(f * uFogAmount, 0.0, 1.0);
  col = mix(col, uFogColor, f);
  fragColor = vec4(col * uExposure, alpha);
}`;

  const VS_FULL = `#version 300 es
precision highp float;
in vec2 aClip;
out vec2 vNdc;
void main() {
  vNdc = aClip;
  gl_Position = vec4(aClip, 0.0, 1.0);
}`;

  const FS_SKY = `#version 300 es
precision highp float;
in vec2 vNdc;
uniform mat4 uInvVP;
uniform vec3 uEye;
uniform vec3 uTop;
uniform vec3 uHorizon;
uniform vec3 uBottom;
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform float uSunPower;
out vec4 fragColor;

void main() {
  vec4 p = uInvVP * vec4(vNdc, 1.0, 1.0);
  vec3 dir = normalize(p.xyz / p.w - uEye);
  float h = dir.y;

  vec3 col = mix(uHorizon, uTop, pow(clamp(h, 0.0, 1.0), 0.55));
  col = mix(col, uBottom, pow(clamp(-h, 0.0, 1.0), 0.5));

  // 太阳光晕
  float sd = max(dot(dir, uSunDir), 0.0);
  /* v1.23：外层光晕收窄（8→20 次幂）并降到 0.16。
     旧的 pow(sd,8)*0.25 是一圈直径约画面 45% 的大白雾，晴天照片也发灰。 */
  col += uSunColor * (pow(sd, 20.0) * 0.16 + pow(sd, 220.0) * 2.0) * uSunPower;
  // 地平线附近轻微大气增亮
  col += uHorizon * 0.08 * exp(-abs(h) * 18.0);
  fragColor = vec4(col, 1.0);
}`;

  const FS_POST = `#version 300 es
precision highp float;
in vec2 vNdc;
uniform sampler2D uScene;
uniform vec2 uRes;
uniform float uTime;
uniform float uFade;
uniform float uVignette;
uniform float uExposure;
uniform float uGrain;
uniform float uAberr;
uniform float uAA;
out vec4 fragColor;

const vec3 LUMA = vec3(0.299, 0.587, 0.114);

vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// 色调映射 + sRGB 编码
vec3 tone(vec3 c) {
  return pow(aces(max(c, 0.0) * uExposure * 1.04), vec3(1.0 / 2.2));
}

vec3 S(vec2 uv) {
  return tone(texture(uScene, uv).rgb);
}

void main() {
  vec2 uv = vNdc * 0.5 + 0.5;
  vec2 rcp = 1.0 / uRes;

  // ---- FXAA（平滑边缘，避免像素感） ----
  vec3 rgbM = S(uv);
  vec3 rgbNW = S(uv + vec2(-1.0, -1.0) * rcp);
  vec3 rgbNE = S(uv + vec2(1.0, -1.0) * rcp);
  vec3 rgbSW = S(uv + vec2(-1.0, 1.0) * rcp);
  vec3 rgbSE = S(uv + vec2(1.0, 1.0) * rcp);
  float lNW = dot(rgbNW, LUMA);
  float lNE = dot(rgbNE, LUMA);
  float lSW = dot(rgbSW, LUMA);
  float lSE = dot(rgbSE, LUMA);
  float lM = dot(rgbM, LUMA);
  float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
  float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));

  vec3 col = rgbM;
  if (uAA > 0.5 && lMax - lMin > 0.035) {
    vec2 dir = vec2(-((lNW + lNE) - (lSW + lSE)), ((lNW + lSW) - (lNE + lSE)));
    float reduce = max((lNW + lNE + lSW + lSE) * 0.25 * 0.02, 1.0 / 128.0);
    float rcpMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcpMin, vec2(-7.0), vec2(7.0)) * rcp;
    vec3 rgbA = 0.5 * (S(uv + dir * (1.0 / 3.0 - 0.5)) + S(uv + dir * (2.0 / 3.0 - 0.5)));
    vec3 rgbB = rgbA * 0.5 + 0.25 * (S(uv - dir * 0.5) + S(uv + dir * 0.5));
    float lB = dot(rgbB, LUMA);
    col = (lB < lMin || lB > lMax) ? rgbA : rgbB;
  }

  // ---- 暗角 ----
  vec2 c = uv - 0.5;
  float v = smoothstep(1.22, 0.44, length(c) * 1.14);
  col *= mix(1.0, v, uVignette);

  // ---- 极轻微胶片颗粒（抗色带，几乎不可见） ----
  col += (hash(uv * uRes + fract(uTime) * 61.0) - 0.5) * uGrain;

  col *= uFade;
  fragColor = vec4(col, 1.0);
}`;


  const VS_POINTS = `#version 300 es
precision highp float;
in vec3 aPos;
in float aSize;
in vec4 aColor;
uniform mat4 uVP;
uniform vec3 uEye;
uniform float uScale;
out vec4 vC;
out float vDist;
out vec3 vWorld;
void main() {
  gl_Position = uVP * vec4(aPos, 1.0);
  vWorld = aPos;
  vDist = length(aPos - uEye);
  vC = aColor;
  gl_PointSize = clamp(aSize * uScale / max(vDist, 0.6), 1.0, 64.0);
}`;

  const FS_POINTS = `#version 300 es
precision highp float;
in vec4 vC;
in float vDist;
uniform vec3 uFogColor;
uniform float uFogDensity;
out vec4 fragColor;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r = length(d) * 2.0;
  if (r > 1.0) discard;
  float a = (1.0 - r) * (1.0 - r) * vC.a;
  float f = clamp(1.0 - exp(-pow(vDist * uFogDensity, 2.2)), 0.0, 1.0);
  vec3 col = mix(vC.rgb, uFogColor, f * 0.7);
  fragColor = vec4(col, a * (1.0 - f * 0.85));
}`;

  /* ---- 点云 / surfel 渲染（v1.24.7）----
   * 参考 GitHub 上的纯前端方案（miggyh27/ImgCloud 的 batched POINTS、
   * warhammer50K/WebPointCloud 的点云 viewer、activeguild/render-image-depth 的
   * 分层透视）：三角形网格在掠射角必然拉伸、单层薄壳侧面就是「竖条幕」，
   * 而点云没有「连接」，每个像素独立成点，天然不会拉丝。
   * 这里用照片 + 深度反投影出 surfel，点大小随距离衰减、圆形 alpha 羽化。 */
  const VS_SPLAT = `#version 300 es
precision highp float;
in vec3 aPos;
in vec2 aUV;
in vec4 aColor;
uniform mat4 uVP;
uniform vec3 uEye;
uniform float uScale;
out vec2 vUV;
out vec4 vC;
out float vDist;
void main() {
  gl_Position = uVP * vec4(aPos, 1.0);
  vUV = aUV;
  vC = aColor;
  vDist = length(aPos - uEye);
  // 下限 3.2px：远处点太小会在竖面上露出「点列之间」的背景，读起来像栅栏；
  // 3.2px 覆盖足够密，代价是远景稍微糊一点。
  gl_PointSize = clamp(uScale / max(vDist, 1.0), 3.2, 20.0);
}`;

  const FS_SPLAT = `#version 300 es
precision highp float;
in vec2 vUV;
in vec4 vC;
in float vDist;
uniform sampler2D uTex;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform float uFogAmt;
out vec4 fragColor;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r = length(d) * 2.0;
  if (r > 1.0) discard;
  vec3 col = texture(uTex, vUV).rgb * vC.rgb;
  float f = clamp(1.0 - exp(-pow(vDist * uFogDensity, 2.2)), 0.0, 1.0);
  col = mix(col, uFogColor, f * uFogAmt);
  fragColor = vec4(col, (1.0 - r * r) * 0.9 * vC.a);
}`;

  function compile(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh);
      console.error('shader compile error:', log, '\n', src.split('\n').map((l, i) => (i + 1) + ': ' + l).join('\n'));
      throw new Error('Shader compile failed: ' + log);
    }
    return sh;
  }

  function createProgram(gl, vsSrc, fsSrc) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error('Program link failed: ' + gl.getProgramInfoLog(p));
    }
    const uniforms = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveUniform(p, i);
      const name = info.name.replace(/\[0\]$/, '');
      uniforms[name] = { loc: gl.getUniformLocation(p, info.name), type: info.type };
    }
    return { prog: p, uniforms, gl };
  }

  function setUniforms(gl, program, values) {
    const us = program.uniforms;
    for (const key in values) {
      const u = us[key];
      if (!u) continue;
      const v = values[key];
      switch (u.type) {
        case gl.FLOAT: gl.uniform1f(u.loc, v); break;
        case gl.FLOAT_VEC2: gl.uniform2fv(u.loc, v); break;
        case gl.FLOAT_VEC3: gl.uniform3fv(u.loc, v); break;
        case gl.FLOAT_VEC4: gl.uniform4fv(u.loc, v); break;
        case gl.FLOAT_MAT4: gl.uniformMatrix4fv(u.loc, false, v); break;
        case gl.INT: case gl.BOOL: gl.uniform1i(u.loc, v); break;
        case gl.SAMPLER_2D: gl.uniform1i(u.loc, v); break;
        default: break;
      }
    }
  }

  /* 网格：位置 / 法线 / UV / 颜色（vec4）四路独立缓冲 + 索引 */
  function createMesh(gl, data) {
    const mesh = { count: 0, mode: data.mode === undefined ? gl.TRIANGLES : data.mode };
    const mk = (arr, size) => {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, arr instanceof Float32Array ? arr : new Float32Array(arr), gl.STATIC_DRAW);
      return { buf: b, size };
    };
    mesh.aPos = mk(data.pos, 3);
    mesh.aNormal = data.normal ? mk(data.normal, 3) : null;
    mesh.aUV = data.uv ? mk(data.uv, 2) : null;
    mesh.aColor = data.color ? mk(data.color, 4) : null;
    mesh.aEdge = data.edge ? mk(data.edge, 2) : null;
    mesh.count = data.count !== undefined ? data.count : data.pos.length / 3;
    if (data.index) {
      const ib = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(data.index), gl.STATIC_DRAW);
      mesh.index = { buf: ib, count: data.index.length };
      mesh.count = data.index.length;
    }
    return mesh;
  }

  function bindMesh(gl, program, mesh) {
    const loc = (name) => gl.getAttribLocation(program.prog, name);
    const bind = (attrib, buf, zero) => {
      if (attrib < 0) return;
      if (!buf) {
        gl.disableVertexAttribArray(attrib);
        // aEdge 缺失的网格（GLB 资产、程序化地面等）必须默认 0，
        // 否则整张网格会被当成边缘融进雾色。
        if (zero) gl.vertexAttrib2f(attrib, 0, 0);
        else gl.vertexAttrib4f(attrib, 1, 1, 1, 1);
        return;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, buf.buf);
      gl.enableVertexAttribArray(attrib);
      gl.vertexAttribPointer(attrib, buf.size, gl.FLOAT, false, 0, 0);
    };
    bind(loc('aPos'), mesh.aPos);
    bind(loc('aNormal'), mesh.aNormal);
    bind(loc('aUV'), mesh.aUV);
    bind(loc('aColor'), mesh.aColor);
    bind(loc('aEdge'), mesh.aEdge, true);
    if (mesh.index) gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.index.buf);
    else gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  }

  /* texImage2D 遇到超过 MAX_TEXTURE_SIZE 的源会静默失败（黑贴图而不报错）。
     v1.20.3 起主照片贴图升到 2048，手机常见上限 4096、桌面 16384；
     这里统一把超限源等比收进上限，程序化画布与对象裁片也一并受保护。 */
  function fitSource(gl, source) {
    let max = 0;
    try { max = gl.getParameter(gl.MAX_TEXTURE_SIZE) || 0; } catch (e) { return source; }
    const sw = source.width || source.videoWidth || 0;
    const sh = source.height || source.videoHeight || 0;
    if (!max || !sw || !sh || (sw <= max && sh <= max)) return source;
    const s = Math.min(max / sw, max / sh);
    const cv = document.createElement('canvas');
    cv.width = Math.max(1, Math.floor(sw * s));
    cv.height = Math.max(1, Math.floor(sh * s));
    const ctx = cv.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(source, 0, 0, cv.width, cv.height);
    return cv;
  }

  function createTexture(gl, source, opts) {
    opts = opts || {};
    source = fitSource(gl, source);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, opts.flipY !== false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    const wrap = opts.wrap === 'mirror' ? gl.MIRRORED_REPEAT : (opts.wrap === 'clamp' ? gl.CLAMP_TO_EDGE : gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, opts.clampV ? gl.CLAMP_TO_EDGE : wrap);
    const aniso = gl.getExtension('EXT_texture_filter_anisotropic');
    if (aniso && opts.aniso) {
      const max = gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
      gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(opts.aniso, max));
    }
    return tex;
  }

  global.PW.GL = {
    VS_MAIN, FS_MAIN, VS_FULL, FS_SKY, FS_POST, VS_POINTS, FS_POINTS,
    VS_SPLAT, FS_SPLAT,
    createProgram, setUniforms, createMesh, bindMesh, createTexture
  };
})(window);
