/* 第一人称控制：键鼠 / 触屏，行走 + 贴地 + 边界 */
(function (global) {
  'use strict';

  const { clamp, lerp } = global.PW.util;

  function createPlayer(canvas, world) {
    const EYE = (world && world.eyeHeight) || 1.68;
    // walkRadius === 0 表示不限制（深度重建模式）；undefined 才用默认值
    /* v1.19：不再有任何移动半径夹紧。以前 fallback（无深度模型）路径是
       walkRadius=116 —— 走出去 116m 就被按回原点，正是用户说的「r≈120 空气墙」。
       现在所有模式都不限制：脚下用解析高度场，四周用实时生成。 */
    const MAX_R = 0;

    const state = {
      x: 0, z: 0, y: EYE,
      yaw: 0, pitch: -0.02,
      vx: 0, vz: 0,
      bob: 0, phase: 0,
      locked: false,
      enabled: true,
      speedMul: 1,
      fovMul: 1.06,
      /* 飞行（创意模式）：双击空格切换；空格升 / Shift 降，松开后自由下落。
         flyY 是"绝对视点高度"，落地前一直用它 —— 不能复用 y，
         因为 y 每帧都被贴地高度覆写。 */
      flying: false,
      flyY: EYE,
      vy: 0
    };

    const keys = Object.create(null);
    const touch = { joy: null, look: null, jx: 0, jy: 0 };

    let lastSpaceAt = 0;
    function onKey(e, down) {
      const k = e.key.toLowerCase();
      keys[k] = down;
      /* 双击空格切换飞行。用 320ms 的双击窗口；e.repeat 要挡掉（按住空格会连发
         keydown，不挡的话一直按着就等于"每帧双击"，飞机会疯狂开关）。 */
      if (down && k === ' ' && !e.repeat) {
        const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
        if (now - lastSpaceAt < 320) {
          state.flying = !state.flying;
          if (state.flying) {
            state.flyY = state.y;
            state.vy = 0;
          }
          lastSpaceAt = 0;          // 用完即清，避免三次点击又触发
        } else {
          lastSpaceAt = now;
        }
      }
      if (down && ['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' '].includes(k)) {
        e.preventDefault();
      }
    }
    const kd = (e) => onKey(e, true);
    const ku = (e) => onKey(e, false);
    window.addEventListener('keydown', kd);
    window.addEventListener('keyup', ku);

    /* 鼠标视角 */
    let dragging = false, lastX = 0, lastY = 0;
    const SENS = 0.0026;

    function look(dx, dy) {
      state.yaw += dx * SENS;
      state.pitch = clamp(state.pitch - dy * SENS, -1.45, 1.42);
    }

    const onMouseMove = (e) => {
      if (!state.enabled) return;
      if (state.locked) {
        look(e.movementX || 0, e.movementY || 0);
      } else if (dragging) {
        look(e.clientX - lastX, e.clientY - lastY);
        lastX = e.clientX; lastY = e.clientY;
      }
    };
    const onMouseDown = (e) => {
      if (!state.enabled) return;
      if (!state.locked && e.button === 0) {
        dragging = true; lastX = e.clientX; lastY = e.clientY;
      }
    };
    const onMouseUp = () => { dragging = false; };
    const onWheel = (e) => {
      if (!state.enabled) return;
      state.fovMul = clamp(state.fovMul - (e.deltaY > 0 ? 0.06 : -0.06), 0.72, 1.55);
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mouseup', onMouseUp);
    window.addEventListener('wheel', onWheel, { passive: true });

    /* 指针锁定 */
    function requestLock() {
      if (canvas.requestPointerLock) canvas.requestPointerLock();
    }
    function onLockChange() {
      state.locked = document.pointerLockElement === canvas;
      if (global.PW.onLockChange) global.PW.onLockChange(state.locked);
    }
    document.addEventListener('pointerlockchange', onLockChange);

    /* 触屏 */
    const joyEl = document.getElementById('joystick');
    const joyKnob = document.getElementById('joystickKnob');
    let joyId = null, lookId = null;

    const onTouchStart = (e) => {
      for (const t of e.changedTouches) {
        const half = window.innerWidth * 0.45;
        if (t.clientX < half && joyId === null) {
          joyId = t.identifier;
          touch.joy = { ox: t.clientX, oy: t.clientY };
          if (joyEl) {
            joyEl.style.display = 'block';
            joyEl.style.left = (t.clientX - 60) + 'px';
            joyEl.style.top = (t.clientY - 60) + 'px';
          }
        } else if (lookId === null) {
          lookId = t.identifier;
          touch.look = { x: t.clientX, y: t.clientY };
        }
      }
      e.preventDefault();
    };

    const onTouchMove = (e) => {
      for (const t of e.changedTouches) {
        if (t.identifier === joyId && touch.joy) {
          const dx = t.clientX - touch.joy.ox;
          const dy = t.clientY - touch.joy.oy;
          const len = Math.hypot(dx, dy);
          const max = 58;
          const k = len > max ? max / len : 1;
          touch.jx = (dx * k) / max;
          touch.jy = (dy * k) / max;
          if (joyKnob) {
            joyKnob.style.transform = `translate(${dx * k}px, ${dy * k}px)`;
          }
        } else if (t.identifier === lookId && touch.look) {
          look((t.clientX - touch.look.x) * 1.7, (t.clientY - touch.look.y) * 1.7);
          touch.look.x = t.clientX; touch.look.y = t.clientY;
        }
      }
      e.preventDefault();
    };

    const endTouch = (e) => {
      for (const t of e.changedTouches) {
        if (t.identifier === joyId) {
          joyId = null; touch.jx = 0; touch.jy = 0;
          if (joyEl) joyEl.style.display = 'none';
          if (joyKnob) joyKnob.style.transform = 'translate(0,0)';
        }
        if (t.identifier === lookId) { lookId = null; touch.look = null; }
      }
    };
    canvas.addEventListener('touchstart', onTouchStart, { passive: false });
    canvas.addEventListener('touchmove', onTouchMove, { passive: false });
    canvas.addEventListener('touchend', endTouch);
    canvas.addEventListener('touchcancel', endTouch);

    function update(dt) {
      dt = Math.min(dt, 0.05);
      if (!state.enabled) return;

      let fwd = 0, side = 0;
      if (keys['w'] || keys['arrowup']) fwd += 1;
      if (keys['s'] || keys['arrowdown']) fwd -= 1;
      if (keys['a'] || keys['arrowleft']) side -= 1;
      if (keys['d'] || keys['arrowright']) side += 1;
      fwd -= touch.jy;
      side += touch.jx;

      const len = Math.hypot(fwd, side);
      if (len > 1) { fwd /= len; side /= len; }

      /* 键位（用户指定）：Ctrl 疾跑、Shift 缓行。
         飞行时 Shift 另有用途（下降），所以缓行只在非飞行时生效；
         触屏摇杆拉到最上仍然是疾跑。 */
      const sprint = (keys['control'] || touch.jy < -0.85) ? 2.05 : 1;
      const slow = (!state.flying && keys['shift']) ? 0.42 : 1;
      // 飞行时水平更快一点：飞行的用途就是快速看地形
      const target = 4.5 * state.speedMul * sprint * slow * (state.flying ? 1.5 : 1);
      const sy = Math.sin(state.yaw), cy = Math.cos(state.yaw);
      const tx = (sy * fwd + cy * side) * target;
      const tz = (-cy * fwd + sy * side) * target;

      const acc = 1 - Math.exp(-11 * dt);
      state.vx = lerp(state.vx, tx, acc);
      state.vz = lerp(state.vz, tz, acc);

      state.x += state.vx * dt;
      state.z += state.vz * dt;

      // 软边界
      const r = Math.hypot(state.x, state.z);
      // v1.14：不再用圆形半径把玩家锁住（用户要求移动不受限制）。
      // 世界边界交给几何与雾；heightField 在区域外会 clamp 到边缘高度。
      if (MAX_R > 0 && r > MAX_R) {
        const k = MAX_R / r;
        state.x *= k; state.z *= k;
        state.vx *= 0.4; state.vz *= 0.4;
      }

      const speed = Math.hypot(state.vx, state.vz);
      state.phase += dt * speed * 2.1;
      const bobTarget = speed > 0.4 ? Math.sin(state.phase) * 0.045 * Math.min(speed / 4.5, 1.4) : 0;
      state.bob = lerp(state.bob, bobTarget, 1 - Math.exp(-12 * dt));

      /* v3：把水平活动范围锁在出生点周围（单视图重建只在拍摄点附近成立） */
      if (world && typeof world.moveRadius === 'number') {
        const r = Math.hypot(state.x, state.z);
        if (r > world.moveRadius) {
          const k = world.moveRadius / r;
          state.x *= k; state.z *= k;
          state.vx = 0; state.vz = 0;
        }
      }

      const groundY = world.heightAt(state.x, state.z);
      const eyeH = (state.eyeOverride === null || state.eyeOverride === undefined) ? EYE : state.eyeOverride;
      const floorY = groundY + eyeH;

      if (state.flying) {
        /* 飞行：空格升 / Shift 降（与《我的世界》创意模式一致），
           水平速度不变（略微加速，飞起来更顺手）。 */
        const up = (keys[' '] ? 1 : 0) - (keys['shift'] ? 1 : 0);
        const vTarget = up * 9.0;
        state.vy = lerp(state.vy, vTarget, 1 - Math.exp(-10 * dt));
        state.flyY += state.vy * dt;
        // 下限：不能钻到地面以下；上限：别飞出世界盒
        /* v3 世界把飞行上限压到离地 12m：再高就会从照片壳的破口往下看，
           看到的不是地面而是虚空/天空卡（v3 是单视图重建，头顶本来没有几何）。
           12m 既够俯瞰，又保证起飞后看到的与站在地上是**同一个世界**。 */
        const maxRel = (world && typeof world.flyMaxRel === 'number') ? world.flyMaxRel : 260;
        state.flyY = clamp(state.flyY, floorY, groundY + maxRel);
        state.y = state.flyY;
        state.bob = 0;
      } else if (state.y > floorY + 0.06) {
        /* 关掉飞行后自由下落（简单重力），落地即贴回地面。
           不做下落的话，关飞行会"啪"地瞬移回地面，很突兀。 */
        state.vy = Math.max(-16, state.vy - 26 * dt);
        state.y += state.vy * dt;
        if (state.y <= floorY) { state.y = floorY; state.vy = 0; state.bob = 0; }
      } else {
        state.vy = 0;
        state.y = floorY + state.bob;
      }

      state.speed = speed;
    }

    function reset(x, z, yaw) {
      state.flying = false;
      state.vy = 0;
      state.eyeOverride = null;
      state.x = x || 0; state.z = z || 0;
      state.yaw = yaw || 0;
      state.pitch = -0.02;
      state.vx = state.vz = 0;
      state.y = world.heightAt(state.x, state.z) + EYE;
    }

    function dispose() {
      /* F10（v2.15.12）：这里原来只解绑 6 个 window 监听 —— canvas 上的三组触摸
         监听与 document 的 pointerlockchange 一直挂着。每次进世界都会再挂一组，
         旧 player + 旧 world 被闭包钉住不释放（配合 world 不回收，显存/内存双泄漏）。 */
      window.removeEventListener('keydown', kd);
      window.removeEventListener('keyup', ku);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mouseup', onMouseUp);
      window.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('touchstart', onTouchStart);
      canvas.removeEventListener('touchmove', onTouchMove);
      canvas.removeEventListener('touchend', endTouch);
      canvas.removeEventListener('touchcancel', endTouch);
      document.removeEventListener('pointerlockchange', onLockChange);
    }

    reset(0, 0, 0);

    return { state, update, reset, dispose, requestLock };
  }

  global.PW.createPlayer = createPlayer;
})(window);
