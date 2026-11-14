/* 第一人称控制：键鼠 / 触屏，行走 + 贴地 + 边界 */
(function (global) {
  'use strict';

  const { clamp, lerp } = global.PW.util;

  function createPlayer(canvas, world) {
    const EYE = (world && world.eyeHeight) || 1.68;
    // walkRadius === 0 表示不限制（深度重建模式）；undefined 才用默认值
    const MAX_R = (world && world.walkRadius === 0) ? 0
      : ((world && world.walkRadius) || 116);

    const state = {
      x: 0, z: 0, y: EYE,
      yaw: 0, pitch: -0.02,
      vx: 0, vz: 0,
      bob: 0, phase: 0,
      locked: false,
      enabled: true,
      speedMul: 1,
      fovMul: 1.06
    };

    const keys = Object.create(null);
    const touch = { joy: null, look: null, jx: 0, jy: 0 };

    function onKey(e, down) {
      const k = e.key.toLowerCase();
      keys[k] = down;
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
    document.addEventListener('pointerlockchange', () => {
      state.locked = document.pointerLockElement === canvas;
      if (global.PW.onLockChange) global.PW.onLockChange(state.locked);
    });

    /* 触屏 */
    const joyEl = document.getElementById('joystick');
    const joyKnob = document.getElementById('joystickKnob');
    let joyId = null, lookId = null;

    canvas.addEventListener('touchstart', (e) => {
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
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
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
    }, { passive: false });

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

      const sprint = (keys['shift'] || touch.jy < -0.85) ? 2.05 : 1;
      const target = 4.5 * state.speedMul * sprint;
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

      const groundY = world.heightAt(state.x, state.z);
      state.y = groundY + EYE + state.bob;

      state.speed = speed;
    }

    function reset(x, z, yaw) {
      state.x = x || 0; state.z = z || 0;
      state.yaw = yaw || 0;
      state.pitch = -0.02;
      state.vx = state.vz = 0;
      state.y = world.heightAt(state.x, state.z) + EYE;
    }

    function dispose() {
      window.removeEventListener('keydown', kd);
      window.removeEventListener('keyup', ku);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mouseup', onMouseUp);
      window.removeEventListener('wheel', onWheel);
    }

    reset(0, 0, 0);

    return { state, update, reset, dispose, requestLock };
  }

  global.PW.createPlayer = createPlayer;
})(window);
