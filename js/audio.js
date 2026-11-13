/* 环境声：程序化风声（无外部音频文件） */
(function (global) {
  'use strict';

  function createAmbience() {
    let ctx = null, master = null, on = false, ready = false;

    function init() {
      if (ctx) return;
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      try { ctx = new AC(); } catch (e) { return; }

      master = ctx.createGain();
      master.gain.value = 0.0001;
      master.connect(ctx.destination);

      const len = Math.floor(ctx.sampleRate * 5);
      const buf = ctx.createBuffer(1, len, ctx.sampleRate);
      const d = buf.getChannelData(0);
      let last = 0;
      for (let i = 0; i < len; i++) {
        const w = Math.random() * 2 - 1;
        last = (last + 0.02 * w) / 1.02;
        d[i] = last * 3.2;
      }

      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.loop = true;

      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.value = 480;
      lp.Q.value = 0.7;

      const g1 = ctx.createGain();
      g1.gain.value = 0.55;
      src.connect(lp);
      lp.connect(g1);
      g1.connect(master);

      // 阵风
      const lfo = ctx.createOscillator();
      lfo.frequency.value = 0.055;
      const lfoG = ctx.createGain();
      lfoG.gain.value = 230;
      lfo.connect(lfoG);
      lfoG.connect(lp.frequency);

      // 细微空气嘶声
      const bp = ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = 1650;
      bp.Q.value = 0.6;
      const g2 = ctx.createGain();
      g2.gain.value = 0.028;
      src.connect(bp);
      bp.connect(g2);
      g2.connect(master);

      src.start();
      lfo.start();
      ready = true;
    }

    function setOn(v) {
      init();
      if (!ready) return false;
      on = v;
      if (ctx.state === 'suspended') ctx.resume();
      const now = ctx.currentTime;
      master.gain.cancelScheduledValues(now);
      master.gain.setValueAtTime(Math.max(master.gain.value, 0.0001), now);
      master.gain.exponentialRampToValueAtTime(v ? 0.16 : 0.0001, now + (v ? 1.6 : 0.7));
      return on;
    }

    return {
      toggle() { return setOn(!on); },
      get on() { return on; }
    };
  }

  global.PW.createAmbience = createAmbience;
})(window);
