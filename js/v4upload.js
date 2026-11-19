/* v4：多图 / 视频 → 重建面板
 *
 * 设计上刻意**不碰主程序的模块作用域**：自己建 DOM、自己发请求，成功后用
 * `?glb=...` 重新进站——这样它和 v1/v3 那条链路完全解耦，坏了也不影响看老世界。
 *
 * 依赖本机服务：tools/serve_api.py（默认 http://127.0.0.1:8200）。
 * 没起服务时给出明确提示，而不是静默失败。
 */
(function () {
  'use strict';

  const API = (location.protocol + '//' + location.hostname + ':8200').replace('http://', 'http://');
  const $ = (sel) => document.querySelector(sel);

  function elt(tag, cls, txt) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt != null) n.textContent = txt;
    return n;
  }

  /* ---------------- 面板 DOM ---------------- */
  const wrap = elt('div', 'v4w hidden');
  wrap.innerHTML = `
    <div class="v4w-box" role="dialog" aria-label="多图重建">
      <div class="v4w-head">
        <b>多图 / 视频重建</b>
        <button class="v4w-x" title="关闭">✕</button>
      </div>
      <div class="v4w-body">
        <p class="v4w-tip">
          选 <b>6~20 张</b>照片（或一段 5~15 秒视频）。要<b>走动</b>拍、相邻两张重叠三分之一以上，
          原地转圈拍不出深度。
        </p>
        <div class="v4w-pick">
          <button class="v4w-btn" id="v4wPick">选择照片 / 视频</button>
          <input type="file" id="v4wFile" multiple accept="image/*,video/*" hidden>
          <span class="v4w-count" id="v4wCount">未选择</span>
        </div>
        <div class="v4w-bar"><i id="v4wBar"></i></div>
        <div class="v4w-state" id="v4wState">待开始</div>
        <pre class="v4w-log" id="v4wLog"></pre>
        <div class="v4w-qc hidden" id="v4wQc"></div>
        <div class="v4w-foot">
          <button class="v4w-btn primary" id="v4wRun" disabled>开始重建</button>
          <button class="v4w-btn hidden" id="v4wGo">进入这个世界</button>
        </div>
      </div>
    </div>`;
  document.body.appendChild(wrap);

  const style = elt('style');
  style.textContent = `
  .v4w{position:fixed;inset:0;z-index:70;background:rgba(8,10,13,.72);backdrop-filter:blur(6px);
       display:flex;align-items:center;justify-content:center}
  .v4w.hidden{display:none}
  .v4w-box{width:min(660px,92vw);max-height:86vh;overflow:auto;border-radius:16px;
       background:#15181d;border:1px solid #2a2f37;box-shadow:0 24px 80px rgba(0,0,0,.6)}
  .v4w-head{display:flex;align-items:center;justify-content:space-between;
       padding:14px 18px;border-bottom:1px solid #23272e;font-size:15px;color:#e8eaee}
  .v4w-x{background:none;border:0;color:#8b929c;font-size:16px;cursor:pointer;padding:4px 8px}
  .v4w-body{padding:16px 18px 18px}
  .v4w-tip{margin:0 0 14px;font-size:12.5px;line-height:1.7;color:#9aa1ab}
  .v4w-tip b{color:#cfd5dd}
  .v4w-pick{display:flex;align-items:center;gap:12px;margin-bottom:12px}
  .v4w-count{font-size:12px;color:#8b929c}
  .v4w-btn{padding:8px 14px;border-radius:9px;border:1px solid #343a44;background:#1c2027;
       color:#dfe3e9;font-size:13px;cursor:pointer;transition:.15s}
  .v4w-btn:hover:not(:disabled){background:#232833;border-color:#414a57}
  .v4w-btn.primary{background:#2f6df6;border-color:#2f6df6;color:#fff}
  .v4w-btn.primary:hover:not(:disabled){background:#3b78ff}
  .v4w-btn:disabled{opacity:.45;cursor:not-allowed}
  .v4w-bar{height:5px;border-radius:3px;background:#20242b;overflow:hidden;margin:6px 0 10px}
  .v4w-bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,#2f6df6,#63b3ff);
       transition:width .25s}
  .v4w-state{font-size:12.5px;color:#a8b0ba;margin-bottom:8px}
  .v4w-log{margin:0 0 12px;max-height:150px;overflow:auto;padding:10px;border-radius:9px;
       background:#0f1216;border:1px solid #20242b;font-size:11.5px;line-height:1.6;color:#8d96a1;
       white-space:pre-wrap}
  .v4w-qc{border-radius:11px;border:1px solid #2a2f37;padding:12px 14px;margin-bottom:12px;font-size:12.5px}
  .v4w-qc h4{margin:0 0 8px;font-size:13px;display:flex;align-items:center;gap:8px}
  .v4w-chip{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600}
  .v4w-pass{background:#13351f;color:#5ee08a28;color:#63e08b;border:1px solid #235c37}
  .v4w-warn{background:#3a2f12;color:#f0c05a;border:1px solid #5c4a1e}
  .v4w-fail{background:#3a1618;color:#ff7b7b;border:1px solid #5c2529}
  .v4w-kv{display:grid;grid-template-columns:repeat(3,1fr);gap:8px 12px;color:#9aa1ab}
  .v4w-kv b{color:#dfe3e9;font-weight:600}
  .v4w-adv{margin:10px 0 0;padding-left:16px;color:#c9b07a}
  .v4w-adv li{margin:4px 0}
  .v4w-foot{display:flex;gap:10px;justify-content:flex-end}
  .v4w-off{color:#ff9a6b}`;
  document.head.appendChild(style);

  /* ---------------- 状态 ---------------- */
  let files = [];
  let job = null;
  let timer = null;

  function setState(txt) { $('#v4wState').textContent = txt; }
  function setBar(p) { $('#v4wBar').style.width = Math.max(0, Math.min(100, p)) + '%'; }
  function log(txt) {
    const box = $('#v4wLog');
    box.textContent = txt || '';
    box.scrollTop = box.scrollHeight;
  }

  async function api(path, opt) {
    const r = await fetch(API + path, Object.assign({ cache: 'no-store' }, opt || {}));
    const txt = await r.text();
    let data = null;
    try { data = JSON.parse(txt); } catch (e) { data = { raw: txt }; }
    if (!r.ok) throw new Error((data && data.error) || ('HTTP ' + r.status));
    return data;
  }

  /* ---------------- 交互 ---------------- */
  $('#v4wPick').addEventListener('click', () => $('#v4wFile').click());
  $('#v4wFile').addEventListener('change', (e) => {
    files = Array.from(e.target.files || []);
    $('#v4wCount').textContent = files.length
      ? (files.length + ' 个文件 · ' + (files.reduce((a, f) => a + f.size, 0) / 1048576).toFixed(1) + ' MB')
      : '未选择';
    $('#v4wRun').disabled = files.length === 0;
  });

  $('#v4wRun').addEventListener('click', async () => {
    if (!files.length) return;
    $('#v4wRun').disabled = true;
    $('#v4wGo').classList.add('hidden');
    $('#v4wQc').classList.add('hidden');
    try {
      setState('创建任务…'); setBar(3);
      const j = await api('/api/v4/job', { method: 'POST' });
      job = j.job;
      for (let i = 0; i < files.length; i++) {
        const f = files[i];
        setState(`上传 ${i + 1}/${files.length}：${f.name}`);
        setBar(5 + 45 * (i / files.length));
        await api('/api/v4/upload?job=' + encodeURIComponent(job) +
                  '&name=' + encodeURIComponent(f.name),
                  { method: 'POST', body: f });
      }
      setBar(52);
      setState('启动重建（本机计算，可能需要几十秒）…');
      await api('/api/v4/build?job=' + encodeURIComponent(job), { method: 'POST' });
      poll();
    } catch (err) {
      setState('失败：' + err.message);
      $('#v4wRun').disabled = false;
      log('如果是「Failed to fetch」，说明本机重建服务没起。\n' +
          '请运行：python tools/serve_api.py --port 8200 --static .\n' +
          '（或双击桌面上的「PHOTOWORLD v4 - 重建服务.bat」）');
    }
  });

  function poll() {
    clearInterval(timer);
    timer = setInterval(async () => {
      try {
        const s = await api('/api/v4/status?job=' + encodeURIComponent(job));
        log(s.log || '');
        if (s.qc) renderQc(s.qc);
        if (s.state === 'running') {
          setState('重建中…'); setBar((b) => b);
          $('#v4wBar').style.width = '78%';
        } else if (s.state === 'done') {
          clearInterval(timer); setBar(100);
          setState('完成 —— 世界已生成');
          if (s.glb) {
            $('#v4wGo').classList.remove('hidden');
            $('#v4wGo').onclick = () => {
              location.href = '?glb=' + encodeURIComponent(s.glb) + '&instant=1';
            };
          }
          $('#v4wRun').disabled = false;
        } else if (s.state === 'fail') {
          clearInterval(timer); setBar(100);
          setState('未通过质量门 —— 这个素材不适合重建（报告见下）');
          $('#v4wRun').disabled = false;
        }
      } catch (e) { /* 网络抖动时继续轮询 */ }
    }, 1200);
  }

  function renderQc(qc) {
    const box = $('#v4wQc');
    box.classList.remove('hidden');
    const cls = qc.verdict === 'PASS' ? 'v4w-pass' : qc.verdict === 'WARN' ? 'v4w-warn' : 'v4w-fail';
    const kv = [
      ['帧数', qc.frames], ['定位', qc.localized + '/' + qc.frames],
      ['视差', qc.parallax_median_deg + '°'], ['重投影', qc.reproj_rmse_px + 'px'],
      ['稀疏点', qc.points], ['填充率', qc.fill_ratio == null ? '—' : Math.round(qc.fill_ratio * 100) + '%']
    ];
    box.innerHTML =
      `<h4>重建质量 <span class="v4w-chip ${cls}">${qc.verdict}</span></h4>` +
      '<div class="v4w-kv">' +
      kv.map(([k, v]) => `<div>${k} <b>${v}</b></div>`).join('') +
      '</div>' +
      (qc.advice && qc.advice.length
        ? '<ul class="v4w-adv">' + qc.advice.map((a) => '<li>' + a + '</li>').join('') + '</ul>'
        : '');
  }

  /* ---------------- 打开 / 关闭 ---------------- */
  function open() {
    wrap.classList.remove('hidden');
    // 先探一下服务在不在，给个即时反馈
    fetch(API + '/api/v4/status?job=probe', { cache: 'no-store' })
      .then((r) => r.ok ? setState('待开始 · 本机服务已连接') : null)
      .catch(() => setState('⚠ 本机重建服务未运行：请运行 tools/serve_api.py（或双击桌面 v4 服务快捷方式）'));
  }
  function close() { wrap.classList.add('hidden'); clearInterval(timer); }

  wrap.addEventListener('click', (e) => { if (e.target === wrap) close(); });
  wrap.querySelector('.v4w-x').addEventListener('click', close);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !wrap.classList.contains('hidden')) close();
  });

  // 顶栏按钮：插在「用我的照片」旁边（找不到就挂到 body 右上角，保证可用）
  function mountButton() {
    const ref = document.getElementById('btnUploadTop');
    const btn = elt('button', 'ghost', '多图重建');
    btn.id = 'btnV4Open';
    btn.addEventListener('click', open);
    if (ref && ref.parentNode) ref.parentNode.insertBefore(btn, ref.nextSibling);
    else document.body.appendChild(btn);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mountButton);
  } else { mountButton(); }

  window.__pwV4 = { open, close };
})();
