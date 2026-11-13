#!/usr/bin/env node
/**
 * 从 glb 世界渲染带真值位姿的轨迹（合成数据源）
 *
 *   node scripts/render_traj.mjs --glb assets/world-alpine.glb --out data/syn \
 *        --trajs 6 --frames 48
 *
 * 产出 data/syn/traj_XXXX/{frames/000000.jpg, poses.json}
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((a, v, i, arr) => {
  if (v.startsWith('--')) a.push([v.slice(2), arr[i + 1]]);
  return a;
}, []));
const GLB = (args.glb || 'assets/world-alpine.glb').replace(/^(?!\/)/, '/');
const OUT = args.out || 'data/syn';
const N_TRAJ = parseInt(args.trajs || '4', 10);
const N_FRAMES = parseInt(args.frames || '48', 10);
const PORT = 9600 + Math.floor(Math.random() * 200);
const EDGE = args.edge || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const W = 384, H = 256;

const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'pwtraj-'));
const proc = spawn(EDGE, ['--headless=new', `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`, '--use-gl=angle', '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader', '--hide-scrollbars', '--no-first-run',
  `--window-size=${W},${H}`, 'about:blank'], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function ws() {
  for (let i = 0; i < 80; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/list`);
      const t = (await r.json()).find((x) => x.type === 'page');
      if (t) return t.webSocketDebuggerUrl;
    } catch (e) { /* not yet */ }
    await sleep(250);
  }
  throw new Error('CDP 未就绪');
}

function connect(url) {
  return new Promise((res, rej) => {
    const sock = new WebSocket(url);
    let id = 0; const pend = new Map();
    sock.addEventListener('open', () => res({ sock, send }));
    sock.addEventListener('error', () => rej(new Error('ws error')));
    sock.addEventListener('message', (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && pend.has(m.id)) {
        const { resolve, reject } = pend.get(m.id); pend.delete(m.id);
        if (m.error) reject(new Error(JSON.stringify(m.error))); else resolve(m.result);
      }
    });
    function send(method, params) {
      const mid = ++id;
      sock.send(JSON.stringify({ id: mid, method, params: params || {} }));
      return new Promise((resolve, reject) => pend.set(mid, { resolve, reject }));
    }
  });
}

function makePoses(seed, n) {
  let s = seed * 9301 + 49297;
  const rnd = () => ((s = (s * 9301 + 49297) % 233280) / 233280);
  const poses = [], acts = [];
  let x = 0, z = 0, yaw = (rnd() - 0.5) * 0.5, pitch = (rnd() - 0.5) * 0.12;
  for (let i = 0; i < n; i++) {
    poses.push([x, 1.7, z, yaw, pitch]);
    const r = rnd();
    let act;
    if (r < 0.45) {
      act = 'forward';
      const d = 0.06 + rnd() * 0.06;
      x -= Math.sin(yaw) * d; z -= Math.cos(yaw) * d;
    } else if (r < 0.65) { act = 'turn_left'; yaw += 0.035 + rnd() * 0.03; }
    else if (r < 0.85) { act = 'turn_right'; yaw -= 0.035 + rnd() * 0.03; }
    else if (r < 0.93) { act = 'look_up'; pitch = Math.min(0.45, pitch + 0.02); }
    else { act = 'idle'; }
    acts.push(act);
  }
  return { poses, acts };
}

function trsArray(x, y, z, yaw, pitch) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  return [
    cy, 0, sy, 0,
    sy * sp, cp, -cy * sp, 0,
    -sy * cp, sp, cy * cp, 0,
    x, y, z, 1
  ].map((v) => Math.round(v * 1e5) / 1e5);
}

try {
  const u = await ws();
  const { sock, send } = await connect(u);
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url: 'http://127.0.0.1:8099/tools/traj.html' });
  await sleep(1600);

  const init = await send('Runtime.evaluate', {
    expression: `__traj.init(${JSON.stringify(GLB)})`, awaitPromise: true, returnByValue: true
  });
  if (init.exceptionDetails) throw new Error('init 失败：' + JSON.stringify(init.exceptionDetails.text));
  console.log('世界已加载', JSON.stringify(init.result.value));

  const fovY = 2 * Math.atan(Math.tan(69.4 * Math.PI / 360) * (H / W));
  fs.mkdirSync(OUT, { recursive: true });

  for (let t = 0; t < N_TRAJ; t++) {
    const { poses, acts } = makePoses(t + 1, N_FRAMES);
    const r = await send('Runtime.evaluate', {
      expression: `__traj.runPath(${JSON.stringify(poses)}, ${fovY})`,
      returnByValue: true
    });
    if (r.exceptionDetails) throw new Error('渲染失败：' + JSON.stringify(r.exceptionDetails.text));
    const frames = r.result.value;
    const dir = path.join(OUT, 'traj_' + String(t).padStart(4, '0'));
    fs.mkdirSync(path.join(dir, 'frames'), { recursive: true });
    const meta = { fovY: fovY, fovX: 69.4 * Math.PI / 180, width: W, height: H, frames: [] };
    frames.forEach((b64, i) => {
      const name = String(i).padStart(6, '0') + '.jpg';
      fs.writeFileSync(path.join(dir, 'frames', name), Buffer.from(b64, 'base64'));
      const p = poses[i];
      meta.frames.push({ file: name, camToWorld: trsArray(p[0], p[1], p[2], p[3], p[4]), action: acts[i] });
    });
    fs.writeFileSync(path.join(dir, 'poses.json'), JSON.stringify(meta, null, 1));
    console.log('traj_' + String(t).padStart(4, '0') + '  ' + frames.length + ' 帧');
  }
  sock.close();
} catch (e) {
  console.error('失败：' + e.message);
  process.exitCode = 1;
} finally {
  try { spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { stdio: 'ignore' }); } catch (e) { /* ignore */ }
}
