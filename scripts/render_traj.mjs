#!/usr/bin/env node
/**
 * 从 glb 世界渲染带真值位姿的轨迹（兼容入口）。
 *
 * 逻辑已用 Python 重写在 render_traj.py（"尽量用 Python"），本文件保留为薄包装，
 * 让旧的 `node scripts/render_traj.mjs ...` 调用方无需改动。参数/输出/退出码透传。
 *
 *   node scripts/render_traj.mjs --glb assets/world-alpine.glb --out data/syn --trajs 6 --frames 48
 */
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PY = path.join(HERE, 'render_traj.py');
const args = process.argv.slice(2);

const candidates = process.platform === 'win32' ? ['python', 'python3'] : ['python3', 'python'];
let r = null;
for (const cmd of candidates) {
  r = spawnSync(cmd, [PY, ...args], { stdio: 'inherit' });
  if (r.error && r.error.code === 'ENOENT') continue;
  break;
}
if (!r || r.error) {
  console.error('找不到 Python 解释器，无法运行 render_traj.py');
  process.exit(1);
}
process.exit(r.status === null ? 1 : r.status);
