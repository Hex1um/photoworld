#!/usr/bin/env node
/**
 * 兼容入口：逻辑已迁到 tools/probe_depth.py，本文件只转发参数/输出/退出码。
 *   node tools/probe_depth.mjs --url ... --out reports/depth_probe.json
 */
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PY = path.join(HERE, 'probe_depth.py');
const args = process.argv.slice(2);
const candidates = process.platform === 'win32' ? ['python', 'python3'] : ['python3', 'python'];
let r = null;
for (const cmd of candidates) {
  r = spawnSync(cmd, [PY, ...args], { stdio: 'inherit' });
  if (r.error && r.error.code === 'ENOENT') continue;
  break;
}
if (!r || r.error) { console.error('找不到 Python 解释器，无法运行 probe_depth.py'); process.exit(1); }
process.exit(r.status === null ? 1 : r.status);
