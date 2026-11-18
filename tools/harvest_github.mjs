#!/usr/bin/env node
/**
 * GitHub 图片采集（兼容入口）。
 *
 * 逻辑已用 Python 重写在 harvest_github.py（"尽量用 Python"），本文件保留为薄包装，
 * 让旧的 `node tools/harvest_github.mjs ...` 调用方无需改动。参数/输出/退出码透传。
 *
 *   node tools/harvest_github.mjs --target 900 --out corpus
 */
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PY = path.join(HERE, 'harvest_github.py');
const args = process.argv.slice(2);

const candidates = process.platform === 'win32' ? ['python', 'python3'] : ['python3', 'python'];
let r = null;
for (const cmd of candidates) {
  r = spawnSync(cmd, [PY, ...args], { stdio: 'inherit' });
  if (r.error && r.error.code === 'ENOENT') continue;
  break;
}
if (!r || r.error) {
  console.error('找不到 Python 解释器，无法运行 harvest_github.py');
  process.exit(1);
}
process.exit(r.status === null ? 1 : r.status);
