#!/usr/bin/env node
/**
 * 静态资源缓存击穿（兼容入口）。
 *
 * 逻辑已用 Python 重写在 bump_cache.py（"尽量用 Python"），本文件保留为薄包装，
 * 让旧的 `node tools/bump_cache.mjs ...` 调用方无需改动。参数/输出/退出码透传。
 *
 *   node tools/bump_cache.mjs 1.10.0    # 升版并写回
 *   node tools/bump_cache.mjs --check  # 只校验一致性
 */
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PY = path.join(HERE, 'bump_cache.py');
const args = process.argv.slice(2);

// Windows 上解释器名可能是 python 或 python3，逐个尝试
const candidates = process.platform === 'win32' ? ['python', 'python3'] : ['python3', 'python'];
let r = null;
for (const cmd of candidates) {
  r = spawnSync(cmd, [PY, ...args], { stdio: 'inherit' });
  if (r.error && r.error.code === 'ENOENT') continue;
  break;
}
if (!r || r.error) {
  console.error('找不到 Python 解释器，无法运行 bump_cache.py');
  process.exit(1);
}
process.exit(r.status === null ? 1 : r.status);
