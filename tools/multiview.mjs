#!/usr/bin/env node
/**
 * 多视角批量质检（兼容入口）。
 *
 * 逻辑已用 Python 重写在 multiview.py（"尽量用 Python"，v3.0.6）：
 *   - 浏览器链路复用 tools/cdp.py（标准库手写 WebSocket）；
 *   - 清晰度六项复用 tools/clarity.py（与 clarity.mjs 镜像 parity 锁定）；
 *   - PNG 解码用 Pillow（缺失时回退标准库 zlib）。
 * 本文件保留为薄包装，让旧的 `node tools/multiview.mjs ...` 调用方无需改动。
 * 参数/输出/退出码透传。
 *
 *   node tools/multiview.mjs --base http://127.0.0.1:8099 --list corpus/sample.txt
 */
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PY = path.join(HERE, 'multiview.py');
const args = process.argv.slice(2);

const candidates = process.platform === 'win32' ? ['python', 'python3'] : ['python3', 'python'];
let r = null;
for (const cmd of candidates) {
  r = spawnSync(cmd, [PY, ...args], { stdio: 'inherit' });
  if (r.error && r.error.code === 'ENOENT') continue;
  break;
}
if (!r || r.error) {
  console.error('找不到 Python 解释器，无法运行 multiview.py');
  process.exit(1);
}
process.exit(r.status === null ? 1 : r.status);
