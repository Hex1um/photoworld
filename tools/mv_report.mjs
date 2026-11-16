#!/usr/bin/env node
/**
 * 多视角质检汇总器（兼容入口）。
 *
 * 逻辑已用 Python 重写在 mv_report.py（CLARITY_KEYS 内联其中，改那边要同步
 * tools/clarity.mjs）；本文件保留为薄包装，让旧的 `node tools/mv_report.mjs ...`
 * 调用方无需改动。参数/输出/退出码透传。
 *
 *   node tools/mv_report.mjs reports/all32.json [--top 8] [--ring C]
 */
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PY = path.join(HERE, 'mv_report.py');
const args = process.argv.slice(2);

const candidates = process.platform === 'win32' ? ['python', 'python3'] : ['python3', 'python'];
let r = null;
for (const cmd of candidates) {
  r = spawnSync(cmd, [PY, ...args], { stdio: 'inherit' });
  if (r.error && r.error.code === 'ENOENT') continue;
  break;
}
if (!r || r.error) {
  console.error('找不到 Python 解释器，无法运行 mv_report.py');
  process.exit(1);
}
process.exit(r.status === null ? 1 : r.status);
