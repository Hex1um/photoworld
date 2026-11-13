#!/usr/bin/env node
/**
 * 静态资源缓存击穿：index.html 里所有 `?v=x.y.z` 必须同版本，否则浏览器
 * 会混用新旧 JS/CSS（改了代码但拿不到新代码，是这类无构建站点最常见的事故）。
 *
 *   node tools/bump_cache.mjs 1.8.0     # 升版并写回
 *   node tools/bump_cache.mjs --check  # 只校验一致性，供 CI 使用
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FILE = path.join(ROOT, 'index.html');
const arg = process.argv[2] || '--check';
const src = fs.readFileSync(FILE, 'utf8');
const versions = [...src.matchAll(/\?v=(\d+\.\d+\.\d+)/g)].map((m) => m[1]);
const tags = [...src.matchAll(/\?v=/g)].length;

if (!tags || versions.length !== tags) {
  console.error(`index.html 有 ${tags} 个版本串，但只解析出 ${versions.length} 个合法版本号`);
  process.exit(1);
}
const current = [...new Set(versions)];
if (current.length !== 1) {
  console.error('版本串不一致：' + current.join(', '));
  process.exit(1);
}

if (arg === '--check') {
  console.log(`index.html ${tags} 个资源引用全部为 v${current[0]}`);
} else {
  if (!/^\d+\.\d+\.\d+$/.test(arg)) {
    console.error('版本号格式应为 x.y.z，例如 1.8.0');
    process.exit(1);
  }
  const out = src.replace(/\?v=\d+\.\d+\.\d+/g, `?v=${arg}`);
  fs.writeFileSync(FILE, out);
  console.log(`index.html: v${current[0]} → v${arg}（${tags} 处缓存串已同步）`);
}
