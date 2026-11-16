#!/usr/bin/env python3
"""静态资源缓存击穿：index.html 里所有 `?v=x.y.z` 必须同版本。

  python tools/bump_cache.py 1.10.0    # 升版并写回
  python tools/bump_cache.py --check   # 只校验一致性（供 CI / 自检）

从 bump_cache.mjs 翻译而来，行为/输出保持一致；.mjs 保留为薄包装转调本脚本，
旧的 `node tools/bump_cache.mjs` 调用方无需改动。
"""
from __future__ import annotations

import datetime
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "index.html")
VERSION_RE = re.compile(r"\?v=(\d+\.\d+\.\d+)")
ANY_V_RE = re.compile(r"\?v=\d+\.\d+\.\d+")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def main(argv: list[str]) -> int:
    arg = argv[1] if len(argv) > 1 else "--check"
    with open(INDEX, encoding="utf-8") as f:
        src = f.read()
    versions = VERSION_RE.findall(src)
    tags = len(re.findall(r"\?v=", src))
    if not tags or len(versions) != tags:
        print("index.html 有 %d 个版本串，但只解析出 %d 个合法版本号" % (tags, len(versions)),
              file=sys.stderr)
        return 1
    current = sorted(set(versions))
    if len(current) != 1:
        print("版本串不一致：" + ", ".join(current), file=sys.stderr)
        return 1

    if arg == "--check":
        print("index.html %d 个资源引用全部为 v%s" % (tags, current[0]))
        return 0

    if not SEMVER_RE.match(arg):
        print("版本号格式应为 x.y.z，例如 1.10.0", file=sys.stderr)
        return 1

    out = ANY_V_RE.sub("?v=" + arg, src)
    with open(INDEX, "w", encoding="utf-8", newline="") as f:
        f.write(out)

    # version.json：页面里的「过期缓存自检」比对用
    payload = {"version": arg,
               "builtAt": datetime.datetime.now(datetime.timezone.utc)
               .isoformat(timespec="milliseconds").replace("+00:00", "Z")}
    with open(os.path.join(ROOT, "version.json"), "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")

    # js/app.js 的 VERSION 是第三处手改点，必须一起自动化
    appf = os.path.join(ROOT, "js", "app.js")
    if os.path.exists(appf):
        with open(appf, encoding="utf-8") as f:
            a = f.read()
        b = re.sub(r"const VERSION = '\d+\.\d+\.\d+';", "const VERSION = '%s';" % arg, a)
        if b != a:
            with open(appf, "w", encoding="utf-8", newline="") as f:
                f.write(b)

    print("index.html: v%s → v%s（%d 处缓存串已同步；version.json 与 js/app.js VERSION 已更新）"
          % (current[0], arg, tags))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
