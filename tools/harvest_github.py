#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GitHub 图片采集（Python 版）：从公开仓库 raw 直链拉风景 JPEG，用于图集训练。

    python tools/harvest_github.py --target 900 --out corpus
    python tools/harvest_github.py --repos a/b,c/d --target 200 --dry

为什么不是 Pixabay：pixabay.com 对 curl 返回 403，headless 拿到 JS 挑战页；
GitHub 是本机唯一稳定可达的源。过滤（宁可少收不要脏数据）：JPEG 真解码通过、
宽 ≥ 900、横向 1.15~2.6、≥ 45KB、天空启发式（顶部 12% 亮度方差小且整体不比底部暗）。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from PIL import Image
except ImportError:
    Image = None

TOKEN_ENV = "GITHUB_TOKEN"
UA = "photoworld-trainer"

DEFAULT_REPOS = [
    "Doble-2/scenery-wallpapers",
    "Joao2Pereira1/Wallpapers",
    "na-ive/wallpapers",
    "malwarepad/wallpapers",
    "ItsTerm1n4l/Wallpapers-old-archive",
    "Nirupam-Ghosh2004/VisualVault",
    "IcePocket/Wallpapers",
    "OntologicalShock/nord-stalker-wallpapers",
]


def token() -> str:
    t = os.environ.get(TOKEN_ENV, "").strip()
    if t:
        return t
    try:
        out = subprocess.run(["git", "credential", "fill"],
                             input="protocol=https\nhost=github.com\n\n",
                             capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            if line.startswith("password="):
                return line[len("password="):].strip()
    except Exception:
        pass
    return ""


def api(path: str, tok: str):
    args = ["curl", "-k", "-s", "-m", "40", "-H", "Accept: application/vnd.github+json"]
    if tok:
        args += ["-H", "Authorization: Bearer " + tok]
    args.append("https://api.github.com" + path)
    try:
        out = subprocess.run(args, capture_output=True, timeout=60).stdout
        return json.loads(out.decode("utf-8", "replace"))
    except Exception:
        return None


def fetch_raw(repo: str, file: str) -> bytes | None:
    segs = "/".join(urllib.parse.quote(s, safe="") for s in file.split("/"))
    url = "https://raw.githubusercontent.com/%s/HEAD/%s" % (repo, segs)
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE  # 本机代理是自签证书
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
            if r.status != 200:
                return None
            data = r.read(40 * 1024 * 1024 + 1)
        return data if len(data) <= 40 * 1024 * 1024 else None
    except Exception:
        return None


def looks_like_landscape(data: bytes, w: int, h: int) -> bool:
    """顶部 12% 与底部 25% 的亮度差 + 顶部方差。"""
    rows = max(2, int(h * 0.12))
    rows_b = max(2, int(h * 0.25))
    top_l = bot_l = 0.0
    top_n = bot_n = 0
    tops = []
    for y in range(rows):
        for x in range(0, w, 4):
            i = (y * w + x) * 4
            l = (0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]) / 255
            top_l += l
            top_n += 1
            tops.append(l)
    for y in range(h - rows_b, h):
        for x in range(0, w, 4):
            i = (y * w + x) * 4
            bot_l += (0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]) / 255
            bot_n += 1
    top_l /= max(1, top_n)
    bot_l /= max(1, bot_n)
    varr = sum((l - top_l) ** 2 for l in tops) / max(1, len(tops))
    return (top_l > bot_l * 0.92) and varr < 0.06


def verify(buf: bytes | None, min_w: int, min_bytes: int):
    if not buf or len(buf) < min_bytes or Image is None:
        return None
    try:
        im = Image.open(io.BytesIO(buf)).convert("RGBA")
        w, h = im.size
    except Exception:
        return None
    if w < min_w or h < 400:
        return None
    ar = w / h
    if ar < 1.15 or ar > 2.6:
        return None
    if not looks_like_landscape(im.tobytes(), w, h):
        return None
    return {"w": w, "h": h, "digest": hashlib.md5(buf).hexdigest()[:16]}


def _arg(name, default):
    argv = sys.argv[1:]
    if "--" + name in argv:
        i = argv.index("--" + name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, _arg("out", "corpus"))
    target = int(_arg("target", "600"))
    per_repo = int(_arg("per-repo", "220"))
    min_w = int(_arg("min-w", "900"))
    min_bytes = int(_arg("min-bytes", "45000"))
    dry = "--dry" in sys.argv
    workers = int(_arg("workers", "6"))

    repos_arg = _arg("repos", "")
    repos = [s.strip() for s in repos_arg.split(",") if s.strip()] or DEFAULT_REPOS
    tok = token()

    os.makedirs(out, exist_ok=True)
    manifest_path = os.path.join(out, "manifest.json")
    manifest = []
    if os.path.isfile(manifest_path):
        try:
            manifest = json.load(open(manifest_path, encoding="utf-8"))
        except Exception:
            manifest = []
    seen = {m["file"] for m in manifest}

    added = tried = rejected = 0
    t0 = time.monotonic()
    queue = []
    for repo in repos:
        if added >= target:
            break
        tree = api("/repos/%s/git/trees/HEAD?recursive=1" % repo, tok)
        if not tree or not tree.get("tree"):
            print("✗ %s  无法读取目录树" % repo)
            continue
        files = [x for x in tree["tree"]
                 if x.get("type") == "blob" and x.get("path", "").lower().endswith((".jpg", ".jpeg"))
                 and (x.get("size") or 0) >= min_bytes]
        random.shuffle(files)
        files = files[:per_repo]
        print("· %s  候选 %d 张" % (repo, len(files)))
        for f in files:
            queue.append((repo, f["path"]))
    print("共 %d 个候选，目标再收 %d 张" % (len(queue), target))

    counter_lock = __import__("threading").Lock()

    def worker():
        nonlocal added, tried, rejected
        while True:
            with counter_lock:
                if added >= target:
                    return
                i = tried
                tried += 1
            if i >= len(queue):
                return
            repo, path = queue[i]
            buf = fetch_raw(repo, path)
            info = verify(buf, min_w, min_bytes)
            if not info:
                with counter_lock:
                    rejected += 1
                continue
            name = info["digest"] + ".jpg"
            with counter_lock:
                if name in seen:
                    continue
                seen.add(name)
                if not dry:
                    with open(os.path.join(out, name), "wb") as f:
                        f.write(buf)
                    manifest.append({
                        "file": name, "cat": "landscape", "query": "github:" + repo,
                        "src": "https://github.com/%s/blob/HEAD/%s" % (repo, path),
                        "repo": repo, "path": path,
                        "license": "public repo (见源仓库 LICENSE)",
                        "w": info["w"], "h": info["h"],
                    })
                added += 1
                n = added
            if n % 20 == 0:
                if not dry:
                    with open(manifest_path, "w", encoding="utf-8") as f:
                        json.dump(manifest, f, ensure_ascii=False, indent=1)
                sp = max(1e-6, time.monotonic() - t0)
                print("  已收 %d / %d（尝试 %d，拒绝 %d）%ds ≈ %.1f 张/s"
                      % (n, target, tried, rejected, int(sp), n / sp))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker) for _ in range(workers)]
        for f in as_completed(futures):
            f.result()

    if not dry:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("完成：新增 %d 张（尝试 %d，过滤 %d）· 清单 %d 张 → %s"
          % (added, tried, rejected, len(manifest), os.path.relpath(manifest_path, root)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
