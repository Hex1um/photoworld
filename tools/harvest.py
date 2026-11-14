#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHOTOWORLD 语料采集器
从 360 图片搜索按场景类别抓取横构图风景照，用于批量评测与参数标定。
仅本地使用（不参与发布），manifest 记录来源，方便复核。

用法:
  python tools/harvest.py --target 520 --out corpus
"""
import argparse, io, json, os, ssl, subprocess, sys, threading, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

QUERIES = [
    ("alpine",   "山脉风景 摄影"),
    ("lake",     "湖泊 倒影 风景"),
    ("coast",    "海岸 悬崖 风景"),
    ("forest",   "森林 晨雾"),
    ("desert",   "沙漠 戈壁 风景"),
    ("snow",     "雪山 高原"),
    ("grass",    "草原 牧场"),
    ("canyon",   "峡谷 地貌"),
    ("field",    "农田 田野 风景"),
    ("city",     "城市 天际线 黄昏"),
    ("autumn",   "秋天 红叶 森林"),
    ("tropic",   "热带 海滩"),
    ("water",    "瀑布 溪流 风景"),
    ("night",    "星空 银河 山"),
    ("road",     "公路 风景"),
    ("valley",   "山谷 溪谷 风景"),
    ("fjord",    "峡湾 北欧 风景"),
    ("rice",     "梯田 田园 风景"),
    ("sunrise",  "日出 云海 山"),
    ("rain",     "雨后 风景 湿润"),
    # —— 第二批（语料扩到 1000+ 时补的场景多样性，避免只有"山+湖+海"）——
    ("cabin",    "小木屋 森林 田野"),
    ("lighthouse", "灯塔 海岸 风景"),
    ("bridge",   "古桥 溪水 风景"),
    ("village",  "乡村 房屋 风景"),
    ("glacier",  "冰川 极地 风景"),
    ("volcano",  "火山 地貌 风景"),
    ("reef",     "海岛 礁石 海水"),
    ("wetland",  "湿地 芦苇 风景"),
    ("dune",     "沙丘 沙漠 光影"),
    ("cliff",    "悬崖 海边 峭壁"),
    ("plateau",  "青藏高原 草原"),
    ("terrace",  "茶园 梯田 山"),
    ("lava",     "火山岩 熔岩 地貌"),
    ("aurora",   "极光 雪原 星空"),
    ("birch",    "白桦林 秋天"),
    ("mangrove", "红树林 水边"),
    ("ruins",    "废墟 古迹 风景"),
    ("harbor",   "港口 船只 风景"),
    ("railway",  "铁路 风景 山"),
    ("windmill", "风车 风电场 风景"),
]

DEFAULT_PAGES = list(range(0, 960, 60))   # pn 0..900，20 类 × 16 页够撑到 1000+
RN = 60
MIN_W = 1100
MIN_AR = 1.25


def curl_json(url, tries=3):
    for i in range(tries):
        p = subprocess.run(
            ["curl", "-k", "-s", "-m", "25", "-A", UA,
             "-H", "Referer: https://image.so.com/",
             "-H", "Accept: application/json, text/plain, */*", url],
            capture_output=True)
        try:
            return json.loads(p.stdout.decode("utf-8", "ignore"))
        except Exception:
            time.sleep(1.2 * (i + 1))
    return None


def gather(query, pages, min_w=None):
    out = []
    for pn in pages:
        url = ("https://image.so.com/j?q=%s&src=srp&pn=%d&rn=%d"
               % (subprocess.list2cmdline([query]).strip('"'), pn, RN))
        from urllib.parse import quote
        url = ("https://image.so.com/j?q=%s&src=srp&pn=%d&rn=%d"
               % (quote(query), pn, RN))
        d = curl_json(url)
        if not d or not d.get("list"):
            continue
        for it in d["list"]:
            u = it.get("img") or ""
            try:
                w, h = int(it.get("width") or 0), int(it.get("height") or 0)
            except (TypeError, ValueError):
                continue
            if not u.startswith("http"):
                continue
            if w < (min_w or MIN_W) or h <= 0 or (w / max(h, 1)) < MIN_AR:
                continue
            out.append({
                "url": u.replace("http://", "https://"),
                "w": w, "h": h, "title": (it.get("title") or "")[:80],
                "site": it.get("site") or "",
                "query": query,
            })
        time.sleep(0.35)
    return out


def verify_bytes(buf):
    """字节级校验：PIL 真解码 + 尺寸/构图门槛。返回 (w,h,md5) 或 None。

    走内存而不是先落盘再校验：热链返 404/502 时 curl 也会写出几 KB 的 HTML，
    旧版靠「先写 .part 再删」清理，批量删除会被安全策略拦截（且留下垃圾）。
    现在只有校验通过的字节才落盘，文件名用内容哈希，天然去重。
    """
    try:
        from PIL import Image
        import hashlib
        im = Image.open(io.BytesIO(buf))
        w, h = im.size
        if w < 700 or h < 400 or w / max(1, h) < 1.15:
            return None
        im.verify()
        return (w, h, hashlib.md5(buf).hexdigest()[:16])
    except Exception:
        return None


def fetch(url, timeout=25):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Referer": "https://image.so.com/",
        "Accept": "image/avif,image/webp,image/jpeg,image/*,*/*;q=0.8",
    })
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE          # 本机代理是自签证书
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read(12 * 1024 * 1024)


def download(item, outdir):
    """下载 → 内存校验 → 内容哈希命名落盘。返回 (path, name, existed)。"""
    url = item["url"]
    for attempt in range(3):
        try:
            buf = fetch(url)
        except Exception:
            time.sleep(0.8 * (attempt + 1))
            continue
        info = verify_bytes(buf)
        if not info:
            continue                          # 拿到 HTML / 尺寸不够 → 换下一次重试
        w, h, digest = info
        name = digest + ".jpg"
        path = os.path.join(outdir, name)
        if os.path.exists(path):
            return path, name, True
        with open(path, "wb") as f:
            f.write(buf)
        return path, name, False
    return None, "", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=520)
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--workers", type=int, default=6)
    # 翻页深度可调：语料从 300 张扩到 1000+ 时，580 个候选明显不够（去重后）
    ap.add_argument("--pages", type=int, default=len(DEFAULT_PAGES))
    ap.add_argument("--min-w", type=int, default=1100)
    args = ap.parse_args()
    pages = DEFAULT_PAGES[:max(1, min(len(DEFAULT_PAGES), args.pages))]
    MIN_W = args.min_w

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path, encoding="utf-8"))
    seen = {m["file"] for m in manifest}

    print("[1/2] 收集候选 URL …")
    cands = []
    lock = threading.Lock()

    def work(q):
        got = gather(q[1], pages, MIN_W)
        with lock:
            cands.extend([dict(c, cat=q[0]) for c in got])
        print("   %-8s %-16s +%d" % (q[0], q[1], len(got)))

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(work, QUERIES))

    # 去重
    uniq, urls = [], set()
    for c in cands:
        if c["url"] in urls:
            continue
        urls.add(c["url"])
        uniq.append(c)
    print("   候选 %d 条（去重后）" % len(uniq))

    print("[2/2] 下载图片 …")
    need = max(0, args.target - len(manifest))
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download, c, args.out): c for c in uniq[: need * 3]}
        for f in as_completed(futs):
            c = futs[f]
            try:
                path, name, existed = f.result()
            except Exception:
                continue
            if not path:
                continue
            if name in seen:
                continue
            seen.add(name)
            manifest.append({
                "file": name, "cat": c["cat"], "query": c["query"],
                "src": c["url"], "site": c["site"], "title": c["title"],
                "w": c["w"], "h": c["h"],
            })
            done += 1
            if done % 25 == 0:
                print("   已下载 %d / %d" % (len(manifest), args.target))
                json.dump(manifest, open(manifest_path, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
            if len(manifest) >= args.target:
                break

    json.dump(manifest, open(manifest_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # 收尾自检：只「读取 + 核对」，不删文件（批量删除会被安全策略拦截）。
    # 磁盘上可能残留旧版留下的不可读文件，它们不进入 manifest，也就不会进入语料。
    good, dropped = [], 0
    for m in manifest:
        p = os.path.join(args.out, m["file"])
        try:
            from PIL import Image
            with Image.open(p) as im:
                w, h = im.size
            if w >= 700 and h >= 400 and w / max(1, h) >= 1.15:
                good.append(m)
                continue
        except Exception:
            pass
        dropped += 1
    if dropped:
        manifest = good
        json.dump(manifest, open(manifest_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print("完成：可训练 %d 张（剔除损坏 %d）→ %s" % (len(manifest), dropped, manifest_path))


if __name__ == "__main__":
    main()
