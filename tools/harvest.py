#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHOTOWORLD 语料采集器
从 360 图片搜索按场景类别抓取横构图风景照，用于批量评测与参数标定。
仅本地使用（不参与发布），manifest 记录来源，方便复核。

用法:
  python tools/harvest.py --target 520 --out corpus
"""
import argparse, hashlib, json, os, subprocess, sys, threading, time
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
]

PAGES = [0, 60, 120, 180, 240]
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


def gather(query, pages):
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
            if w < MIN_W or h <= 0 or (w / max(h, 1)) < MIN_AR:
                continue
            out.append({
                "url": u.replace("http://", "https://"),
                "w": w, "h": h, "title": (it.get("title") or "")[:80],
                "site": it.get("site") or "",
                "query": query,
            })
        time.sleep(0.35)
    return out


def download(item, outdir):
    url = item["url"]
    ext = ".jpg"
    name = hashlib.md5(url.encode()).hexdigest()[:16] + ext
    path = os.path.join(outdir, name)
    if os.path.exists(path) and os.path.getsize(path) > 20000:
        return path, name, True
    p = subprocess.run(
        ["curl", "-k", "-s", "-m", "30", "-A", UA,
         "-H", "Referer: https://image.so.com/", "-o", path, url],
        capture_output=True)
    if not os.path.exists(path) or os.path.getsize(path) < 20000:
        return None, name, False
    return path, name, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=520)
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

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
        got = gather(q[1], PAGES)
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
    print("完成：%d 张，写入 %s" % (len(manifest), manifest_path))


if __name__ == "__main__":
    main()
