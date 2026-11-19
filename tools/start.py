#!/usr/bin/env python3
"""PHOTOWORLD 统一启动器 —— 零第三方依赖，可从任意目录启动。

设计目标（针对旧版 .ps1 启动器的实际故障）：
  * 不依赖 PowerShell 执行策略：双击 .bat 或直接 `python tools/start.py` 都可用；
  * 不因为 v2 的可选重依赖（fastapi/uvicorn/torch/transformers）缺失就整个启动失败
    —— 静态站和 v3 资产路径不需要它们；
  * 只打开确实存在的资产的 URL，不再把浏览器送到 404 的 world-v3-*.glb；
  * 后台服务脱离启动器进程组，关掉启动窗口也不会被一起回收；
  * 停止时按 PID 文件精确关闭，不误杀无关进程；
  * 默认不提问、不阻塞（需要确认时用 --interactive 显式开启）。

用法：
    python tools/start.py                      # 静态站 + 首页（画廊）
    python tools/start.py --world alpine       # 直接进入 alpine 世界（优先本地 GLB）
    python tools/start.py --v2                 # 附带尝试启动 v2 GPU 服务（缺依赖则跳过）
    python tools/start.py --photo D:/pics/a.jpg --out out/v3/a
    python tools/start.py --stop               # 关闭本启动器起的服务
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
PID_FILE = LOG_DIR / "pids_v3.json"


QUIET = False


def say(msg: str = "", kind: str = "info") -> None:
    if QUIET and kind != "err":
        return
    prefix = {"ok": "[OK]  ", "warn": "[WARN]", "err": "[ERR] ", "info": "       "}.get(kind, "       ")
    print(prefix + " " + msg, flush=True)


def http_ok(url: str, timeout: float = 1.5) -> bool:
    """探测 HTTP 是否就绪。显式绕过会话代理（本机 127.0.0.1 不该走代理）。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as r:
            return 200 <= r.status < 400
    except Exception:
        return False


def port_owner(port: int) -> int:
    """返回监听 port 的 PID；用 netstat，Windows 全版本可用，不依赖 PowerShell 模块。"""
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"], capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=8,
        ).stdout or ""
    except Exception:
        return 0
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 5 and f[3].upper() == "LISTENING":
            if f[1].endswith(":" + str(port)):
                try:
                    return int(f[4])
                except ValueError:
                    pass
    return 0


def is_python_pid(pid: int) -> bool:
    """确认 PID 是 python 进程，避免误杀抢占端口的其他程序。"""
    if pid <= 0:
        return False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "PID eq %d" % pid, "/FO", "CSV", "/NH"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8,
        ).stdout or ""
    except Exception:
        return True  # 查不到时仍按 PID 文件记录处理（该 PID 由本脚本写入）
    low = out.lower()
    return ("python" in low) or (not low.strip()) or ("no tasks" in low)


def find_python_with(modules: list[str]) -> str | None:
    """找一个装了指定模块的 Python；用于 v2 可选服务，找不到就跳过而不是失败。"""
    candidates: list[str] = []
    py = shutil.which("py")
    if py:
        candidates.append(py + " -3.11")
        candidates.append(py + " -3")
    for name in ("python", "python3"):
        p = shutil.which(name)
        if p and p not in candidates:
            candidates.append(p)
    for base in (Path.home() / "AppData/Local/Programs/Python",):
        candidates.extend(str(p) for p in sorted(base.glob("Python3*/python.exe"), reverse=True))
    code = (
        "import importlib.util as u,sys;"
        "m=" + repr(modules) + ";"
        "sys.exit(0 if all(u.find_spec(x) for x in m) else 1)"
    )
    for cmd in candidates:
        argv = cmd.split() if " " in cmd else [cmd]
        try:
            r = subprocess.run(argv + ["-c", code], capture_output=True, timeout=8)
            if r.returncode == 0:
                return cmd
        except Exception:
            continue
    return None


def start_background(argv: list[str], cwd: Path, log_name: str, visible: bool = False) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = (LOG_DIR / log_name).open("w", encoding="utf-8", errors="replace")
    err = (LOG_DIR / log_name.replace(".log", ".err.log")).open("w", encoding="utf-8", errors="replace")
    env = dict(os.environ)
    # 本机 WorkBuddy 会话会注入短命代理端口；子进程继承后容易 ECONNREFUSED。
    # 本地回环明确不走代理，其余保持不变。
    env["NO_PROXY"] = (env.get("NO_PROXY", "") + ",127.0.0.1,localhost,::1").strip(",")
    env["no_proxy"] = env["NO_PROXY"]
    flags = 0
    if os.name == "nt" and not visible:
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    p = subprocess.Popen(argv, cwd=str(cwd), stdout=out, stderr=err, env=env, creationflags=flags)
    out.close()
    err.close()
    return p.pid


def wait_http(url: str, pid: int, seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if http_ok(url, 0.6):
            return True
        if pid > 0 and not is_python_pid(pid):
            return False
        time.sleep(0.25)
    return False


def save_pids(data: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_pids() -> dict:
    try:
        return json.loads(PID_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def do_stop(port: int, api_port: int) -> int:
    say("停止 PHOTOWORLD 服务…", "ok")
    killed = []
    seen = set()
    info = read_pids()
    for key in ("static", "api"):
        pid = int(info.get(key) or 0)
        if pid > 0 and pid not in seen:
            seen.add(pid)
            if is_python_pid(pid):
                try:
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=8)
                    killed.append("%s=%d" % (key, pid))
                except Exception:
                    pass
    for label, p in (("static", port), ("api", api_port)):
        pid = port_owner(p)
        if pid > 0 and pid not in seen and is_python_pid(pid):
            seen.add(pid)
            try:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=8)
                killed.append("port%d=%d" % (p, pid))
            except Exception:
                pass
    try:
        PID_FILE.unlink()
    except Exception:
        pass
    time.sleep(0.5)
    left = [p for p in (port, api_port) if port_owner(p) > 0]
    if killed:
        say("已停止：" + ", ".join(killed), "ok")
    else:
        say("没有本启动器在跑的服务。", "info")
    if left:
        say("注意：端口仍被占用 " + ", ".join(str(x) for x in left) + "（可能是别的程序）", "warn")
    else:
        say("端口 %d / %d 已释放" % (port, api_port), "ok")
    return 0


def resolve_asset(world: str) -> tuple[str, str] | None:
    """返回 (URL 参数值, 来源描述)；按 v3 → out → v2 内置 GLB 顺序挑存在的。"""
    candidates = [
        ("assets/world-v3-%s.glb" % world, "v3 预烘"),
        ("out/v3/worlds/%s/world.glb" % world, "v3 烘焙"),
        ("out/v3/%s/world.glb" % world, "v3 烘焙"),
        ("assets/world-%s.glb" % world, "v2 内置 GLB"),
    ]
    for rel, src in candidates:
        if (ROOT / rel).is_file():
            return rel, src
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="PHOTOWORLD 统一启动器（Python，零依赖）")
    ap.add_argument("--stop", action="store_true", help="停止本启动器管理的服务")
    ap.add_argument("--no-browser", action="store_true", help="不开浏览器")
    ap.add_argument("--visible", action="store_true", help="显示服务窗口（默认隐藏）")
    ap.add_argument("--quiet", action="store_true", help="只输出错误")
    ap.add_argument("--port", type=int, default=8199, help="静态站端口（默认 8199）")
    ap.add_argument("--api-port", type=int, default=8100, help="v2 GPU 服务端口（默认 8100）")
    ap.add_argument("--v2", action="store_true", help="尝试启动 v2 GPU 服务（缺依赖则跳过）")
    ap.add_argument("--world", default="", help="直接进入内置世界：alpine|desert|coast|winter")
    ap.add_argument("--photo", default="", help="先烘焙照片再打开（需要 v3 重依赖）")
    ap.add_argument("--out", default="", help="配合 --photo 的输出目录")
    ap.add_argument("--no-diffusion", action="store_true", help="配合 --photo：只做视锥内")
    ap.add_argument("--interactive", action="store_true", help="出错时等待回车（默认不阻塞）")
    args = ap.parse_args()
    global QUIET
    QUIET = args.quiet

    if args.stop:
        return do_stop(args.port, args.api_port)

    static_url = "http://127.0.0.1:%d" % args.port
    api_url = "http://127.0.0.1:%d" % args.api_port
    say("PHOTOWORLD 启动器")
    say("仓库：" + str(ROOT), "info")
    if not (ROOT / "index.html").is_file():
        say("找不到 index.html —— 请把 start.py 放在仓库 tools/ 目录下", "err")
        return 1

    info = read_pids()

    # ---- 1) 静态站 ----
    static_pid = int(info.get("static") or 0)
    if http_ok(static_url + "/index.html"):
        say("静态站已在运行（复用）：" + static_url, "ok")
        static_pid = port_owner(args.port) or static_pid
    else:
        owner = port_owner(args.port)
        if owner > 0:
            say("端口 %d 被 PID=%d 占用且不是 PHOTOWORLD 静态站。" % (args.port, owner), "err")
            say("换端口：python tools/start.py --port 8200", "info")
            return 1
        static_pid = start_background(
            [sys.executable, "-m", "http.server", str(args.port), "--bind", "127.0.0.1"],
            ROOT, "v3_static.log", args.visible,
        )
        if not wait_http(static_url + "/index.html", static_pid, 12):
            say("静态站没起来。看日志：logs/v3_static.err.log", "err")
            return 1
        say("静态站就绪（PID=%d）：%s" % (static_pid, static_url), "ok")

    # ---- 1b) v4 重建服务（本机计算；失败只警告，不影响看世界）----
    v4_url = "http://127.0.0.1:8200"
    if not http_ok(v4_url + "/api/v4/status?job=probe", 1.0):
        try:
            v4_pid = start_background(
                [sys.executable, str(ROOT / "tools" / "serve_api.py"),
                 "--port", "8200", "--static", str(ROOT)],
                ROOT, "v4_api.log", args.visible,
            )
            if wait_http(v4_url + "/api/v4/status?job=probe", v4_pid, 8):
                say("v4 重建服务就绪（PID=%d）：%s" % (v4_pid, v4_url), "ok")
            else:
                say("v4 重建服务没起来（多图重建不可用，看世界不受影响）。", "warn")
        except Exception as e:
            say("v4 重建服务启动失败：%s（多图重建不可用）" % e, "warn")
    else:
        say("v4 重建服务已在运行（复用）：%s" % v4_url, "ok")

    # ---- 2) v2 GPU 服务（可选；缺依赖只警告，不拖垮静态站）----
    api_pid = 0
    if args.v2:
        if http_ok(api_url + "/health"):
            say("v2 GPU 服务已在运行（复用）：" + api_url, "ok")
            api_pid = port_owner(args.api_port)
        else:
            need = ["fastapi", "uvicorn", "torch", "transformers", "scipy"]
            py = find_python_with(need)
            if not py:
                say("跳过 v2 GPU 服务：找不到装有 %s 的 Python。" % ",".join(need), "warn")
                say("安装：python -m pip install fastapi uvicorn torch transformers scipy pillow numpy", "info")
                say("页面仍可用（在线链路 / v3 资产），只是没有 v2 重建加速。", "info")
            else:
                argv = py.split() if " " in py else [py]
                api_pid = start_background(
                    argv + ["-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(args.api_port)],
                    ROOT / "server", "api.log", args.visible,
                )
                if wait_http(api_url + "/health", api_pid, 90):
                    say("v2 GPU 服务就绪：" + api_url, "ok")
                else:
                    say("v2 GPU 服务未就绪（不影响静态站）。看日志：logs/api.err.log", "warn")
                    api_pid = 0

    # ---- 3) 目标 URL ----
    # 默认进**主页**，并带 ?pipe=v3：主页徽标与画廊都按 v3（离线资产）走。
    # 不带这个参数时，页面会先探测到本机 v2 服务并把徽标写成 "v2 就绪"，
    # 用户会以为启动器起的还是 v2（实际反馈过）。
    target = static_url + "/?pipe=v3"
    if args.photo:
        src = Path(args.photo)
        if not src.is_file():
            say("照片不存在：" + str(src), "err")
            return 1
        out_dir = ROOT / (args.out or ("out/v3/" + src.stem))
        need = ["torch", "transformers", "scipy", "PIL", "numpy"]
        py = find_python_with(need)
        if not py:
            say("烘焙需要 %s，当前未安装。" % ",".join(need), "err")
            say("安装：python -m pip install torch transformers scipy pillow numpy", "info")
            return 1
        argv = py.split() if " " in py else [py]
        bake = [argv[0]] + (argv[1:] if len(argv) > 1 else []) + [
            str(ROOT / "pipeline/v3/bake.py"), str(src), "--out", str(out_dir), "--step", "4"
        ]
        if args.no_diffusion:
            bake.append("--no-diffusion")
        say("烘焙中（约 2 分钟/张）：" + str(src), "warn")
        r = subprocess.run(bake, cwd=str(ROOT))
        if r.returncode != 0:
            say("烘焙失败（退出码 %d）" % r.returncode, "err")
            return r.returncode
        glb = out_dir / "world.glb"
        if glb.is_file():
            target = static_url + "/?pipe=v3&glb=" + urllib.request.quote(
                str(glb.relative_to(ROOT)).replace("\\", "/"), safe="/"
            ) + "&instant=1"
    elif args.world:
        found = resolve_asset(args.world)
        if found:
            rel, src = found
            target = static_url + "/?pipe=v3&glb=" + urllib.request.quote(rel, safe="/") + "&instant=1"
            say("世界 %s：%s（%s）" % (args.world, rel, src), "ok")
        else:
            target = static_url + "/?pipe=v3&w=" + urllib.request.quote(args.world) + "&instant=1"
            say("没有本地 GLB，走在线链路：" + target, "warn")
    if api_pid:
        sep = "&" if "?" in target else "?"
        target += sep + "api=" + urllib.request.quote(api_url, safe=":/")

    save_pids({"static": static_pid, "api": api_pid, "url": target, "started": time.strftime("%Y-%m-%dT%H:%M:%S")})

    say("")
    say("打开地址：" + target, "ok")
    if args.no_browser:
        say("(-no-browser) 请手动复制上面的地址。", "info")
    else:
        try:
            webbrowser.open(target)
            say("已打开浏览器。首次进世界要下载/重建，稍等。", "ok")
        except Exception as e:
            say("浏览器打开失败（%s），请手动访问上面的地址。" % e, "warn")
    say("关闭服务：python tools/start.py --stop", "info")
    say("日志目录：" + str(LOG_DIR), "info")
    say("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
