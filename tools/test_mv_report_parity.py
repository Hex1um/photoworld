#!/usr/bin/env python3
"""mv_report 入口一致性测试：mjs 薄包装与 py 直调输出须逐字节一致。

  python tools/test_mv_report_parity.py

在临时沙箱里放合成 reports JSON + 两个脚本，分别跑 node mjs / python py，
比对 stdout 字节。不触碰真实 reports/，避免污染。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJS = os.path.join(ROOT, "tools", "mv_report.mjs")
PY = os.path.join(ROOT, "tools", "mv_report.py")

SAMPLE = """{"results":[
 {"ok":true,"image":"worlds/a.jpg","views":[
   {"ring":"A","a":0,"file":"shots/a_A.png","voidFrac":0.02,"innerVoid":0.01,"lowVoid":0.03,"borderVoid":0.01,"darkFrac":0.02,"stripe":0.5,"detail":0.4,"hfEnergy":0.3,"farHF":0.2,"centerHF":0.35,"lapVar":0.1,"centerLapVar":0.12,"localContrast":0.2},
   {"ring":"C","a":180,"file":"shots/a_C.png","voidFrac":0.30,"innerVoid":0.20,"lowVoid":0.25,"borderVoid":0.05,"darkFrac":0.09,"stripe":3.5,"detail":0.2,"hfEnergy":0.4,"farHF":0.3,"centerHF":0.25,"lapVar":0.2,"centerLapVar":0.22,"localContrast":0.3}]},
 {"ok":true,"image":"worlds/b.jpg","views":[
   {"ring":"B","a":90,"file":"shots/b_B.png","voidFrac":0.05,"innerVoid":0.02,"lowVoid":0.04,"borderVoid":0.02,"darkFrac":0.01,"stripe":1.2,"detail":0.5,"hfEnergy":0.2,"farHF":0.1,"centerHF":0.3,"lapVar":0.05,"centerLapVar":0.08,"localContrast":0.15}]}
]}
"""


def test_mv_report_parity():
    d = tempfile.mkdtemp(prefix="pw_mvreport_")
    try:
        os.makedirs(os.path.join(d, "tools"))
        os.makedirs(os.path.join(d, "reports"))
        shutil.copyfile(MJS, os.path.join(d, "tools", "mv_report.mjs"))
        shutil.copyfile(PY, os.path.join(d, "tools", "mv_report.py"))
        data = os.path.join(d, "reports", "all32.json")
        with open(data, "w", encoding="utf-8", newline="") as f:
            f.write(SAMPLE)

        r1 = subprocess.run(["node", os.path.join(d, "tools", "mv_report.mjs"), data, "--top", "8"],
                            cwd=d, capture_output=True, timeout=30)
        r2 = subprocess.run([sys.executable, os.path.join(d, "tools", "mv_report.py"), data, "--top", "8"],
                            cwd=d, capture_output=True, timeout=30)
        assert r1.returncode == r2.returncode == 0, (r1.stderr, r2.stderr)
        assert r1.stdout == r2.stdout, "mjs/py 输出字节不一致"
        print("PASS  M1  mv_report mjs 包装与 py 直调输出逐字节一致（%d 行）"
              % len(r1.stdout.decode("utf-8", "replace").splitlines()))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_mv_report_ring_filter():
    d = tempfile.mkdtemp(prefix="pw_mvreport_")
    try:
        os.makedirs(os.path.join(d, "tools"))
        os.makedirs(os.path.join(d, "reports"))
        shutil.copyfile(PY, os.path.join(d, "tools", "mv_report.py"))
        data = os.path.join(d, "reports", "all32.json")
        with open(data, "w", encoding="utf-8", newline="") as f:
            f.write(SAMPLE)
        r = subprocess.run([sys.executable, os.path.join(d, "tools", "mv_report.py"),
                            data, "--ring", "C"],
                           cwd=d, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0 and "最差 stripe" in r.stdout, (r.returncode, r.stderr)
        # ring=C 只保留 a_C.png（stripe 3.5），不应出现 b_B
        assert "b_B.png" not in r.stdout, "--ring C 过滤未生效"
        print("PASS  M2  --ring 过滤生效（只统计 C 圈视角）")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    test_mv_report_parity()
    test_mv_report_ring_filter()
    print("\nmv_report 入口一致性全部通过（2 项）")
