#!/usr/bin/env python
"""T3 EXIF 摄取的数值验收（v1.22）。python tools/test_exif.py

覆盖：f35 = 13 / 26 / 200 / 300（旧区间 [20,200] 会把 13 和 300 拒掉）、
     物理焦距 + FocalPlaneXResolution（真实传感器宽度）、
     物理焦距但无传感器信息（标低置信度）、完全无 EXIF（低置信度 + 26mm）。
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import exif as exifmod  # noqa: E402

FAIL = []


class FakeImg:
    """最小 EXIF 壳：getexif() 返回 dict 语义的映射。"""

    def __init__(self, size, tags):
        self.size = size
        self._tags = tags

    def getexif(self):
        return self._tags


def run(name, size, tags, want_hfov_deg, want_conf, tol=0.6):
    img = FakeImg(size, tags)
    hfov, src, meta = exifmod.hfov_from_image(img)
    deg = math.degrees(hfov)
    ok = abs(deg - want_hfov_deg) <= tol and meta.get("fov_confidence") == want_conf
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} hfov={deg:7.2f}° (期望 {want_hfov_deg:6.2f}°) "
          f"conf={meta.get('fov_confidence')} src={src}")
    if not ok:
        FAIL.append(name)


def main() -> int:
    print("T3 EXIF → 水平视场角")
    size = (4000, 3000)
    # 1) 35mm 等效 13mm（超广角）：旧区间 [20,200] 直接拒 → 落 26mm(69.4°)
    run("f35=13 超广角", size, {0xA405: 13}, 2 * math.degrees(math.atan(36 / 26.0)), "high")
    # 2) 26mm 手机主摄
    run("f35=26 主摄", size, {0xA405: 26}, 2 * math.degrees(math.atan(36 / 52.0)), "high")
    # 3) 200mm 长焦
    run("f35=200 长焦", size, {0xA405: 200}, 2 * math.degrees(math.atan(36 / 400.0)), "high")
    # 4) 300mm（旧区间上限 200 会拒）
    run("f35=300 远摄", size, {0xA405: 300}, 2 * math.degrees(math.atan(36 / 600.0)), "high")

    # 5) 物理焦距 4.3mm + 传感器 6.2mm 宽（1/2.3" 手机）：fpx 使 sensor≈6.2mm
    #    sensor = w * 10 / fpx  →  fpx = w*10/6.2 = 4000*10/6.2 = 6451.6
    fpx = 4000 * 10.0 / 6.2
    run("focal=4.3 传感器 6.2mm", size, {0x920A: 4.3, 0xA20E: fpx, 0xA210: 3},
        2 * math.degrees(math.atan(6.2 / (2 * 4.3))), "high")

    # 6) 物理焦距 17.3mm，无传感器信息 → 假设 36mm，但必须标低置信度
    run("focal=17.3 无传感器", size, {0x920A: 17.3},
        2 * math.degrees(math.atan(36 / (2 * 17.3))), "low")

    # 7) 长焦 250mm 物理焦距（旧代码 hfov 会被 <0.35 的消息滤掉）
    run("focal=250 长焦", size, {0x920A: 250},
        2 * math.degrees(math.atan(36 / 500.0)), "low")

    # 8) 完全无 EXIF → 26mm 默认 + 低置信度
    run("无 EXIF", size, {}, 2 * math.degrees(math.atan(36 / 52.0)), "low")

    print()
    if FAIL:
        print(f"✗ {len(FAIL)} 项未通过：" + "、".join(FAIL))
        return 3
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
