"""EXIF 解析：焦距 → 水平视场角，等比处理。

没有 EXIF 时不给"魔法常数"，而是按 35mm 等效焦距的常见取值给一个有依据的默认值，
并把不确定性写进产物元数据（前端可以据此提示用户）。
"""
from __future__ import annotations

import math
from typing import Optional


def _ratio(value) -> Optional[float]:
    try:
        if isinstance(value, tuple) and len(value) == 2 and value[1]:
            return float(value[0]) / float(value[1])
        f = float(value)
        return f if f > 0 else None
    except Exception:
        return None


def hfov_from_image(img) -> tuple[float, str, dict]:
    """返回 (水平视场角弧度, 来源说明, 元数据)。

    优先级：EXIF 35mm 等效焦距 → EXIF 物理焦距 + 传感器宽度 → 默认。
    注意：竖直方向按等比换算，绝不对照片做非等比缩放。
    """
    meta: dict = {}
    exif = getattr(img, "getexif", lambda: None)()
    w, h = img.size

    f35 = None
    focal = None
    sensor = None
    if exif:
        try:
            # 0xA405 FocalLengthIn35mmFilm, 0x920A FocalLength, 0xA002/0xA003 尺寸
            f35 = _ratio(exif.get(0xA405))
            focal = _ratio(exif.get(0x920A))
            # 0x0002 FocalPlaneXResolution 等一般不可靠，这里只用前两项
        except Exception:
            pass

    # T3（v1.22）：区间太窄会把真实的超广角/长焦全部拒掉，然后悄悄落到 26mm 默认值。
    #   13mm 等效（运动相机/手机超广）和 250mm 等效（长焦）都是常见输入，
    #   旧区间 [20,200] 把它们分别推到 26mm —— 视场角错 26%，而验收全绿。
    if f35 and 8.0 <= f35 <= 400.0:
        hfov = 2.0 * math.atan(36.0 / (2.0 * f35))
        # 0.06 rad ≈ 3.4°：覆盖到 400mm 等效（5.15°）。0.14(8°) 会把 300mm 拒掉。
        if 0.06 <= hfov <= 2.4:
            meta["focal35"] = f35
            meta["fov_confidence"] = "high"
            meta["fov_source"] = "exif-35mm"
            return hfov, "exif-35mm", meta

    # 物理焦距分支：**先读 FocalPlaneXResolution 求真实传感器宽度**，
    #   查不到才退回 36mm 全画幅假设，并明确标成低置信度。
    #   旧实现无条件假设 36mm：1/2.3" 手机 CMOS（宽约 6.2mm）会把视场角算大 5 倍。
    sensor_w = None
    try:
        fpx = _ratio(exif.get(0xA20E)) if exif else None      # FocalPlaneXResolution
        unit = exif.get(0xA210) if exif else None             # FocalPlaneResolutionUnit
        if fpx and fpx > 1.0:
            # 单位 3 = cm（Exif 标准），2 = inch；像素数取照片宽度
            per_unit = 10.0 if (unit in (3, "3") or unit is None) else 25.4
            sensor_w = float(w) * per_unit / fpx
            if not (2.0 <= sensor_w <= 90.0):                 # 物理上不合理的值直接丢
                sensor_w = None
    except Exception:
        sensor_w = None

    if focal and 2.0 <= focal <= 400.0:
        sw = sensor_w if sensor_w else 36.0
        hfov = 2.0 * math.atan(sw / (2.0 * focal))
        if 0.06 <= hfov <= 2.4:
            meta["focal"] = focal
            meta["sensor_w_mm"] = round(sw, 3)
            meta["fov_confidence"] = "high" if sensor_w else "low"
            meta["fov_source"] = "exif-focal"
            return hfov, ("exif-focal(sensor)" if sensor_w else "exif-focal(assume36mm)"), meta

    # 无 EXIF：按"手机主摄 26mm 等效"，但必须标成低置信度，让上层能据此拒绝验收
    hfov = 2.0 * math.atan(36.0 / (2.0 * 26.0))
    meta["note"] = "no-exif: assume 26mm equiv"
    meta["fov_confidence"] = "low"
    meta["fov_source"] = "default-26mm"
    return hfov, "default-26mm", meta


def vfov_of(hfov: float, w: int, h: int) -> float:
    """由水平视场角与图像宽高比得到垂直视场角（等比）。"""
    return 2.0 * math.atan(math.tan(hfov * 0.5) * (h / float(w)))


def intrinsics(hfov: float, w: int, h: int) -> dict:
    fx = 0.5 * w / math.tan(hfov * 0.5)
    fy = fx  # 方形像素
    return {"fx": fx, "fy": fy, "cx": 0.5 * w, "cy": 0.5 * h, "w": w, "h": h}
