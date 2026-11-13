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

    if f35 and 20 <= f35 <= 200:
        # 36mm 宽的 35mm 画幅
        hfov = 2.0 * math.atan(36.0 / (2.0 * f35))
        meta["focal35"] = f35
        return hfov, "exif-35mm", meta

    if focal and 4 <= focal <= 400:
        # 常见 1/2.3"~全画幅，取 36mm 宽作为保守假设
        hfov = 2.0 * math.atan(36.0 / (2.0 * focal))
        if 0.35 < hfov < 2.2:
            meta["focal"] = focal
            return hfov, "exif-focal(assume36mm)", meta

    # 无 EXIF：按"手机主摄 26mm 等效"，等比输出
    hfov = 2.0 * math.atan(36.0 / (2.0 * 26.0))
    meta["note"] = "no-exif: assume 26mm equiv"
    return hfov, "default-26mm", meta


def vfov_of(hfov: float, w: int, h: int) -> float:
    """由水平视场角与图像宽高比得到垂直视场角（等比）。"""
    return 2.0 * math.atan(math.tan(hfov * 0.5) * (h / float(w)))


def intrinsics(hfov: float, w: int, h: int) -> dict:
    fx = 0.5 * w / math.tan(hfov * 0.5)
    fy = fx  # 方形像素
    return {"fx": fx, "fy": fy, "cx": 0.5 * w, "cy": 0.5 * h, "w": w, "h": h}
