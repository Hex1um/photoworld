"""photoworld 的数据加载器：每个 item 就是一张待重建的照片。"""
from __future__ import annotations

import json
from pathlib import Path

from skillopt.datasets.base import SplitDataLoader


def _normalize(raw: dict) -> dict:
    return {
        "id": str(raw["id"]),
        # image 是相对 PHOTOWORLD 仓库根的路径
        "image": raw["image"],
        # task_type 用来做分层抽样：风景 / 建筑 / 人像 / 室内，各有各的难点
        "task_type": raw.get("task_type", "landscape"),
        # 允许给每张图一个"人工备注"，作为反射阶段的上下文
        "note": raw.get("note", ""),
    }


class PhotoWorldDataLoader(SplitDataLoader):
    """从 JSON 清单读 train / val / test。

    清单格式：{"train": [{...}], "val": [{...}], "test": [{...}]}
    """

    def load_split_items(self, split_path: str) -> list[dict]:
        p = Path(split_path)
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        if not files:
            raise FileNotFoundError(f"photoworld: 找不到 {split_path} 下的清单")

        items: list[dict] = []
        for f in files:
            raw = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "train" in raw:
                # 传进来的是总清单时，按 split 名过滤
                split_name = p.name if p.is_dir() else ""
                rows = raw.get(split_name) or raw.get("train") or []
            elif isinstance(raw, list):
                rows = raw
            else:
                rows = raw.get("items") or []
            items.extend(_normalize(r) for r in rows)
        return items
