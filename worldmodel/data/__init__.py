"""训练数据层：webdataset tar 分片读取。

数据由 `python scripts/prepare_data.py --traj ... --out data/wm` 生成。
"""
from .dataset import TarWindows, collate, split_dataset

__all__ = ["TarWindows", "collate", "split_dataset"]
