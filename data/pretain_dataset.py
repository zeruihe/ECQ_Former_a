from __future__ import annotations
from typing import Any, Dict, List
from torch.utils.data import Dataset

from .sources import UnifiedSample

## 合并多数据集数据，对pytorch统一
class CaptionDataset(Dataset):
    """
    Returns a dict for each sample:
      {
        "image_path": str,
        "text": str,
        "dataset": str,
        "sample_id": str,
        "meta": dict
      }
    """
    ## 初始化样本列表存入
    def __init__(self, samples: List[UnifiedSample]):
        self.samples = samples
    ## 返回样本数量
    def __len__(self) -> int:
        return len(self.samples)
    ## 返回指定索引的样本
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "image_path": s.image_path,
            "text": s.caption,
            "dataset": s.dataset,
            "sample_id": s.sample_id,
            "meta": s.meta,
        }

## 按照数量对数据集存储点位，使编码器可以输入编号，定位到某个数据集的某个索引中找到对应的样本
class ConcatCaptionDataset(Dataset):
    """Concatenate multiple CaptionDataset instances."""
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.cum = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cum.append(total)

    def __len__(self) -> int:
        return self.cum[-1] if self.cum else 0

    def __getitem__(self, idx: int):
        # binary search
        lo, hi = 0, len(self.cum) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if idx < self.cum[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        ds_i = lo
        prev = 0 if ds_i == 0 else self.cum[ds_i - 1]
        return self.datasets[ds_i][idx - prev]
