from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List

## 定义数据配置的数据类

@dataclass
class AnnSplitCfg:
    train: str
    valid: str
    test: str

@dataclass
class DataSourceCfg:
    name: str
    root: str
    ann: AnnSplitCfg
    image_key: str = "image"
    text_key: str = "caption"
    id_key: Optional[str] = None  # e.g., ROCOv2 uses "image_id"

@dataclass
class DataCfg:
    batch_size: int = 4
    num_workers: int = 4
    return_pil: bool = True
    pin_memory: bool = True
    sources: List[DataSourceCfg] = None
