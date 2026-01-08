from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .io import read_json_or_jsonl
from .path_resolver import resolve_image_path

## 映射不同数据集字段成为统一样本格式
@dataclass
class UnifiedSample:
    image_path: str
    caption: str
    dataset: str
    sample_id: str
    meta: Dict[str, Any]

def build_samples_from_source(
    *,
    dataset_name: str,
    ann_path: str,
    root: str,
    image_key: str,
    text_key: str,
    id_key: Optional[str] = None,
) -> List[UnifiedSample]:
    ## 读取json文件
    records = read_json_or_jsonl(ann_path)
    ## 初始化输出列表
    samples: List[UnifiedSample] = []

    for idx, r in enumerate(records):
        if image_key not in r or text_key not in r:
            continue

        rel_img = r[image_key]
        cap = r[text_key]
        ## 拼凑图片完整路径
        img_abs = resolve_image_path(root, rel_img, ann_path)

        # id priority: explicit id_key -> fallback to image_path -> fallback to idx
        if id_key and id_key in r:
            sid = str(r[id_key])
        elif isinstance(rel_img, str) and rel_img:
            sid = rel_img
        else:
            sid = f"{dataset_name}:{idx}"

        meta = {k: v for k, v in r.items() if k not in [image_key, text_key]}
        ## 将数据整合加入样本列表
        samples.append(UnifiedSample(
            image_path=img_abs,
            caption=cap,
            dataset=dataset_name,
            sample_id=sid,
            meta=meta,
        ))

    return samples
