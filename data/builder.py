from __future__ import annotations
from typing import List
from torch.utils.data import DataLoader

from .schema import DataSourceCfg
from .sources import build_samples_from_source
from .pretain_dataset import CaptionDataset, ConcatCaptionDataset
from .collate import PretrainCaptionCollator

def build_caption_dataset(split: str, sources: List[DataSourceCfg]):
    datasets = []
    for src in sources:
        ann_path = getattr(src.ann, split)
        samples = build_samples_from_source(
            dataset_name=src.name,
            ann_path=ann_path,
            root=src.root,
            image_key=src.image_key,
            text_key=src.text_key,
            id_key=src.id_key,
        )
        datasets.append(CaptionDataset(samples))
    return ConcatCaptionDataset(datasets)

def build_caption_loader(
    *,
    split: str,
    sources: List[DataSourceCfg],
    batch_size: int,
    num_workers: int,
    return_pil: bool,
    pin_memory: bool,
    shuffle: bool,
):
    ds = build_caption_dataset(split, sources)
    collate = PretrainCaptionCollator(return_pil=return_pil)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        drop_last=False,
    )
