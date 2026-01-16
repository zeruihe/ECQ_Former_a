from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from PIL import Image

import torch
from torch.utils.data import Dataset


@dataclass
class VQAExample:
    image_pil: Image.Image
    question: str
    answer: str
    qid: str
    is_closed: bool
    meta: Dict[str, Any]


def _norm_text(x: str) -> str:
    return " ".join((x or "").strip().lower().split())


def _is_closed_answer(ans: str) -> bool:
    a = _norm_text(ans)
    return a in {"yes", "no"}


class HuggingFaceVQADataset(Dataset):
    """Lightweight adapter for Med-VQA datasets downloaded from Hugging Face.

    Design goals:
    - Works in OFFLINE mode (datasets are already cached on disk).
    - Returns PIL images, so each vision encoder can apply its own processor.
    - Robust to mild schema differences across PathVQA / VQA-RAD / SLAKE.
    """

    def __init__(
        self,
        *,
        dataset_id: str,
        split: str,
        cache_dir: Optional[str] = None,
        local_dir: Optional[str] = None,
        images_dir: Optional[str] = None,
        language: str = "en",
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_id = dataset_id
        self.split = split
        self.cache_dir = cache_dir
        self.local_dir = local_dir
        self.images_dir = images_dir
        self.language = language
        self.max_samples = max_samples

        # Load dataset
        self.ds = self._load_dataset()

        if max_samples is not None:
            self.ds = self.ds.select(range(min(int(max_samples), len(self.ds))))

    def _load_dataset(self):
        try:
            from datasets import load_dataset, load_from_disk
        except Exception as e:
            raise RuntimeError(
                "Missing dependency: datasets. Please `pip install datasets` in your env."
            ) from e

        # 1) Prefer load_from_disk if the user saved a dataset locally.
        if self.local_dir and os.path.isdir(self.local_dir):
            cand = os.path.join(self.local_dir, "dataset_dict.json")
            if os.path.exists(cand):
                dd = load_from_disk(self.local_dir)
                # dd can be DatasetDict
                if hasattr(dd, "__getitem__") and self.split in dd:
                    return dd[self.split]
                return dd

        # 2) Otherwise, rely on HF cache (offline).
        return load_dataset(self.dataset_id, split=self.split, cache_dir=self.cache_dir)

    def __len__(self) -> int:
        return len(self.ds)

    def _get_lang_field(self, v: Any) -> Any:
        # Some datasets store bilingual fields as dicts.
        if isinstance(v, dict):
            if self.language in v:
                return v[self.language]
            if "en" in v:
                return v["en"]
            # fall back to any value
            return next(iter(v.values()))
        return v

    def _get_image_pil(self, ex: Dict[str, Any]) -> Image.Image:
        # HF Image feature
        if "image" in ex and ex["image"] is not None:
            im = ex["image"]
            if isinstance(im, Image.Image):
                return im.convert("RGB")
            # datasets.Image may return a dict with 'path' or bytes
            if isinstance(im, dict) and "path" in im and os.path.exists(im["path"]):
                return Image.open(im["path"]).convert("RGB")

        # Common SLAKE keys
        for k in ["img_path", "img", "img_name", "image_path", "image_id", "img_id"]:
            if k in ex and ex[k]:
                rel = str(ex[k])
                if self.images_dir:
                    cand = os.path.join(self.images_dir, rel)
                    if os.path.exists(cand):
                        return Image.open(cand).convert("RGB")
                    # sometimes rel contains only basename
                    cand2 = os.path.join(self.images_dir, os.path.basename(rel))
                    if os.path.exists(cand2):
                        return Image.open(cand2).convert("RGB")
                # allow absolute
                if os.path.isabs(rel) and os.path.exists(rel):
                    return Image.open(rel).convert("RGB")

        raise RuntimeError(
            f"Cannot resolve image for dataset={self.dataset_id}. "
            f"If this is SLAKE, please set data.images_dir to the extracted image folder."
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[idx]

        # question/answer
        q = self._get_lang_field(ex.get("question", ex.get("query", "")))
        a = self._get_lang_field(ex.get("answer", ex.get("label", "")))
        q = str(q)
        a = str(a)

        # id
        qid = ex.get("qid", ex.get("question_id", ex.get("id", idx)))
        qid = str(qid)

        # closed/open
        if "answer_type" in ex and ex["answer_type"] is not None:
            at = str(ex["answer_type"]).lower()
            is_closed = at in {"closed", "yes/no", "yesno", "binary"}
        else:
            is_closed = _is_closed_answer(a)

        image_pil = self._get_image_pil(ex)

        return {
            "image_pil": image_pil,
            "question": q,
            "answer": a,
            "qid": qid,
            "is_closed": is_closed,
            "meta": {"dataset_id": self.dataset_id, "split": self.split},
        }
