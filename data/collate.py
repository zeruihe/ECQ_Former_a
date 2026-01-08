from __future__ import annotations
from typing import Any, Dict, List
from PIL import Image

def load_pil_rgb(path: str) -> Image.Image:
    im = Image.open(path)
    # 医学影像很多是灰度/单通道，这里统一转 RGB，避免后续 processor 报错
    return im.convert("RGB")

## 打包样本为batch
class PretrainCaptionCollator:
    """
    Collate for caption-style pretraining.
    Output is encoder-agnostic:
      - images_pil: List[PIL.Image]  (next step: each encoder uses its own processor)
      - captions:  List[str]
      - meta:      List[dict]
    """
    def __init__(self, return_pil: bool = True):
        self.return_pil = return_pil

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images_pil = []
        captions = []
        metas = []

        for ex in batch:
            images_pil.append(load_pil_rgb(ex["image_path"]))
            captions.append(ex["text"])
            metas.append({
                "dataset": ex.get("dataset"),
                "sample_id": ex.get("sample_id"),
                "image_path": ex.get("image_path"),
                **(ex.get("meta") or {}),
            })

        return {
            "images_pil": images_pil,
            "captions": captions,
            "meta": metas,
        }
