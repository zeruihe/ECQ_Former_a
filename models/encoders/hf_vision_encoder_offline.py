from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image

from transformers import AutoModel, AutoImageProcessor, AutoProcessor
from transformers import PreTrainedTokenizerBase, CLIPImageProcessor


@dataclass
class VisionTokens:
    tokens: torch.Tensor
    attn_mask: Optional[torch.Tensor] = None


class HFVisionEncoderOffline(nn.Module):
    """
    Offline-only HF vision encoder wrapper.
    Loads model & processor from local_dir with local_files_only=True.
    Outputs: last_hidden_state (B, N, D).
    """

    def __init__(self, local_dir: str, torch_dtype=torch.bfloat16):
        super().__init__()
        self.local_dir = local_dir
        self.torch_dtype = torch_dtype

        # processor
        # processor (prefer image processor)
        self.processor = None
        proc_err = None

        # 1) try AutoImageProcessor first
        try:
            self.processor = AutoImageProcessor.from_pretrained(local_dir, local_files_only=True)
        except Exception as e:
            proc_err = e

        # 2) fallback to AutoProcessor
        if self.processor is None:
            try:
                self.processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
            except Exception as e:
                proc_err = e

        if self.processor is None:
            raise RuntimeError(f"Failed to load processor from {local_dir}: {proc_err}")

        # 3) If it resolves to a tokenizer (text-only), force an image processor.
        #    This happens for some CLIP/OpenCLIP-derived repos (e.g., BiomedCLIP).
        if isinstance(self.processor, PreTrainedTokenizerBase):
            try:
                self.processor = CLIPImageProcessor.from_pretrained(local_dir, local_files_only=True)
            except Exception as e:
                raise RuntimeError(
                    f"Processor for {local_dir} is a tokenizer, and CLIPImageProcessor load failed: {e}\n"
                    "Please ensure the repo directory contains preprocessor_config.json / image processor files."
                )

        # model
        self.model = AutoModel.from_pretrained(
            local_dir,
            local_files_only=True,
            torch_dtype=torch_dtype,
        )

        cfg = getattr(self.model, "config", None)
        out_dim = getattr(cfg, "hidden_size", None)
        if out_dim is None and hasattr(cfg, "vision_config"):
            out_dim = getattr(cfg.vision_config, "hidden_size", None)
        if out_dim is None:
            raise ValueError(f"Cannot infer hidden_size from {local_dir}")
        self.out_dim = int(out_dim)

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.inference_mode()
    def forward(self, images_pil: List[Image.Image], device: torch.device) -> VisionTokens:
        inputs = self.processor(images=images_pil, return_tensors="pt")
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        pixel_values = inputs.get("pixel_values", None)

        # CLIP-like: has vision_model; internal vision_model may NOT accept return_dict
        if hasattr(self.model, "vision_model") and pixel_values is not None:
            out = self.model.vision_model(pixel_values=pixel_values)  # no return_dict
            tokens = getattr(out, "last_hidden_state", None)
            if tokens is None:
                tokens = out[0]
            return VisionTokens(tokens=tokens)

        # ViT/DINOv2-like
        if pixel_values is not None:
            out = self.model(pixel_values=pixel_values)  # no return_dict
            tokens = getattr(out, "last_hidden_state", None)
            if tokens is None:
                tokens = out[0]
            return VisionTokens(tokens=tokens)

        # fallback
        out = self.model(**inputs)
        tokens = getattr(out, "last_hidden_state", None)
        if tokens is None:
            tokens = out[0]
        return VisionTokens(tokens=tokens)