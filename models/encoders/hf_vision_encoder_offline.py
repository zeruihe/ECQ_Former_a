from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any, Dict

import torch
import torch.nn as nn
from PIL import Image

from transformers import AutoModel, AutoImageProcessor, AutoProcessor, PreTrainedTokenizerBase

# manual vision preprocess fallback (for repos without preprocessor_config.json, e.g., some BiomedCLIP layouts)
from torchvision import transforms
from torchvision.transforms import InterpolationMode


@dataclass
class VisionTokens:
    tokens: torch.Tensor
    attn_mask: Optional[torch.Tensor] = None


def _infer_image_size(model: nn.Module, default: int = 224) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return default
    # common patterns
    if hasattr(cfg, "image_size") and isinstance(cfg.image_size, int):
        return int(cfg.image_size)
    vc = getattr(cfg, "vision_config", None)
    if vc is not None and hasattr(vc, "image_size") and isinstance(vc.image_size, int):
        return int(vc.image_size)
    return default


class HFVisionEncoderOffline(nn.Module):
    """
    Offline-only HF vision encoder wrapper.

    - Loads model from local_dir with local_files_only=True
    - Tries to load an image processor; if it resolves to a tokenizer (text-only),
      falls back to manual CLIP-style image preprocessing to produce pixel_values.
    - Output: last_hidden_state (B, N, D)
    """

    def __init__(self, local_dir: str, torch_dtype=torch.bfloat16):
        super().__init__()
        self.local_dir = local_dir
        self.torch_dtype = torch_dtype

        # model first (so we can infer image_size for fallback preprocess)
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

        # processor
        self.processor: Any = None
        self.use_manual_preprocess = False

        # 1) try AutoImageProcessor
        try:
            self.processor = AutoImageProcessor.from_pretrained(local_dir, local_files_only=True)
        except Exception:
            self.processor = None

        # 2) fallback to AutoProcessor
        if self.processor is None:
            try:
                self.processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
            except Exception:
                self.processor = None

        # 3) If processor is tokenizer or missing, use manual preprocess
        if (self.processor is None) or isinstance(self.processor, PreTrainedTokenizerBase):
            self.use_manual_preprocess = True
            image_size = _infer_image_size(self.model, default=224)

            # CLIP-default normalization (widely used and works well as a fallback)
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)

            self.manual_transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.manual_transform = None

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def _to_device(self, inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    @torch.inference_mode()
    def forward(self, images_pil: List[Image.Image], device: torch.device) -> VisionTokens:
        if self.use_manual_preprocess:
            assert self.manual_transform is not None
            pixel_values = torch.stack(
                [self.manual_transform(img.convert("RGB")) for img in images_pil],
                dim=0
            ).to(device)
        else:
            inputs = self.processor(images=images_pil, return_tensors="pt")
            inputs = self._to_device(inputs, device)
            pixel_values = inputs.get("pixel_values", None)
            if pixel_values is None:
                # unexpected processor output -> fallback to manual
                assert self.manual_transform is not None
                pixel_values = torch.stack(
                    [self.manual_transform(img.convert("RGB")) for img in images_pil],
                    dim=0
                ).to(device)

        # CLIP-like: has vision_model; internal vision_model may NOT accept return_dict
        if hasattr(self.model, "vision_model"):
            out = self.model.vision_model(pixel_values=pixel_values)  # no return_dict
            tokens = getattr(out, "last_hidden_state", None)
            if tokens is None:
                tokens = out[0]
            return VisionTokens(tokens=tokens)

        # ViT/DINOv2-like
        out = self.model(pixel_values=pixel_values)  # no return_dict
        tokens = getattr(out, "last_hidden_state", None)
        if tokens is None:
            tokens = out[0]
        return VisionTokens(tokens=tokens)