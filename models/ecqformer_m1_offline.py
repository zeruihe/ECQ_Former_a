from __future__ import annotations
from typing import List
from PIL import Image

import torch
import torch.nn as nn

from .encoders.hf_vision_encoder_offline import HFVisionEncoderOffline
from .bridge.projectors import LinearProjector
from .bridge.meqformer import MEQFormer
from .lm.frozen_llama_softprompt_offline import FrozenLlamaWithSoftPromptOffline


class ECQFormerM1Offline(nn.Module):
    """
    M1: BRAVE-MEQ Medical Base (Offline)
    Frozen: 3 vision encoders + Llama LM
    Trainable: Pk projectors + MEQFormer + soft_proj
    """
    def __init__(
        self,
        *,
        llama_local_dir: str,
        clip_local_dir: str,
        dinov2_local_dir: str,
        biomedclip_local_dir: str,
        d_bridge: int = 768,
        meq_layers: int = 2,
        meq_heads: int = 12,
        m_queries: int = 96,         # 32*K (K=3 => 96)
        torch_dtype=torch.bfloat16,
    ):
        super().__init__()
        self.torch_dtype = torch_dtype

        # LM (frozen)
        self.lm = FrozenLlamaWithSoftPromptOffline(llama_local_dir, torch_dtype=torch_dtype)

        # Vision encoders (frozen)
        self.enc_clip = HFVisionEncoderOffline(clip_local_dir, torch_dtype=torch_dtype)
        self.enc_dino = HFVisionEncoderOffline(dinov2_local_dir, torch_dtype=torch_dtype)
        self.enc_bio  = HFVisionEncoderOffline(biomedclip_local_dir, torch_dtype=torch_dtype)
        self.enc_clip.freeze()
        self.enc_dino.freeze()
        self.enc_bio.freeze()

        # Trainable projectors -> d_bridge
        self.proj_clip = LinearProjector(self.enc_clip.out_dim, d_bridge)
        self.proj_dino = LinearProjector(self.enc_dino.out_dim, d_bridge)
        self.proj_bio  = LinearProjector(self.enc_bio.out_dim,  d_bridge)

        # MEQFormer
        self.meq = MEQFormer(d=d_bridge, nhead=meq_heads, num_layers=meq_layers, m_queries=m_queries, dropout=0.0)

        # soft prompt projector -> LM embedding dim
        self.soft_proj = nn.Linear(d_bridge, self.lm.embed_dim, bias=True)

        # 将可训练模块转换为正确的 dtype
        self.proj_clip = self.proj_clip.to(torch_dtype)
        self.proj_dino = self.proj_dino.to(torch_dtype)
        self.proj_bio = self.proj_bio.to(torch_dtype)
        self.meq = self.meq.to(torch_dtype)
        self.soft_proj = self.soft_proj.to(torch_dtype)

    def encode_vision(self, images_pil: List[Image.Image], device: torch.device) -> torch.Tensor:
        # encoders are inference_mode; output dtype likely bf16/fp16 depending on torch_dtype
        vt_clip = self.enc_clip(images_pil, device=device).tokens
        vt_dino = self.enc_dino(images_pil, device=device).tokens
        vt_bio  = self.enc_bio(images_pil,  device=device).tokens

        x_clip = self.proj_clip(vt_clip)
        x_dino = self.proj_dino(vt_dino)
        x_bio  = self.proj_bio(vt_bio)

        x_v = torch.cat([x_clip, x_bio, x_dino], dim=1)  # 顺序可固定，便于复现实验
        return x_v

    def forward_caption_pretrain(
        self,
        *,
        images_pil: List[Image.Image],
        captions: List[str],
        device: torch.device,
        prompt_prefix: str = "Describe the medical image:\n",
        max_length: int = 256,
    ) -> torch.Tensor:
        x_v = self.encode_vision(images_pil, device=device)
        z = self.meq(x_v).z
        soft = self.soft_proj(z)

        prompts = [prompt_prefix for _ in captions]
        targets = [c for c in captions]
        out = self.lm.forward_with_soft_prompt(
            soft_prompt=soft,
            prompts=prompts,
            targets=targets,
            device=device,
            max_length=max_length,
        )
        return out.loss

    @torch.inference_mode()
    def build_soft_prompt(self, images_pil: List[Image.Image], device: torch.device) -> torch.Tensor:
        """
        构建 soft prompt: image -> vision tokens -> MEQ resample -> project to LLM dim
        Returns: (B, M, D_lm)
        """
        x_v = self.encode_vision(images_pil, device=device)
        z = self.meq(x_v).z
        soft = self.soft_proj(z)
        return soft

    @torch.inference_mode()
    def generate_caption(
        self,
        *,
        images_pil: List[Image.Image],
        device: torch.device = None,
        prompt_prefix: str = "Describe the medical image:\n",
        max_new_tokens: int = 64,
        num_beams: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        # 自动获取设备
        if device is None:
            device = next(self.parameters()).device
        
        soft = self.build_soft_prompt(images_pil, device=device)
        prompts = [prompt_prefix for _ in images_pil]
        
        return self.lm.generate_with_soft_prompt(
            soft_prompt=soft,
            prompts=prompts,
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
        )

     # --- trainable-only checkpoint helpers ---
    def trainable_state_dict(self) -> dict:
        """
        Only save bridge-side trainables to keep checkpoint small:
        proj_* + meq + soft_proj
        """
        full = self.state_dict()
        keep_prefix = ("proj_clip.", "proj_dino.", "proj_bio.", "meq.", "soft_proj.")
        return {k: v.cpu() for k, v in full.items() if k.startswith(keep_prefix)}

    def load_trainable_state_dict(self, state: dict, strict: bool = False):
        """
        Load only trainable parts. Returns (missing_keys, unexpected_keys) like load_state_dict.
        """
        full = self.state_dict()
        # Merge: overwrite trainable keys only
        for k, v in state.items():
            if k in full:
                full[k] = v
        return self.load_state_dict(full, strict=strict)
