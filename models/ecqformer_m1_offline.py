from __future__ import annotations
from typing import List, Optional
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
        enabled_encoders: Optional[List[str]] = None,
        torch_dtype=torch.bfloat16,
        attn_type: str = "standard",         # standard | param_free
        phi: str = "identity",               # identity | relu | silu
        score_scale: bool = True,
        score_norm: bool = False,
        adaptive_drop: dict | None = None,  # YAML 里 adaptive_drop 字段原样传进来
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

        # encoder routing (for ablations)
        # Use a fixed canonical order to ensure reproducibility across runs.
        canonical = ["clip", "biomedclip", "dinov2"]
        if enabled_encoders is None:
            enabled_encoders = canonical
        enabled_encoders = [e.lower() for e in enabled_encoders]
        unknown = [e for e in enabled_encoders if e not in canonical]
        if unknown:
            raise ValueError(f"Unknown encoders in enabled_encoders={unknown}. Supported: {canonical}")
        self.enabled_encoders = [e for e in canonical if e in enabled_encoders]

        # Trainable projectors -> d_bridge
        self.proj_clip = LinearProjector(self.enc_clip.out_dim, d_bridge)
        self.proj_dino = LinearProjector(self.enc_dino.out_dim, d_bridge)
        self.proj_bio  = LinearProjector(self.enc_bio.out_dim,  d_bridge)

        # Freeze unused projectors to make Trainable Params comparable across ablations.
        if "clip" not in self.enabled_encoders:
            for p in self.proj_clip.parameters():
                p.requires_grad_(False)
        if "dinov2" not in self.enabled_encoders:
            for p in self.proj_dino.parameters():
                p.requires_grad_(False)
        if "biomedclip" not in self.enabled_encoders:
            for p in self.proj_bio.parameters():
                p.requires_grad_(False)

        # MEQFormer
        self.meq = MEQFormer(d=d_bridge, nhead=meq_heads, num_layers=meq_layers, m_queries=m_queries, dropout=0.0, attn_type=attn_type, 
        phi=phi, score_scale=score_scale, score_norm=score_norm, adaptive_drop=adaptive_drop)

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
        xs = []
        # fixed concat order: clip -> biomedclip -> dinov2
        if "clip" in self.enabled_encoders:
            vt = self.enc_clip(images_pil, device=device).tokens
            xs.append(self.proj_clip(vt))
        if "biomedclip" in self.enabled_encoders:
            vt = self.enc_bio(images_pil, device=device).tokens
            xs.append(self.proj_bio(vt))
        if "dinov2" in self.enabled_encoders:
            vt = self.enc_dino(images_pil, device=device).tokens
            xs.append(self.proj_dino(vt))
        if not xs:
            raise RuntimeError("enabled_encoders is empty: at least one vision encoder must be enabled")
        return torch.cat(xs, dim=1)

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

    def forward_vqa_finetune(
        self,
        *,
        images_pil: List[Image.Image],
        questions: List[str],
        answers: List[str],
        device: torch.device,
        prompt_template: str = "You are a helpful medical assistant. Answer the question based on the image.\nQuestion: {question}\nAnswer:",
        max_length: int = 256,
        debug_nan: bool = True,
    ) -> torch.Tensor:
        """Supervised fine-tuning for Med-VQA.

        We keep the training objective identical to caption pretrain:
        soft_prompt + textual prompt -> next-token LM loss on target answer.
        """
        x_v = self.encode_vision(images_pil, device=device)
        
        if debug_nan and (torch.isnan(x_v).any() or torch.isinf(x_v).any()):
            raise RuntimeError(f"NaN/Inf in vision encoding output. x_v stats: min={x_v.min()}, max={x_v.max()}")
        
        z = self.meq(x_v).z
        
        if debug_nan and (torch.isnan(z).any() or torch.isinf(z).any()):
            raise RuntimeError(f"NaN/Inf after MEQFormer. z stats: min={z.min()}, max={z.max()}")
        
        soft = self.soft_proj(z)
        
        if debug_nan and (torch.isnan(soft).any() or torch.isinf(soft).any()):
            raise RuntimeError(f"NaN/Inf after soft_proj. soft stats: min={soft.min()}, max={soft.max()}")

        prompts = [prompt_template.format(question=q) for q in questions]
        targets = [a for a in answers]
        out = self.lm.forward_with_soft_prompt(
            soft_prompt=soft,
            prompts=prompts,
            targets=targets,
            device=device,
            max_length=max_length,
        )
        return out.loss

    @torch.inference_mode()
    def generate_vqa(
        self,
        *,
        images_pil: List[Image.Image],
        questions: List[str],
        device: torch.device = None,
        prompt_template: str = "You are a helpful medical assistant. Answer the question based on the image.\nQuestion: {question}\nAnswer:",
        max_new_tokens: int = 32,
        num_beams: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
        early_stopping: bool = False,
    ) -> List[str]:
        if device is None:
            device = next(self.parameters()).device

        soft = self.build_soft_prompt(images_pil, device=device)
        prompts = [prompt_template.format(question=q) for q in questions]
        return self.lm.generate_with_soft_prompt(
            soft_prompt=soft,
            prompts=prompts,
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            early_stopping=early_stopping,
        )

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
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
        early_stopping: bool = False,
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
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            early_stopping=early_stopping,
        )

    @torch.inference_mode()
    def generate_caption_with_timing(
        self,
        *,
        images_pil: List[Image.Image],
        device: torch.device = None,
        prompt_prefix: str = "Describe the medical image:\n",
        max_new_tokens: int = 64,
        num_beams: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
        early_stopping: bool = False,
    ) -> tuple:
        """
        生成描述并返回详细计时信息。
        Returns: (captions, timing_dict)
            timing_dict: {
                "vision_bridge_ms": float,  # encode_vision + meq + soft_proj 时间
                "decode_ms": float,         # LM generate 时间
                "num_tokens": int,          # 生成的 token 数
                "tokens_per_s": float,      # 生成速度
            }
        """
        import time
        
        if device is None:
            device = next(self.parameters()).device
        
        # 1) Vision + Bridge timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        
        soft = self.build_soft_prompt(images_pil, device=device)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_vision_bridge = time.time() - t0
        
        # 2) Decode timing
        prompts = [prompt_prefix for _ in images_pil]
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        
        captions, num_tokens = self.lm.generate_with_soft_prompt_and_count(
            soft_prompt=soft,
            prompts=prompts,
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            early_stopping=early_stopping,
        )
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_decode = time.time() - t1
        
        # 计算 tokens per second
        tokens_per_s = num_tokens / t_decode if t_decode > 0 else 0.0
        
        timing = {
            "vision_bridge_ms": t_vision_bridge * 1000,
            "decode_ms": t_decode * 1000,
            "num_tokens": num_tokens,
            "tokens_per_s": tokens_per_s,
        }
        
        return captions, timing

     # --- trainable-only checkpoint helpers ---
    def trainable_state_dict(self) -> dict:
        """
        Only save bridge-side trainables to keep checkpoint small:
        proj_* + meq + soft_proj
        """
        full = self.state_dict()
        keep_prefix = ["meq.", "soft_proj."]
        # Only save enabled projectors (and any other trainable ones) to keep ckpt small.
        if any(p.requires_grad for p in self.proj_clip.parameters()):
            keep_prefix.append("proj_clip.")
        if any(p.requires_grad for p in self.proj_bio.parameters()):
            keep_prefix.append("proj_bio.")
        if any(p.requires_grad for p in self.proj_dino.parameters()):
            keep_prefix.append("proj_dino.")
        keep_prefix = tuple(keep_prefix)
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
