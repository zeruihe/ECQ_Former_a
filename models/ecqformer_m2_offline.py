# models/ecqformer_m2_offline.py
"""
ECQFormerM2Offline：第二个创新点完整模型

在 M1（多编码器融合 + 无参数注意力 + 自适应去噪）基础上新增：
  Part 1：文本引导的编码器门控（EncoderTextGating）
    - 用问题语义动态重分配三路视觉编码器的通道贡献
    - Encoder Dropout 防门控塌缩（训练期随机置零一路gate）

  Part 2：FiLM 文本驱动查询调制（FiLMQueryModulator in MEQFormerV2）
    - 可学习 Queries 在进入 cross-attn 前先与问题语义 concat+投影
    - 再施加残差 FiLM 调制，赋予查询任务相关的检索方向

冻结模块（与 M1 相同）：
  - LLaMA-3.1-8B-UltraMedical
  - CLIP / BiomedCLIP / DINOv2 视觉编码器

可训练模块（M1 + 新增）：
  - proj_clip / proj_bio / proj_dino      （同 M1）
  - MEQFormerV2（含 queries + blocks）    （同 M1，但 forward 支持 text_vec）
  - soft_proj                             （同 M1）
  [新增]
  - text_enc   : TextSemanticEncoder
  - gating     : EncoderTextGating
  - film_mod   : FiLMQueryModulator（内嵌于 MEQFormerV2）
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

import torch
import torch.nn as nn

from .encoders.hf_vision_encoder_offline import HFVisionEncoderOffline
from .bridge.projectors import LinearProjector
from .bridge.meqformer import MEQFormerV2
from .bridge.text_gating import TextSemanticEncoder, EncoderTextGating
from .bridge.film_modulation import FiLMQueryModulator
from .bridge.dropping import AdaptiveDropCfg
from .lm.frozen_llama_softprompt_offline import FrozenLlamaWithSoftPromptOffline


class ECQFormerM2Offline(nn.Module):
    """
    M2: 文本驱动的编码器门控 + FiLM 查询调制（Offline）

    YAML 新增参数（models 节）：
      gating:
        enabled        : bool   (default True)
        mlp_ratio      : int    (default 2)
        dropout        : float  (default 0.1)
        encoder_drop_p : float  (default 0.1)
      film:
        enabled  : bool   (default True)
        dropout  : float  (default 0.1)
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
        m_queries: int = 96,
        enabled_encoders: Optional[List[str]] = None,
        torch_dtype=torch.bfloat16,
        # M1 引入的 bridge 参数
        attn_type: str = "param_free",
        phi: str = "silu",
        score_scale: bool = True,
        score_norm: bool = False,
        adaptive_drop: dict | None = None,
        # M2 新增：门控配置
        gating: dict | None = None,
        # M2 新增：FiLM 配置
        film: dict | None = None,
    ):
        super().__init__()
        self.torch_dtype = torch_dtype

        # ─── 冻结的 LLM ─────────────────────────────────────────
        self.lm = FrozenLlamaWithSoftPromptOffline(llama_local_dir, torch_dtype=torch_dtype)

        # ─── 冻结的视觉编码器 ─────────────────────────────────────
        self.enc_clip = HFVisionEncoderOffline(clip_local_dir, torch_dtype=torch_dtype)
        self.enc_dino = HFVisionEncoderOffline(dinov2_local_dir, torch_dtype=torch_dtype)
        self.enc_bio  = HFVisionEncoderOffline(biomedclip_local_dir, torch_dtype=torch_dtype)
        self.enc_clip.freeze()
        self.enc_dino.freeze()
        self.enc_bio.freeze()

        # 编码器路由（消融用）
        canonical = ["clip", "biomedclip", "dinov2"]
        if enabled_encoders is None:
            enabled_encoders = canonical
        enabled_encoders = [e.lower() for e in enabled_encoders]
        unknown = [e for e in enabled_encoders if e not in canonical]
        if unknown:
            raise ValueError(f"Unknown encoders: {unknown}. Supported: {canonical}")
        self.enabled_encoders = [e for e in canonical if e in enabled_encoders]

        # ─── 可训练：视觉投影器 ──────────────────────────────────
        self.proj_clip = LinearProjector(self.enc_clip.out_dim, d_bridge)
        self.proj_dino = LinearProjector(self.enc_dino.out_dim, d_bridge)
        self.proj_bio  = LinearProjector(self.enc_bio.out_dim, d_bridge)

        # 冻结未启用编码器的投影器（消融时保持参数量可比）
        if "clip" not in self.enabled_encoders:
            for p in self.proj_clip.parameters(): p.requires_grad_(False)
        if "dinov2" not in self.enabled_encoders:
            for p in self.proj_dino.parameters(): p.requires_grad_(False)
        if "biomedclip" not in self.enabled_encoders:
            for p in self.proj_bio.parameters(): p.requires_grad_(False)

        # ─── M2 Part 1：文本语义编码器 + 编码器门控 ─────────────
        gating_cfg = gating or {}
        self.gating_enabled = bool(gating_cfg.get("enabled", True))
        if self.gating_enabled:
            lm_embed_dim = self.lm.embed_dim   # LLaMA embedding 维度
            self.text_enc = TextSemanticEncoder(
                lm_embed_dim=lm_embed_dim,
                d_bridge=d_bridge,
                max_len=64,
                dropout=float(gating_cfg.get("dropout", 0.1)),
            )
            self.gating = EncoderTextGating(
                d_bridge=d_bridge,
                n_encoders=len(self.enabled_encoders),
                mlp_ratio=int(gating_cfg.get("mlp_ratio", 2)),
                dropout=float(gating_cfg.get("dropout", 0.1)),
                encoder_drop_p=float(gating_cfg.get("encoder_drop_p", 0.1)),
            )
        else:
            self.text_enc = None
            self.gating = None

        # ─── M2 Part 2：FiLM 调制器 ─────────────────────────────
        film_cfg = film or {}
        self.film_enabled = bool(film_cfg.get("enabled", True))
        if self.film_enabled:
            film_mod = FiLMQueryModulator(
                d_model=d_bridge,
                dropout=float(film_cfg.get("dropout", 0.1)),
            )
        else:
            film_mod = None

        # ─── MEQFormerV2（支持 FiLM） ──────────────────────────
        self.meq = MEQFormerV2(
            d=d_bridge,
            nhead=meq_heads,
            num_layers=meq_layers,
            m_queries=m_queries,
            dropout=0.0,
            attn_type=attn_type,
            phi=phi,
            score_scale=score_scale,
            score_norm=score_norm,
            adaptive_drop=adaptive_drop,
            film_modulator=film_mod,
        )

        # ─── soft prompt 投影器 ─────────────────────────────────
        self.soft_proj = nn.Linear(d_bridge, self.lm.embed_dim, bias=True)

        # ─── dtype 统一 ──────────────────────────────────────────
        trainable_modules = [
            self.proj_clip, self.proj_dino, self.proj_bio,
            self.meq, self.soft_proj,
        ]
        if self.text_enc is not None:
            trainable_modules.append(self.text_enc)
        if self.gating is not None:
            trainable_modules.append(self.gating)
        for m in trainable_modules:
            m.to(torch_dtype)

    # ─────────────────────────────────────────────────────────────────
    # 内部工具：提取文本语义向量
    # ─────────────────────────────────────────────────────────────────

    def _encode_text(self, questions: List[str], device: torch.device) -> Optional[torch.Tensor]:
        """提取问题语义向量 t ∈ (B, d_bridge)。若门控禁用则返回 None。"""
        if self.text_enc is None:
            return None
        embed_layer = self.lm.model.get_input_embeddings()  # 复用 LLaMA embedding（冻结）
        return self.text_enc(
            questions=questions,
            tokenizer=self.lm.tokenizer,
            embed_layer=embed_layer,
            device=device,
        )

    # ─────────────────────────────────────────────────────────────────
    # 视觉编码（带可选门控）
    # ─────────────────────────────────────────────────────────────────

    def encode_vision_gated(
        self,
        images_pil: List[Image.Image],
        device: torch.device,
        text_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        编码图像 → 投影 → [可选] 门控重标定 → 拼接。

        Args:
            text_vec : (B, D_bridge) 文本语义向量，为 None 时跳过门控
        Returns:
            x_v : (B, N, D_bridge)
        """
        # 各路编码结果
        vis_list = []
        if "clip" in self.enabled_encoders:
            vt = self.enc_clip(images_pil, device=device).tokens
            vis_list.append(self.proj_clip(vt))
        if "biomedclip" in self.enabled_encoders:
            vt = self.enc_bio(images_pil, device=device).tokens
            vis_list.append(self.proj_bio(vt))
        if "dinov2" in self.enabled_encoders:
            vt = self.enc_dino(images_pil, device=device).tokens
            vis_list.append(self.proj_dino(vt))
        if not vis_list:
            raise RuntimeError("enabled_encoders is empty: at least one encoder must be enabled")

        # 编码器门控
        if self.gating is not None and text_vec is not None:
            vis_list = self.gating(text_vec, vis_list)

        return torch.cat(vis_list, dim=1)   # (B, N_total, D)

    def encode_vision_gated_with_splits(
        self,
        images_pil: List[Image.Image],
        device: torch.device,
        text_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[str, int, int]]]:
        """带 kv_splits 信息的视觉编码（用于注意力可视化）。"""
        vis_list = []
        names = []
        if "clip" in self.enabled_encoders:
            vt = self.enc_clip(images_pil, device=device).tokens
            vis_list.append(self.proj_clip(vt))
            names.append("clip")
        if "biomedclip" in self.enabled_encoders:
            vt = self.enc_bio(images_pil, device=device).tokens
            vis_list.append(self.proj_bio(vt))
            names.append("biomedclip")
        if "dinov2" in self.enabled_encoders:
            vt = self.enc_dino(images_pil, device=device).tokens
            vis_list.append(self.proj_dino(vt))
            names.append("dinov2")
        if not vis_list:
            raise RuntimeError("enabled_encoders is empty")

        if self.gating is not None and text_vec is not None:
            vis_list = self.gating(text_vec, vis_list)

        kv_splits = []
        offset = 0
        for name, vt in zip(names, vis_list):
            n = vt.size(1)
            kv_splits.append((name, offset, offset + n))
            offset += n

        return torch.cat(vis_list, dim=1), kv_splits

    # ─────────────────────────────────────────────────────────────────
    # 构建 soft prompt（推理共用）
    # ─────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def build_soft_prompt(
        self,
        images_pil: List[Image.Image],
        device: torch.device,
        questions: Optional[List[str]] = None,
        *,
        return_attn: bool = False,
        return_debug: bool = False,
        attn_layer: int = -1,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        image → soft prompt (B, M, D_lm)

        Args:
            questions : 问题文本列表（M2 新增，用于门控和 FiLM 调制）
        """
        # 文本语义向量
        text_vec = None
        if questions is not None and self.text_enc is not None:
            text_vec = self._encode_text(questions, device)

        # 视觉编码（含门控）
        x_v, kv_splits = self.encode_vision_gated_with_splits(
            images_pil, device=device, text_vec=text_vec
        )

        # MEQFormerV2（含 FiLM 调制）
        meq_out = self.meq(
            x_v,
            text_vec=text_vec,
            return_attn=return_attn,
            return_debug=return_debug,
            kv_splits=kv_splits,
            attn_layer=attn_layer,
        )
        z = meq_out.z
        soft = self.soft_proj(z)

        debug = meq_out.debug
        if debug is not None:
            debug["kv_splits"] = kv_splits

        return soft, debug

    # ─────────────────────────────────────────────────────────────────
    # 训练 forward
    # ─────────────────────────────────────────────────────────────────

    def forward_caption_pretrain(
        self,
        *,
        images_pil: List[Image.Image],
        captions: List[str],
        device: torch.device,
        prompt_prefix: str = "Describe the medical image:\n",
        max_length: int = 256,
    ) -> torch.Tensor:
        """
        M2 预训练 forward（图像描述任务）。

        预训练阶段无真实问题文本，以 prompt_prefix 作为统一的文本条件信号，
        驱动门控和 FiLM 学习面向通用医学图像描述的特征分配策略。
        训练目标：给定 [soft_prompt + prompt_prefix]，最大化 caption 的似然。
        """
        B = len(captions)
        # 将 prompt_prefix 作为所有样本的"指令文本"，驱动门控和 FiLM
        instructions = [prompt_prefix] * B
        text_vec = self._encode_text(instructions, device)

        x_v = self.encode_vision_gated(images_pil, device=device, text_vec=text_vec)
        z = self.meq(x_v, text_vec=text_vec).z
        soft = self.soft_proj(z)

        prompts = [prompt_prefix] * B
        out = self.lm.forward_with_soft_prompt(
            soft_prompt=soft,
            prompts=prompts,
            targets=captions,
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
        prompt_template: str = (
            "You are a helpful medical assistant. "
            "Answer the question based on the image.\n"
            "Question: {question}\nAnswer:"
        ),
        max_length: int = 256,
        debug_nan: bool = True,
    ) -> torch.Tensor:
        """M2 VQA 微调 forward（gate 使用 training=True 启用 Encoder Dropout）。"""
        # 文本语义向量（训练时也计算，用于门控和 FiLM）
        text_vec = self._encode_text(questions, device)

        # 视觉编码 + 门控（self.training 控制 Encoder Dropout）
        x_v = self.encode_vision_gated(images_pil, device=device, text_vec=text_vec)

        if debug_nan and (torch.isnan(x_v).any() or torch.isinf(x_v).any()):
            raise RuntimeError(f"NaN/Inf in gated vision encoding")

        # MEQFormerV2（FiLM 调制 + cross-attn）
        z = self.meq(x_v, text_vec=text_vec).z

        if debug_nan and (torch.isnan(z).any() or torch.isinf(z).any()):
            raise RuntimeError(f"NaN/Inf after MEQFormerV2")

        soft = self.soft_proj(z)

        if debug_nan and (torch.isnan(soft).any() or torch.isinf(soft).any()):
            raise RuntimeError(f"NaN/Inf after soft_proj")

        prompts = [prompt_template.format(question=q) for q in questions]
        out = self.lm.forward_with_soft_prompt(
            soft_prompt=soft,
            prompts=prompts,
            targets=answers,
            device=device,
            max_length=max_length,
        )
        return out.loss

    # ─────────────────────────────────────────────────────────────────
    # 推理接口
    # ─────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate_vqa(
        self,
        *,
        images_pil: List[Image.Image],
        questions: List[str],
        device: torch.device = None,
        prompt_template: str = (
            "You are a helpful medical assistant. "
            "Answer the question based on the image.\n"
            "Question: {question}\nAnswer:"
        ),
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

        soft, _ = self.build_soft_prompt(images_pil, device=device, questions=questions)
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

    # ─────────────────────────────────────────────────────────────────
    # 门控权重获取（可视化用）
    # ─────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def get_gate_weights(
        self,
        questions: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """
        返回各路编码器的门控权重均值，用于可视化分析。

        Returns:
            weights: (B, n_encoders)  值域约 [0,1]
        """
        if self.text_enc is None or self.gating is None:
            raise RuntimeError("Gating module is disabled (gating.enabled=false).")
        text_vec = self._encode_text(questions, device)
        return self.gating.get_gate_weights(text_vec)

    # ─────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ─────────────────────────────────────────────────────────────────

    def trainable_state_dict(self) -> dict:
        """仅保存可训练参数（用于小体积 checkpoint）。"""
        full = self.state_dict()
        keep_prefix = ["meq.", "soft_proj."]
        if any(p.requires_grad for p in self.proj_clip.parameters()):
            keep_prefix.append("proj_clip.")
        if any(p.requires_grad for p in self.proj_bio.parameters()):
            keep_prefix.append("proj_bio.")
        if any(p.requires_grad for p in self.proj_dino.parameters()):
            keep_prefix.append("proj_dino.")
        if self.text_enc is not None:
            keep_prefix.append("text_enc.")
        if self.gating is not None:
            keep_prefix.append("gating.")
        keep_prefix = tuple(keep_prefix)
        return {k: v.cpu() for k, v in full.items() if k.startswith(keep_prefix)}

    def load_trainable_state_dict(self, state: dict, strict: bool = False):
        full = self.state_dict()
        for k, v in state.items():
            if k in full:
                full[k] = v
        return self.load_state_dict(full, strict=strict)
