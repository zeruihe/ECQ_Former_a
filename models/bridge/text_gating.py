# models/bridge/text_gating.py
"""
文本引导的编码器门控模块（第二个创新点 Part 1）

包含两个组件：
1. TextSemanticEncoder
   - 复用冻结的 LLaMA Embedding 层，对问题 token 做加权均值池化
   - 通过 Linear + LayerNorm 投影到 d_bridge 维度
   - 不引入任何额外大型文本模型，仅增加约 1.2M 可训练参数

2. EncoderTextGating
   - 为每路视觉编码器独立维护一个 MLP 门控生成器
   - 生成通道级 gate 向量 g_k ∈ (B, D)，与 token 做逐元素乘法
   - 防门控塌缩：Encoder Dropout（训练期）
       以概率 encoder_drop_p 触发，随机选择一路编码器将其 gate 置为零向量
       推理期关闭，所有编码器门控完整生效

形状约定：
  text_vec  : (B, D_bridge)
  vis_tokens: (B, N_k, D_bridge)
  gate_k    : (B, D_bridge)  → broadcast to (B, 1, D_bridge)
  output_k  : (B, N_k, D_bridge)  = vis_tokens_k × gate_k (after broadcast)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────
# 1. Text Semantic Encoder
# ─────────────────────────────────────────────────

class TextSemanticEncoder(nn.Module):
    """
    将文本问题编码为语义向量 t ∈ (B, d_bridge)。

    流程：
      tokenize → LLaMA embed_layer（冻结）→ attention_mask 加权均值池化
      → Dropout → Linear(D_lm, d_bridge) → LayerNorm

    参数：
      lm_embed_dim : LLaMA embedding 维度（通常为 4096，由上层传入）
      d_bridge     : 桥接维度（与视觉 token 维度一致）
      max_len      : tokenize 最大长度（仅用于问题，64 已充分）
      dropout      : 投影层前的 dropout
    """

    def __init__(
        self,
        lm_embed_dim: int,
        d_bridge: int,
        max_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(lm_embed_dim, d_bridge, bias=True)
        self.norm = nn.LayerNorm(d_bridge)

    def forward(
        self,
        questions: List[str],
        tokenizer,
        embed_layer: nn.Embedding,   # LLaMA 的 input_embeddings，冻结，外部传入
        device: torch.device,
    ) -> torch.Tensor:
        """
        Args:
            questions  : List[str]，长度 B
            tokenizer  : LLaMA tokenizer
            embed_layer: model.get_input_embeddings()，冻结
            device     : 目标设备
        Returns:
            t : (B, d_bridge)
        """
        tok = tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
        )
        input_ids  = tok["input_ids"].to(device)         # (B, L)
        attn_mask  = tok["attention_mask"].to(device)    # (B, L)  1=real, 0=pad

        # 复用冻结 LLaMA embedding（不计入梯度）
        with torch.no_grad():
            emb = embed_layer(input_ids)                 # (B, L, D_lm)

        # 确保精度与投影层一致
        emb = emb.to(self.proj.weight.dtype)

        # 加权均值池化：仅对非 padding token 有效
        mask_f = attn_mask.unsqueeze(-1).to(emb.dtype)  # (B, L, 1)
        t = (emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)  # (B, D_lm)

        # 投影到 d_bridge
        t = self.proj(self.drop(t))   # (B, d_bridge)
        t = self.norm(t)
        return t


# ─────────────────────────────────────────────────
# 2. Encoder Gating Module
# ─────────────────────────────────────────────────

class _GateMLP(nn.Module):
    """单路编码器的轻量 MLP 门控生成器。"""

    def __init__(self, d_bridge: int, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = d_bridge * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_bridge, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_bridge, bias=True),
        )

    def forward(self, text_vec: torch.Tensor) -> torch.Tensor:
        """text_vec: (B, D) → gate_logits: (B, D)"""
        return self.net(text_vec)


class EncoderTextGating(nn.Module):
    """
    文本引导多路编码器门控。

    对每路编码器维护独立的 MLP 门控生成器，输出通道级 sigmoid 门控：
        g_k = sigmoid(MLP_k(text_vec))  ∈ (B, D)
        vis_k_gated = vis_k × g_k.unsqueeze(1)   逐元素乘法

    防门控塌缩：Encoder Dropout（仅训练期）
        以概率 encoder_drop_p 随机选择一路编码器，将其 gate 乘以零向量（r_k=0）
        推理期所有 r_k=1，门控完整生效

    Args:
        d_bridge        : 桥接维度
        n_encoders      : 编码器路数（默认 3：clip / biomedclip / dinov2）
        mlp_ratio       : 门控 MLP 隐层扩展比例
        dropout         : MLP 内部 dropout
        encoder_drop_p  : Encoder Dropout 触发概率（仅 training=True 时生效）
    """

    def __init__(
        self,
        d_bridge: int,
        n_encoders: int = 3,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        encoder_drop_p: float = 0.1,
    ):
        super().__init__()
        self.n_encoders = n_encoders
        self.encoder_drop_p = encoder_drop_p

        # 每路独立的门控 MLP
        self.gate_mlps = nn.ModuleList(
            [_GateMLP(d_bridge, mlp_ratio, dropout) for _ in range(n_encoders)]
        )

    def forward(
        self,
        text_vec: torch.Tensor,              # (B, D_bridge)
        vis_tokens_list: List[torch.Tensor], # List[(B, N_k, D_bridge)]
    ) -> List[torch.Tensor]:
        """
        Returns:
            gated_list: List[(B, N_k, D_bridge)]，门控后的视觉 token 列表
        """
        assert len(vis_tokens_list) == self.n_encoders, (
            f"Expected {self.n_encoders} encoders, got {len(vis_tokens_list)}"
        )
        B = text_vec.shape[0]

        # ── 生成各路门控向量 ──
        gates = [
            torch.sigmoid(self.gate_mlps[k](text_vec))  # (B, D)
            for k in range(self.n_encoders)
        ]

        # ── Encoder Dropout（仅训练期） ──
        if self.training and self.encoder_drop_p > 0.0:
            if torch.rand(1).item() < self.encoder_drop_p:
                drop_idx = torch.randint(0, self.n_encoders, (1,)).item()
                gates[drop_idx] = torch.zeros_like(gates[drop_idx])

        # ── 门控乘法 ──
        gated_list = [
            vis_tokens_list[k] * gates[k].unsqueeze(1)  # broadcast (B,1,D)
            for k in range(self.n_encoders)
        ]
        return gated_list

    def get_gate_weights(
        self,
        text_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        （可视化辅助）返回各路门控向量的通道均值，用于直观展示编码器贡献。

        Returns:
            weights: (B, n_encoders)  值域 [0,1]，越接近 1 表示该路编码器贡献越大
        """
        with torch.no_grad():
            gates = [
                torch.sigmoid(self.gate_mlps[k](text_vec)).mean(dim=-1)  # (B,)
                for k in range(self.n_encoders)
            ]
        return torch.stack(gates, dim=1)   # (B, n_encoders)
