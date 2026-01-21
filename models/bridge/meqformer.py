# models/bridge/meqformer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from .attention import build_cross_attention
from .dropping import AdaptiveDropCfg


@dataclass
class MEQOutput:
    """MEQFormer 输出：z 为重采样后的查询表示 (B, M, D)。"""
    z: torch.Tensor


class MEQBlock(nn.Module):
    """
    单层 MEQ block：

    q <- CrossAttn(q, kv)   # 第三章改造点：standard / param_free 可切换，并可叠加 adaptive_drop
    q <- SelfAttn(q)        # 保持标准 MHA（可训练）
    q <- FFN(q)             # 保持标准 FFN（可训练）
    """
    def __init__(
        self,
        d: int,
        nhead: int,
        dropout: float = 0.0,
        *,
        attn_type: str = "standard",       # standard | param_free
        phi: str = "identity",            # identity | relu | silu
        score_scale: bool = True,
        score_norm: bool = False,
        adaptive_drop: Optional[AdaptiveDropCfg] = None,
    ):
        super().__init__()

        # cross-attn：由工厂按配置构建（解耦实现）
        self.cross_attn = build_cross_attention(
            attn_type=attn_type,
            d_model=d,
            nhead=nhead,
            dropout=dropout,
            phi=phi,
            score_scale=score_scale,
            score_norm=score_norm,
            adaptive_drop=adaptive_drop,
        )

        # self-attn：仍用标准 MultiheadAttention（可训练）
        self.self_attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)

        # FFN：标准 transformer FFN（可训练）
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1) cross-attention
        x = self.norm1(q)
        x2, _ = self.cross_attn(x, kv, kv_mask=kv_mask, need_weights=False)
        q = q + self.drop(x2)

        # 2) self-attention (queries only)
        x = self.norm2(q)
        x2, _ = self.self_attn(query=x, key=x, value=x, need_weights=False)
        q = q + self.drop(x2)

        # 3) FFN
        x = self.norm3(q)
        q = q + self.drop(self.ffn(x))
        return q


class MEQFormer(nn.Module):
    """
    输入：kv（视觉 token）
    输出：固定长度 M 的重采样查询表示 z（后续投到 soft prompt）

    支持 YAML 直接传 dict 构造 adaptive_drop。
    """
    def __init__(
        self,
        d: int,
        nhead: int,
        num_layers: int,
        m_queries: int,
        dropout: float = 0.0,
        *,
        attn_type: str = "standard",
        phi: str = "identity",
        score_scale: bool = True,
        score_norm: bool = False,
        adaptive_drop: Optional[Union[AdaptiveDropCfg, dict]] = None,
    ):
        super().__init__()

        # learnable queries：MEQ 的重采样核心（可训练）
        self.queries = nn.Parameter(torch.randn(m_queries, d) * 0.02)

        # YAML dict -> AdaptiveDropCfg
        if isinstance(adaptive_drop, dict):
            adaptive_drop = AdaptiveDropCfg(
                enabled=bool(adaptive_drop.get("enabled", False)),
                gamma=float(adaptive_drop.get("gamma", 0.0)),
                mode=str(adaptive_drop.get("mode", "topk")),
                post=str(adaptive_drop.get("post", "no_softmax")),
            )

        self.blocks = nn.ModuleList([
            MEQBlock(
                d, nhead, dropout=dropout,
                attn_type=attn_type,
                phi=phi,
                score_scale=score_scale,
                score_norm=score_norm,
                adaptive_drop=adaptive_drop,
            )
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d)

    def forward(self, kv: torch.Tensor, kv_mask: Optional[torch.Tensor] = None) -> MEQOutput:
        B = kv.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for blk in self.blocks:
            q = blk(q, kv, kv_mask=kv_mask)
        q = self.norm_out(q)
        return MEQOutput(z=q)
