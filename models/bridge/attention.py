# models/bridge/attention.py
"""
实现并暴露两种 cross-attn：
1) standard：PyTorch 原生 MultiheadAttention（baseline）
2) param_free：无参数 cross-attn（不引入任何可训练投影矩阵），仅使用激活函数 ϕ

并支持 “自适应去噪”：
- 在 softmax 前基于 scores 生成 keep_mask，将被丢弃 token 的 score 置为 -inf
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dropping import AdaptiveDropCfg, build_adaptive_drop_mask


def _phi(x: torch.Tensor, phi: str) -> torch.Tensor:
    """固定投影函数 ϕ，不包含可训练参数。"""
    phi = (phi or "identity").lower()
    if phi == "identity":
        return x
    if phi == "relu":
        return F.relu(x)
    if phi == "silu":
        return F.silu(x)
    raise ValueError(f"Unsupported phi: {phi}. Choose from identity|relu|silu")


def _maybe_norm(x: torch.Tensor, enabled: bool, eps: float = 1e-6) -> torch.Tensor:
    """可选稳定化：对最后一维做 L2 normalize，抑制 dot-product 动态范围。"""
    if not enabled:
        return x
    return F.normalize(x, dim=-1, eps=eps)


class StandardCrossAttention(nn.Module):
    """
    baseline cross-attention（带参数）：
    内部包含 Q/K/V/O 投影矩阵，用于标准对比实验。
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out, attn = self.mha(query=q, key=kv, value=kv, key_padding_mask=kv_mask, need_weights=need_weights)
        return out, attn


@dataclass
class ParamFreeAttnCfg:
    """无参数 attention 的配置（与你 YAML 对齐）。"""
    phi: str = "identity"
    score_scale: bool = True
    score_norm: bool = False
    attn_dropout: float = 0.0
    adaptive_drop: Optional[AdaptiveDropCfg] = None


class ParamFreeCrossAttention(nn.Module):
    """
    无参数 multi-head cross-attention：
    - 不引入任何 nn.Parameter
    - Q/K/V 均直接来自输入（上游 projectors/queries 仍是可训练的）
    """
    def __init__(self, d_model: int, nhead: int, cfg: ParamFreeAttnCfg):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.cfg = cfg
        self.drop = nn.Dropout(cfg.attn_dropout)  # Dropout 不含参数

    def forward(
        self,
        q: torch.Tensor,                 # (B, M, D)
        kv: torch.Tensor,                # (B, N, D)
        kv_mask: Optional[torch.Tensor] = None,   # (B, N) True 表示 padding
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, M, D = q.shape
        _, N, _ = kv.shape

        # 1) reshape 为多头
        qh = q.view(B, M, self.nhead, self.d_head).transpose(1, 2)   # (B,H,M,d)
        kh = kv.view(B, N, self.nhead, self.d_head).transpose(1, 2)  # (B,H,N,d)
        vh = kv.view(B, N, self.nhead, self.d_head).transpose(1, 2)  # (B,H,N,d)

        # 2) ϕ + 可选归一化
        qh = _maybe_norm(_phi(qh, self.cfg.phi), self.cfg.score_norm)
        kh = _maybe_norm(_phi(kh, self.cfg.phi), self.cfg.score_norm)

        # 3) logits: (B,H,M,N)
        scores = torch.matmul(qh, kh.transpose(-2, -1))
        if self.cfg.score_scale:
            scores = scores / math.sqrt(self.d_head)

        # 4) padding mask
        if kv_mask is not None:
            scores = scores.masked_fill(kv_mask[:, None, None, :].to(torch.bool), float("-inf"))

        # 5) 自适应去噪：生成 keep_mask，并在 softmax 前把 drop token 置 -inf
        if self.cfg.adaptive_drop is not None and self.cfg.adaptive_drop.enabled and self.cfg.adaptive_drop.gamma > 0:
            keep_mask = build_adaptive_drop_mask(
                scores=scores,
                kv_mask=kv_mask,
                gamma=self.cfg.adaptive_drop.gamma,
                mode=self.cfg.adaptive_drop.mode,
                post=self.cfg.adaptive_drop.post,
            )  # (B,N) True=keep
            scores = scores.masked_fill((~keep_mask)[:, None, None, :], float("-inf"))

        # 6) softmax：为稳定性在 fp32 做 softmax，再 cast 回来
        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attn = self.drop(attn)

        # 7) 加权求和
        out = torch.matmul(attn, vh)  # (B,H,M,d)

        # 8) 合并多头
        out = out.transpose(1, 2).contiguous().view(B, M, D)  # (B,M,D)

        if need_weights:
            return out, attn.mean(dim=1)  # (B,M,N)
        return out, None


def build_cross_attention(
    *,
    attn_type: str,
    d_model: int,
    nhead: int,
    dropout: float,
    phi: str = "identity",
    score_scale: bool = True,
    score_norm: bool = False,
    adaptive_drop: Optional[AdaptiveDropCfg] = None,
) -> nn.Module:
    """工厂：根据配置构建 cross-attn。"""
    attn_type = (attn_type or "standard").lower()
    if attn_type == "standard":
        return StandardCrossAttention(d_model, nhead, dropout=dropout)
    if attn_type == "param_free":
        cfg = ParamFreeAttnCfg(
            phi=phi,
            score_scale=score_scale,
            score_norm=score_norm,
            attn_dropout=dropout,
            adaptive_drop=adaptive_drop,
        )
        return ParamFreeCrossAttention(d_model, nhead, cfg)
    raise ValueError(f"Unknown attn_type: {attn_type}. Choose from standard|param_free")
