# models/bridge/dropping.py
"""
自适应去噪（Adaptive Denoising / Adaptive Dropping）模块。

核心目标：
- 在多源视觉 token 冗余/噪声较多时，自动抑制“与当前查询不相关”的 token。
- 该文件只负责“如何根据相似度/注意力分数生成 token-level keep/drop mask”，
  并不绑定具体 attention 实现，便于后续做 dropping 规则对比。

我们这里采用一个工程上稳定、实现简单、易于消融的策略：
1) 基于 cross-attn 的 logits (或其 softmax 后权重) 计算每个视觉 token 的 relevance。
2) 按 gamma 指定的丢弃比例，对 token relevance 做 top-k 保留。

术语与形状约定：
- scores: (B, H, M, N)
    B: batch, H: head, M: query length（桥接查询数）, N: 视觉 token 数
- kv_mask: (B, N)  True 表示 padding token（应当永远 drop）
- 输出 keep_mask: (B, N) True 表示保留

gamma 的含义（与你实验方案一致）：
- gamma = 0   ：不丢弃（等价于无去噪）
- gamma = 0.2 ：丢弃 20% 的视觉 token（仅在非 padding token 内）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AdaptiveDropCfg:
    """Config for adaptive dropping/denoising."""
    enabled: bool = False
    gamma: float = 0.0          # drop ratio in [0,1)
    mode: str = "topk"          # "topk" | "sort"
    post: str = "no_softmax"    # "no_softmax" | "softmax"


def _token_relevance(scores: torch.Tensor, post: str) -> torch.Tensor:
    """
    计算每个 token 的 relevance（重要性分数）。

    - post == "no_softmax": 直接对 logits 做 mean 聚合（更省算力）
    - post == "softmax"   : 先对 logits 做 softmax 得到注意力权重，再 mean 聚合（更直观）

    输出 relevance: (B, N)
    """
    post = (post or "no_softmax").lower()

    if post == "softmax":
        w = torch.softmax(scores.float(), dim=-1)  # (B,H,M,N)
        return w.mean(dim=(1, 2))                  # (B,N)

    if post == "no_softmax":
        return scores.float().mean(dim=(1, 2))     # (B,N)

    raise ValueError(f"Unsupported post: {post}. Choose from no_softmax|softmax")


def build_adaptive_drop_mask(
    *,
    scores: torch.Tensor,                      # (B, H, M, N)
    kv_mask: Optional[torch.Tensor],           # (B, N) True=padding
    gamma: float,
    mode: str = "topk",
    post: str = "no_softmax",
) -> torch.Tensor:
    """
    根据 adaptive dropping 规则生成 keep_mask (B, N)。

    - gamma: 丢弃比例（0~1）。保留 ceil((1-gamma)*有效token数) 个。
    - mode : topk/sort（topk 更快）
    - padding token 永远不会保留
    - 至少保留 1 个 token（避免全 -inf 导致 NaN）
    """
    if gamma <= 0:
        if kv_mask is None:
            return torch.ones(scores.size(0), scores.size(-1), device=scores.device, dtype=torch.bool)
        return ~kv_mask.to(torch.bool)

    B, _, _, N = scores.shape
    rel = _token_relevance(scores, post=post)  # (B,N)

    # padding token relevance 置为 -inf，避免被选中
    if kv_mask is not None:
        rel = rel.masked_fill(kv_mask.to(torch.bool), float("-inf"))

    # 计算有效 token 数（每个样本可能不同）
    if kv_mask is None:
        valid_counts = torch.full((B,), N, device=scores.device, dtype=torch.long)
    else:
        valid_counts = (~kv_mask.to(torch.bool)).sum(dim=1).clamp(min=1)

    keep_mask = torch.zeros((B, N), device=scores.device, dtype=torch.bool)
    mode = (mode or "topk").lower()

    # 逐样本做 top-k：便于严格控制丢弃比例
    for b in range(B):
        n_valid = int(valid_counts[b].item())
        k_keep = int(max(1, round((1.0 - float(gamma)) * n_valid)))

        if mode == "topk":
            idx = torch.topk(rel[b], k=k_keep, largest=True).indices
        elif mode == "sort":
            idx = torch.argsort(rel[b], descending=True)[:k_keep]
        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose from topk|sort")

        keep_mask[b, idx] = True

    # padding token 彻底 drop
    if kv_mask is not None:
        keep_mask = keep_mask & (~kv_mask.to(torch.bool))

    return keep_mask
