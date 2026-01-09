# models/bridge/meqformer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

@dataclass
class MEQOutput:
    z: torch.Tensor  # (B, M, D)

class MEQBlock(nn.Module):
    def __init__(self, d: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.self_attn  = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)

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
        x = self.norm1(q)
        x2, _ = self.cross_attn(query=x, key=kv, value=kv, key_padding_mask=kv_mask)
        q = q + self.drop(x2)

        x = self.norm2(q)
        x2, _ = self.self_attn(query=x, key=x, value=x)
        q = q + self.drop(x2)

        x = self.norm3(q)
        q = q + self.drop(self.ffn(x))
        return q

class MEQFormer(nn.Module):
    def __init__(self, d: int, nhead: int, num_layers: int, m_queries: int, dropout: float = 0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(m_queries, d) * 0.02)
        self.blocks = nn.ModuleList([MEQBlock(d, nhead, dropout=dropout) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(d)

    def forward(self, kv: torch.Tensor, kv_mask: Optional[torch.Tensor] = None) -> MEQOutput:
        B = kv.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for blk in self.blocks:
            q = blk(q, kv, kv_mask=kv_mask)
        q = self.norm_out(q)
        return MEQOutput(z=q)
