# models/bridge/meqformer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, List, Tuple

import torch
import torch.nn as nn

from .attention import build_cross_attention
from .dropping import AdaptiveDropCfg


@dataclass
class MEQOutput:
    """MEQFormer 输出：z 为重采样后的查询表示 (B, M, D)。"""
    z: torch.Tensor
    debug: Optional[Dict[str, Any]] = None


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

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_mask: Optional[torch.Tensor] = None, *, return_attn: bool = False, return_debug: bool = False,) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 1) cross-attention
        x = self.norm1(q)
        x2, attn, keep_mask = self.cross_attn(
            x,
            kv,
            kv_mask=kv_mask,
            need_weights=return_attn,
            return_debug=return_debug,
        )
        q = q + self.drop(x2)

        # 2) self-attention (queries only)
        x = self.norm2(q)
        x2, _ = self.self_attn(query=x, key=x, value=x, need_weights=False)
        q = q + self.drop(x2)

        # 3) FFN
        x = self.norm3(q)
        q = q + self.drop(self.ffn(x))
        return q, attn, keep_mask


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

    def forward(
        self,
        kv: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
        *,
        return_attn: bool = False,
        return_debug: bool = False,
        kv_splits: Optional[List[Tuple[str, int, int]]] = None,
        attn_layer: int = -1,
    ) -> MEQOutput:
        """
        Args:
          return_attn: 是否返回 cross-attn 权重（用于可视化）
          return_debug: 是否返回 keep_mask 等调试信息（用于解释去噪）
          kv_splits: 多编码器拼接区间 [(name,start,end), ...]
          attn_layer: 取哪一层的 cross-attn 来可视化（默认最后一层 -1）

        debug 返回字段：
          - attn: (B,H,M,N) 某层 cross-attn 权重
          - keep_mask: (B,N) 若开启 adaptive_drop
          - kv_splits: 编码器 token 区间
          - attn_enc: (B,H,M,K) 按编码器聚合后的注意力
        """
        B = kv.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)

        attn_to_save = None
        keep_to_save = None

        # 将 attn_layer 归一化为正索引
        L = len(self.blocks)
        layer_idx = attn_layer if attn_layer >= 0 else (L + attn_layer)

        for i, blk in enumerate(self.blocks):
            want_attn = return_attn and (i == layer_idx)
            want_dbg = return_debug and (i == layer_idx)

            q, attn, keep_mask = blk(
                q, kv, kv_mask=kv_mask,
                return_attn=want_attn,
                return_debug=want_dbg,
            )
            if want_attn:
                attn_to_save = attn
            if want_dbg:
                keep_to_save = keep_mask

        q = self.norm_out(q)

        debug = None
        if return_attn or return_debug:
            debug = {
                "attn": attn_to_save,        # (B,H,M,N) or None
                "keep_mask": keep_to_save,   # (B,N) or None
                "kv_splits": kv_splits,
            }

            # 如果有 splits，则按编码器聚合注意力（更适合论文图）
            if attn_to_save is not None and kv_splits is not None:
                # attn_to_save: (B,H,M,N)
                enc_chunks = []
                for name, s, e in kv_splits:
                    enc_chunks.append(attn_to_save[..., s:e].sum(dim=-1, keepdim=True))  # (B,H,M,1)
                debug["attn_enc"] = torch.cat(enc_chunks, dim=-1)  # (B,H,M,K)

        return MEQOutput(z=q, debug=debug)
