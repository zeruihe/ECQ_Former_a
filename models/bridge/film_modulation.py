# models/bridge/film_modulation.py
"""
FiLM 文本驱动查询调制模块（第二个创新点 Part 2）

Feature-wise Linear Modulation (FiLM) 残差调制：

原理：
  1. Query 增强：将可学习 query Q ∈ (B, M, D) 与文本语义向量 t ∈ (B, D) 进行
     特征维度拼接（concat on dim=-1 并投影回 D），使每个 query 都感知到当前问题语义，
     得到文本感知查询 Q_hat ∈ (B, M, D)。

  2. FiLM 调制：
     从文本语义向量 t 生成通道级参数 (γ, β) ∈ (B, D)。
     以残差形式施加：
         Q_out = Q_hat + γ ⊙ Q_hat + β
                = (1 + γ) ⊙ Q_hat + β
     此处 γ 和 β 对 M 维（query 序列长度）广播，实现逐通道调制。

优势：
  - 不改变 query 序列的形状，MEQFormer cross-attn 不需要任何修改
  - 参数量极小（约 2.4M），计算开销可忽略
  - 残差设计确保若 γ≈0, β≈0 则退化为 concat-proj 路径，训练稳定

形状约定：
  q       : (B, M, D)
  text_vec: (B, D)
  Q_hat   : (B, M, D)
  γ, β    : (B, D) → unsqueeze(1) → (B, 1, D) → broadcast to (B, M, D)
  output  : (B, M, D)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLMQueryModulator(nn.Module):
    """
    文本驱动的 FiLM 查询调制器。

    Args:
        d_model  : query 与文本向量的维度（均为 d_bridge）
        dropout  : concat 投影层前的 dropout
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # ① 特征拼接投影：(2D → D)，将文本信息融入 query
        self.concat_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model, bias=True),
            nn.LayerNorm(d_model),
        )

        # ② FiLM 参数生成器：从文本向量生成通道级 (γ, β)
        # 输出 2D，split 为 γ 和 β
        hidden = d_model * 2
        self.modulator = nn.Sequential(
            nn.Linear(d_model, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model * 2, bias=True),
        )

        # 初始化 modulator 最后一层为零，确保训练初期 γ≈0, β≈0（恒等残差启动）
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)

    def forward(
        self,
        q: torch.Tensor,          # (B, M, D)
        text_vec: torch.Tensor,   # (B, D)
    ) -> torch.Tensor:
        """
        Returns:
            q_out: (B, M, D)，经 FiLM 调制后的查询序列
        """
        B, M, D = q.shape

        # 1) Query 增强：concat + 投影
        t_exp = text_vec.unsqueeze(1).expand(B, M, D)   # (B, M, D)
        q_cat = torch.cat([q, t_exp], dim=-1)            # (B, M, 2D)
        q_hat = self.concat_proj(q_cat)                  # (B, M, D)

        # 2) 生成 FiLM 参数 γ, β
        film_params = self.modulator(text_vec)            # (B, 2D)
        gamma, beta = film_params.split(D, dim=-1)       # each (B, D)
        gamma = gamma.unsqueeze(1)                       # (B, 1, D) → broadcast
        beta  = beta.unsqueeze(1)                        # (B, 1, D) → broadcast

        # 3) 残差 FiLM：Q_out = Q_hat + γ⊙Q_hat + β = (1+γ)⊙Q_hat + β
        q_out = q_hat + gamma * q_hat + beta             # (B, M, D)
        return q_out
