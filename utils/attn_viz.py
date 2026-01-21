# utils/attn_viz.py
"""
Attention visualization utilities for multi-encoder setting.

We provide two plots:
1) Encoder-level heatmap: (M queries) x (K encoders)
   - attn_enc is computed by summing attention mass over token ranges for each encoder.
   - This plot supports the paper claim: different encoders contribute differently.

2) Token-level heatmap: (M queries) x (N tokens), with vertical split lines for encoders.
   - This plot supports the paper claim: attention becomes more concentrated/structured.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import os
import numpy as np
import torch


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def plot_encoder_level_heatmap(
    attn_enc: torch.Tensor,                 # (B,H,M,K) or (H,M,K) or (M,K)
    kv_splits: List[Tuple[str, int, int]],
    out_path: str,
    sample_idx: int = 0,
    head_reduce: str = "mean",              # "mean" | "max" | "none"
):
    import matplotlib.pyplot as plt

    # 统一成 (H,M,K)
    if attn_enc.dim() == 4:
        x = attn_enc[sample_idx]  # (H,M,K)
    elif attn_enc.dim() == 3:
        x = attn_enc
    elif attn_enc.dim() == 2:
        x = attn_enc.unsqueeze(0)  # (1,M,K)
    else:
        raise ValueError(f"Unexpected attn_enc shape: {tuple(attn_enc.shape)}")

    if head_reduce == "mean":
        x2 = x.mean(dim=0)  # (M,K)
    elif head_reduce == "max":
        x2 = x.max(dim=0).values
    elif head_reduce == "none":
        # 若不聚合 head，会得到 H 张图；这里简单取 mean 作为默认行为
        x2 = x.mean(dim=0)
    else:
        raise ValueError("head_reduce must be mean|max|none")

    names = [n for (n, _, _) in kv_splits]

    plt.figure()
    plt.imshow(_to_numpy(x2), aspect="auto")
    plt.yticks(range(x2.size(0)))
    plt.xticks(range(len(names)), names, rotation=30, ha="right")
    plt.xlabel("Encoder source")
    plt.ylabel("Query index")
    plt.colorbar()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_token_level_heatmap(
    attn: torch.Tensor,                      # (B,H,M,N)
    kv_splits: List[Tuple[str, int, int]],
    out_path: str,
    sample_idx: int = 0,
    head_reduce: str = "mean",
    max_tokens_plot: int = 512,
):
    """
    Token-level heatmap can be large (N can be hundreds).
    We provide a simple downsampling if N > max_tokens_plot:
      - average pooling tokens into bins so width is bounded for readability.
    """
    import matplotlib.pyplot as plt

    if attn.dim() != 4:
        raise ValueError(f"attn must be (B,H,M,N), got {tuple(attn.shape)}")

    x = attn[sample_idx]  # (H,M,N)

    if head_reduce == "mean":
        x2 = x.mean(dim=0)  # (M,N)
    elif head_reduce == "max":
        x2 = x.max(dim=0).values
    else:
        raise ValueError("head_reduce must be mean|max")

    M, N = x2.shape
    x_plot = x2

    # downsample tokens if needed
    if N > max_tokens_plot:
        bins = max_tokens_plot
        # 将 N 切成 bins 段做均值，得到 (M, bins)
        idx = torch.linspace(0, N, bins + 1).long()
        pooled = []
        for i in range(bins):
            s, e = int(idx[i].item()), int(idx[i + 1].item())
            if e <= s:
                e = min(N, s + 1)
            pooled.append(x_plot[:, s:e].mean(dim=1, keepdim=True))  # (M,1)
        x_plot = torch.cat(pooled, dim=1)  # (M,bins)

        # split 线也要映射到 bins 上
        splits_mapped = []
        for name, s, e in kv_splits:
            s2 = int(round(s / N * bins))
            e2 = int(round(e / N * bins))
            splits_mapped.append((name, s2, e2))
        kv_splits = splits_mapped

    plt.figure(figsize=(10, 4))
    plt.imshow(_to_numpy(x_plot), aspect="auto")
    plt.xlabel("Visual token index (pooled)" if N > max_tokens_plot else "Visual token index")
    plt.ylabel("Query index")
    plt.colorbar()

    # draw encoder split lines
    for name, s, e in kv_splits:
        plt.axvline(x=s, linewidth=1)
        # 在中间位置标注 encoder 名称
        mid = (s + e) / 2
        plt.text(mid, -1.5, name, ha="center", va="top", fontsize=8, rotation=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_keep_mask(
    keep_mask: torch.Tensor,                 # (B,N)
    kv_splits: List[Tuple[str, int, int]],
    out_path: str,
    sample_idx: int = 0,
):
    """
    用于解释自适应去噪：保存每个 token 是否保留（1/0），并按 encoder 分段输出。
    """
    km = keep_mask[sample_idx].detach().cpu().to(torch.int32).numpy().tolist()

    lines = []
    for name, s, e in kv_splits:
        seg = km[s:e]
        kept = sum(seg)
        lines.append(f"{name}: kept {kept}/{len(seg)} ({kept/ max(1,len(seg)):.3f})")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
