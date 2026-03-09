# scripts/visualize_gating_weights.py
"""
编码器门控权重可视化脚本（第二个创新点 Part 1 可视化）

功能：
  给定一张医学图像 + 多个不同问题，
  对每个问题输出三路编码器（CLIP / BiomedCLIP / DINOv2）的门控贡献权重，
  并生成可视化图表：
    - 图1：水平条形图，对比不同问题下三路编码器的门控均值
    - 图2：热力图，横轴=问题，纵轴=编码器，颜色=门控权重

使用方法：
  python scripts/visualize_gating_weights.py \\
    --config config/m3.yaml \\
    --ckpt /root/autodl-tmp/outputs/m3_innovation2_vqa/checkpoints/best.pt \\
    --image /path/to/medical_image.jpg \\
    --questions "What is the main finding?" \\
                "Is there any pneumonia?" \\
                "Describe the left lung structure." \\
                "Any abnormality in right lung?" \\
    --output_dir /root/autodl-tmp/vis_gating

  不指定 --ckpt 时使用随机初始化参数（用于调试维度）。
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")   # 服务器无 GUI，使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from models.ecqformer_m2_offline import ECQFormerM2Offline


# ─────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualize encoder gating weights of ECQFormerM2")
    p.add_argument("--config",  type=str, default="config/m3.yaml", help="M3 config yaml")
    p.add_argument("--ckpt",    type=str, default=None,  help="Checkpoint path (trainable_state_dict)")
    p.add_argument("--image",   type=str, required=True, help="Path to one medical image")
    p.add_argument("--questions", nargs="+", required=True,
                   help="List of questions (space-separated, wrap in quotes)")
    p.add_argument("--output_dir", type=str, default="outputs/vis_gating",
                   help="Directory to save visualization figures")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────

def _load_model(cfg_path: str, ckpt_path: str | None, device: torch.device):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    m_cfg = cfg["models"]
    enabled_encoders = [str(e) for e in m_cfg.get("enabled_encoders", ["clip", "biomedclip", "dinov2"])]
    m_queries = 32 * len(enabled_encoders) if m_cfg.get("auto_m_queries", True) else int(m_cfg["m_queries"])

    model = ECQFormerM2Offline(
        llama_local_dir      = m_cfg["llama_dir"],
        clip_local_dir       = m_cfg["clip_dir"],
        dinov2_local_dir     = m_cfg["dinov2_dir"],
        biomedclip_local_dir = m_cfg["biomedclip_dir"],
        d_bridge      = int(m_cfg["d_bridge"]),
        meq_layers    = int(m_cfg["meq_layers"]),
        meq_heads     = int(m_cfg["meq_heads"]),
        m_queries     = int(m_queries),
        attn_type     = str(m_cfg.get("attn_type", "param_free")),
        phi           = str(m_cfg.get("phi", "silu")),
        score_scale   = bool(m_cfg.get("score_scale", True)),
        score_norm    = bool(m_cfg.get("score_norm", False)),
        adaptive_drop = m_cfg.get("adaptive_drop", None),
        gating        = m_cfg.get("gating", None),
        film          = m_cfg.get("film", None),
        enabled_encoders = enabled_encoders,
        torch_dtype   = torch.bfloat16,
    ).to(device)
    model.eval()

    if ckpt_path is not None:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state   = payload.get("model_trainable", payload)
        model.load_trainable_state_dict(state, strict=False)
        print(f"[load] checkpoint: {ckpt_path}")
    else:
        print("[load] No checkpoint provided, using random initialization (for debugging).")

    return model, enabled_encoders


# ─────────────────────────────────────────────────────────────
# 门控权重提取
# ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def extract_gate_weights(
    model: ECQFormerM2Offline,
    image: Image.Image,
    questions: list[str],
    device: torch.device,
) -> np.ndarray:
    """
    对每个问题提取三路编码器的门控权重均值。

    Returns:
        weights: (Q, K)  Q=问题数, K=编码器数
    """
    images_pil = [image] * len(questions)   # 同一张图，多个问题

    weights_list = []
    for q in questions:
        w = model.get_gate_weights(
            questions=[q],
            device=device,
        )   # (1, K)
        weights_list.append(w[0].float().cpu().numpy())

    return np.stack(weights_list, axis=0)   # (Q, K)


# ─────────────────────────────────────────────────────────────
# 可视化：条形图 + 热力图
# ─────────────────────────────────────────────────────────────

def plot_barh(
    weights: np.ndarray,
    questions: list[str],
    encoder_names: list[str],
    save_path: str,
    dpi: int = 150,
):
    """
    水平分组条形图：每个问题一组，每组三根条（对应三路编码器）。
    """
    Q, K = weights.shape
    x     = np.arange(Q)
    width = 0.25
    colors = ["#4C9BE8", "#E87B4C", "#5CC77A"]   # 蓝 / 橙 / 绿

    fig, ax = plt.subplots(figsize=(max(10, Q * 1.8), 5), dpi=dpi)

    for k, (enc, color) in enumerate(zip(encoder_names, colors)):
        offset = (k - 1) * width
        bars = ax.bar(x + offset, weights[:, k], width, label=enc, color=color, alpha=0.85)
        # 在条顶显示数值
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    #  横轴：截短过长的问题文字
    short_qs = [q[:40] + "…" if len(q) > 40 else q for q in questions]
    ax.set_xticks(x)
    ax.set_xticklabels(short_qs, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean Gate Weight", fontsize=11)
    ax.set_title("Encoder Gating Weights per Question (M2)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[vis] bar chart saved: {save_path}")


def plot_heatmap(
    weights: np.ndarray,
    questions: list[str],
    encoder_names: list[str],
    save_path: str,
    dpi: int = 150,
):
    """
    热力图：行=编码器，列=问题，颜色=门控权重。
    """
    Q, K = weights.shape
    data = weights.T   # (K, Q)

    short_qs = [q[:35] + "…" if len(q) > 35 else q for q in questions]

    fig, ax = plt.subplots(figsize=(max(8, Q * 1.5), 3.5), dpi=dpi)
    im = ax.imshow(data, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(Q))
    ax.set_xticklabels(short_qs, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(K))
    ax.set_yticklabels(encoder_names, fontsize=11)
    ax.set_title("Encoder Gate Weight Heatmap (M2)", fontsize=13, fontweight="bold")

    # 格内标注数值
    for ki in range(K):
        for qi in range(Q):
            v = data[ki, qi]
            ax.text(qi, ki, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color="black" if v < 0.65 else "white")

    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Gate Weight")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[vis] heatmap saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # 加载模型
    model, enabled_encoders = _load_model(args.config, args.ckpt, device)

    # 加载图像
    image = Image.open(args.image).convert("RGB")
    print(f"[image] {args.image}  size={image.size}")

    # 提取门控权重
    questions = args.questions
    print(f"[questions] {len(questions)} questions")
    weights = extract_gate_weights(model, image, questions, device)  # (Q, K)

    print("\n── Gate Weights ───────────────────────────────────────────")
    for qi, q in enumerate(questions):
        row = "  ".join(
            f"{enc}={weights[qi, ki]:.4f}"
            for ki, enc in enumerate(enabled_encoders)
        )
        print(f"  Q{qi+1}: \"{q[:50]}\"")
        print(f"        {row}")
    print("────────────────────────────────────────────────────────────\n")

    # 保存可视化
    os.makedirs(args.output_dir, exist_ok=True)
    stem = Path(args.image).stem

    plot_barh(
        weights, questions, enabled_encoders,
        save_path=os.path.join(args.output_dir, f"{stem}_gate_barh.png"),
        dpi=args.dpi,
    )
    plot_heatmap(
        weights, questions, enabled_encoders,
        save_path=os.path.join(args.output_dir, f"{stem}_gate_heatmap.png"),
        dpi=args.dpi,
    )
    print("[done] Gating visualization complete.")


if __name__ == "__main__":
    main()
