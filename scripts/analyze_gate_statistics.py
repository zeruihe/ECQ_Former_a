# scripts/analyze_gate_statistics.py
# -*- coding: utf-8 -*-
"""
门控可解释性统计：不同问题类型下三路编码器门控权重均值与方差
对应论文第四章可视化实验（2）：门控随问题语义变化的自适应调整

数据集：VQA-RAD (flaviagiammarino/vqa-rad)
VQA-RAD 原始 question_type 字段取值：
  abnormality / attribute / color / count / modality /
  object/organ / plane / size / yes/no

输出（保存至 --out-dir）：
  gate_boxplot.png          — 箱线图：各问题类型下三路编码器门控权重分布
  gate_barplot_mean_std.png — 分组条形图：均值 ± 方差（每类题型一组，每组三根条）
  gate_stats.json           — 完整统计数据（均值/方差/中位数/四分位距/样本数）

用法：
  python scripts/analyze_gate_statistics.py \\
    --config config/m3.yaml \\
    --ckpt /root/autodl-tmp/outputs/m3_finetune_vqa-rad/checkpoints/best.pt \\
    --split test \\
    --out-dir /root/autodl-tmp/vis_gate_stats \\
    --max-samples 0          # 0 表示全量

    # 调试（不需要真实 checkpoint）
  python scripts/analyze_gate_statistics.py \\
    --config config/m3.yaml \\
    --split test --max-samples 20 --debug
"""
from __future__ import annotations

import os, sys, json, argparse
from collections import defaultdict
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import yaml
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from data.vqa_dataset import HuggingFaceVQADataset
from models.ecqformer_m2_offline import ECQFormerM2Offline


# ─────────────────────────────────────────────────────────────
# VQA-RAD question_type 归一化
# ─────────────────────────────────────────────────────────────

# VQA-RAD 官方类型 → 论文展示标签
_QTYPE_DISPLAY = {
    "abnormality":   "Abnormality",
    "attribute":     "Attribute",
    "color":         "Color",
    "count":         "Count",
    "modality":      "Modality",
    "object/organ":  "Object/Organ",
    "organ":         "Object/Organ",
    "object":        "Object/Organ",
    "plane":         "Plane",
    "size":          "Size",
    "yes/no":        "Yes/No",
    "yesno":         "Yes/No",
    "closed":        "Yes/No",
}

# 论文期望保持的展示顺序
_QTYPE_ORDER = [
    "Abnormality", "Attribute", "Color", "Count",
    "Modality", "Object/Organ", "Plane", "Size", "Yes/No",
]

def normalize_qtype(raw: str) -> str:
    return _QTYPE_DISPLAY.get(str(raw).lower().strip(), str(raw).capitalize())


# ─────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────

def _get_enabled_encoders(m_cfg) -> List[str]:
    enc = m_cfg.get("enabled_encoders", None)
    return [str(e) for e in enc] if enc else ["clip", "biomedclip", "dinov2"]

def _auto_m_queries(m_cfg) -> int:
    enabled = _get_enabled_encoders(m_cfg)
    return 32 * len(enabled) if m_cfg.get("auto_m_queries", True) else int(m_cfg["m_queries"])

def load_model(cfg_path: str, ckpt_path: str | None, device: torch.device):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m_cfg = cfg["models"]
    amp   = torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16
    
    model = ECQFormerM2Offline(
        llama_local_dir      = m_cfg["llama_dir"],
        clip_local_dir       = m_cfg["clip_dir"],
        dinov2_local_dir     = m_cfg["dinov2_dir"],
        biomedclip_local_dir = m_cfg["biomedclip_dir"],
        d_bridge             = int(m_cfg["d_bridge"]),
        meq_layers           = int(m_cfg["meq_layers"]),
        meq_heads            = int(m_cfg["meq_heads"]),
        m_queries            = _auto_m_queries(m_cfg),
        attn_type            = m_cfg.get("attn_type", "param_free"),
        phi                  = m_cfg.get("phi", "silu"),
        score_scale          = bool(m_cfg.get("score_scale", True)),
        score_norm           = bool(m_cfg.get("score_norm", False)),
        adaptive_drop        = m_cfg.get("adaptive_drop", None),
        gating               = m_cfg.get("gating", None),
        film                 = m_cfg.get("film", None),
        enabled_encoders     = _get_enabled_encoders(m_cfg),
        torch_dtype          = amp,
    ).to(device)

    if ckpt_path:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state   = payload.get("model_trainable", payload)
        model.load_state_dict(state, strict=False)
        print(f"[load] checkpoint: {ckpt_path}")
    else:
        print("[load] No checkpoint — using random init (debug)")

    model.eval()
    return model, cfg


# ─────────────────────────────────────────────────────────────
# 数据收集：遍历数据集，按问题类型收集门控权重
# ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def collect_gate_weights(
    model: ECQFormerM2Offline,
    cfg: dict,
    split: str,
    device: torch.device,
    max_samples: int = 0,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Returns:
        gate_data[question_type][encoder_name] = [w1, w2, ...]
        门控权重为通道均值 sigmoid 值，约 [0,1]
    """
    d_cfg = cfg["data"]
    dataset = HuggingFaceVQADataset(
        dataset_id  = str(d_cfg["dataset_id"]),
        split       = split,
        cache_dir   = d_cfg.get("cache_dir", None),
        local_dir   = d_cfg.get("local_dir", None),
        images_dir  = d_cfg.get("images_dir", None),
        language    = str(d_cfg.get("language", "en")),
        max_samples = int(max_samples) if max_samples > 0 else None,
    )
    print(f"[data] {d_cfg['dataset_id']} / {split}  n={len(dataset)}")

    enabled_encoders = model.enabled_encoders

    # gate_data[q_type][enc] = list of scalar weights
    gate_data: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {enc: [] for enc in enabled_encoders}
    )

    n = len(dataset)
    for i in range(n):
        if i % 50 == 0:
            print(f"  [{i}/{n}]", end="\r", flush=True)

        item  = dataset[i]
        raw   = dataset.ds[i]   # 直接访问原始 HF 数据，获取 question_type
        qtype = normalize_qtype(raw.get("question_type", raw.get("answer_type", "unknown")))
        q     = item["question"]
        image = item["image_pil"]

        # 提取门控权重 (1, K)
        weights = model.get_gate_weights(questions=[q], device=device)[0]  # (K,)
        weights = weights.float().cpu().tolist()

        for enc_name, w in zip(enabled_encoders, weights):
            gate_data[qtype][enc_name].append(w)

    print(f"\n[data] collected {n} samples across {len(gate_data)} question types")
    return dict(gate_data)


# ─────────────────────────────────────────────────────────────
# 统计计算
# ─────────────────────────────────────────────────────────────

def compute_stats(gate_data: Dict, enabled_encoders: List[str]) -> dict:
    stats = {}
    for qtype, enc_data in gate_data.items():
        stats[qtype] = {}
        for enc in enabled_encoders:
            vals = np.array(enc_data[enc], dtype=np.float32)
            stats[qtype][enc] = {
                "mean":   float(vals.mean()),
                "std":    float(vals.std()),
                "median": float(np.median(vals)),
                "q25":    float(np.percentile(vals, 25)),
                "q75":    float(np.percentile(vals, 75)),
                "n":      int(len(vals)),
            }
    return stats


# ─────────────────────────────────────────────────────────────
# 绘图：箱线图
# ─────────────────────────────────────────────────────────────

ENC_COLORS  = {"clip": "#4C9BE8", "biomedclip": "#E87B4C", "dinov2": "#5CC77A"}
ENC_LABELS  = {"clip": "CLIP",    "biomedclip": "BiomedCLIP", "dinov2": "DINOv2"}


def _sort_qtypes(keys: List[str]) -> List[str]:
    ordered = [q for q in _QTYPE_ORDER if q in keys]
    rest    = sorted(k for k in keys if k not in ordered)
    return ordered + rest


def plot_boxplot(
    gate_data: Dict,
    enabled_encoders: List[str],
    save_path: str, dpi: int = 150,
):
    """
    箱线图：每个问题类型一组（x轴），每组内三根箱（对应三路编码器）。
    直观展示门控分布随题型的变化。
    """
    qtypes = _sort_qtypes(list(gate_data.keys()))
    Q = len(qtypes)
    K = len(enabled_encoders)

    fig, ax = plt.subplots(figsize=(max(12, Q * 1.8), 6), dpi=dpi)

    # boxplot 每组 K 个，间隔 1.5
    group_width = K * 0.22 + 0.1
    positions_base = np.arange(Q) * (group_width + 0.5)

    for ki, enc in enumerate(enabled_encoders):
        positions = positions_base + (ki - (K - 1) / 2) * 0.22
        data = [gate_data[qt][enc] for qt in qtypes]
        bp = ax.boxplot(
            data, positions=positions, widths=0.18,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="white", linewidth=2),
            boxprops=dict(facecolor=ENC_COLORS[enc], alpha=0.85, linewidth=0.8),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
        )
        # 图例代理
        ax.plot([], [], color=ENC_COLORS[enc], linewidth=6,
                label=ENC_LABELS[enc], alpha=0.85)

    ax.set_xticks(positions_base)
    ax.set_xticklabels(qtypes, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("门控权重（通道均值 sigmoid）", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("不同问题类型下三路编码器门控权重分布\n（VQA-RAD 测试集）",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[vis] boxplot: {save_path}")


def plot_barplot_mean_std(
    stats: dict,
    enabled_encoders: List[str],
    save_path: str, dpi: int = 150,
):
    """
    分组条形图（均值 ± std）：每个题型一组，每组三根条。
    适合论文直接引用的样式。
    """
    qtypes = _sort_qtypes(list(stats.keys()))
    Q  = len(qtypes)
    K  = len(enabled_encoders)
    x  = np.arange(Q)
    bw = 0.22   # 单条宽度

    fig, ax = plt.subplots(figsize=(max(11, Q * 1.6), 5.5), dpi=dpi)

    for ki, enc in enumerate(enabled_encoders):
        offset = (ki - (K - 1) / 2) * bw
        means  = [stats[qt][enc]["mean"] for qt in qtypes]
        stds   = [stats[qt][enc]["std"]  for qt in qtypes]
        bars   = ax.bar(x + offset, means, bw, label=ENC_LABELS[enc],
                        color=ENC_COLORS[enc], alpha=0.85, zorder=3)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none",
                    color="gray", elinewidth=1.2, capsize=3, zorder=4)

        # 在条顶标注均值
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(qtypes, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("门控权重均值（±标准差）", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("不同问题类型下三路编码器门控均值与方差\n（VQA-RAD 测试集，门控权重越高说明该编码器贡献越大）",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[vis] barplot: {save_path}")


# ─────────────────────────────────────────────────────────────
# 控制台汇总打印
# ─────────────────────────────────────────────────────────────

def print_stats_table(stats: dict, enabled_encoders: List[str]):
    qtypes = _sort_qtypes(list(stats.keys()))
    enc_w  = max(len(ENC_LABELS.get(e, e)) for e in enabled_encoders)
    print("\n" + "=" * 80)
    print("  门控权重统计（均值 ± 标准差）")
    print("-" * 80)
    hdr = f"{'Question Type':<20s}"
    for enc in enabled_encoders:
        hdr += f"  {ENC_LABELS.get(enc, enc):>{enc_w+8}s}"
    print(hdr)
    print("-" * 80)
    for qt in qtypes:
        row = f"{qt:<20s}"
        for enc in enabled_encoders:
            s   = stats[qt][enc]
            row += f"  {s['mean']:.4f}±{s['std']:.4f} (n={s['n']:4d})"
        print(row)
    print("=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      required=True, help="M2 config yaml (config/m3.yaml)")
    ap.add_argument("--ckpt",        default=None,  help="Checkpoint 路径（可省略，debug 用）")
    ap.add_argument("--split",       default="test", choices=["train", "validation", "test"])
    ap.add_argument("--max-samples", type=int, default=0, help="0=全量")
    ap.add_argument("--out-dir",     default="outputs/vis_gate_stats")
    ap.add_argument("--dpi",         type=int, default=150)
    ap.add_argument("--debug",       action="store_true",
                    help="调试模式：不加载真实模型，用随机门控权重填充")
    args = ap.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    model, cfg = load_model(args.config, args.ckpt, device)
    enabled_encoders = model.enabled_encoders

    if args.debug:
        # 生成随机统计数据用于调试图表样式
        print("[debug] generating random gate weights...")
        import random
        random.seed(42)
        gate_data = {}
        for qt in _QTYPE_ORDER:
            gate_data[qt] = {}
            base = [0.55 + 0.1 * i for i in range(len(enabled_encoders))]
            random.shuffle(base)
            for enc, b in zip(enabled_encoders, base):
                gate_data[qt][enc] = [
                    min(1.0, max(0.0, b + random.gauss(0, 0.08)))
                    for _ in range(30)
                ]
    else:
        gate_data = collect_gate_weights(
            model, cfg, args.split, device, max_samples=args.max_samples
        )

    stats = compute_stats(gate_data, enabled_encoders)
    print_stats_table(stats, enabled_encoders)

    os.makedirs(args.out_dir, exist_ok=True)

    # 图1：箱线图
    plot_boxplot(
        gate_data, enabled_encoders,
        save_path=os.path.join(args.out_dir, "gate_boxplot.png"),
        dpi=args.dpi,
    )

    # 图2：分组条形图（均值 ± std）
    plot_barplot_mean_std(
        stats, enabled_encoders,
        save_path=os.path.join(args.out_dir, "gate_barplot_mean_std.png"),
        dpi=args.dpi,
    )

    # 保存 JSON 统计数据
    stats_path = os.path.join(args.out_dir, "gate_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "split":    args.split,
            "encoders": enabled_encoders,
            "stats":    stats,
        }, f, ensure_ascii=False, indent=2)
    print(f"[vis] stats json: {stats_path}")
    print(f"\n[done] All outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
