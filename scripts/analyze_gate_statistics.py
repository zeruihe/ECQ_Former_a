# scripts/analyze_gate_statistics.py
# -*- coding: utf-8 -*-
"""
门控可解释性统计：不同问题类型下三路编码器门控权重均值与方差
对应论文第四章可视化实验（2）：门控随问题语义变化的自适应调整

数据来源（二选一）：
  A. 原始 VQA-RAD JSON（推荐，字段完整，含 question_type 9 类）
       --json-path  "/root/autodl-tmp/datasets/data_RAD/VQA_RAD Dataset Public.json"
       --images-dir "/root/autodl-tmp/datasets/data_RAD/VQA_RAD Image Folder"
  B. HuggingFace 格式（备用，question_type 已被 processing.py 删除，仅 Open/Yes-No 两组）

VQA-RAD question_type（原始 JSON 为大写）：
  ABNORMALITY / ATTRIBUTE / COLOR / COUNT / MODALITY /
  OBJECT/ORGAN / PLANE / SIZE / YES/NO

输出（--out-dir）：
  gate_boxplot.png          — 箱线图
  gate_barplot_mean_std.png — 分组条形图（均值 ± std，适合论文直接使用）
  gate_stats.json           — 完整统计数据

用法（单行）：
  python scripts/analyze_gate_statistics.py --config config/m3.yaml --ckpt /root/autodl-tmp/outputs/m3_finetune_vqa-rad/checkpoints/best.pt --json-path "/root/autodl-tmp/datasets/data_RAD/VQA_RAD Dataset Public.json" --images-dir "/root/autodl-tmp/datasets/data_RAD/VQA_RAD Image Folder" --split test --out-dir /root/autodl-tmp/vis_gate_stats

  # 调试图表样式（不加载真实模型，随机权重）
  python scripts/analyze_gate_statistics.py --config config/m3.yaml --json-path "..." --images-dir "..." --split test --debug
"""
from __future__ import annotations

import os, sys, json, argparse, re
from collections import defaultdict
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import yaml
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from models.ecqformer_m2_offline import ECQFormerM2Offline


# ─────────────────────────────────────────────────────────────
# VQA-RAD question_type 归一化
# ─────────────────────────────────────────────────────────────

_QTYPE_DISPLAY = {
    "abnormality":  "Abnormality",
    "attribute":    "Attribute",
    "color":        "Color",
    "count":        "Count",
    "modality":     "Modality",
    "object/organ": "Object/Organ",
    "organ":        "Object/Organ",
    "object":       "Object/Organ",
    "plane":        "Plane",
    "size":         "Size",
    "yes/no":       "Yes/No",
    "yesno":        "Yes/No",
}

_QTYPE_ORDER = [
    "Abnormality", "Attribute", "Color", "Count",
    "Modality", "Object/Organ", "Plane", "Size", "Yes/No",
]

def normalize_qtype(raw) -> str:
    if raw is None:
        return "Unknown"
    return _QTYPE_DISPLAY.get(str(raw).lower().strip(), str(raw).title())


# ─────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────

_TRAIN_PHRASE = {"freeform", "para"}
_TEST_PHRASE  = {"test_freeform", "test_para"}


def load_vqa_rad_json(
    json_path: str,
    images_dir: str,
    split: str,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    从原始 VQA-RAD JSON 加载（推荐）。
    phrase_type in {freeform,para} → train；
    phrase_type in {test_freeform,test_para} → test。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    phrase_types = _TEST_PHRASE if split in ("test", "validation") else _TRAIN_PHRASE

    def clean(x: str) -> str:
        return re.sub(r" +", " ", str(x).lower()).replace(" ?", "?").strip()

    records, skipped = [], 0
    for item in data:
        pt = str(item.get("phrase_type", "")).lower()
        if pt not in phrase_types:
            continue

        image_name = item.get("image_name", "")
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            skipped += 1
            continue

        qt = item.get("question_type", None)
        at = str(item.get("answer_type", "")).upper()

        records.append({
            "image_path":    image_path,
            "question":      clean(item.get("question", "")),
            "answer":        clean(item.get("answer", "")),
            "question_type": normalize_qtype(qt),
            "is_closed":     at in ("CLOSED", "YES/NO", "YESNO"),
        })

        if max_samples and len(records) >= max_samples:
            break

    if skipped:
        print(f"[data] skipped {skipped} items (image not found)")
    print(f"[data] raw JSON / {split}: {len(records)} samples")
    return records


def load_vqa_rad_hf(cfg_data: dict, split: str,
                    max_samples: Optional[int] = None) -> List[Dict]:
    """回退：从 HuggingFace 数据集加载（无 question_type，仅 Open/Yes-No）。"""
    from data.vqa_dataset import HuggingFaceVQADataset
    dataset = HuggingFaceVQADataset(
        dataset_id  = str(cfg_data["dataset_id"]),
        split       = split,
        cache_dir   = cfg_data.get("cache_dir"),
        local_dir   = cfg_data.get("local_dir"),
        images_dir  = cfg_data.get("images_dir"),
        language    = str(cfg_data.get("language", "en")),
        max_samples = max_samples,
    )
    print(f"[data] HF dataset / {split}: {len(dataset)} samples (no question_type → Open/Yes-No only)")
    records = []
    for i in range(len(dataset)):
        item = dataset[i]
        is_c = bool(item["is_closed"])
        records.append({
            "image_pil":     item["image_pil"],
            "question":      item["question"],
            "question_type": "Yes/No" if is_c else "Open",
            "is_closed":     is_c,
        })
    return records


# ─────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────

def _get_enabled_encoders(m_cfg) -> List[str]:
    enc = m_cfg.get("enabled_encoders", None)
    return [str(e) for e in enc] if enc else ["clip", "biomedclip", "dinov2"]

def _auto_m_queries(m_cfg) -> int:
    enabled = _get_enabled_encoders(m_cfg)
    return 32 * len(enabled) if m_cfg.get("auto_m_queries", True) else int(m_cfg["m_queries"])

def load_model(cfg_path: str, ckpt_path: Optional[str], device: torch.device):
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
        print("[load] No checkpoint — random init (debug)")
    model.eval()
    return model, cfg


# ─────────────────────────────────────────────────────────────
# 门控权重收集
# ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def collect_gate_weights(
    model: ECQFormerM2Offline,
    records: List[Dict],
    device: torch.device,
) -> Dict[str, Dict[str, List[float]]]:
    enabled_encoders = model.enabled_encoders
    gate_data: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {enc: [] for enc in enabled_encoders}
    )
    n = len(records)
    for i, rec in enumerate(records):
        if i % 50 == 0:
            print(f"  [{i}/{n}]", end="\r", flush=True)

        image_pil = (rec["image_pil"] if "image_pil" in rec
                     else Image.open(rec["image_path"]).convert("RGB"))
        q     = rec["question"]
        qtype = rec["question_type"]

        weights = model.get_gate_weights(questions=[q], device=device)[0]
        weights = weights.float().cpu().tolist()

        for enc_name, w in zip(enabled_encoders, weights):
            gate_data[qtype][enc_name].append(w)

    print(f"\n[data] done: {n} samples across {len(gate_data)} question types")
    for qt in _sort_qtypes(list(gate_data.keys())):
        n_qt = len(next(iter(gate_data[qt].values())))
        print(f"       {qt:<15s}: {n_qt:4d} samples")
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
# 绘图
# ─────────────────────────────────────────────────────────────

ENC_COLORS = {"clip": "#4C9BE8", "biomedclip": "#E87B4C", "dinov2": "#5CC77A"}
ENC_LABELS = {"clip": "CLIP",    "biomedclip": "BiomedCLIP", "dinov2": "DINOv2"}


def _sort_qtypes(keys: List[str]) -> List[str]:
    ordered = [q for q in _QTYPE_ORDER if q in keys]
    rest    = sorted(k for k in keys if k not in ordered)
    return ordered + rest


def plot_boxplot(gate_data: Dict, enabled_encoders: List[str],
                 save_path: str, dpi: int = 150):
    qtypes = _sort_qtypes(list(gate_data.keys()))
    Q, K   = len(qtypes), len(enabled_encoders)
    fig, ax = plt.subplots(figsize=(max(12, Q * 1.8), 6), dpi=dpi)
    positions_base = np.arange(Q) * (K * 0.22 + 0.5)

    for ki, enc in enumerate(enabled_encoders):
        positions = positions_base + (ki - (K - 1) / 2) * 0.22
        ax.boxplot(
            [gate_data[qt][enc] for qt in qtypes],
            positions=positions, widths=0.18,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="white", linewidth=2),
            boxprops=dict(facecolor=ENC_COLORS[enc], alpha=0.85, linewidth=0.8),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
        )
        ax.plot([], [], color=ENC_COLORS[enc], linewidth=6,
                label=ENC_LABELS[enc], alpha=0.85)

    ax.set_xticks(positions_base)
    ax.set_xticklabels(qtypes, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Gate Weight (channel-mean sigmoid)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Encoder Gate Weight Distribution by Question Type\n(VQA-RAD Test Set)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[vis] boxplot: {save_path}")


def plot_barplot_mean_std(stats: dict, enabled_encoders: List[str],
                          save_path: str, dpi: int = 150):
    qtypes = _sort_qtypes(list(stats.keys()))
    Q, K   = len(qtypes), len(enabled_encoders)
    x, bw  = np.arange(Q), 0.22

    fig, ax = plt.subplots(figsize=(max(11, Q * 1.6), 5.5), dpi=dpi)

    for ki, enc in enumerate(enabled_encoders):
        offset = (ki - (K - 1) / 2) * bw
        means  = [stats[qt][enc]["mean"] for qt in qtypes]
        stds   = [stats[qt][enc]["std"]  for qt in qtypes]
        bars   = ax.bar(x + offset, means, bw, label=ENC_LABELS[enc],
                        color=ENC_COLORS[enc], alpha=0.85, zorder=3)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none",
                    color="gray", elinewidth=1.2, capsize=3, zorder=4)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.012,
                    f"{m:.3f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(qtypes, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Mean Gate Weight (±Std)", fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_title("Mean & Variance of Encoder Gate Weights by Question Type\n"
                 "(VQA-RAD Test Set — higher weight = greater encoder contribution)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[vis] barplot: {save_path}")


def print_stats_table(stats: dict, enabled_encoders: List[str]):
    qtypes = _sort_qtypes(list(stats.keys()))
    print("\n" + "=" * 80)
    print("  Gate Weight Stats (mean ± std)")
    print("-" * 80)
    hdr = f"{'Question Type':<16s}"
    for enc in enabled_encoders:
        hdr += f"  {ENC_LABELS.get(enc, enc):>22s}"
    print(hdr)
    print("-" * 80)
    for qt in qtypes:
        row = f"{qt:<16s}"
        for enc in enabled_encoders:
            s = stats[qt][enc]
            row += f"  {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']:4d})"
        print(row)
    print("=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Gate weight statistics by VQA-RAD question type (M2 model)")
    ap.add_argument("--config",      required=True,
                    help="M2 config yaml, e.g. config/m3.yaml")
    ap.add_argument("--ckpt",        default=None,
                    help="Checkpoint path (omit for debug/random-init)")
    ap.add_argument("--split",       default="test",
                    choices=["train", "validation", "test"])
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Max samples to process (0 = all)")

    # 原始 JSON 模式（推荐）
    ap.add_argument("--json-path",
                    default=None,
                    help='Raw VQA-RAD JSON path, e.g. '
                         '"/root/autodl-tmp/datasets/data_RAD/VQA_RAD Dataset Public.json"')
    ap.add_argument("--images-dir",
                    default=None,
                    help='Raw VQA-RAD image folder, e.g. '
                         '"/root/autodl-tmp/datasets/data_RAD/VQA_RAD Image Folder"')

    ap.add_argument("--out-dir",     default="outputs/vis_gate_stats")
    ap.add_argument("--dpi",         type=int, default=150)
    ap.add_argument("--debug",       action="store_true",
                    help="Use random gate weights (no model/data needed) to test chart style")
    args = ap.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    model, cfg = load_model(args.config, args.ckpt, device)
    enabled_encoders = model.enabled_encoders

    # ── 数据 ──────────────────────────────────────────────────
    if args.debug:
        import random; random.seed(42)
        K = len(enabled_encoders)
        gate_data = {}
        for qt in _QTYPE_ORDER:
            base = [0.45 + 0.12 * i for i in range(K)]
            random.shuffle(base)
            gate_data[qt] = {
                enc: [min(1.0, max(0.0, b + random.gauss(0, 0.09))) for _ in range(40)]
                for enc, b in zip(enabled_encoders, base)
            }
        print("[debug] using random gate weights")
    else:
        max_s = int(args.max_samples) if args.max_samples > 0 else None

        if args.json_path:
            if not args.images_dir:
                ap.error("--images-dir is required when --json-path is set")
            records = load_vqa_rad_json(
                json_path   = args.json_path,
                images_dir  = args.images_dir,
                split       = args.split,
                max_samples = max_s,
            )
        else:
            print("[warn] --json-path not set, falling back to HuggingFace dataset")
            print("       question_type is missing in that dataset; only Open/Yes-No grouping available")
            records = load_vqa_rad_hf(cfg["data"], args.split, max_s)

        gate_data = collect_gate_weights(model, records, device)

    # ── 统计 & 输出 ───────────────────────────────────────────
    stats = compute_stats(gate_data, enabled_encoders)
    print_stats_table(stats, enabled_encoders)

    os.makedirs(args.out_dir, exist_ok=True)

    plot_boxplot(gate_data, enabled_encoders,
                 save_path=os.path.join(args.out_dir, "gate_boxplot.png"),
                 dpi=args.dpi)
    plot_barplot_mean_std(stats, enabled_encoders,
                          save_path=os.path.join(args.out_dir, "gate_barplot_mean_std.png"),
                          dpi=args.dpi)

    stats_path = os.path.join(args.out_dir, "gate_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "encoders": enabled_encoders,
                   "stats": stats}, f, ensure_ascii=False, indent=2)
    print(f"[vis] stats json: {stats_path}")
    print(f"\n[done] All outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
