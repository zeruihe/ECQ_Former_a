# scripts/visualize_attn_lesion_focus.py
# -*- coding: utf-8 -*-
"""
注意力热图对比：单路 CLIP 基线 vs M2 完整模型（三路编码器 + FiLM + 门控）
对应论文第四章可视化实验（1）：证明完整模型对病灶区域的注意力聚焦更优

设计逻辑：
  - 从原始 VQA-RAD JSON 随机抽取 N 张图像及其对应问题
  - 分别运行单-CLIP 基线与 M2 完整模型，提取桥接交叉注意力后的空间热图
  - 输出论文级对比布局：每行一个样本（图像|问题|CLIP热图|M2热图）

输出（--out-dir）：
  lesion_focus_compare.png   — 主对比图（N行 × 4列）
  lesion_focus_grid.png      — 精简版（仅2列：CLIP叠加 vs M2叠加）

单行命令（用实际路径替换占位符）：
  python scripts/visualize_attn_lesion_focus.py \\
    --config_clip config/m2_cd.yaml \\
    --config_m2   config/m3.yaml \\
    --ckpt_clip   /path/to/clip_only_ckpt.pt \\
    --ckpt_m2     /path/to/m2_full_ckpt.pt \\
    --json-path   "/root/autodl-tmp/datasets/data_RAD/VQA_RAD Dataset Public.json" \\
    --images-dir  "/root/autodl-tmp/datasets/data_RAD/VQA_RAD Image Folder" \\
    --n-samples   6 --split test \\
    --out-dir     /root/autodl-tmp/vis_lesion_focus
"""
from __future__ import annotations

import os, sys, json, re, argparse, math, random
from typing import Dict, List, Optional, Tuple

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
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


# ─────────────────────────────────────────────────────────────
# 自定义热图颜色（医学图像友好：蓝→青→黄→红，高亮病灶）
# ─────────────────────────────────────────────────────────────
_CMAP = LinearSegmentedColormap.from_list(
    "lesion", ["#000080", "#0080FF", "#00FFFF", "#FFFF00", "#FF4000"], N=256
)


# ─────────────────────────────────────────────────────────────
# 数据加载（直接读原始 VQA-RAD JSON）
# ─────────────────────────────────────────────────────────────

_TRAIN_PHRASE = {"freeform", "para"}
_TEST_PHRASE  = {"test_freeform", "test_para"}


def sample_vqa_rad(
    json_path: str,
    images_dir: str,
    split: str,
    n_samples: int,
    seed: int = 42,
) -> List[Dict]:
    """从原始 VQA-RAD JSON 随机抽取 n_samples 条记录。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    phrase_types = _TEST_PHRASE if split in ("test", "validation") else _TRAIN_PHRASE

    def clean(x): return re.sub(r" +", " ", str(x).lower()).replace(" ?", "?").strip()

    candidates = []
    for item in data:
        pt = str(item.get("phrase_type", "")).lower()
        if pt not in phrase_types:
            continue
        img_path = os.path.join(images_dir, item.get("image_name", ""))
        if not os.path.exists(img_path):
            continue
        candidates.append({
            "image_path":    img_path,
            "image_name":    item.get("image_name", ""),
            "question":      clean(item.get("question", "")),
            "answer":        clean(item.get("answer", "")),
            "question_type": str(item.get("question_type", "")).title(),
        })

    random.seed(seed)
    n = min(n_samples, len(candidates))
    samples = random.sample(candidates, n)
    print(f"[data] sampled {n} / {len(candidates)} {split} records")
    return samples


# ─────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────

def _get_enabled_encoders(m_cfg) -> List[str]:
    enc = m_cfg.get("enabled_encoders", None)
    return [str(e) for e in enc] if enc else ["clip", "biomedclip", "dinov2"]

def _auto_m_queries(m_cfg) -> int:
    n = len(_get_enabled_encoders(m_cfg))
    return 32 * n if m_cfg.get("auto_m_queries", True) else int(m_cfg["m_queries"])

def _load_ckpt(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state   = payload.get("model_trainable", payload)
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"  [ckpt] missing={len(miss)} unexpected={len(unexp)}")

def load_clip_model(cfg_path: str, ckpt_path: Optional[str], device: torch.device):
    """加载单-CLIP 基线（ECQFormerM1Offline，enabled_encoders=['clip']）。"""
    from models.ecqformer_m1_offline import ECQFormerM1Offline
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m = cfg["models"]
    amp = torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16
    model = ECQFormerM1Offline(
        llama_local_dir      = m["llama_dir"],
        clip_local_dir       = m["clip_dir"],
        dinov2_local_dir     = m["dinov2_dir"],
        biomedclip_local_dir = m["biomedclip_dir"],
        d_bridge             = int(m["d_bridge"]),
        meq_layers           = int(m["meq_layers"]),
        meq_heads            = int(m["meq_heads"]),
        m_queries            = _auto_m_queries(m),
        attn_type            = m.get("attn_type", "param_free"),
        phi                  = m.get("phi", "silu"),
        score_scale          = bool(m.get("score_scale", True)),
        score_norm           = bool(m.get("score_norm", False)),
        adaptive_drop        = m.get("adaptive_drop", None),
        enabled_encoders     = ["clip"],   # 强制单路
        torch_dtype          = amp,
    ).to(device)
    if ckpt_path:
        _load_ckpt(model, ckpt_path)
    else:
        print("  [ckpt] no checkpoint (random init)")
    model.eval()
    return model

def load_m2_model(cfg_path: str, ckpt_path: Optional[str], device: torch.device):
    """加载 M2 完整模型（ECQFormerM2Offline）。"""
    from models.ecqformer_m2_offline import ECQFormerM2Offline
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m = cfg["models"]
    amp = torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16
    model = ECQFormerM2Offline(
        llama_local_dir      = m["llama_dir"],
        clip_local_dir       = m["clip_dir"],
        dinov2_local_dir     = m["dinov2_dir"],
        biomedclip_local_dir = m["biomedclip_dir"],
        d_bridge             = int(m["d_bridge"]),
        meq_layers           = int(m["meq_layers"]),
        meq_heads            = int(m["meq_heads"]),
        m_queries            = _auto_m_queries(m),
        attn_type            = m.get("attn_type", "param_free"),
        phi                  = m.get("phi", "silu"),
        score_scale          = bool(m.get("score_scale", True)),
        score_norm           = bool(m.get("score_norm", False)),
        adaptive_drop        = m.get("adaptive_drop", None),
        gating               = m.get("gating", None),
        film                 = m.get("film", None),
        enabled_encoders     = _get_enabled_encoders(m),
        torch_dtype          = amp,
    ).to(device)
    if ckpt_path:
        _load_ckpt(model, ckpt_path)
    else:
        print("  [ckpt] no checkpoint (random init)")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# 注意力提取 & 空间热图生成
# ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def get_attn_debug_m1(model, image_pil, device):
    """M1 / 单-CLIP：无 questions 参数。"""
    _, debug = model.build_soft_prompt(
        [image_pil], device=device,
        return_attn=True, return_debug=True, attn_layer=-1,
    )
    return debug

@torch.inference_mode()
def get_attn_debug_m2(model, image_pil, question, device):
    """M2：需要 questions 参数（触发 FiLM + 门控）。"""
    _, debug = model.build_soft_prompt(
        [image_pil], device=device, questions=[question],
        return_attn=True, return_debug=True, attn_layer=-1,
    )
    return debug


def debug_to_cpu(debug: Optional[dict]) -> Optional[dict]:
    """将 debug 字典中所有 Tensor 移到 CPU，释放 GPU 显存。"""
    if debug is None:
        return None
    out = {}
    for k, v in debug.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.cpu()
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.cpu() for t in v]
        else:
            out[k] = v
    return out


def attn_to_spatial_map(
    attn: torch.Tensor,      # (1, H, M, N_total)
    kv_splits: List,         # [(name, start, end), ...]
    image_pil: Image.Image,
    enc_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    将所有（或指定）编码器的注意力汇总到图像空间热图。
    - 对所有指定编码器的 token 区间取并集
    - 对 head 维和 query 维做平均
    - reshape 为近似正方形网格后上采样到图像尺寸
    Returns: (H_img, W_img) float32 ndarray，值域 [0,1]
    """
    if kv_splits is None:
        return np.zeros((image_pil.size[1], image_pil.size[0]), dtype=np.float32)

    # 收集需要聚合的 token 索引
    if enc_names is None:
        enc_names = [name for name, s, e in kv_splits]

    token_weights = []
    for name, s, e in kv_splits:
        if name in enc_names:
            seg = attn[0, :, :, s:e]           # (H, M, N_enc)
            seg = seg.mean(dim=0).mean(dim=0)   # (N_enc,)
            token_weights.append(seg.float().cpu())

    if not token_weights:
        return np.zeros((image_pil.size[1], image_pil.size[0]), dtype=np.float32)

    weights = torch.cat(token_weights, dim=0).numpy()   # (N_total_selected,)
    N = len(weights)
    grid = int(round(math.sqrt(N)))
    w_grid = math.ceil(N / grid)
    pad = grid * w_grid - N
    if pad > 0:
        weights = np.concatenate([weights, np.zeros(pad)])
    heat = weights.reshape(grid, w_grid)

    # 归一化
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    # 上采样到图像尺寸
    W_img, H_img = image_pil.size
    heat_pil = Image.fromarray((heat * 255).astype(np.uint8)).resize(
        (W_img, H_img), Image.BILINEAR
    )
    return np.array(heat_pil).astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────
# 绘图：论文级对比布局
# ─────────────────────────────────────────────────────────────

def wrap_text(text: str, max_len: int = 50) -> str:
    """简单断行，避免标题过长溢出。"""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_len:
            lines.append(cur.strip())
            cur = w
        else:
            cur += " " + w
    if cur:
        lines.append(cur.strip())
    return "\n".join(lines)


def plot_comparison(
    samples: List[Dict],
    attn_clip: List[Optional[dict]],
    attn_m2:   List[Optional[dict]],
    save_path: str,
    alpha: float = 0.55,
    dpi: int = 150,
):
    """
    主对比图：每行一个样本，四列：原图 | 问题 | CLIP热图 | M2热图
    """
    N = len(samples)
    col_widths = [2.5, 2.5, 3.5, 3.5]
    col_labels = ["Original Image", "Question & Answer", "Single-CLIP", "M2 Full Model"]

    fig = plt.figure(figsize=(sum(col_widths), N * 3.2 + 0.6), dpi=dpi)
    fig.patch.set_facecolor("#1a1a2e")

    # 列标题行 (row 0) + N 数据行
    gs = gridspec.GridSpec(
        N + 1, 4,
        height_ratios=[0.25] + [1.0] * N,
        width_ratios=col_widths,
        hspace=0.04, wspace=0.03,
    )

    # 列标题
    header_colors = ["#16213e", "#16213e", "#1a4066", "#0f3460"]
    for ci, (label, hc) in enumerate(zip(col_labels, header_colors)):
        ax = fig.add_subplot(gs[0, ci])
        ax.set_facecolor(hc)
        ax.text(0.5, 0.5, label,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white", transform=ax.transAxes)
        ax.axis("off")

    for ri, (rec, dbg_clip, dbg_m2) in enumerate(zip(samples, attn_clip, attn_m2)):
        image_pil = Image.open(rec["image_path"]).convert("RGB")
        img_arr   = np.array(image_pil)
        row = ri + 1

        # ── 列 0：原始图像 ─────────────────────────────────────
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(img_arr)
        ax0.set_title(f"#{ri+1}  [{rec['question_type']}]",
                      fontsize=7.5, color="#aaaacc", pad=2)
        ax0.axis("off")

        # ── 列 1：问题 + 答案文本 ─────────────────────────────
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.set_facecolor("#0d0d1a")
        q_text = wrap_text("Q: " + rec["question"], 38)
        a_text = "A: " + rec["answer"]
        ax1.text(0.05, 0.65, q_text,
                 ha="left", va="top", fontsize=7.5, color="#e0e0ff",
                 transform=ax1.transAxes, wrap=True)
        ax1.text(0.05, 0.28, a_text,
                 ha="left", va="top", fontsize=7.5, color="#88ff88",
                 transform=ax1.transAxes)
        ax1.axis("off")

        # ── 列 2：单-CLIP 注意力叠加 ──────────────────────────
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(img_arr)
        if dbg_clip and dbg_clip.get("attn") is not None:
            heat_clip = attn_to_spatial_map(
                dbg_clip["attn"], dbg_clip.get("kv_splits"), image_pil
            )
            ax2.imshow(heat_clip, cmap=_CMAP, alpha=alpha,
                       vmin=0, vmax=1,
                       extent=[0, img_arr.shape[1], img_arr.shape[0], 0])
        ax2.axis("off")

        # ── 列 3：M2 完整模型注意力叠加 ───────────────────────
        ax3 = fig.add_subplot(gs[row, 3])
        ax3.imshow(img_arr)
        if dbg_m2 and dbg_m2.get("attn") is not None:
            heat_m2 = attn_to_spatial_map(
                dbg_m2["attn"], dbg_m2.get("kv_splits"), image_pil
            )
            im = ax3.imshow(heat_m2, cmap=_CMAP, alpha=alpha,
                            vmin=0, vmax=1,
                            extent=[0, img_arr.shape[1], img_arr.shape[0], 0])
        ax3.axis("off")

    plt.suptitle(
        "Cross-Modal Attention Heatmap: Single-CLIP vs M2 Full Model\n"
        "(Higher attention on lesion region → better evidence localization)",
        fontsize=12, color="white", y=1.005, fontweight="bold",
    )
    plt.savefig(save_path, bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=dpi)
    plt.close()
    print(f"[vis] main compare: {save_path}")


def plot_grid_overlay(
    samples: List[Dict],
    attn_clip: List[Optional[dict]],
    attn_m2:   List[Optional[dict]],
    save_path: str,
    alpha: float = 0.55,
    dpi: int = 150,
):
    """
    精简双列对比（CLIP 叠加 | M2 叠加），适合论文单栏插图。
    """
    N = len(samples)
    fig, axes = plt.subplots(N, 2, figsize=(8, N * 3.2), dpi=dpi)
    fig.patch.set_facecolor("#111122")
    if N == 1:
        axes = [axes]

    for ri, (rec, dbg_clip, dbg_m2) in enumerate(zip(samples, attn_clip, attn_m2)):
        image_pil = Image.open(rec["image_path"]).convert("RGB")
        img_arr   = np.array(image_pil)

        for ci, (ax, dbg, label) in enumerate(zip(
            axes[ri],
            [dbg_clip, dbg_m2],
            ["Single-CLIP Baseline", "M2 Full Model (Proposed)"],
        )):
            ax.imshow(img_arr)
            if dbg and dbg.get("attn") is not None:
                heat = attn_to_spatial_map(
                    dbg["attn"], dbg.get("kv_splits"), image_pil
                )
                ax.imshow(heat, cmap=_CMAP, alpha=alpha,
                          vmin=0, vmax=1,
                          extent=[0, img_arr.shape[1], img_arr.shape[0], 0])
            q_short = rec["question"][:55] + "…" if len(rec["question"]) > 55 else rec["question"]
            title = f"[{rec['question_type']}] {label}"
            if ri == 0:
                ax.set_title(title, fontsize=9, color="white", pad=4)
            else:
                ax.set_title(f"[{rec['question_type']}]", fontsize=8,
                             color="#aaaacc", pad=2)
            ax.set_xlabel(q_short, fontsize=7, color="#ccccff", labelpad=2)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=dpi)
    plt.close()
    print(f"[vis] grid overlay: {save_path}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Attention heatmap: Single-CLIP vs M2 Full Model on VQA-RAD")
    ap.add_argument("--config_clip",  required=True,
                    help="单-CLIP 基线 config（通常 config/m2_cd.yaml）")
    ap.add_argument("--config_m2",    required=True,
                    help="M2 完整模型 config（config/m3.yaml）")
    ap.add_argument("--ckpt_clip",    default=None,
                    help="单-CLIP checkpoint 路径（占位，用实际路径替换）")
    ap.add_argument("--ckpt_m2",      default=None,
                    help="M2 checkpoint 路径（占位，用实际路径替换）")
    ap.add_argument("--json-path",    required=True,
                    help='VQA-RAD JSON, e.g. "...VQA_RAD Dataset Public.json"')
    ap.add_argument("--images-dir",   required=True,
                    help='VQA-RAD 图像文件夹, e.g. "...VQA_RAD Image Folder"')
    ap.add_argument("--split",        default="test",
                    choices=["train", "validation", "test"])
    ap.add_argument("--n-samples",    type=int, default=6,
                    help="随机抽取的样本数（建议 4-8）")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--alpha",        type=float, default=0.55,
                    help="热图叠加透明度（0=不可见 1=完全覆盖）")
    ap.add_argument("--out-dir",      default="outputs/vis_lesion_focus")
    ap.add_argument("--dpi",          type=int, default=150)
    args = ap.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── 采样数据 ────────────────────────────────────────────
    samples = sample_vqa_rad(
        json_path  = args.json_path,
        images_dir = args.images_dir,
        split      = args.split,
        n_samples  = args.n_samples,
        seed       = args.seed,
    )
    for i, s in enumerate(samples):
        print(f"  [{i+1}] {s['image_name']:30s}  Q: {s['question'][:60]}")

    # ── 阶段 A：加载单-CLIP → 推理 → 释放 ─────────────────
    print("\n[model] Loading Single-CLIP baseline...")
    model_clip = load_clip_model(args.config_clip, args.ckpt_clip, device)

    attn_clip = []
    for i, rec in enumerate(samples):
        img = Image.open(rec["image_path"]).convert("RGB")
        dbg = get_attn_debug_m1(model_clip, img, device)
        attn_clip.append(debug_to_cpu(dbg))
        print(f"  CLIP [{i+1}/{len(samples)}] done")

    del model_clip
    torch.cuda.empty_cache()
    if device.type == "cuda":
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"[mem] CLIP released. Free GPU: {free_gb:.2f} GB")

    # ── 阶段 B：加载 M2 → 推理 → 释放 ────────────────────
    print("\n[model] Loading M2 Full Model...")
    model_m2 = load_m2_model(args.config_m2, args.ckpt_m2, device)

    attn_m2 = []
    for i, rec in enumerate(samples):
        img = Image.open(rec["image_path"]).convert("RGB")
        dbg = get_attn_debug_m2(model_m2, img, rec["question"], device)
        attn_m2.append(debug_to_cpu(dbg))
        print(f"  M2  [{i+1}/{len(samples)}] done")

    del model_m2
    torch.cuda.empty_cache()

    # ── 阶段 C：绘图（纯 CPU） ─────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    plot_comparison(
        samples, attn_clip, attn_m2,
        save_path = os.path.join(args.out_dir, "lesion_focus_compare.png"),
        alpha     = args.alpha,
        dpi       = args.dpi,
    )

    plot_grid_overlay(
        samples, attn_clip, attn_m2,
        save_path = os.path.join(args.out_dir, "lesion_focus_grid.png"),
        alpha     = args.alpha,
        dpi       = args.dpi,
    )

    print(f"\n[done] Outputs saved to: {args.out_dir}")
    print("  lesion_focus_compare.png  — 4-column detailed comparison")
    print("  lesion_focus_grid.png     — 2-column grid for paper")


if __name__ == "__main__":
    main()
