# scripts/visualize_attn_compare_m1_m2.py
# -*- coding: utf-8 -*-
"""
多编码器注意力热图对比：M1（无显式控制）vs M2（FiLM + 门控）
对应论文第四章可视化实验（1）：注意力由"扩散式"转向"结构化聚焦"

输出（保存至 --out-dir）：
  attn_encoder_compare.png   — encoder-level 热图并排对比（M1 vs M2）
  attn_spatial_clip.png      — CLIP 编码器注意力空间叠加（M1 左 | M2 右）
  attn_spatial_bio.png       — BiomedCLIP 编码器注意力空间叠加
  attn_spatial_dino.png      — DINOv2 编码器注意力空间叠加
  attn_token_compare.png     — query×token token级别热图并排对比

用法：
  python scripts/visualize_attn_compare_m1_m2.py \\
    --config_m1 config/m2_cd.yaml \\
    --config_m2 config/m3.yaml \\
    --ckpt_m1 /root/outputs/m1_finetune/checkpoints/best.pt \\
    --ckpt_m2 /root/outputs/m3_finetune_vqa-rad/checkpoints/best.pt \\
    --image /path/to/medical_image.jpg \\
    --question "What is the main finding in this image?" \\
    --out-dir /root/outputs/vis_attn_compare
"""
from __future__ import annotations

import os, sys, argparse, math
from pathlib import Path
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
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ─────────────────────────────────────────────────────────────
# 模型加载辅助
# ─────────────────────────────────────────────────────────────

def _get_enabled_encoders(m_cfg) -> List[str]:
    enc = m_cfg.get("enabled_encoders", None)
    return [str(e) for e in enc] if enc else ["clip", "biomedclip", "dinov2"]

def _auto_m_queries(m_cfg) -> int:
    enabled = _get_enabled_encoders(m_cfg)
    return 32 * len(enabled) if m_cfg.get("auto_m_queries", True) else int(m_cfg["m_queries"])

def _load_ckpt(model: torch.nn.Module, ckpt_path: str) -> None:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state   = payload.get("model_trainable", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] {ckpt_path}  missing={len(missing)} unexpected={len(unexpected)}")

def load_m1(cfg_path: str, ckpt_path: Optional[str], device: torch.device):
    from models.ecqformer_m1_offline import ECQFormerM1Offline
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m_cfg = cfg["models"]
    amp   = torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16
    model = ECQFormerM1Offline(
        llama_local_dir      = m_cfg["llama_dir"],
        clip_local_dir       = m_cfg["clip_dir"],
        dinov2_local_dir     = m_cfg["dinov2_dir"],
        biomedclip_local_dir = m_cfg["biomedclip_dir"],
        d_bridge      = int(m_cfg["d_bridge"]),
        meq_layers    = int(m_cfg["meq_layers"]),
        meq_heads     = int(m_cfg["meq_heads"]),
        m_queries     = _auto_m_queries(m_cfg),
        attn_type     = m_cfg.get("attn_type", "param_free"),
        phi           = m_cfg.get("phi", "silu"),
        score_scale   = bool(m_cfg.get("score_scale", True)),
        score_norm    = bool(m_cfg.get("score_norm", False)),
        adaptive_drop = m_cfg.get("adaptive_drop", None),
        enabled_encoders = _get_enabled_encoders(m_cfg),
        torch_dtype   = amp,
    ).to(device)
    if ckpt_path:
        _load_ckpt(model, ckpt_path)
    model.eval()
    return model, cfg

def load_m2(cfg_path: str, ckpt_path: Optional[str], device: torch.device):
    from models.ecqformer_m2_offline import ECQFormerM2Offline
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m_cfg = cfg["models"]
    amp   = torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16
    model = ECQFormerM2Offline(
        llama_local_dir      = m_cfg["llama_dir"],
        clip_local_dir       = m_cfg["clip_dir"],
        dinov2_local_dir     = m_cfg["dinov2_dir"],
        biomedclip_local_dir = m_cfg["biomedclip_dir"],
        d_bridge      = int(m_cfg["d_bridge"]),
        meq_layers    = int(m_cfg["meq_layers"]),
        meq_heads     = int(m_cfg["meq_heads"]),
        m_queries     = _auto_m_queries(m_cfg),
        attn_type     = m_cfg.get("attn_type", "param_free"),
        phi           = m_cfg.get("phi", "silu"),
        score_scale   = bool(m_cfg.get("score_scale", True)),
        score_norm    = bool(m_cfg.get("score_norm", False)),
        adaptive_drop = m_cfg.get("adaptive_drop", None),
        gating        = m_cfg.get("gating", None),
        film          = m_cfg.get("film", None),
        enabled_encoders = _get_enabled_encoders(m_cfg),
        torch_dtype   = amp,
    ).to(device)
    if ckpt_path:
        _load_ckpt(model, ckpt_path)
    model.eval()
    return model, cfg


# ─────────────────────────────────────────────────────────────
# 注意力提取
# ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def extract_attn_m1(model, image_pil: Image.Image, device: torch.device):
    _, debug = model.build_soft_prompt(
        [image_pil], device=device,
        return_attn=True, return_debug=True, attn_layer=-1,
    )
    return debug   # attn:(1,H,M,N), attn_enc:(1,H,M,K), kv_splits:[...]

@torch.inference_mode()
def extract_attn_m2(model, image_pil: Image.Image, question: str, device: torch.device):
    _, debug = model.build_soft_prompt(
        [image_pil], device=device, questions=[question],
        return_attn=True, return_debug=True, attn_layer=-1,
    )
    return debug


# ─────────────────────────────────────────────────────────────
# 空间注意力映射（将 query 均值注意力映射回图像空间）
# ─────────────────────────────────────────────────────────────

def _attn_to_spatial(
    attn: torch.Tensor,          # (1,H,M,N_total)
    kv_splits: List,             # [(name,s,e), ...]
    enc_name: str,
    image_pil: Image.Image,
) -> np.ndarray:
    """
    提取指定编码器的空间注意力热图，上采样到图像尺寸。
    Returns: (H_img, W_img) float32 ndarray，值域 [0,1]
    """
    s, e = next((s, e) for name, s, e in kv_splits if name == enc_name)
    attn_enc = attn[0, :, :, s:e]        # (H, M, N_enc)
    attn_enc = attn_enc.mean(dim=0)       # (M, N_enc)  —— head 平均
    attn_map = attn_enc.mean(dim=0)       # (N_enc,)    —— query 平均
    attn_map = attn_map.float().cpu().numpy()

    N = len(attn_map)
    grid = int(round(math.sqrt(N)))
    # 如果 sqrt 不整除，取最接近的矩形
    h_grid = grid
    w_grid = grid if grid * grid == N else math.ceil(N / grid)
    pad = h_grid * w_grid - N
    if pad > 0:
        attn_map = np.concatenate([attn_map, np.zeros(pad)])
    attn_map = attn_map.reshape(h_grid, w_grid)

    # 归一化到 [0,1]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # 上采样到图像尺寸（双线性）
    W_img, H_img = image_pil.size
    attn_pil = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        (W_img, H_img), Image.BILINEAR
    )
    return np.array(attn_pil) / 255.0


# ─────────────────────────────────────────────────────────────
# 绘图函数
# ─────────────────────────────────────────────────────────────

ENC_COLORS = {"clip": "#4C9BE8", "biomedclip": "#E87B4C", "dinov2": "#5CC77A"}
ENC_LABELS = {"clip": "CLIP", "biomedclip": "BiomedCLIP", "dinov2": "DINOv2"}


def plot_encoder_compare(
    debug_m1, debug_m2,
    enabled_encoders: List[str],
    save_path: str, dpi: int = 150,
):
    """
    并排 encoder-level 热图（M1 左 | M2 右）。
    attn_enc: (1, H, M, K)  → 每格 = 某 query 对某编码器的注意力
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)
    titles = ["M1（无显式控制）", "M2（FiLM + 门控）"]

    for ax, debug, title in zip(axes, [debug_m1, debug_m2], titles):
        if debug is None or debug.get("attn_enc") is None:
            ax.text(0.5, 0.5, "No attn_enc", ha="center", va="center")
            ax.set_title(title)
            continue

        enc_attn = debug["attn_enc"][0].float().cpu()   # (H, M, K)
        enc_attn = enc_attn.mean(dim=0).numpy()         # (M, K)
        im = ax.imshow(enc_attn, cmap="Blues", aspect="auto",
                       vmin=0, vmax=enc_attn.max())
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("编码器", fontsize=11)
        ax.set_ylabel("桥接查询 Query 索引", fontsize=11)
        ax.set_xticks(range(len(enabled_encoders)))
        ax.set_xticklabels([ENC_LABELS.get(e, e) for e in enabled_encoders], fontsize=10)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.suptitle("多编码器注意力热图对比（Query × Encoder）", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[vis] encoder compare: {save_path}")


def plot_token_level_compare(
    debug_m1, debug_m2,
    kv_splits, save_path: str, dpi: int = 150,
):
    """
    并排 token-level 热图（M1 左 | M2 右），带编码器分割线。
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=dpi)
    titles = ["M1（无显式控制）", "M2（FiLM + 门控）"]

    for ax, debug, title in zip(axes, [debug_m1, debug_m2], titles):
        if debug is None or debug.get("attn") is None:
            ax.text(0.5, 0.5, "No attn", ha="center", va="center")
            ax.set_title(title)
            continue

        attn = debug["attn"][0].float().mean(0).cpu().numpy()  # (M, N)
        im = ax.imshow(attn, cmap="hot", aspect="auto")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("视觉 Token 索引", fontsize=11)
        ax.set_ylabel("桥接查询 Query 索引", fontsize=11)

        if kv_splits:
            for name, s, e in kv_splits:
                ax.axvline(x=s - 0.5, color="cyan", linewidth=1.5, linestyle="--", alpha=0.8)
                mid = (s + e) / 2
                ax.text(mid, -2, ENC_LABELS.get(name, name),
                        ha="center", va="top", fontsize=8, color="white",
                        bbox=dict(boxstyle="round,pad=0.15", fc="#333", ec="none", alpha=0.7))

        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.suptitle("Token 级别注意力热图对比（Query × Token）", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[vis] token compare: {save_path}")


def plot_spatial_compare(
    debug_m1, debug_m2,
    image_pil: Image.Image,
    enc_name: str, save_path: str, dpi: int = 150,
):
    """
    单路编码器：空间注意力叠加原图（M1 左 | M2 右）。
    展示 M1 "扩散" vs M2 "聚焦" 的空间分布差异。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
    img_arr = np.array(image_pil.convert("RGB"))
    titles  = ["M1（无显式控制）", "M2（FiLM + 门控）"]

    for ax, debug, title in zip(axes, [debug_m1, debug_m2], titles):
        ax.set_title(f"{title}\n{ENC_LABELS.get(enc_name, enc_name)} 注意力", fontsize=11)
        ax.axis("off")
        if debug is None or debug.get("attn") is None:
            ax.imshow(img_arr)
            continue
        try:
            heat = _attn_to_spatial(debug["attn"], debug["kv_splits"], enc_name, image_pil)
            ax.imshow(img_arr)
            im = ax.imshow(heat, cmap="jet", alpha=0.5,
                           vmin=0, vmax=1, extent=[0, img_arr.shape[1], img_arr.shape[0], 0])
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="4%", pad=0.05)
            plt.colorbar(im, cax=cax)
        except Exception as ex:
            ax.imshow(img_arr)
            ax.set_title(f"{title}\n{ENC_LABELS.get(enc_name, enc_name)} (err: {ex})", fontsize=9)

    plt.suptitle(
        f"注意力空间分布对比 — {ENC_LABELS.get(enc_name, enc_name)} 编码器",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[vis] spatial {enc_name}: {save_path}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_m1", required=True, help="M1 config yaml (e.g. config/m2_cd.yaml)")
    ap.add_argument("--config_m2", required=True, help="M2 config yaml (e.g. config/m3.yaml)")
    ap.add_argument("--ckpt_m1",   default=None,  help="M1 checkpoint (可省略，使用随机初始化调试)")
    ap.add_argument("--ckpt_m2",   default=None,  help="M2 checkpoint")
    ap.add_argument("--image",     required=True, help="医学图像路径")
    ap.add_argument("--question",  required=True, help="问题文本（M2 门控和 FiLM 的条件）")
    ap.add_argument("--out-dir",   default="outputs/vis_attn_compare")
    ap.add_argument("--dpi",       type=int, default=150)
    args = ap.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    image_pil = Image.open(args.image).convert("RGB")
    print(f"[image] {args.image}  size={image_pil.size}")
    print(f"[question] {args.question}")

    # 加载双模型
    print("\n[load] M1 model...")
    m1, _ = load_m1(args.config_m1, args.ckpt_m1, device)
    print("[load] M2 model...")
    m2, cfg_m2 = load_m2(args.config_m2, args.ckpt_m2, device)

    enabled_encoders = _get_enabled_encoders(cfg_m2["models"])

    # 推理
    print("\n[infer] M1 attention...")
    debug_m1 = extract_attn_m1(m1, image_pil, device)
    print("[infer] M2 attention...")
    debug_m2 = extract_attn_m2(m2, image_pil, args.question, device)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1：encoder-level 并排对比
    plot_encoder_compare(
        debug_m1, debug_m2, enabled_encoders,
        save_path=os.path.join(args.out_dir, "attn_encoder_compare.png"),
        dpi=args.dpi,
    )

    # 2：token-level 并排对比（用 M1 的 kv_splits，两者应相同）
    kv_splits = (debug_m1 or debug_m2 or {}).get("kv_splits", None)
    plot_token_level_compare(
        debug_m1, debug_m2, kv_splits,
        save_path=os.path.join(args.out_dir, "attn_token_compare.png"),
        dpi=args.dpi,
    )

    # 3：各路编码器空间注意力叠加
    for enc_name in enabled_encoders:
        plot_spatial_compare(
            debug_m1, debug_m2, image_pil, enc_name,
            save_path=os.path.join(args.out_dir, f"attn_spatial_{enc_name}.png"),
            dpi=args.dpi,
        )

    print(f"\n[done] All outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
