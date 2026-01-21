# scripts/visualize_attn_multi_encoder.py
"""
Visualize multi-encoder attention heatmaps.

This script:
- loads model + checkpoint
- reads a few images
- runs model.build_soft_prompt(return_attn=True, return_debug=True)
- saves:
  1) encoder-level heatmap: queries x encoders
  2) token-level heatmap: queries x tokens (with split lines)
  3) keep_mask stats (if adaptive_drop enabled)
"""

from __future__ import annotations

import os
import argparse
from typing import List
from PIL import Image

import torch
import yaml

from models.ecqformer_m1_offline import ECQFormerM1Offline
from utils.attn_viz import plot_encoder_level_heatmap, plot_token_level_heatmap, save_keep_mask


def load_images(paths: List[str]) -> List[Image.Image]:
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(img)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, help="Path to YAML config (e.g., config/m2_cd.yaml or m1.yaml)")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (final.pt / best.pt)")
    ap.add_argument("--images", type=str, nargs="+", required=True, help="Image paths to visualize")
    ap.add_argument("--out-dir", type=str, default="outputs/attn_viz", help="Output directory")
    ap.add_argument("--attn-layer", type=int, default=-1, help="Which MEQ layer's cross-attn to visualize (default last)")
    ap.add_argument("--max-tokens-plot", type=int, default=512, help="Token heatmap width cap (pooled if exceed)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mcfg = cfg.get("models", cfg)  # 兼容你的 config 结构

    model = ECQFormerM1Offline(
        llama_local_dir=mcfg.get("llama_local_dir", mcfg.get("llama_dir")),
        clip_local_dir=mcfg.get("clip_local_dir", mcfg.get("clip_dir")),
        dinov2_local_dir=mcfg.get("dinov2_local_dir", mcfg.get("dinov2_dir")),
        biomedclip_local_dir=mcfg.get("biomedclip_local_dir", mcfg.get("biomedclip_dir")),
        d_bridge=int(mcfg.get("d_bridge", 768)),
        meq_layers=int(mcfg.get("meq_layers", 2)),
        meq_heads=int(mcfg.get("meq_heads", 12)),
        m_queries=int(mcfg.get("m_queries", 96)),
        # third-chapter switches (if present)
        attn_type=str(mcfg.get("attn_type", "standard")),
        phi=str(mcfg.get("phi", "identity")),
        score_scale=bool(mcfg.get("score_scale", True)),
        score_norm=bool(mcfg.get("score_norm", False)),
        adaptive_drop=mcfg.get("adaptive_drop", None),
        torch_dtype=torch.bfloat16,
    ).to(device)

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # 你的工程里通常是 trainable-only 存储；这里用 strict=False 更稳
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()

    images_pil = load_images(args.images)

    with torch.inference_mode():
        _, debug = model.build_soft_prompt(
            images_pil, device=device,
            return_attn=True,
            return_debug=True,
            attn_layer=args.attn_layer,
        )

    if debug is None or debug.get("attn", None) is None:
        raise RuntimeError("No attention returned. Ensure return_attn=True and your MEQFormer is updated.")

    attn = debug["attn"]                # (B,H,M,N)
    kv_splits = debug.get("kv_splits")  # [(name,s,e), ...]

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) token-level heatmap with split lines
    token_path = os.path.join(args.out_dir, "token_level.png")
    plot_token_level_heatmap(
        attn=attn,
        kv_splits=kv_splits,
        out_path=token_path,
        sample_idx=0,
        head_reduce="mean",
        max_tokens_plot=args.max_tokens_plot,
    )

    # 2) encoder-level heatmap (queries x encoders)
    if "attn_enc" in debug and debug["attn_enc"] is not None:
        enc_path = os.path.join(args.out_dir, "encoder_level.png")
        plot_encoder_level_heatmap(
            attn_enc=debug["attn_enc"],  # (B,H,M,K)
            kv_splits=kv_splits,
            out_path=enc_path,
            sample_idx=0,
            head_reduce="mean",
        )

    # 3) keep_mask stats (if adaptive drop enabled)
    if debug.get("keep_mask", None) is not None:
        keep_path = os.path.join(args.out_dir, "keep_mask.txt")
        save_keep_mask(
            keep_mask=debug["keep_mask"],
            kv_splits=kv_splits,
            out_path=keep_path,
            sample_idx=0,
        )

    print(f"[OK] Saved to: {args.out_dir}")
    print(f" - token heatmap: {token_path}")
    if "attn_enc" in debug:
        print(f" - encoder heatmap: {os.path.join(args.out_dir, 'encoder_level.png')}")
    if debug.get("keep_mask", None) is not None:
        print(f" - keep stats: {os.path.join(args.out_dir, 'keep_mask.txt')}")


if __name__ == "__main__":
    main()

"""
python scripts/visualize_attn_multi_encoder.py config/m2_cd.yaml \
  --ckpt /root/autodl-tmp/outputs/xxx/checkpoints/final.pt \
  --images /root/autodl-tmp/some_image1.jpg /root/autodl-tmp/some_image2.jpg \
  --out-dir outputs/attn_viz_demo \
  --attn-layer -1
"""