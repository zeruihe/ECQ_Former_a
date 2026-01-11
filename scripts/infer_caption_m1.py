from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, Any, List, Optional

import torch

from data import build_caption_loader
from models.ecqformer_m1_offline import ECQFormerM1Offline


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_latest_ckpt(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    pts = [p for p in os.listdir(ckpt_dir) if p.endswith(".pt")]
    if not pts:
        return None
    # prefer final.pt
    if "final.pt" in pts:
        return os.path.join(ckpt_dir, "final.pt")
    # step_*.pt
    def key_fn(name: str) -> int:
        if name.startswith("step_") and name.endswith(".pt"):
            try:
                return int(name[len("step_"):-3])
            except Exception:
                return -1
        return -1
    pts.sort(key=key_fn)
    return os.path.join(ckpt_dir, pts[-1])


def _load_trainable_state(model: torch.nn.Module, ckpt_path: str) -> None:
    # PyTorch 2.6+: torch.load defaults weights_only=True, may fail for our ckpt dict.
    # We trust our own checkpoints -> load with weights_only=False.
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint is not a dict: type={type(ckpt)}")

    # Support your training script key names:
    cand_keys = [
        "model_trainable",        # <-- your current checkpoint key
        "model_trainable_state",
        "trainable_state",
        "model_state",
        "model",
        "state_dict",
    ]

    state = None
    for k in cand_keys:
        if k in ckpt and isinstance(ckpt[k], dict):
            state = ckpt[k]
            break

    # if checkpoint itself is a state_dict
    if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt

    if state is None:
        raise KeyError(
            f"Cannot find model state in checkpoint: {ckpt_path}. "
            f"keys={list(ckpt.keys())}"
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys (strict=False): {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys (strict=False): {len(unexpected)}")



@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out_jsonl", type=str, default=None)

    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    args = parser.parse_args()
    cfg = _load_yaml(args.config)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = os.path.join(cfg["output"]["out_dir"], cfg["output"]["run_name"])
    os.makedirs(out_dir, exist_ok=True)

    if args.out_jsonl is None:
        args.out_jsonl = os.path.join(out_dir, f"samples_{args.split}_{args.num_samples}.jsonl")

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = _find_latest_ckpt(os.path.join(out_dir, "checkpoints"))
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found. Provide --ckpt or ensure outputs/.../checkpoints exists.")
    print(f"[infer] ckpt = {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    mcfg = cfg["models"]
    model = ECQFormerM1Offline(
        llama_local_dir=mcfg["llama_dir"],
        clip_local_dir=mcfg["clip_dir"],
        dinov2_local_dir=mcfg["dinov2_dir"],
        biomedclip_local_dir=mcfg["biomedclip_dir"],
        d_bridge=int(mcfg.get("d_bridge", 768)),
        m_queries=int(mcfg.get("m_queries", 96)),
        meq_layers=int(mcfg.get("meq_layers", 2)),
        meq_heads=int(mcfg.get("meq_heads", 12)),
        torch_dtype=torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16,
    ).to(device)

    _load_trainable_state(model, ckpt_path)
    model.eval()

    # Data
    loader = build_caption_loader(
        split=args.split,
        sources_cfg=cfg["data"]["sources"],
        batch_size=1,  # generation per-sample
        shuffle=True,  # sample randomly
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        return_pil=True,
    )

    # sample
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    n = 0
    for batch in loader:
        images_pil = batch["images_pil"]  # List[PIL] length=1
        captions = batch["captions"]      # List[str] length=1
        meta = batch["meta"][0]           # dict

        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

        t_gen0 = time.time()
        pred = model.generate_caption(
            images_pil=images_pil,
            prompt_prefix=cfg["train"].get("prompt_prefix", "Describe the medical image:\n"),
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_p=args.top_p,
        )[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_gen = (time.time() - t_gen0) * 1000.0

        peak_mem_gb = None
        if device.type == "cuda":
            peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        row = {
            "dataset": meta.get("dataset"),
            "sample_id": meta.get("sample_id"),
            "image_path": meta.get("image_path"),
            "gt_caption": captions[0],
            "pred_caption": pred,
            "gen_time_ms": round(t_gen, 3),
            "peak_mem_gb": None if peak_mem_gb is None else round(float(peak_mem_gb), 4),
        }
        rows.append(row)
        n += 1
        if n >= args.num_samples:
            break

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dt = time.time() - t0
    print(f"[infer] wrote {len(rows)} samples -> {args.out_jsonl}")
    print(f"[infer] time = {dt:.2f}s")


if __name__ == "__main__":
    main()