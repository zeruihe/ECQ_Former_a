# scripts/eval_caption_m1.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
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
    if "final.pt" in pts:
        return os.path.join(ckpt_dir, "final.pt")

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
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cand_keys = ["model_trainable_state", "trainable_state", "model_state", "model", "state_dict"]
    state = None
    for k in cand_keys:
        if k in ckpt and isinstance(ckpt[k], dict):
            state = ckpt[k]
            break
    if state is None and isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    if state is None:
        raise KeyError(f"Cannot find model state in checkpoint: {ckpt_path}")

    model.load_state_dict(state, strict=False)


def _rouge_l_f1(preds: List[str], refs: List[str]) -> float:
    # offline ROUGE-L via rouge-score
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)["rougeL"].fmeasure
        scores.append(s)
    return float(sum(scores) / max(1, len(scores)))


def _bertscore_f1(preds: List[str], refs: List[str], model_type_or_path: str, device: str) -> Optional[float]:
    # optional (needs local bertscore model)
    try:
        from bert_score import score as bert_score
    except Exception:
        return None

    P, R, F1 = bert_score(
        cands=preds,
        refs=refs,
        model_type=model_type_or_path,
        device=device,
        lang="en",
        rescale_with_baseline=True,
        verbose=False,
    )
    return float(F1.mean().item())


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--max_eval", type=int, default=512)  # 控制评测量，避免太慢
    parser.add_argument("--save_preds", action="store_true")

    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    # BERTScore optional: point to a local path if offline
    parser.add_argument("--bertscore_model", type=str, default=None)

    args = parser.parse_args()
    cfg = _load_yaml(args.config)

    out_dir = os.path.join(cfg["output"]["out_dir"], cfg["output"]["run_name"])
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"metrics_{args.split}.json")

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = _find_latest_ckpt(os.path.join(out_dir, "checkpoints"))
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found. Provide --ckpt or ensure outputs/.../checkpoints exists.")
    print(f"[eval] ckpt = {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if device.type == "cuda" else "cpu"

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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loader = build_caption_loader(
        split=args.split,
        sources_cfg=cfg["data"]["sources"],
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        return_pil=True,
    )

    preds: List[str] = []
    refs: List[str] = []
    metas: List[Dict[str, Any]] = []

    times_ms: List[float] = []
    peak_mems: List[float] = []

    n = 0
    t0 = time.time()
    for batch in loader:
        images_pil = batch["images_pil"]
        gt = batch["captions"][0]
        meta = batch["meta"][0]

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t1 = time.time()
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
        dt = (time.time() - t1) * 1000.0

        preds.append(pred)
        refs.append(gt)
        metas.append(meta)
        times_ms.append(dt)

        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            peak_mems.append(float(peak))

        n += 1
        if n >= args.max_eval:
            break

    rougeL = _rouge_l_f1(preds, refs)

    bert_f1 = None
    if args.bertscore_model is not None:
        bert_f1 = _bertscore_f1(preds, refs, model_type_or_path=args.bertscore_model, device=device_str)

    avg_time = float(sum(times_ms) / max(1, len(times_ms)))
    avg_peak = float(sum(peak_mems) / max(1, len(peak_mems))) if peak_mems else None

    metrics = {
        "split": args.split,
        "num_eval": len(preds),
        "rougeL_f1": rougeL,
        "bertscore_f1": bert_f1,
        "trainable_params": trainable_params,
        "avg_gen_time_ms": avg_time,
        "avg_peak_mem_gb": avg_peak,
        "ckpt": ckpt_path,
        "elapsed_s": time.time() - t0,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(
        f"[eval] split={args.split} n={metrics['num_eval']} "
        f"ROUGE-L(F1)={metrics['rougeL_f1']:.4f} "
        + (f"BERTScore(F1)={metrics['bertscore_f1']:.4f} " if metrics["bertscore_f1"] is not None else "BERTScore=NA ")
        + f"trainable={metrics['trainable_params']/1e6:.2f}M "
        + f"lat={metrics['avg_gen_time_ms']:.2f}ms "
        + (f"peak={metrics['avg_peak_mem_gb']:.3f}GB " if metrics["avg_peak_mem_gb"] is not None else "")
        + f"-> {metrics_path}"
    )

    if args.save_preds:
        pred_path = os.path.join(out_dir, f"preds_{args.split}.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for p, r, m in zip(preds, refs, metas):
                f.write(json.dumps({"meta": m, "gt": r, "pred": p}, ensure_ascii=False) + "\n")
        print(f"[eval] wrote preds -> {pred_path}")


if __name__ == "__main__":
    main()
