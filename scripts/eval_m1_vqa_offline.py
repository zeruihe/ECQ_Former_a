# eval_m1_vqa_offline.py
from __future__ import annotations

import os
import sys
import json
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
import torch
from tqdm import tqdm

from data import build_vqa_loader
from models.ecqformer_m1_offline import ECQFormerM1Offline
from utils.checkpoint import load_checkpoint
from utils.vqa_metrics import compute_vqa_metrics


def _get_enabled_encoders(cfg_models) -> List[str]:
    enc = cfg_models.get("enabled_encoders", None)
    if enc is None:
        return ["clip", "biomedclip", "dinov2"]
    return [str(e) for e in enc]


def _auto_m_queries(cfg_models) -> int:
    enabled = _get_enabled_encoders(cfg_models)
    if bool(cfg_models.get("auto_m_queries", False)):
        return 32 * len(enabled)
    return int(cfg_models["m_queries"])


@torch.inference_mode()
def main(config_path: str = "config/finetune.yaml"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    eval_cfg = cfg.get("eval", {})
    split = str(eval_cfg.get("split", "validation"))
    loader = build_vqa_loader(
        dataset_id=str(cfg["data"]["dataset_id"]),
        split=split,
        cache_dir=cfg["data"].get("cache_dir", None),
        local_dir=cfg["data"].get("local_dir", None),
        images_dir=cfg["data"].get("images_dir", None),
        language=str(cfg["data"].get("language", "en")),
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        shuffle=False,
        max_samples=eval_cfg.get("max_samples", None),
    )

    # model
    use_bf16 = bool(cfg["train"].get("bf16", True))
    use_fp16 = bool(cfg["train"].get("fp16", False))
    if use_bf16 and use_fp16:
        raise ValueError("bf16 and fp16 cannot both be true.")
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    enabled_encoders = _get_enabled_encoders(cfg["models"])
    m_queries = _auto_m_queries(cfg["models"])

    model = ECQFormerM1Offline(
        llama_local_dir=cfg["models"]["llama_dir"],
        clip_local_dir=cfg["models"]["clip_dir"],
        dinov2_local_dir=cfg["models"]["dinov2_dir"],
        biomedclip_local_dir=cfg["models"]["biomedclip_dir"],
        d_bridge=int(cfg["models"]["d_bridge"]),
        meq_layers=int(cfg["models"]["meq_layers"]),
        meq_heads=int(cfg["models"]["meq_heads"]),
        m_queries=int(m_queries),
        enabled_encoders=enabled_encoders,
        torch_dtype=amp_dtype,
    ).to(device)

    # checkpoint
    ckpt_path = eval_cfg.get("ckpt", None)
    if ckpt_path is None:
        out_dir = cfg["output"]["out_dir"]
        run_name = cfg["output"]["run_name"]
        ckpt_path = os.path.join(out_dir, run_name, "checkpoints", "best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(out_dir, run_name, "checkpoints", "final.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[eval] loading ckpt: {ckpt_path}")
    load_checkpoint(ckpt_path=ckpt_path, model=model, optimizer=None, scaler=None, map_location="cpu")

    prompt_template = str(cfg["train"].get("prompt_template", "Question: {question}\nAnswer:"))
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 32))
    num_beams = int(eval_cfg.get("num_beams", 1))
    temperature = float(eval_cfg.get("temperature", 0.7))
    top_p = float(eval_cfg.get("top_p", 0.95))

    preds: List[str] = []
    golds: List[str] = []
    closed: List[bool] = []
    qids: List[str] = []
    questions_all: List[str] = []

    for batch in tqdm(loader, desc=f"[eval {split}]"):
        images_pil = batch["images_pil"]
        questions = batch["questions"]
        answers = batch["answers"]
        is_closed = batch["is_closed"]
        qid_batch = batch["qids"]

        out = model.generate_vqa(
            images_pil=images_pil,
            questions=questions,
            device=device,
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
        )
        preds.extend(out)
        golds.extend(answers)
        closed.extend(is_closed)
        qids.extend(qid_batch)
        questions_all.extend(questions)

    m = compute_vqa_metrics(preds, golds, closed)
    print("\n==============================")
    print(f"  Dataset: {cfg['data']['dataset_id']}")
    print(f"  Split:   {split}")
    print("------------------------------")
    print(f"  Open Acc:    {m.open_acc:.4f} (n={m.n_open})")
    print(f"  Closed Acc:  {m.closed_acc:.4f} (n={m.n_closed})")
    print(f"  Overall Acc: {m.overall_acc:.4f} (n={m.n_total})")
    print("==============================\n")

    # save per-example predictions for paper appendix / error analysis
    out_dir = cfg["output"]["out_dir"]
    run_name = cfg["output"]["run_name"]
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    pred_path = os.path.join(run_dir, f"preds_{split}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for qid, q, g, p, c in zip(qids, questions_all, golds, preds, closed):
            f.write(json.dumps({
                "qid": qid,
                "question": q,
                "gold": g,
                "pred": p,
                "is_closed": bool(c),
            }, ensure_ascii=False) + "\n")
    print(f"[eval] saved: {pred_path}")

    metrics_path = os.path.join(run_dir, f"metrics_{split}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "open_acc": m.open_acc,
            "closed_acc": m.closed_acc,
            "overall_acc": m.overall_acc,
            "n_open": m.n_open,
            "n_closed": m.n_closed,
            "n_total": m.n_total,
        }, f, ensure_ascii=False, indent=2)
    print(f"[eval] saved: {metrics_path}")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/finetune.yaml"
    main(cfg_path)
