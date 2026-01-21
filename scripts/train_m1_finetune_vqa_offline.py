# train_m1_finetune_vqa_offline.py
from __future__ import annotations

import os
import sys
import time
import json
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from data import build_vqa_loader
from models.ecqformer_m1_offline import ECQFormerM1Offline
from utils.checkpoint import set_seed, save_checkpoint, load_checkpoint, rotate_checkpoints
from utils.vqa_metrics import compute_vqa_metrics, normalize_answer, clean_prediction


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def _compute_max_steps(cfg_train: dict, dataset_size: int, batch_size: int) -> int:
    """
    Compute max_steps from num_epochs if specified, otherwise use max_steps directly.
    
    Priority:
      1. If num_epochs is set -> max_steps = ceil(dataset_size / batch_size) * num_epochs
      2. Otherwise use max_steps from config
    
    Typical epochs:
      - Pretrain: 1-3 epochs
      - Finetune: 3-10 epochs
    """
    import math
    num_epochs = cfg_train.get("num_epochs", None)
    if num_epochs is not None:
        steps_per_epoch = math.ceil(dataset_size / batch_size)
        max_steps = steps_per_epoch * int(num_epochs)
        print(f"[train] dataset_size={dataset_size}, batch_size={batch_size}")
        print(f"[train] steps_per_epoch={steps_per_epoch}, num_epochs={num_epochs}")
        print(f"[train] => max_steps={max_steps}")
        return max_steps
    else:
        return int(cfg_train["max_steps"])


def _load_trainable_only(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load trainable weights from checkpoint with shape and dtype validation.
    
    If encoder config differs between pretrain and finetune, some weights
    (like meq.queries) may have different shapes. We skip those and warn.
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("model_trainable", payload)
    
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    
    for k, v in state.items():
        if k not in model_state:
            skipped_keys.append((k, "not in model"))
            continue
        if model_state[k].shape != v.shape:
            skipped_keys.append((k, f"shape mismatch: ckpt={v.shape} vs model={model_state[k].shape}"))
            continue
        
        # Convert dtype to match model (crucial for bf16/fp16)
        target_dtype = model_state[k].dtype
        if v.dtype != target_dtype:
            v = v.to(target_dtype)
        
        # Check for NaN/Inf in checkpoint
        if torch.isnan(v).any() or torch.isinf(v).any():
            skipped_keys.append((k, "contains NaN/Inf values"))
            continue
            
        model_state[k] = v
        loaded_keys.append(k)
    
    model.load_state_dict(model_state, strict=False)
    
    print(f"[init] loaded {len(loaded_keys)} keys from checkpoint")
    if skipped_keys:
        print(f"[init] skipped {len(skipped_keys)} keys due to mismatch:")
        for k, reason in skipped_keys:
            print(f"       - {k}: {reason}")


@torch.inference_mode()
def evaluate(
    *,
    model: ECQFormerM1Offline,
    loader,
    device: torch.device,
    prompt_template: str,
    max_new_tokens: int,
    num_beams: int,
    temperature: float,
    top_p: float,
) -> Dict[str, float]:
    model.eval()
    preds: List[str] = []
    golds: List[str] = []
    closed: List[bool] = []

    for batch in tqdm(loader, desc="[eval]", leave=False):
        images_pil = batch["images_pil"]
        questions = batch["questions"]
        answers = batch["answers"]
        is_closed = batch["is_closed"]

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

    m = compute_vqa_metrics(preds, golds, closed, use_clean=True)
    
    # Debug: show first 5 predictions vs gold (with cleaning)
    print("[eval] Sample predictions (after cleaning):")
    for i in range(min(5, len(preds))):
        cleaned = clean_prediction(preds[i])
        print(f"  [{i}] pred='{cleaned}' | gold='{golds[i]}' | em={normalize_answer(cleaned) == normalize_answer(golds[i])}")
    
    print(f"[eval] BERTScore helped {m.open_bert_helped} open questions")
    
    return {
        "open_acc": m.open_acc,           # Hybrid (EM + BERTScore)
        "closed_acc": m.closed_acc,       # EM only
        "overall_acc": m.overall_acc,     # (C + O) / N
        "open_em_only": m.open_em_only,   # Open with EM only (for comparison)
        "bert_helped": m.open_bert_helped,
        "n_open": m.n_open,
        "n_closed": m.n_closed,
        "n_total": m.n_total,
    }


def main(config_path: str = "config/finetune.yaml"):
    # hard offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["output"]["out_dir"]
    run_name = cfg["output"]["run_name"]
    run_dir = os.path.join(out_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # seed
    seed = int(cfg["train"]["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # data
    train_loader = build_vqa_loader(
        dataset_id=str(cfg["data"]["dataset_id"]),
        split=str(cfg["data"]["split"]),
        cache_dir=cfg["data"].get("cache_dir", None),
        local_dir=cfg["data"].get("local_dir", None),
        images_dir=cfg["data"].get("images_dir", None),
        language=str(cfg["data"].get("language", "en")),
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        shuffle=bool(cfg["data"]["shuffle"]),
        max_samples=cfg["data"].get("max_samples", None),
    )

    eval_cfg = cfg.get("eval", {})
    do_eval = bool(eval_cfg.get("enabled", False))
    eval_loader = None
    if do_eval:
        eval_loader = build_vqa_loader(
            dataset_id=str(cfg["data"]["dataset_id"]),
            split=str(eval_cfg.get("split", "validation")),
            cache_dir=cfg["data"].get("cache_dir", None),
            local_dir=cfg["data"].get("local_dir", None),
            images_dir=cfg["data"].get("images_dir", None),
            language=str(cfg["data"].get("language", "en")),
            batch_size=int(cfg["data"]["batch_size"]),
            num_workers=int(cfg["data"]["num_workers"]),
            pin_memory=bool(cfg["data"]["pin_memory"]),
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
        # ---- Chapter-3 efficiency switches (optional) ----
        attn_type=str(cfg["models"].get("attn_type", "standard")),
        phi=str(cfg["models"].get("phi", "identity")),
        score_scale=bool(cfg["models"].get("score_scale", True)),
        score_norm=bool(cfg["models"].get("score_norm", False)),
        adaptive_drop=cfg["models"].get("adaptive_drop", None),
        # ---- End Chapter-3 efficiency switches ----
        enabled_encoders=enabled_encoders,
        torch_dtype=amp_dtype,
    ).to(device)

    print(f"[M1 finetune] trainable params = {count_trainable_params(model)/1e6:.2f}M")

    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    scaler = GradScaler(enabled=use_fp16)

    # resume / init
    start_step = 0
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    resume_from = cfg["train"].get("resume_from", None)
    auto_resume = bool(cfg["train"].get("auto_resume", True))
    init_from = cfg["train"].get("init_from", None)

    if resume_from:
        print(f"[resume] loading from explicit: {resume_from}")
        st = load_checkpoint(ckpt_path=resume_from, model=model, optimizer=optim, scaler=scaler if use_fp16 else None)
        start_step = st.step + 1
    elif auto_resume and os.path.exists(latest_path):
        print(f"[resume] auto resume from: {latest_path}")
        st = load_checkpoint(ckpt_path=latest_path, model=model, optimizer=optim, scaler=scaler if use_fp16 else None)
        start_step = st.step + 1
    else:
        print("[resume] start from scratch")
        if init_from:
            # Support both file path and directory path
            # If directory: try best.pt first, then final.pt as fallback
            if os.path.isdir(init_from):
                best_path = os.path.join(init_from, "best.pt")
                final_path = os.path.join(init_from, "final.pt")
                if os.path.exists(best_path):
                    init_from = best_path
                    print(f"[init] found best.pt, loading from: {init_from}")
                elif os.path.exists(final_path):
                    init_from = final_path
                    print(f"[init] best.pt not found, fallback to final.pt: {init_from}")
                else:
                    raise FileNotFoundError(f"No checkpoint found in {init_from}")
            else:
                print(f"[init] loading trainable from: {init_from}")
            _load_trainable_only(model, init_from)

    # train settings
    model.train()
    grad_accum = int(cfg["train"]["grad_accum"])
    
    # Compute max_steps from num_epochs or use max_steps directly
    dataset_size = len(train_loader.dataset)
    batch_size = int(cfg["data"]["batch_size"])
    max_steps = _compute_max_steps(cfg["train"], dataset_size, batch_size)
    max_length = int(cfg["train"]["max_length"])
    prompt_template = str(cfg["train"]["prompt_template"])
    log_every = int(cfg["train"]["log_every"])
    save_every = int(cfg["train"]["save_every"])
    keep_last_k = int(cfg["train"].get("keep_last_k", 3))
    save_latest = bool(cfg["train"].get("save_latest", True))

    # fast-forward dataloader when resuming
    it = iter(train_loader)
    for _ in range(start_step):
        try:
            next(it)
        except StopIteration:
            it = iter(train_loader)

    running = 0.0
    t0 = time.time()
    best_overall: float = -1.0

    optim.zero_grad(set_to_none=True)

    pbar = tqdm(range(start_step, max_steps), total=max_steps - start_step)
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        images_pil = batch["images_pil"]
        questions = batch["questions"]
        answers = batch["answers"]

        with autocast(enabled=True, dtype=amp_dtype):
            loss = model.forward_vqa_finetune(
                images_pil=images_pil,
                questions=questions,
                answers=answers,
                device=device,
                prompt_template=prompt_template,
                max_length=max_length,
            )
            loss = loss / grad_accum

        # NaN detection with debug info
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[ERROR] NaN/Inf loss detected at step {step}")
            print(f"  questions: {questions[:2]}...")
            print(f"  answers: {answers[:2]}...")
            # Check model parameters for NaN
            for name, param in model.named_parameters():
                if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
                    print(f"  NaN/Inf in param: {name}")
            raise RuntimeError(f"NaN loss at step {step}. Check model weights and data.")

        if use_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running += loss.item()

        if (step + 1) % grad_accum == 0:
            if use_fp16:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
            optim.zero_grad(set_to_none=True)

        if step % log_every == 0:
            avg_loss = running / (step - start_step + 1)
            pbar.set_description(f"step={step} avg_loss={avg_loss:.4f}")

        # checkpoint + (optional) eval
        if save_every > 0 and step > 0 and (step % save_every == 0):
            extra = {"avg_loss": running / (step - start_step + 1)}
            if do_eval and eval_loader is not None:
                metrics = evaluate(
                    model=model,
                    loader=eval_loader,
                    device=device,
                    prompt_template=prompt_template,
                    max_new_tokens=int(eval_cfg.get("max_new_tokens", 32)),
                    num_beams=int(eval_cfg.get("num_beams", 1)),
                    temperature=float(eval_cfg.get("temperature", 0.7)),
                    top_p=float(eval_cfg.get("top_p", 0.95)),
                )
                extra.update({"eval": metrics})
                if float(metrics["overall_acc"]) > best_overall:
                    best_overall = float(metrics["overall_acc"])
                    best_path = os.path.join(ckpt_dir, "best.pt")
                    save_checkpoint(
                        ckpt_path=best_path,
                        model_trainable_state=model.trainable_state_dict(),
                        optimizer_state=optim.state_dict(),
                        scaler_state=(scaler.state_dict() if use_fp16 else None),
                        step=step,
                        config=cfg,
                        extra={"best_metric": best_overall, **extra},
                    )
                    print(f"[ckpt] best saved: {best_path} (overall={best_overall:.4f})")

            ckpt_path = os.path.join(ckpt_dir, f"step_{step}.pt")
            save_checkpoint(
                ckpt_path=ckpt_path,
                model_trainable_state=model.trainable_state_dict(),
                optimizer_state=optim.state_dict(),
                scaler_state=(scaler.state_dict() if use_fp16 else None),
                step=step,
                config=cfg,
                extra=extra,
            )
            if save_latest:
                save_checkpoint(
                    ckpt_path=latest_path,
                    model_trainable_state=model.trainable_state_dict(),
                    optimizer_state=optim.state_dict(),
                    scaler_state=(scaler.state_dict() if use_fp16 else None),
                    step=step,
                    config=cfg,
                    extra=extra,
                )
            rotate_checkpoints(ckpt_dir, keep_last_k)
            print(f"[ckpt] saved: {ckpt_path}")

    # final
    final_path = os.path.join(ckpt_dir, "final.pt")
    final_step = max_steps - 1
    save_checkpoint(
        ckpt_path=final_path,
        model_trainable_state=model.trainable_state_dict(),
        optimizer_state=optim.state_dict(),
        scaler_state=(scaler.state_dict() if use_fp16 else None),
        step=final_step,
        config=cfg,
        extra={"avg_loss": running / (final_step - start_step + 1) if final_step > start_step else running, "best_metric": best_overall},
    )
    print(f"[ckpt] final saved: {final_path}")

    # write summary
    summary_path = os.path.join(run_dir, "summary.json")
    summary = {
        "trainable_params": count_trainable_params(model),
        "enabled_encoders": enabled_encoders,
        "m_queries": m_queries,
        "best_overall": best_overall,
        "time_min": (time.time() - t0) / 60.0,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[summary] saved: {summary_path}")

    dt = time.time() - t0
    print(f"[M1 finetune] done. time={dt/60:.1f} min, steps={max_steps-start_step}")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/finetune.yaml"
    main(cfg_path)
