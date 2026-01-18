# train_m1_pretrain_caption_offline.py
from __future__ import annotations
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import yaml
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from data import build_caption_loader
from data.schema import DataSourceCfg, AnnSplitCfg

from models.ecqformer_m1_offline import ECQFormerM1Offline
from utils.checkpoint import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    rotate_checkpoints,
)

def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_sources(cfg_sources):
    sources = []
    for s in cfg_sources:
        sources.append(
            DataSourceCfg(
                name=s["name"],
                root=s["root"],
                ann=AnnSplitCfg(
                    train=s["ann"]["train"],
                    valid=s["ann"]["valid"],
                    test=s["ann"]["test"],
                ),
                image_key=s["image_key"],
                text_key=s["text_key"],
                id_key=s.get("id_key", None),
            )
        )
    return sources

def _get_enabled_encoders(cfg_models):
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


@torch.inference_mode()
def evaluate_caption(
    *,
    model: ECQFormerM1Offline,
    loader,
    device: torch.device,
    prompt_prefix: str,
    max_length: int,
    amp_dtype: torch.dtype,
    max_batches: int = 100,
) -> dict:
    """Evaluate caption loss on validation set.
    
    Returns dict with avg_val_loss.
    Uses a subset of batches for efficiency.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    for batch in loader:
        if n_batches >= max_batches:
            break
            
        images_pil = batch["images_pil"]
        captions = batch["captions"]
        
        with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
            loss = model.forward_caption_pretrain(
                images_pil=images_pil,
                captions=captions,
                device=device,
                prompt_prefix=prompt_prefix,
                max_length=max_length,
            )
        
        total_loss += loss.item()
        n_batches += 1
    
    model.train()
    
    avg_loss = total_loss / n_batches if n_batches > 0 else float('inf')
    return {"val_loss": avg_loss, "n_batches": n_batches}


def main(config_path: str = "config/m1.yaml"):
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
    sources = build_sources(cfg["data"]["sources"])
    loader = build_caption_loader(
        split=cfg["data"]["split"],
        sources=sources,
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        return_pil=bool(cfg["data"]["return_pil"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        shuffle=bool(cfg["data"]["shuffle"]),
    )

    # Validation loader (optional)
    eval_cfg = cfg.get("eval", {})
    do_eval = bool(eval_cfg.get("enabled", False))
    val_loader = None
    if do_eval:
        val_loader = build_caption_loader(
            split="valid",  # Use validation split
            sources=sources,
            batch_size=int(cfg["data"]["batch_size"]),
            num_workers=int(cfg["data"]["num_workers"]),
            return_pil=bool(cfg["data"]["return_pil"]),
            pin_memory=bool(cfg["data"]["pin_memory"]),
            shuffle=False,  # Don't shuffle validation
        )
        print(f"[eval] validation loader ready, samples={len(val_loader.dataset)}")

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

    print(f"[M1] trainable params = {count_trainable_params(model)/1e6:.2f}M")

    optim = AdamW([p for p in model.parameters() if p.requires_grad],
                  lr=float(cfg["train"]["lr"]),
                  weight_decay=float(cfg["train"]["weight_decay"]))

    # scaler only for fp16
    scaler = GradScaler(enabled=use_fp16)

    # resume logic
    start_step = 0
    resume_from = cfg["train"].get("resume_from", None)
    auto_resume = bool(cfg["train"].get("auto_resume", True))
    latest_path = os.path.join(ckpt_dir, "latest.pt")

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

    # train settings
    model.train()
    grad_accum = int(cfg["train"]["grad_accum"])
    
    # Compute max_steps from num_epochs or use max_steps directly
    dataset_size = len(loader.dataset)
    batch_size = int(cfg["data"]["batch_size"])
    max_steps = _compute_max_steps(cfg["train"], dataset_size, batch_size)
    max_length = int(cfg["train"]["max_length"])
    prompt_prefix = str(cfg["train"]["prompt_prefix"])
    log_every = int(cfg["train"]["log_every"])
    save_every = int(cfg["train"]["save_every"])
    keep_last_k = int(cfg["train"].get("keep_last_k", 3))
    save_latest = bool(cfg["train"].get("save_latest", True))

    # Validation settings
    max_val_batches = int(eval_cfg.get("max_batches", 100))

    # fast-forward dataloader if resuming (simple approach)
    # Note: for large start_step, you may want a sampler with set_epoch / stateful dataloader;
    # for M1 short runs this is acceptable.
    it = iter(loader)
    for _ in range(start_step):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)

    running = 0.0
    t0 = time.time()
    best_val_loss = float('inf')  # Track best validation loss

    optim.zero_grad(set_to_none=True)

    pbar = tqdm(range(start_step, max_steps), total=max_steps - start_step)
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        images_pil = batch["images_pil"]
        captions = batch["captions"]

        with autocast(enabled=True, dtype=amp_dtype):
            loss = model.forward_caption_pretrain(
                images_pil=images_pil,
                captions=captions,
                device=device,
                prompt_prefix=prompt_prefix,
                max_length=max_length,
            )
            loss = loss / grad_accum

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

        # save checkpoint + (optional) validation
        if save_every > 0 and step > 0 and (step % save_every == 0):
            extra = {"avg_loss": running / (step - start_step + 1)}
            
            # Validation evaluation
            if do_eval and val_loader is not None:
                val_metrics = evaluate_caption(
                    model=model,
                    loader=val_loader,
                    device=device,
                    prompt_prefix=prompt_prefix,
                    max_length=max_length,
                    amp_dtype=amp_dtype,
                    max_batches=max_val_batches,
                )
                extra.update({"val": val_metrics})
                print(f"[eval] val_loss={val_metrics['val_loss']:.4f}")
                
                # Save best checkpoint based on validation loss
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    best_path = os.path.join(ckpt_dir, "best.pt")
                    save_checkpoint(
                        ckpt_path=best_path,
                        model_trainable_state=model.trainable_state_dict(),
                        optimizer_state=optim.state_dict(),
                        scaler_state=(scaler.state_dict() if use_fp16 else None),
                        step=step,
                        config=cfg,
                        extra={"best_val_loss": best_val_loss, **extra},
                    )
                    print(f"[ckpt] best saved: {best_path} (val_loss={best_val_loss:.4f})")
            
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

    # 训练结束后保存 final.pt
    final_path = os.path.join(ckpt_dir, "final.pt")
    final_step = max_steps - 1
    save_checkpoint(
        ckpt_path=final_path,
        model_trainable_state=model.trainable_state_dict(),
        optimizer_state=optim.state_dict(),
        scaler_state=(scaler.state_dict() if use_fp16 else None),
        step=final_step,
        config=cfg,
        extra={"avg_loss": running / (final_step - start_step + 1) if final_step > start_step else running},
    )
    print(f"[ckpt] final saved: {final_path}")

    dt = time.time() - t0
    print(f"[M1] done. time={dt/60:.1f} min, steps={max_steps-start_step}")

if __name__ == "__main__":
    # 允许：python train_m1_pretrain_caption_offline.py configs/m1.yaml
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/m1.yaml"
    main(cfg_path)
