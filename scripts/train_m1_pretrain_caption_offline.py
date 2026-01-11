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

def main(config_path: str = "configs/m1.yaml"):
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

    # model
    use_bf16 = bool(cfg["train"].get("bf16", True))
    use_fp16 = bool(cfg["train"].get("fp16", False))
    if use_bf16 and use_fp16:
        raise ValueError("bf16 and fp16 cannot both be true.")
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model = ECQFormerM1Offline(
        llama_local_dir=cfg["models"]["llama_dir"],
        clip_local_dir=cfg["models"]["clip_dir"],
        dinov2_local_dir=cfg["models"]["dinov2_dir"],
        biomedclip_local_dir=cfg["models"]["biomedclip_dir"],
        d_bridge=int(cfg["models"]["d_bridge"]),
        meq_layers=int(cfg["models"]["meq_layers"]),
        meq_heads=int(cfg["models"]["meq_heads"]),
        m_queries=int(cfg["models"]["m_queries"]),
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
    max_steps = int(cfg["train"]["max_steps"])
    max_length = int(cfg["train"]["max_length"])
    prompt_prefix = str(cfg["train"]["prompt_prefix"])
    log_every = int(cfg["train"]["log_every"])
    save_every = int(cfg["train"]["save_every"])
    keep_last_k = int(cfg["train"].get("keep_last_k", 3))
    save_latest = bool(cfg["train"].get("save_latest", True))

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

        # save checkpoint
        if save_every > 0 and step > 0 and (step % save_every == 0):
            ckpt_path = os.path.join(ckpt_dir, f"step_{step}.pt")
            save_checkpoint(
                ckpt_path=ckpt_path,
                model_trainable_state=model.trainable_state_dict(),
                optimizer_state=optim.state_dict(),
                scaler_state=(scaler.state_dict() if use_fp16 else None),
                step=step,
                config=cfg,
                extra={"avg_loss": running / (step - start_step + 1)},
            )
            if save_latest:
                save_checkpoint(
                    ckpt_path=latest_path,
                    model_trainable_state=model.trainable_state_dict(),
                    optimizer_state=optim.state_dict(),
                    scaler_state=(scaler.state_dict() if use_fp16 else None),
                    step=step,
                    config=cfg,
                    extra={"avg_loss": running / (step - start_step + 1)},
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
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/m1.yaml"
    main(cfg_path)
