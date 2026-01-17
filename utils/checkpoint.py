# utils/checkpoint.py
from __future__ import annotations
import os
import glob
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

try:
    import numpy as np
except Exception:
    np = None


@dataclass
class CheckpointState:
    step: int
    best_metric: Optional[float] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if np is not None:
        np.random.seed(seed)


def get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if np is not None:
        state["numpy"] = np.random.get_state()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if np is not None and "numpy" in state:
        np.random.set_state(state["numpy"])


def save_checkpoint(
    *,
    ckpt_path: str,
    model_trainable_state: Dict[str, torch.Tensor],
    optimizer_state: Dict[str, Any],
    scaler_state: Optional[Dict[str, Any]],
    step: int,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    payload = {
        "step": step,
        "model_trainable": model_trainable_state,
        "optimizer": optimizer_state,
        "scaler": scaler_state,
        "rng": get_rng_state(),
        "config": config,
        "extra": extra or {},
    }
    torch.save(payload, ckpt_path)


def load_checkpoint(
    *,
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    map_location: str = "cpu",
) -> CheckpointState:
    payload = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    # trainable-only load
    missing, unexpected = model.load_trainable_state_dict(payload["model_trainable"], strict=False)
    if missing:
        print(f"[resume] missing keys (trainable): {len(missing)}")
    if unexpected:
        print(f"[resume] unexpected keys (trainable): {len(unexpected)}")

    if optimizer is not None and "optimizer" in payload and payload["optimizer"] is not None:
        optimizer.load_state_dict(payload["optimizer"])

    if scaler is not None and "scaler" in payload and payload["scaler"] is not None:
        try:
            scaler.load_state_dict(payload["scaler"])
        except Exception as e:
            print(f"[resume] scaler load skipped: {e}")

    if "rng" in payload and payload["rng"] is not None:
        set_rng_state(payload["rng"])

    step = int(payload.get("step", 0))
    best_metric = payload.get("extra", {}).get("best_metric", None)
    return CheckpointState(step=step, best_metric=best_metric)


def rotate_checkpoints(ckpt_dir: str, keep_last_k: int) -> None:
    if keep_last_k <= 0:
        return
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    if len(paths) <= keep_last_k:
        return
    to_delete = paths[: len(paths) - keep_last_k]
    for p in to_delete:
        try:
            os.remove(p)
        except OSError:
            pass
