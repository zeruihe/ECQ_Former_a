# scripts/eval_vqa_performance.py
# -*- coding: utf-8 -*-
"""
VQA Performance Evaluation Script
评估微调后的VQA模型的效率指标：
- 视觉编码+Bridge耗时
- 解码耗时
- 总延迟
- Token生成速度
- 峰值显存
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import yaml
from tqdm import tqdm

from data import build_vqa_loader
from models.ecqformer_m1_offline import ECQFormerM1Offline
from utils.vqa_metrics import compute_vqa_metrics


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def find_checkpoint(ckpt_dir: str) -> str:
    """Find best.pt or final.pt in checkpoint directory."""
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    
    best_path = os.path.join(ckpt_dir, "best.pt")
    final_path = os.path.join(ckpt_dir, "final.pt")
    
    if os.path.exists(best_path):
        return best_path
    elif os.path.exists(final_path):
        return final_path
    else:
        # Find latest step checkpoint
        pts = [p for p in os.listdir(ckpt_dir) if p.endswith(".pt")]
        if not pts:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        
        def step_key(name: str) -> int:
            if name.startswith("step_") and name.endswith(".pt"):
                try:
                    return int(name[5:-3])
                except:
                    return -1
            return -1
        
        pts.sort(key=step_key)
        return os.path.join(ckpt_dir, pts[-1])


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> int:
    """Load checkpoint and return step number."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint is not a dict: {type(ckpt)}")
    
    cand_keys = ["model_trainable", "model_trainable_state", "trainable_state", "model_state", "state_dict"]
    state = None
    for k in cand_keys:
        if k in ckpt and isinstance(ckpt[k], dict):
            state = ckpt[k]
            break
    
    if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    
    if state is None:
        raise KeyError(f"Cannot find model state. keys={list(ckpt.keys())}")
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)}")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)}")
    
    return ckpt.get("step", 0)


def count_params(model: torch.nn.Module) -> Dict[str, int]:
    """统计参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


@torch.inference_mode()
def generate_vqa_with_timing(
    model: ECQFormerM1Offline,
    images_pil: List,
    questions: List[str],
    device: torch.device,
    prompt_template: str,
    gen_params: Dict,
) -> tuple:
    """生成VQA回答并记录详细计时。
    
    Returns:
        (predictions, timing_dict)
    """
    torch.cuda.synchronize() if device.type == "cuda" else None
    
    # Step 1: Vision encoding + Bridge (soft prompt)
    t_vision_start = time.perf_counter()
    soft = model.build_soft_prompt(images_pil, device=device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t_vision_end = time.perf_counter()
    vision_bridge_ms = (t_vision_end - t_vision_start) * 1000
    
    # Step 2: Text generation
    prompts = [prompt_template.format(question=q) for q in questions]
    
    t_decode_start = time.perf_counter()
    outputs = model.lm.generate_with_soft_prompt(
        soft_prompt=soft,
        prompts=prompts,
        device=device,
        max_new_tokens=gen_params.get("max_new_tokens", 32),
        num_beams=gen_params.get("num_beams", 1),
        temperature=gen_params.get("temperature", 0.7),
        top_p=gen_params.get("top_p", 0.95),
    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    t_decode_end = time.perf_counter()
    decode_ms = (t_decode_end - t_decode_start) * 1000
    
    # Estimate tokens generated (approximate)
    total_tokens = sum(len(o.split()) for o in outputs)  # word-based estimate
    tokens_per_s = total_tokens / (decode_ms / 1000) if decode_ms > 0 else 0
    
    timing = {
        "vision_bridge_ms": vision_bridge_ms,
        "decode_ms": decode_ms,
        "total_latency_ms": vision_bridge_ms + decode_ms,
        "num_tokens": total_tokens,
        "tokens_per_s": tokens_per_s,
    }
    
    return outputs, timing


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="VQA Performance Evaluation")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--ckpt", type=str, default=None, help="检查点路径（可选）")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max_samples", type=int, default=500, help="最大评估样本数")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (建议1以精确测量延迟)")
    args = parser.parse_args()

    # 离线模式
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # 加载配置
    cfg = load_config(args.config)
    
    # 输出目录
    out_dir = os.path.join(cfg["output"]["out_dir"], cfg["output"]["run_name"])
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    # 查找检查点
    ckpt_path = args.ckpt or find_checkpoint(ckpt_dir)
    print(f"[eval] checkpoint: {ckpt_path}")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device: {device}")

    # 构建模型
    mcfg = cfg["models"]
    amp_dtype = torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16
    
    enabled_encoders = _get_enabled_encoders(mcfg)
    m_queries = _auto_m_queries(mcfg)
    
    print(f"[eval] enabled_encoders: {enabled_encoders}")
    print(f"[eval] m_queries: {m_queries}")
    
    model = ECQFormerM1Offline(
        llama_local_dir=mcfg["llama_dir"],
        clip_local_dir=mcfg["clip_dir"],
        dinov2_local_dir=mcfg["dinov2_dir"],
        biomedclip_local_dir=mcfg["biomedclip_dir"],
        d_bridge=int(mcfg.get("d_bridge", 768)),
        meq_layers=int(mcfg.get("meq_layers", 12)),
        meq_heads=int(mcfg.get("meq_heads", 12)),
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

    step = load_checkpoint(model, ckpt_path)
    model.eval()
    
    # 参数统计
    params = count_params(model)
    print(f"[eval] loaded step={step}")
    print(f"[eval] params: total={params['total']/1e6:.2f}M, trainable={params['trainable']/1e6:.2f}M")

    # 构建数据集
    eval_cfg = cfg.get("eval", {})
    loader = build_vqa_loader(
        dataset_id=str(cfg["data"]["dataset_id"]),
        split=args.split,
        cache_dir=cfg["data"].get("cache_dir", None),
        local_dir=cfg["data"].get("local_dir", None),
        images_dir=cfg["data"].get("images_dir", None),
        language=str(cfg["data"].get("language", "en")),
        batch_size=args.batch_size,
        num_workers=0,  # 单线程以精确测量
        pin_memory=True,
        shuffle=False,
        max_samples=args.max_samples,
    )
    
    print(f"[eval] dataset: {cfg['data']['dataset_id']}, split: {args.split}")
    print(f"[eval] evaluating {min(args.max_samples, len(loader.dataset))} samples...")

    # 生成参数
    prompt_template = str(cfg["train"].get("prompt_template", "Question: {question}\nAnswer:"))
    gen_params = {
        "max_new_tokens": int(eval_cfg.get("max_new_tokens", 10)),
        "num_beams": int(eval_cfg.get("num_beams", 3)),
        "temperature": float(eval_cfg.get("temperature", 0.1)),
        "top_p": float(eval_cfg.get("top_p", 0.9)),
    }
    print(f"[eval] generation params: {gen_params}")

    # 计时列表
    vision_bridge_ms_list = []
    decode_ms_list = []
    total_latency_ms_list = []
    num_tokens_list = []
    tokens_per_s_list = []
    peak_mems_gb = []
    
    preds = []
    golds = []
    closed = []

    # Warmup
    print("[eval] warmup...")
    for batch in loader:
        images_pil = batch["images_pil"]
        questions = batch["questions"]
        _, _ = generate_vqa_with_timing(
            model, images_pil, questions, device, prompt_template, gen_params
        )
        break

    # 评估循环
    print("[eval] running evaluation...")
    t0 = time.time()
    
    for i, batch in enumerate(tqdm(loader, desc=f"[eval {args.split}]")):
        images_pil = batch["images_pil"]
        questions = batch["questions"]
        answers = batch["answers"]
        is_closed = batch["is_closed"]
        
        # 重置显存统计
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        
        # 生成
        outputs, timing = generate_vqa_with_timing(
            model, images_pil, questions, device, prompt_template, gen_params
        )
        
        # 记录计时
        vision_bridge_ms_list.append(timing["vision_bridge_ms"])
        decode_ms_list.append(timing["decode_ms"])
        total_latency_ms_list.append(timing["total_latency_ms"])
        num_tokens_list.append(timing["num_tokens"])
        tokens_per_s_list.append(timing["tokens_per_s"])
        
        # 记录峰值显存
        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            peak_mems_gb.append(peak_mem)
        
        preds.extend(outputs)
        golds.extend(answers)
        closed.extend(is_closed)

    total_time = time.time() - t0
    
    # 计算VQA指标
    print("\n[eval] computing VQA metrics...")
    m = compute_vqa_metrics(preds, golds, closed, use_clean=True)
    
    # 计算效率指标
    n = len(vision_bridge_ms_list)
    avg_vision_bridge_ms = sum(vision_bridge_ms_list) / n if n > 0 else 0
    avg_decode_ms = sum(decode_ms_list) / n if n > 0 else 0
    avg_total_latency_ms = sum(total_latency_ms_list) / n if n > 0 else 0
    avg_num_tokens = sum(num_tokens_list) / n if n > 0 else 0
    avg_tokens_per_s = sum(tokens_per_s_list) / n if n > 0 else 0
    avg_peak_mem_gb = sum(peak_mems_gb) / len(peak_mems_gb) if peak_mems_gb else None
    
    # 吞吐量：每秒处理的样本数
    throughput_samples_per_s = len(preds) / total_time if total_time > 0 else 0

    # 汇总结果
    metrics = {
        "model": "ECQFormer-VQA",
        "checkpoint": ckpt_path,
        "step": step,
        "split": args.split,
        "dataset": cfg["data"]["dataset_id"],
        "enabled_encoders": enabled_encoders,
        "m_queries": m_queries,
        "num_evaluated": len(preds),
        
        # VQA 指标
        "vqa_metrics": {
            "closed_acc": round(m.closed_acc, 4),
            "open_acc": round(m.open_acc, 4),
            "overall_acc": round(m.overall_acc, 4),
            "n_closed": m.n_closed,
            "n_open": m.n_open,
            "n_total": m.n_total,
        },
        
        # 效率指标
        "efficiency": {
            "trainable_params": params["trainable"],
            "trainable_params_M": round(params["trainable"] / 1e6, 2),
            "total_params": params["total"],
            "total_params_M": round(params["total"] / 1e6, 2),
            
            # 延迟指标
            "avg_vision_bridge_ms": round(avg_vision_bridge_ms, 2),
            "avg_decode_ms": round(avg_decode_ms, 2),
            "avg_total_latency_ms": round(avg_total_latency_ms, 2),
            
            # Token 生成速度
            "avg_num_tokens": round(avg_num_tokens, 1),
            "avg_tokens_per_s": round(avg_tokens_per_s, 2),
            
            # 显存
            "avg_peak_mem_gb": round(avg_peak_mem_gb, 3) if avg_peak_mem_gb else None,
            
            # 吞吐量
            "throughput_samples_per_s": round(throughput_samples_per_s, 2),
        },
        
        "total_eval_time_s": round(total_time, 2),
    }

    # 保存结果
    metrics_file = os.path.join(out_dir, f"performance_{args.split}.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 打印结果表格
    print("\n" + "=" * 65)
    print(f"  ECQFormer VQA Performance Evaluation ({args.split})")
    print("=" * 65)
    print(f"  Checkpoint: {os.path.basename(ckpt_path)} (step {step})")
    print(f"  Dataset: {cfg['data']['dataset_id']}")
    print(f"  Encoders: {enabled_encoders}")
    print(f"  Samples: {len(preds)}")
    print("-" * 65)
    print("  [VQA Accuracy]")
    print(f"    Closed Acc:  {m.closed_acc:.4f} (n={m.n_closed})")
    print(f"    Open Acc:    {m.open_acc:.4f} (n={m.n_open})")
    print(f"    Overall Acc: {m.overall_acc:.4f} (n={m.n_total})")
    print("-" * 65)
    print("  [Efficiency Metrics]")
    print(f"    Trainable Params:    {params['trainable']/1e6:.2f}M")
    print(f"    Vision+Bridge Time:  {avg_vision_bridge_ms:.2f}ms")
    print(f"    Decode Time:         {avg_decode_ms:.2f}ms")
    print(f"    Total Latency:       {avg_total_latency_ms:.2f}ms")
    print(f"    Avg Tokens:          {avg_num_tokens:.1f}")
    print(f"    Tokens/sec:          {avg_tokens_per_s:.2f}")
    if avg_peak_mem_gb:
        print(f"    Peak Memory:         {avg_peak_mem_gb:.3f}GB")
    print(f"    Throughput:          {throughput_samples_per_s:.2f} samples/s")
    print("=" * 65)
    print(f"  Results saved to: {metrics_file}")


if __name__ == "__main__":
    main()
