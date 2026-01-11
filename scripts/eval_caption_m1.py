# scripts/eval_caption_m1.py
# -*- coding: utf-8 -*-
"""
1.2 & 1.3 评估脚本
- 基础指标：ROUGE-L、BERTScore
- 效率指标：可训练参数量、单样本推理延迟、峰值显存
- 输出：metrics_{split}.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import yaml

from data import build_caption_loader
from data.schema import DataSourceCfg, AnnSplitCfg
from models.ecqformer_m1_offline import ECQFormerM1Offline


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_sources(cfg_sources: List[Dict]) -> List[DataSourceCfg]:
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


def find_checkpoint(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    pts = [p for p in os.listdir(ckpt_dir) if p.endswith(".pt")]
    if not pts:
        return None
    if "final.pt" in pts:
        return os.path.join(ckpt_dir, "final.pt")
    if "latest.pt" in pts:
        return os.path.join(ckpt_dir, "latest.pt")
    
    def step_key(name: str) -> int:
        if name.startswith("step_") and name.endswith(".pt"):
            try:
                return int(name[len("step_"):-3])
            except:
                return -1
        return -1
    pts.sort(key=step_key)
    return os.path.join(ckpt_dir, pts[-1])


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> int:
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


def compute_rouge_l(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """计算 ROUGE-L"""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("[warn] rouge-score not installed, skipping ROUGE-L")
        return {"precision": None, "recall": None, "f1": None}
    
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = {"precision": [], "recall": [], "f1": []}
    
    for pred, ref in zip(preds, refs):
        s = scorer.score(ref, pred)["rougeL"]
        scores["precision"].append(s.precision)
        scores["recall"].append(s.recall)
        scores["f1"].append(s.fmeasure)
    
    return {
        "precision": sum(scores["precision"]) / len(scores["precision"]),
        "recall": sum(scores["recall"]) / len(scores["recall"]),
        "f1": sum(scores["f1"]) / len(scores["f1"]),
    }


def compute_bertscore(
    preds: List[str], 
    refs: List[str], 
    model_path: Optional[str],
    device: str
) -> Dict[str, Optional[float]]:
    """计算 BERTScore"""
    if model_path is None:
        return {"precision": None, "recall": None, "f1": None}
    
    try:
        from bert_score import score as bert_score
    except ImportError:
        print("[warn] bert-score not installed, skipping BERTScore")
        return {"precision": None, "recall": None, "f1": None}
    
    P, R, F1 = bert_score(
        cands=preds,
        refs=refs,
        model_type=model_path,
        device=device,
        lang="en",
        rescale_with_baseline=True,
        verbose=False,
    )
    
    return {
        "precision": float(P.mean().item()),
        "recall": float(R.mean().item()),
        "f1": float(F1.mean().item()),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="M1 评估脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--ckpt", type=str, default=None, help="检查点路径")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--max_eval", type=int, default=500, help="最大评估样本数")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--bertscore_model", type=str, default=None, 
                        help="BERTScore 模型路径（如 /path/to/roberta-large）")
    parser.add_argument("--save_preds", action="store_true", help="保存预测结果")
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
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    print(f"[eval] checkpoint: {ckpt_path}")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device: {device}")

    # 构建模型
    mcfg = cfg["models"]
    amp_dtype = torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16
    
    model = ECQFormerM1Offline(
        llama_local_dir=mcfg["llama_dir"],
        clip_local_dir=mcfg["clip_dir"],
        dinov2_local_dir=mcfg["dinov2_dir"],
        biomedclip_local_dir=mcfg["biomedclip_dir"],
        d_bridge=int(mcfg.get("d_bridge", 768)),
        meq_layers=int(mcfg.get("meq_layers", 2)),
        meq_heads=int(mcfg.get("meq_heads", 12)),
        m_queries=int(mcfg.get("m_queries", 96)),
        torch_dtype=amp_dtype,
    ).to(device)

    step = load_checkpoint(model, ckpt_path)
    model.eval()
    
    # 参数统计
    params = count_params(model)
    print(f"[eval] loaded step={step}")
    print(f"[eval] params: total={params['total']/1e6:.2f}M, trainable={params['trainable']/1e6:.2f}M, frozen={params['frozen']/1e6:.2f}M")

    # 构建数据加载器
    sources = build_sources(cfg["data"]["sources"])
    loader = build_caption_loader(
        split=args.split,
        sources=sources,
        batch_size=1,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        return_pil=True,
        pin_memory=True,
        shuffle=False,  # 评估时不打乱
    )

    # 推理
    prompt_prefix = cfg["train"].get("prompt_prefix", "Describe the medical image:\n")
    
    preds: List[str] = []
    refs: List[str] = []
    metas: List[Dict] = []
    latencies_ms: List[float] = []
    peak_mems_gb: List[float] = []
    
    print(f"[eval] evaluating up to {args.max_eval} samples from {args.split}...")
    t0 = time.time()
    
    for i, batch in enumerate(loader):
        if i >= args.max_eval:
            break
        
        images_pil = batch["images_pil"]
        gt_caption = batch["captions"][0]
        meta = batch["meta"][0]
        
        # 重置显存统计
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        
        # 生成
        t_gen = time.time()
        pred_caption = model.generate_caption(
            images_pil=images_pil,
            device=device,
            prompt_prefix=prompt_prefix,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_p=args.top_p,
        )[0]
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        latency_ms = (time.time() - t_gen) * 1000
        latencies_ms.append(latency_ms)
        
        # 记录峰值显存
        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            peak_mems_gb.append(peak_mem)
        
        preds.append(pred_caption)
        refs.append(gt_caption)
        metas.append(meta)
        
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{min(args.max_eval, len(loader))}] lat={latency_ms:.1f}ms")

    total_time = time.time() - t0
    
    # 计算指标
    print("\n[eval] computing metrics...")
    
    # ROUGE-L
    rouge_scores = compute_rouge_l(preds, refs)
    
    # BERTScore
    bertscore_scores = compute_bertscore(
        preds, refs, 
        model_path=args.bertscore_model,
        device="cuda" if device.type == "cuda" else "cpu"
    )
    
    # 效率指标
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0
    avg_peak_mem_gb = sum(peak_mems_gb) / len(peak_mems_gb) if peak_mems_gb else None
    
    # 汇总结果
    metrics = {
        "model": "ECQFormer-M1",
        "checkpoint": ckpt_path,
        "step": step,
        "split": args.split,
        "num_evaluated": len(preds),
        
        # 基础指标
        "rouge_l": rouge_scores,
        "bertscore": bertscore_scores,
        
        # 效率指标
        "efficiency": {
            "trainable_params": params["trainable"],
            "trainable_params_M": round(params["trainable"] / 1e6, 2),
            "total_params": params["total"],
            "total_params_M": round(params["total"] / 1e6, 2),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "avg_peak_mem_gb": round(avg_peak_mem_gb, 3) if avg_peak_mem_gb else None,
        },
        
        "total_eval_time_s": round(total_time, 2),
    }
    
    # 保存结果
    metrics_file = os.path.join(out_dir, f"metrics_{args.split}.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # 打印结果表格
    print("\n" + "=" * 60)
    print(f"  ECQFormer-M1 Evaluation Results ({args.split})")
    print("=" * 60)
    print(f"  Checkpoint: {os.path.basename(ckpt_path)} (step {step})")
    print(f"  Samples: {len(preds)}")
    print("-" * 60)
    print("  [Quality Metrics]")
    print(f"    ROUGE-L (F1):    {rouge_scores['f1']:.4f}" if rouge_scores['f1'] else "    ROUGE-L: N/A")
    print(f"    BERTScore (F1):  {bertscore_scores['f1']:.4f}" if bertscore_scores['f1'] else "    BERTScore: N/A")
    print("-" * 60)
    print("  [Efficiency Metrics]")
    print(f"    Trainable Params: {params['trainable']/1e6:.2f}M")
    print(f"    Avg Latency:      {avg_latency_ms:.2f}ms")
    if avg_peak_mem_gb:
        print(f"    Peak Memory:      {avg_peak_mem_gb:.3f}GB")
    print("=" * 60)
    print(f"  Results saved to: {metrics_file}")
    
    # 保存预测结果
    if args.save_preds:
        preds_file = os.path.join(out_dir, f"preds_{args.split}.jsonl")
        with open(preds_file, "w", encoding="utf-8") as f:
            for pred, ref, meta in zip(preds, refs, metas):
                f.write(json.dumps({
                    "meta": meta,
                    "gt": ref,
                    "pred": pred,
                }, ensure_ascii=False) + "\n")
        print(f"  Predictions saved to: {preds_file}")


if __name__ == "__main__":
    main()
