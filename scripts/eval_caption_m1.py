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

from data import build_caption_dataset
from data.schema import DataSourceCfg, AnnSplitCfg
from data.collate import PretrainCaptionCollator
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


def build_balanced_subset(
    concat_ds,
    total_samples: int,
    seed: int = 42,
) -> torch.utils.data.Subset:
    """
    从 ConcatCaptionDataset 中按数据源均匀采样。
    每个数据源抽取 total_samples // num_sources 条样本。
    """
    import random
    random.seed(seed)
    
    # 计算每个子数据集的区间
    num_sources = len(concat_ds.datasets)
    samples_per_source = total_samples // num_sources
    
    all_indices = []
    prev_cum = 0
    
    for i, ds in enumerate(concat_ds.datasets):
        ds_len = len(ds)
        # 全局索引区间: [prev_cum, cum[i])
        global_start = prev_cum
        global_end = concat_ds.cum[i]
        
        # 从该数据源随机抽取
        ds_indices = list(range(global_start, global_end))
        k = min(samples_per_source, len(ds_indices))
        sampled = random.sample(ds_indices, k)
        all_indices.extend(sampled)
        
        prev_cum = global_end
    
    # 打乱最终顺序以避免数据源顺序偏差
    random.shuffle(all_indices)
    
    return torch.utils.data.Subset(concat_ds, all_indices)


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
    
    # 对于本地路径，需要额外指定 num_layers
    # roberta-large: 17 layers, roberta-base: 10 layers
    is_local_path = os.path.isdir(model_path)
    
    try:
        if is_local_path:
            # 本地路径需要指定 num_layers
            # 检测是否是 roberta-large
            if "roberta-large" in model_path.lower():
                num_layers = 17
            elif "roberta-base" in model_path.lower():
                num_layers = 10
            else:
                num_layers = 17  # 默认使用 large 的层数
            
            P, R, F1 = bert_score(
                cands=preds,
                refs=refs,
                model_type=model_path,
                num_layers=num_layers,
                device=device,
                lang="en",
                rescale_with_baseline=False,  # 本地模式不使用基线（避免网络请求）
                verbose=False,
            )
        else:
            # 在线模型名称
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
    except Exception as e:
        print(f"[warn] BERTScore failed: {e}")
        return {"precision": None, "recall": None, "f1": None}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="M1 评估脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--ckpt", type=str, default=None, help="检查点路径")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--max_eval", type=int, default=500, help="最大评估样本数")
    # 以下参数优先使用配置文件中的值，命令行可覆盖
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--early_stopping", action="store_true", default=None)
    parser.add_argument("--bertscore_model", type=str, default=None, 
                        help="BERTScore 模型路径（如 /path/to/roberta-large）")
    parser.add_argument("--save_preds", action="store_true", help="保存预测结果")
    args = parser.parse_args()

    # 离线模式
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # 加载配置
    cfg = load_config(args.config)
    
    # 读取 eval 配置（命令行参数覆盖配置文件）
    eval_cfg = cfg.get("eval", {})
    gen_params = {
        "max_new_tokens": args.max_new_tokens if args.max_new_tokens is not None else eval_cfg.get("max_new_tokens", 32),
        "num_beams": args.num_beams if args.num_beams is not None else eval_cfg.get("num_beams", 1),
        "temperature": args.temperature if args.temperature is not None else eval_cfg.get("temperature", 0.7),
        "top_p": args.top_p if args.top_p is not None else eval_cfg.get("top_p", 0.95),
        "no_repeat_ngram_size": args.no_repeat_ngram_size if args.no_repeat_ngram_size is not None else eval_cfg.get("no_repeat_ngram_size", 3),
        "repetition_penalty": args.repetition_penalty if args.repetition_penalty is not None else eval_cfg.get("repetition_penalty", 1.15),
        "early_stopping": args.early_stopping if args.early_stopping is not None else eval_cfg.get("early_stopping", True),
    }
    print(f"[eval] generation params: {gen_params}")
    
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

    # 构建数据集（按数据源均匀采样）
    sources = build_sources(cfg["data"]["sources"])
    concat_ds = build_caption_dataset(split=args.split, sources=sources)
    
    # 打印各数据源信息
    print(f"[eval] data sources:")
    prev = 0
    for i, ds in enumerate(concat_ds.datasets):
        ds_name = sources[i].name
        ds_len = concat_ds.cum[i] - prev
        print(f"  [{i+1}] {ds_name}: {ds_len} samples")
        prev = concat_ds.cum[i]
    print(f"  total: {len(concat_ds)} samples")
    
    # 按数据源均匀采样
    eval_seed = cfg.get("eval", {}).get("seed", 42)
    subset = build_balanced_subset(concat_ds, total_samples=args.max_eval, seed=eval_seed)
    print(f"[eval] balanced sampling: {len(subset)} samples ({args.max_eval // len(sources)} per source)")
    
    # 创建 DataLoader
    collate = PretrainCaptionCollator(return_pil=True)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=1,
        shuffle=False,  # Subset 已打乱
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate,
    )

    # 推理
    prompt_prefix = cfg["train"].get("prompt_prefix", "Describe the medical image:\n")
    
    preds: List[str] = []
    refs: List[str] = []
    metas: List[Dict] = []
    
    # 详细计时列表
    vision_bridge_ms_list: List[float] = []
    decode_ms_list: List[float] = []
    tokens_per_s_list: List[float] = []
    num_tokens_list: List[int] = []
    peak_mems_gb: List[float] = []
    
    print(f"[eval] evaluating {len(subset)} samples from {args.split}...")
    t0 = time.time()
    
    for i, batch in enumerate(loader):
        
        images_pil = batch["images_pil"]
        gt_caption = batch["captions"][0]
        meta = batch["meta"][0]
        
        # 重置显存统计
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        
        # 生成（带详细计时）
        pred_list, timing = model.generate_caption_with_timing(
            images_pil=images_pil,
            device=device,
            prompt_prefix=prompt_prefix,
            max_new_tokens=gen_params["max_new_tokens"],
            num_beams=gen_params["num_beams"],
            temperature=gen_params["temperature"],
            top_p=gen_params["top_p"],
            no_repeat_ngram_size=gen_params["no_repeat_ngram_size"],
            repetition_penalty=gen_params["repetition_penalty"],
            early_stopping=gen_params["early_stopping"],
        )
        pred_caption = pred_list[0]
        
        # 记录计时
        vision_bridge_ms_list.append(timing["vision_bridge_ms"])
        decode_ms_list.append(timing["decode_ms"])
        tokens_per_s_list.append(timing["tokens_per_s"])
        num_tokens_list.append(timing["num_tokens"])
        
        # 记录峰值显存
        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            peak_mems_gb.append(peak_mem)
        
        preds.append(pred_caption)
        refs.append(gt_caption)
        metas.append(meta)
        
        if (i + 1) % 50 == 0:
            total_ms = timing["vision_bridge_ms"] + timing["decode_ms"]
            print(f"  [{i+1}/{min(args.max_eval, len(loader))}] "
                  f"vb={timing['vision_bridge_ms']:.1f}ms dec={timing['decode_ms']:.1f}ms "
                  f"tok/s={timing['tokens_per_s']:.1f}")

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
    
    # 效率指标 - 详细计时
    avg_vision_bridge_ms = sum(vision_bridge_ms_list) / len(vision_bridge_ms_list) if vision_bridge_ms_list else 0
    avg_decode_ms = sum(decode_ms_list) / len(decode_ms_list) if decode_ms_list else 0
    avg_tokens_per_s = sum(tokens_per_s_list) / len(tokens_per_s_list) if tokens_per_s_list else 0
    avg_num_tokens = sum(num_tokens_list) / len(num_tokens_list) if num_tokens_list else 0
    avg_total_latency_ms = avg_vision_bridge_ms + avg_decode_ms
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
        
        # 效率指标 - 详细计时
        "efficiency": {
            "trainable_params": params["trainable"],
            "trainable_params_M": round(params["trainable"] / 1e6, 2),
            "total_params": params["total"],
            "total_params_M": round(params["total"] / 1e6, 2),
            
            # 拆分计时
            "avg_vision_bridge_ms": round(avg_vision_bridge_ms, 2),
            "avg_decode_ms": round(avg_decode_ms, 2),
            "avg_total_latency_ms": round(avg_total_latency_ms, 2),
            
            # token 生成速度
            "avg_num_tokens": round(avg_num_tokens, 1),
            "avg_tokens_per_s": round(avg_tokens_per_s, 2),
            
            "avg_peak_mem_gb": round(avg_peak_mem_gb, 3) if avg_peak_mem_gb else None,
        },
        
        "total_eval_time_s": round(total_time, 2),
    }
    
    # 保存结果
    metrics_file = os.path.join(out_dir, f"metrics_{args.split}.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # 打印结果表格
    print("\n" + "=" * 65)
    print(f"  ECQFormer-M1 Evaluation Results ({args.split})")
    print("=" * 65)
    print(f"  Checkpoint: {os.path.basename(ckpt_path)} (step {step})")
    print(f"  Samples: {len(preds)}")
    print("-" * 65)
    print("  [Quality Metrics]")
    print(f"    ROUGE-L (F1):    {rouge_scores['f1']:.4f}" if rouge_scores['f1'] else "    ROUGE-L: N/A")
    print(f"    BERTScore (F1):  {bertscore_scores['f1']:.4f}" if bertscore_scores['f1'] else "    BERTScore: N/A")
    print("-" * 65)
    print("  [Efficiency Metrics]")
    print(f"    Trainable Params:    {params['trainable']/1e6:.2f}M")
    print(f"    Vision+Bridge Time:  {avg_vision_bridge_ms:.2f}ms")
    print(f"    Decode Time:         {avg_decode_ms:.2f}ms")
    print(f"    Total Latency:       {avg_total_latency_ms:.2f}ms")
    print(f"    Tokens/sec:          {avg_tokens_per_s:.2f}")
    print(f"    Avg Tokens:          {avg_num_tokens:.1f}")
    if avg_peak_mem_gb:
        print(f"    Peak Memory:         {avg_peak_mem_gb:.3f}GB")
    print("=" * 65)
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
