# scripts/infer_caption_m1.py
# -*- coding: utf-8 -*-
"""
M1 推理样例生成脚本
- 从 config 读取解码参数
- 按数据源均匀采样（与 eval 脚本一致）
- 详细计时（vision_bridge_ms, decode_ms, tokens_per_s）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import random
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
    random.seed(seed)
    
    num_sources = len(concat_ds.datasets)
    samples_per_source = total_samples // num_sources
    
    all_indices = []
    prev_cum = 0
    
    for i, ds in enumerate(concat_ds.datasets):
        global_start = prev_cum
        global_end = concat_ds.cum[i]
        
        ds_indices = list(range(global_start, global_end))
        k = min(samples_per_source, len(ds_indices))
        sampled = random.sample(ds_indices, k)
        all_indices.extend(sampled)
        
        prev_cum = global_end
    
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


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="M1 推理样例生成")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--ckpt", type=str, default=None, help="检查点路径")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--num_samples", type=int, default=50, help="采样数量")
    parser.add_argument("--seed", type=int, default=3407)
    # 以下参数优先使用配置文件中的值，命令行可覆盖
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--early_stopping", action="store_true", default=None)
    args = parser.parse_args()

    # 设置离线模式
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    print(f"[infer] generation params: {gen_params}")
    
    # 输出目录
    out_dir = os.path.join(cfg["output"]["out_dir"], cfg["output"]["run_name"])
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    # 查找检查点
    ckpt_path = args.ckpt or find_checkpoint(ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}. Use --ckpt to specify.")
    print(f"[infer] checkpoint: {ckpt_path}")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] device: {device}")

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
    print(f"[infer] loaded step={step}")

    # 构建数据集（按数据源均匀采样）
    sources = build_sources(cfg["data"]["sources"])
    concat_ds = build_caption_dataset(split=args.split, sources=sources)
    
    # 打印各数据源信息
    print(f"[infer] data sources:")
    prev = 0
    for i, ds in enumerate(concat_ds.datasets):
        ds_name = sources[i].name
        ds_len = concat_ds.cum[i] - prev
        print(f"  [{i+1}] {ds_name}: {ds_len} samples")
        prev = concat_ds.cum[i]
    print(f"  total: {len(concat_ds)} samples")
    
    # 按数据源均匀采样
    eval_seed = cfg.get("eval", {}).get("seed", 42)
    subset = build_balanced_subset(concat_ds, total_samples=args.num_samples, seed=eval_seed)
    print(f"[infer] balanced sampling: {len(subset)} samples ({args.num_samples // len(sources)} per source)")
    
    # 创建 DataLoader
    collate = PretrainCaptionCollator(return_pil=True)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate,
    )

    # 推理
    prompt_prefix = cfg["train"].get("prompt_prefix", "Describe the medical image:\n")
    samples: List[Dict[str, Any]] = []
    
    # 详细计时列表
    vision_bridge_ms_list: List[float] = []
    decode_ms_list: List[float] = []
    tokens_per_s_list: List[float] = []
    num_tokens_list: List[int] = []
    
    print(f"[infer] generating {len(subset)} samples from {args.split}...")
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
        
        # 峰值显存
        peak_mem_gb = None
        if device.type == "cuda":
            peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        
        sample = {
            "sample_idx": i,
            "dataset": meta.get("dataset", "unknown"),
            "image_id": meta.get("sample_id", meta.get("image_path", f"sample_{i}")),
            "image_path": meta.get("image_path", ""),
            "gt_caption": gt_caption,
            "pred_caption": pred_caption,
            "timing": {
                "vision_bridge_ms": round(timing["vision_bridge_ms"], 2),
                "decode_ms": round(timing["decode_ms"], 2),
                "num_tokens": timing["num_tokens"],
                "tokens_per_s": round(timing["tokens_per_s"], 2),
            },
            "peak_mem_gb": round(peak_mem_gb, 3) if peak_mem_gb else None,
        }
        samples.append(sample)
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(subset)}] vb={timing['vision_bridge_ms']:.1f}ms dec={timing['decode_ms']:.1f}ms tok/s={timing['tokens_per_s']:.1f}")

    total_time = time.time() - t0
    
    # 统计
    avg_vision_bridge_ms = sum(vision_bridge_ms_list) / len(vision_bridge_ms_list) if vision_bridge_ms_list else 0
    avg_decode_ms = sum(decode_ms_list) / len(decode_ms_list) if decode_ms_list else 0
    avg_tokens_per_s = sum(tokens_per_s_list) / len(tokens_per_s_list) if tokens_per_s_list else 0
    avg_num_tokens = sum(num_tokens_list) / len(num_tokens_list) if num_tokens_list else 0
    
    # 保存结果
    out_file = os.path.join(out_dir, f"samples_{args.split}_{len(samples)}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    # 打印总结
    print("\n" + "=" * 60)
    print(f"  ECQFormer-M1 Inference Results ({args.split})")
    print("=" * 60)
    print(f"  Samples: {len(samples)}")
    print(f"  Total Time: {total_time:.1f}s")
    print("-" * 60)
    print("  [Timing Statistics]")
    print(f"    Avg Vision+Bridge: {avg_vision_bridge_ms:.2f}ms")
    print(f"    Avg Decode:        {avg_decode_ms:.2f}ms")
    print(f"    Avg Tokens/sec:    {avg_tokens_per_s:.2f}")
    print(f"    Avg Tokens:        {avg_num_tokens:.1f}")
    print("=" * 60)
    print(f"  Output: {out_file}")
    
    # 打印几个样例
    print("\n[infer] Example outputs:")
    for s in samples[:3]:
        print(f"  ID: {s['image_id']}")
        print(f"  GT: {s['gt_caption'][:80]}...")
        print(f"  Pred: {s['pred_caption'][:80]}...")
        print()


if __name__ == "__main__":
    main()