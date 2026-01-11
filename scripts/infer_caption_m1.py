# scripts/infer_caption_m1.py
# -*- coding: utf-8 -*-
"""
1.1 生成样例脚本
- 载入 final.pt
- 从 rocov2/mimic-cxr val/test 抽样 N 条
- 输出：image_id、GT caption、生成 caption -> outputs/.../samples.jsonl
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


def find_checkpoint(ckpt_dir: str, prefer: str = "final") -> Optional[str]:
    """查找检查点，优先 final.pt"""
    if not os.path.isdir(ckpt_dir):
        return None
    pts = [p for p in os.listdir(ckpt_dir) if p.endswith(".pt")]
    if not pts:
        return None
    
    # 优先级: final.pt > latest.pt > 最大 step
    if "final.pt" in pts:
        return os.path.join(ckpt_dir, "final.pt")
    if "latest.pt" in pts:
        return os.path.join(ckpt_dir, "latest.pt")
    
    # 按 step 排序
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
    """加载检查点，返回训练步数"""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint is not a dict: {type(ckpt)}")
    
    # 查找模型状态
    cand_keys = ["model_trainable", "model_trainable_state", "trainable_state", "model_state", "state_dict"]
    state = None
    for k in cand_keys:
        if k in ckpt and isinstance(ckpt[k], dict):
            state = ckpt[k]
            break
    
    if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    
    if state is None:
        raise KeyError(f"Cannot find model state in checkpoint. keys={list(ckpt.keys())}")
    
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
    parser.add_argument("--ckpt", type=str, default=None, help="检查点路径（默认自动查找 final.pt）")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--num_samples", type=int, default=50, help="采样数量")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    # 设置离线模式
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载配置
    cfg = load_config(args.config)
    
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

    # 构建数据加载器
    sources = build_sources(cfg["data"]["sources"])
    loader = build_caption_loader(
        split=args.split,
        sources=sources,
        batch_size=1,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        return_pil=True,
        pin_memory=True,
        shuffle=True,  # 随机采样
    )

    # 推理
    prompt_prefix = cfg["train"].get("prompt_prefix", "Describe the medical image:\n")
    samples: List[Dict[str, Any]] = []
    
    print(f"[infer] sampling {args.num_samples} from {args.split}...")
    t0 = time.time()
    
    for i, batch in enumerate(loader):
        if i >= args.num_samples:
            break
        
        images_pil = batch["images_pil"]
        gt_caption = batch["captions"][0]
        meta = batch["meta"][0]
        
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
        gen_time_ms = (time.time() - t_gen) * 1000
        
        sample = {
            "sample_idx": i,
            "dataset": meta.get("dataset", "unknown"),
            "image_id": meta.get("sample_id", meta.get("image_path", f"sample_{i}")),
            "image_path": meta.get("image_path", ""),
            "gt_caption": gt_caption,
            "pred_caption": pred_caption,
            "gen_time_ms": round(gen_time_ms, 2),
        }
        samples.append(sample)
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.num_samples}] {sample['image_id'][:30]}...")

    total_time = time.time() - t0
    
    # 保存结果
    out_file = os.path.join(out_dir, f"samples_{args.split}_{args.num_samples}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"\n[infer] Done!")
    print(f"  samples: {len(samples)}")
    print(f"  total time: {total_time:.1f}s")
    print(f"  avg time per sample: {total_time/len(samples)*1000:.1f}ms")
    print(f"  output: {out_file}")
    
    # 打印几个样例
    print("\n[infer] Example outputs:")
    for s in samples[:3]:
        print(f"  ID: {s['image_id']}")
        print(f"  GT: {s['gt_caption'][:100]}...")
        print(f"  Pred: {s['pred_caption'][:100]}...")
        print()


if __name__ == "__main__":
    main()