# scripts/eval_m2_vqa_performance.py
# -*- coding: utf-8 -*-
"""
M2 VQA 效率性能评估脚本（第二个创新点）

在 eval_vqa_performance.py（M1 版）基础上适配 ECQFormerM2Offline：
  - build_soft_prompt 需传入 questions（驱动门控和 FiLM）
  - 计时分解：vision_bridge 阶段现在包含文本编码 + 门控 + FiLM
  - 额外记录每 batch 三路编码器门控均值（可分析文本引导效果）

输出：{out_dir}/{run_name}/performance_{split}.json

用法：
  python scripts/eval_m2_vqa_performance.py \\
    --config config/m3.yaml \\
    --split test \\
    --max_samples 500 \\
    --batch_size 1

  # 手动指定 checkpoint
  python scripts/eval_m2_vqa_performance.py \\
    --config config/m3.yaml \\
    --ckpt /root/autodl-tmp/outputs/m3_finetune_vqa-rad/checkpoints/best.pt \\
    --split test
"""
from __future__ import annotations

import argparse, json, os, sys, time
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import yaml
from tqdm import tqdm

from data import build_vqa_loader
from models.ecqformer_m2_offline import ECQFormerM2Offline
from utils.vqa_metrics import compute_vqa_metrics


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

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
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")
    for name in ["best.pt", "final.pt"]:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
    pts = sorted(
        [p for p in os.listdir(ckpt_dir) if p.endswith(".pt")],
        key=lambda n: int(n[5:-3]) if n.startswith("step_") else -1,
    )
    if pts:
        return os.path.join(ckpt_dir, pts[-1])
    raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")


def load_checkpoint_weights(model: torch.nn.Module, ckpt_path: str) -> int:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint is not a dict: {type(ckpt)}")
    for key in ["model_trainable", "model_trainable_state", "trainable_state", "model_state", "state_dict"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    else:
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
        else:
            raise KeyError(f"Cannot find model state. keys={list(ckpt.keys())}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)}")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)}")
    return ckpt.get("step", 0)


def count_params(model: torch.nn.Module) -> Dict[str, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ─────────────────────────────────────────────────────────────
# 单 batch 推理 + 计时（M2 版）
# ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_m2_with_timing(
    model: ECQFormerM2Offline,
    images_pil: List,
    questions: List[str],
    device: torch.device,
    prompt_template: str,
    gen_params: Dict,
) -> tuple:
    """
    M2 单 batch 推理 + 精确计时。

    计时分解：
      vision_bridge_ms  — 文本编码 + 门控 + 视觉编码 + MEQFormerV2 + soft_proj
      decode_ms         — LLaMA 生成阶段

    额外返回 gate_weights (B, K) 用于分析门控行为。
    """
    sync = lambda: torch.cuda.synchronize() if device.type == "cuda" else None

    # ── Stage 1：Vision + Bridge（含 M2 文本编码 + 门控 + FiLM） ──
    sync()
    t0 = time.perf_counter()
    soft, debug = model.build_soft_prompt(
        images_pil, device=device, questions=questions
    )
    sync()
    vision_bridge_ms = (time.perf_counter() - t0) * 1000

    # 提取门控权重（用于分析，不计入计时）
    gate_weights = None
    if model.gating is not None:
        gate_weights = model.get_gate_weights(questions, device).float().cpu()  # (B, K)

    # ── Stage 2：LLaMA 生成 ────────────────────────────────────
    prompts = [prompt_template.format(question=q) for q in questions]
    sync()
    t1 = time.perf_counter()
    outputs = model.lm.generate_with_soft_prompt(
        soft_prompt    = soft,
        prompts        = prompts,
        device         = device,
        max_new_tokens = gen_params.get("max_new_tokens", 10),
        num_beams      = gen_params.get("num_beams", 3),
        temperature    = gen_params.get("temperature", 0.1),
        top_p          = gen_params.get("top_p", 0.9),
    )
    sync()
    decode_ms = (time.perf_counter() - t1) * 1000

    total_tokens = sum(len(o.split()) for o in outputs)
    tokens_per_s = total_tokens / (decode_ms / 1000) if decode_ms > 0 else 0.0

    timing = {
        "vision_bridge_ms": vision_bridge_ms,
        "decode_ms":        decode_ms,
        "total_latency_ms": vision_bridge_ms + decode_ms,
        "num_tokens":       total_tokens,
        "tokens_per_s":     tokens_per_s,
    }
    return outputs, timing, gate_weights


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="M2 VQA Performance Evaluation")
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--ckpt",        type=str, default=None)
    parser.add_argument("--split",       type=str, default="test",
                        choices=["train", "validation", "test"])
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size",  type=int, default=1,
                        help="建议 1 以精确测量单样本延迟")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    cfg   = load_config(args.config)
    m_cfg = cfg["models"]

    out_dir  = os.path.join(cfg["output"]["out_dir"], cfg["output"]["run_name"])
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = args.ckpt or find_checkpoint(ckpt_dir)
    print(f"[eval] checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device: {device}")

    # ── 构建模型 ─────────────────────────────────────────────
    amp_dtype        = torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16
    enabled_encoders = _get_enabled_encoders(m_cfg)
    m_queries        = _auto_m_queries(m_cfg)

    print(f"[eval] enabled_encoders: {enabled_encoders}, m_queries: {m_queries}")

    model = ECQFormerM2Offline(
        llama_local_dir      = m_cfg["llama_dir"],
        clip_local_dir       = m_cfg["clip_dir"],
        dinov2_local_dir     = m_cfg["dinov2_dir"],
        biomedclip_local_dir = m_cfg["biomedclip_dir"],
        d_bridge      = int(m_cfg["d_bridge"]),
        meq_layers    = int(m_cfg["meq_layers"]),
        meq_heads     = int(m_cfg["meq_heads"]),
        m_queries     = int(m_queries),
        attn_type     = str(m_cfg.get("attn_type", "param_free")),
        phi           = str(m_cfg.get("phi", "silu")),
        score_scale   = bool(m_cfg.get("score_scale", True)),
        score_norm    = bool(m_cfg.get("score_norm", False)),
        adaptive_drop = m_cfg.get("adaptive_drop", None),
        gating        = m_cfg.get("gating", None),
        film          = m_cfg.get("film", None),
        enabled_encoders = enabled_encoders,
        torch_dtype   = amp_dtype,
    ).to(device)

    step = load_checkpoint_weights(model, ckpt_path)
    model.eval()

    params = count_params(model)
    print(f"[eval] loaded step={step}")
    print(f"[eval] total={params['total']/1e6:.2f}M  trainable={params['trainable']/1e6:.2f}M")
    if model.text_enc is not None:
        print(f"  M2 text_enc : {sum(p.numel() for p in model.text_enc.parameters())/1e6:.2f}M")
    if model.gating is not None:
        print(f"  M2 gating   : {sum(p.numel() for p in model.gating.parameters())/1e6:.2f}M")
    if model.meq.film_mod is not None:
        print(f"  M2 film_mod : {sum(p.numel() for p in model.meq.film_mod.parameters())/1e6:.2f}M")

    # ── 数据 ─────────────────────────────────────────────────
    eval_cfg = cfg.get("eval", {})
    loader = build_vqa_loader(
        dataset_id  = str(cfg["data"]["dataset_id"]),
        split       = args.split,
        cache_dir   = cfg["data"].get("cache_dir", None),
        local_dir   = cfg["data"].get("local_dir", None),
        images_dir  = cfg["data"].get("images_dir", None),
        language    = str(cfg["data"].get("language", "en")),
        batch_size  = args.batch_size,
        num_workers = 0,
        pin_memory  = True,
        shuffle     = False,
        max_samples = args.max_samples,
    )
    print(f"[eval] {cfg['data']['dataset_id']} / {args.split}  samples={min(args.max_samples, len(loader.dataset))}")

    prompt_template = str(cfg["train"].get("prompt_template",
        "You are a helpful medical assistant. "
        "Answer the question based on the image.\n"
        "Question: {question}\nAnswer:"))
    gen_params = {
        "max_new_tokens": int(eval_cfg.get("max_new_tokens", 10)),
        "num_beams":      int(eval_cfg.get("num_beams", 3)),
        "temperature":    float(eval_cfg.get("temperature", 0.1)),
        "top_p":          float(eval_cfg.get("top_p", 0.9)),
    }
    print(f"[eval] gen_params: {gen_params}")

    # ── Warmup ───────────────────────────────────────────────
    print("[eval] warmup...")
    for batch in loader:
        generate_m2_with_timing(
            model, batch["images_pil"], batch["questions"],
            device, prompt_template, gen_params
        )
        break

    # ── 评估循环 ─────────────────────────────────────────────
    vision_bridge_ms_list, decode_ms_list, total_ms_list = [], [], []
    tokens_list, tps_list, peak_mems_gb = [], [], []
    preds, golds, closed = [], [], []
    all_gate_weights: List[torch.Tensor] = []   # (B, K) per batch

    print("[eval] running evaluation...")
    t0 = time.time()

    for batch in tqdm(loader, desc=f"[eval {args.split}]"):
        images_pil = batch["images_pil"]
        questions  = batch["questions"]
        answers    = batch["answers"]
        is_closed  = batch["is_closed"]

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        outputs, timing, gate_w = generate_m2_with_timing(
            model, images_pil, questions, device, prompt_template, gen_params
        )

        vision_bridge_ms_list.append(timing["vision_bridge_ms"])
        decode_ms_list.append(timing["decode_ms"])
        total_ms_list.append(timing["total_latency_ms"])
        tokens_list.append(timing["num_tokens"])
        tps_list.append(timing["tokens_per_s"])

        if device.type == "cuda":
            peak_mems_gb.append(torch.cuda.max_memory_allocated(device) / (1024 ** 3))

        if gate_w is not None:
            all_gate_weights.append(gate_w)

        preds.extend(outputs)
        golds.extend(answers)
        closed.extend(is_closed)

    total_time = time.time() - t0

    # ── 指标汇总 ─────────────────────────────────────────────
    print("\n[eval] computing VQA metrics...")
    m  = compute_vqa_metrics(preds, golds, closed, use_clean=True)
    n  = len(vision_bridge_ms_list)
    av = lambda lst: sum(lst) / len(lst) if lst else 0.0

    avg_vision_bridge  = av(vision_bridge_ms_list)
    avg_decode         = av(decode_ms_list)
    avg_total_latency  = av(total_ms_list)
    avg_peak_mem       = av(peak_mems_gb) if peak_mems_gb else None
    throughput         = len(preds) / total_time if total_time > 0 else 0.0

    # 门控均值（每路编码器，越分散说明门控越有效）
    gate_summary = None
    if all_gate_weights:
        stacked = torch.cat(all_gate_weights, dim=0)  # (N_samples, K)
        means   = stacked.mean(dim=0).tolist()
        gate_summary = {enc: round(means[i], 4) for i, enc in enumerate(enabled_encoders)}

    metrics = {
        "model":      "ECQFormer-M2-VQA",
        "checkpoint": ckpt_path,
        "step":       step,
        "split":      args.split,
        "dataset":    cfg["data"]["dataset_id"],
        "enabled_encoders": enabled_encoders,
        "m_queries":  m_queries,
        "num_evaluated": len(preds),

        "vqa_metrics": {
            "closed_acc":  round(m.closed_acc, 4),
            "open_acc":    round(m.open_acc, 4),
            "overall_acc": round(m.overall_acc, 4),
            "n_closed": m.n_closed, "n_open": m.n_open, "n_total": m.n_total,
        },

        "efficiency": {
            "trainable_params":   params["trainable"],
            "trainable_params_M": round(params["trainable"] / 1e6, 2),
            "total_params":       params["total"],
            "total_params_M":     round(params["total"] / 1e6, 2),

            # 延迟（M2 vision_bridge 包含文本编码 + 门控 + FiLM）
            "avg_vision_bridge_ms": round(avg_vision_bridge, 2),
            "avg_decode_ms":        round(avg_decode, 2),
            "avg_total_latency_ms": round(avg_total_latency, 2),

            "avg_num_tokens":       round(av(tokens_list), 1),
            "avg_tokens_per_s":     round(av(tps_list), 2),
            "avg_peak_mem_gb":      round(avg_peak_mem, 3) if avg_peak_mem else None,
            "throughput_samples_per_s": round(throughput, 2),
        },

        # M2 独有：门控均值（反映三路编码器的平均贡献）
        "gate_means": gate_summary,

        "total_eval_time_s": round(total_time, 2),
    }

    metrics_file = os.path.join(out_dir, f"performance_{args.split}.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ── 打印报告 ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  ECQFormer-M2 VQA Performance ({args.split})")
    print("=" * 65)
    print(f"  Checkpoint:  {os.path.basename(ckpt_path)} (step {step})")
    print(f"  Dataset:     {cfg['data']['dataset_id']}")
    print(f"  Encoders:    {enabled_encoders}")
    print(f"  Samples:     {len(preds)}")
    print("-" * 65)
    print("  [VQA Accuracy]")
    print(f"    Closed Acc:  {m.closed_acc:.4f} (n={m.n_closed})")
    print(f"    Open Acc:    {m.open_acc:.4f} (n={m.n_open})")
    print(f"    Overall Acc: {m.overall_acc:.4f} (n={m.n_total})")
    print("-" * 65)
    print("  [Efficiency Metrics]  (* vision_bridge includes text enc + gating + FiLM)")
    print(f"    Trainable Params:    {params['trainable']/1e6:.2f}M")
    print(f"    Vision+Bridge* Time: {avg_vision_bridge:.2f}ms")
    print(f"    Decode Time:         {avg_decode:.2f}ms")
    print(f"    Total Latency:       {avg_total_latency:.2f}ms")
    print(f"    Avg Tokens:          {av(tokens_list):.1f}")
    print(f"    Tokens/sec:          {av(tps_list):.2f}")
    if avg_peak_mem:
        print(f"    Peak Memory:         {avg_peak_mem:.3f}GB")
    print(f"    Throughput:          {throughput:.2f} samples/s")
    if gate_summary:
        print("-" * 65)
        print("  [M2 Gate Weights (avg across samples)]")
        for enc, val in gate_summary.items():
            bar = "█" * int(val * 20)
            print(f"    {enc:<12s}: {val:.4f}  {bar}")
    print("=" * 65)
    print(f"  Results saved to: {metrics_file}")


if __name__ == "__main__":
    main()
