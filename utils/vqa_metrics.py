from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

# BERTScore lazy import to avoid startup overhead
_bertscore_scorer = None

# Local path for RoBERTa-large (offline mode)
ROBERTA_LOCAL_PATH = "/root/autodl-tmp/hf_models/roberta-large"


def _get_bertscore_scorer():
    """Lazy load BERTScore scorer with RoBERTa-large backbone.
    
    Supports offline mode by setting HF_HOME to local cache.
    """
    global _bertscore_scorer
    if _bertscore_scorer is None:
        try:
            import os
            from bert_score import BERTScorer
            
            # For offline mode: set environment variable before loading
            # This tells HuggingFace to look in the local directory
            if os.path.exists(ROBERTA_LOCAL_PATH):
                # Create symlink in HF cache format if needed
                hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                os.makedirs(hf_cache_dir, exist_ok=True)
                
                # Set offline mode
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                
                print(f"[vqa_metrics] Using local RoBERTa-large: {ROBERTA_LOCAL_PATH}")
            else:
                print(f"[vqa_metrics] Local path not found, using HuggingFace online")
            
            # Use standard model name - BERTScore handles it internally
            # With rescale_with_baseline=False to avoid baseline download issues
            _bertscore_scorer = BERTScorer(
                model_type=ROBERTA_LOCAL_PATH if os.path.exists(ROBERTA_LOCAL_PATH) else "roberta-large",
                lang="en",
                rescale_with_baseline=False,  # Disable baseline rescaling to avoid download
                device="cuda"
            )
            print("[vqa_metrics] BERTScore initialized successfully")
        except ImportError:
            print("[vqa_metrics] WARNING: bert-score not installed. Run: pip install bert-score")
            _bertscore_scorer = None
        except Exception as e:
            print(f"[vqa_metrics] WARNING: BERTScore init failed: {e}")
            _bertscore_scorer = None
    return _bertscore_scorer


def normalize_answer(s: str) -> str:
    """A simple, robust normalization for exact-match VQA accuracy.

    - lowercase
    - strip
    - collapse whitespaces
    - remove trailing punctuation

    Note: Med-VQA leaderboards often use exact match on normalized strings.
    This function is intentionally conservative and avoids aggressive rules
    (e.g., article removal) that may harm medical terminology.
    """
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\s\.,;:!\?\-]+$", "", s)
    return s


def clean_prediction(pred: str) -> str:
    """Clean up long model predictions to extract the core answer.
    
    Strategies:
    1. Take only the first line
    2. Take content before first period/comma
    3. Remove common prefixes
    4. Limit to first few words
    """
    if not pred:
        return ""
    
    # Remove common prefixes
    prefixes_to_remove = [
        "the answer is ",
        "answer: ",
        "answer is ",
        "it is ",
        "this is ",
    ]
    pred_lower = pred.lower().strip()
    for prefix in prefixes_to_remove:
        if pred_lower.startswith(prefix):
            pred = pred[len(prefix):]
            break
    
    # Take first line
    pred = pred.split('\n')[0].strip()
    
    # Take content before first sentence-ending punctuation
    pred = re.split(r'[\.!]', pred)[0].strip()
    
    # For closed questions (yes/no), extract just yes or no
    pred_lower = pred.lower().strip()
    if 'yes' in pred_lower and 'no' not in pred_lower:
        return 'yes'
    if 'no' in pred_lower and 'yes' not in pred_lower:
        return 'no'
    
    # Limit to first 5 words for open-ended
    words = pred.split()
    if len(words) > 5:
        pred = ' '.join(words[:5])
    
    return pred.strip()


def exact_match(pred: str, gold: str) -> bool:
    """Exact match after normalization (Equation 3-18 in thesis)."""
    return normalize_answer(pred) == normalize_answer(gold)


def compute_bertscore(preds: List[str], golds: List[str]) -> List[float]:
    """Compute BERTScore F1 for a batch of predictions.
    
    Uses RoBERTa-large as backbone (as specified in thesis).
    Returns list of F1 scores.
    """
    scorer = _get_bertscore_scorer()
    if scorer is None:
        # Fallback: return zeros if BERTScore not available
        return [0.0] * len(preds)
    
    try:
        # BERTScore returns (P, R, F1)
        P, R, F1 = scorer.score(preds, golds)
        return F1.cpu().tolist()
    except Exception as e:
        print(f"[vqa_metrics] BERTScore computation failed: {e}")
        return [0.0] * len(preds)


def hybrid_semantic_match(pred: str, gold: str, bertscore_f1: float, threshold: float = 0.85) -> bool:
    """Hybrid Semantic Matching for open-ended questions (Equation 3-19 in thesis).
    
    S_open = 1 if EM(P, G) = 1 OR BERTScore(P, G) >= τ
             0 otherwise
    
    Args:
        pred: Predicted answer
        gold: Gold answer
        bertscore_f1: Pre-computed BERTScore F1 value
        threshold: Semantic similarity threshold τ (default 0.85)
    
    Returns:
        True if match, False otherwise
    """
    # First try exact match
    if exact_match(pred, gold):
        return True
    
    # If EM fails, check BERTScore
    return bertscore_f1 >= threshold


@dataclass
class VQAMetrics:
    """VQA evaluation metrics following thesis equations 3-18, 3-19, 3-20."""
    closed_acc: float       # Acc_closed (Equation 3-18)
    open_acc: float         # Acc_open based on hybrid matching (Equation 3-19)
    overall_acc: float      # Acc_overall (Equation 3-20)
    
    # Detailed counts
    n_closed: int           # N_closed
    n_open: int             # N_open
    n_total: int            # N_total
    c_closed: int           # C_correct (closed)
    c_open: int             # O_correct (open)
    
    # Additional metrics for analysis
    open_em_only: float     # Open accuracy with EM only (for comparison)
    open_bert_helped: int   # Number of open questions where BERTScore helped


def compute_vqa_metrics(
    preds: Iterable[str],
    golds: Iterable[str],
    is_closed: Iterable[bool],
    use_clean: bool = True,
    bertscore_threshold: float = 0.80,
) -> VQAMetrics:
    """Compute VQA metrics using hybrid semantic matching strategy.
    
    - Closed questions: Exact Match (Equation 3-18)
    - Open questions: EM first, then BERTScore >= threshold (Equation 3-19)
    - Overall: (C_correct + O_correct) / N_total (Equation 3-20)
    
    Args:
        preds: List of predictions
        golds: List of gold answers
        is_closed: List of bool flags (True = closed question)
        use_clean: Whether to clean predictions before matching
        bertscore_threshold: τ threshold for semantic matching (default 0.85)
    """
    # Convert to lists
    preds_list = list(preds)
    golds_list = list(golds)
    is_closed_list = list(is_closed)
    
    n_total = len(preds_list)
    if n_total == 0:
        return VQAMetrics(
            closed_acc=0.0, open_acc=0.0, overall_acc=0.0,
            n_closed=0, n_open=0, n_total=0, c_closed=0, c_open=0,
            open_em_only=0.0, open_bert_helped=0
        )
    
    # Apply cleaning if requested
    if use_clean:
        preds_list = [clean_prediction(p) for p in preds_list]
    
    # Separate closed and open questions
    closed_indices = [i for i, c in enumerate(is_closed_list) if c]
    open_indices = [i for i, c in enumerate(is_closed_list) if not c]
    
    n_closed = len(closed_indices)
    n_open = len(open_indices)
    
    # ========== Closed Questions: Exact Match (Equation 3-18) ==========
    c_closed = 0
    for i in closed_indices:
        if exact_match(preds_list[i], golds_list[i]):
            c_closed += 1
    
    # ========== Open Questions: Hybrid Semantic Matching (Equation 3-19) ==========
    c_open = 0
    c_open_em_only = 0
    bert_helped = 0
    
    if n_open > 0:
        # Separate open questions for BERTScore batch computation
        open_preds = [preds_list[i] for i in open_indices]
        open_golds = [golds_list[i] for i in open_indices]
        
        # First pass: check exact match
        open_em_results = [exact_match(p, g) for p, g in zip(open_preds, open_golds)]
        c_open_em_only = sum(open_em_results)
        
        # Second pass: for non-EM matches, compute BERTScore
        non_em_indices = [j for j, em in enumerate(open_em_results) if not em]
        
        if non_em_indices:
            non_em_preds = [open_preds[j] for j in non_em_indices]
            non_em_golds = [open_golds[j] for j in non_em_indices]
            
            # Batch compute BERTScore
            bert_scores = compute_bertscore(non_em_preds, non_em_golds)
            
            # Check which ones pass threshold
            for j, bs in zip(non_em_indices, bert_scores):
                if bs >= bertscore_threshold:
                    open_em_results[j] = True
                    bert_helped += 1
        
        c_open = sum(open_em_results)
    
    # ========== Overall Accuracy (Equation 3-20) ==========
    # Acc_overall = (C_correct + O_correct) / N_total
    
    closed_acc = c_closed / n_closed if n_closed > 0 else 0.0
    open_acc = c_open / n_open if n_open > 0 else 0.0
    overall_acc = (c_closed + c_open) / n_total if n_total > 0 else 0.0
    open_em_only_acc = c_open_em_only / n_open if n_open > 0 else 0.0
    
    return VQAMetrics(
        closed_acc=closed_acc,
        open_acc=open_acc,
        overall_acc=overall_acc,
        n_closed=n_closed,
        n_open=n_open,
        n_total=n_total,
        c_closed=c_closed,
        c_open=c_open,
        open_em_only=open_em_only_acc,
        open_bert_helped=bert_helped,
    )


def split_open_closed(question: str, answer: str) -> Tuple[bool, str]:
    """Utility for datasets without explicit answer_type.

    Returns (is_closed, normalized_answer).
    """
    a = normalize_answer(answer)
    return (a in {"yes", "no"}), a
