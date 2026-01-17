from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Tuple


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
    return normalize_answer(pred) == normalize_answer(gold)


def contain_match(pred: str, gold: str) -> bool:
    """Check if gold answer is contained in prediction (more lenient)."""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if not gold_norm:
        return False
    return gold_norm in pred_norm or pred_norm in gold_norm


@dataclass
class VQAMetrics:
    open_acc: float
    closed_acc: float
    overall_acc: float
    contain_acc: float      # 新增：包含匹配准确率
    n_open: int
    n_closed: int
    n_total: int


def compute_vqa_metrics(
    preds: Iterable[str],
    golds: Iterable[str],
    is_closed: Iterable[bool],
    use_clean: bool = True,  # 是否使用答案清理
) -> VQAMetrics:
    n_open = n_closed = 0
    c_open = c_closed = 0
    c_contain = 0
    n_total = 0
    c_total = 0

    for p, g, c in zip(preds, golds, is_closed):
        n_total += 1
        
        # 清理预测
        if use_clean:
            p = clean_prediction(p)
        
        ok = exact_match(p, g)
        ok_contain = contain_match(p, g)
        
        c_total += int(ok)
        c_contain += int(ok_contain)
        
        if c:
            n_closed += 1
            c_closed += int(ok)
        else:
            n_open += 1
            c_open += int(ok)

    open_acc = c_open / n_open if n_open > 0 else 0.0
    closed_acc = c_closed / n_closed if n_closed > 0 else 0.0
    overall_acc = c_total / n_total if n_total > 0 else 0.0
    contain_acc = c_contain / n_total if n_total > 0 else 0.0
    
    return VQAMetrics(
        open_acc=open_acc,
        closed_acc=closed_acc,
        overall_acc=overall_acc,
        contain_acc=contain_acc,
        n_open=n_open,
        n_closed=n_closed,
        n_total=n_total,
    )


def split_open_closed(question: str, answer: str) -> Tuple[bool, str]:
    """Utility for datasets without explicit answer_type.

    Returns (is_closed, normalized_answer).
    """
    a = normalize_answer(answer)
    return (a in {"yes", "no"}), a
