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


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


@dataclass
class VQAMetrics:
    open_acc: float
    closed_acc: float
    overall_acc: float
    n_open: int
    n_closed: int
    n_total: int


def compute_vqa_metrics(
    preds: Iterable[str],
    golds: Iterable[str],
    is_closed: Iterable[bool],
) -> VQAMetrics:
    n_open = n_closed = 0
    c_open = c_closed = 0
    n_total = 0
    c_total = 0

    for p, g, c in zip(preds, golds, is_closed):
        n_total += 1
        ok = exact_match(p, g)
        c_total += int(ok)
        if c:
            n_closed += 1
            c_closed += int(ok)
        else:
            n_open += 1
            c_open += int(ok)

    open_acc = c_open / n_open if n_open > 0 else 0.0
    closed_acc = c_closed / n_closed if n_closed > 0 else 0.0
    overall_acc = c_total / n_total if n_total > 0 else 0.0
    return VQAMetrics(
        open_acc=open_acc,
        closed_acc=closed_acc,
        overall_acc=overall_acc,
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
