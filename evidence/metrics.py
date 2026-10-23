"""
Answer metrics for the evidence pipeline.

Same definition as the repo's own evaluation (evaluation/eval.py): lower-case, drop
punctuation, drop the articles a/an/the, collapse whitespace; EM and token-F1 are the
max over all gold aliases. tests/test_evidence_offline.py checks the two stay identical.
"""
import re
import string
from collections import Counter

_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_f1(pred: str, golds):
    """(EM, F1), each the max over the gold aliases. An empty prediction scores 0 on a non-empty gold."""
    best_em, best_f1 = 0.0, 0.0
    p = normalize_answer(pred).split()
    for g in golds or []:
        gt = normalize_answer(str(g)).split()
        best_em = max(best_em, float(p == gt))
        common = sum((Counter(p) & Counter(gt)).values())
        if common and p and gt:
            pr, rc = common / len(p), common / len(gt)
            best_f1 = max(best_f1, 2 * pr * rc / (pr + rc))
    return best_em, best_f1
