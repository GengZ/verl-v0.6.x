#!/usr/bin/env python3
"""
ScanQA-style EM (Exact Match) and token-level F1 with shared normalization.

Normalization (SQuAD-style + light number mapping):
- lowercase
- strip whitespace, collapse multiple spaces
- remove punctuation
- remove English articles: a, an, the
- map simple number-words -> digits (e.g., "two" -> "2")

Both metrics take the max over references (gold answers + aliases).
"""

from __future__ import annotations
from collections import Counter
from typing import Iterable, List
import string

# ---------- Normalization helpers ----------

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

def _normalize(s: str | None) -> str:
    """SQuAD-style normalization plus simple number-word mapping."""
    if s is None:
        return ""
    s = s.lower().strip()
    # number words -> digits (token-wise; avoids 'stone' -> 'st1e')
    toks = [ _NUM_WORDS.get(t, t) for t in s.split() ]
    s = " ".join(toks)
    # strip punctuation
    s = s.translate(_PUNCT_TABLE)
    # remove articles
    toks = [t for t in s.split() if t not in _ARTICLES]
    # collapse spaces
    return " ".join(toks)

def _tokenize(s: str | None) -> List[str]:
    s = _normalize(s)
    return s.split() if s else []


# ---------- Metrics ----------

def exact_match(pred: str, refs: Iterable[str]) -> int:
    """Return 1 if normalized pred equals any normalized ref, else 0."""
    p = _normalize(pred)
    for r in refs:
        if p == _normalize(r):
            return 1
    return 0

def _f1_single(pred: str, ref: str) -> float:
    """Token-level F1 for a single reference."""
    p_toks = _tokenize(pred)
    r_toks = _tokenize(ref)

    if not p_toks and not r_toks:
        return 1.0
    if not p_toks or not r_toks:
        return 0.0

    p_cnt = Counter(p_toks)
    r_cnt = Counter(r_toks)
    overlap = sum((p_cnt & r_cnt).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(p_toks)
    recall = overlap / len(r_toks)
    return 2 * precision * recall / (precision + recall)

def f1_max_over_refs(pred: str, refs: Iterable[str]) -> float:
    """Max token-level F1 over references."""
    refs = list(refs)
    if not refs:
        return 0.0
    return max(_f1_single(pred, r) for r in refs)


# ---------- Tests ----------

def _approx_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol

def _run_tests():
    print("Running EM & F1 tests...\n")

    # 1) Canonical exact match with alias set
    pred = "red apple"
    refs = ["apple", "red apple"]
    em = exact_match(pred, refs)
    f1 = f1_max_over_refs(pred, refs)
    print(f"[exact] pred={pred!r} refs={refs} -> EM={em}, F1={f1:.4f}")
    assert em == 1, "Exact match should be 1 when a ref equals pred after normalization"
    assert _approx_equal(f1, 1.0), "F1 should be 1.0 when there is an exact ref"

    # 2) Articles/punctuation/spacing normalization
    pred = "  The,   Chair!! "
    refs = ["chair", "a chair"]
    em = exact_match(pred, refs)
    f1 = f1_max_over_refs(pred, refs)
    print(f"[norm]  pred={pred!r} refs={refs} -> EM={em}, F1={f1:.4f}")
    assert em == 1
    assert _approx_equal(f1, 1.0)

    # 3) Partial overlap (no EM, partial F1)
    pred = "blue apple"
    refs = ["red apple"]
    em = exact_match(pred, refs)
    f1 = f1_max_over_refs(pred, refs)
    print(f"[part]  pred={pred!r} refs={refs} -> EM={em}, F1={f1:.4f}")
    # Tokens: pred ['blue','apple'], ref ['red','apple'] -> overlap=1/2 & 1/2 => F1=0.5
    assert em == 0
    assert _approx_equal(f1, 0.5)

    # 4) Number-word → digit mapping (no unit unification here)
    pred = "two meters"
    refs = ["2 m", "two m", "2 meters"]
    em = exact_match(pred, refs)
    f1 = f1_max_over_refs(pred, refs)
    print(f"[num]   pred={pred!r} refs={refs} -> EM={em}, F1={f1:.4f}")
    # Normalized:
    # pred -> "2 meters"
    # refs -> ["2 m","2 m","2 meters"] => EM==1 via "2 meters"
    assert em == 1
    assert _approx_equal(f1, 1.0)

    # 5) Empty predictions/answers
    pred = ""
    refs = [""]
    em = exact_match(pred, refs)
    f1 = f1_max_over_refs(pred, refs)
    print(f"[empty] pred={pred!r} refs={refs} -> EM={em}, F1={f1:.4f}")
    assert em == 1
    assert _approx_equal(f1, 1.0)

    # 6) Empty pred vs non-empty gold
    pred = ""
    refs = ["chair"]
    em = exact_match(pred, refs)
    f1 = f1_max_over_refs(pred, refs)
    print(f"[empty] pred={pred!r} refs={refs} -> EM={em}, F1={f1:.4f}")
    assert em == 0
    assert _approx_equal(f1, 0.0)

    # 7) Whitespace collapse, case-insensitivity
    pred = "Red    APPLE"
    refs = ["red apple"]
    em = exact_match(pred, refs)
    f1 = f1_max_over_refs(pred, refs)
    print(f"[case]  pred={pred!r} refs={refs} -> EM={em}, F1={f1:.4f}")
    assert em == 1
    assert _approx_equal(f1, 1.0)

    # 8) Multi-set overlap counts
    pred = "red red chair"
    refs = ["red chair"]
    em = exact_match(pred, refs)
    f1 = f1_max_over_refs(pred, refs)
    print(f"[multi] pred={pred!r} refs={refs} -> EM={em}, F1={f1:.4f}")
    # p: ['red','red','chair'] (len=3), r: ['red','chair'] (len=2)
    # overlap multiset is 2 (one 'red', one 'chair') -> P=2/3, R=2/2=1 -> F1=0.8
    assert em == 0
    assert _approx_equal(f1, 0.8)

    print("\nAll tests passed ✅")

if __name__ == "__main__":
    _run_tests()
