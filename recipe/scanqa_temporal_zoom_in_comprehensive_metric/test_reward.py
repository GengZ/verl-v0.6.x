import math
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Iterable, Optional

# ----------------------------
# Helpers
# ----------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")

def _maybe_extract_answer(s: str) -> str:
    """
    If your outputs are wrapped like <answer>...</answer>, extract that span.
    Otherwise return the original string unchanged.
    """
    m = re.search(r"<answer>(.*?)</answer>", s, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else s

def normalize_text(s: str) -> str:
    """
    'Refined' normalization commonly used for EM in QA:
    - extract <answer>...</answer> if present
    - lowercase
    - strip
    - collapse whitespace
    - remove punctuation (tweak if your official eval keeps punctuation)
    """
    s = _maybe_extract_answer(s)
    s = s.lower().strip()
    s = _WHITESPACE_RE.sub(" ", s)
    s = _PUNCT_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s

def tokenize_words(s: str) -> List[str]:
    s = normalize_text(s)
    return s.split() if s else []

def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)] if n > 0 else []

# ----------------------------
# 1) Exact Match (refined)
# ----------------------------

def em_refined(output_str: str, target_str: str) -> int:
    """
    Returns 1 if normalized output == normalized target, else 0.
    """
    return int(normalize_text(output_str) == normalize_text(target_str))

# ----------------------------
# 2) BLEU-4 (sentence-level with smoothing)
#   - Returns 0..100 (percentage scale)
# ----------------------------

def bleu4_score(output_str: str, target_str: str) -> float:
    """
    Sentence-level BLEU-4 with smoothing (method-1: add-one).
    Single reference version.
    Returns BLEU-4 on a 0..100 scale.
    """
    cand = tokenize_words(output_str)
    ref = tokenize_words(target_str)
    if len(cand) == 0:
        return 0.0
    # Modified n-gram precisions with clipping
    precisions = []
    for n in range(1, 5):
        cand_ngrams = Counter(ngrams(cand, n))
        ref_ngrams = Counter(ngrams(ref, n))
        overlap = 0
        total = 0
        for g, c in cand_ngrams.items():
            overlap += min(c, ref_ngrams.get(g, 0))
            total += c
        # smoothing: +1 / +1
        p_n = (overlap + 1.0) / (total + 1.0) if total > 0 else 1.0
        precisions.append(p_n)

    # Brevity penalty (single ref)
    c = len(cand)
    r = len(ref)
    if c == 0:
        return 0.0
    bp = 1.0 if c > r else math.exp(1 - float(r) / max(1, c))

    # geometric mean of precisions
    log_prec = sum(math.log(p) for p in precisions) / 4.0
    bleu = bp * math.exp(log_prec)
    return 100.0 * bleu

# ----------------------------
# 3) ROUGE-L (F1 over LCS)
#   - Returns 0..100
# ----------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    """
    Longest Common Subsequence length (O(len(a)*len(b)) DP).
    """
    la, lb = len(a), len(b)
    dp = [0] * (lb + 1)
    for i in range(1, la + 1):
        prev = 0
        for j in range(1, lb + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[lb]

def rouge_l_score(output_str: str, target_str: str, beta: float = 1.0) -> float:
    """
    ROUGE-L F-measure between candidate and single reference (0..100).
    """
    cand = tokenize_words(output_str)
    ref = tokenize_words(target_str)
    if len(cand) == 0 or len(ref) == 0:
        return 0.0
    lcs = _lcs_length(cand, ref)
    prec = lcs / len(cand)
    rec = lcs / len(ref)
    if prec == 0 and rec == 0:
        return 0.0
    beta2 = beta * beta
    f1 = (1 + beta2) * prec * rec / (rec + beta2 * prec) if (rec + beta2 * prec) > 0 else 0.0
    return 100.0 * f1

# ----------------------------
# 4) CIDEr (lightweight proxy of CIDEr-D)
#   - Returns cider on 0..100 scale
#   - Accepts optional 'idf_corpus' to compute IDF; if None, uses TF-only (idf=1)
# ----------------------------

def _build_df(corpus_refs: Iterable[str], n_max: int = 4) -> Dict[int, Dict[Tuple[str, ...], int]]:
    """
    Build document frequency (DF) counts for n-grams 1..n_max over a reference corpus.
    DF is number of refs in which the n-gram appears at least once.
    """
    df: Dict[int, Dict[Tuple[str, ...], int]] = {n: defaultdict(int) for n in range(1, n_max + 1)}
    for ref_text in corpus_refs:
        tokens = tokenize_words(ref_text)
        for n in range(1, n_max + 1):
            seen = set(ngrams(tokens, n))
            for g in seen:
                df[n][g] += 1
    return df

def _tf_vector(tokens: List[str], n: int) -> Dict[Tuple[str, ...], float]:
    g = ngrams(tokens, n)
    counts = Counter(g)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}  # normalized TF

def _cosine_sim_weighted(tf_c: Dict, tf_r: Dict, idf: Dict[Tuple[str, ...], float]) -> float:
    # dot
    dot = 0.0
    for g, wc in tf_c.items():
        if g in tf_r:
            w = idf.get(g, 1.0)
            dot += wc * tf_r[g] * (w * w)
    # norms
    def _norm(tf):
        return math.sqrt(sum((v * idf.get(g, 1.0)) ** 2 for g, v in tf.items()))
    nc = _norm(tf_c)
    nr = _norm(tf_r)
    if nc == 0 or nr == 0:
        return 0.0
    return dot / (nc * nr)

def cider_score_0_100(
    output_str: str,
    target_str: str,
    idf_corpus: Optional[Iterable[str]] = None,
    n_max: int = 4,
    sigma: float = 6.0
) -> float:
    """
    A lightweight CIDEr-D style proxy:
    - TF-IDF cosine similarity averaged over n=1..n_max
    - Gaussian length penalty (sigma as in CIDEr-D)
    - Scaled to 0..100 for convenience
    If idf_corpus is None, uses idf=1 for all n-grams (still useful; just lacks DF weighting).
    """
    cand = tokenize_words(output_str)
    ref = tokenize_words(target_str)

    # Build IDF if corpus provided
    idf_by_n: Dict[int, Dict[Tuple[str, ...], float]] = {n: defaultdict(lambda: 1.0) for n in range(1, n_max + 1)}
    if idf_corpus is not None:
        df = _build_df(idf_corpus, n_max=n_max)
        N = max(1, len(list(idf_corpus)))
        for n in range(1, n_max + 1):
            for g, d in df[n].items():
                # IDF per CIDEr: log((N + 1) / (df + 1))
                idf_by_n[n][g] = math.log((N + 1) / (d + 1))

    # Gaussian length penalty
    len_pen = math.exp(-((len(cand) - len(ref)) ** 2) / (2 * (sigma ** 2))) if sigma > 0 else 1.0

    sims = []
    for n in range(1, n_max + 1):
        tf_c = _tf_vector(cand, n)
        tf_r = _tf_vector(ref, n)
        sim = _cosine_sim_weighted(tf_c, tf_r, idf_by_n[n])
        sims.append(sim)

    cider = len_pen * (sum(sims) / max(1, len(sims)))  # 0..1-ish
    return 100.0 * cider  # scale to 0..100

# ----------------------------
# 5) Normalized CIDEr in [0,1]
# ----------------------------

def cider_n(output_str: str, target_str: str, idf_corpus: Optional[Iterable[str]] = None) -> float:
    """
    Normalized CIDEr proxy in [0,1], by dividing the 0..100 variant by 100 and clipping.
    """
    c = cider_score_0_100(output_str, target_str, idf_corpus=idf_corpus)
    return max(0.0, min(c / 100.0, 1.0))

def total_reward(output_str: str, target_str: str) -> float:
    em_reward = em_refined(output_str, target_str)
    bleu4_reward = bleu4_score(output_str, target_str)
    rouge_l_reward = rouge_l_score(output_str, target_str)
    cider_n_reward = cider_n(output_str, target_str) * 0.5
    return 0.2 * em_reward + 0.005 * bleu4_reward + 0.005 * rouge_l_reward + 0.2 * cider_n_reward


if __name__ == "__main__":
    # output_str = "The answer is 1."
    # target_str = "The answer is 1."
    # output_str = "SpongeBob Big Bang."
    # target_str = "SpongeBob Big Bang."
    output_str = "I am Batman."
    target_str = "I am Batman."
    print(em_refined(output_str, target_str))
    print(bleu4_score(output_str, target_str))
    print(rouge_l_score(output_str, target_str))
    print(cider_n(output_str, target_str))
    print(total_reward(output_str, target_str))