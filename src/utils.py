"""
Shared utilities: the official SQuAD normalization / EM / F1 functions
(reimplemented from the canonical SQuAD evaluation script, Rajpurkar et al. 2016/2018,
https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)
so that scores here are directly comparable to published SQuAD numbers.
"""
import re
import string
import collections


def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles, and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is a no-answer/empty string, F1 is 1 only if both are empty.
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list) -> float:
    """SQuAD allows multiple acceptable gold answers; score against the best-matching one."""
    if not ground_truths:
        ground_truths = [""]
    scores = [metric_fn(gt, prediction) for gt in ground_truths]
    return max(scores)


# --- Hallucination-proxy heuristics -----------------------------------------

UNCERTAINTY_MARKERS = [
    "i dont know",
    "i do not know",
    "im not sure",
    "i am not sure",
    "not sure",
    "cannot determine",
    "cant determine",
    "no information",
    "not mentioned",
    "not stated",
    "not provided",
    "unclear",
    "unknown",
    "i cannot answer",
    "i cant answer",
    "not specified",
    "not certain",
    "no answer",
    "not given",
    "not found in",
    "does not contain",
    "doesnt contain",
    "doesnt say",
    "does not say",
    "no idea",
]


def _strip_apostrophes(text: str) -> str:
    # Normalize straight ('), curly right (’) and curly left (‘)
    # apostrophes to nothing, so "don't" / "don't" / "dont" all match the
    # same marker. Model output punctuation is not fully predictable, so
    # matching should not depend on which apostrophe character it used.
    for ch in ("'", "’", "‘"):
        text = text.replace(ch, "")
    return text


def is_explicit_uncertainty(pred: str) -> bool:
    """Rule-based check: does the prediction explicitly hedge / abstain?
    Used as a proxy for 'the model admitted it didn't know' vs.
    'the model stated a concrete, confident, but wrong answer'.
    """
    p = _strip_apostrophes(pred.strip().lower())
    if p == "" or p in {"n/a", "na", "-"}:
        return True
    return any(marker in p for marker in UNCERTAINTY_MARKERS)


def classify_wrong_answer(pred: str) -> str:
    """Only call this on answers that already scored EM=0 (i.e., wrong)."""
    return "explicit_uncertainty" if is_explicit_uncertainty(pred) else "confident_fabrication"
