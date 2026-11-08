"""Purpose: the measurements a judge evaluation reports over labelled rows:
ROC AUC, direction accuracy over one-way pairs, the dev-calibrated threshold,
and accuracy, balanced accuracy, precision and recall at a threshold.

Scores and labels are parallel sequences; a row is admitted when its score
reaches the threshold, as the engine admits a rewrite. Unequal lengths raise
ValueError.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Sequence


# Plain Python, so the core package imports this without numpy; the results
# equal the numpy version's bit for bit, since every sum is of integers or
# half-integers and every other step is the same IEEE operation.
def _rows(scores: Sequence[float], labels: Sequence[bool]) -> tuple[list[float], list[bool]]:
    s = [float(x) for x in scores]
    y = [bool(v) for v in labels]
    if len(s) != len(y):
        msg = f"{len(s)} scores against {len(y)} labels"
        raise ValueError(msg)
    return s, y


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """The chance a random true row outscores a random false one, ties counted half; None without both classes.

    Time: n log n for the ranking, n = len(scores).

    >>> roc_auc([0.9, 0.2, 0.5, 0.5], [True, False, True, False])
    0.875
    """
    s, y = _rows(scores, labels)
    positives = sum(y)
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(s)), key=s.__getitem__)
    ranks = [0.0] * len(s)
    # Average ranks over ties (1-based), so tied pairs count half in the Mann-Whitney U statistic.
    start = 0
    while start < len(s):
        end = start
        while end + 1 < len(s) and s[order[end + 1]] == s[order[start]]:
            end += 1
        for i in order[start : end + 1]:
            ranks[i] = (start + end) / 2 + 1
        start = end + 1
    u = sum(r for r, true in zip(ranks, y) if true) - positives * (positives + 1) / 2
    return u / (positives * negatives)


def direction(true_scores: Sequence[float], false_scores: Sequence[float]) -> dict[str, float | int | None]:
    """How often a one-way pair's true direction outscores its false one.

    Each pair counts 1 when the true direction scores higher, 0.5 when tied and
    0 when lower; `ties` is the share tied, so a symmetric judge reads accuracy
    0.5 with ties 1.0.

    >>> direction([0.9, 0.4, 0.5], [0.1, 0.6, 0.5])
    {'pairs': 3, 'accuracy': 0.5, 'ties': 0.3333333333333333}
    """
    t = [float(x) for x in true_scores]
    f = [float(x) for x in false_scores]
    if len(t) != len(f):
        msg = f"{len(t)} true-direction scores against {len(f)} false-direction scores"
        raise ValueError(msg)
    if not t:
        return {"pairs": 0, "accuracy": None, "ties": None}
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in zip(t, f))
    ties = sum(a == b for a, b in zip(t, f))
    return {"pairs": len(t), "accuracy": wins / len(t), "ties": ties / len(t)}


def calibrate(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """The threshold maximising balanced accuracy of "admit when score >= threshold" on these rows.

    Balanced accuracy weights each class by the inverse of its count, so the
    threshold ignores the rows' class ratio: on rows one fifth true, plain
    accuracy rewards admitting nothing whenever a judge's precision stays
    under one half. With one class absent it is plain accuracy.

    Candidates sit halfway between 0 and the lowest score (admit all), between
    each two adjacent distinct scores, and halfway between the highest score and
    1 (admit none; just above 1 when a score is 1); among the candidates reaching
    the best accuracy the middle one is taken, so a plateau resolves to its centre.

    Time: n log n for the sort, then one bisection of the sorted scores per
    candidate, n = len(scores), at most n + 1 candidates.

    >>> calibrate([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
    0.5
    >>> calibrate([0.1, 0.2, 0.3, 0.4, 0.6, 0.9], [False, False, False, False, True, True])
    0.5
    >>> calibrate([0.3, 0.4], [False, False])
    0.7
    """
    s, y = _rows(scores, labels)
    if not s:
        msg = "cannot calibrate a threshold on no rows"
        raise ValueError(msg)
    distinct = sorted(set(s))
    above = (distinct[-1] + 1.0) / 2 if distinct[-1] < 1.0 else math.nextafter(1.0, 2.0)
    candidates = [distinct[0] / 2, *((a + b) / 2 for a, b in itertools.pairwise(distinct)), above]
    # Rows admitted at a candidate are those scoring at or above it; count each class below a cut with prefix sums.
    order = sorted(range(len(s)), key=s.__getitem__)
    sorted_s = [s[i] for i in order]
    below_true, below_false = [0], [0]
    for i in order:
        below_true.append(below_true[-1] + y[i])
        below_false.append(below_false[-1] + (not y[i]))
    positives, negatives = below_true[-1], below_false[-1]
    weight_true = 1.0 / positives if positives else 0.0
    weight_false = 1.0 / negatives if negatives else 0.0
    correct = []
    for candidate in candidates:
        cut = bisect.bisect_left(sorted_s, candidate)
        # Rows below a candidate are rejected: true negatives there, true positives above it.
        value = weight_false * below_false[cut] + weight_true * (positives - below_true[cut])
        # Rounded to 12 decimals as numpy.round does it (scale, round half to even, unscale), so sums equal
        # but for their last bits tie.
        correct.append(round(value * 1e12) / 1e12)
    top = max(correct)
    best = [i for i, value in enumerate(correct) if value == top]
    return candidates[best[len(best) // 2]]


def at_threshold(scores: Sequence[float], labels: Sequence[bool], threshold: float) -> dict[str, float | int | None]:
    """Accuracy, balanced accuracy, precision and recall of admitting rows scoring at or above `threshold`.

    Precision is None when nothing is admitted, recall None when nothing is true,
    balanced accuracy None unless both classes are present.

    >>> at_threshold([0.9, 0.6, 0.2], [True, False, False], 0.5)
    {'threshold': 0.5, 'rows': 3, 'admitted': 2, 'accuracy': 0.6666666666666666, 'balanced_accuracy': 0.75, 'precision': 0.5, 'recall': 1.0}
    """
    s, y = _rows(scores, labels)
    admitted = [x >= threshold for x in s]
    hits = sum(a and true for a, true in zip(admitted, y))
    rejected = sum(not a and not true for a, true in zip(admitted, y))
    trues = sum(y)
    falses = len(y) - trues
    count = sum(admitted)
    return {
        "threshold": float(threshold),
        "rows": len(s),
        "admitted": count,
        "accuracy": sum(a == true for a, true in zip(admitted, y)) / len(s) if s else None,
        "balanced_accuracy": (hits / trues + rejected / falses) / 2 if trues and falses else None,
        "precision": hits / count if count else None,
        "recall": hits / trues if trues else None,
    }
