"""Purpose: the measurements a judge evaluation reports over labelled rows:
ROC AUC, direction accuracy over one-way pairs, the dev-calibrated threshold,
and accuracy, balanced accuracy, precision and recall at a threshold.

Scores and labels are parallel sequences; a row is admitted when its score
reaches the threshold, as the engine admits a rewrite.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """The chance a random true row outscores a random false one, ties counted half; None without both classes.

    Time: n log n for the ranking, n = len(scores).

    >>> roc_auc([0.9, 0.2, 0.5, 0.5], [True, False, True, False])
    0.875
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    positives, negatives = int(y.sum()), int((~y).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_scores = s[order]
    # Average ranks over ties (1-based), so tied pairs count half in the Mann-Whitney U statistic.
    start = 0
    while start < len(s):
        end = start
        while end + 1 < len(s) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    u = ranks[y].sum() - positives * (positives + 1) / 2
    return float(u / (positives * negatives))


def direction(true_scores: Sequence[float], false_scores: Sequence[float]) -> dict[str, float | int | None]:
    """How often a one-way pair's true direction outscores its false one.

    Each pair counts 1 when the true direction scores higher, 0.5 when tied and
    0 when lower; `ties` is the share tied, so a symmetric judge reads accuracy
    0.5 with ties 1.0.

    >>> direction([0.9, 0.4, 0.5], [0.1, 0.6, 0.5])
    {'pairs': 3, 'accuracy': 0.5, 'ties': 0.3333333333333333}
    """
    t = np.asarray(true_scores, dtype=float)
    f = np.asarray(false_scores, dtype=float)
    if len(t) != len(f):
        msg = f"{len(t)} true-direction scores against {len(f)} false-direction scores"
        raise ValueError(msg)
    if len(t) == 0:
        return {"pairs": 0, "accuracy": None, "ties": None}
    wins = np.where(t > f, 1.0, np.where(t == f, 0.5, 0.0))
    return {"pairs": len(t), "accuracy": float(wins.mean()), "ties": float((t == f).mean())}


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

    Time: n log n for the sort, then one pass, n = len(scores).

    >>> calibrate([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
    0.5
    >>> calibrate([0.1, 0.2, 0.3, 0.4, 0.6, 0.9], [False, False, False, False, True, True])
    0.5
    >>> calibrate([0.3, 0.4], [False, False])
    0.7
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if len(s) == 0:
        msg = "cannot calibrate a threshold on no rows"
        raise ValueError(msg)
    distinct = np.unique(s)
    above = (distinct[-1] + 1.0) / 2 if distinct[-1] < 1.0 else np.nextafter(1.0, 2.0)
    candidates = np.concatenate(([distinct[0] / 2], (distinct[:-1] + distinct[1:]) / 2, [above]))
    # Rows admitted at a candidate are those scoring at or above it; count errors with cumulative sums.
    order = np.argsort(s, kind="mergesort")
    sorted_s, sorted_y = s[order], y[order]
    below_true = np.concatenate(([0], np.cumsum(sorted_y)))
    below_false = np.concatenate(([0], np.cumsum(~sorted_y)))
    cut = np.searchsorted(sorted_s, candidates, side="left")
    positives, negatives = int(sorted_y.sum()), int((~sorted_y).sum())
    weight_true = 1.0 / positives if positives else 0.0
    weight_false = 1.0 / negatives if negatives else 0.0
    # Rows below a candidate are rejected: true negatives there, true positives above it.
    correct = weight_false * below_false[cut] + weight_true * (positives - below_true[cut])
    correct = np.round(correct, 12)
    best = np.flatnonzero(correct == correct.max())
    return float(candidates[best[len(best) // 2]])


def at_threshold(scores: Sequence[float], labels: Sequence[bool], threshold: float) -> dict[str, float | int | None]:
    """Accuracy, balanced accuracy, precision and recall of admitting rows scoring at or above `threshold`.

    Precision is None when nothing is admitted, recall None when nothing is true,
    balanced accuracy None unless both classes are present.

    >>> at_threshold([0.9, 0.6, 0.2], [True, False, False], 0.5)
    {'threshold': 0.5, 'rows': 3, 'admitted': 2, 'accuracy': 0.6666666666666666, 'balanced_accuracy': 0.75, 'precision': 0.5, 'recall': 1.0}
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    admitted = s >= threshold
    hits = int((admitted & y).sum())
    rejected = int((~admitted & ~y).sum())
    both = y.any() and (~y).any()
    return {
        "threshold": float(threshold),
        "rows": len(s),
        "admitted": int(admitted.sum()),
        "accuracy": float((admitted == y).mean()) if len(s) else None,
        "balanced_accuracy": (hits / int(y.sum()) + rejected / int((~y).sum())) / 2 if both else None,
        "precision": hits / int(admitted.sum()) if admitted.any() else None,
        "recall": hits / int(y.sum()) if y.any() else None,
    }
