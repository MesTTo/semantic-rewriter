"""Purpose: the measurements a judge evaluation reports over labelled rows:
ROC AUC, direction accuracy over one-way pairs, the dev-calibrated threshold,
accuracy, balanced accuracy, precision and recall at a threshold, the trust
fitted to a judge's opinions, and the log loss and Brier score of probabilities.

Scores and labels are parallel sequences; a row is admitted when its score
reaches the threshold, as the engine admits a rewrite. Unequal lengths raise
ValueError.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Sequence

from semrewrite.opinion import Opinion


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


def fit_trust(opinions: Sequence[Opinion], labels: Sequence[bool]) -> float:
    """The trust t in [0, 1] maximising the likelihood of the labels when each row holds with the projection of its
    opinion discounted by t, the analyst's trust in the judge (Jøsang 2016, Eq. 14.6).

    Discounting scales belief and disbelief by t and moves the rest to
    uncertainty, so a row's projection is linear in t, from the base rate at
    t = 0 to the opinion's own projection at t = 1; the log-likelihood is then
    concave in t, and its derivative, decreasing in t, is bisected for its
    root. A judge whose probability p reads as the dogmatic opinion (p, 1 - p, 0)
    holds at a + t (p - a): its score shrunk toward the base rate, worth
    W t / (1 - t) observations (Jøsang 2016, Eq. 3.11 with W = 2).

    Time: 2n opinion operations, then n per bisection step over 64 steps, n = len(opinions).

    >>> round(fit_trust([Opinion(0.9, 0.1, 0.0)] * 10, [True] * 9 + [False]), 6)
    1.0
    >>> round(fit_trust([Opinion(0.9, 0.1, 0.0)] * 10, [True] * 7 + [False] * 3), 6)
    0.5
    """
    if len(opinions) != len(labels):
        msg = f"{len(opinions)} opinions against {len(labels)} labels"
        raise ValueError(msg)
    if not opinions:
        msg = "cannot fit a trust on no rows"
        raise ValueError(msg)
    rows = [(o.discount(0.0).projection, o.discount(1.0).projection, bool(y)) for o, y in zip(opinions, labels)]

    def slope(t: float) -> float:
        total = 0.0
        for low, high, y in rows:
            if high == low:
                continue
            p = low + t * (high - low)
            # A projection on the bound its label contradicts, which rounding can reach just inside t = 1, has a
            # likelihood falling to or rising from zero: an infinite slope with the sign of the row's own.
            if y:
                total += (high - low) / p if p > 0.0 else math.copysign(math.inf, high - low)
            else:
                total -= (high - low) / (1.0 - p) if p < 1.0 else math.copysign(math.inf, high - low)
        return total

    if slope(0.0) <= 0.0:
        return 0.0
    # At t = 1 a row whose own projection is 0 or 1 against its label makes the slope minus infinity, so the end is
    # tested just inside it.
    if slope(math.nextafter(1.0, 0.0)) >= 0.0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low + high) / 2
        if slope(middle) > 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


# Time: 101 grid evaluations, then 60 golden-section steps, each 2n opinion operations, n = len(judged).
def fit_data_trust(judged: Sequence[Opinion], data: Sequence[Opinion], labels: Sequence[bool]) -> float:
    """The trust t in [0, 1] in a second source maximising the likelihood of the labels when each row holds with the
    projection of the judge's opinion fused with the source's opinion discounted by t (Jøsang 2016, Def. 12.5 and
    Eq. 14.6). At t = 0 the source is vacuous and the judge's opinion stands alone. Fusion weighs each opinion by the
    other's uncertainty, so the likelihood need not be concave in t: a grid of 101 points finds the best cell, and
    golden-section search refines t within its two neighbours.

    >>> judged = [Opinion(0.4, 0.4, 0.2)] * 4
    >>> data = [Opinion(0.9, 0.0, 0.1)] * 2 + [Opinion(0.0, 0.9, 0.1)] * 2
    >>> fit_data_trust(judged, data, [True, True, False, False]) > 0.9
    True
    >>> fit_data_trust(judged, data, [False, False, True, True])
    0.0
    """
    if not len(judged) == len(data) == len(labels):
        msg = f"{len(judged)} judge opinions, {len(data)} source opinions and {len(labels)} labels"
        raise ValueError(msg)
    if not judged:
        msg = "cannot fit a trust on no rows"
        raise ValueError(msg)
    rows = list(zip(judged, data, [bool(y) for y in labels], strict=True))

    def loss(t: float) -> float:
        total = 0.0
        for j, d, y in rows:
            p = j.fuse(d.discount(t)).projection
            q = p if y else 1.0 - p
            if q <= 0.0:
                return math.inf
            total -= math.log(q)
        return total

    grid = [loss(g / 100) for g in range(101)]
    best = min(range(101), key=lambda g: (grid[g], g))
    low, high = max(best - 1, 0) / 100, min(best + 1, 100) / 100
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        a, b = high - ratio * (high - low), low + ratio * (high - low)
        if loss(a) <= loss(b):
            high = b
        else:
            low = a
    middle = (low + high) / 2
    # The ends are candidates too: a source worth nothing should read exactly 0.
    return min((best / 100, middle), key=lambda t: (loss(t), t))


def probability_scores(probabilities: Sequence[float], labels: Sequence[bool]) -> dict[str, float]:
    """The mean log loss and Brier score of `probabilities` against the labels, both proper scoring rules, so a
    calibrated judge's opinions score better than shaded ones.

    >>> probability_scores([0.5, 0.5], [True, False])
    {'rows': 2, 'log_loss': 0.6931471805599453, 'brier': 0.25}
    """
    p, y = _rows(probabilities, labels)
    if not p:
        msg = "cannot score no rows"
        raise ValueError(msg)
    loss = -sum(math.log(q) if true else math.log(1.0 - q) for q, true in zip(p, y)) / len(p)
    brier = sum((q - true) ** 2 for q, true in zip(p, y)) / len(p)
    return {"rows": len(p), "log_loss": loss, "brier": brier}


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
