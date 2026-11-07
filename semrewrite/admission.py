"""Purpose: admission policies, which decide which scored proposals become rewrites,
and the calibration of their parameters on labelled pairs.

Threshold admits a proposal whose score reaches the judge's threshold.
Directional also asks the judge about each proposal's converse and admits a
proposal only when, in addition, score + slack >= converse score: a judge that
reads two related predicates as entailing each other scores both directions
high, and comparing the two cancels that shared relatedness. A true
equivalence, both directions near equal, survives; a one-way implication read
backwards does not. A slack of 1 or more admits what Threshold admits.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from semrewrite.model import Proposal


@dataclass(frozen=True)
class Threshold:
    """Admit a proposal whose score reaches the judge's threshold."""

    needs_converse: bool = False

    def admit(self, proposals: Sequence[Proposal], scores: Sequence[float],
              converse: Sequence[float] | None, threshold: float) -> list[bool]:
        return [s >= threshold for s in scores]


@dataclass(frozen=True)
class Directional:
    """Admit a proposal that reaches the threshold and that its converse does not outscore by more than `slack`."""

    slack: float
    needs_converse: bool = True

    def admit(self, proposals: Sequence[Proposal], scores: Sequence[float],
              converse: Sequence[float] | None, threshold: float) -> list[bool]:
        if converse is None or len(converse) != len(scores):
            msg = "direction-aware admission needs one converse score per proposal"
            raise ValueError(msg)
        return [s >= threshold and s + self.slack >= c for s, c in zip(scores, converse, strict=True)]


def balanced_accuracy(admitted: Sequence[bool], labels: Sequence[bool]) -> float:
    """Mean of the true-positive and true-negative rates; 0.5 for a constant answer."""
    tp = sum(a and y for a, y in zip(admitted, labels, strict=True))
    tn = sum(not a and not y for a, y in zip(admitted, labels, strict=True))
    pos = sum(labels)
    neg = len(labels) - pos
    tpr = tp / pos if pos else 1.0
    tnr = tn / neg if neg else 1.0
    return (tpr + tnr) / 2


def calibrate_threshold(scores: Sequence[float], labels: Sequence[bool]) -> tuple[float, float]:
    """The threshold maximising balanced accuracy on labelled scores, the higher threshold on ties, with that accuracy.

    Candidate thresholds are the distinct scores themselves, so the cut always
    falls exactly on an observed score.
    """
    best = (0.5, -1.0)
    for t in sorted(set(scores)):
        acc = balanced_accuracy([s >= t for s in scores], labels)
        if acc > best[1] or (acc == best[1] and t > best[0]):
            best = (t, acc)
    return best


def calibrate_slack(
    forward: Sequence[float], backward: Sequence[float], labels: Sequence[bool], threshold: float,
    grid: Sequence[float] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0),
) -> tuple[float, dict[float, float]]:
    """The slack maximising balanced accuracy on labelled rows at `threshold`, the larger slack on ties.

    Returns the chosen slack and the balanced accuracy at every slack tried.
    """
    curve = {}
    for slack in grid:
        admitted = [f >= threshold and f + slack >= b for f, b in zip(forward, backward, strict=True)]
        curve[slack] = balanced_accuracy(admitted, labels)
    best = max(grid, key=lambda s: (curve[s], s))
    return best, curve
