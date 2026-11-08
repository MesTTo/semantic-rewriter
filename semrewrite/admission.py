"""Purpose: admission policies, which decide which scored proposals become rewrites,
and the calibration of the direction-aware policy's slack on labelled pairs; the
threshold's calibration is judges.metrics.calibrate.

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
    """Mean of the true-positive and true-negative rates; 0.5 for a constant answer.

    With one class absent it is the rate on the class present, plain accuracy,
    as judges.metrics.calibrate scores it.
    """
    if not labels:
        msg = "balanced accuracy needs at least one row"
        raise ValueError(msg)
    tp = sum(a and y for a, y in zip(admitted, labels, strict=True))
    tn = sum(not a and not y for a, y in zip(admitted, labels, strict=True))
    pos = sum(labels)
    neg = len(labels) - pos
    rates = [r for r, present in ((tp / pos if pos else 0.0, pos), (tn / neg if neg else 0.0, neg)) if present]
    return sum(rates) / len(rates)


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
