"""Purpose: a direction-aware admission policy wrapped around any judge: admit
"Q entails P" only when the judge does not score the reverse, "P entails Q",
higher by more than a slack calibrated on labelled dev pairs.

A judge that reads two related predicates as entailing each other scores both
directions high; comparing the two cancels that shared relatedness, which is
what the judges' direction accuracy measures. With slack s, a proposal keeps its
score when score(fwd) + s >= score(rev) and scores 0 otherwise, so a true
equivalence (both directions near equal) survives and a one-way implication
read backwards does not. Slack 1 or more is the plain judge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from semrewrite.judging import Judge, Proposal


def reverse(p: Proposal) -> Proposal:
    """The converse claim: conclusion entails premise over the same argument order."""
    return Proposal(p.conclusion, p.premise, p.swapped)


@dataclass
class DirectionalJudge:
    """`base` with direction-aware admission; scores each proposal and its converse in one batch."""

    base: Judge
    slack: float
    name: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = self.name or f"{self.base.name}+direction"
        self.usage = self.base.usage

    @property
    def threshold(self) -> float:
        """The base judge's admission threshold."""
        return self.base.threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self.base.threshold = value

    def score(self, proposals: Sequence[Proposal]) -> list[float]:
        """The base score where the converse does not outscore it by more than the slack, else 0."""
        n = len(proposals)
        both = self.base.score([*proposals, *(reverse(p) for p in proposals)])
        return [f if f + self.slack >= r else 0.0 for f, r in zip(both[:n], both[n:], strict=True)]

    def close(self) -> None:
        """Release the base judge's model."""
        close = getattr(self.base, "close", None)
        if callable(close):
            close()


def balanced_accuracy(admitted: Sequence[bool], labels: Sequence[bool]) -> float:
    """Mean of the true-positive and true-negative rates; 0.5 for a constant answer."""
    tp = sum(a and y for a, y in zip(admitted, labels, strict=True))
    tn = sum(not a and not y for a, y in zip(admitted, labels, strict=True))
    pos = sum(labels)
    neg = len(labels) - pos
    tpr = tp / pos if pos else 1.0
    tnr = tn / neg if neg else 1.0
    return (tpr + tnr) / 2


def calibrate_slack(
    forward: Sequence[float], backward: Sequence[float], labels: Sequence[bool], threshold: float,
    grid: Sequence[float] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0),
) -> tuple[float, dict[float, float]]:
    """The slack maximising balanced accuracy on dev rows at the base threshold, the larger slack on ties.

    Returns the chosen slack and the balanced accuracy at every slack tried.
    """
    curve = {}
    for slack in grid:
        admitted = [f >= threshold and f + slack >= b for f, b in zip(forward, backward, strict=True)]
        curve[slack] = balanced_accuracy(admitted, labels)
    best = max(grid, key=lambda s: (curve[s], s))
    return best, curve
