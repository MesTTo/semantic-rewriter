"""Purpose: tests of the admission policies through the discoverer: direction-aware
admission drops an implication read backwards and keeps an equivalence, slack 1
admits what the threshold admits, and calibration picks the separating slack.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from semrewrite.admission import Directional, Threshold, balanced_accuracy, calibrate_slack, calibrate_threshold
from semrewrite.discovery import Discoverer, ScoreCache
from semrewrite.model import Predicate, Proposal

CATALOGUE = {p.name: p for p in (
    Predicate("mother", "{x} is the mother of {y}"),
    Predicate("mum", "{x} is the mum of {y}"),
    Predicate("parent", "{x} is a parent of {y}"),
)}
SCORES = {
    ("parent", "mum", (0, 1)): 0.955, ("mum", "parent", (0, 1)): 0.998,
    ("mum", "mother", (0, 1)): 0.998, ("mother", "mum", (0, 1)): 0.996,
    ("mother", "parent", (0, 1)): 0.99, ("parent", "mother", (0, 1)): 0.97,
}


@dataclass
class StubJudge:
    """Scores from a fixed table keyed by proposal key; 0 for anything else."""

    table: dict
    name: str = "stub"
    threshold: float = 0.5
    usage: dict[str, int] = field(default_factory=lambda: {"pairs": 0})

    def score(self, proposals: Sequence[Proposal]) -> list[float]:
        self.usage["pairs"] += len(proposals)
        return [self.table.get(p.key, 0.0) for p in proposals]


def _admitted(policy, goal: str) -> set[str]:  # noqa: ANN001
    disc = Discoverer(CATALOGUE, None, ScoreCache(StubJudge(SCORES)), admission=policy)
    return {rw.premise for rw in disc.rewrites(goal)}


def test_a_direction_read_backwards_is_dropped_and_an_equivalence_survives() -> None:
    """mum => parent stays and parent => mum (0.955 against 0.998) goes; mum <=> mother (0.998, 0.996) survives both ways."""
    assert _admitted(Threshold(), "mum") == {"parent", "mother"}
    assert _admitted(Directional(slack=0.01), "mum") == {"mother"}
    assert _admitted(Directional(slack=0.01), "mother") == {"mum"}
    assert _admitted(Directional(slack=0.01), "parent") == {"mum", "mother"}


def test_slack_one_admits_what_the_threshold_admits() -> None:
    for goal in CATALOGUE:
        assert _admitted(Directional(slack=1.0), goal) == _admitted(Threshold(), goal)


def test_directional_admission_asks_each_converse_once_more() -> None:
    judge = StubJudge(SCORES)
    disc = Discoverer(CATALOGUE, None, ScoreCache(judge), admission=Directional(slack=0.01))
    disc.rewrites("parent")
    assert disc.stats.proposals == 2 * (2 * 2 + 1)


def test_calibration_picks_the_slack_and_threshold_that_separate_the_labels() -> None:
    forward = [0.95, 0.99, 0.97, 0.9]
    backward = [0.99, 0.2, 0.96, 0.1]
    labels = [False, True, True, True]
    slack, curve = calibrate_slack(forward, backward, labels, threshold=0.5)
    assert curve[1.0] == balanced_accuracy([True, True, True, True], labels)
    assert curve[0.02] == 1.0
    assert slack == max(s for s, v in curve.items() if v == 1.0)
    threshold, accuracy = calibrate_threshold([0.1, 0.4, 0.6, 0.9], [False, False, True, True])
    assert (threshold, accuracy) == (0.6, 1.0)
