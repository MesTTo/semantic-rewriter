"""Purpose: tests of direction-aware admission (semrewrite/directional.py) on a
stub judge with fixed scores: the converse it asks, the slack rule, an
equivalence surviving, and slack calibration on labelled rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from semrewrite.directional import DirectionalJudge, balanced_accuracy, calibrate_slack, reverse
from semrewrite.judging import Predicate, Proposal

MOTHER = Predicate("mother", "{x} is the mother of {y}")
MUM = Predicate("mum", "{x} is the mum of {y}")
PARENT = Predicate("parent", "{x} is a parent of {y}")


@dataclass
class StubJudge:
    """Scores from a fixed table keyed by (premise, conclusion, swapped); 0 for anything else."""

    table: dict[tuple[str, str, bool], float]
    name: str = "stub"
    threshold: float = 0.5
    usage: dict[str, int] = field(default_factory=lambda: {"pairs": 0})

    def score(self, proposals: Sequence[Proposal]) -> list[float]:
        self.usage["pairs"] += len(proposals)
        return [self.table.get(p.key, 0.0) for p in proposals]


def test_reverse_keeps_the_argument_order_flag() -> None:
    """The converse of child_of(y, x) => parent(x, y) is parent(y, x) => child_of(x, y)."""
    p = Proposal(MOTHER, PARENT, swapped=True)
    assert reverse(p).key == ("parent", "mother", True)


def test_a_direction_read_backwards_is_dropped_and_an_equivalence_survives() -> None:
    """parent => mum scored 0.955 against mum => parent 0.998 is dropped; mum <=> mother (0.998, 0.996) survives both ways."""
    stub = StubJudge({
        ("parent", "mum", False): 0.955, ("mum", "parent", False): 0.998,
        ("mum", "mother", False): 0.998, ("mother", "mum", False): 0.996,
    })
    judge = DirectionalJudge(stub, slack=0.01)
    scores = judge.score([Proposal(PARENT, MUM), Proposal(MUM, PARENT), Proposal(MUM, MOTHER), Proposal(MOTHER, MUM)])
    assert scores == [0.0, 0.998, 0.998, 0.996]
    assert stub.usage["pairs"] == 8


def test_slack_one_is_the_plain_judge() -> None:
    """With slack 1 no score is ever dropped."""
    stub = StubJudge({("parent", "mum", False): 0.955, ("mum", "parent", False): 0.998})
    assert DirectionalJudge(stub, slack=1.0).score([Proposal(PARENT, MUM)]) == [0.955]


def test_calibration_picks_the_slack_that_separates_the_labels() -> None:
    """Rows a direction check gets right make a small slack win; ties go to the larger slack."""
    forward = [0.95, 0.99, 0.97, 0.9]
    backward = [0.99, 0.2, 0.96, 0.1]
    labels = [False, True, True, True]
    slack, curve = calibrate_slack(forward, backward, labels, threshold=0.5)
    assert curve[1.0] == balanced_accuracy([True, True, True, True], labels)
    assert curve[0.02] == 1.0
    assert slack == max(s for s, v in curve.items() if v == 1.0)
