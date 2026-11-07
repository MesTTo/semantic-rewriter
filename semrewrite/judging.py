"""Purpose: the contract between the semantic rewriter and its direction judges.

A judge scores implication proposals between two binary predicates. The
rewriter asks, for a goal predicate P and a candidate Q, whether Q(x, y)
entails P(x, y), and whether Q(y, x) does, and admits a rewrite
(= (P $x $y) (Q $x $y)) or (= (P $x $y) (Q $y $x)) when the score reaches
the judge's threshold. Rewriting a P goal into a Q goal is sound only in
that direction: every Q answer is then a P answer, never the reverse.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Predicate:
    """A binary predicate with the sentence it states about its two arguments.

    `gloss` is a template over `{x}` and `{y}`, the predicate's first and
    second argument: mother with gloss "{x} is the mother of {y}" says the
    fact (mother alice tom) means "alice is the mother of tom".
    """

    name: str
    gloss: str

    def __post_init__(self) -> None:
        if "{x}" not in self.gloss or "{y}" not in self.gloss:
            msg = f"gloss of {self.name!r} must mention both {{x}} and {{y}}: {self.gloss!r}"
            raise ValueError(msg)

    def sentence(self, x: str, y: str) -> str:
        """The gloss with its arguments filled, such as "A is the mother of B"."""
        return self.gloss.format(x=x, y=y)


@dataclass(frozen=True, slots=True)
class Proposal:
    """The claim that `premise` entails `conclusion` over shared arguments.

    Unswapped: premise(x, y) entails conclusion(x, y), as mother entails parent.
    Swapped: premise(y, x) entails conclusion(x, y), as child_of(tom, alice)
    entails parent(alice, tom).
    """

    premise: Predicate
    conclusion: Predicate
    swapped: bool = False

    def sentences(self, x: str = "A", y: str = "B") -> tuple[str, str]:
        """The premise and conclusion sentences over placeholder arguments."""
        premise = self.premise.sentence(y, x) if self.swapped else self.premise.sentence(x, y)
        return premise, self.conclusion.sentence(x, y)

    @property
    def key(self) -> tuple[str, str, bool]:
        """Names-only identity, for caches and gold lookups."""
        return (self.premise.name, self.conclusion.name, self.swapped)


@runtime_checkable
class Judge(Protocol):
    """Scores proposals; the rewriter admits one whose score reaches `threshold`.

    `usage` counts the model work spent so far under named units, such as
    {"pairs": 120, "batches": 4} for a cross-encoder or
    {"requests": 3, "questions": 60} for an API judge, so the experiment can
    report semantic model calls per query.
    """

    name: str
    threshold: float
    usage: dict[str, int]

    def score(self, proposals: Sequence[Proposal]) -> list[float]:
        """A score in [0, 1] per proposal, in order, higher meaning more likely to hold."""
        ...


@dataclass
class OracleJudge:
    """A judge that knows the gold implications, so engine behaviour can be
    measured apart from judge error. `gold` holds (premise, conclusion,
    swapped) name triples; reflexive proposals never reach a judge.
    """

    gold: frozenset[tuple[str, str, bool]]
    name: str = "oracle"
    threshold: float = 0.5
    usage: dict[str, int] = field(default_factory=lambda: {"pairs": 0})

    @classmethod
    def of(cls, triples: Iterable[tuple[str, str, bool]]) -> OracleJudge:
        """Build from any iterable of gold triples."""
        return cls(frozenset(triples))

    def score(self, proposals: Sequence[Proposal]) -> list[float]:
        """1.0 for a gold implication and 0.0 otherwise."""
        self.usage["pairs"] += len(proposals)
        return [1.0 if p.key in self.gold else 0.0 for p in proposals]
