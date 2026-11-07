"""Purpose: semantic rule discovery: for a goal predicate, retrieve candidate
predicates, ask a judge whether each entails the goal in either argument
order, and return the admitted rewrites, counting the model work spent.

A rewrite (= (parent $X $Y) (mother $X $Y)) is admitted when the judge scores
"mother(A, B) entails parent(A, B)" at or above its threshold; a swapped one,
(= (parent $X $Y) (child_of $Y $X)), when it scores "child_of(B, A) entails
parent(A, B)". The goal against itself swapped asks whether the goal is
symmetric, as spouse is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from semrewrite.judging import Judge, Predicate, Proposal
from semrewrite.kb import Rule, implication_rule
from semrewrite.retrieve import Retriever


@dataclass(frozen=True, slots=True)
class Rewrite:
    """An admitted implication premise => conclusion, read as a rewrite of conclusion goals."""

    conclusion: str
    premise: str
    swapped: bool
    score: float

    @property
    def key(self) -> tuple[str, str, bool]:
        """The implication triple, as semrewrite.judging.Proposal.key."""
        return (self.premise, self.conclusion, self.swapped)

    def rule(self) -> Rule:
        """The Horn rule the rewrite states."""
        return implication_rule(self.premise, self.conclusion, self.swapped)

    def metta(self) -> str:
        """The rewrite as a native equation, such as (= (parent $X $Y) (child_of $Y $X))."""
        return self.rule().metta()


class ScoreCache:
    """A judge wrapper that remembers scores, so strategies compared on one
    judge share its forward passes while each still counts every proposal it
    asked for. `logical` counts proposals asked, `computed` those the judge saw.
    """

    def __init__(self, judge: Judge) -> None:
        self.judge = judge
        self._scores: dict[tuple[str, str, bool], float] = {}
        self.logical = 0
        self.computed = 0

    @property
    def threshold(self) -> float:
        """The wrapped judge's admission threshold."""
        return self.judge.threshold

    @staticmethod
    def _key(p: Proposal) -> tuple[str, str, bool]:
        return (p.premise.gloss, p.conclusion.gloss, p.swapped)

    def scores(self) -> list[float]:
        """Every score computed so far, one per distinct proposal."""
        return list(self._scores.values())

    def score(self, proposals: Sequence[Proposal]) -> list[float]:
        """Scores for `proposals`, computing only those never seen."""
        self.logical += len(proposals)
        missing = list({self._key(p): p for p in proposals if self._key(p) not in self._scores}.values())
        if missing:
            self.computed += len(missing)
            for p, s in zip(missing, self.judge.score(missing), strict=True):
                self._scores[self._key(p)] = float(s)
        return [self._scores[self._key(p)] for p in proposals]


@dataclass
class DiscoveryStats:
    """Model work one discoverer spent: expansions run, proposals asked, retrievals made."""

    expansions: int = 0
    proposals: int = 0
    retrievals: int = 0
    expanded: set[str] = field(default_factory=set)


class Discoverer:
    """Admitted rewrites per goal predicate over a frozen catalogue.

    With `memo` each predicate is expanded once and later asks answer from the
    memo; without it every ask retrieves and judges again, which is what a
    rewriter that rediscovers on every reduction step does.
    """

    def __init__(
        self,
        catalogue: Mapping[str, Predicate],
        retriever: Retriever | None,
        scorer: ScoreCache,
        *,
        k: int,
        memo: bool,
    ) -> None:
        self.catalogue = dict(catalogue)
        self.retriever = retriever
        self.scorer = scorer
        self.k = k
        self.memo = memo
        self.stats = DiscoveryStats()
        self._memo: dict[str, tuple[Rewrite, ...]] = {}

    def candidates(self, goal: str) -> list[str]:
        """The k retrieved neighbours of `goal`, or the whole catalogue when there is no retriever."""
        self.stats.retrievals += 1
        if self.retriever is None:
            return sorted(name for name in self.catalogue if name != goal)
        return [name for name, _ in self.retriever.top_k(self.catalogue[goal], self.catalogue, self.k)]

    # Time: one retrieval plus one judge call over 2k + 1 proposals per expansion.
    def rewrites(self, goal: str) -> tuple[Rewrite, ...]:
        """Admitted rewrites concluding `goal`, strongest first."""
        if self.memo and goal in self._memo:
            return self._memo[goal]
        if goal not in self.catalogue:
            msg = f"goal predicate {goal!r} has no gloss in the catalogue"
            raise KeyError(msg)
        self.stats.expansions += 1
        self.stats.expanded.add(goal)
        target = self.catalogue[goal]
        proposals = [
            Proposal(self.catalogue[q], target, swapped)
            for q in self.candidates(goal)
            for swapped in (False, True)
        ]
        proposals.append(Proposal(target, target, swapped=True))
        scores = self.scorer.score(proposals)
        self.stats.proposals += len(proposals)
        admitted = sorted(
            (
                Rewrite(goal, p.premise.name, p.swapped, s)
                for p, s in zip(proposals, scores, strict=True)
                if s >= self.scorer.threshold
            ),
            key=lambda r: (-r.score, r.premise, r.swapped),
        )
        found = tuple(admitted)
        if self.memo:
            self._memo[goal] = found
        return found
