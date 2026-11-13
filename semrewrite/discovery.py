"""Purpose: semantic rule discovery: for a goal predicate, retrieve candidate premises
of the same arity, ask a judge whether each entails the goal under every
argument mapping, admit what the admission policy accepts, and return the
rewrites, counting the model work spent.

For a binary goal parent with candidate mother, the judge is asked whether
"A is the mother of B" entails "A is a parent of B" (mapping (0, 1)) and
whether "B is the mother of A" does (mapping (1, 0)); the goal against itself
under each non-identity mapping asks whether the goal is symmetric, as spouse
is. An expansion of an n-ary goal with k candidates asks k·n! + n! - 1
proposals, 2k + 1 for binary goals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from semrewrite.admission import Threshold
from semrewrite.model import Key, Predicate, Proposal, Rewrite, identity, mappings
from semrewrite.opinion import Opinion
from semrewrite.protocols import Admission, Judge, Retriever, calibrated
from semrewrite.trace import Expanded, Remembered, Tracer


class ScoreCache:
    """A judge wrapper that remembers the judge's opinion of each proposal, by both predicates' names and sentences
    and the mapping, and gives a proposal's score as that opinion's belief.

    Strategies compared on one judge share its model calls while each still
    counts every proposal it asked for: `logical` counts proposals asked,
    `computed` those the judge actually judged. The key holds the names as well
    as the sentences because a judge may read either: the oracle judges by
    name, so mum and mom glossed alike can still differ in what they entail.
    """

    def __init__(self, judge: Judge) -> None:
        self.judge = judge
        self._opinions: dict[tuple[str, str, str, str, tuple[int, ...]], Opinion] = {}
        self.logical = 0
        self.computed = 0

    @property
    def threshold(self) -> float:
        """The wrapped judge's admission threshold."""
        return self.judge.threshold

    @staticmethod
    def _key(p: Proposal) -> tuple[str, str, str, str, tuple[int, ...]]:
        return (p.premise.name, p.premise.gloss, p.conclusion.name, p.conclusion.gloss, p.mapping)

    def scores(self) -> list[float]:
        """Every score computed so far, one per distinct proposal."""
        return [o.belief for o in self._opinions.values()]

    def _judged(self, proposals: Sequence[Proposal]) -> list[Opinion]:
        """The judge's own opinions of `proposals`, asking it only for those never seen."""
        self.logical += len(proposals)
        missing = list({self._key(p): p for p in proposals if self._key(p) not in self._opinions}.values())
        if missing:
            self.computed += len(missing)
            for p, o in zip(missing, self.judge.opinions(missing), strict=True):
                if not isinstance(o, Opinion):
                    msg = f"judge {self.judge.name} gave {o!r} for {p.key}; a judge gives Opinion values"
                    raise TypeError(msg)
                self._opinions[self._key(p)] = o
        return [self._opinions[self._key(p)] for p in proposals]

    def score(self, proposals: Sequence[Proposal]) -> list[float]:
        """Scores for `proposals`: the belief of the judge's opinion of each."""
        return [o.belief for o in self._judged(proposals)]

    def opinions(self, proposals: Sequence[Proposal]) -> list[Opinion]:
        """The judge's opinion of each proposal at the judge's base rate, discounted by the judge's trust (Jøsang
        2016, Eq. 14.6), the opinion an answer's belief weighs."""
        return calibrated(self.judge, self._judged(proposals))


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
    memo, which is what makes rewriting terminate under tables; without it
    every ask retrieves and judges again, as a rewriter that rediscovers on
    every reduction step does. The memo is keyed on the goal and the judge's
    threshold at the time, so a discoverer shared across queries admits afresh
    once calibration moves the threshold; the score cache makes that free of
    model calls.
    """

    def __init__(
        self,
        catalogue: Mapping[str, Predicate],
        retriever: Retriever | None,
        scorer: ScoreCache,
        *,
        k: int = 8,
        memo: bool = True,
        admission: Admission | None = None,
        trace: Tracer | None = None,
    ) -> None:
        self.catalogue = dict(catalogue)
        self.retriever = retriever
        self.scorer = scorer
        self.k = k
        self.memo = memo
        self.admission: Admission = admission if admission is not None else Threshold()
        # Receives an Expanded event per expansion and a Remembered event per ask the memo answers.
        self.trace = trace
        self.stats = DiscoveryStats()
        self._memo: dict[tuple[str, float], tuple[Rewrite, ...]] = {}

    def _among(self, goal: str) -> dict[str, Predicate]:
        """Every other predicate of the goal's arity, which retrieval ranks."""
        target = self.catalogue[goal]
        return {n: p for n, p in self.catalogue.items() if n != goal and p.arity == target.arity}

    def candidates(self, goal: str, among: Mapping[str, Predicate] | None = None) -> list[tuple[str, float]]:
        """The k retrieved neighbours of `goal` of its arity with their retrieval scores, best first; without a
        retriever, every predicate of its arity by name, each scored 1.0. `among` is the goal's same-arity
        predicates when the caller has them."""
        self.stats.retrievals += 1
        among = self._among(goal) if among is None else among
        if self.retriever is None:
            return [(name, 1.0) for name in sorted(among)]
        found = self.retriever.top_k(self.catalogue[goal], among, self.k)
        stray = [n for n, _ in found if n not in among]
        if stray:
            msg = f"retriever proposed {stray} for {goal}, outside the same-arity catalogue"
            raise ValueError(msg)
        return found

    def proposals(self, goal: str, candidates: Sequence[str] | None = None) -> list[Proposal]:
        """Every proposal one expansion of `goal` asks: each candidate, retrieved unless given, under each mapping,
        and the goal's own symmetries."""
        target = self.catalogue[goal]
        names = [name for name, _ in self.candidates(goal)] if candidates is None else candidates
        out = [Proposal(self.catalogue[q], target, m) for q in names for m in mappings(target.arity)]
        out += [Proposal(target, target, m) for m in mappings(target.arity) if m != identity(target.arity)]
        return out

    # Time: one retrieval plus one judge call over k·n! + n! - 1 proposals per expansion,
    # twice that with a policy that needs converse scores.
    def rewrites(self, goal: str) -> tuple[Rewrite, ...]:
        """Admitted rewrites concluding `goal`, strongest first."""
        key = (goal, self.scorer.threshold)
        if self.memo and key in self._memo:
            if self.trace is not None:
                self.trace(Remembered(goal))
            return self._memo[key]
        if goal not in self.catalogue:
            msg = f"goal predicate {goal!r} has no gloss in the catalogue"
            raise KeyError(msg)
        self.stats.expansions += 1
        self.stats.expanded.add(goal)
        among = self._among(goal)
        kept = self.candidates(goal, among)
        proposals = self.proposals(goal, [name for name, _ in kept])
        scores = self.scorer.score(proposals)
        converse = self.scorer.score([p.converse() for p in proposals]) if self.admission.needs_converse else None
        self.stats.proposals += len(proposals) * (2 if converse is not None else 1)
        verdicts = self.admission.admit(proposals, scores, converse, self.scorer.threshold)
        admitted = sorted(
            (p.rewrite(s) for p, s, ok in zip(proposals, scores, verdicts, strict=True) if ok),
            key=lambda r: (-r.score, r.premise, r.mapping),
        )
        found = tuple(admitted)
        if self.memo:
            self._memo[key] = found
        if self.trace is not None:
            self.trace(Expanded(self.catalogue[goal], self.retriever.name if self.retriever is not None else None,
                                len(among), tuple(kept), self.scorer.judge.name, self.scorer.threshold,
                                tuple(proposals), tuple(scores), found))
        return found

    def keys(self) -> set[Key]:
        """Every rewrite key this discoverer has admitted so far."""
        return {rw.key for found in self._memo.values() for rw in found}
