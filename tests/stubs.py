"""Purpose: the stand-ins the rewriter's tests share: a judge giving the dogmatic opinion of a
probability looked up by proposal key in a fixed table, so a test states which
rewrites a judge would admit and at what score; sentence judges reading each
pair of sentences as a fixed table says, which the claim check's tests read
statements against claims with; and an embedder giving cosines from a table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from semrewrite.judges.base import SentenceJudge
from semrewrite.model import Proposal
from semrewrite.opinion import Opinion


@dataclass
class TableJudge:
    """Opinions (p, 1 - p, 0) from a fixed table of probabilities p keyed by proposal key; 0 for anything else."""

    table: dict
    name: str = "stub"
    threshold: float = 0.5
    trust: float = 1.0
    base_rate: float = 0.5
    usage: dict[str, int] = field(default_factory=lambda: {"pairs": 0})

    def opinions(self, proposals: Sequence[Proposal]) -> list[Opinion]:
        self.usage["pairs"] += len(proposals)
        return [Opinion(s, 1.0 - s, 0.0) for s in (self.table.get(p.key, 0.0) for p in proposals)]

    def close(self) -> None:
        """The table holds nothing to release."""


class SentenceTableJudge(SentenceJudge):
    """Reads each (premise, conclusion) pair of sentences as the opinion `table` gives it, and a pair it lacks as
    neutral, (0, 0, 1); its disbelief is contradiction, as an NLI judge's is. Proposals are read through their
    sentences under the letters A, B, ..., as every sentence judge reads them."""

    reads_contradiction: ClassVar[bool] = True

    def __init__(self, table: Mapping[tuple[str, str], Opinion], *, threshold: float = 0.5, trust: float = 1.0,
                 base_rate: float = 0.5, name: str = "sentence stub") -> None:
        super().__init__(name, threshold)
        self.table = dict(table)
        self.trust = trust
        self.base_rate = base_rate
        self.asked: list[tuple[str, str]] = []

    def _opinions(self, pairs: Sequence[tuple[str, str]]) -> list[Opinion]:
        self.asked += pairs
        return [self.table.get(pair, Opinion(0.0, 0.0, 1.0)) for pair in pairs]


class YesNoTableJudge(SentenceTableJudge):
    """A SentenceTableJudge whose disbelief says only that a conclusion does not follow, as a judge of how likely it
    follows gives it."""

    reads_contradiction: ClassVar[bool] = False


@dataclass
class TableEmbedder:
    """Cosines between sentences looked up in a fixed table keyed by (anchor, sentence), and `default` for any pair
    it lacks, so a test states which statements lie near which claims."""

    table: Mapping[tuple[str, str], float]
    default: float = 0.0
    asked: list[str] = field(default_factory=list)

    def cosines(self, anchors: Sequence[str], sentences: Sequence[str]) -> list[list[float]]:
        self.asked += anchors
        return [[self.table.get((a, s), self.default) for s in sentences] for a in anchors]
