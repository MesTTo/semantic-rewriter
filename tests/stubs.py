"""Purpose: the stand-in judge the rewriter's tests share: the dogmatic opinion of a
probability looked up by proposal key in a fixed table, so a test states which
rewrites a judge would admit and at what score.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

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
