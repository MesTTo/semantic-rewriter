"""Purpose: the oracle judge, which admits exactly a given set of implications, so
engine behaviour can be measured apart from judge error and tests can say
which rewrites a judge would admit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from semrewrite.model import Key, Mapping, Proposal, keys_of


@dataclass
class OracleJudge:
    """Scores 1.0 for a proposal whose (premise, conclusion, mapping) is in `gold` and 0.0 otherwise."""

    gold: frozenset[Key]
    name: str = "oracle"
    threshold: float = 0.5
    usage: dict[str, int] = field(default_factory=lambda: {"pairs": 0})

    @classmethod
    def of(cls, triples: Iterable[tuple[str, str, Mapping | bool]]) -> OracleJudge:
        """Build from (premise, conclusion, mapping) triples; a bool stands for the swap of a binary pair."""
        return cls(keys_of(triples))

    def score(self, proposals: Sequence[Proposal]) -> list[float]:
        """1.0 for a gold implication and 0.0 otherwise."""
        self.usage["pairs"] += len(proposals)
        return [1.0 if p.key in self.gold else 0.0 for p in proposals]
