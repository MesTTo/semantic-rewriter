"""Purpose: the oracle judge, which admits exactly a given set of implications, so
engine behaviour can be measured apart from judge error and tests can say
which rewrites a judge would admit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from semrewrite.model import Key, Mapping, Proposal, keys_of
from semrewrite.opinion import Opinion


@dataclass
class OracleJudge:
    """Certain belief for a proposal whose (premise, conclusion, mapping) is in `gold`, certain disbelief otherwise."""

    gold: frozenset[Key]
    name: str = "oracle"
    threshold: float = 0.5
    trust: float = 1.0
    base_rate: float = 0.5
    usage: dict[str, int] = field(default_factory=lambda: {"pairs": 0})

    @classmethod
    def of(cls, triples: Iterable[tuple[str, str, Mapping | bool]]) -> OracleJudge:
        """Build from (premise, conclusion, mapping) triples; a bool stands for the swap of a binary pair."""
        return cls(keys_of(triples))

    def opinions(self, proposals: Sequence[Proposal]) -> list[Opinion]:
        """(1, 0, 0) for a gold implication and (0, 1, 0) otherwise."""
        self.usage["pairs"] += len(proposals)
        return [Opinion(1.0, 0.0, 0.0) if p.key in self.gold else Opinion(0.0, 1.0, 0.0) for p in proposals]

    def close(self) -> None:
        """The oracle holds no model, so closing it releases nothing."""
