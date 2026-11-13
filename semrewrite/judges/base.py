"""Purpose: the part every model-backed direction judge shares: proposals become
premise and conclusion sentences under argument fillers, the model's outputs
become opinions in input order, a probability as its dogmatic opinion, usage is
counted, and the model is released on close.

A subclass implements `_score_sentences`, mapping (premise sentence, conclusion
sentence) pairs to probabilities in [0, 1], or `_opinions` where its model says
more than a probability, as an NLI model's three classes do; and `_release`
when it holds a model.
"""

from __future__ import annotations

import gc
import math
import sys
from collections.abc import Sequence
from types import TracebackType
from typing import Self

from semrewrite.model import LETTERS as _LETTERS
from semrewrite.model import Proposal
from semrewrite.opinion import Opinion

Fillers = tuple[str, ...]
"""The texts standing for the conclusion's arguments in order, such as ("A", "B"); at least the arity many."""

LETTERS: Fillers = _LETTERS



def sentence_pairs(proposals: Sequence[Proposal], fillers: Sequence[Fillers]) -> list[tuple[str, str]]:
    """Each proposal's (premise, conclusion) sentences under its own fillers.

    >>> from semrewrite.model import Predicate
    >>> mother = Predicate("mother", "{x} is the mother of {y}")
    >>> parent = Predicate("parent", "{x} is a parent of {y}")
    >>> sentence_pairs([Proposal(mother, parent, (0, 1))], [("the first person", "the second person")])
    [('the first person is the mother of the second person', 'the first person is a parent of the second person')]
    """
    if len(fillers) != len(proposals):
        msg = f"{len(fillers)} filler tuples for {len(proposals)} proposals"
        raise ValueError(msg)
    for p, f in zip(proposals, fillers, strict=True):
        if len(f) < p.conclusion.arity:
            msg = f"{len(f)} fillers for the {p.conclusion.arity} arguments of {p.conclusion.name}"
            raise ValueError(msg)
    return [p.sentences(f) for p, f in zip(proposals, fillers, strict=True)]


class SentenceJudge:
    """A judge giving opinions of proposals through the sentences they state.

    `usage["pairs"]` counts every proposal judged and `usage["calls"]` every
    non-empty `opinions` call; nothing is memoised, so a caller asking twice is
    charged twice, which is what the engine's call counts measure. `trust` is
    complete, 1, and `base_rate` one half, until the judge factory sets the
    calibrated ones.
    """

    def __init__(self, name: str, threshold: float, fillers: Fillers = LETTERS) -> None:
        self.name = name
        self.threshold = threshold
        self.trust = 1.0
        self.base_rate = 0.5
        self.fillers = fillers
        self.usage: dict[str, int] = {"pairs": 0, "calls": 0}

    def opinions(self, proposals: Sequence[Proposal], fillers: Sequence[Fillers] | None = None) -> list[Opinion]:
        """An opinion per proposal, in order, that its premise entails its conclusion.

        `fillers`, one pair per proposal, overrides the judge's own fillers, so an
        evaluation can fill each row's arguments with its typed nouns or instances.
        """
        chosen = fillers if fillers is not None else [self.fillers] * len(proposals)
        pairs = sentence_pairs(proposals, chosen)
        if not pairs:
            return []
        opinions = list(self._opinions(pairs))
        if len(opinions) != len(pairs):
            msg = f"{self.name} returned {len(opinions)} opinions for {len(pairs)} proposals"
            raise RuntimeError(msg)
        self.usage["pairs"] += len(pairs)
        self.usage["calls"] += 1
        return opinions

    def score(self, proposals: Sequence[Proposal], fillers: Sequence[Fillers] | None = None) -> list[float]:
        """A score per proposal, in order: the belief of the judge's opinion that its premise entails its conclusion."""
        return [o.belief for o in self.opinions(proposals, fillers)]

    def _opinions(self, pairs: Sequence[tuple[str, str]]) -> Sequence[Opinion]:
        """The model's probability p for each pair as the dogmatic opinion (p, 1 - p, 0), a probability's equivalent
        opinion (Jøsang 2016, Table 3.2). The uncertainty a probability leaves unstated enters when the judge's
        trust discounts the opinion."""
        scores = [float(s) for s in self._score_sentences(pairs)]
        bad = [s for s in scores if not (0.0 <= s <= 1.0) or math.isnan(s)]
        if bad:
            msg = f"{self.name} returned scores outside [0, 1]: {bad[:5]}"
            raise RuntimeError(msg)
        return [Opinion(s, 1.0 - s, 0.0) for s in scores]

    def _score_sentences(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        raise NotImplementedError

    def _release(self) -> None:
        """Drop the model; a judge holding none has nothing to do."""

    def close(self) -> None:
        """Release the model and the GPU memory it held, so the next judge can load.

        Only a process that imported torch can hold CUDA memory, so a judge on an
        install without torch, such as Jev's extra alone, closes without it.
        """
        self._release()
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None) -> None:
        self.close()
