"""Purpose: candidate retrieval, the step before a judge sees any proposal:
EmbeddingRetriever ranks catalogue predicates by gloss-embedding similarity,
JudgeRetriever ranks them by a judge's entailment score, the first stage of a
cascade for large catalogues, and AllCandidates proposes every predicate, which
suits small catalogues.

Each predicate's gloss is filled with neutral letters ("A is the mother of B")
and embedded once; a goal's candidates are the k predicates of its arity
nearest it by cosine. Retrieval is symmetric and says nothing about direction;
the judge decides which way, and how the arguments map.
"""

from __future__ import annotations

import gc
import sys
from collections.abc import Iterable, Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Self

from semrewrite.model import LETTERS, Predicate, Proposal, mappings

# numpy names only a type here: the vectors come from the embedding model, so
# a core install can import AllCandidates without numpy.
if TYPE_CHECKING:
    import numpy as np

    from semrewrite.discovery import ScoreCache
    from semrewrite.protocols import Judge

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingRetriever:
    """Top-k gloss neighbours over a catalogue, with embedding work counted in `usage`.

    Vectors are cached by gloss sentence, so a predicate name reused with a
    different gloss in another knowledge base gets its own vector. close()
    releases the model, as a judge's does, and the cached vectors still serve
    the glosses already embedded.
    """

    def __init__(self, model: str = DEFAULT_MODEL, fillers: tuple[str, ...] = LETTERS) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.fillers = fillers
        self._model = SentenceTransformer(model)
        self._vectors: dict[str, np.ndarray] = {}
        self.usage: dict[str, int] = {"encoded": 0, "queries": 0}

    def _sentence(self, predicate: Predicate) -> str:
        return predicate.sentence(*self.fillers[: predicate.arity])

    # Time: one batched forward pass over the glosses not yet cached.
    def index(self, predicates: Iterable[Predicate]) -> None:
        """Embed every gloss not yet cached, in one batch."""
        fresh = sorted({self._sentence(p) for p in predicates} - set(self._vectors))
        if not fresh:
            return
        if self._model is None:
            msg = f"the retriever was closed, so it cannot embed {len(fresh)} new glosses; make a new one"
            raise RuntimeError(msg)
        vectors = self._model.encode(fresh, normalize_embeddings=True, convert_to_numpy=True)
        self._vectors.update(zip(fresh, vectors, strict=True))
        self.usage["encoded"] += len(fresh)

    # Time: |among| dot products of the embedding width, then a sort of |among|.
    def top_k(self, goal: Predicate, among: Mapping[str, Predicate], k: int) -> list[tuple[str, float]]:
        """The k predicates of `among` of the goal's arity, `goal` excluded, nearest `goal` by cosine, nearest first."""
        self.index([goal, *among.values()])
        self.usage["queries"] += 1
        anchor = self._vectors[self._sentence(goal)]
        scored = [
            (name, float(anchor @ self._vectors[self._sentence(p)]))
            for name, p in among.items()
            if name != goal.name and p.arity == goal.arity
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:k]

    def close(self) -> None:
        """Release the model and the GPU memory it held; only a process that imported torch can hold any."""
        self._model = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None) -> None:
        self.close()


class AllCandidates:
    """Every predicate of the goal's arity, by name; right when the catalogue is small enough to judge whole."""

    def top_k(self, goal: Predicate, among: Mapping[str, Predicate], k: int) -> list[tuple[str, float]]:
        return [(n, 1.0) for n in sorted(among) if n != goal.name and among[n].arity == goal.arity]


class JudgeRetriever:
    """Candidates ranked by a judge: each predicate of the goal's arity is scored as a premise of the goal under every
    argument mapping, ranked by its best score, and the k best are kept.

    With a local cross-encoder as the judge this is the first stage of a
    cascade: it reads the whole catalogue, and the admitting judge sees only the
    k candidates it keeps. Embedding similarity is symmetric and lexical, so on
    a catalogue of Wikidata's 1,790 entity-valued properties it ranked author
    327th and notable work 794th among the premises of "{x} created {y}", where
    nli-deberta-large ranked them 26th and 2nd. Scores go through a ScoreCache,
    so a goal's candidates are scored once however often it is expanded.
    """

    def __init__(self, judge: Judge | ScoreCache) -> None:
        from semrewrite.discovery import ScoreCache

        self.scorer = judge if isinstance(judge, ScoreCache) else ScoreCache(judge)

    # Time: |among|·n! judge scores the first time a goal is ranked, n its arity, then |among| maxima and a sort.
    def top_k(self, goal: Predicate, among: Mapping[str, Predicate], k: int) -> list[tuple[str, float]]:
        """The k predicates of `among` of the goal's arity, `goal` excluded, most likely to entail it first."""
        candidates = [(name, p) for name, p in among.items() if name != goal.name and p.arity == goal.arity]
        maps = list(mappings(goal.arity))
        scores = self.scorer.score([Proposal(p, goal, m) for _, p in candidates for m in maps])
        best = [(name, max(scores[i * len(maps):(i + 1) * len(maps)])) for i, (name, _) in enumerate(candidates)]
        best.sort(key=lambda item: (-item[1], item[0]))
        return best[:k]

    def close(self) -> None:
        """Release the ranking judge's model, as EmbeddingRetriever.close releases its own."""
        close = getattr(self.scorer.judge, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None) -> None:
        self.close()
