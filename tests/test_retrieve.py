"""Purpose: tests of candidate retrieval with a stub embedding model: the k nearest
predicates of the goal's arity come back nearest first, ties by name, each gloss is
embedded once, a closed retriever serves only what it embedded, and AllCandidates
proposes every predicate of the goal's arity.
"""

from __future__ import annotations

import sys
import types

import pytest

from semrewrite.model import Predicate
from semrewrite.retrieve import AllCandidates, EmbeddingRetriever


class _Vector(tuple):
    """A vector supporting the dot product the retriever takes with @."""

    def __matmul__(self, other: tuple) -> float:
        return sum(a * b for a, b in zip(self, other, strict=True))


class _StubModel:
    """Embeds each sentence as the fixed unit vector _VECTORS gives it, and records what it encodes."""

    def __init__(self, name: str) -> None:
        self.encoded: list[str] = []

    def encode(self, sentences, normalize_embeddings: bool, convert_to_numpy: bool):
        self.encoded += sentences
        return [_Vector(_VECTORS[s]) for s in sentences]


# Sentences as the retriever fills them, with letters for the arguments.
_VECTORS = {
    "A is the mother of B": (1.0, 0.0),
    "A is the mum of B": (0.99, 0.141),
    "A is a parent of B": (0.8, 0.6),
    "A is the boss of B": (0.6, 0.8),
    "A is near B": (0.6, 0.8),
    "A is tall": (1.0, 0.0),
}


@pytest.fixture
def retriever(monkeypatch: pytest.MonkeyPatch) -> EmbeddingRetriever:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _StubModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return EmbeddingRetriever()


CATALOGUE = {p.name: p for p in (
    Predicate("mother", "{x} is the mother of {y}"),
    Predicate("mum", "{x} is the mum of {y}"),
    Predicate("parent", "{x} is a parent of {y}"),
    Predicate("boss_of", "{x} is the boss of {y}"),
    Predicate("near", "{x} is near {y}"),
    Predicate("tall", "{x} is tall"),
)}


def test_nearest_of_the_goal_arity_first_ties_by_name(retriever: EmbeddingRetriever) -> None:
    got = retriever.top_k(CATALOGUE["mother"], CATALOGUE, 4)
    assert [name for name, _ in got] == ["mum", "parent", "boss_of", "near"]
    assert got[2][1] == got[3][1]
    assert [name for name, _ in retriever.top_k(CATALOGUE["mother"], CATALOGUE, 2)] == ["mum", "parent"]


def test_each_gloss_is_embedded_once(retriever: EmbeddingRetriever) -> None:
    retriever.top_k(CATALOGUE["mother"], CATALOGUE, 3)
    retriever.top_k(CATALOGUE["parent"], CATALOGUE, 3)
    assert retriever.usage == {"encoded": len(CATALOGUE), "queries": 2}


def test_a_closed_retriever_serves_only_what_it_embedded(retriever: EmbeddingRetriever) -> None:
    with retriever:
        retriever.top_k(CATALOGUE["mother"], CATALOGUE, 3)
    assert retriever._model is None
    assert retriever.top_k(CATALOGUE["parent"], CATALOGUE, 1)[0][0] in CATALOGUE
    with pytest.raises(RuntimeError, match="was closed"):
        retriever.top_k(Predicate("aunt", "{x} is the aunt of {y}"), CATALOGUE, 1)


def test_all_candidates_proposes_every_predicate_of_the_goal_arity() -> None:
    got = AllCandidates().top_k(CATALOGUE["mother"], CATALOGUE, 2)
    assert got == [(n, 1.0) for n in ("boss_of", "mum", "near", "parent")]
