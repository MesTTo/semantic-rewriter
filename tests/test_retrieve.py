"""Purpose: tests of candidate retrieval with a stub embedding model: the k nearest
predicates of the goal's arity come back nearest first, ties by name, each gloss is
embedded once, a closed retriever serves only what it embedded, the claim check's
cosines come back a row per anchor, and AllCandidates proposes every predicate of
the goal's arity; and of JudgeRetriever with a stub judge,
which ranks each candidate by its best score over the argument mappings.
"""

from __future__ import annotations

import sys
import types

import pytest

from semrewrite.model import Predicate
from semrewrite.opinion import Opinion
from semrewrite.retrieve import AllCandidates, EmbeddingRetriever, JudgeRetriever


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


def test_cosines_give_each_anchor_its_row_embedding_each_sentence_once(retriever: EmbeddingRetriever) -> None:
    """The claim check's cosines: a row per anchor over the sentences, in order, each sentence embedded once."""
    pytest.importorskip(
        "numpy", reason="cosines multiplies the model's vectors with numpy, which the models extra installs"
    )
    rows = retriever.cosines(["A is the mother of B", "A is near B"], ["A is the mum of B", "A is near B"])
    # pytest.approx compares flat sequences only, so each row is compared alone.
    assert len(rows) == 2
    assert rows[0] == pytest.approx([0.99, 0.6])
    assert rows[1] == pytest.approx([0.6 * 0.99 + 0.8 * 0.141, 1.0])
    assert retriever.cosines(["A is near B"], []) == [[]]
    assert retriever.cosines([], ["A is near B"]) == []
    assert retriever.usage["encoded"] == 3


def test_all_candidates_proposes_every_predicate_of_the_goal_arity() -> None:
    got = AllCandidates().top_k(CATALOGUE["mother"], CATALOGUE, 2)
    assert got == [(n, 1.0) for n in ("boss_of", "mum", "near", "parent")]


# The stub judge's score for a premise under a mapping; any other proposal scores 0.
_JUDGED = {("mum", (0, 1)): 0.9, ("parent", (1, 0)): 0.95, ("parent", (0, 1)): 0.1, ("near", (0, 1)): 0.3}


class _StubJudge:
    """Judges a proposal by its premise's name and mapping, as the dogmatic opinion of a probability, and records how
    many it judged."""

    name, threshold, trust = "stub", 0.5, 1.0

    def __init__(self) -> None:
        self.scored = 0

    def opinions(self, proposals) -> list[Opinion]:
        self.scored += len(proposals)
        return [Opinion(s, 1.0 - s, 0.0) for s in (_JUDGED.get((p.premise.name, p.mapping), 0.0) for p in proposals)]

    def close(self) -> None:
        """The stub holds nothing to release."""


def test_the_judge_retriever_ranks_by_the_best_mapping_and_scores_each_candidate_once() -> None:
    judge = _StubJudge()
    retriever = JudgeRetriever(judge)
    got = retriever.top_k(CATALOGUE["mother"], CATALOGUE, 3)
    # parent's swapped reading scores 0.95, so it outranks mum; tall has another arity and mother is the goal.
    assert got == [("parent", 0.95), ("mum", 0.9), ("near", 0.3)]
    assert judge.scored == 4 * 2
    assert retriever.top_k(CATALOGUE["mother"], CATALOGUE, 1) == [("parent", 0.95)] and judge.scored == 8
