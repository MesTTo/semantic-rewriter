"""Purpose: tests of the semantic rewriter engine on pymetta: the knowledge-base
round trip through MeTTa text, the fixpoint and implication closure, alias
canonicalisation, and every strategy against the gold answers of every
benchmark query under a perfect (oracle) judge.
"""

from __future__ import annotations

import pytest

from semrewrite.benchmarks import BENCHMARKS
from semrewrite.engine import STRATEGIES, Budget, Evaluation, ParityUnionFind, Rewriter
from semrewrite.judging import OracleJudge
from semrewrite.kb import KB, Lit, Query, Rule, fixpoint, from_metta, implication_closure

KBS = {name: make() for name, make in BENCHMARKS.items()}
TABLED = [name for name, s in STRATEGIES.items() if s.evaluation is Evaluation.TABLED and s.timing.value != "none"]


@pytest.mark.parametrize("name", sorted(KBS))
def test_metta_text_round_trips(name: str) -> None:
    """from_metta(kb.to_metta()) gives back the same glosses, facts and rules."""
    kb = KBS[name]
    back = from_metta(kb.to_metta(), name)
    assert dict(back.glosses) == dict(kb.glosses)
    assert sorted(back.facts) == sorted(kb.facts)
    assert back.rules == kb.rules


def test_fixpoint_chains_and_swaps() -> None:
    """mother => parent and child_of swapped => parent chain into grandparent."""
    rules = (
        Rule(Lit("parent", "X", "Y"), (Lit("mother", "X", "Y"),)),
        Rule(Lit("parent", "X", "Y"), (Lit("child_of", "Y", "X"),)),
        Rule(Lit("grandparent", "X", "Z"), (Lit("parent", "X", "Y"), Lit("parent", "Y", "Z"))),
    )
    facts = (("mother", "carol", "alice"), ("child_of", "tom", "alice"))
    closed = fixpoint(facts, rules)
    assert closed["parent"] == {("carol", "alice"), ("alice", "tom")}
    assert closed["grandparent"] == {("carol", "tom")}


def test_implication_closure_composes_swaps() -> None:
    """Two swaps cancel, one survives, and a symmetric conclusion takes both parities."""
    closed = implication_closure({("son_of", "child", False), ("child", "parent", True), ("spouse", "spouse", True), ("wife_of", "spouse", False)})
    assert ("son_of", "parent", True) in closed
    assert ("son_of", "parent", False) not in closed
    assert {("wife_of", "spouse", False), ("wife_of", "spouse", True), ("spouse", "spouse", True)} <= closed
    assert ("parent", "parent", False) not in closed


def test_parity_union_find_joins_aliases_and_spots_symmetry() -> None:
    """child_of swapped joins parent; joining a class to itself at odd parity marks it symmetric."""
    uf = ParityUnionFind()
    uf.union("parent", "child_of", True)
    uf.union("mother", "mum", False)
    root, parity = uf.find("child_of")
    assert root == uf.find("parent")[0]
    assert parity != uf.find("parent")[1]
    assert uf.find("mum")[0] == uf.find("mother")[0]
    uf.union("spouse", "spouse", True)
    assert "spouse" in uf.symmetric


def _oracle_rewriter(kb: KB) -> Rewriter:
    return Rewriter(kb, OracleJudge.of(implication_closure(kb.gold)), None, budget=Budget(inferences=3_000_000, seconds=60))


@pytest.mark.parametrize("strategy", TABLED)
@pytest.mark.parametrize("name", sorted(KBS))
def test_tabled_strategies_answer_every_query_exactly(name: str, strategy: str) -> None:
    """With a perfect judge every tabled strategy returns exactly the gold answers and terminates."""
    kb = KBS[name]
    rewriter = _oracle_rewriter(kb)
    closure = kb.closure()
    for query in kb.queries:
        outcome = rewriter.ask(STRATEGIES[strategy], query)
        assert outcome.terminated, f"{strategy} on {query}: {outcome.status} {outcome.detail}"
        assert outcome.answers == kb.answers(query, closure), f"{strategy} on {query}"


@pytest.mark.parametrize("name", sorted(KBS))
def test_exact_matching_is_sound_and_misses_mismatched_vocabulary(name: str) -> None:
    """Without discovery every answer is right but most gold answers are missing."""
    kb = KBS[name]
    rewriter = _oracle_rewriter(kb)
    closure = kb.closure()
    found = gold = 0
    for query in kb.queries:
        outcome = rewriter.ask(STRATEGIES["exact"], query)
        expected = kb.answers(query, closure)
        assert outcome.answers <= expected
        found += len(outcome.answers)
        gold += len(expected)
    assert found < gold / 2


def test_untabled_rewriting_never_finishes_on_an_alias_cycle() -> None:
    """mother <=> mum makes SLD resolution loop, with or without a discovery memo, until the budget ends it."""
    kb = KBS["family"]
    rewriter = Rewriter(kb, OracleJudge.of(implication_closure(kb.gold)), None, budget=Budget(inferences=200_000, seconds=30))
    for strategy in ("naive", "memo-untabled"):
        outcome = rewriter.ask(STRATEGIES[strategy], Query("mother", "X", "Y"))
        assert outcome.status in ("inference_limit", "time_limit"), (strategy, outcome.status)


def test_naive_rediscovery_keeps_calling_the_judge() -> None:
    """Without a memo the same predicate is expanded again on every call, so expansions far exceed the catalogue."""
    kb = KBS["family"]
    rewriter = Rewriter(kb, OracleJudge.of(implication_closure(kb.gold)), None, budget=Budget(inferences=200_000, seconds=30))
    naive = rewriter.ask(STRATEGIES["naive"], Query("mother", "X", "Y"))
    memo = rewriter.ask(STRATEGIES["memo-untabled"], Query("mother", "X", "Y"))
    assert memo.expansions <= len(kb.glosses)
    assert naive.expansions > 10 * len(kb.glosses)
