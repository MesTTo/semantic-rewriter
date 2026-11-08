"""Purpose: tests of the semantic rewriter on every installed engine: every strategy
the engine can run answers every benchmark query exactly under a perfect judge,
strategies it cannot run are refused by name, names spelled like engine
keywords keep their answers, and without tables rewriting loops on a cycle.
"""

from __future__ import annotations

import pytest

from semrewrite.benchmarks import BENCHMARKS
from semrewrite.engines import available, make
from semrewrite.judges.oracle import OracleJudge
from semrewrite.kb import KB, atom
from semrewrite.protocols import Budget, Capabilities
from semrewrite.rewriter import STRATEGIES, CapabilityError, Rewriter

KBS = {name: make_kb() for name, make_kb in BENCHMARKS.items()}
ENGINES = available()


def _runnable(engine_name: str) -> list[str]:
    caps = make(engine_name).capabilities
    return [n for n, s in STRATEGIES.items() if all(getattr(caps, c) for c, need in s.needs().items() if need)]


def _oracle_rewriter(kb: KB, engine_name: str, work: int = 3_000_000) -> Rewriter:
    return Rewriter(kb, OracleJudge(kb.entailed()), make(engine_name), None, budget=Budget(work=work, seconds=60))


@pytest.mark.parametrize("engine_name", ENGINES)
@pytest.mark.parametrize("name", sorted(KBS))
def test_tabled_strategies_answer_every_query_exactly(engine_name: str, name: str) -> None:
    """With a perfect judge every strategy with tables and no depth bound returns exactly the gold answers."""
    kb = KBS[name]
    rewriter = _oracle_rewriter(kb, engine_name)
    closure = kb.closure()
    for strategy in _runnable(engine_name):
        s = STRATEGIES[strategy]
        if not s.tabled or s.depth or s.timing.value == "none":
            continue
        for query in kb.queries:
            outcome = rewriter.ask(query, s)
            assert outcome.finished, f"{engine_name} {strategy} on {query}: {outcome.status} {outcome.detail}"
            assert outcome.answers == kb.answers(query, closure), f"{engine_name} {strategy} on {query}"


@pytest.mark.parametrize("engine_name", ENGINES)
@pytest.mark.parametrize("name", sorted(KBS))
def test_exact_matching_is_sound_and_misses_mismatched_vocabulary(engine_name: str, name: str) -> None:
    """Without discovery every answer is right but most gold answers are missing."""
    kb = KBS[name]
    rewriter = _oracle_rewriter(kb, engine_name)
    closure = kb.closure()
    found = gold = 0
    for query in kb.queries:
        outcome = rewriter.ask(query, "exact")
        expected = kb.answers(query, closure)
        assert outcome.answers <= expected
        found += len(outcome.answers)
        gold += len(expected)
    assert found < gold / 2


def test_a_strategy_the_engine_cannot_run_is_refused_by_name() -> None:
    class Bare:
        name = "bare"
        capabilities = Capabilities(tabling=True)

        def solve(self, program, query, budget, expand=None):
            raise AssertionError("must not be called")

    kb = KBS["family"]
    rewriter = Rewriter(kb, OracleJudge(kb.entailed()), Bare(), None)
    with pytest.raises(CapabilityError, match="lacks lazy"):
        rewriter.ask("parent(X, Y)", "lazy-tabled")
    with pytest.raises(CapabilityError, match="lacks untabled, depth"):
        rewriter.ask("parent(X, Y)", "depth-bounded")


@pytest.mark.parametrize("engine_name", ENGINES)
def test_names_spelled_like_engine_keywords_keep_their_answers(engine_name: str) -> None:
    """max and min are pymetta builtins and here also people; if is a predicate. Every runnable strategy keeps the answers."""
    kb = KB(
        name="keywords",
        glosses={"parent": "{x} is a parent of {y}", "daughter_of": "{x} is a daughter of {y}", "if": "{x} if {y}"},
        facts=(("daughter_of", "max", "hope"), ("daughter_of", "min", "max"), ("if", "max", "min")),
        gold=frozenset({("daughter_of", "parent", True)}),
    )
    rewriter = Rewriter(kb, OracleJudge.of(kb.gold), make(engine_name), None, budget=Budget(work=300_000, seconds=30))
    for strategy in _runnable(engine_name):
        if STRATEGIES[strategy].timing.value == "none":
            assert rewriter.ask(atom("if", "X", "min"), strategy).answers == {("max", "min")}, strategy
            continue
        assert rewriter.ask(atom("parent", "X", "max"), strategy).answers == {("hope", "max")}, strategy
        assert rewriter.ask(atom("parent", "max", "Y"), strategy).answers == {("max", "min")}, strategy


@pytest.mark.parametrize("engine_name", [e for e in ENGINES if make(e).capabilities.untabled])
def test_untabled_rewriting_never_finishes_on_an_alias_cycle(engine_name: str) -> None:
    """mother <=> mum makes top-down resolution loop, with or without a discovery memo, until the budget ends it."""
    kb = KBS["family"]
    rewriter = _oracle_rewriter(kb, engine_name, work=200_000)
    for strategy in ("naive", "memo-untabled"):
        outcome = rewriter.ask(atom("mother", "X", "Y"), strategy)
        assert outcome.status in ("work_limit", "time_limit"), (strategy, outcome.status)
        assert outcome.answers == set()


@pytest.mark.parametrize("engine_name", [e for e in ENGINES if make(e).capabilities.untabled])
def test_naive_rediscovery_keeps_calling_the_judge(engine_name: str) -> None:
    """Without a memo the same predicate is expanded again on every call, so expansions far exceed the catalogue."""
    kb = KBS["family"]
    rewriter = _oracle_rewriter(kb, engine_name, work=200_000)
    naive = rewriter.ask(atom("mother", "X", "Y"), "naive")
    memo = rewriter.ask(atom("mother", "X", "Y"), "memo-untabled")
    assert memo.expansions <= len(kb.glosses)
    assert naive.expansions > 10 * len(kb.glosses)


@pytest.mark.parametrize("engine_name", [e for e in ENGINES if make(e).capabilities.depth])
def test_depth_tabled_answers_what_depth_bounded_answers_on_the_benchmarks(engine_name: str) -> None:
    """With the same depth bound, tables change only the cost: answers agree query by query."""
    caps = make(engine_name).capabilities
    for kb in KBS.values():
        rewriter = _oracle_rewriter(kb, engine_name, work=5_000_000)
        for query in kb.queries:
            tabled = rewriter.ask(query, "depth-tabled")
            assert tabled.finished, (kb.name, str(query), tabled.status)
            if caps.untabled:
                bounded = rewriter.ask(query, "depth-bounded")
                if bounded.finished:
                    assert bounded.answers == tabled.answers, (kb.name, str(query))
