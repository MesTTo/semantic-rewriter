"""Purpose: tests of the semantic rewriter on every installed engine: every strategy
the engine can run answers every benchmark query exactly under a perfect judge,
strategies it cannot run are refused by name, names spelled like engine
keywords keep their answers, and without tables rewriting loops on a cycle.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import semrewrite
from semrewrite import conformance as kit
from semrewrite.benchmarks import BENCHMARKS
from semrewrite.engines import available, make
from semrewrite.engines.python import PythonEngine
from semrewrite.judges.oracle import OracleJudge
from semrewrite.kb import KB, atom
from semrewrite.model import Atom, Rewrite
from semrewrite.protocols import Budget, Capabilities, Program
from semrewrite.rewriter import STRATEGIES, CapabilityError, Rewriter, Strategy, Timing

KBS = {name: make_kb() for name, make_kb in BENCHMARKS.items()}
ENGINES = available()


def _runnable(engine_name: str) -> list[str]:
    caps = make(engine_name).capabilities
    return [n for n, s in STRATEGIES.items() if all(getattr(caps, c) for c, need in s.needs().items() if need)]


def _oracle_rewriter(kb: KB, engine_name: str, work: int = 3_000_000) -> Rewriter:
    return Rewriter(kb, OracleJudge(kb.entailed()), make(engine_name), None, budget=Budget(work=work, seconds=math.inf))


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


@pytest.mark.parametrize(("knobs", "message"), [
    ({"timing": Timing.LAZY, "depth": -1}, "can't be negative"),
    ({"timing": Timing.LAZY, "depth": 3, "depth_counts": "rules"}, "depth_counts is 'rules'"),
    ({"timing": Timing.EAGER, "depth": 3}, "only lazy discovery"),
    ({"timing": Timing.ROUNDS, "tabled": False}, "only lazy discovery"),
    ({"timing": Timing.SATURATE, "memo": False}, "only lazy discovery"),
    ({"timing": Timing.LAZY, "canonical": True}, "merging alias classes"),
])
def test_a_strategy_outside_its_knobs_is_refused_where_it_is_defined(knobs: dict, message: str) -> None:
    """A setting no implementation has fails when the strategy is built, not at its first run."""
    with pytest.raises(ValueError, match=message):
        Strategy("custom", **knobs)


def test_a_strategy_named_by_its_timing_string_is_the_same_strategy() -> None:
    """Timing is a StrEnum, so "lazy" once passed the dispatch but failed the identity checks, skipping the lazy need."""
    by_name = Strategy("by-name", "lazy", depth=3)
    assert by_name.timing is Timing.LAZY and by_name.needs() == Strategy("by-member", Timing.LAZY, depth=3).needs()
    assert Strategy("by-name", "eager", canonical=True).timing is Timing.EAGER
    with pytest.raises(ValueError, match="sideways"):
        Strategy("by-name", "sideways")

    class Bare:
        name = "bare"
        capabilities = Capabilities(tabling=True)

        def solve(self, program, query, budget, expand=None):
            raise AssertionError("must not be called")

    kb = KBS["family"]
    with pytest.raises(CapabilityError, match="lacks lazy"):
        Rewriter(kb, OracleJudge(kb.entailed()), Bare(), None).ask("parent(X, Y)", Strategy("by-name", "lazy"))


@pytest.mark.parametrize("engine_name", [e for e in ENGINES if "rounds" in _runnable(e)])
def test_rounds_share_one_budget_and_stop_once_a_round_adds_nothing(engine_name: str) -> None:
    """Each evaluation gets only what the earlier ones left, and a round whose expansions add no rewrite is the last."""
    kb = KB("chain", {"p": "{x} is p", "q": "{x} is q", "r": "{x} is r"}, tuple(atom("q", f"c{i}") for i in range(40)),
            gold=frozenset({("q", "p", (0,))}))
    roomy = Rewriter(kb, OracleJudge(kb.entailed()), make(engine_name), None, budget=Budget(work=10**9, seconds=math.inf))
    whole = roomy.ask("p(X)", "rounds")
    # Round 1 reaches p and admits q => p; round 2 reaches q, whose expansion admits nothing, so no third round.
    assert whole.finished and whole.rounds == 2 and len(whole.answers) == 40
    tight = Rewriter(kb, OracleJudge(kb.entailed()), make(engine_name), None,
                     budget=Budget(work=whole.work - 1, seconds=math.inf))
    short = tight.ask("p(X)", "rounds")
    assert not short.finished and short.status == "work_limit" and not short.answers, short


@pytest.mark.parametrize("engine_name", ENGINES)
def test_a_shared_discoverer_admits_afresh_when_the_threshold_moves(engine_name: str) -> None:
    kb = KB("k", {"mother": "{x} is the mother of {y}", "parent": "{x} is a parent of {y}"}, (atom("mother", "ann", "tom"),),
            gold=frozenset({("mother", "parent", (0, 1))}))
    judge = OracleJudge(kb.entailed())
    rewriter = Rewriter(kb, judge, make(engine_name), None)
    shared = rewriter.discoverer()
    assert rewriter.ask("parent(X, tom)", "lazy-tabled", shared).answers == {("ann", "tom")}
    judge.threshold = 2.0
    assert rewriter.ask("parent(X, tom)", "lazy-tabled", shared).answers == set()


@pytest.mark.parametrize("engine_name", [e for e in ENGINES if "naive" in _runnable(e)])
def test_the_first_solve_of_a_process_stops_at_its_work_budget(engine_name: str) -> None:
    """A fresh process's first run of a rewrite loop stops at its inference budget, not at its time limit.

    With an object-annotated op, pymetta 0.9.4 ran the first naive solve of mother(X, bea) to 16 million
    inferences against 300,000, so the run gets a process of its own, importing this checkout's package.
    """
    code = (
        "import json\n"
        "import math\n"
        "from semrewrite.benchmarks import BENCHMARKS\n"
        "from semrewrite.engines import make\n"
        "from semrewrite.judges.oracle import OracleJudge\n"
        "from semrewrite.protocols import Budget\n"
        "from semrewrite.rewriter import Rewriter\n"
        "kb = BENCHMARKS['family']()\n"
        f"rewriter = Rewriter(kb, OracleJudge(kb.entailed()), make({engine_name!r}), None, "
        "budget=Budget(work=300_000, seconds=math.inf))\n"
        "o = rewriter.ask('mother(X, bea)', 'naive')\n"
        "print(json.dumps({'status': o.status, 'work': o.work}))\n"
    )
    package_root = str(Path(semrewrite.__file__).resolve().parents[1])
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                          env={**os.environ, "PYTHONPATH": package_root})
    outcome = json.loads(done.stdout.strip().splitlines()[-1])
    assert outcome["status"] == "work_limit", outcome
    assert outcome["work"] <= 330_000, outcome


@pytest.mark.parametrize("engine_name", ENGINES)
def test_names_spelled_like_engine_keywords_keep_their_answers(engine_name: str) -> None:
    """max and min are pymetta builtins and here also people; if is a predicate. Every runnable strategy keeps the answers."""
    kb = KB(
        name="keywords",
        glosses={"parent": "{x} is a parent of {y}", "daughter_of": "{x} is a daughter of {y}", "if": "{x} if {y}"},
        facts=(("daughter_of", "max", "hope"), ("daughter_of", "min", "max"), ("if", "max", "min")),
        gold=frozenset({("daughter_of", "parent", True)}),
    )
    rewriter = Rewriter(kb, OracleJudge.of(kb.gold), make(engine_name), None, budget=Budget(work=300_000, seconds=math.inf))
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


@pytest.mark.parametrize("engine_name", [e for e in ENGINES if make(e).capabilities.proofs])
@pytest.mark.parametrize("strategy", [name for name, s in STRATEGIES.items() if not s.canonical])
def test_explained_answers_are_derived_from_the_knowledge_base_and_the_admitted_rewrites(
        engine_name: str, strategy: str) -> None:
    """Asked for proofs, each answer comes with a derivation from the knowledge base's facts and rules and the
    rewrites the run admitted, within the strategy's depth bound, whichever encoding the engine picked. Without
    tables a run may stop at its budget, so an untabled strategy is checked on the runs that finish."""
    if strategy not in _runnable(engine_name):
        pytest.skip(f"{engine_name} cannot run {strategy}")
    s = STRATEGIES[strategy]
    explained = 0
    for kb in KBS.values():
        rewriter = Rewriter(kb, OracleJudge(kb.entailed()), make(engine_name), None,
                            budget=Budget(work=5_000_000 if s.tabled else 300_000, seconds=math.inf), proofs=True)
        for query in kb.queries:
            out = rewriter.ask(query, strategy)
            if not s.tabled and not out.finished:
                continue
            assert out.finished, (kb.name, str(query), out.status, out.detail)
            admitted = tuple(Rewrite(c, p, m) for p, c, m in sorted(out.admitted))
            program = Program(facts=kb.facts, rules=kb.rules, rewrites=admitted, depth=s.depth,
                              depth_counts=s.depth_counts)
            assert out.proofs is not None and set(out.proofs) == out.answers, (kb.name, str(query))
            for row, proof in out.proofs.items():
                problem = kit.proof_problem(proof, Atom(query.pred, row), program)
                assert problem is None, (kb.name, str(query), problem, proof.lines())
            explained += 1
    assert explained, f"no {strategy} run finished"


def test_proofs_are_refused_where_none_can_come() -> None:
    """An engine without the capability, or a strategy answering over merged predicates, can't explain its answers."""

    class NoProofs(PythonEngine):
        capabilities = Capabilities(tabling=True, depth=True, rewrite_depth=True, lazy=True, reached=True)

    kb = KBS["family"]
    with pytest.raises(CapabilityError, match="lacks proofs"):
        Rewriter(kb, OracleJudge(kb.entailed()), NoProofs(), None, proofs=True).ask(kb.queries[0], "lazy-tabled")
    with pytest.raises(CapabilityError, match="no derivations"):
        Rewriter(kb, OracleJudge(kb.entailed()), make("python"), None, proofs=True).ask(kb.queries[0],
                                                                                         "alias-canonical")
    assert Rewriter(kb, OracleJudge(kb.entailed()), NoProofs(), None).ask(kb.queries[0], "lazy-tabled").finished


@pytest.mark.skipif("pymetta" not in ENGINES, reason="pymetta does not import here")
def test_proofs_converge_on_a_rewrite_cycle_and_leave_no_cache_rows() -> None:
    """a and b rewrite into each other, so a plain table of proof terms never completes; each answer still gets
    one valid proof, and the lattice rows each solve declared in &metta, which outlives it, are gone after."""
    from metta import MeTTa

    engine = make("pymetta")
    cycle = (Rewrite("b", "a", (0,), 0.9), Rewrite("a", "b", (0,), 0.8))
    program = Program(facts=(Atom("a", ("k",)), Atom("c", ("j",))), rewrites=cycle, proofs=True)
    for expand in (None, lambda p: [rw for rw in cycle if rw.conclusion == p]):
        lazy = expand is not None
        s = engine.solve(program if not lazy else Program(facts=program.facts, proofs=True), Atom("b", ("X",)),
                         Budget(work=1_000_000, seconds=math.inf), expand)
        assert s.status == "ok" and s.answers == {("k",)}, (lazy, s.status, s.detail)
        assert kit.proof_problem(s.proofs[("k",)], Atom("b", ("k",)), program) is None
        assert s.proofs[("k",)].lines() == ["b(k)    [by b(X0) ⇐ a(X0), score 0.900]", "  a(k)    [fact]"]
    with MeTTa() as m:
        assert m.self.run("!(size-atom (collapse (match &metta (cache $name $policy) $name)))") == [[0]]
