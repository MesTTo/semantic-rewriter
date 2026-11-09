"""Purpose: unit tests of the pure-Python engine: derivation heights and depth bounds,
bounds counting rewrites alone, expansion pruned by the depth bound, proofs, budget
statuses and the work unit, lazy expansion once per predicate, any arity, repeated
variables and constants, malformed input, names spelled like reserved words, and runs
that repeat exactly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from semrewrite import conformance as kit
from semrewrite import fixpoint
from semrewrite.engines import make
from semrewrite.engines.python import PythonEngine
from semrewrite.model import Atom, Rewrite, Rule
from semrewrite.protocols import Budget, Capabilities, Engine, Program, Proof

ENGINE = PythonEngine()
GENEROUS = Budget()


def solve(program: Program, query: Atom, budget: Budget = GENEROUS, expand=None):
    """The engine's solve with proofs asked for, which every test reading them needs."""
    return ENGINE.solve(replace(program, proofs=True), query, budget, expand)


def height(proof: Proof) -> int:
    """Proof.steps() without recursion, for proofs deeper than Python's recursion limit."""
    done: dict[int, int] = {}
    stack = [(proof, False)]
    while stack:
        node, ready = stack.pop()
        if id(node) in done:
            continue
        if not ready:
            stack.append((node, True))
            stack.extend((child, False) for child in node.children)
        else:
            done[id(node)] = 0 if node.by is None else 1 + max((done[id(c)] for c in node.children), default=0)
    return done[id(proof)]


def distances(program: Program, start: str) -> dict[str, int]:
    """Breadth-first distance from `start` along rule bodies and rewrite premises."""
    graph: dict[str, list[str]] = {}
    for rule in program.rules:
        graph.setdefault(rule.head.pred, []).extend(a.pred for a in rule.body)
    for rewrite in program.rewrites:
        graph.setdefault(rewrite.conclusion, []).append(rewrite.premise)
    dist = {start: 0}
    queue = deque([start])
    while queue:
        pred = queue.popleft()
        for nxt in graph.get(pred, ()):
            if nxt not in dist:
                dist[nxt] = dist[pred] + 1
                queue.append(nxt)
    return dist


def rewrite_distances(program: Program, start: str) -> dict[str, int]:
    """The fewest rewrite steps from `start` to each predicate, rule bodies costing none (Dijkstra on 0-1 weights)."""
    graph: dict[str, list[tuple[str, int]]] = {}
    for rule in program.rules:
        graph.setdefault(rule.head.pred, []).extend((a.pred, 0) for a in rule.body)
    for rewrite in program.rewrites:
        graph.setdefault(rewrite.conclusion, []).append((rewrite.premise, 1))
    dist = {start: 0}
    frontier = [start]
    while frontier:
        pred = min(frontier, key=dist.__getitem__)
        frontier.remove(pred)
        for nxt, cost in graph.get(pred, ()):
            if nxt not in dist or dist[pred] + cost < dist[nxt]:
                dist[nxt] = dist[pred] + cost
                frontier.append(nxt)
    return dist


class Recorder:
    """An expander serving fixed rewrites by conclusion and recording each call."""

    def __init__(self, rewrites, delay: float = 0.0) -> None:
        self.rewrites = tuple(rewrites)
        self.calls: list[str] = []
        self.delay = delay

    def __call__(self, pred: str):
        self.calls.append(pred)
        time.sleep(self.delay)
        return [r for r in self.rewrites if r.conclusion == pred]


def closure(n: int) -> tuple[Program, Atom]:
    """The transitive closure of an n-node chain, queried for every path."""
    facts = tuple(Atom("edge", (f"n{i}", f"n{i + 1}")) for i in range(n - 1))
    rules = (Rule(Atom("path", ("X", "Y")), (Atom("edge", ("X", "Y")),)),
             Rule(Atom("path", ("X", "Z")), (Atom("edge", ("X", "Y")), Atom("path", ("Y", "Z")))))
    return Program(facts=facts, rules=rules), Atom("path", ("X", "Y"))


FAMILY = Program(
    facts=(Atom("mother", ("alice", "tom")), Atom("dad", ("bob", "tom")),
           Atom("child_of", ("tom", "erin")), Atom("guardian", ("gus", "tom"))),
    rewrites=(Rewrite("parent", "mother", (0, 1)), Rewrite("parent", "dad", (0, 1)),
              Rewrite("parent", "child_of", (1, 0))))


def test_it_is_the_registered_engine() -> None:
    engine = make("python")
    assert isinstance(engine, PythonEngine) and isinstance(engine, Engine)
    assert engine.name == "python"
    assert engine.capabilities == Capabilities(tabling=True, depth=True, rewrite_depth=True, lazy=True,
                                               untabled=False, reached=True, proofs=True)


def test_the_design_example() -> None:
    s = solve(FAMILY, Atom("parent", ("X", "tom")))
    assert s.status == "ok"
    assert s.answers == {("alice", "tom"), ("bob", "tom"), ("erin", "tom")}
    assert s.derivations == 3
    assert s.reached == {"parent", "mother", "dad", "child_of"} and s.tables == 4
    assert s.proofs[("erin", "tom")].lines() == [
        "parent(erin, tom)    [by parent(X0, X1) ⇐ child_of(X1, X0), score 1.000]",
        "  child_of(tom, erin)    [fact]",
    ]


def test_proofs_come_only_when_asked_for() -> None:
    asked = ENGINE.solve(replace(FAMILY, proofs=True), Atom("parent", ("X", "tom")), GENEROUS)
    unasked = ENGINE.solve(FAMILY, Atom("parent", ("X", "tom")), GENEROUS)
    assert set(asked.proofs) == unasked.answers == asked.answers and unasked.proofs is None
    assert unasked.work == asked.work


@settings(max_examples=300, deadline=None)
@given(kit.cases())
def test_round_k_derives_the_facts_of_height_k(case: kit.Case) -> None:
    """Each answer's proof is its first derivation, so its height is the least height any derivation has."""
    s = solve(case.program, case.query)
    assert s.status == "ok" and s.answers == kit.reference(case.program, case.query)
    tallest = max((height(p) for p in s.proofs.values()), default=0)
    within = {d: kit.reference(replace(case.program, depth=d), case.query) for d in range(1, tallest + 2)}
    base = {f.args for f in case.program.facts if f.pred == case.query.pred}
    for answer, proof in s.proofs.items():
        least = 0 if answer in base else min(d for d, rows in within.items() if answer in rows)
        assert height(proof) == least, (answer, proof.lines())


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 4])
def test_depth_bound_on_a_chain(depth: int) -> None:
    chain = Program(facts=(Atom("p3", ("a",)),),
                    rewrites=tuple(Rewrite(f"p{i}", f"p{i + 1}", (0,)) for i in range(3)), depth=depth)
    s = solve(chain, Atom("p0", ("X",)))
    reachable = depth == 0 or depth >= 3
    assert s.status == "ok"
    assert s.answers == ({("a",)} if reachable else set())
    assert [height(p) for p in s.proofs.values()] == ([3] if reachable else [])


def test_a_bound_on_rewrites_keeps_user_recursion_complete() -> None:
    """A rewrite on top of a 30-node closure: a step bound of 3 cuts the paths, one rewrite answers them all."""
    program, _ = closure(30)
    program = replace(program, rewrites=(Rewrite("link", "path", (0, 1)),))
    query = Atom("link", ("X", "Y"))
    every = {(f"n{i}", f"n{j}") for i in range(30) for j in range(i + 1, 30)}
    assert solve(replace(program, depth=1, depth_counts="rewrites"), query).answers == every
    # path(ni, nj) takes j - i rule steps, and link one rewrite more.
    steps = solve(replace(program, depth=3), query)
    assert steps.answers == {(a, b) for a, b in every if int(b[1:]) - int(a[1:]) <= 2}
    assert solve(replace(program, depth=1, depth_counts="rewrites"), Atom("path", ("X", "Y"))).answers == every


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_a_bound_on_rewrites_cuts_rewrite_chains_and_ignores_rules_between_them(depth: int) -> None:
    """p0 needs three rewrites, each followed by a user rule: six steps, but three rewrites."""
    rewrites = tuple(Rewrite(f"p{i}", f"q{i}", (0,)) for i in range(3))
    rules = tuple(Rule(Atom(f"q{i}", ("X",)), (Atom(f"p{i + 1}", ("X",)),)) for i in range(3))
    chain = Program(facts=(Atom("p3", ("a",)),), rules=rules, rewrites=rewrites, depth=depth)
    by_rewrites = solve(replace(chain, depth_counts="rewrites"), Atom("p0", ("X",)))
    assert by_rewrites.answers == ({("a",)} if depth >= 3 else set())
    assert [p.steps("rewrites") for p in by_rewrites.proofs.values()] == ([3] if depth >= 3 else [])
    assert solve(chain, Atom("p0", ("X",))).answers == set()


@settings(max_examples=300, deadline=None)
@given(kit.cases(), st.integers(1, 4))
def test_level_k_derives_the_facts_needing_k_rewrites(case: kit.Case, depth: int) -> None:
    """Under a bound on rewrites each answer's proof is its first derivation, so it uses the fewest rewrites any
    derivation needs on its busiest path, and the answers are the reference's."""
    bounded = replace(case.program, depth=depth, depth_counts="rewrites")
    s = solve(bounded, case.query)
    assert s.status == "ok" and s.answers == kit.reference(bounded, case.query)
    within = {0: fixpoint.answers(fixpoint.least_model(case.program.facts, case.program.rules), case.query)}
    within |= {d: kit.reference(replace(bounded, depth=d), case.query) for d in range(1, depth + 1)}
    for answer, proof in s.proofs.items():
        assert proof.steps("rewrites") == min(d for d, rows in within.items() if answer in rows), proof.lines()
        assert kit.proof_problem(proof, Atom(case.query.pred, answer), bounded) is None


@settings(max_examples=300, deadline=None)
@given(kit.cases(), st.integers(1, 4))
def test_a_bound_on_rewrites_expands_only_predicates_fewer_rewrites_away(case: kit.Case, depth: int) -> None:
    """With a bound of n rewrites, expand runs once for each predicate fewer than n rewrites from the query."""
    recorder = Recorder(case.program.rewrites)
    s = solve(replace(case.program, rewrites=(), depth=depth, depth_counts="rewrites"), case.query, expand=recorder)
    dist = rewrite_distances(case.program, case.query.pred)
    assert len(recorder.calls) == len(set(recorder.calls))
    assert set(recorder.calls) == {p for p, d in dist.items() if d < depth}
    assert s.reached == {p for p, d in dist.items() if d <= depth}
    bounded = replace(case.program, depth=depth, depth_counts="rewrites")
    assert s.status == "ok" and s.answers == kit.reference(bounded, case.query)


@settings(max_examples=300, deadline=None)
@given(kit.cases(), st.integers(1, 4))
def test_depth_bound_expands_only_predicates_closer_than_it(case: kit.Case, depth: int) -> None:
    """With depth n, expand runs once for each predicate at distance < n and never further; answers and proofs keep least_model(depth=n)."""
    recorder = Recorder(case.program.rewrites)
    s = solve(replace(case.program, rewrites=(), depth=depth), case.query, expand=recorder)
    dist = distances(case.program, case.query.pred)
    assert len(recorder.calls) == len(set(recorder.calls))
    assert set(recorder.calls) == {p for p, d in dist.items() if d < depth}
    assert s.reached == {p for p, d in dist.items() if d <= depth} and s.tables == len(s.reached)
    bounded = replace(case.program, depth=depth)
    assert s.status == "ok" and s.answers == kit.reference(bounded, case.query)
    for answer, proof in s.proofs.items():
        assert kit.proof_problem(proof, Atom(case.query.pred, answer), bounded) is None


def test_expand_runs_once_per_relevant_predicate_and_never_elsewhere() -> None:
    rewrites = (Rewrite("p", "q", (0,)), Rewrite("q", "p", (0,)), Rewrite("r", "q", (0,)), Rewrite("u", "p", (0,)))
    program = Program(facts=(Atom("q", ("a",)), Atom("u", ("b",))))
    recorder = Recorder(rewrites)
    s = solve(program, Atom("p", ("X",)), expand=recorder)
    assert recorder.calls == ["p", "q"]
    assert s.answers == {("a",)} and s.reached == {"p", "q"}
    eager = solve(replace(program, rewrites=rewrites), Atom("p", ("X",)))
    both = solve(replace(program, rewrites=rewrites), Atom("p", ("X",)), expand=Recorder(rewrites))
    assert eager.answers == both.answers and eager.work == both.work


def test_expand_misbehaving_raises() -> None:
    program = Program(facts=(Atom("q", ("a",)),))
    with pytest.raises(ValueError, match="concludes 'r'"):
        solve(program, Atom("p", ("X",)), expand=lambda pred: [Rewrite("r", "q", (0,))])
    with pytest.raises(TypeError, match="not a Rewrite"):
        solve(program, Atom("p", ("X",)), expand=lambda pred: [("q", "p")])
    with pytest.raises(ValueError, match="arity"):
        solve(program, Atom("p", ("X",)), expand=lambda pred: [Rewrite("p", "q", (1, 0))] if pred == "p" else [])

    def broken(pred: str):
        raise RuntimeError("judge offline")

    with pytest.raises(RuntimeError, match="judge offline"):
        solve(program, Atom("p", ("X",)), expand=broken)


def test_work_counts_candidate_rows() -> None:
    """Round 1 examines p's two rows and derives q's; round 2 has no plan reading q; reading q's answers examines two more."""
    program = Program(facts=(Atom("p", ("a",)), Atom("p", ("b",))), rewrites=(Rewrite("q", "p", (0,)),))
    s = solve(program, Atom("q", ("X",)))
    assert s.answers == {("a",), ("b",)} and s.work == 4


def test_budget_statuses() -> None:
    program, query = closure(60)
    full = solve(program, query)
    assert full.status == "ok" and len(full.answers) == 60 * 59 // 2
    assert solve(program, query).work == full.work
    assert solve(program, query, Budget(work=full.work)).status == "ok"
    short = solve(program, query, Budget(work=full.work - 1))
    assert short.status == "work_limit" and short.work == full.work
    assert short.answers == set() and short.proofs == {} and short.derivations == 0
    assert short.reached == {"path", "edge"} and "work budget" in short.detail
    tiny = solve(program, query, Budget(work=50))
    assert tiny.status == "work_limit" and tiny.work == 51 and not tiny.answers
    instant = solve(program, query, Budget(seconds=0.0))
    assert instant.status == "time_limit" and not instant.answers and "time budget" in instant.detail
    trivial = solve(Program(facts=(Atom("p", ("a",)),)), Atom("p", ("X",)), Budget(seconds=0.0))
    assert trivial.status == "time_limit" and not trivial.answers


def test_time_limit_while_expanding() -> None:
    recorder = Recorder([Rewrite("p", "q", (0,))], delay=0.05)
    s = solve(Program(facts=(Atom("q", ("a",)),)), Atom("p", ("X",)), Budget(seconds=0.01), expand=recorder)
    assert s.status == "time_limit" and not s.answers and "while expanding 'p'" in s.detail
    assert recorder.calls == ["p"]


def test_any_arity() -> None:
    nullary = Program(facts=(Atom("on", ()),), rewrites=(Rewrite("lit", "on", ()),))
    assert solve(nullary, Atom("lit", ())).answers == {()}
    assert solve(nullary, Atom("off", ())).answers == set()
    five = Program(facts=(Atom("p", ("a", "b", "c", "d", "e")), Atom("p", ("e", "d", "c", "b", "a"))),
                   rewrites=(Rewrite("q", "p", (4, 3, 2, 1, 0)), Rewrite("r", "q", (1, 2, 3, 4, 0)),
                             Rewrite("r", "r", (4, 0, 1, 2, 3))))
    for query in (Atom("r", ("V", "W", "X", "Y", "Z")), Atom("r", ("a", "W", "X", "Y", "Z")), Atom("q", ("V", "W", "c", "Y", "Z"))):
        s = solve(five, query)
        assert s.answers == kit.reference(five, query) and s.answers


def test_repeated_variables_and_constants() -> None:
    facts = (Atom("person", ("ann",)), Atom("person", ("bob",)),
             Atom("likes", ("ann", "ann")), Atom("likes", ("ann", "tom")), Atom("likes", ("bob", "tom")),
             Atom("parent", ("ann", "bob")), Atom("parent", ("bob", "cy")))
    rules = (Rule(Atom("same", ("X", "X")), (Atom("person", ("X",)),)),
             Rule(Atom("tagged", ("X", "k")), (Atom("person", ("X",)),)),
             Rule(Atom("likes_tom", ("X",)), (Atom("likes", ("X", "tom")),)),
             Rule(Atom("narcissist", ("X",)), (Atom("likes", ("X", "X")),)),
             Rule(Atom("grand", ("X", "Z")), (Atom("parent", ("X", "Y")), Atom("parent", ("Y", "Z")))))
    program = Program(facts=facts, rules=rules)
    expect = {
        Atom("same", ("X", "Y")): {("ann", "ann"), ("bob", "bob")},
        Atom("tagged", ("X", "Y")): {("ann", "k"), ("bob", "k")},
        Atom("likes_tom", ("X",)): {("ann",), ("bob",)},
        Atom("narcissist", ("X",)): {("ann",)},
        Atom("grand", ("X", "Z")): {("ann", "cy")},
        Atom("likes", ("X", "X")): {("ann", "ann")},
        Atom("likes", ("X", "tom")): {("ann", "tom"), ("bob", "tom")},
        Atom("same", ("bob", "X")): {("bob", "bob")},
    }
    for query, rows in expect.items():
        assert solve(program, query).answers == rows == kit.reference(program, query), query


def test_arity_disagreement_raises_and_irrelevant_parts_are_ignored() -> None:
    with pytest.raises(ValueError, match=r"'p' has arity 2 in rewrite q\(X0, X1\) ⇐ p\(X1, X0\) but arity 1 in fact p\(a\)"):
        solve(Program(facts=(Atom("p", ("a",)),), rewrites=(Rewrite("q", "p", (1, 0)),)), Atom("q", ("X", "Y")))
    with pytest.raises(ValueError, match="arity"):
        solve(Program(facts=(Atom("p", ("a",)),), rules=(Rule(Atom("q", ("X",)), (Atom("p", ("X", "Y")),)),)),
              Atom("q", ("X",)))
    with pytest.raises(ValueError, match="query"):
        solve(Program(facts=(Atom("p", ("a", "b")),)), Atom("p", ("X",)))
    with pytest.raises(ValueError, match="arity"):
        solve(Program(facts=(Atom("q", ("a",)),), rules=(Rule(Atom("p", ("X",)), (Atom("q", ("X",)),)),
                                                         Rule(Atom("p", ("X", "X")), (Atom("q", ("X",)),)))),
              Atom("p", ("X",)))
    elsewhere = Program(facts=(Atom("p", ("a",)), Atom("u", ("a",)), Atom("u", ("a", "b"))),
                        rewrites=(Rewrite("w", "u", (1, 0)),))
    assert solve(elsewhere, Atom("p", ("X",))).answers == {("a",)}


def test_facts_of_irrelevant_predicates_are_never_read() -> None:
    noise = tuple(Atom("noise", (f"k{i}", f"k{i + 1}")) for i in range(10_000))
    lean = solve(FAMILY, Atom("parent", ("X", "tom")))
    heavy = solve(replace(FAMILY, facts=FAMILY.facts + noise), Atom("parent", ("X", "tom")))
    assert heavy.answers == lean.answers and heavy.work == lean.work and heavy.reached == lean.reached


def test_names_are_data() -> None:
    program = Program(facts=(Atom("max", ("if", "not")), Atom("holds", ("if", "max"))),
                      rules=(Rule(Atom("let", ("X", "Y")), (Atom("min", ("X", "Z")), Atom("holds", ("Z", "Y")))),),
                      rewrites=(Rewrite("min", "max", (1, 0)), Rewrite("is", "let", (0, 1))))
    assert solve(program, Atom("min", ("X", "Y"))).answers == {("not", "if")}
    assert solve(program, Atom("is", ("X", "Y"))).answers == {("not", "max")}


def test_tabled_flag_and_duplicates_change_nothing() -> None:
    program, query = closure(20)
    base = solve(program, query)
    untabled = solve(replace(program, tabled=False), query)
    assert untabled.answers == base.answers and untabled.work == base.work
    doubled = replace(program, facts=program.facts * 2, rules=program.rules * 2)
    again = solve(doubled, query)
    assert again.answers == base.answers and again.work == base.work
    rewrites = FAMILY.rewrites + tuple(replace(r, score=0.5) for r in FAMILY.rewrites)
    family = solve(replace(FAMILY, rewrites=rewrites), Atom("parent", ("X", "tom")))
    plain = solve(FAMILY, Atom("parent", ("X", "tom")))
    assert family.answers == plain.answers and family.work == plain.work


def test_long_chains_need_no_recursion() -> None:
    n = 3000
    chain = Program(facts=(Atom(f"p{n}", ("a",)),),
                    rewrites=tuple(Rewrite(f"p{i}", f"p{i + 1}", (0,)) for i in range(n)))
    s = solve(chain, Atom("p0", ("X",)))
    assert s.answers == {("a",)} and height(s.proofs[("a",)]) == n and s.tables == n + 1


def test_runs_repeat_exactly_across_hash_seeds() -> None:
    """Work and proofs come out the same whatever PYTHONHASHSEED orders sets by."""
    script = (
        "import random\n"
        "from semrewrite.engines.python import PythonEngine\n"
        "from semrewrite.model import Atom, Rewrite, Rule\n"
        "from semrewrite.protocols import Budget, Program\n"
        "rng = random.Random(7)\n"
        "P = [f'q{i}' for i in range(12)]\n"
        "facts = tuple(Atom(rng.choice(P), (rng.choice('abcde'), rng.choice('abcde'))) for _ in range(30))\n"
        "rws = tuple(Rewrite(c, p, rng.choice([(0, 1), (1, 0)])) for c in P for p in rng.sample(P, 4) if p != c)\n"
        "join = (Rule(Atom('q0', ('X', 'Z')), (Atom('q1', ('X', 'Y')), Atom('q2', ('Y', 'Z')))),)\n"
        "s = PythonEngine().solve(Program(facts=facts, rules=join, rewrites=rws, proofs=True), Atom('q0', ('X', 'Y')),\n"
        "                         Budget())\n"
        "print(s.work, len(s.answers))\n"
        "for answer in sorted(s.proofs):\n"
        "    print('\\n'.join(s.proofs[answer].lines()))\n"
    )
    outputs = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outputs.add(subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                   check=True, env=env).stdout)
    assert len(outputs) == 1 and len(next(iter(outputs)).splitlines()) > 20
