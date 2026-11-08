"""Purpose: property tests of when semantic rewriting stops, over generated
catalogues that always contain an alias cycle.

The claims, each checked against a Python fixpoint over the gold rules:

| strategy | claim |
|---|---|
| every tabled strategy without a depth bound | terminates and answers exactly the fixpoint, on every catalogue |
| untabled (naive, memo-untabled) | never terminates when the query predicate lies on a cycle of rewrites |
| untabled (naive, memo-untabled) | terminates with the fixpoint when everything the query reaches is acyclic |
| depth-bounded, depth-tabled, rewrite-depth-tabled | terminate, every answer is a gold answer, and tables leave the answers under a given bound unchanged |

The judge is an oracle that admits only the direct gold triples, so a chain
p0 <= p1 <= p2 must be found by discovering p1 and then p2: recursive
discovery is exercised, not handed over as a closure.
"""

from __future__ import annotations

from collections import defaultdict

import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from semrewrite.engines import available, make
from semrewrite.judges.oracle import OracleJudge
from semrewrite.kb import KB, atom
from semrewrite.model import Rule
from semrewrite.protocols import Budget
from semrewrite.rewriter import STRATEGIES, Rewriter

CONSTANTS = ("c0", "c1", "c2", "c3")
TABLED = [n for n, s in STRATEGIES.items() if s.tabled and not s.depth and s.timing.value != "none"]
UNTABLED = [n for n, s in STRATEGIES.items() if not s.tabled and not s.depth]
DEPTH = [n for n, s in STRATEGIES.items() if s.depth]


@st.composite
def catalogues(draw: st.DrawFn) -> KB:
    """A random catalogue of 2 to 6 predicates whose gold holds at least one alias cycle.

    Gold implications are random (premise, conclusion, swapped) triples, self-swaps
    included, plus one planted equivalence p_i <=> p_j; facts are random over four
    constants; at most one user rule joins two predicates; the query binds one
    argument, both or neither.
    """
    n = draw(st.integers(2, 6))
    preds = [f"p{i}" for i in range(n)]
    pred = st.sampled_from(preds)
    a, b = draw(st.lists(pred, min_size=2, max_size=2, unique=True))
    swapped = draw(st.booleans())
    gold = {(a, b, swapped), (b, a, swapped)}
    gold |= set(draw(st.lists(st.tuples(pred, pred, st.booleans()), max_size=2 * n)))
    gold = {(p, q, s) for p, q, s in gold if not (p == q and not s)}
    facts = draw(st.lists(st.tuples(pred, st.sampled_from(CONSTANTS), st.sampled_from(CONSTANTS)), min_size=1, max_size=8))
    rules: tuple[Rule, ...] = ()
    if n >= 3 and draw(st.booleans()):
        head, left, right = draw(st.lists(pred, min_size=3, max_size=3))
        rules = (Rule(atom(head, "X", "Z"), (atom(left, "X", "Y"), atom(right, "Y", "Z"))),)
    term = st.sampled_from(("X",) + CONSTANTS)
    qa, qb = draw(term), draw(st.sampled_from(("Y",) + CONSTANTS))
    query = atom(draw(pred), qa, qb)
    glosses = {p: f"{{x}} is {p} of {{y}}" for p in preds}
    return KB("generated", glosses, tuple(sorted(set(facts))), rules, frozenset(gold), (query,))


def _edges(kb: KB) -> dict[str, set[str]]:
    """Predicate dependency edges: a goal on p calls q when q => p is gold or q is in a rule body for p."""
    calls: dict[str, set[str]] = defaultdict(set)
    for premise, conclusion, _ in kb.gold:
        calls[conclusion].add(premise)
    for rule in kb.rules:
        calls[rule.head.pred].update(lit.pred for lit in rule.body)
    return calls


def _reachable(calls: dict[str, set[str]], start: str) -> set[str]:
    seen, stack = {start}, [start]
    while stack:
        for q in calls.get(stack.pop(), ()):
            if q not in seen:
                seen.add(q)
                stack.append(q)
    return seen


def _on_rewrite_cycle(kb: KB, start: str) -> bool:
    """True when rewrites alone lead from `start` back to `start`."""
    rewrites: dict[str, set[str]] = defaultdict(set)
    for premise, conclusion, _ in kb.gold:
        rewrites[conclusion].add(premise)
    frontier, seen = list(rewrites.get(start, ())), set()
    while frontier:
        q = frontier.pop()
        if q == start:
            return True
        if q not in seen:
            seen.add(q)
            frontier.extend(rewrites.get(q, ()))
    return False


def _acyclic_from(kb: KB, start: str) -> bool:
    """True when no dependency cycle is reachable from `start`."""
    calls = _edges(kb)
    reach = _reachable(calls, start)
    state: dict[str, int] = {}

    def visit(p: str) -> bool:
        state[p] = 1
        for q in calls.get(p, ()):
            if q in reach and (state.get(q) == 1 or (q not in state and not visit(q))):
                return False
        state[p] = 2
        return True

    return visit(start)


def _rewriter(kb: KB, work: int, engine_name: str) -> Rewriter:
    return Rewriter(kb, OracleJudge(kb.gold), make(engine_name), None, budget=Budget(work=work, seconds=60))


def _engines(*capabilities: str) -> list[str]:
    return [e for e in available() if all(getattr(make(e).capabilities, c) for c in capabilities)]


# The case count comes from the Hypothesis profile tests/conftest.py loads: 400, or 100 under ci.
SETTINGS = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.mark.parametrize("engine_name", _engines("tabling"))
@SETTINGS
@given(catalogues())
def test_tabled_strategies_reach_exactly_the_fixpoint(engine_name: str, kb: KB) -> None:
    """Every tabled strategy stops inside the budget with exactly the fixpoint's answers."""
    query = kb.queries[0]
    gold = kb.answers(query)
    event("query on a rewrite cycle" if _on_rewrite_cycle(kb, query.pred) else
          "query reaches no cycle" if _acyclic_from(kb, query.pred) else "query reaches a cycle off itself")
    rewriter = _rewriter(kb, 2_000_000, engine_name)
    caps = make(engine_name).capabilities
    for name in TABLED:
        if not all(getattr(caps, c) for c, need in STRATEGIES[name].needs().items() if need):
            continue
        outcome = rewriter.ask(query, STRATEGIES[name])
        assert outcome.finished, f"{name} did not terminate on {query}: {outcome.status} {outcome.detail}"
        assert outcome.answers == gold, f"{name} on {query}"


@pytest.mark.parametrize("engine_name", _engines("untabled", "lazy"))
@SETTINGS
@given(catalogues())
def test_untabled_rewriting_loops_exactly_when_a_cycle_is_in_reach(engine_name: str, kb: KB) -> None:
    """On a rewrite cycle through the query, untabled rewriting never stops; with no cycle in reach it stops correctly."""
    query = kb.queries[0]
    looping = _on_rewrite_cycle(kb, query.pred)
    acyclic = _acyclic_from(kb, query.pred)
    event("looping case" if looping else "acyclic case" if acyclic else "no claim")
    if not (looping or acyclic):
        return
    rewriter = _rewriter(kb, 150_000, engine_name)
    for name in UNTABLED:
        outcome = rewriter.ask(query, STRATEGIES[name])
        if looping:
            assert outcome.status in ("work_limit", "time_limit"), f"{name} stopped on a cycle through {query}"
        else:
            assert outcome.finished, f"{name} did not stop on an acyclic {query}: {outcome.status}"
            assert outcome.answers == kb.answers(query), f"{name} on acyclic {query}"


@pytest.mark.parametrize("engine_name", _engines("depth", "lazy"))
@SETTINGS
@given(catalogues())
def test_depth_bounded_rewriting_stops_and_is_sound(engine_name: str, kb: KB) -> None:
    """A depth bound always stops the search, what it finds is true, and tables change only its cost."""
    query = kb.queries[0]
    rewriter = _rewriter(kb, 2_000_000, engine_name)
    caps = make(engine_name).capabilities
    by_bound: dict[tuple[int, str], list[tuple[str, frozenset]]] = defaultdict(list)
    for name in DEPTH:
        if not all(getattr(caps, c) for c, need in STRATEGIES[name].needs().items() if need):
            continue
        outcome = rewriter.ask(query, STRATEGIES[name])
        assert outcome.finished, f"{name} on {query}: {outcome.status}"
        assert outcome.answers <= kb.answers(query), f"{name} on {query}"
        by_bound[STRATEGIES[name].depth, STRATEGIES[name].depth_counts].append((name, frozenset(outcome.answers)))
    for bound, found in by_bound.items():
        assert len({answers for _, answers in found}) == 1, f"depth {bound} on {query}: {found}"
