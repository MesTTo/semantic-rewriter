"""Purpose: property tests of when semantic rewriting stops, over generated
catalogues that always contain an alias cycle.

The claims, each checked against a Python fixpoint over the gold rules:

| strategy | claim |
|---|---|
| every tabled strategy | terminates and answers exactly the fixpoint, on every catalogue |
| untabled (naive, memo-untabled) | never terminates when the query predicate lies on a cycle of rewrites |
| untabled (naive, memo-untabled) | terminates with the fixpoint when everything the query reaches is acyclic |
| depth-bounded | terminates, and every answer is a gold answer |

The judge is an oracle that admits only the direct gold triples, so a chain
p0 <= p1 <= p2 must be found by discovering p1 and then p2: recursive
discovery is exercised, not handed over as a closure.
"""

from __future__ import annotations

from collections import defaultdict

from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from semrewrite.engine import STRATEGIES, Budget, Evaluation, Rewriter
from semrewrite.judging import OracleJudge
from semrewrite.kb import KB, Lit, Query, Rule

CONSTANTS = ("c0", "c1", "c2", "c3")
TABLED = [n for n, s in STRATEGIES.items() if s.evaluation is Evaluation.TABLED and s.timing.value != "none"]
UNTABLED = [n for n, s in STRATEGIES.items() if s.evaluation is Evaluation.UNTABLED]


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
        rules = (Rule(Lit(head, "X", "Z"), (Lit(left, "X", "Y"), Lit(right, "Y", "Z"))),)
    term = st.sampled_from(("X",) + CONSTANTS)
    qa, qb = draw(term), draw(st.sampled_from(("Y",) + CONSTANTS))
    query = Query(draw(pred), qa, qb)
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


def _rewriter(kb: KB, inferences: int) -> Rewriter:
    return Rewriter(kb, OracleJudge.of(kb.gold), None, budget=Budget(inferences=inferences, seconds=60))


SETTINGS = settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@SETTINGS
@given(catalogues())
def test_tabled_strategies_reach_exactly_the_fixpoint(kb: KB) -> None:
    """Every tabled strategy stops inside the budget with exactly the fixpoint's answers."""
    query = kb.queries[0]
    gold = kb.answers(query)
    event("query on a rewrite cycle" if _on_rewrite_cycle(kb, query.pred) else
          "query reaches no cycle" if _acyclic_from(kb, query.pred) else "query reaches a cycle off itself")
    rewriter = _rewriter(kb, inferences=2_000_000)
    for name in TABLED:
        outcome = rewriter.ask(STRATEGIES[name], query)
        assert outcome.terminated, f"{name} did not terminate on {query}: {outcome.status} {outcome.detail}"
        assert outcome.answers == gold, f"{name} on {query}"


@SETTINGS
@given(catalogues())
def test_untabled_rewriting_loops_exactly_when_a_cycle_is_in_reach(kb: KB) -> None:
    """On a rewrite cycle through the query, untabled rewriting never stops; with no cycle in reach it stops correctly."""
    query = kb.queries[0]
    looping = _on_rewrite_cycle(kb, query.pred)
    acyclic = _acyclic_from(kb, query.pred)
    event("looping case" if looping else "acyclic case" if acyclic else "no claim")
    if not (looping or acyclic):
        return
    rewriter = _rewriter(kb, inferences=150_000)
    for name in UNTABLED:
        outcome = rewriter.ask(STRATEGIES[name], query)
        if looping:
            assert outcome.status in ("inference_limit", "time_limit"), f"{name} stopped on a cycle through {query}"
        else:
            assert outcome.terminated, f"{name} did not stop on an acyclic {query}: {outcome.status}"
            assert outcome.answers == kb.answers(query), f"{name} on acyclic {query}"


@SETTINGS
@given(catalogues())
def test_depth_bounded_rewriting_stops_and_is_sound(kb: KB) -> None:
    """A depth bound always stops the search, and what it finds is true."""
    query = kb.queries[0]
    outcome = _rewriter(kb, inferences=2_000_000).ask(STRATEGIES["depth-bounded"], query)
    assert outcome.terminated, f"depth-bounded on {query}: {outcome.status}"
    assert outcome.answers <= kb.answers(query)
