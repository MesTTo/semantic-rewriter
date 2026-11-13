"""Purpose: tests of answer lineage: on random small programs, each answer's lineage is exactly the minimal sets of
rewrites whose program alone derives it, found by running the Python engine on every subset of the rewrites; its
probability and second moment equal the sums over those subsets; a lineage kept to k sets keeps the k most probable;
and a work budget too small stops the evaluation unfinished.
"""

from __future__ import annotations

import itertools
import math

from hypothesis import given, settings
from hypothesis import strategies as st

from semrewrite.engines.python import PythonEngine
from semrewrite.lineage import lineages, minimise, moments, probability
from semrewrite.model import Atom, Rewrite, Rule
from semrewrite.protocols import Budget, Program

CONSTANTS = ("a", "b", "c")
PREDICATES = ("p", "q", "r", "s")
TRANSITIVE = Rule(Atom("p", ("X", "Z")), (Atom("p", ("X", "Y")), Atom("p", ("Y", "Z"))))
GENEROUS = Budget(work=10_000_000, seconds=math.inf)


@st.composite
def programs(draw):
    """A program over four binary predicates and three constants: some facts, up to four rewrites with scores, and
    sometimes p transitive."""
    facts = draw(st.sets(st.tuples(st.sampled_from(PREDICATES), st.sampled_from(CONSTANTS), st.sampled_from(CONSTANTS)),
                         max_size=8))
    keys = draw(st.sets(st.tuples(st.sampled_from(PREDICATES), st.sampled_from(PREDICATES), st.booleans()),
                        max_size=4))
    rewrites = []
    for conclusion, premise, swap in sorted(keys):
        if conclusion == premise and not swap:
            continue
        score = draw(st.floats(min_value=0.05, max_value=0.95))
        rewrites.append(Rewrite(conclusion, premise, (1, 0) if swap else (0, 1), round(score, 3)))
    rules = (TRANSITIVE,) if draw(st.booleans()) else ()
    return Program(facts=tuple(Atom(p, (x, y)) for p, x, y in sorted(facts)), rules=rules, rewrites=tuple(rewrites))


def _derives(program: Program, subset: tuple[Rewrite, ...], query: Atom) -> set[tuple[str, ...]]:
    s = PythonEngine().solve(Program(facts=program.facts, rules=program.rules, rewrites=subset), query, GENEROUS)
    assert s.status == "ok"
    return s.answers


def _brute(program: Program, query: Atom) -> dict[tuple[str, ...], list[frozenset]]:
    """For each answer, every subset of the rewrites whose program alone derives it."""
    found: dict[tuple[str, ...], list[frozenset]] = {}
    for n in range(len(program.rewrites) + 1):
        for subset in itertools.combinations(program.rewrites, n):
            for row in _derives(program, subset, query):
                found.setdefault(row, []).append(frozenset(rw.key for rw in subset))
    return found


@settings(max_examples=60)
@given(programs(), st.sampled_from(CONSTANTS))
def test_each_answer_s_lineage_is_the_minimal_rewrite_sets_that_derive_it(program: Program, constant: str) -> None:
    query = Atom("p", ("X", constant))
    weights = {rw.key: rw.score for rw in program.rewrites}
    got = lineages(program, query, weights, k=64, work=10_000_000)
    assert got.finished
    expected = {row: minimise(sets) for row, sets in _brute(program, query).items()}
    assert got.answers == expected


@settings(max_examples=60)
@given(programs(), st.sampled_from(CONSTANTS))
def test_probability_and_second_moment_equal_the_sums_over_rewrite_subsets(program: Program, constant: str) -> None:
    query = Atom("p", ("X", constant))
    rewrites = program.rewrites
    mean = {rw.key: rw.score for rw in rewrites}
    # A Beta with that mean and four pseudo-observations: E[p^2] = m (m n + 1) / (n + 1) with n = 4.
    square = {key: m * (m * 4 + 1) / 5 for key, m in mean.items()}
    got = lineages(program, query, mean, k=64, work=10_000_000)
    worlds = {}
    for n in range(len(rewrites) + 1):
        for subset in itertools.combinations(rewrites, n):
            worlds[frozenset(rw.key for rw in subset)] = _derives(program, subset, query)
    keys = [rw.key for rw in rewrites]
    for row, lin in got.answers.items():
        holds = [w for w, answers in worlds.items() if row in answers]
        p = sum(math.prod(mean[k] if k in w else 1 - mean[k] for k in keys) for w in holds)
        second = 0.0
        for w1 in holds:
            for w2 in holds:
                factor = 1.0
                for k in keys:
                    if k in w1 and k in w2:
                        factor *= square[k]
                    elif k in w1 or k in w2:
                        factor *= mean[k] - square[k]
                    else:
                        factor *= 1 - 2 * mean[k] + square[k]
                second += factor
        assert math.isclose(probability(lin, mean), p, abs_tol=1e-12)
        assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(moments(lin, mean, square), (p, second), strict=True))


def test_a_rewrite_reused_down_a_chain_counts_once() -> None:
    """Five generations through one rewrite need that rewrite alone, so the answer's probability is the rewrite's."""
    people = ["g0", "g1", "g2", "g3", "g4", "g5"]
    facts = tuple(Atom("mother", (child, parent)) for parent, child in itertools.pairwise(people))
    rules = (Rule(Atom("descends", ("X", "Y")), (Atom("parent", ("Y", "X")),)),
             Rule(Atom("descends", ("X", "Z")), (Atom("parent", ("Y", "X")), Atom("descends", ("Y", "Z")))))
    mother = Rewrite("parent", "mother", (1, 0), 0.99)
    program = Program(facts=facts, rules=rules, rewrites=(mother,))
    got = lineages(program, Atom("descends", ("X", "g0")), {mother.key: 0.99}, k=16, work=1_000_000)
    assert set(got.answers) == {(p, "g0") for p in people[1:]}
    assert all(lin == frozenset({frozenset({mother.key})}) for lin in got.answers.values())
    assert probability(got.answers[("g5", "g0")], {mother.key: 0.99}) == 0.99


def test_a_bounded_lineage_keeps_the_most_probable_sets_and_a_small_budget_stops_unfinished() -> None:
    weak, strong = Rewrite("q", "r", (0, 1), 0.2), Rewrite("q", "s", (0, 1), 0.9)
    program = Program(facts=(Atom("r", ("a", "b")), Atom("s", ("a", "b"))), rewrites=(weak, strong))
    weights = {weak.key: 0.2, strong.key: 0.9}
    both = lineages(program, Atom("q", ("X", "Y")), weights, k=2, work=1_000).answers[("a", "b")]
    assert both == frozenset({frozenset({weak.key}), frozenset({strong.key})})
    one = lineages(program, Atom("q", ("X", "Y")), weights, k=1, work=1_000).answers[("a", "b")]
    assert one == frozenset({frozenset({strong.key})})
    assert not lineages(program, Atom("q", ("X", "Y")), weights, k=2, work=0).finished
