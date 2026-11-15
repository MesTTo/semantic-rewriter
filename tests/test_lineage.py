"""Purpose: tests of answer lineage: on random small programs, each answer's lineage is exactly the minimal sets of
rewrites whose program alone derives it, found by running the Python engine on every subset of the rewrites; its
probability and second moment equal the sums over those subsets; with exception leaves on some applications, they equal
the sums over every world of rewrite validities and leaves, each world run with the leaves' applications guarded, and a
defeated application derives nothing; an exception never moves an answer whose probability does not depend on its
application, and moves the others by its leaf's weight alone; a lineage kept to k sets keeps the k most probable; and
a work budget too small stops the evaluation unfinished.
"""

from __future__ import annotations

import itertools
import math

from hypothesis import given, settings
from hypothesis import strategies as st

from semrewrite.engines.python import PythonEngine
from semrewrite.lineage import Leaf, leaf, lineages, minimise, moments, probability
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


@st.composite
def concluded(draw):
    """A program as `programs` draws it with at least one rewrite, a fact one of its rewrites applies to, and a query
    for that rewrite's conclusion, its first argument free and its second the constant the application concludes for,
    so every case applies a rewrite on the way to an answer."""
    program = draw(programs().filter(lambda p: p.rewrites))
    rw = draw(st.sampled_from(program.rewrites))
    row = (draw(st.sampled_from(CONSTANTS)), draw(st.sampled_from(CONSTANTS)))
    facts = tuple(sorted(set(program.facts) | {Atom(rw.premise, tuple(row[m] for m in rw.mapping))},
                         key=lambda f: (f.pred, f.args)))
    return Program(facts=facts, rules=program.rules, rewrites=program.rewrites), Atom(rw.conclusion, ("X", row[1]))


def _excepted(program: Program, query: Atom, data) -> dict:
    """Up to three of the applications the lineage of `query` makes, each with a leaf: a weight in (0.05, 0.95), or
    0 for a defeated application."""
    weights = {rw.key: rw.score for rw in program.rewrites}
    applied = lineages(program, query, weights, k=64, work=10_000_000).applied
    made = sorted(leaf(key, row) for key, rows in applied.items() for row in rows)
    chosen = data.draw(st.lists(st.sampled_from(made), min_size=1, max_size=3, unique=True))
    return {v: data.draw(st.sampled_from([0.0, 0.3, 0.75])) for v in sorted(chosen)}


def _guarded(program: Program, valid: tuple[Rewrite, ...], normal: set[Leaf], leaves: set[Leaf]) -> Program:
    """The program of one world: the rewrites `valid`, each one with leaves turned into its rule guarded by a fact per
    row it may apply to, every row of the constants but its leaves' rows outside `normal`."""
    facts, rules, rewrites = list(program.facts), list(program.rules), []
    keyed = {(v[0], v[1], v[2]) for v in leaves}
    for i, rw in enumerate(valid):
        if rw.key not in keyed:
            rewrites.append(rw)
            continue
        rule = rw.rule()
        guard = f"ok{i}"
        rules.append(Rule(rule.head, (*rule.body, Atom(guard, rule.head.args))))
        facts += [Atom(guard, (x, y)) for x in CONSTANTS for y in CONSTANTS
                  if leaf(rw.key, (x, y)) not in leaves or leaf(rw.key, (x, y)) in normal]
    return Program(facts=tuple(facts), rules=tuple(rules), rewrites=tuple(rewrites))


# Time: 2^r rewrite subsets times 2^l leaf subsets, one engine solve each; r <= 4 and l <= 3, so at most 128 solves.
def _worlds(program: Program, weights: dict, query: Atom, square: dict) -> dict[tuple[str, ...], tuple[float, float]]:
    """Each row's probability and second moment by brute force: every world of valid rewrites and normal leaves, a
    defeated leaf never normal, weighted by its variables' means, the second moment by their pairwise products."""
    leaves = {v for v in weights if len(v) == 4}
    live = sorted(v for v in leaves if weights[v] > 0.0)
    worlds = []
    for n in range(len(program.rewrites) + 1):
        for valid in itertools.combinations(program.rewrites, n):
            for m in range(len(live) + 1):
                for normal in itertools.combinations(live, m):
                    held = {rw.key for rw in valid} | set(normal)
                    world = _guarded(program, valid, set(normal), leaves)
                    answers = _derives(world, world.rewrites, query)
                    worlds.append((held, answers))
    variables = [rw.key for rw in program.rewrites] + live
    out: dict[tuple[str, ...], tuple[float, float]] = {}
    for row in {r for _, answers in worlds for r in answers}:
        holds = [w for w, answers in worlds if row in answers]
        p = sum(math.prod(weights[v] if v in w else 1 - weights[v] for v in variables) for w in holds)
        second = 0.0
        for w1 in holds:
            for w2 in holds:
                factor = 1.0
                for v in variables:
                    if v in w1 and v in w2:
                        factor *= square[v]
                    elif v in w1 or v in w2:
                        factor *= weights[v] - square[v]
                    else:
                        factor *= 1 - 2 * weights[v] + square[v]
                second += factor
        out[row] = (p, second)
    return out


@settings(max_examples=80, deadline=None)
@given(concluded(), st.data())
def test_a_lineage_with_exception_leaves_counts_every_world_of_rewrites_and_leaves(drawn, data) -> None:
    """Each answer's probability and second moment equal the sums over every world of valid rewrites and normal
    leaves, and an answer no world derives, every derivation of which a defeated application carries, is none."""
    program, query = drawn
    leaves = _excepted(program, query, data)
    weights = {rw.key: rw.score for rw in program.rewrites} | leaves
    square = {v: m * (m * 4 + 1) / 5 for v, m in weights.items()}
    got = lineages(program, query, weights, k=64, work=10_000_000)
    assert got.finished
    brute = _worlds(program, weights, query, square)
    assert set(got.answers) == {row for row, (p, _) in brute.items() if p > 0.0}
    for row, lin in got.answers.items():
        p, second = brute[row]
        assert math.isclose(probability(lin, weights), p, abs_tol=1e-12)
        assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(moments(lin, weights, square), (p, second),
                                                                      strict=True))
    # A defeat removes the applications its derivations fed and adds none.
    plain = lineages(program, query, {rw.key: rw.score for rw in program.rewrites}, k=64, work=10_000_000).applied
    assert all(rows <= plain[key] for key, rows in got.applied.items())


@settings(max_examples=80, deadline=None)
@given(concluded(), st.data())
def test_an_exception_never_moves_an_answer_not_derived_through_its_application(drawn, data) -> None:
    """With a leaf of weight w on one application the lineage makes, an answer whose probability is the same with the
    application always normal and never is keeps its lineage exactly; every other answer's probability is w times
    the first plus 1 - w times the second, so the leaf moves it by its own weight alone."""
    program, query = drawn
    weights = {rw.key: rw.score for rw in program.rewrites}
    before = lineages(program, query, weights, k=64, work=10_000_000)
    applications = sorted((key, row) for key, rows in before.applied.items() for row in rows)
    key, row = data.draw(st.sampled_from(applications))
    w = data.draw(st.sampled_from([0.0, 0.25, 0.8]))
    exception = leaf(key, row)
    after = lineages(program, query, {**weights, exception: w}, k=64, work=10_000_000)
    square = {v: m * (m * 4 + 1) / 5 for v, m in {**weights, exception: 1.0}.items()}
    normal = _worlds(program, {**weights, exception: 1.0}, query, square)
    blocked = _worlds(program, {**weights, exception: 0.0}, query, square)
    for answer, lin in before.answers.items():
        p1, p0 = normal[answer][0], blocked.get(answer, (0.0, 0.0))[0]
        if math.isclose(p1, p0, abs_tol=1e-12):
            assert after.answers[answer] == lin
        else:
            assert math.isclose(probability(after.answers.get(answer, frozenset()), {**weights, exception: w}),
                                w * p1 + (1 - w) * p0, abs_tol=1e-12)
