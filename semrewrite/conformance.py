"""Purpose: the engine conformance kit: the property checks that decide whether an engine
implements semrewrite.protocols.Engine, the Hypothesis strategies generating their
cases, and a runner that reports each property's verdict.

    from semrewrite.conformance import run
    from semrewrite.engines import make
    print(run(make("python"), max_examples=200).summary())

or from a shell, `python -m semrewrite.conformance --engine python`.

| property | needs | claim |
|---|---|---|
| tabling | tabling | with tables and no depth bound, answers equal the reference fixpoint |
| depth | depth | with a bound of 1 to 4 steps, answers equal the reference cut at that height, tables on or off (untabled runs deeper than 1 may stop) |
| rewrite-depth | rewrite_depth | with a bound of 1 to 4 rewrites, answers equal the reference cut at that many rewrites and recursion through user rules answers completely, tables on or off (untabled runs may stop) |
| lazy | lazy, and tabling or a depth bound | rewrites given through expand answer as rewrites given up front; expand sees only the program's predicates |
| budget | | a budget too small ends with a limit status and no answers; a run reporting ok stayed in its budget |
| names | tabling or a depth bound | predicates and constants spelled like reserved words keep their answers, eagerly and lazily |
| untabled | untabled | untabled, a query on a rewrite cycle hits a limit and one reaching no cycle answers exactly |
| proofs | proofs, and tabling or a depth bound | every answer has a proof deriving it from the program within the depth bound |
| rounds | reached and tabling | evaluating, adding the rewrites of every predicate a run reached and evaluating again until a run reaches nothing new gives the reference answers |

The reference is semrewrite.fixpoint, a naive evaluator sharing no code with any
engine. Generated programs have 2 to 6 predicates of arity 0 to 3 over four
constants, a planted rewrite cycle p <=> q, random rewrites under any argument
permutation, up to ten distinct facts, up to three user rules of one to three body
atoms, constants and repeated variables anywhere in their heads and bodies, and a
query binding some of its arguments. The rewrite-depth property also runs on chains of 6 to 12 nodes, whose
transitive closure a recursive user rule derives in up to 11 steps, more than any
bound it tries.

A run must finish with exactly the reference answers when it is tabled, or when its
depth bound is 1 step, whose untabled search is a few hundred clause tries on these
cases. Such a run gets the protocol's default work and no time limit, so the verdict
never depends on how loaded the machine is, while the work limit still stops an
engine that does not terminate. An untabled search deeper than that tries every derivation separately, which
through a join of well-derived predicates grows doubly exponentially with the bound
and is a cost no budget in an engine's own unit can be promised to cover, so such a
run may instead stop at a limit, with no answers; so may an untabled run bounding
rewrites alone, which leaves recursion through user rules unbounded, and when such a
run can reach a cycle of user rules it gets the loop budget the untabled property
gives a query on a cycle. Every limit a run reports must be one it reached: work at
least the budget's, or seconds at least the budget's. A passing verdict counts the
runs that were allowed to stop and the ones that did.

Where an engine declares reached, the tabling, lazy and names properties also check
that each run reports the query's predicate and only predicates of the program: the
rounds strategy expands what an engine reports, so a name the engine made up
internally, such as max_ for max, would send the rewriter after a predicate that
doesn't exist. For the same reason expand may only be called with program predicates.
The rounds property checks the other side, that a run reports every predicate it
called, by running the rounds strategy itself: a predicate missing from reached never
has its rewrites added, and the last round then lacks the answers they give.

Each property is a function taking an engine and a case that raises
ConformanceFailure, an AssertionError, when the engine breaks the claim, and
MissingCapability, a unittest.SkipTest that pytest and Hypothesis both treat as a
skip, when the engine lacks what the claim needs. Tests can call them inside
@given with the matching strategy, or call check() or run(), which search with
hypothesis.find, shrink a failing case and report it; check() can also split one
seeded search into shards that run in parallel and together check exactly its
cases. Hypothesis is imported only inside the functions that generate or search,
so importing this module never needs it.
"""

from __future__ import annotations

import argparse
import contextvars
import math
import random
import traceback
import unittest
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, NoReturn

from semrewrite import fixpoint
from semrewrite.engines import REGISTRY, make
from semrewrite.model import Atom, Key, Rewrite, Rule, identity, inverse, is_variable, mappings
from semrewrite.protocols import Budget, Capabilities, DepthCounts, Engine, Program, Proof, Solution

if TYPE_CHECKING:
    from types import ModuleType

    from hypothesis.strategies import DrawFn, SearchStrategy

Row = tuple[str, ...]

#: The default predicate and constant alphabets of generated cases.
PREDICATES: tuple[str, ...] = tuple(f"p{i}" for i in range(6))
CONSTANTS: tuple[str, ...] = ("c0", "c1", "c2", "c3")

#: Names that MeTTa, Prolog or SQL reserve; the names property spells predicates and constants with them.
RESERVED: tuple[str, ...] = ("max", "min", "if", "let", "holds", "fact", "via", "rw", "select", "from", "not", "is", "true")

#: The variables of generated rules, and the ones a generated query may use.
VARIABLES: tuple[str, ...] = ("X", "Y", "Z", "W")
QUERY_VARIABLES: tuple[str, ...] = ("X", "Y", "Z")

#: The bounds, in counted steps, that the depth and rewrite-depth properties try, and the lazy, names and proofs
#: properties try counting steps.
DEPTHS: tuple[int, ...] = (1, 2, 3, 4)

#: The one bound on rewrites the lazy, names and proofs properties also try. What a bound counts is independent of
#: what those properties check, and the rewrite-depth property tries every depth, so one bound covers each pairing
#: of the two with a quarter of the rewrite-bounded runs; 2 needs a user rule's level closed between two rewrite levels.
REWRITE_BOUND: int = 2

#: The statuses of a run stopped by its budget.
LIMITS: frozenset[str] = frozenset({"work_limit", "time_limit"})

#: The budget a run expecting to finish gets: the protocol's default work and no time limit.
GENEROUS: Budget = Budget(seconds=math.inf)

#: A work budget no engine can answer more than 50 answers in, whatever its unit, while
#: each answer costs at least one unit of it; the budget property's chains need 59 or more.
TOO_SMALL: Budget = Budget(work=50, seconds=60.0)

#: The budget an untabled run that may loop gets: a loop renaming one goal exceeds 20,000
#: units of work in milliseconds on any engine counting inferences, and the second bounds
#: what a looping case costs an engine whose unit is coarser.
LOOP: Budget = Budget(work=20_000, seconds=1.0)

#: The budget an untabled bounded run that may stop gets: a second bounds what a search
#: too large to finish costs the kit.
SEARCH: Budget = Budget(work=GENEROUS.work, seconds=LOOP.seconds)

# While check() runs, the depth-bounded runs it allowed to stop and the ones that stopped.
_ALLOWED_TO_STOP: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar("allowed_to_stop", default=None)


class ConformanceFailure(AssertionError):
    """The engine broke a property's claim on a case; the message says how."""


class MissingCapability(unittest.SkipTest):
    """The engine lacks a capability the property needs, so the property is skipped, not failed."""


def _hypothesis() -> ModuleType:
    """The hypothesis package, or an ImportError saying how to install it."""
    try:
        import hypothesis
        import hypothesis.errors
        import hypothesis.strategies
    except ImportError as error:
        msg = "the conformance kit needs hypothesis: pip install 'semantic-rewriter[conformance]'"
        raise ImportError(msg, name="hypothesis") from error
    return hypothesis


@dataclass(frozen=True)
class Case:
    """A generated program with its query.

    `answers` holds the answers without a depth bound when the generator knows them in
    closed form, as for chain closures, whose reference fixpoint is too slow to compute
    for every case; otherwise the reference fixpoint computes them.
    """

    program: Program
    query: Atom
    answers: frozenset[Row] | None = None

    def predicates(self) -> frozenset[str]:
        """Every predicate the case mentions: in facts, rules, rewrites and the query."""
        names = {self.query.pred}
        names.update(f.pred for f in self.program.facts)
        for rule in self.program.rules:
            names.add(rule.head.pred)
            names.update(a.pred for a in rule.body)
        for rewrite in self.program.rewrites:
            names.update((rewrite.conclusion, rewrite.premise))
        return frozenset(names)

    def lines(self) -> list[str]:
        """The case as Prolog-like text: facts, rules, rewrites and the query."""
        out = [f"facts: {' '.join(f'{f}.' for f in self.program.facts) or '(none)'}"]
        out += [f"rule: {rule}" for rule in self.program.rules]
        out += [f"rewrite: {rewrite}" for rewrite in self.program.rewrites]
        out.append(f"query: {self.query}")
        return out

    def __str__(self) -> str:
        return "\n".join(self.lines())


def reference(program: Program, query: Atom) -> set[Row]:
    """The answers of fixpoint.least_model over the facts, the rules and every rewrite's rule, cut at program.depth
    counted steps; with depth_counts "rewrites" the user rules are free, so only rewrite steps count."""
    rewrites = tuple(rewrite.rule() for rewrite in program.rewrites)
    if program.depth_counts == "rewrites":
        model = fixpoint.least_model(program.facts, rewrites, program.depth, free=program.rules)
    else:
        model = fixpoint.least_model(program.facts, (*program.rules, *rewrites), program.depth)
    return fixpoint.answers(model, query)


def expected(case: Case, depth: int = 0, counts: DepthCounts = "steps") -> set[Row]:
    """The answers the case must give at `depth` counted steps: its closed form when it has one and the bound can't
    cut it, which a bound on rewrites can't in a program without rewrites, else the reference."""
    uncut = depth == 0 or (counts == "rewrites" and not case.program.rewrites)
    if uncut and case.answers is not None:
        return set(case.answers)
    return reference(replace(case.program, depth=depth, depth_counts=counts), case.query)


# Strategies ---------------------------------------------------------------------------


def cases(predicates: Sequence[str] = PREDICATES, constants: Sequence[str] = CONSTANTS) -> SearchStrategy[Case]:
    """Random programs with a planted rewrite cycle, as the module docstring describes.

    Predicate names are drawn from `predicates` and the four constants from
    `constants`, so the names property passes RESERVED for both and a constant can
    share its spelling with a predicate.
    """
    if len(set(predicates)) < 2 or len(set(constants)) < 4:
        msg = "cases need at least 2 predicate names and 4 constants to draw from"
        raise ValueError(msg)
    st = _hypothesis().strategies

    @st.composite
    def draw_case(draw: DrawFn) -> Case:
        n = draw(st.integers(2, min(6, len(set(predicates)))))
        # Slices of permutations give distinct names without the rejections unique lists cost.
        names = draw(st.permutations(list(dict.fromkeys(predicates))))[:n]
        consts = tuple(draw(st.permutations(list(dict.fromkeys(constants))))[:4])
        arity = {p: draw(st.sampled_from((0, 1, 2, 3))) for p in names}
        # The planted cycle: names[0] <=> names[1] under a random permutation and its inverse.
        a, b = names[0], names[1]
        arity[b] = arity[a]
        planted = draw(st.sampled_from(list(mappings(arity[a]))))
        rewrites = [Rewrite(a, b, planted), Rewrite(b, a, inverse(planted))]
        for _ in range(draw(st.integers(0, 2 * n))):
            conclusion = draw(st.sampled_from(names))
            premise = draw(st.sampled_from([p for p in names if arity[p] == arity[conclusion]]))
            mapping = draw(st.sampled_from(list(mappings(arity[conclusion]))))
            if premise != conclusion or mapping != identity(len(mapping)):
                rewrites.append(Rewrite(conclusion, premise, mapping))
        facts = []
        for _ in range(draw(st.integers(1, 10))):
            pred = draw(st.sampled_from(names))
            facts.append(Atom(pred, tuple(draw(st.sampled_from(consts)) for _ in range(arity[pred]))))
        rules = [_rule(draw, names, arity, consts) for _ in range(draw(st.integers(0, 3)))]
        pred = draw(st.sampled_from(names))
        terms = st.sampled_from(QUERY_VARIABLES + consts)
        query = Atom(pred, tuple(draw(terms) for _ in range(arity[pred])))
        program = Program(facts=tuple(dict.fromkeys(facts)), rules=tuple(rules), rewrites=tuple(rewrites))
        return Case(program, query)

    return draw_case()


def _rule(draw: DrawFn, names: Sequence[str], arity: dict[str, int], consts: Sequence[str]) -> Rule:
    """A rule of one to three body atoms whose arguments are variables or constants, repeats allowed, and whose head
    takes the body's variables and the constants; a body variable shared by two atoms makes a join."""
    st = _hypothesis().strategies
    term = st.one_of(st.sampled_from(VARIABLES), st.sampled_from(consts))
    body = []
    for _ in range(draw(st.integers(1, 3))):
        pred = draw(st.sampled_from(names))
        body.append(Atom(pred, tuple(draw(term) for _ in range(arity[pred]))))
    bound = sorted({t for a in body for t in a.args if is_variable(t)})
    head_term = st.one_of(st.sampled_from(bound), st.sampled_from(consts)) if bound else st.sampled_from(consts)
    head = draw(st.sampled_from(names))
    return Rule(Atom(head, tuple(draw(head_term) for _ in range(arity[head]))), tuple(body))


def chains(nodes: tuple[int, int] = (60, 90)) -> SearchStrategy[Case]:
    """Transitive closures of chains of nodes[0] to nodes[1] nodes, whose answers are known in closed form.

    path(X, Y) :- edge(X, Y) and path(X, Z) :- edge(X, Y), path(Y, Z) over edges
    n0 -> n1 -> ... derive path(ni, nj) exactly for i < j, so the case carries those
    answers instead of waiting on the naive reference, which takes O(n^4) here. The
    default of 60 nodes or more gives every query shape but path(X, X) at least 59
    answers, more than TOO_SMALL's 50 units of work can produce.
    """
    if nodes[0] < 2 or nodes[0] > nodes[1]:
        msg = f"a chain needs at least 2 nodes and a non-empty range, got {nodes}"
        raise ValueError(msg)
    st = _hypothesis().strategies

    @st.composite
    def draw_chain(draw: DrawFn) -> Case:
        n = draw(st.integers(*nodes))
        names = [f"n{i}" for i in range(n)]
        facts = tuple(Atom("edge", (names[i], names[i + 1])) for i in range(n - 1))
        rules = (Rule(Atom("path", ("X", "Y")), (Atom("edge", ("X", "Y")),)),
                 Rule(Atom("path", ("X", "Z")), (Atom("edge", ("X", "Y")), Atom("path", ("Y", "Z")))))
        query = draw(st.sampled_from((Atom("path", ("X", "Y")), Atom("path", (names[0], "Y")),
                                      Atom("path", ("X", names[-1])), Atom("path", ("X", "X")))))
        closure = ((names[i], names[j]) for i in range(n) for j in range(i + 1, n))
        answers = frozenset(row for row in closure if query.match(row) is not None)
        return Case(Program(facts=facts, rules=rules), query, answers)

    return draw_chain()


def _rewrite_depth_cases() -> SearchStrategy[Case]:
    """Generated programs, and chains of 6 to 12 nodes, whose closure a recursive user rule derives in more steps
    than any bound in DEPTHS."""
    return _hypothesis().strategies.one_of(cases(), chains((6, 12)))


# Capabilities -------------------------------------------------------------------------


def _exact(c: Capabilities) -> str | None:
    if c.tabling or c.depth or c.rewrite_depth:
        return None
    return "the engine has neither tabling nor a depth bound, so no run of it can be compared with the reference"


def _capability(name: str, then: Callable[[Capabilities], str | None] | None = None) -> Callable[[Capabilities], str | None]:
    """The skip reason of a property needing capability `name`, and then whatever `then` needs."""

    def needs(c: Capabilities) -> str | None:
        if not getattr(c, name):
            return f"the engine lacks the {name} capability"
        return then(c) if then else None

    return needs


def _nothing(c: Capabilities) -> str | None:
    return None


def _require(engine: Engine, needs: Callable[[Capabilities], str | None]) -> None:
    reason = needs(engine.capabilities)
    if reason is not None:
        raise MissingCapability(reason)


def _exact_bounds(c: Capabilities) -> list[tuple[int, DepthCounts]]:
    """The bounds at which the engine's answers can be compared with the reference: none with tabling, then each
    of DEPTHS counting steps with depth, and REWRITE_BOUND counting rewrites with rewrite_depth."""
    bounds: list[tuple[int, DepthCounts]] = [(0, "steps")] if c.tabling else []
    if c.depth:
        bounds += [(depth, "steps") for depth in DEPTHS]
    if c.rewrite_depth:
        bounds.append((REWRITE_BOUND, "rewrites"))
    return bounds


def _bound(program: Program) -> str:
    """How a report names the program's bound: no depth bound, depth 3, or rewrite depth 3."""
    if not program.depth:
        return "no depth bound"
    return f"{'rewrite depth' if program.depth_counts == 'rewrites' else 'depth'} {program.depth}"


def _tabled(c: Capabilities, depth: int) -> bool:
    """Ask for tables without a depth bound, and under one whenever the engine tables or can't run untabled."""
    return depth == 0 or c.tabling or not c.untabled


# Checks -------------------------------------------------------------------------------


def _fail(message: str) -> NoReturn:
    raise ConformanceFailure(message)


def _exactly(solution: Solution, want: set[Row], what: str) -> None:
    """Fail unless the run finished with exactly the answers `want`."""
    if solution.status != "ok":
        _fail(f"{what}: status {solution.status} ({solution.detail}), expected ok")
    got = set(solution.answers)
    if got != want:
        _fail(f"{what}: answers {sorted(got)} but the reference gives {sorted(want)}; "
              f"missing {sorted(want - got)}, unexpected {sorted(got - want)}")


def _must_finish(engine: Engine, program: Program) -> bool:
    """Whether a bounded run must finish: tabled where the engine tables, or bounded at one step counting steps."""
    return (program.tabled and engine.capabilities.tabling) or (program.depth == 1 and program.depth_counts == "steps")


def _budget(engine: Engine, program: Program, query: Atom) -> Budget:
    """GENEROUS for a bounded run that must finish; LOOP for an untabled run bounding rewrites alone that can reach a
    cycle of user rules, which its bound leaves free to loop; SEARCH for any other run that may stop."""
    if _must_finish(engine, program):
        return GENEROUS
    if program.depth_counts == "rewrites" and _reaches_cycle(program, query.pred, rules_only=True):
        return LOOP
    return SEARCH


def _bounded(engine: Engine, program: Program, budget: Budget, solution: Solution, want: set[Row], what: str) -> bool:
    """Judge a depth-bounded run: exactly `want`, or, where it may stop, a limit it reached with no answers.

    `budget` is the one the run had. Returns whether the run finished, so callers check
    what only a finished run reports.
    """
    tally = _ALLOWED_TO_STOP.get()
    may_stop = not _must_finish(engine, program)
    if tally is not None and may_stop:
        tally[0] += 1
    if solution.status == "ok" or not may_stop:
        _exactly(solution, want, what)
        return True
    _limited(solution, LIMITS, f"{what} (it may stop: an untabled search its bound does not keep small)", budget)
    if tally is not None:
        tally[1] += 1
    return False


def _reached(engine: Engine, solution: Solution, case: Case, what: str) -> None:
    """With the reached capability, fail unless the run reports the query's predicate and only program predicates."""
    if not engine.capabilities.reached:
        return
    if solution.reached is None:
        _fail(f"{what}: the engine declares the reached capability but reported no reached predicates")
    if case.query.pred not in solution.reached:
        _fail(f"{what}: reached {sorted(solution.reached)} lacks the query's predicate {case.query.pred!r}")
    strangers = sorted(set(solution.reached) - case.predicates())
    if strangers:
        _fail(f"{what}: reached includes {strangers}, which are not predicates of the program")


class _Expander:
    """An expander serving fixed rewrites by conclusion and recording every predicate it is asked about."""

    def __init__(self, rewrites: Iterable[Rewrite]) -> None:
        self.by_conclusion: dict[str, list[Rewrite]] = {}
        for rewrite in rewrites:
            self.by_conclusion.setdefault(rewrite.conclusion, []).append(rewrite)
        self.calls: list[str] = []

    def __call__(self, pred: str) -> tuple[Rewrite, ...]:
        self.calls.append(pred)
        return tuple(self.by_conclusion.get(pred, ()))


def _lazily(engine: Engine, case: Case, program: Program, eager: Sequence[Rewrite], lazy: Sequence[Rewrite],
            want: set[Row], what: str) -> None:
    """Solve with `eager` in program.rewrites and `lazy` behind expand, judging the run as _bounded does.

    It also fails when expand is asked about a name the case lacks, or a finished run reaches one.
    """
    expander = _Expander(lazy)
    split = replace(program, rewrites=tuple(eager))
    budget = _budget(engine, split, case.query)
    solution = engine.solve(split, case.query, budget, expander)
    strangers = sorted(set(expander.calls) - case.predicates())
    if strangers:
        _fail(f"{what}: expand was called with {strangers}, which are not predicates of the program; "
              "an engine that renames names internally must undo the renaming before calling expand")
    if _bounded(engine, split, budget, solution, want, what):
        _reached(engine, solution, case, what)


def check_tabling(engine: Engine, case: Case) -> None:
    """With tables and no depth bound the run finishes with exactly the reference answers, reporting what it reached."""
    _require(engine, TABLING.needs)
    program = replace(case.program, tabled=True, depth=0)
    solution = engine.solve(program, case.query, GENEROUS)
    _exactly(solution, expected(case), "tabled, no depth bound")
    _reached(engine, solution, case, "tabled, no depth bound")


def _check_bounds(engine: Engine, case: Case, counts: DepthCounts) -> None:
    """At each depth in DEPTHS counting `counts`, with tables and without where the engine can, the answers are the
    reference's under that bound; untabled runs may stop where _must_finish allows."""
    c = engine.capabilities
    modes = [tabled for tabled, has in ((True, c.tabling), (False, c.untabled)) if has] or [True]
    for depth in DEPTHS:
        want = expected(case, depth, counts)
        for tabled in modes:
            program = replace(case.program, tabled=tabled, depth=depth, depth_counts=counts)
            budget = _budget(engine, program, case.query)
            solution = engine.solve(program, case.query, budget)
            _bounded(engine, program, budget, solution, want, f"{_bound(program)}, {'tabled' if tabled else 'untabled'}")


def check_depth(engine: Engine, case: Case) -> None:
    """At each depth in DEPTHS, with tables and without where the engine can, the answers are the reference's at that depth.

    Untabled runs deeper than one step may stop instead, as the module docstring says.
    """
    _require(engine, DEPTH.needs)
    _check_bounds(engine, case, "steps")


def check_rewrite_depth(engine: Engine, case: Case) -> None:
    """At each bound in DEPTHS on rewrite steps alone, with tables and without where the engine can, the answers are
    the reference's under that bound. On a chain, whose closure only a recursive user rule derives, that is every
    path, however many steps it takes. Untabled runs may stop instead, as the module docstring says.
    """
    _require(engine, REWRITE_DEPTH.needs)
    _check_bounds(engine, case, "rewrites")


def check_lazy(engine: Engine, case: Case) -> None:
    """Rewrites given up front, through expand, or split between the two give the reference answers at every exact bound."""
    _require(engine, LAZY.needs)
    c = engine.capabilities
    rewrites = case.program.rewrites
    for depth, counts in _exact_bounds(c):
        want = expected(case, depth, counts)
        program = replace(case.program, tabled=_tabled(c, depth), depth=depth, depth_counts=counts)
        what = f"{_bound(program)}, every rewrite up front"
        budget = _budget(engine, program, case.query)
        solution = engine.solve(program, case.query, budget)
        if _bounded(engine, program, budget, solution, want, what):
            _reached(engine, solution, case, what)
        _lazily(engine, case, program, (), rewrites, want, f"{_bound(program)}, every rewrite through expand")
        _lazily(engine, case, program, rewrites[::2], rewrites[1::2], want,
                f"{_bound(program)}, alternate rewrites up front and through expand")


def check_budget(engine: Engine, case: Case) -> None:
    """Budgets are hard: a run past its work or seconds ends with a limit status and no answers.

    Four runs, in an exact mode where the engine has one (tables without a depth
    bound, or untabled when the query reaches no cycle): TOO_SMALL when the case has
    more than 50 answers, which must end in a limit; seconds=0, which must end in
    time_limit; a generous run, which must finish with the reference answers and
    reports its work W; and a run under work W // 2, which must end in a limit or else
    finish exactly within it. Without an exact mode only the first two run.
    """
    want = expected(case)
    c = engine.capabilities
    exact = None
    if c.tabling:
        exact = replace(case.program, tabled=True, depth=0)
    elif c.untabled and not _reaches_cycle(case.program, case.query.pred):
        exact = replace(case.program, tabled=False, depth=0)
    program = exact or case.program
    if len(want) > TOO_SMALL.work:
        _limited(engine.solve(program, case.query, TOO_SMALL), LIMITS, f"work budget {TOO_SMALL.work}", TOO_SMALL)
    instant = Budget(work=GENEROUS.work, seconds=0.0)
    _limited(engine.solve(program, case.query, instant), {"time_limit"}, "a budget of 0 seconds", instant)
    if exact is None:
        return
    full = engine.solve(exact, case.query, GENEROUS)
    _exactly(full, want, "a generous budget")
    half = Budget(work=full.work // 2, seconds=GENEROUS.seconds)
    rerun = engine.solve(exact, case.query, half)
    what = f"work budget {half.work}, half the {full.work} the same run reported"
    if rerun.status == "ok":
        if rerun.work > half.work:
            _fail(f"{what}: status ok after spending {rerun.work}, more than the budget")
        _exactly(rerun, want, what)
    else:
        _limited(rerun, LIMITS, what, half)


def _limited(solution: Solution, statuses: Iterable[str], what: str, budget: Budget) -> None:
    """Fail unless the run stopped with one of `statuses`, a limit of `budget` it reached, and returned no answers."""
    statuses = set(statuses)
    if solution.status not in statuses:
        _fail(f"{what}: status {solution.status} ({solution.detail}), expected {' or '.join(sorted(statuses))}")
    if solution.status == "work_limit" and solution.work < budget.work:
        _fail(f"{what}: reported work_limit after {solution.work} units of a {budget.work} budget")
    if solution.status == "time_limit" and solution.seconds < budget.seconds:
        _fail(f"{what}: reported time_limit after {solution.seconds:.3f}s of a {budget.seconds}s budget")
    if solution.answers:
        _fail(f"{what}: stopped with {solution.status} but returned {len(solution.answers)} answers, "
              "which would read as complete")


def check_names(engine: Engine, case: Case) -> None:
    """A case spelled with RESERVED names answers as the reference at every exact bound, eagerly and, when lazy, lazily."""
    _require(engine, NAMES.needs)
    c = engine.capabilities
    for depth, counts in _exact_bounds(c):
        want = expected(case, depth, counts)
        program = replace(case.program, tabled=_tabled(c, depth), depth=depth, depth_counts=counts)
        budget = _budget(engine, program, case.query)
        solution = engine.solve(program, case.query, budget)
        if _bounded(engine, program, budget, solution, want, f"{_bound(program)}, eager"):
            _reached(engine, solution, case, f"{_bound(program)}, eager")
        if c.lazy:
            _lazily(engine, case, program, (), program.rewrites, want, f"{_bound(program)}, every rewrite through expand")


def _graph(program: Program, rules: bool = True, rewrites: bool = True) -> dict[str, set[str]]:
    """Each predicate's dependencies: the body predicates of its rules and the premises of its rewrites, each when asked."""
    graph: dict[str, set[str]] = {}
    if rules:
        for rule in program.rules:
            graph.setdefault(rule.head.pred, set()).update(a.pred for a in rule.body)
    if rewrites:
        for rewrite in program.rewrites:
            graph.setdefault(rewrite.conclusion, set()).add(rewrite.premise)
    return graph


def _reachable(graph: dict[str, set[str]], start: str) -> set[str]:
    """The predicates reachable from `start` in one step or more."""
    seen: set[str] = set()
    stack = list(graph.get(start, ()))
    while stack:
        pred = stack.pop()
        if pred not in seen:
            seen.add(pred)
            stack.extend(graph.get(pred, ()))
    return seen


def _on_rewrite_cycle(program: Program, pred: str) -> bool:
    """Whether rewrites alone lead from `pred` back to itself, so untabled resolution renames its goal forever."""
    return pred in _reachable(_graph(program, rules=False), pred)


def _reaches_cycle(program: Program, pred: str, rules_only: bool = False) -> bool:
    """Whether some predicate `pred` depends on, itself included, lies on a cycle of rules and rewrites, or with
    `rules_only` on a cycle of user rules alone, which a bound on rewrites does not cut."""
    graph = _graph(program)
    cycles = _graph(program, rewrites=False) if rules_only else graph
    return any(p in _reachable(cycles, p) for p in {pred} | _reachable(graph, pred))


def check_untabled(engine: Engine, case: Case) -> None:
    """Untabled with no depth bound: a query on a rewrite cycle ends in a limit, one reaching no cycle answers exactly.

    A query reaching a cycle without lying on one may or may not finish, depending on
    the order goals are tried; it must then either answer exactly or stop with no answers.
    """
    _require(engine, UNTABLED.needs)
    program = replace(case.program, tabled=False, depth=0)
    pred = case.query.pred
    if _on_rewrite_cycle(program, pred):
        _limited(engine.solve(program, case.query, LOOP), LIMITS, f"untabled query on the rewrite cycle through {pred}", LOOP)
    elif not _reaches_cycle(program, pred):
        _exactly(engine.solve(program, case.query, GENEROUS), expected(case), "untabled query reaching no cycle")
    else:
        solution = engine.solve(program, case.query, LOOP)
        what = "untabled query reaching a cycle"
        if solution.status == "ok":
            _exactly(solution, expected(case), what)
        else:
            _limited(solution, LIMITS, what, LOOP)


def check_proofs(engine: Engine, case: Case) -> None:
    """At every exact bound each answer has exactly one proof, a valid derivation from the program within the bound."""
    _require(engine, PROOFS.needs)
    c = engine.capabilities
    for depth, counts in _exact_bounds(c):
        program = replace(case.program, tabled=_tabled(c, depth), depth=depth, depth_counts=counts, proofs=True)
        budget = _budget(engine, program, case.query)
        solution = engine.solve(program, case.query, budget)
        if not _bounded(engine, program, budget, solution, expected(case, depth, counts), _bound(program)):
            continue
        proofs = solution.proofs
        if proofs is None or set(proofs) != set(solution.answers):
            got = None if proofs is None else sorted(proofs)
            _fail(f"{_bound(program)}: proofs for {got}, but the answers are {sorted(solution.answers)}")
        for answer, proof in sorted(proofs.items()):
            problem = proof_problem(proof, Atom(case.query.pred, answer), program)
            if problem:
                _fail(f"{_bound(program)}: the proof of {Atom(case.query.pred, answer)} {problem}")


def _unifies(patterns: Sequence[Atom], atoms: Sequence[Atom]) -> bool:
    """Whether one binding of the patterns' variables turns every pattern into the atom beside it."""
    binding: dict[str, str] = {}
    for pattern, atom in zip(patterns, atoms, strict=True):
        if pattern.pred != atom.pred or pattern.arity != atom.arity:
            return False
        for term, value in zip(pattern.args, atom.args, strict=True):
            if is_variable(term):
                if binding.setdefault(term, value) != value:
                    return False
            elif term != value:
                return False
    return True


# Time: O(s) over the s distinct proof nodes, each visited once however often it is shared.
def proof_problem(proof: Proof, atom: Atom, program: Program) -> str | None:
    """What is wrong with `proof` as a derivation of `atom` from `program`, or None when it is valid.

    Every leaf must be a program fact; every step must apply a program rule, a program
    rewrite, or a rewrite's rule, whose head and body unify with the step's atom and its
    children; with program.depth > 0 the proof may take at most that many counted steps
    on any path, where a user rule's step doesn't count under a bound on rewrites.
    """
    if proof.atom != atom:
        return f"proves {proof.atom} instead"
    facts = set(program.facts)
    user_rules = set(program.rules)
    rules = user_rules | {rewrite.rule() for rewrite in program.rewrites}
    keys = {rewrite.key for rewrite in program.rewrites}
    rewrites_only = program.depth_counts == "rewrites"
    height: dict[int, int] = {}
    stack: list[tuple[Proof, bool]] = [(proof, False)]
    while stack:
        node, done = stack.pop()
        if id(node) in height:
            continue
        if not done:
            stack.append((node, True))
            stack.extend((child, False) for child in node.children)
            continue
        if node.by is None:
            if node.children or node.atom not in facts:
                return f"has the leaf {node.atom}, which is not a program fact"
            height[id(node)] = 0
            continue
        if isinstance(node.by, Rewrite):
            if node.by.key not in keys:
                return f"applies {node.by}, which is not a rewrite of the program"
            rule = node.by.rule()
        elif isinstance(node.by, Rule) and node.by in rules:
            rule = node.by
        else:
            return f"applies {node.by!r}, which is not a rule or rewrite of the program"
        children = [child.atom for child in node.children]
        if len(children) != len(rule.body) or not _unifies((rule.head, *rule.body), (node.atom, *children)):
            return f"derives {node.atom} from {', '.join(map(str, children)) or 'nothing'} by {node.by}, which doesn't match"
        # Under a bound on rewrites a user rule's step is free; a step by a rewrite, or by a rule only a rewrite has, counts.
        counted = not rewrites_only or isinstance(node.by, Rewrite) or rule not in user_rules
        height[id(node)] = int(counted) + max(height[id(child)] for child in node.children)
    if program.depth and height[id(proof)] > program.depth:
        unit = "rewrites" if rewrites_only else "steps"
        return f"takes {height[id(proof)]} {unit}, more than the depth bound {program.depth} allows"
    return None


# Properties and the runner ------------------------------------------------------------


# Time: at most r + 1 solves, r = the case's rewrites, since every round but the last adds one of them or more.
def check_rounds(engine: Engine, case: Case) -> None:
    """The rounds strategy over the case's rewrites ends with exactly the reference answers.

    Each round evaluates with tables and no depth bound, then adds the rewrites
    concluding every predicate the run reached that no earlier round expanded,
    until a round adds none. A predicate evaluation called but left out of
    reached never has its rewrites added, so the last round lacks what they derive.
    """
    _require(engine, ROUNDS.needs)
    by_conclusion: dict[str, list[Rewrite]] = defaultdict(list)
    for rewrite in case.program.rewrites:
        by_conclusion[rewrite.conclusion].append(rewrite)
    added: dict[Key, Rewrite] = {}
    expanded: set[str] = set()
    rounds = 0
    while True:
        rounds += 1
        program = replace(case.program, rewrites=tuple(added.values()), tabled=True, depth=0)
        solution = engine.solve(program, case.query, GENEROUS)
        what = f"rounds, round {rounds} with {len(added)} of the case's {len(case.program.rewrites)} rewrites"
        if solution.status != "ok":
            _fail(f"{what}: status {solution.status} ({solution.detail}), expected ok")
        _reached(engine, solution, case, what)
        before = len(added)
        for pred in sorted(set(solution.reached or ()) - expanded):
            expanded.add(pred)
            for rewrite in by_conclusion.get(pred, ()):
                added.setdefault(rewrite.key, rewrite)
        if len(added) == before:
            _exactly(solution, expected(case), what)
            return


@dataclass(frozen=True)
class Property:
    """A conformance property: its name, its claim, what it needs, its cases and its check."""

    name: str
    claim: str
    needs: Callable[[Capabilities], str | None]
    strategy: Callable[[], SearchStrategy[Case]]
    check: Callable[[Engine, Case], None]


TABLING = Property("tabling", "with tables and no depth bound, answers equal the reference fixpoint",
                   _capability("tabling"), cases, check_tabling)
DEPTH = Property("depth", "with a bound of 1 to 4 steps, answers equal the reference cut at that height, tables on or off",
                 _capability("depth"), cases, check_depth)
REWRITE_DEPTH = Property("rewrite-depth", "with a bound of 1 to 4 rewrites, answers equal the reference cut at that "
                         "many rewrites and recursion through user rules answers completely, tables on or off",
                         _capability("rewrite_depth"), _rewrite_depth_cases, check_rewrite_depth)
LAZY = Property("lazy", "rewrites given through expand answer as rewrites given up front; expand sees only the program's predicates",
                _capability("lazy", _exact), cases, check_lazy)
BUDGET = Property("budget", "a budget too small ends with a limit status and no answers; a run reporting ok stayed in its budget",
                  _nothing, chains, check_budget)
NAMES = Property("names", "predicates and constants spelled like reserved words keep their answers, eagerly and lazily",
                 _exact, lambda: cases(RESERVED, RESERVED), check_names)
UNTABLED = Property("untabled", "untabled, a query on a rewrite cycle hits a limit and one reaching no cycle answers exactly",
                    _capability("untabled"), cases, check_untabled)
PROOFS = Property("proofs", "every answer has a proof deriving it from the program within the depth bound",
                  _capability("proofs", _exact), cases, check_proofs)
ROUNDS = Property("rounds", "evaluating, adding the rewrites of every predicate a run reached and evaluating again until a "
                  "run reaches nothing new gives the reference answers", _capability("reached", _capability("tabling")),
                  cases, check_rounds)

#: Every property, in the order the report lists them.
PROPERTIES: tuple[Property, ...] = (TABLING, DEPTH, REWRITE_DEPTH, LAZY, BUDGET, NAMES, UNTABLED, PROOFS, ROUNDS)


def property_named(name: str) -> Property:
    """The property called `name`, or a KeyError listing the known ones."""
    for prop in PROPERTIES:
        if prop.name == name:
            return prop
    msg = f"unknown property {name!r}; known properties: {', '.join(p.name for p in PROPERTIES)}"
    raise KeyError(msg)


@dataclass(frozen=True)
class Verdict:
    """One property's outcome: pass, fail with the shrunk falsifying case and what went wrong, or skip with why.

    `may_stop` counts the untabled runs the cases made whose bound does not keep their
    search small, which may stop at a limit instead of finishing, and `stopped` the
    ones that did.
    """

    prop: str
    outcome: Literal["pass", "fail", "skip"]
    cases: int = 0
    case: Case | None = None
    detail: str = ""
    may_stop: int = 0
    stopped: int = 0

    def lines(self) -> list[str]:
        """The verdict as report lines, the falsifying case indented below a failure."""
        head = f"{self.outcome:<5} {self.prop:<13}"
        if self.outcome == "skip":
            return [f"{head} {self.detail}"]
        stops = (f"; {self.stopped} of {self.may_stop} untabled runs allowed to stop at a limit did"
                 if self.may_stop else "")
        if self.outcome == "pass":
            return [f"{head} {self.cases} cases{stops}"]
        out = [f"{head} after {self.cases} cases (shrinking included){stops}"]
        out += [f"        {line}" for line in self.detail.splitlines()]
        if self.case is not None:
            out.append("        falsifying case:")
            out += [f"          {line}" for line in self.case.lines()]
        return out


@dataclass(frozen=True)
class Report:
    """The verdicts of one conformance run of one engine."""

    engine: str
    max_examples: int
    seed: int | None
    verdicts: tuple[Verdict, ...]

    @property
    def passed(self) -> bool:
        """True when no property failed; skipped properties don't count against the engine."""
        return all(v.outcome != "fail" for v in self.verdicts)

    def verdict(self) -> str:
        """The one-line verdict, as the summary's last line."""
        count = {o: sum(v.outcome == o for v in self.verdicts) for o in ("pass", "fail", "skip")}
        tally = f"{count['pass']} passed, {count['fail']} failed, {count['skip']} skipped"
        if self.passed:
            return f"verdict: engine {self.engine!r} conforms: {tally}"
        failed = ", ".join(v.prop for v in self.verdicts if v.outcome == "fail")
        return f"verdict: engine {self.engine!r} does not conform ({failed}): {tally}"

    def summary(self) -> str:
        """The report as text: a header, one block per property and the verdict line."""
        seed = "random" if self.seed is None else self.seed
        out = [(f"conformance of engine {self.engine!r}: {len(self.verdicts)} properties, "
                f"up to {self.max_examples} cases each, seed {seed}")]
        for v in self.verdicts:
            out += [f"  {line}" for line in v.lines()]
        out.append(self.verdict())
        return "\n".join(out)


def _explain(engine: Engine, prop: Property, case: Case) -> str:
    """Why `case` fails `prop` on `engine`, from running the check once more."""
    try:
        prop.check(engine, case)
    except ConformanceFailure as failure:
        return str(failure)
    except Exception as error:  # noqa: BLE001 - an engine crash is a verdict, reported with its traceback
        return "the engine raised:\n" + "".join(traceback.format_exception(error)).rstrip()
    return "the case failed during the search but passed when run again, so the engine is not deterministic"


# Time: max_examples draws of the property's strategy and about max_examples / count property checks for
# shard (index, count), plus the checks the shrinker makes after a failure.
def check(engine: Engine, prop: Property, max_examples: int = 200, seed: int | None = 0,
          shard: tuple[int, int] = (0, 1)) -> Verdict:
    """Search up to `max_examples` cases for one that fails `prop` on `engine`, shrinking any it finds.

    With an int `seed` the search tries the same cases on every run; with None,
    Hypothesis picks them at random. No example database is read or written.

    `shard` (index, count) splits the search across `count` runs, which can go in
    parallel: every shard replays the whole seeded search, since Hypothesis picks
    each case from the outcomes before it, and checks only the cases whose position
    in it is `index` modulo `count`, passing the rest unchecked; once a check fails
    it checks every case the shrinker tries. Together the shards check exactly the
    cases one search would, and the shard holding the first failing case shrinks it
    to the same falsifying case. A random search can't be split, so a shard needs a seed.
    """
    index, count = shard
    if not 0 <= index < count:
        msg = f"shard {shard} is not (index, count) with 0 <= index < count"
        raise ValueError(msg)
    if count > 1 and seed is None:
        msg = "a sharded search needs an int seed, so that every shard replays the same search"
        raise ValueError(msg)
    reason = prop.needs(engine.capabilities)
    if reason is not None:
        return Verdict(prop.name, "skip", detail=reason)
    hypothesis = _hypothesis()
    tried = 0
    tally = [0, 0]
    drawn = 0
    failed = False

    def fails(case: Case) -> bool:
        nonlocal tried, drawn, failed
        mine = failed or drawn % count == index
        drawn += 1
        if not mine:
            return False
        tried += 1
        try:
            prop.check(engine, case)
        except Exception:  # noqa: BLE001 - a failure or a crash both falsify the property
            failed = True
            return True
        return False

    settings = hypothesis.settings(max_examples=max_examples, deadline=None, database=None)
    token = _ALLOWED_TO_STOP.set(tally)
    try:
        with warnings.catch_warnings():
            # find() renders each failing case for a report it never shows; ours prints Case.lines().
            warnings.filterwarnings("ignore", "Generating overly large repr", hypothesis.errors.HypothesisWarning)
            case = hypothesis.find(prop.strategy(), fails, settings=settings,
                                   random=None if seed is None else random.Random(seed))
        detail = _explain(engine, prop, case)
    except hypothesis.errors.NoSuchExample:
        return Verdict(prop.name, "pass", cases=tried, may_stop=tally[0], stopped=tally[1])
    except hypothesis.errors.Flaky as error:
        return Verdict(prop.name, "fail", cases=tried,
                       detail=f"the engine answered one case differently on two runs, so it is not deterministic: {error}",
                       may_stop=tally[0], stopped=tally[1])
    finally:
        _ALLOWED_TO_STOP.reset(token)
    return Verdict(prop.name, "fail", cases=tried, case=case, detail=detail, may_stop=tally[0], stopped=tally[1])


def run(engine: Engine, max_examples: int = 200, seed: int | None = 0,
        properties: Sequence[Property] = PROPERTIES) -> Report:
    """Check every property in `properties` against `engine` and collect the verdicts."""
    verdicts = tuple(check(engine, prop, max_examples, seed) for prop in properties)
    return Report(engine.name, max_examples, seed, verdicts)


def main(argv: Sequence[str] | None = None) -> int:
    """The command line: run the kit against a registered engine, print the report, exit 0 when it conforms."""
    parser = argparse.ArgumentParser(prog="python -m semrewrite.conformance",
                                     description="Check that a logic engine implements the semantic rewriter's engine protocol.")
    parser.add_argument("--engine", default="python", choices=sorted(REGISTRY), help="the registered engine to check")
    parser.add_argument("--max-examples", type=int, default=200, help="cases tried per property (default 200)")
    parser.add_argument("--seed", type=int, default=0, help="the search seed, so a run repeats exactly (default 0)")
    parser.add_argument("--property", action="append", choices=[p.name for p in PROPERTIES],
                        help="check only this property; repeat for several")
    args = parser.parse_args(argv)
    properties = [property_named(n) for n in args.property] if args.property else PROPERTIES
    report = run(make(args.engine), args.max_examples, args.seed, properties)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
