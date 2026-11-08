"""Purpose: tests of the engine conformance kit: every property against every engine
that imports here, at the Hypothesis profile's case count (400 by default, 100
under the ci profile), the kit passing a second, top-down engine it knows
nothing about, and the kit failing each of a dozen deliberately broken engines, since
a kit that passes a broken engine is worthless.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import replace

import pytest
from hypothesis import given, settings

from semrewrite import conformance as kit
from semrewrite.engines import REGISTRY, available, make
from semrewrite.engines.python import PythonEngine
from semrewrite.model import Atom, Rewrite, Rule, inverse, is_variable
from semrewrite.protocols import Budget, Capabilities, Engine, Expander, Program, Proof, Solution

AVAILABLE = available()


@pytest.mark.parametrize("prop", kit.PROPERTIES, ids=lambda p: p.name)
@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_registered_engine_conforms(name: str, prop: kit.Property) -> None:
    if name not in AVAILABLE:
        pytest.skip(f"engine {name!r} does not import in this environment")
    verdict = kit.check(make(name), prop, max_examples=settings().max_examples)
    print("\n".join(verdict.lines()))
    if verdict.outcome == "skip":
        pytest.skip(verdict.detail)
    assert verdict.outcome == "pass", "\n".join(verdict.lines())


# A second engine ---------------------------------------------------------------------


class TopDown:
    """Plain SLD resolution, depth-first and left to right with no tables, as an untabled Prolog runs.

    Like a Prolog machine it binds variables in one store with a trail that
    backtracking unwinds, and keeps goals in a shared linked list, so a step costs the
    size of the clause it tries, however deep the search; facts are indexed on their
    first argument. Work counts the clauses tried; a depth bound n lets a goal use a
    counted rule only while fewer than n counted steps lie above it, where every rule
    counts, or under depth_counts "rewrites" only the rules of rewrites. It has the
    untabled capability the Python engine lacks, so it exercises the untabled property,
    and it shares no code with the kit.
    """

    name = "topdown"
    capabilities = Capabilities(depth=True, rewrite_depth=True, lazy=True, untabled=True, reached=True)

    def solve(self, program: Program, query: Atom, budget: Budget, expand: Expander | None = None) -> Solution:
        start = time.monotonic()
        indexed: dict[tuple[str, str], list[tuple[str, ...]]] = {}
        every: dict[str, list[tuple[str, ...]]] = {}
        for fact in program.facts:
            every.setdefault(fact.pred, []).append(fact.args)
            if fact.args:
                indexed.setdefault((fact.pred, fact.args[0]), []).append(fact.args)
        rules: dict[str, list[Rule]] = {}
        for rule in (*program.rules, *(rewrite.rule() for rewrite in program.rewrites)):
            rules.setdefault(rule.head.pred, []).append(rule)
        # The user rules, whose steps a bound on rewrites alone does not count.
        free = set(program.rules) if program.depth_counts == "rewrites" else set()
        bindings: dict[str, str] = {}
        trail: list[str] = []

        def deref(term: str) -> str:
            while term in bindings:
                term = bindings[term]
            return term

        # A clause's variables are fresh, so binding them to the goal's dereferenced terms,
        # and goal variables only to constants, keeps every chain of bindings a link or two
        # long; binding the other way grows one chain per step down a cycle.
        def unify(goal: tuple[str, ...], head: tuple[str, ...]) -> bool:
            for x, y in zip(goal, head, strict=True):
                x, y = deref(x), deref(y)
                if x == y:
                    continue
                if is_variable(y):
                    bindings[y] = x
                    trail.append(y)
                elif is_variable(x):
                    bindings[x] = y
                    trail.append(x)
                else:
                    return False
            return True

        called: set[str] = set()
        answers: set[tuple[str, ...]] = set()
        work = renamed = 0
        # A goal list is None or ((pred, args, steps left or None), rest). A choice point
        # holds a goal list, the clauses for its first goal, the next one to try and the trail mark.
        goals = ((query.pred, query.args, program.depth or None), None)
        choices: list[tuple[object, list, int, int]] = []
        clauses: list = []
        i = 0
        while True:
            if goals is None:
                answers.add(tuple(deref(t) for t in query.args))
            else:
                (pred, args, left), rest = goals
                if not clauses and i == 0:
                    if pred not in called:
                        called.add(pred)
                        for rewrite in expand(pred) if expand else ():
                            rules.setdefault(pred, []).append(rewrite.rule())
                    first = deref(args[0]) if args else None
                    rows = every.get(pred, []) if first is None or is_variable(first) else indexed.get((pred, first), [])
                    clauses = [*rows, *(r for r in rules.get(pred, []) if left != 0 or r in free)]
                stepped = False
                while i < len(clauses):
                    clause = clauses[i]
                    i += 1
                    work += 1
                    over = ("work_limit" if work > budget.work
                            else "time_limit" if time.monotonic() - start > budget.seconds else None)
                    if over:
                        return Solution(status=over, work=work, seconds=time.monotonic() - start,
                                        reached=frozenset(called))
                    mark = len(trail)
                    if isinstance(clause, Rule):
                        renamed += 1
                        fresh = {t: f"{t}_{renamed}" for a in (clause.head, *clause.body) for t in a.args if is_variable(t)}
                        head, body = tuple(fresh.get(t, t) for t in clause.head.args), clause.body
                    else:
                        head, body, fresh = clause, (), {}
                    if not unify(args, head):
                        while len(trail) > mark:
                            del bindings[trail.pop()]
                        continue
                    choices.append((goals, clauses, i, mark))
                    below = None if left is None else left if clause in free else left - 1
                    for atom in reversed(body):
                        rest = ((atom.pred, tuple(fresh.get(t, t) for t in atom.args), below), rest)
                    goals, clauses, i = rest, [], 0
                    stepped = True
                    break
                if stepped:
                    continue
            if not choices:
                break
            goals, clauses, i, mark = choices.pop()
            while len(trail) > mark:
                del bindings[trail.pop()]
        elapsed = time.monotonic() - start
        if elapsed > budget.seconds:
            return Solution(status="time_limit", work=work, seconds=elapsed, reached=frozenset(called))
        return Solution(answers=answers, work=work, seconds=elapsed, derivations=len(answers), reached=frozenset(called))


def test_kit_passes_a_top_down_engine_and_runs_untabled() -> None:
    report = kit.run(TopDown(), max_examples=150)
    print(report.summary())
    outcomes = {v.prop: v.outcome for v in report.verdicts}
    assert report.passed, report.summary()
    assert outcomes == {"tabling": "skip", "depth": "pass", "rewrite-depth": "pass", "lazy": "pass",
                        "budget": "pass", "names": "pass", "untabled": "pass", "proofs": "skip"}
    depth = next(v for v in report.verdicts if v.prop == "depth")
    assert depth.may_stop == 3 * depth.cases and 0 <= depth.stopped <= depth.may_stop
    assert "untabled runs allowed to stop at a limit did" in depth.lines()[0]
    rewrites = next(v for v in report.verdicts if v.prop == "rewrite-depth")
    assert rewrites.may_stop == 4 * rewrites.cases and 0 <= rewrites.stopped <= rewrites.may_stop


# Broken engines ----------------------------------------------------------------------


class _Delegate:
    """An engine handing its input to PythonEngine and returning its output; each subclass breaks one rule."""

    name = "broken"
    capabilities = PythonEngine.capabilities

    def __init__(self) -> None:
        self.inner = PythonEngine()

    def solve(self, program: Program, query: Atom, budget: Budget, expand: Expander | None = None) -> Solution:
        return self.inner.solve(program, query, budget, expand)


class IgnoresDepth(_Delegate):
    """Evaluates every program to its fixpoint, whatever its depth bound."""

    def solve(self, program, query, budget, expand=None):
        return self.inner.solve(replace(program, depth=0), query, budget, expand)


class DepthOffByOne(_Delegate):
    """Lets a derivation take one step more than the bound."""

    def solve(self, program, query, budget, expand=None):
        return self.inner.solve(replace(program, depth=program.depth and program.depth + 1), query, budget, expand)


class RuleStepsCount(_Delegate):
    """Counts user rule steps under a bound on rewrites, so it cuts recursion through user rules."""

    def solve(self, program, query, budget, expand=None):
        return self.inner.solve(replace(program, depth_counts="steps"), query, budget, expand)


class IgnoresRewriteBound(_Delegate):
    """Evaluates a program bounded on rewrites to its fixpoint, so false rewrites spread without limit."""

    def solve(self, program, query, budget, expand=None):
        return self.inner.solve(replace(program, depth=0) if program.depth_counts == "rewrites" else program,
                                query, budget, expand)


class StepsAsRewrites(_Delegate):
    """Counts only rewrites under a bound on steps, so its step bound lets user rules recurse freely."""

    def solve(self, program, query, budget, expand=None):
        return self.inner.solve(replace(program, depth_counts="rewrites"), query, budget, expand)


class IgnoresExpand(_Delegate):
    """Never calls the expander, so lazily given rewrites are dropped."""

    def solve(self, program, query, budget, expand=None):
        return self.inner.solve(program, query, budget, None)


class ExpandsQueryOnly(_Delegate):
    """Expands the query's predicate and no other, so lazily found chains stop after one link."""

    def solve(self, program, query, budget, expand=None):
        only = None if expand is None else (lambda pred: expand(pred) if pred == query.pred else ())
        return self.inner.solve(program, query, budget, only)


class IgnoresBudget(_Delegate):
    """Runs every program to completion, whatever its budget."""

    def solve(self, program, query, budget, expand=None):
        return self.inner.solve(program, query, Budget(work=10**12, seconds=10**6), expand)


class PartialOnLimit(_Delegate):
    """Reports the limit it hit but returns answers anyway, which would read as complete."""

    def solve(self, program, query, budget, expand=None):
        solution = self.inner.solve(program, query, budget, expand)
        if solution.status != "ok":
            solution.answers = self.inner.solve(program, query, Budget(), expand).answers
        return solution


class EscapesMax(_Delegate):
    """Renames the constant max to max_ to dodge a builtin and forgets to rename it back in the answers."""

    def solve(self, program, query, budget, expand=None):
        def escape(atom: Atom) -> Atom:
            return Atom(atom.pred, tuple("max_" if t == "max" else t for t in atom.args))

        program = replace(program, facts=tuple(map(escape, program.facts)),
                          rules=tuple(Rule(escape(r.head), tuple(map(escape, r.body))) for r in program.rules))
        return self.inner.solve(program, escape(query), budget, expand)


class RenamesMaxPredicate(_Delegate):
    """Renames the predicate max to max_ throughout and answers right eagerly, but asks the expander about max_."""

    def solve(self, program, query, budget, expand=None):
        def rename(pred: str) -> str:
            return "max_" if pred == "max" else pred

        def atom(a: Atom) -> Atom:
            return Atom(rename(a.pred), a.args)

        def rewrite(w: Rewrite) -> Rewrite:
            return Rewrite(rename(w.conclusion), rename(w.premise), w.mapping, w.score)

        program = replace(program, facts=tuple(map(atom, program.facts)),
                          rules=tuple(Rule(atom(r.head), tuple(map(atom, r.body))) for r in program.rules),
                          rewrites=tuple(map(rewrite, program.rewrites)))
        inner = None if expand is None else (lambda pred: tuple(map(rewrite, expand(pred))))
        return self.inner.solve(program, atom(query), budget, inner)


class IgnoresRepeatedVariables(_Delegate):
    """Answers p(X, X) as if it were p(X, Y), dropping the equality a repeated query variable states."""

    def solve(self, program, query, budget, expand=None):
        seen: set[str] = set()
        args = []
        for i, term in enumerate(query.args):
            args.append(f"V{i}" if is_variable(term) and term in seen else term)
            seen.add(term)
        return self.inner.solve(program, Atom(query.pred, tuple(args)), budget, expand)


class LeafProofs(_Delegate):
    """Returns every answer's proof as a bare fact, derived answers included."""

    def solve(self, program, query, budget, expand=None):
        solution = self.inner.solve(program, query, budget, expand)
        if solution.proofs:
            solution.proofs = {answer: Proof(Atom(query.pred, answer)) for answer in solution.proofs}
        return solution


class PaddedProofs(_Delegate):
    """Pads each proof with a detour round a rewrite cycle p <= q <= p: still a derivation, but two steps longer."""

    def solve(self, program, query, budget, expand=None):
        solution = self.inner.solve(program, query, budget, expand)
        pairs = [(there, back) for there in program.rewrites for back in program.rewrites
                 if there.conclusion == query.pred == back.premise and back.conclusion == there.premise
                 and back.mapping == inverse(there.mapping)]
        if solution.proofs and pairs:
            there, back = pairs[0]

            def pad(proof: Proof) -> Proof:
                middle = Atom(there.premise, tuple(proof.atom.args[i] for i in there.mapping))
                return Proof(proof.atom, there, (Proof(middle, back, (proof,)),))

            solution.proofs = {answer: pad(proof) for answer, proof in solution.proofs.items()}
        return solution


class HidesReached(_Delegate):
    """Declares the reached capability but reports nothing reached."""

    def solve(self, program, query, budget, expand=None):
        return replace(self.inner.solve(program, query, budget, expand), reached=None)


class SecretlyTabled(TopDown):
    """Claims plain top-down resolution but evaluates untabled programs to their fixpoint, so cycles never loop."""

    name = "secretly-tabled"

    def solve(self, program, query, budget, expand=None):
        if not program.tabled and not program.depth:
            return PythonEngine().solve(replace(program, tabled=True), query, budget, expand)
        return super().solve(program, query, budget, expand)


class UntabledIgnoresDepth(TopDown):
    """Searches without its depth bound, so it loops on rewrite cycles and over-answers elsewhere.

    It caps its own budget at the kit's LOOP so each loop ends in a fraction of a
    second; looping to the 2,000,000 units a depth-1 run gets took 149 s of shrinking.
    The kit must fail it either way: an over-answer or a limit at depth 1.
    """

    name = "untabled-ignores-depth"

    def solve(self, program, query, budget, expand=None):
        capped = Budget(work=min(budget.work, kit.LOOP.work), seconds=min(budget.seconds, kit.LOOP.seconds))
        return super().solve(replace(program, depth=0), query, capped, expand)


class GivesUpEarly(TopDown):
    """Reports a work limit on every search deeper than one step without spending its budget."""

    name = "gives-up-early"

    def solve(self, program, query, budget, expand=None):
        if program.depth > 1:
            return Solution(status="work_limit", work=0, reached=frozenset({query.pred}))
        return super().solve(program, query, budget, expand)


BROKEN: list[tuple[Engine, str]] = [
    (IgnoresDepth(), "depth"),
    (DepthOffByOne(), "depth"),
    (RuleStepsCount(), "rewrite-depth"),
    (IgnoresRewriteBound(), "rewrite-depth"),
    (StepsAsRewrites(), "depth"),
    (IgnoresExpand(), "lazy"),
    (ExpandsQueryOnly(), "lazy"),
    (IgnoresBudget(), "budget"),
    (PartialOnLimit(), "budget"),
    (EscapesMax(), "names"),
    (RenamesMaxPredicate(), "names"),
    (IgnoresRepeatedVariables(), "tabling"),
    (LeafProofs(), "proofs"),
    (PaddedProofs(), "proofs"),
    (HidesReached(), "tabling"),
    (SecretlyTabled(), "untabled"),
    (UntabledIgnoresDepth(), "depth"),
    (GivesUpEarly(), "depth"),
]


@pytest.mark.parametrize(("engine", "prop"), BROKEN, ids=[f"{type(e).__name__}-{p}" for e, p in BROKEN])
def test_kit_fails_a_broken_engine(engine: Engine, prop: str) -> None:
    verdict = kit.check(engine, kit.property_named(prop), max_examples=400)
    print("\n".join(verdict.lines()))
    assert verdict.outcome == "fail", f"the kit passed {type(engine).__name__} on {prop}"
    assert verdict.case is not None and verdict.detail


def test_broken_engine_fails_the_whole_run() -> None:
    report = kit.run(IgnoresDepth(), max_examples=100)
    assert not report.passed
    assert report.verdict().startswith("verdict: engine 'broken' does not conform (depth")
    assert "falsifying case:" in report.summary()


# The kit's own parts -----------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(kit.chains(nodes=(2, 12)))
def test_chain_closed_form_is_the_reference(case: kit.Case) -> None:
    assert set(case.answers) == kit.reference(case.program, case.query)


@settings(max_examples=100, deadline=None)
@given(kit.cases())
def test_properties_run_inside_given(case: kit.Case) -> None:
    engine = PythonEngine()
    for check in (kit.check_tabling, kit.check_depth, kit.check_rewrite_depth, kit.check_lazy, kit.check_proofs):
        check(engine, case)


@settings(max_examples=100, deadline=None)
@given(kit.cases())
def test_cases_have_a_rewrite_cycle_and_one_arity_per_predicate(case: kit.Case) -> None:
    program = case.program
    assert any(kit._on_rewrite_cycle(program, p) for p in case.predicates())
    arity: dict[str, int] = {}
    atoms = [*program.facts, case.query, *(a for r in program.rules for a in (r.head, *r.body))]
    for atom in atoms:
        assert arity.setdefault(atom.pred, atom.arity) == atom.arity
    for rewrite in program.rewrites:
        assert arity.setdefault(rewrite.premise, rewrite.arity) == rewrite.arity
        assert arity.setdefault(rewrite.conclusion, rewrite.arity) == rewrite.arity
    assert set(arity.values()) <= {1, 2, 3} and 2 <= len(case.predicates()) <= 6
    assert len({t for f in program.facts for t in f.args}) <= 4
    assert len(set(program.facts)) == len(program.facts) <= 10 and len(program.rules) <= 1


def test_missing_capability_is_a_skip() -> None:
    case = kit.Case(Program(facts=(Atom("p", ("a",)),)), Atom("p", ("X",)))
    with pytest.raises(kit.MissingCapability):
        kit.check_untabled(PythonEngine(), case)
    assert kit.check(PythonEngine(), kit.UNTABLED).outcome == "skip"


def test_proof_problem_names_each_defect() -> None:
    fact = Atom("mother", ("ann", "bob"))
    rewrite = Rewrite("parent", "mother", (0, 1))
    swap = Rewrite("parent", "child_of", (1, 0))
    program = Program(facts=(fact,), rewrites=(rewrite, swap), depth=1)
    answer = Atom("parent", ("ann", "bob"))
    good = Proof(answer, rewrite, (Proof(fact),))
    assert kit.proof_problem(good, answer, program) is None
    assert kit.proof_problem(Proof(answer, rewrite.rule(), (Proof(fact),)), answer, program) is None
    assert "instead" in kit.proof_problem(good, Atom("parent", ("bob", "ann")), program)
    assert "not a program fact" in kit.proof_problem(Proof(answer), answer, program)
    assert "doesn't match" in kit.proof_problem(Proof(answer, swap, (Proof(fact),)), answer, program)
    stranger = Rewrite("parent", "dad", (0, 1))
    assert "not a rewrite" in kit.proof_problem(Proof(answer, stranger, (Proof(fact),)), answer, program)
    loop = Rewrite("mother", "parent", (0, 1))
    tall = Proof(answer, rewrite, (Proof(fact, loop, (good,)),))
    two_steps = replace(program, rewrites=(rewrite, swap, loop))
    assert "more than the depth bound" in kit.proof_problem(tall, answer, two_steps)


def test_proof_problem_counts_only_rewrites_under_a_bound_on_rewrites() -> None:
    """grand(ann, cy) by a user rule over two rewrite steps: two steps on each path, one of them a rewrite."""
    facts = (Atom("mother", ("ann", "bob")), Atom("mother", ("bob", "cy")))
    rewrite = Rewrite("parent", "mother", (0, 1))
    grand = Rule(Atom("grand", ("X", "Z")), (Atom("parent", ("X", "Y")), Atom("parent", ("Y", "Z"))))
    answer = Atom("grand", ("ann", "cy"))
    proof = Proof(answer, grand, tuple(Proof(Atom("parent", f.args), rewrite, (Proof(f),)) for f in facts))
    program = Program(facts=facts, rules=(grand,), rewrites=(rewrite,), depth=1)
    problem = kit.proof_problem(proof, answer, program)
    assert problem is not None and "takes 2 steps, more than the depth bound 1 allows" in problem
    assert kit.proof_problem(proof, answer, replace(program, depth_counts="rewrites")) is None
    assert proof.steps() == 2 and proof.steps("rewrites") == 1
    rule_twice = Proof(answer, grand, tuple(Proof(Atom("parent", f.args), rewrite.rule(), (Proof(f),)) for f in facts))
    assert kit.proof_problem(rule_twice, answer, replace(program, depth_counts="rewrites")) is None


def test_summary_and_command_line(capsys: pytest.CaptureFixture[str]) -> None:
    report = kit.run(PythonEngine(), max_examples=10)
    lines = report.summary().splitlines()
    assert lines[0].startswith("conformance of engine 'python': 8 properties")
    assert [line.split()[1] for line in lines[1:-1]] == [p.name for p in kit.PROPERTIES]
    assert lines[-1] == "verdict: engine 'python' conforms: 7 passed, 0 failed, 1 skipped"
    assert kit.main(["--engine", "python", "--max-examples", "5", "--property", "depth"]) == 0
    out = capsys.readouterr().out
    assert "pass  depth" in out and out.rstrip().endswith("conforms: 1 passed, 0 failed, 0 skipped")


def test_import_needs_no_hypothesis() -> None:
    code = ("import sys; sys.modules['hypothesis'] = None\n"
            "import semrewrite.conformance as kit\n"
            "try:\n    kit.cases()\nexcept ImportError as e:\n    print(e)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert "pip install 'semantic-rewriter[conformance]'" in out
