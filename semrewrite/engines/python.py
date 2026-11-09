"""Purpose: the pure-Python logic engine, PythonEngine: demand-driven semi-naive bottom-up
evaluation of function-free Horn programs of any arity, with rewrites as rules, lazy
expansion, depth bounds, hard work and time budgets, and one proof per answer.

    from semrewrite.engines.python import PythonEngine
    from semrewrite.model import Atom, Rewrite
    from semrewrite.protocols import Budget, Program

    program = Program(facts=(Atom("child_of", ("tom", "erin")),),
                      rewrites=(Rewrite("parent", "child_of", (1, 0)),))
    PythonEngine().solve(program, Atom("parent", ("X", "tom")), Budget()).answers
    # {("erin", "tom")}

A solve runs in two phases. Relevance starts at the query's predicate and walks
breadth-first: a relevant predicate adds the body predicates of the rules for it and
the premises of the rewrites concluding it, taken from program.rewrites and, when an
expander is given, from expand(P), called once per predicate per solve when the walk
reaches P. Facts of other predicates are never read. Evaluation is then a semi-naive
fixpoint over the relevant rules and the rule of every relevant rewrite, in levels.
A step the depth bound counts is a counted step, and the others are free: without a
bound or with depth_counts "steps" every step counts, and with "rewrites" a user rule's
step is free. A level first closes under the free plans, each round joining a rule
body with at least one fact the round before derived; one round of the counted plans
then joins their bodies with at least one fact the whole level derived, and starts the
next level. Level k so holds exactly the facts with a derivation of at most k counted
steps on every path down to a fact, and a depth bound n is a stop once level n is
closed. With every step counted a level is one round, the round deriving exactly the
facts of derivation height k.

With a depth bound n the walk also records each predicate's distance d from the query,
the fewest counted steps between them, walking free steps first (a 0-1 breadth-first
search). A row of that predicate derived at level h can only take part in an answer
when d + h <= n, so a predicate at distance n is read but its counted plans never
fire and it is never expanded, and its plans fire only for levels h with d + h <= n.
The answers are those of fixpoint.least_model with the same bound; only the work and
the expander calls shrink.

Work counts the candidate rows examined while joining rule bodies and while reading the
query's answers, so the same program always costs the same work. Every candidate checks
the work and time budget, and a run that exceeds either returns its limit status and no
answers. Iteration follows insertion order throughout, so the proof of each answer, its
first derivation, is also the same on every run.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from semrewrite.model import Atom, Key, Rewrite, Rule, is_variable
from semrewrite.protocols import Budget, Capabilities, Expander, Program, Proof, Solution, Status

Row = tuple[str, ...]
"""The constants of a ground atom in argument order: one tuple of a relation."""

Fact = tuple[str, Row]
"""A ground atom as (predicate, row)."""

Term = int | str
"""A compiled argument: an int is a variable's slot in the binding, a str a constant."""

# Where a body atom's candidate rows come from in a round: the delta it joins, the facts
# derived before the delta, or every fact derived before the round.
_DELTA, _OLD, _FULL = 0, 1, 2

# The join loops count work on every candidate but read the clock on every 256th: reading
# it on every candidate cost 15% of the loop (980 against 840 ns a candidate, measured on
# 40 binary predicates, 150 facts, 316 rewrites and a join), while 256 candidates overshoot a time
# budget by about a quarter of a millisecond. solve reads it once more before returning.
_CLOCK_MASK = 256 - 1


class _Limit(Exception):
    """Raised inside a solve when it exceeds its budget; solve turns it into the limit's Solution."""

    def __init__(self, status: Status, detail: str) -> None:
        super().__init__(detail)
        self.status: Status = status
        self.detail = detail


@dataclass(frozen=True, slots=True)
class _Step:
    """One body atom compiled for a join order.

    `positions` are the argument positions whose values are known before the atom is
    matched, and `key` gives each one's value: a constant, or the slot of a variable an
    earlier atom bound. The first step of a plan reads the delta and uses them as
    filters; later steps look them up in a hash index. `checks` pairs the positions of
    a variable repeated within this atom, and `binds` stores each variable the atom
    binds first into its slot.
    """

    pred: str
    source: int
    positions: tuple[int, ...]
    key: tuple[Term, ...]
    checks: tuple[tuple[int, int], ...]
    binds: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class _Plan:
    """A rule compiled for one delta position: its steps in join order and how to build the head.

    `by` is what a proof names for the step, the user Rule or the Rewrite; `step_of[j]`
    is the step that matches body atom j, so a derived fact's children come out in body
    order; `distance` is the head predicate's distance from the query, and `counted`
    whether the depth bound counts the plan's step. A rule with one body atom, such as
    every rewrite's, also gets a `projection`: per head argument, the position of the
    body row holding it (an int) or a constant, so it derives its head without a binding.
    """

    by: Rule | Rewrite
    head_pred: str
    head: tuple[Term, ...]
    steps: tuple[_Step, ...]
    step_of: tuple[int, ...]
    body_preds: tuple[str, ...]
    slots: int
    distance: int
    counted: bool
    projection: tuple[Term, ...] | None


def _compile(by: Rule | Rewrite, rule: Rule, distance: int, counted: bool) -> list[_Plan]:
    """The semi-naive plans of `rule`, one per body position i taking the delta.

    Atom i is matched first against the delta, then the others in body order; atoms
    before i read only facts older than the delta and atoms after i read every fact, so
    a body instantiation using several new facts is found once, by the plan of its
    first new fact.
    """
    body = rule.body
    plans = []
    for delta in range(len(body)):
        order = (delta, *(j for j in range(len(body)) if j != delta))
        slots: dict[str, int] = {}
        steps = []
        for j in order:
            atom = body[j]
            positions: list[int] = []
            key: list[Term] = []
            checks: list[tuple[int, int]] = []
            binds: list[tuple[int, int]] = []
            first: dict[str, int] = {}
            for pos, term in enumerate(atom.args):
                if not is_variable(term):
                    positions.append(pos)
                    key.append(term)
                elif term in first:
                    checks.append((pos, first[term]))
                elif term in slots:
                    positions.append(pos)
                    key.append(slots[term])
                else:
                    first[term] = pos
                    slots[term] = len(slots)
                    binds.append((pos, slots[term]))
            source = _DELTA if j == delta else _OLD if j < delta else _FULL
            steps.append(_Step(atom.pred, source, tuple(positions), tuple(key), tuple(checks), tuple(binds)))
        head = tuple(slots[t] if is_variable(t) else t for t in rule.head.args)
        step_of = tuple(order.index(j) for j in range(len(body)))
        projection = None
        if len(body) == 1:
            position_of = {slot: pos for pos, slot in steps[0].binds}
            projection = tuple(position_of[t] if isinstance(t, int) else t for t in head)
        plans.append(_Plan(by, rule.head.pred, head, tuple(steps), step_of,
                           tuple(a.pred for a in body), len(slots), distance, counted, projection))
    return plans


class _Solve:
    """The state of one call to PythonEngine.solve: the relevant predicates, their rows and indexes, and the meter."""

    def __init__(self, program: Program, query: Atom, budget: Budget, expand: Expander | None) -> None:
        self.program = program
        self.query = query
        self.budget = budget
        self.expand = expand
        self.start = time.monotonic()
        self.deadline = self.start + budget.seconds
        self.work = 0
        self.rounds = 0
        self.levels = 0
        # Relevant predicates in the order the walk reached them, with their distance from the query.
        self.distance: dict[str, int] = {}
        # Each predicate's arity and the first place that fixed it, for error messages.
        self.arity: dict[str, tuple[int, str]] = {}
        self.plans: list[_Plan] = []
        # pred -> the indexes of the plans whose first, delta, atom reads it.
        self.by_delta: dict[str, list[int]] = {}
        # pred -> row -> the round that first derived it, 0 for a program fact; insertion order is round order.
        self.round_of: dict[str, dict[Row, int]] = {}
        # pred -> bound positions -> key values -> rows, each bucket appended in round order.
        self.index: dict[str, dict[tuple[int, ...], dict[tuple[str, ...], list[Row]]]] = {}
        # The first derivation of every derived fact: the rule or rewrite and the facts it joined.
        self.derived_by: dict[Fact, tuple[Rule | Rewrite, tuple[Fact, ...]]] = {}

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def clock(self, where: str) -> None:
        """Stop with time_limit once the run has spent more than its seconds."""
        if time.monotonic() > self.deadline:
            raise _Limit("time_limit", f"time budget of {self.budget.seconds}s exceeded {where}")

    def note_arity(self, pred: str, arity: int, where: str) -> None:
        """Fix `pred`'s arity at its first use and raise ValueError when a later use disagrees."""
        known = self.arity.setdefault(pred, (arity, where))
        if known[0] != arity:
            msg = f"predicate {pred!r} has arity {known[0]} in {known[1]} but arity {arity} in {where}"
            raise ValueError(msg)

    # Time: one pass over the facts, rules and program rewrites to group them by predicate,
    # then O(1) per rule, rewrite and fact of a relevant predicate, plus one expand call per
    # relevant predicate closer to the query than the depth bound. A predicate is queued at
    # most twice, once more when a free step finds it closer than a counted one did.
    def relevance(self) -> None:
        """Walk from the query's predicate in order of distance, collecting relevant predicates, their facts and plans."""
        program, query, depth = self.program, self.query, self.program.depth
        # Without a bound nothing is free, so evaluation runs plain rounds whatever depth_counts says.
        rules_count = not depth or program.depth_counts == "steps"
        facts: dict[str, list[Atom]] = {}
        for fact in program.facts:
            facts.setdefault(fact.pred, []).append(fact)
        rules: dict[str, list[Rule]] = {}
        for rule in dict.fromkeys(program.rules):
            rules.setdefault(rule.head.pred, []).append(rule)
        rewrites: dict[str, list[Rewrite]] = {}
        for rewrite in program.rewrites:
            rewrites.setdefault(rewrite.conclusion, []).append(rewrite)
        seen: set[Key] = set()
        queue: deque[str] = deque()
        walked: set[str] = set()

        def reach(pred: str, arity: int, where: str, distance: int, free: bool) -> None:
            self.note_arity(pred, arity, where)
            if pred in self.distance:
                # A free step can find a predicate closer than the counted step that reached it first.
                if pred not in walked and distance < self.distance[pred]:
                    self.distance[pred] = distance
                    queue.appendleft(pred)
                return
            self.distance[pred] = distance
            rows = self.round_of[pred] = {}
            for fact in facts.get(pred, ()):
                self.note_arity(pred, fact.arity, f"fact {fact}")
                rows.setdefault(fact.args, 0)
            if free:
                queue.appendleft(pred)
            else:
                queue.append(pred)

        reach(query.pred, query.arity, f"query {query}", 0, False)
        while queue:
            pred = queue.popleft()
            if pred in walked:
                continue
            walked.add(pred)
            distance = self.distance[pred]
            # A row a counted step derives for a predicate at the depth bound could only reach the
            # query in more than `depth` counted steps, so such steps are never needed there.
            at_bound = bool(depth) and distance >= depth
            if not (rules_count and at_bound):
                for rule in rules.get(pred, ()):
                    self.note_arity(pred, rule.head.arity, f"rule {rule}")
                    for atom in rule.body:
                        reach(atom.pred, atom.arity, f"rule {rule}", distance + rules_count, not rules_count)
                    self.add(rule, rule, distance, rules_count)
            if at_bound:
                continue
            found = list(rewrites.get(pred, ()))
            if self.expand is not None:
                for rewrite in self.expand(pred):
                    if not isinstance(rewrite, Rewrite):
                        msg = f"expand({pred!r}) returned {rewrite!r}, which is not a Rewrite"
                        raise TypeError(msg)
                    if rewrite.conclusion != pred:
                        msg = f"expand({pred!r}) returned {rewrite}, which concludes {rewrite.conclusion!r}"
                        raise ValueError(msg)
                    found.append(rewrite)
                self.clock(f"while expanding {pred!r}")
            for rewrite in found:
                if rewrite.key in seen:
                    continue
                seen.add(rewrite.key)
                self.note_arity(pred, rewrite.arity, f"rewrite {rewrite}")
                reach(rewrite.premise, rewrite.arity, f"rewrite {rewrite}", distance + 1, False)
                self.add(rewrite, rewrite.rule(), distance, True)

    def add(self, by: Rule | Rewrite, rule: Rule, distance: int, counted: bool) -> None:
        """Compile `rule` and build the hash index each of its later join steps probes."""
        for plan in _compile(by, rule, distance, counted):
            self.by_delta.setdefault(plan.steps[0].pred, []).append(len(self.plans))
            self.plans.append(plan)
            for step in plan.steps[1:]:
                indexes = self.index.setdefault(step.pred, {})
                if step.positions not in indexes:
                    buckets: dict[tuple[str, ...], list[Row]] = {}
                    for row in self.round_of[step.pred]:
                        buckets.setdefault(tuple(row[p] for p in step.positions), []).append(row)
                    indexes[step.positions] = buckets

    # Time: the sum over rounds of the candidate rows every plan firing in the round examines
    # (the work this engine reports), each costing O(a) for a row of arity a, plus
    # O(p log p) a round to order the p plans whose delta predicate has new rows. A plan's
    # candidates are the delta rows of its first atom, times, for each later atom, the rows
    # of the index bucket its bound arguments select. A counted plan joins each row of the
    # level that derived it once, and a free plan each row once in the round after it, so
    # every plan meets every row it can use once as its delta. Levels number at most the
    # depth bound, or one more than the greatest level of a relevant fact without one.
    # Space: the relevant facts, one index per (predicate, bound positions) a plan probes,
    # and one provenance entry per derived fact.
    def evaluate(self) -> None:
        """Run semi-naive levels until a level's counted plans derive nothing or the depth bound is reached."""
        depth = self.program.depth
        free = any(not plan.counted for plan in self.plans)
        # The rows the last round derived, the rows the current level derived, and the round that began
        # the level: the program facts' round 0, or the counted round that derived the level's first rows.
        delta = {pred: list(rows) for pred, rows in self.round_of.items() if rows}
        level_delta = {pred: list(rows) for pred, rows in delta.items()}
        level_start = 0
        while True:
            # Close the level under the free plans; each round's delta is the round before's rows.
            while free and delta:
                delta = self.round(delta, self.rounds, counted=False, level=self.levels)
                for pred, rows in delta.items():
                    level_delta.setdefault(pred, []).extend(rows)
            if not level_delta or (depth and self.levels == depth):
                return
            # One counted round over everything the level derived starts the next level.
            delta = self.round(level_delta, level_start, counted=True, level=self.levels + 1)
            if not delta:
                return
            self.levels += 1
            level_start = self.rounds
            level_delta = {pred: list(rows) for pred, rows in delta.items()}

    def round(self, delta: dict[str, list[Row]], since: int, counted: bool, level: int) -> dict[str, list[Row]]:
        """Fire the counted or the free plans on `delta`, rows derived in round `since` or later, deriving rows
        of `level`; record the new rows and return them by predicate."""
        self.rounds += 1
        k, depth = self.rounds, self.program.depth
        fresh: dict[str, dict[Row, tuple[Rule | Rewrite, tuple[Fact, ...]]]] = {}
        # Plans fire in the order they were compiled, so first derivations don't depend on which rows are new.
        for i in sorted(i for pred in delta for i in self.by_delta.get(pred, ())):
            plan = self.plans[i]
            if plan.counted != counted or (depth and plan.distance + level > depth):
                continue
            if plan.projection is not None:
                self.project(plan, k, delta, fresh)
            else:
                self.join(plan, k, delta, since, fresh)
        new: dict[str, list[Row]] = {}
        for pred, rows in fresh.items():
            if not rows:
                continue
            born = self.round_of[pred]
            indexes = self.index.get(pred, {})
            for row, derivation in rows.items():
                born[row] = k
                self.derived_by[(pred, row)] = derivation
                for positions, buckets in indexes.items():
                    buckets.setdefault(tuple(row[p] for p in positions), []).append(row)
            new[pred] = list(rows)
        return new

    def project(self, plan: _Plan, k: int, delta: dict[str, list[Row]],
                fresh: dict[str, dict[Row, tuple[Rule | Rewrite, tuple[Fact, ...]]]]) -> None:
        """Apply a one-atom `plan` to the delta in round k: each matching row gives one head row, no join needed."""
        step = plan.steps[0]
        pred, by, projection = step.pred, plan.by, plan.projection
        assert projection is not None, "project runs only one-atom plans, which carry a projection"
        filters, checks = tuple(zip(step.positions, step.key, strict=True)), step.checks
        known = self.round_of[plan.head_pred]
        out = fresh.setdefault(plan.head_pred, {})
        limit, deadline, mask = self.budget.work, self.deadline, _CLOCK_MASK
        work = self.work
        try:
            for row in delta[pred]:
                work += 1
                if work > limit:
                    raise _Limit("work_limit", f"work budget of {limit} exceeded in round {k}")
                if not work & mask and time.monotonic() > deadline:
                    raise _Limit("time_limit", f"time budget of {self.budget.seconds}s exceeded in round {k}")
                if any(row[p] != c for p, c in filters) or any(row[p] != row[q] for p, q in checks):
                    continue
                head = tuple([row[t] if isinstance(t, int) else t for t in projection])
                if head not in known and head not in out:
                    out[head] = (by, ((pred, row),))
        finally:
            self.work = work

    def join(self, plan: _Plan, k: int, delta: dict[str, list[Row]], since: int,
             fresh: dict[str, dict[Row, tuple[Rule | Rewrite, tuple[Fact, ...]]]]) -> None:
        """Join `plan`'s body in round k, its first atom over `delta` and its old atoms over rows derived before round
        `since`, and record each head row not yet known with its first derivation."""
        steps = plan.steps
        last = len(steps)
        env: list[str] = [""] * plan.slots
        matched: list[Row] = [()] * last
        head, by, body_preds, step_of = plan.head, plan.by, plan.body_preds, plan.step_of
        known = self.round_of[plan.head_pred]
        out = fresh.setdefault(plan.head_pred, {})
        index, round_of = self.index, self.round_of
        limit, deadline, mask = self.budget.work, self.deadline, _CLOCK_MASK
        work = self.work

        def walk(s: int) -> None:
            nonlocal work
            if s == last:
                row = tuple([env[t] if isinstance(t, int) else t for t in head])
                if row not in known and row not in out:
                    out[row] = (by, tuple([(body_preds[j], matched[step_of[j]]) for j in range(len(body_preds))]))
                return
            step = steps[s]
            if s == 0:
                candidates: list[Row] | tuple[()] = delta[step.pred]
                filters = tuple(zip(step.positions, step.key, strict=True))
            else:
                key = tuple([env[t] if isinstance(t, int) else t for t in step.key])
                candidates = index[step.pred][step.positions].get(key, ())
                filters = ()
            born = round_of[step.pred] if step.source == _OLD else None
            for row in candidates:
                work += 1
                if work > limit:
                    raise _Limit("work_limit", f"work budget of {limit} exceeded in round {k}")
                if not work & mask and time.monotonic() > deadline:
                    raise _Limit("time_limit", f"time budget of {self.budget.seconds}s exceeded in round {k}")
                # Buckets hold rows in round order, so the facts older than the delta are a prefix.
                if born is not None and born[row] >= since:
                    break
                if any(row[p] != c for p, c in filters) or any(row[p] != row[q] for p, q in step.checks):
                    continue
                for p, slot in step.binds:
                    env[slot] = row[p]
                matched[s] = row
                walk(s + 1)

        try:
            walk(0)
        finally:
            self.work = work

    def answers(self) -> set[Row]:
        """The rows of the query's predicate that match its constants and repeated variables."""
        query = self.query
        rows = set()
        for row in self.round_of[query.pred]:
            self.work += 1
            if self.work > self.budget.work:
                raise _Limit("work_limit", f"work budget of {self.budget.work} exceeded reading the answers")
            if not self.work & _CLOCK_MASK:
                self.clock("reading the answers")
            if query.match(row) is not None:
                rows.add(row)
        return rows

    # Time: O(f log f + s) for the f distinct facts and s premise links in the answers' first derivations.
    def proofs(self, answers: set[Row]) -> dict[Row, Proof]:
        """The first derivation of each answer, built bottom-up so no recursion depth limits it."""
        pred = self.query.pred
        needed: dict[Fact, int] = {}
        stack: list[Fact] = [(pred, row) for row in answers]
        while stack:
            fact = stack.pop()
            if fact in needed:
                continue
            needed[fact] = self.round_of[fact[0]][fact[1]]
            derivation = self.derived_by.get(fact)
            if derivation is not None:
                stack.extend(derivation[1])
        built: dict[Fact, Proof] = {}
        # A derived fact's premises were derived in earlier rounds, so building in round order finds them built.
        for fact in sorted(needed, key=needed.__getitem__):
            atom = Atom(fact[0], fact[1])
            derivation = self.derived_by.get(fact)
            if derivation is None:
                built[fact] = Proof(atom)
            else:
                rule, premises = derivation
                built[fact] = Proof(atom, rule, tuple(built[p] for p in premises))
        return {row: built[(pred, row)] for row in answers}


class PythonEngine:
    """The pure-Python engine: exact, terminating, lazy, depth-bounded and budgeted, with proofs.

    It has no untabled mode: bottom-up evaluation terminates on every program, so
    Program.tabled changes nothing here, and a rewriter strategy that needs plain
    top-down resolution must run on another engine. A malformed program, such as a
    predicate used with two arities or an expander returning a rewrite for another
    predicate, raises ValueError; an exception raised by the expander propagates.
    """

    name: str = "python"
    capabilities: Capabilities = Capabilities(
        tabling=True, depth=True, rewrite_depth=True, lazy=True, untabled=False, reached=True, proofs=True)

    def solve(self, program: Program, query: Atom, budget: Budget, expand: Expander | None = None) -> Solution:
        """Answer `query` over `program`, calling `expand` once per relevant predicate, within `budget`."""
        run = _Solve(program, query, budget, expand)
        try:
            run.relevance()
            run.evaluate()
            answers = run.answers()
            proofs = run.proofs(answers) if program.proofs else None
            run.clock("before returning")
        except _Limit as limit:
            return Solution(answers=set(), status=limit.status, work=run.work, seconds=run.elapsed(),
                            tables=len(run.distance), reached=frozenset(run.distance),
                            proofs={} if program.proofs else None, detail=limit.detail)
        detail = (f"{run.rounds} rounds in {run.levels} levels derived {len(run.derived_by)} facts "
                  f"over {len(run.distance)} relevant predicates")
        return Solution(answers=answers, status="ok", work=run.work, seconds=run.elapsed(),
                        derivations=len(answers), tables=len(run.distance), reached=frozenset(run.distance),
                        proofs=proofs, detail=detail)
