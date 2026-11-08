"""Purpose: the PyMeTTa engine: evaluates semantic-rewriting programs on upstream
pymetta (MeTTa on SWI-Prolog), with SLG tabling from lib_tabling, inference and
time budgets, and the lazy discovery callback as a pymetta op.

Two encodings carry the protocol. The native encoding compiles facts, user
rules and every rewrite into ordinary equations, (= (mother alice tom) True)
and (= (parent $X0 $X1) (child_of $X1 $X0)), and tables every defined
predicate. The meta encoding keeps predicates as data, (holds parent $x0 $x1),
and one clause asks the `impliers` op for rewrites of a goal's predicate while
evaluation runs and follows each through a `via` clause per argument mapping;
tabling `holds` makes that SLG resolution with semantic goal expansion, and an
extra argument holding the depth left makes it depth-bounded: every rewrite step
spends one, and so does every user rule step unless the bound counts rewrites
alone. The native encoding runs when the program has no callback, no depth bound
and tables on; otherwise the meta one.

Names are data: every predicate is written as p<k>__<name> and every constant as
c__<name> inside the engine and read back in the answers, so no knowledge-base
name can collide with a pymetta builtin (pymetta 0.9.4 silently loses answers
when a query binds a constant spelled like one, repros/builtin_named_constant_untabled.py),
a Prolog core predicate such as is, or the encodings' own symbols. k counts the
engine's solves, and the meta encoding's own functions are holds<k>, via<k> and
fact<k>, so no solve defines a function name an earlier one used: pymetta 0.9.4
makes every call of a name that earlier contexts tabled cost more, by a few
inferences per such context (repros/tabled_contexts_raise_inference_cost.py),
which over a long process starved untabled runs of their budget.

pymetta 0.9.4 charges an op's calls differently by its parameter's
annotation, so each op's annotation is the one measured cheapest where it is
called. The discovery op takes a Symbol: annotated object, an inference limit
reached in it arrives as an EngineError from the second context of a process
on (repros/op_object_parameter_wraps_limit.py), and the first naive solve of a
process ran to 16 to 23 million inferences against a budget of 300,000; with the
oracle on family, lazy-tabled cost 17,761 inferences a query with Symbol,
19,254 unannotated and 19,810 with object. The reach op takes object, which
cost 174 inferences a table on a tabled path closure against 287 unannotated.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterator

from metta import Expression, MeTTa, S, Symbol, parse
from metta.errors import EngineError, InferenceLimitError, TimeLimitError
from metta.vocabularies import EffectClass

from semrewrite.model import Atom, Rewrite, is_variable, mappings
from semrewrite.protocols import Budget, Capabilities, Expander, Program, Solution

# The solves this process has run, which tag every function name a solve defines.
_SOLVES = itertools.count()


def _mapping_symbol(m: tuple[int, ...]) -> str:
    """The symbol naming an argument mapping in the meta encoding, such as m1_0 for the binary swap.

    The meta encoding writes one via clause per mapping of each arity present,
    n! of them for arity n, the same growth as discovery's k·n! proposals.
    """
    return "m" + "_".join(map(str, m))


class _Names:
    """One namespace: a name becomes prefix + name in the engine and comes back without the prefix.

    The prefix makes every written symbol differ from every builtin, Prolog
    core predicate and encoding symbol, and the mapping is injective, so no
    list of reserved words has to be kept in step with pymetta. The bare
    prefix itself names nothing in any knowledge base, which makes it a safe
    sentinel.
    """

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def __call__(self, name: str) -> str:
        return f"${name}" if is_variable(name) else self.prefix + name

    def undo(self, symbol: str) -> str:
        return symbol.removeprefix(self.prefix)


class PyMeTTaEngine:
    """semrewrite.protocols.Engine on upstream pymetta; its work unit is the SWI-Prolog inference."""

    name = "pymetta"
    capabilities = Capabilities(tabling=True, depth=True, rewrite_depth=True, lazy=True, untabled=True, reached=True,
                                proofs=False)

    # -- the protocol ----------------------------------------------------------

    def solve(self, program: Program, query: Atom, budget: Budget, expand: Expander | None = None) -> Solution:
        if budget.work <= 0 or budget.seconds <= 0:
            # pymetta refuses non-positive limits, and a run with nothing to spend is already out of budget.
            status = "work_limit" if budget.work <= 0 else "time_limit"
            return Solution(status=status, detail=f"budget {budget} leaves nothing to spend", reached=frozenset())
        tag = next(_SOLVES)
        preds, consts = _Names(f"p{tag}__"), _Names("c__")
        native = expand is None and program.depth == 0 and program.tabled
        # Evaluation always calls the query's predicate, even when a fact answers it before any fallback runs.
        reached: set[str] = {query.pred}
        if native:
            source, tabled = _native_source(program, query, preds, consts)
            term = _query_term(query, preds, consts, meta=False, depth=0, tag=tag)
            ops = {"reach": _reach(reached, preds)}
        else:
            source, tabled = _meta_source(program, query, preds, consts, tag), []
            term = _query_term(query, preds, consts, meta=True, depth=program.depth, tag=tag)
            ops = {"impliers": _impliers(program, expand, reached, preds)}
        return self._run(source, tabled, ops, term, budget, query, consts, reached)

    # -- evaluation ----------------------------------------------------------------

    def _run(self, source: str, tabled: list[str], ops: dict[str, Callable[..., Iterator[Expression]]], term: str,
             budget: Budget, query: Atom, consts: _Names, reached: set[str]) -> Solution:
        solution = Solution()
        result: list = []
        started = time.perf_counter()
        with MeTTa() as m:
            for name, fn in ops.items():
                m.op(fn, name=name, effect=EffectClass.nondeterministicReadOnly)
            if tabled or "!(tabled" in source:
                m.self.run("!(import! (context-space) (library lib_tabling))")
            m.self.run(source)
            for decl in tabled:
                if m.self.run(decl) != [[True]]:
                    msg = f"pymetta refused the tabling declaration {decl}"
                    raise RuntimeError(msg)
            with m.stats() as stats:
                try:
                    # pymetta takes None for no time limit; SWI overflows on an infinite timeout.
                    timeout = None if math.isinf(budget.seconds) else budget.seconds
                    result = m.eval(parse(term), inferences=budget.work, timeout=timeout)
                except InferenceLimitError as e:
                    solution.status, solution.detail = "work_limit", str(e)
                except TimeLimitError as e:
                    solution.status, solution.detail = "time_limit", str(e)
                except EngineError as e:
                    solution.status, solution.detail = "error", str(e)
            # A limit reached inside a Python op can surface as a plain EngineError, so an engine error
            # is read as a limit when the engine's own inference count or the elapsed time reached the budget.
            if solution.status == "error" and stats.inferences >= budget.work:
                solution.status = "work_limit"
            elif solution.status == "error" and time.perf_counter() - started >= budget.seconds:
                solution.status = "time_limit"
            solution.work = stats.inferences
            solution.tables = _table_count(m)
        solution.seconds = time.perf_counter() - started
        # Budgets are hard: SWI checks its limits only now and then, so a run that finished past either
        # bound counts as stopped by it rather than answering for work its caller did not allow.
        if solution.status == "ok" and solution.work > budget.work:
            solution.status, solution.detail = "work_limit", f"finished after {solution.work} inferences, over {budget.work}"
        elif solution.status == "ok" and solution.seconds > budget.seconds:
            solution.status, solution.detail = "time_limit", f"finished after {solution.seconds:.3f} s, over {budget.seconds} s"
        solution.derivations = len(result)
        solution.reached = frozenset(reached)
        if solution.status != "ok":
            return solution
        for atom in result:
            children = atom.children if isinstance(atom, Expression) else ()
            if len(children) == query.arity + 1 and str(children[0]) == "ans":
                solution.answers.add(tuple(consts.undo(str(c)) for c in children[1:]))
            else:
                solution.status, solution.detail = "error", f"unexpected answer {atom}"
                solution.answers = set()
                break
        return solution


def _term(t: str, consts: _Names) -> str:
    return f"${t}" if is_variable(t) else consts(t)


def _call(a: Atom, preds: _Names, consts: _Names) -> str:
    return f"({preds(a.pred)} {' '.join(_term(t, consts) for t in a.args)})"


def _body(body: tuple[Atom, ...], call: Callable[[Atom], str]) -> str:
    """A rule body as a left-to-right let chain ending in its last call, which answers True when the conjunction holds."""
    text = call(body[-1])
    for a in reversed(body[:-1]):
        text = f"(let True {call(a)} {text})"
    return text


def _native_source(program: Program, query: Atom, preds: _Names, consts: _Names) -> tuple[str, list[str]]:
    """Facts, rewrite equations and rules as native equations, a reach fallback per predicate, and the tabling declarations.

    Rewrites come before user rules, so a rule body's call to a predicate only
    rewrites define is compiled after that predicate exists. Every predicate
    gets a fallback equation calling the reach op, which yields nothing but
    records that evaluation called the predicate, for the rounds strategy. The
    op call also keeps pymetta's tabling declarations linear in the number of
    predicates: without a clause calling an op, declaring 240 tabled functions
    that call each other took 2.6 s against 0.7 s with one, and a geography
    solve three times as long. Its calls are about half of a native solve's
    inferences.
    """
    lines = [f"(= {_call(f, preds, consts)} True)" for f in program.facts]
    arity: dict[str, int] = {f.pred: f.arity for f in program.facts}
    seen = set()
    for rw in program.rewrites:
        if rw.key in seen:
            continue
        seen.add(rw.key)
        rule = rw.rule()
        lines.append(f"(= {_call(rule.head, preds, consts)} {_body(rule.body, lambda a: _call(a, preds, consts))})")
        arity[rw.conclusion] = arity[rw.premise] = rw.arity
    for rule in program.rules:
        lines.append(f"(= {_call(rule.head, preds, consts)} {_body(rule.body, lambda a: _call(a, preds, consts))})")
        for a in (rule.head, *rule.body):
            arity[a.pred] = a.arity
    arity[query.pred] = query.arity
    for p, n in sorted(arity.items()):
        args = " ".join(f"$X{i}" for i in range(n))
        lines.append(f'(= ({preds(p)} {args}) (reach "{preds(p)}"))')
    tabled = [f"!(tabled ({preds(p)} {' '.join(f'$X{i}' for i in range(n))}))" for p, n in sorted(arity.items())]
    return "\n".join(lines), tabled


def _meta_source(program: Program, query: Atom, preds: _Names, consts: _Names, tag: int) -> str:
    """The program as data under holds, with the discovery clause, a via clause per mapping, and tabling when asked.

    For each arity n present, holds takes the predicate and n arguments, plus
    the depth left when the program has a depth bound: every rewrite step spends
    one unit, and every user rule step too unless the bound counts rewrites
    alone, when a rule passes the depth left to its body unchanged; facts hold at
    any depth, and tabling then keys each table on the depth left too. holds, via
    and fact carry the solve's tag.
    """
    holds_, via_, fact_ = f"holds{tag}", f"via{tag}", f"fact{tag}"
    depth = program.depth > 0
    d, dm = (" $d", " (- $d 1)") if depth else ("", "")
    # The depth left that a user rule's body goals get, and whether a rule needs depth left to fire.
    rule_counts = depth and program.depth_counts == "steps"
    body_left = dm if rule_counts else d
    arities = {query.arity} | {f.arity for f in program.facts} | {a.arity for r in program.rules for a in (r.head, *r.body)}
    arities |= {rw.arity for rw in program.rewrites}
    lines = [f"(= ({fact_} {preds(f.pred)} {' '.join(consts(t) for t in f.args)}) True)" for f in program.facts]
    for n in sorted(arities):
        xs = " ".join(f"$x{i}" for i in range(n))
        # A sentinel fact under the bare prefix defines fact for every arity holds calls, so an arity with
        # no facts answers nothing instead of raising an unknown-procedure error.
        lines.append(f"(= ({fact_} {preds('')} {' '.join(consts('') for _ in range(n))}) True)")
        lines.append(f"(= ({holds_} $p {xs}{d}) ({fact_} $p {xs}))")
        discover = f"(let (rw $q $m) (impliers $p) ({via_} $q $m {xs}{dm}))"
        lines.append(f"(= ({holds_} $p {xs}{d}) {f'(if (> $d 0) {discover} (empty))' if depth else discover})")
        for m in mappings(n):
            permuted = " ".join(f"$x{i}" for i in m)
            lines.append(f"(= ({via_} $q {_mapping_symbol(m)} {xs}{d}) ({holds_} $q {permuted}{d}))")
        if program.tabled:
            lines.append(f"!(tabled ({holds_} $p {xs}{d}))")

    def holds(a: Atom) -> str:
        return f"({holds_} {preds(a.pred)} {' '.join(_term(t, consts) for t in a.args)}{body_left})"

    for rule in program.rules:
        head = f"({holds_} {preds(rule.head.pred)} {' '.join(_term(t, consts) for t in rule.head.args)}{d})"
        body = _body(rule.body, holds)
        lines.append(f"(= {head} {f'(if (> $d 0) {body} (empty))' if rule_counts else body})")
    return "\n".join(lines)


def _query_term(query: Atom, preds: _Names, consts: _Names, *, meta: bool, depth: int, tag: int) -> str:
    args = " ".join(_term(t, consts) for t in query.args)
    call = f"(holds{tag} {preds(query.pred)} {args}{f' {depth}' if depth else ''})" if meta else f"({preds(query.pred)} {args})"
    return f"(let True {call} (ans {args}))"


def _impliers(program: Program, expand: Expander | None, reached: set[str], preds: _Names) -> Callable[[Symbol], Iterator[Expression]]:
    """The discovery op: the program's rewrites for a predicate, then whatever the callback returns for it, each once.

    It calls the callback on every call, as evaluation reaches the predicate,
    so a discoverer without a memo is asked again each time; a memoised one
    answers from its memo.
    """
    static: dict[str, list[Rewrite]] = defaultdict(list)
    for rw in program.rewrites:
        static[rw.conclusion].append(rw)

    # Symbol, not object: object lets an inference limit be missed or arrive wrapped (module docstring).
    def impliers(p: Symbol) -> Iterator[Expression]:
        name = preds.undo(str(p))
        reached.add(name)
        seen = set()
        for rw in (*static.get(name, ()), *(expand(name) if expand is not None else ())):
            if rw.key in seen:
                continue
            seen.add(rw.key)
            yield Expression((S.rw, S[preds(rw.premise)], S[_mapping_symbol(rw.mapping)]))

    return impliers


def _reach(reached: set[str], preds: _Names) -> Callable[[object], Iterator[Expression]]:
    # object is the cheapest of the annotations measured for this call (module docstring).
    def reach(p: object) -> Iterator[Expression]:
        reached.add(preds.undo(str(getattr(p, "value", p))))
        yield from ()

    return reach


def _table_count(m: MeTTa) -> int:
    row = m.runtime.once(
        "space_module(Space, _M), aggregate_all(count, current_table(_M:_G, _), N)",
        Space=m.self.name,
    )
    return int(row["N"])
