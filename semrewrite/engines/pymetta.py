"""Purpose: the PyMeTTa engine: evaluates semantic-rewriting programs on upstream
pymetta (MeTTa on SWI-Prolog), with SLG tabling from lib_tabling, inference and
time budgets, the lazy discovery callback as a pymetta op, and a derivation of
every answer when the program asks for proofs.

Two encodings carry the protocol. The native encoding compiles facts, user
rules and every rewrite into ordinary equations, (= (mother alice tom) True)
and (= (parent $X0 $X1) (child_of $X1 $X0)), and tables every defined
predicate. The meta encoding keeps predicates as data, (holds parent $x0 $x1),
its facts are atoms in the space, (fact parent alice tom), read with match, and
one clause asks the `impliers` op for rewrites of a goal's predicate while
evaluation runs and follows each through a `via` clause per argument mapping;
tabling `holds` makes that SLG resolution with semantic goal expansion, and an
extra argument holding the depth left makes it depth-bounded: every rewrite step
spends one, and so does every user rule step unless the bound counts rewrites
alone. The native encoding runs when the program has no callback, no depth bound
and tables on; otherwise the meta one. pymetta 0.9.4 takes time growing faster
than quadratically in a function's equations to drop them when a context closes:
10,000 ground equations of one function took 106 s against 0.06 s for the same
facts as atoms (repros/equation_teardown_superquadratic.py), so the meta encoding,
which would hold every fact in one function, holds them as atoms; the native one
spreads them over a function per predicate.

With Program.proofs, every equation of either encoding answers a proof term
where it answered True, and the engine reads each answer's term back as a Proof:
every proof is (pf n node), n its number of rule and rewrite steps, and its node
is proof-fact for a fact, (proof-rewrite premise mapping proof) for a rewrite
step, or (proof-rule j v1 ... vk proof1 ... proofn) for a step by the program's
rule j, where v1 ... vk bind the variables its body has and its head lacks. A
rewrite cycle gives an answer infinitely many proof terms, so plain tables never
complete there; each table is instead declared by a (cache name (plain private
(lattice keep))) row whose join keeps the proof with fewer steps, one per answer.
Steps are counted in naturals, so the join stops improving and the cycle
converges, at each answer's shortest derivation (Knuth's superior-function
argument: a proof's size is 1 plus its premises' sizes); a proof that went round
a cycle is never the shortest. An untabled run returns every derivation, and the
shortest is kept. The proof-free source is unchanged.

Names are data: every predicate is written as p<k>__<name> and every constant as
c__<name> inside the engine and read back in the answers, so no knowledge-base
name can collide with a pymetta builtin (pymetta 0.9.4 silently loses answers
when a query binds a constant spelled like one, repros/builtin_named_constant_untabled.py),
a Prolog core predicate such as is, or the encodings' own symbols. k counts the
engine's solves, and the meta encoding's own names are holds<k>, via<k> and
fact<k>, so no solve defines a function name an earlier one used: pymetta 0.9.4
makes every call of a name that earlier contexts tabled cost more, by a few
inferences per such context (repros/tabled_contexts_raise_inference_cost.py),
which over a long process starved untabled runs of their budget.

Work is the inferences evaluating the query spends, and a solve that finishes
counts the same work whatever the process ran before it, so a finished run's work
is the exact budget the same run needs. pymetta 0.9.4 counts some of its own upkeep
inside an evaluation, and the engine keeps that upkeep out of the count four ways
(repros/history_changes_inference_count.py). It parses the query before counting:
pymetta's reader spends three inferences a character, so a count that included it
grew by three for each digit a solve's tag gained. The first solve on a thread runs
every encoding once on a small program, outside any count: the first evaluation to
reach a builtin, op or library predicate also resolves its import, fills pymetta's
dispatch cache and creates the thread's global variables, which made the first
solve of a process count 61 more inferences than the same solve later, and the
first under a time limit 3 more than the next one under a time limit; building
the first engine on a thread runs it, so a caller's first timed solve does not
wait for it, and a solve on a thread no engine was built on runs it first. Every
context also imports lib_tabling, tabled or not: pymetta keeps the hooks that
invalidate its cache of translated forms installed only while some translation is
cached, so an untabled context, which evaluated nothing before its query, installed
them inside its count whenever the context before it had left the cache empty, 4
more inferences for depth-bounded parent(X, max) after another untabled run. And
Python's cyclic collector stays off while the engine counts, as timeit keeps it off
while timing: a collection there ran the finalizers of earlier solves' garbage,
which call into the engine, 3 more inferences now and then. pymetta counts only the
thread that started its runtime, so a solve on another thread reports a meaningless
count, -40 for a solve the main thread counts 2,358.

pymetta 0.9.4 charges an op's calls differently by its parameter's
annotation, so each op's annotation is the one measured cheapest where it is
called. The discovery op takes a Symbol: annotated object, an inference limit
reached in it arrives as an EngineError from the second context of a process
on (repros/op_object_parameter_wraps_limit.py), and while the meta encoding held
its facts as equations, the first naive solve of a process ran to 16 to 23
million inferences against a budget of 300,000. With the oracle on family,
lazy-tabled cost 19,253 inferences a query with Symbol, 20,746 unannotated and
21,301 with object, counting each query's parse as the engine then did. The reach
op takes object, which cost 174 inferences a table on a tabled path closure
against 287 unannotated.
"""

from __future__ import annotations

import gc
import itertools
import math
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator

from metta import Expression, MeTTa, S, Symbol, parse
from metta.errors import EngineError, InferenceLimitError, TimeLimitError
from metta.vocabularies import EffectClass

from semrewrite.model import Atom, Key, Rewrite, Rule, is_variable, mappings
from semrewrite.protocols import Budget, Capabilities, Expander, Program, Proof, Solution

# The solves this process has run, which tag every function name a solve defines.
_SOLVES = itertools.count()

# Whether this thread's first solve has run every encoding outside any count (_settle).
_SETTLED = threading.local()

# The heads of a proof term (module docstring).
_PROOF, _FACT, _REWRITE, _RULE = "pf", "proof-fact", "proof-rewrite", "proof-rule"


class _ProofTerms:
    """How one solve that asks for proofs writes them: (pf n node) terms, its size and join functions named after
    the solve's tag, and the let chains that bind a step's premises' proofs and count its steps."""

    leaf = f"({_PROOF} 0 {_FACT})"

    def __init__(self, tag: int) -> None:
        self.keep, self.size = f"keep{tag}", f"size{tag}"

    def definitions(self) -> list[str]:
        """The size of a proof term, and the join a proof table keeps the proof with fewer steps by."""
        return [f"(= ({self.size} ({_PROOF} $n $node)) $n)",
                f"(= ({self.keep} $old $new) (if (< ({self.size} $new) ({self.size} $old)) $new $old))"]

    def body(self, calls: list[str], node: str) -> str:
        """The premises' calls as a left-to-right let chain binding each one's proof to $pf<i>, ending in the
        step's proof: `node` over those proofs, with one step more than they have together."""
        total = f"({self.size} $pf0)"
        for i in range(1, len(calls)):
            total = f"(+ {total} ({self.size} $pf{i}))"
        text = f"(let $n (+ 1 {total}) ({_PROOF} $n {node}))"
        for i in reversed(range(len(calls))):
            text = f"(let $pf{i} {calls[i]} {text})"
        return text


def _mapping_symbol(m: tuple[int, ...]) -> str:
    """The symbol naming an argument mapping in the meta encoding, such as m1_0 for the binary swap.

    The meta encoding writes one via clause per mapping of each arity present,
    n! of them for arity n, the same growth as discovery's k·n! proposals.
    """
    return "m" + "_".join(map(str, m))


def _mapping_of(symbol: str) -> tuple[int, ...]:
    """The argument mapping _mapping_symbol named."""
    return tuple(int(i) for i in symbol[1:].split("_")) if len(symbol) > 1 else ()


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
    """semrewrite.protocols.Engine on upstream pymetta; its work unit is the SWI-Prolog inference, and a solve that
    finishes counts the same work whatever ran before it (module docstring)."""

    name = "pymetta"
    capabilities = Capabilities(tabling=True, depth=True, rewrite_depth=True, lazy=True, untabled=True, reached=True,
                                proofs=True, repeatable_work=True)

    def __init__(self) -> None:
        if not getattr(_SETTLED, "done", False):
            _settle(self)

    # -- the protocol ----------------------------------------------------------

    def solve(self, program: Program, query: Atom, budget: Budget, expand: Expander | None = None) -> Solution:
        if budget.work <= 0 or budget.seconds <= 0:
            # pymetta refuses non-positive limits, and a run with nothing to spend is already out of budget.
            status = "work_limit" if budget.work <= 0 else "time_limit"
            return Solution(status=status, detail=f"budget {budget} leaves nothing to spend", reached=frozenset(),
                            proofs={} if program.proofs else None)
        if not getattr(_SETTLED, "done", False):
            _settle(self)
        tag = next(_SOLVES)
        preds, consts = _Names(f"p{tag}__"), _Names("c__")
        native = expand is None and program.depth == 0 and program.tabled
        # Evaluation always calls the query's predicate, even when a fact answers it before any fallback runs.
        reached: set[str] = {query.pred}
        # With proofs, how this solve writes them, and every rewrite a proof step can name, by key: the program's,
        # then each one discovery yields.
        terms = _ProofTerms(tag) if program.proofs else None
        known: dict[Key, Rewrite] = {}
        for rw in program.rewrites:
            known.setdefault(rw.key, rw)
        if native:
            source, tables = _native_source(program, query, known.values(), preds, consts, terms)
            ops = {"reach": _reach(reached, preds)}
        else:
            source, tables = _meta_source(program, query, preds, consts, tag, terms)
            ops = {"impliers": _impliers(program, expand, reached, preds, known if terms else None)}
        term = _query_term(query, preds, consts, meta=not native, depth=program.depth, tag=tag, proofs=terms is not None)
        solution, result = self._run(source, _declarations(tables, terms), ops, term, budget, reached)
        if program.proofs:
            solution.proofs = {}
        if solution.status == "ok":
            reader = _ProofReader(program, known, preds, consts) if terms else None
            _read_answers(solution, result, query, consts, reader)
        return solution

    # -- evaluation ----------------------------------------------------------------

    def _run(self, source: str, declarations: tuple[list[str], list[str]],
             ops: dict[str, Callable[..., Iterator[Expression]]], term: str, budget: Budget,
             reached: set[str]) -> tuple[Solution, list]:
        """Load the source, declare its tables, evaluate the term within the budget, then undo the declarations
        that outlive the context; the raw results come back beside the solution for the caller to read. Only the
        evaluation is counted: the term is parsed before, and every context imports lib_tabling, whose import caches
        a translation, so the count never includes installing pymetta's translation-cache hooks (module docstring)."""
        solution = Solution()
        result: list = []
        declare, undo = declarations
        started = time.perf_counter()
        with MeTTa() as m:
            for name, fn in ops.items():
                m.op(fn, name=name, effect=EffectClass.nondeterministicReadOnly)
            m.self.run("!(import! (context-space) (library lib_tabling))")
            m.self.run(source)
            try:
                for decl in declare:
                    if m.self.run(decl) != [[True]]:
                        msg = f"pymetta refused the tabling declaration {decl}"
                        raise RuntimeError(msg)
                goal = parse(term)
                # The collector stays off while counting, as timeit keeps it off while timing: a collection inside the
                # count would run the finalizers of earlier solves' garbage, some of which call into the engine.
                collecting = gc.isenabled()
                gc.disable()
                try:
                    with m.stats() as stats:
                        try:
                            # pymetta takes None for no time limit; SWI overflows on an infinite timeout.
                            timeout = None if math.isinf(budget.seconds) else budget.seconds
                            result = m.eval(goal, inferences=budget.work, timeout=timeout)
                        except InferenceLimitError as e:
                            solution.status, solution.detail = "work_limit", str(e)
                        except TimeLimitError as e:
                            solution.status, solution.detail = "time_limit", str(e)
                        except EngineError as e:
                            solution.status, solution.detail = "error", str(e)
                finally:
                    if collecting:
                        gc.enable()
                # A limit reached inside a Python op can surface as a plain EngineError, so an engine error
                # is read as a limit when the engine's own inference count or the elapsed time reached the budget.
                if solution.status == "error" and stats.inferences >= budget.work:
                    solution.status = "work_limit"
                elif solution.status == "error" and time.perf_counter() - started >= budget.seconds:
                    solution.status = "time_limit"
                solution.work = stats.inferences
                solution.tables = _table_count(m)
            finally:
                for form in undo:
                    m.self.run(form)
        solution.seconds = time.perf_counter() - started
        # Budgets are hard: SWI checks its limits only now and then, so a run that finished past either
        # bound counts as stopped by it rather than answering for work its caller did not allow.
        if solution.status == "ok" and solution.work > budget.work:
            solution.status, solution.detail = "work_limit", f"finished after {solution.work} inferences, over {budget.work}"
        elif solution.status == "ok" and solution.seconds > budget.seconds:
            solution.status, solution.detail = "time_limit", f"finished after {solution.seconds:.3f} s, over {budget.seconds} s"
        solution.derivations = len(result)
        solution.reached = frozenset(reached)
        return solution, result


def _settle(engine: PyMeTTaEngine) -> None:
    """Solve a small program once in every encoding the engine writes, outside any count, so that no later solve on
    this thread counts pymetta resolving a builtin, op or library predicate, or creating a global, for the first time.

    The program has facts of arity one and two, a rule with a two-atom body, and rewrites with and without a swap of
    their arguments, two of which make the cycle f <= g <= f, so the sources hold every clause shape _native_source
    and _meta_source write. It runs under every setting of tables, depth bound, what the bound counts, discovery
    callback and proofs, with and without a time limit, since the first evaluation under a time limit cost 3
    inferences more. e(X, Y) reaches no cycle and runs under each; f(X, Y) runs wherever a run on a cycle
    terminates, with tables or a bound, since the first tabled cycle of a process, and the first proof table to join
    two proofs of one answer, cost their first solve more too. Each must answer as the program says, since a
    warm-up that failed would hide the engine's own fault.
    """
    # Time: 88 solves of this eight-clause program, once per thread.
    _SETTLED.done = True
    rewrites = (Rewrite("d", "c", (1, 0)), Rewrite("e", "d", (0, 1)), Rewrite("g", "d", (0, 1)),
                Rewrite("f", "g", (1, 0)), Rewrite("g", "f", (1, 0)))
    facts = (Atom("a", ("k", "j")), Atom("b", ("j",)))
    rules = (Rule(Atom("c", ("X", "Y")), (Atom("a", ("X", "Y")), Atom("b", ("Y",)))),)
    queries = ((Atom("e", ("X", "Y")), {("j", "k")}), (Atom("f", ("X", "Y")), {("k", "j")}))

    def expand(p: str) -> list[Rewrite]:
        return [rw for rw in rewrites if rw.conclusion == p]

    # f(k, j) takes three rewrite steps and one rule step, so a bound of 4 reaches it, and e(j, k), counting either.
    bounds = ((0, "steps"), (4, "steps"), (4, "rewrites"))
    try:
        for tabled, (depth, counts), lazy, proofs, seconds in itertools.product(
                (True, False), bounds, (False, True), (False, True), (math.inf, 60.0)):
            program = Program(facts=facts, rules=rules, rewrites=() if lazy else rewrites, tabled=tabled,
                              depth=depth, depth_counts=counts, proofs=proofs)
            for query, answers in queries if tabled or depth else queries[:1]:
                solution = engine.solve(program, query, Budget(work=1_000_000, seconds=seconds),
                                        expand if lazy else None)
                if solution.status != "ok" or solution.answers != answers:
                    msg = (f"the warm-up solve of {query} in {program} ended {solution.status} with "
                           f"{sorted(solution.answers)}: {solution.detail}")
                    raise RuntimeError(msg)
    except BaseException:
        _SETTLED.done = False
        raise


def _read_answers(solution: Solution, result: list, query: Atom, consts: _Names, reader: _ProofReader | None) -> None:
    """Fill the solution's answers from the evaluation's (ans a1 ... an) terms, each with its proof term last when
    proofs were asked for, the shortest proof of each answer kept; anything else makes the run an error."""
    width = query.arity + 1 + (reader is not None)
    answers: set[tuple[str, ...]] = set()
    shortest: dict[tuple[str, ...], tuple[int, object]] = {}
    for atom in result:
        children: list[object] = list(atom.children) if isinstance(atom, Expression) else []
        if len(children) != width or str(children[0]) != "ans":
            solution.status, solution.detail = "error", f"unexpected answer {atom}"
            return
        row = tuple(consts.undo(str(c)) for c in children[1:query.arity + 1])
        answers.add(row)
        if reader is not None:
            size = reader.size(children[-1])
            if size is None:
                solution.status, solution.detail = "error", f"answer {atom} has no proof term"
                return
            if row not in shortest or size < shortest[row][0]:
                shortest[row] = (size, children[-1])
    proofs: dict[tuple[str, ...], Proof] = {}
    for row, (_, term) in shortest.items():
        try:
            proofs[row] = reader.read(term, Atom(query.pred, row)) if reader is not None else Proof(Atom(query.pred, row))
        except ValueError as e:
            solution.status, solution.detail = "error", f"the proof of {Atom(query.pred, row)}: {e}"
            return
    solution.answers = answers
    if reader is not None:
        solution.proofs = proofs


class _ProofReader:
    """Reads the proof terms of either encoding back as Proof trees over the program's own rules and rewrites."""

    def __init__(self, program: Program, known: dict[Key, Rewrite], preds: _Names, consts: _Names) -> None:
        self.rules = program.rules
        self.free = [_free(rule) for rule in program.rules]
        self.known = known
        self.preds, self.consts = preds, consts

    @staticmethod
    def size(term: object) -> int | None:
        """The step count a (pf n node) term carries, or None when the term is not one."""
        parts = term.children if isinstance(term, Expression) else ()
        if len(parts) == 3 and str(parts[0]) == _PROOF and str(parts[1]).isdigit():
            return int(str(parts[1]))
        return None

    # Time: O(n) for a term of n nodes, built bottom-up from an explicit stack, so no recursion limit bounds a proof.
    def read(self, term: object, atom: Atom) -> Proof:
        """The Proof that `term` gives of `atom`; ValueError when the term is no derivation of it."""
        built: list[Proof] = []
        # A pending node is (term, atom); a node whose premises are pending is (None, atom, step, premise count).
        todo: list[tuple] = [(term, atom)]
        while todo:
            item = todo.pop()
            if item[0] is None:
                _, done, by, n = item
                children = tuple(built[len(built) - n:])
                del built[len(built) - n:]
                built.append(Proof(done, by, children))
                continue
            by, premises = self._step(*item)
            todo.append((None, item[1], by, len(premises)))
            todo.extend(reversed(premises))
        return built[0]

    def _step(self, term: object, atom: Atom) -> tuple[Rule | Rewrite | None, list[tuple[object, Atom]]]:
        """What derived `atom` at this node, and each premise's proof term with the atom it proves."""
        if not isinstance(term, Expression) or self.size(term) is None:
            msg = f"unexpected proof term {term}"
            raise ValueError(msg)
        node = term.children[2]
        if isinstance(node, Symbol) and str(node) == _FACT:
            return None, []
        parts = node.children if isinstance(node, Expression) else ()
        head = str(parts[0]) if parts else ""
        if head == _REWRITE and len(parts) == 4:
            key = (self.preds.undo(str(parts[1])), atom.pred, _mapping_of(str(parts[2])))
            rw = self.known.get(key)
            if rw is None:
                msg = f"a step names the rewrite {key}, which the solve never had"
                raise ValueError(msg)
            return rw, [(parts[3], Atom(rw.premise, tuple(atom.args[i] for i in rw.mapping)))]
        if head == _RULE and len(parts) >= 2 and str(parts[1]).isdigit() and int(str(parts[1])) < len(self.rules):
            j = int(str(parts[1]))
            rule, free = self.rules[j], self.free[j]
            if len(parts) != 2 + len(free) + len(rule.body):
                msg = f"the step {term} doesn't fit rule {j}, {rule}"
                raise ValueError(msg)
            binding = dict(zip(free, (self.consts.undo(str(v)) for v in parts[2:2 + len(free)]), strict=True))
            binding.update((t, v) for t, v in zip(rule.head.args, atom.args, strict=True) if is_variable(t))
            body = [Atom(a.pred, tuple(binding[t] if is_variable(t) else t for t in a.args)) for a in rule.body]
            return rule, list(zip(parts[2 + len(free):], body, strict=True))
        msg = f"unexpected proof step {node}"
        raise ValueError(msg)


def _free(rule: Rule) -> tuple[str, ...]:
    """The variables a rule's body binds and its head lacks, in order of first occurrence: what a proof-rule step
    carries beside its premises' proofs, so each premise's atom can be rebuilt."""
    head = rule.head.variables()
    return tuple(dict.fromkeys(t for a in rule.body for t in a.args if is_variable(t) and t not in head))


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


def _rule_step(j: int, rule: Rule) -> str:
    """The node of a step by the program's rule j, over its free variables and its premises' proofs."""
    parts = [_RULE, str(j), *(f"${v}" for v in _free(rule)), *(f"$pf{i}" for i in range(len(rule.body)))]
    return f"({' '.join(parts)})"


def _declarations(tables: list[tuple[str, int]], terms: _ProofTerms | None) -> tuple[list[str], list[str]]:
    """The forms that table each (name, arity) once the source is loaded, and those that undo them after evaluation.

    Without proofs each is a plain tabled declaration, gone with its context. With proofs each name gets a cache
    row in &metta whose lattice join keeps a table's shortest proof; &metta outlives the context, which would keep
    every solve's rows, so each row is removed again.
    """
    if terms is None:
        return [f"!(tabled ({name} {' '.join(f'$X{i}' for i in range(n))}))" for name, n in tables], []
    rows = [f"(cache {name} (plain private (lattice {terms.keep})))" for name in dict.fromkeys(n for n, _ in tables)]
    return [f"!(add-atom &metta {row})" for row in rows], [f"!(remove-atom &metta {row})" for row in rows]


def _native_source(program: Program, query: Atom, rewrites: Iterable[Rewrite], preds: _Names, consts: _Names,
                   terms: _ProofTerms | None) -> tuple[str, list[tuple[str, int]]]:
    """Facts, rewrite equations and rules as native equations, a reach fallback per predicate, and the tables.

    `rewrites` holds each of the program's rewrites once. Rewrites come before
    user rules, so a rule body's call to a predicate only rewrites define is
    compiled after that predicate exists. Every predicate gets a fallback
    equation calling the reach op, which yields nothing but records that
    evaluation called the predicate, for the rounds strategy. The op call also
    keeps pymetta's tabling declarations linear in the number of predicates:
    without a clause calling an op, declaring 240 tabled functions that call
    each other took 2.6 s against 0.7 s with one, and a geography solve three
    times as long. Its calls are about half of a native solve's inferences.
    With proof `terms`, every equation answers its proof term instead of True.
    """
    def call(a: Atom) -> str:
        return _call(a, preds, consts)

    lines = [f"(= {call(f)} {terms.leaf if terms else 'True'})" for f in program.facts]
    arity: dict[str, int] = {f.pred: f.arity for f in program.facts}
    for rw in rewrites:
        rule = rw.rule()
        node = f"({_REWRITE} {preds(rw.premise)} {_mapping_symbol(rw.mapping)} $pf0)"
        body = terms.body([call(a) for a in rule.body], node) if terms else _body(rule.body, call)
        lines.append(f"(= {call(rule.head)} {body})")
        arity[rw.conclusion] = arity[rw.premise] = rw.arity
    for j, rule in enumerate(program.rules):
        body = terms.body([call(a) for a in rule.body], _rule_step(j, rule)) if terms else _body(rule.body, call)
        lines.append(f"(= {call(rule.head)} {body})")
        for a in (rule.head, *rule.body):
            arity[a.pred] = a.arity
    arity[query.pred] = query.arity
    for p, n in sorted(arity.items()):
        args = " ".join(f"$X{i}" for i in range(n))
        lines.append(f'(= ({preds(p)} {args}) (reach "{preds(p)}"))')
    if terms:
        lines += terms.definitions()
    return "\n".join(lines), [(preds(p), n) for p, n in sorted(arity.items())]


def _meta_source(program: Program, query: Atom, preds: _Names, consts: _Names, tag: int,
                 terms: _ProofTerms | None) -> tuple[str, list[tuple[str, int]]]:
    """The program as data under holds, with the discovery clause, a via clause per mapping, and tabling when asked.

    For each arity n present, holds takes the predicate and n arguments, plus
    the depth left when the program has a depth bound: every rewrite step spends
    one unit, and every user rule step too unless the bound counts rewrites
    alone, when a rule passes the depth left to its body unchanged; facts hold at
    any depth, and tabling then keys each table on the depth left too. holds, via
    and fact carry the solve's tag; facts are fact atoms in the space, which a
    match over an arity with none finds empty. Without proofs the tabling
    declarations are in the source; with proof `terms`, holds answers proof terms
    and the table to declare through a lattice row comes back beside the source.
    """
    holds_, via_, fact_ = f"holds{tag}", f"via{tag}", f"fact{tag}"
    depth = program.depth > 0
    d, dm = (" $d", " (- $d 1)") if depth else ("", "")
    # The depth left that a user rule's body goals get, and whether a rule needs depth left to fire.
    rule_counts = depth and program.depth_counts == "steps"
    body_left = dm if rule_counts else d
    arities = {query.arity} | {f.arity for f in program.facts} | {a.arity for r in program.rules for a in (r.head, *r.body)}
    arities |= {rw.arity for rw in program.rewrites}
    lines = [f"({fact_} {preds(f.pred)} {' '.join(consts(t) for t in f.args)})" for f in program.facts]
    for n in sorted(arities):
        xs = " ".join(f"$x{i}" for i in range(n))
        found = f"(match &self ({fact_} $p {xs}) {terms.leaf if terms else 'True'})"
        lines.append(f"(= ({holds_} $p {xs}{d}) {found})")
        via = f"({via_} $q $m {xs}{dm})"
        follow = terms.body([via], f"({_REWRITE} $q $m $pf0)") if terms else via
        discover = f"(let (rw $q $m) (impliers $p) {follow})"
        lines.append(f"(= ({holds_} $p {xs}{d}) {f'(if (> $d 0) {discover} (empty))' if depth else discover})")
        for m in mappings(n):
            permuted = " ".join(f"$x{i}" for i in m)
            lines.append(f"(= ({via_} $q {_mapping_symbol(m)} {xs}{d}) ({holds_} $q {permuted}{d}))")
        if program.tabled and not terms:
            lines.append(f"!(tabled ({holds_} $p {xs}{d}))")

    def holds(a: Atom) -> str:
        return f"({holds_} {preds(a.pred)} {' '.join(_term(t, consts) for t in a.args)}{body_left})"

    for j, rule in enumerate(program.rules):
        head = f"({holds_} {preds(rule.head.pred)} {' '.join(_term(t, consts) for t in rule.head.args)}{d})"
        body = terms.body([holds(a) for a in rule.body], _rule_step(j, rule)) if terms else _body(rule.body, holds)
        lines.append(f"(= {head} {f'(if (> $d 0) {body} (empty))' if rule_counts else body})")
    if terms:
        lines += terms.definitions()
    return "\n".join(lines), [(holds_, 0)] if terms and program.tabled else []


def _query_term(query: Atom, preds: _Names, consts: _Names, *, meta: bool, depth: int, tag: int, proofs: bool) -> str:
    args = " ".join(_term(t, consts) for t in query.args)
    call = f"(holds{tag} {preds(query.pred)} {args}{f' {depth}' if depth else ''})" if meta else f"({preds(query.pred)} {args})"
    return f"(let $pf {call} (ans {args} $pf))" if proofs else f"(let True {call} (ans {args}))"


def _impliers(program: Program, expand: Expander | None, reached: set[str], preds: _Names,
              known: dict[Key, Rewrite] | None) -> Callable[[Symbol], Iterator[Expression]]:
    """The discovery op: the program's rewrites for a predicate, then whatever the callback returns for it, each once.

    It calls the callback on every call, as evaluation reaches the predicate,
    so a discoverer without a memo is asked again each time; a memoised one
    answers from its memo. With proofs it files each rewrite it yields in
    `known`, where the proof reader finds the rewrite a step names.
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
            if known is not None:
                known.setdefault(rw.key, rw)
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
