"""Purpose: the semantic rewriter on pymetta: strategies that build a MeTTa
program from a knowledge base and its admitted rewrites, evaluate a query under
an inference budget, and report the answers with what they cost.

Every strategy is a setting of four independent knobs:

| knob | settings |
|---|---|
| discovery timing | none, eager worklist, lazy inside evaluation, rounds, saturate |
| per-predicate memo | on, off |
| evaluation | tabled (SLG), untabled (SLD), depth-bounded |
| alias canonicalisation | on, off |

Two encodings carry them. The native encoding compiles each admitted rewrite
into an ordinary equation (= (parent $X $Y) (mother $X $Y)) beside the
knowledge base's own facts, so pymetta evaluates it unchanged. The meta
encoding keeps predicates as data, (holds parent $X $Y), and lets one clause
call the discovery op `impliers` while evaluation runs, which is how discovery
happens inside reduction; tabling `holds` makes that SLG resolution with
semantic goal expansion.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum

from metta import Expression, MeTTa, S, parse
from metta.errors import EngineError, InferenceLimitError, TimeLimitError
from metta.vocabularies import EffectClass

from semrewrite.discovery import Discoverer, Rewrite, ScoreCache
from semrewrite.judging import Judge
from semrewrite.kb import KB, Lit, Pair, Query, Rule, is_variable
from semrewrite.retrieve import Retriever


class Timing(StrEnum):
    """When rules are discovered relative to evaluation."""

    NONE = "none"
    EAGER = "eager"
    LAZY = "lazy"
    ROUNDS = "rounds"
    SATURATE = "saturate"


class Evaluation(StrEnum):
    """How the engine resolves goals: SLG with tables, plain SLD, or SLD cut at a depth."""

    TABLED = "tabled"
    UNTABLED = "untabled"
    DEPTH = "depth"


@dataclass(frozen=True, slots=True)
class Strategy:
    """One setting of the knobs, named for the implementation it is."""

    name: str
    timing: Timing
    memo: bool = True
    evaluation: Evaluation = Evaluation.TABLED
    depth: int = 0
    canonical: bool = False

    def __post_init__(self) -> None:
        if (self.evaluation is Evaluation.DEPTH) != (self.depth > 0):
            msg = f"strategy {self.name}: a depth bound goes with depth-bounded evaluation and nothing else"
            raise ValueError(msg)
        if self.timing in (Timing.EAGER, Timing.ROUNDS, Timing.SATURATE, Timing.NONE) and (
            self.evaluation is Evaluation.DEPTH or not self.memo
        ):
            msg = f"strategy {self.name}: only lazy discovery runs without a memo or under a depth bound"
            raise ValueError(msg)
        if self.canonical and self.timing is not Timing.EAGER:
            msg = f"strategy {self.name}: canonicalisation needs the eager rule set before evaluation"
            raise ValueError(msg)


STRATEGIES: dict[str, Strategy] = {
    s.name: s
    for s in (
        Strategy("exact", Timing.NONE),
        Strategy("naive", Timing.LAZY, memo=False, evaluation=Evaluation.UNTABLED),
        Strategy("memo-untabled", Timing.LAZY, memo=True, evaluation=Evaluation.UNTABLED),
        Strategy("nomemo-tabled", Timing.LAZY, memo=False, evaluation=Evaluation.TABLED),
        Strategy("depth-bounded", Timing.LAZY, memo=True, evaluation=Evaluation.DEPTH, depth=3),
        Strategy("lazy-tabled", Timing.LAZY),
        Strategy("rounds", Timing.ROUNDS),
        Strategy("eager-worklist", Timing.EAGER),
        Strategy("alias-canonical", Timing.EAGER, canonical=True),
        Strategy("saturate", Timing.SATURATE),
    )
}


@dataclass(frozen=True, slots=True)
class Budget:
    """The engine work one query may spend before it counts as not terminating."""

    inferences: int = 2_000_000
    seconds: float = 60.0


@dataclass
class Outcome:
    """One query's answers under one strategy and judge, with its cost and how it ended.

    status is ok, inference_limit, time_limit or error; derivations counts
    answers with repeats, so derivations above len(answers) shows the same
    answer proved more than once. `admitted` holds the implication triples the
    judge admitted while answering, as the judge saw them (before any alias
    renaming), so they can be scored against the gold closure.
    """

    kb: str
    strategy: str
    judge: str
    query: str
    answers: set[Pair] = field(default_factory=set)
    derivations: int = 0
    status: str = "ok"
    detail: str = ""
    expansions: int = 0
    proposals: int = 0
    retrievals: int = 0
    admitted: set[tuple[str, str, bool]] = field(default_factory=set)
    rounds: int = 0
    inferences: int = 0
    tables: int = 0
    seconds: float = 0.0

    @property
    def terminated(self) -> bool:
        """True when evaluation finished inside its budget."""
        return self.status == "ok"

    @property
    def rewrites(self) -> int:
        """How many distinct rewrites the judge admitted while answering."""
        return len(self.admitted)


class ParityUnionFind:
    """Union-find whose edges carry an argument-swap bit, joining aliases.

    union(p, q, swapped) records p(x, y) <=> q(x, y), or p(x, y) <=> q(y, x)
    when swapped. find(p) answers (root, parity): p(x, y) is root(x, y) at
    parity False and root(y, x) at parity True. A class asked to hold one
    member at both parities is symmetric: root(x, y) <=> root(y, x).
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._parity: dict[str, bool] = {}
        self.symmetric: set[str] = set()

    # Time: amortised inverse-Ackermann with path compression; parity composes by xor.
    def find(self, p: str) -> tuple[str, bool]:
        """The root of p's class and p's argument parity relative to it."""
        if p not in self._parent:
            self._parent[p] = p
            self._parity[p] = False
        parent = self._parent[p]
        if parent == p:
            return p, False
        root, parity = self.find(parent)
        self._parent[p] = root
        self._parity[p] = self._parity[p] ^ parity
        return root, self._parity[p]

    def union(self, p: str, q: str, swapped: bool) -> None:
        """Join p and q as aliases, p(x, y) <=> q(x, y), or q(y, x) when swapped."""
        rp, pp = self.find(p)
        rq, pq = self.find(q)
        if rp == rq:
            if pp ^ pq != swapped:
                self.symmetric.add(rp)
            return
        lo, hi = sorted((rp, rq))
        self._parent[hi] = lo
        self._parity[hi] = pp ^ pq ^ swapped
        if hi in self.symmetric:
            self.symmetric.discard(hi)
            self.symmetric.add(lo)


def _term(t: str) -> str:
    return f"${t}" if is_variable(t) else t


def _holds(lit: Lit, extra: str = "") -> str:
    tail = f" {extra}" if extra else ""
    return f"(holds {lit.pred} {_term(lit.a)} {_term(lit.b)}{tail})"


def native_source(kb: KB, rewrites: Iterable[Rewrite], *, tabled: bool) -> tuple[str, list[str]]:
    """The knowledge base plus rewrite equations, and the predicates to table.

    Rewrites come before user rules, so a rule body's call to a predicate only
    rewrites define is compiled after that predicate exists. Only predicates
    with at least one equation are tabled; a predicate with none has nothing
    to table and its calls answer nothing.
    """
    lines = [f"(= ({p} {a} {b}) True)" for p, a, b in kb.facts]
    seen: set[tuple[str, str, bool]] = set()
    for rw in rewrites:
        if rw.key not in seen:
            seen.add(rw.key)
            lines.append(rw.metta())
    lines += [rule.metta() for rule in kb.rules]
    defined = {p for p, _, _ in kb.facts} | {r.head.pred for r in kb.rules} | {k[1] for k in seen}
    return "\n".join(lines), sorted(defined) if tabled else []


def meta_source(kb: KB, *, tabled: bool, depth: bool) -> str:
    """The knowledge base as data under holds/3 (holds/4 with a depth), with the discovery clause.

    Facts become (= (fact p a b) True); a user rule p :- q, r becomes
    (= (holds p ...) (let True (holds q ...) (holds r ...))). One clause asks
    the impliers op for admitted rewrites (rw q same|swap) and follows each,
    so discovery runs when evaluation first reaches a goal predicate.
    """
    lines = [f"(= (fact {p} {a} {b}) True)" for p, a, b in kb.facts]
    if depth:
        lines += [
            "(= (holds $p $x $y $d) (fact $p $x $y))",
            "(= (holds $p $x $y $d) (if (> $d 0) (let (rw $q $s) (impliers $p) (via $q $s $x $y (- $d 1))) (empty)))",
            "(= (via $q same $x $y $d) (holds $q $x $y $d))",
            "(= (via $q swap $x $y $d) (holds $q $y $x $d))",
        ]
        for rule in kb.rules:
            head = _holds(rule.head, "$d")
            body = rule.metta_body(lambda lit: _holds(lit, "(- $d 1)"))
            lines.append(f"(= {head} (if (> $d 0) {body} (empty)))")
    else:
        lines += [
            "(= (holds $p $x $y) (fact $p $x $y))",
            "(= (holds $p $x $y) (let (rw $q $s) (impliers $p) (via $q $s $x $y)))",
            "(= (via $q same $x $y) (holds $q $x $y))",
            "(= (via $q swap $x $y) (holds $q $y $x))",
        ]
        lines += [f"(= {_holds(rule.head)} {rule.metta_body(_holds)})" for rule in kb.rules]
        if tabled:
            lines.append("!(tabled (holds $p $x $y))")
    return "\n".join(lines)


def canonicalise(kb: KB, rewrites: Iterable[Rewrite], query: Query) -> tuple[KB, list[Rewrite], Query, bool]:
    """Rename every alias class to one root predicate, egglog's union-find applied to predicates.

    Mutual implications p <=> q (same order or swapped) join a class; facts,
    rules, cross-class rewrites and the query are renamed to the class root,
    swapping arguments where the parity says so, and rewrites inside a class
    vanish. A symmetric class keeps one self-swap rewrite. Returns the renamed
    knowledge base, rewrites, query, and whether the query's answers come back
    swapped.
    """
    rewrites = list(rewrites)
    keys = {rw.key for rw in rewrites}
    uf = ParityUnionFind()
    for premise, conclusion, swapped in keys:
        if premise == conclusion:
            if swapped:
                uf.union(premise, premise, True)
        elif (conclusion, premise, swapped) in keys:
            uf.union(premise, conclusion, swapped)

    def lit(l: Lit) -> Lit:
        root, parity = uf.find(l.pred)
        return Lit(root, l.b, l.a) if parity else Lit(root, l.a, l.b)

    facts = tuple(
        (renamed.pred, renamed.a, renamed.b)
        for p, a, b in kb.facts
        for renamed in (lit(Lit(p, a, b)),)
    )
    rules = tuple(Rule(lit(r.head), tuple(lit(b) for b in r.body)) for r in kb.rules)
    out: dict[tuple[str, str, bool], Rewrite] = {}
    for rw in rewrites:
        rc, pc = uf.find(rw.conclusion)
        rp, pp = uf.find(rw.premise)
        swapped = rw.swapped ^ pc ^ pp
        if rc == rp:
            # Inside one class a rewrite is the identity unless its parity is odd:
            # p0 <=> p2 with p0(y, x) => p2(x, y) says the class is symmetric.
            if swapped:
                uf.symmetric.add(rc)
            continue
        renamed = Rewrite(rc, rp, swapped, rw.score)
        out.setdefault(renamed.key, renamed)
    for root in sorted(uf.symmetric):
        out.setdefault((root, root, True), Rewrite(root, root, True, 1.0))
    roots = {uf.find(p)[0] for p in kb.glosses}
    glosses = {r: kb.glosses[r] for r in roots}
    renamed_kb = KB(name=kb.name, glosses=glosses, facts=facts, rules=rules)
    q = lit(query.lit)
    _, parity = uf.find(query.pred)
    return renamed_kb, list(out.values()), Query(q.pred, q.a, q.b), parity


class Rewriter:
    """Answers queries over one knowledge base with one judge, under any strategy."""

    def __init__(
        self,
        kb: KB,
        judge: Judge | ScoreCache,
        retriever: Retriever | None,
        *,
        k: int = 8,
        budget: Budget = Budget(),
    ) -> None:
        self.kb = kb
        self.catalogue = kb.catalogue()
        self.scorer = judge if isinstance(judge, ScoreCache) else ScoreCache(judge)
        self.retriever = retriever
        self.k = k
        self.budget = budget
        self._rules_by_head: dict[str, list[Rule]] = defaultdict(list)
        for rule in kb.rules:
            self._rules_by_head[rule.head.pred].append(rule)

    def discoverer(self, memo: bool = True) -> Discoverer:
        """A fresh discoverer over this knowledge base's catalogue."""
        return Discoverer(self.catalogue, self.retriever, self.scorer, k=self.k, memo=memo)

    def ask(self, strategy: Strategy, query: Query, discoverer: Discoverer | None = None) -> Outcome:
        """Answer `query` under `strategy`; pass a discoverer to share its memo across queries."""
        if query.pred not in self.catalogue:
            msg = f"query predicate {query.pred!r} has no gloss in {self.kb.name}"
            raise KeyError(msg)
        disc = discoverer if discoverer is not None else self.discoverer(memo=strategy.memo)
        if disc.memo != strategy.memo:
            msg = f"strategy {strategy.name} needs memo={strategy.memo}, the discoverer has memo={disc.memo}"
            raise ValueError(msg)
        before = (disc.stats.expansions, disc.stats.proposals, disc.stats.retrievals)
        outcome = Outcome(self.kb.name, strategy.name, self.scorer.judge.name, str(query))
        started = time.perf_counter()
        run = {
            Timing.NONE: self._exact,
            Timing.EAGER: self._eager,
            Timing.SATURATE: self._saturate,
            Timing.LAZY: self._lazy,
            Timing.ROUNDS: self._rounds,
        }[strategy.timing]
        run(strategy, query, disc, outcome)
        outcome.seconds = time.perf_counter() - started
        outcome.expansions = disc.stats.expansions - before[0]
        outcome.proposals = disc.stats.proposals - before[1]
        outcome.retrievals = disc.stats.retrievals - before[2]
        return outcome

    # -- strategies ----------------------------------------------------------

    def _exact(self, strategy: Strategy, query: Query, disc: Discoverer, out: Outcome) -> None:
        source, tabled = native_source(self.kb, (), tabled=True)
        with self._context(source, tabled) as m:
            self._evaluate(m, query.metta(), out)

    def _worklist(self, start: str, disc: Discoverer) -> list[Rewrite]:
        """Rewrites reachable from `start`: each expanded predicate's premises and rule bodies join the worklist."""
        pending, seen, found = [start], set(), {}
        while pending:
            p = pending.pop()
            if p in seen:
                continue
            seen.add(p)
            for rw in disc.rewrites(p):
                found.setdefault(rw.key, rw)
                pending.append(rw.premise)
            for rule in self._rules_by_head.get(p, ()):
                pending.extend(lit.pred for lit in rule.body)
        return list(found.values())

    def _eager(self, strategy: Strategy, query: Query, disc: Discoverer, out: Outcome) -> None:
        rewrites = self._worklist(query.pred, disc)
        out.admitted = {rw.key for rw in rewrites}
        kb, swapped = self.kb, False
        if strategy.canonical:
            kb, rewrites, query, swapped = canonicalise(self.kb, rewrites, query)
        source, tabled = native_source(kb, rewrites, tabled=True)
        with self._context(source, tabled) as m:
            self._evaluate(m, query.metta(), out)
        if swapped:
            out.answers = {(b, a) for a, b in out.answers}

    def _saturate(self, strategy: Strategy, query: Query, disc: Discoverer, out: Outcome) -> None:
        found: dict[tuple[str, str, bool], Rewrite] = {}
        for p in self.catalogue:
            for rw in disc.rewrites(p):
                found.setdefault(rw.key, rw)
        out.admitted = set(found)
        source, tabled = native_source(self.kb, found.values(), tabled=True)
        with self._context(source, tabled) as m:
            self._evaluate(m, query.metta(), out)

    def _lazy(self, strategy: Strategy, query: Query, disc: Discoverer, out: Outcome) -> None:
        admitted: set[tuple[str, str, bool]] = set()

        def impliers(p: object) -> Iterator[Expression]:
            for rw in disc.rewrites(str(p)):
                admitted.add(rw.key)
                yield Expression((S.rw, S[rw.premise], S.swap if rw.swapped else S.same))

        depth = strategy.evaluation is Evaluation.DEPTH
        source = meta_source(self.kb, tabled=strategy.evaluation is Evaluation.TABLED, depth=depth)
        lit = query.lit
        a, b = _term(lit.a), _term(lit.b)
        call = f"(holds {lit.pred} {a} {b} {strategy.depth})" if depth else f"(holds {lit.pred} {a} {b})"
        term = f"(let True {call} (ans {a} {b}))"
        with self._context(source, [], ops={"impliers": impliers}, meta=True) as m:
            self._evaluate(m, term, out)
        out.admitted = admitted

    def _rounds(self, strategy: Strategy, query: Query, disc: Discoverer, out: Outcome) -> None:
        """Evaluate, expand every predicate the evaluation reached, add the rewrites, evaluate again,
        until a round reaches nothing new: the discover-then-reduce loop run to its fixpoint.
        """
        reached: set[str] = set()

        def reach(p: object) -> Iterator[Expression]:
            reached.add(str(getattr(p, "value", p)))
            yield from ()

        fallbacks = [f'(= ({p} $X $Y) (reach "{p}"))' for p in self.catalogue]
        source, _ = native_source(self.kb, (), tabled=True)
        source = source + "\n" + "\n".join(fallbacks)
        tabled = sorted(self.catalogue)
        added: dict[tuple[str, str, bool], Rewrite] = {}
        expanded: set[str] = set()
        with self._context(source, tabled, ops={"reach": reach}) as m:
            while True:
                out.rounds += 1
                self._evaluate(m, query.metta(), out, accumulate=True)
                if not out.terminated:
                    break
                fresh = sorted(reached - expanded)
                if not fresh:
                    break
                for p in fresh:
                    expanded.add(p)
                    for rw in disc.rewrites(p):
                        if rw.key not in added:
                            added[rw.key] = rw
                            m.self.run(rw.metta())
        out.admitted = set(added)

    # -- engine plumbing ----------------------------------------------------------

    def _context(
        self,
        source: str,
        tabled: list[str],
        *,
        ops: dict[str, Callable[..., Iterator[Expression]]] | None = None,
        meta: bool = False,
    ) -> _Context:
        return _Context(self, source, tabled, ops or {}, meta)

    def _evaluate(self, m: MeTTa, term: str, out: Outcome, *, accumulate: bool = False) -> None:
        """Evaluate `term` under the budget into `out`; a limit or engine error becomes its status.

        A limit reached inside a Python op can surface as a plain EngineError
        (repros/limit_wrapped_after_exact.py), so an engine error is read as a limit when the
        engine's own inference count or the elapsed time reached the budget.
        """
        result: list = []
        started = time.perf_counter()
        with m.stats() as stats:
            try:
                result = m.eval(parse(term), inferences=self.budget.inferences, timeout=self.budget.seconds)
            except InferenceLimitError as e:
                out.status, out.detail = "inference_limit", str(e)
            except TimeLimitError as e:
                out.status, out.detail = "time_limit", str(e)
            except EngineError as e:
                out.status, out.detail = "error", str(e)
        if out.status == "error" and stats.inferences >= self.budget.inferences:
            out.status = "inference_limit"
        elif out.status == "error" and time.perf_counter() - started >= self.budget.seconds:
            out.status = "time_limit"
        out.inferences = out.inferences + stats.inferences if accumulate else stats.inferences
        answers: set[Pair] = set()
        for atom in result:
            children = atom.children if isinstance(atom, Expression) else ()
            if len(children) == 3 and str(children[0]) == "ans":
                answers.add((str(children[1]), str(children[2])))
            else:
                out.status, out.detail = "error", f"unexpected answer {atom}"
        out.answers = answers
        out.derivations = len(result)
        out.tables = _table_count(m)


def _table_count(m: MeTTa) -> int:
    row = m.runtime.once(
        "space_module(Space, _M), aggregate_all(count, current_table(_M:_G, _), N)",
        Space=m.self.name,
    )
    return int(row["N"])


class _Context:
    """A fresh pymetta context holding one program, with the predicate-name guard applied.

    Predicate names become function names (native) or data (meta); one that is
    also a pymetta builtin would silently change meaning, so it is refused.
    """

    def __init__(self, rewriter: Rewriter, source: str, tabled: list[str], ops: dict, meta: bool) -> None:
        self.rewriter = rewriter
        self.source = source
        self.tabled = tabled
        self.ops = ops
        self.meta = meta
        self.m: MeTTa | None = None

    def __enter__(self) -> MeTTa:
        m = MeTTa()
        self.m = m
        clash = sorted(set(self.rewriter.catalogue) & set(m.self.builtins()))
        if clash:
            m.close()
            msg = f"predicates {clash} collide with pymetta builtins; rename them in the knowledge base"
            raise ValueError(msg)
        for name, fn in self.ops.items():
            m.op(fn, name=name, effect=EffectClass.nondeterministicReadOnly)
        if self.tabled or "!(tabled" in self.source:
            m.self.run("!(import! (context-space) (library lib_tabling))")
        m.self.run(self.source)
        for p in self.tabled:
            declared = m.self.run(f"!(tabled ({p} $X $Y))")
            if declared != [[True]]:
                msg = f"tabling {p} was refused: {declared}"
                raise RuntimeError(msg)
        return m

    def __exit__(self, *exc: object) -> None:
        if self.m is not None:
            self.m.close()
