"""Purpose: the semantic rewriter itself, independent of any engine: strategies as
settings of five knobs, the rewriter that runs a strategy over any engine
implementing semrewrite.protocols.Engine, and alias canonicalisation.

| knob | settings |
|---|---|
| discovery timing | none, lazy inside evaluation, eager worklist, rounds, saturate |
| per-predicate memo | on, off |
| tables | on, off |
| depth bound | none, at most n rule and rewrite steps, or at most n rewrite steps |
| alias classes | merged, kept |

The rewriter owns discovery and hands the engine either every rewrite up
front (eager, saturate, rounds) or a callback it calls while it evaluates
(lazy). It refuses a strategy whose needs the engine's capabilities do not
cover, rather than running something else under its name.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, get_args

from semrewrite.claims import weigh
from semrewrite.discovery import Discoverer, ScoreCache
from semrewrite.evidence import rewrite_evidence, statements
from semrewrite.kb import KB
from semrewrite.lineage import lineages, moments, probability
from semrewrite.model import Atom, Key, Mapping, Proposal, Rewrite, Rule, compose, identity, inverse, parse_atom
from semrewrite.opinion import Opinion
from semrewrite.protocols import Admission, Budget, DepthCounts, Engine, Judge, Program, Proof, Retriever, Solution
from semrewrite.trace import Solved, Solving, Tracer

if TYPE_CHECKING:
    from semrewrite.claims import ClaimCheck, Reading

# The protocol's default budget; a Budget is frozen, so one instance serves every rewriter.
DEFAULT_BUDGET = Budget()


class Timing(StrEnum):
    """When rewrites are discovered relative to evaluation."""

    NONE = "none"
    EAGER = "eager"
    LAZY = "lazy"
    ROUNDS = "rounds"
    SATURATE = "saturate"


@dataclass(frozen=True, slots=True)
class Strategy:
    """One setting of the knobs, named for the implementation it is; depth 0 means no depth bound, and depth_counts
    says whether a bound counts every rule and rewrite step or rewrite steps alone."""

    name: str
    timing: Timing
    memo: bool = True
    tabled: bool = True
    depth: int = 0
    depth_counts: DepthCounts = "steps"
    canonical: bool = False

    def __post_init__(self) -> None:
        # Timing is a StrEnum, so "lazy" compares equal to Timing.LAZY but is not it; store the member, so every
        # check below sees one representation and an unknown name fails here.
        object.__setattr__(self, "timing", Timing(self.timing))
        if self.depth < 0:
            msg = f"strategy {self.name}: a depth bound counts steps, so it can't be negative"
            raise ValueError(msg)
        if self.depth_counts not in get_args(DepthCounts):
            msg = (f"strategy {self.name}: depth_counts is {self.depth_counts!r}; "
                   f"a depth bound counts one of {get_args(DepthCounts)}")
            raise ValueError(msg)
        # Discovery before evaluation needs no memo switch, tables-off or depth variants: those exist
        # to show what lazy discovery does without them.
        if self.timing is not Timing.LAZY and (not self.memo or not self.tabled or self.depth):
            msg = f"strategy {self.name}: only lazy discovery runs without a memo, without tables or under a depth bound"
            raise ValueError(msg)
        if self.canonical and self.timing is not Timing.EAGER:
            msg = f"strategy {self.name}: merging alias classes needs the eager rule set before evaluation"
            raise ValueError(msg)

    def needs(self) -> dict[str, bool]:
        """The engine capabilities this strategy relies on."""
        return {
            "tabling": self.tabled,
            "untabled": not self.tabled,
            "depth": self.depth > 0 and self.depth_counts == "steps",
            "rewrite_depth": self.depth > 0 and self.depth_counts == "rewrites",
            "lazy": self.timing is Timing.LAZY,
            "reached": self.timing is Timing.ROUNDS,
        }


STRATEGIES: dict[str, Strategy] = {
    s.name: s
    for s in (
        Strategy("exact", Timing.NONE),
        Strategy("naive", Timing.LAZY, memo=False, tabled=False),
        Strategy("memo-untabled", Timing.LAZY, tabled=False),
        Strategy("nomemo-tabled", Timing.LAZY, memo=False),
        Strategy("depth-bounded", Timing.LAZY, tabled=False, depth=3),
        Strategy("depth-tabled", Timing.LAZY, depth=3),
        Strategy("rewrite-depth-tabled", Timing.LAZY, depth=3, depth_counts="rewrites"),
        Strategy("lazy-tabled", Timing.LAZY),
        Strategy("rounds", Timing.ROUNDS),
        Strategy("eager-worklist", Timing.EAGER),
        Strategy("alias-canonical", Timing.EAGER, canonical=True),
        Strategy("saturate", Timing.SATURATE),
    )
}


class CapabilityError(RuntimeError):
    """The engine cannot run the strategy asked for."""


@dataclass
class Outcome:
    """One query's answers under one strategy, judge and engine, with what they cost and how the run ended.

    `admitted` holds the keys of the rewrites the judge admitted while
    answering, as discovery saw them (before any alias renaming); `proofs`
    holds a derivation per answer when the engine provides them; `beliefs`, when the rewriter was asked for them,
    holds each answer's opinion that some derivation's rewrites are all valid, and `rewrite_beliefs` each admitted
    rewrite's opinion that it is valid, which those answers' opinions are computed from; `readings`, when the
    rewriter has a claim check, holds the statements read against each admitted rewrite (semrewrite.claims).
    """

    kb: str
    strategy: str
    judge: str
    engine: str
    query: str
    answers: set[tuple[str, ...]] = field(default_factory=set)
    derivations: int = 0
    status: str = "ok"
    detail: str = ""
    expansions: int = 0
    proposals: int = 0
    retrievals: int = 0
    admitted: set[Key] = field(default_factory=set)
    rounds: int = 0
    work: int = 0
    tables: int = 0
    seconds: float = 0.0
    proofs: dict[tuple[str, ...], Proof] | None = None
    beliefs: dict[tuple[str, ...], Opinion] | None = None
    rewrite_beliefs: dict[Key, Opinion] | None = None
    readings: dict[Key, tuple[Reading, ...]] | None = None

    @property
    def finished(self) -> bool:
        """True when evaluation finished inside its budget."""
        return self.status == "ok"


class Rewriter:
    """Answers queries over one knowledge base with one judge on one engine, under any strategy the engine can run."""

    def __init__(
        self,
        kb: KB,
        judge: Judge | ScoreCache,
        engine: Engine,
        retriever: Retriever | None = None,
        *,
        k: int = 8,
        budget: Budget = DEFAULT_BUDGET,
        admission: Admission | None = None,
        proofs: bool = False,
        trace: Tracer | None = None,
        beliefs: int = 0,
        data_trust: float = 0.0,
        claims: ClaimCheck | None = None,
    ) -> None:
        self.kb = kb
        self.catalogue = kb.catalogue()
        self.scorer = judge if isinstance(judge, ScoreCache) else ScoreCache(judge)
        self.engine = engine
        self.retriever = retriever
        self.k = k
        self.budget = budget
        self.admission = admission
        # Ask the engine for a derivation of every answer, which Outcome.proofs then holds.
        self.proofs = proofs
        # Receives each solve's start and stop, and is given to every discoverer this rewriter makes.
        self.trace = trace
        # With beliefs = k > 0, each finished ask gives every answer a belief from its lineage, keeping the k most
        # probable rewrite sets per row (semrewrite.lineage says why the bound).
        self.beliefs = beliefs
        # The probability that the knowledge base's evidence on a rewrite is right (semrewrite.evidence), which
        # discounts that evidence before it is fused with the judge's opinion; 0 leaves the judge's opinion alone.
        if not 0.0 <= data_trust <= 1.0:
            msg = f"data trust is a probability, not {data_trust}"
            raise ValueError(msg)
        self.data_trust = data_trust
        # Reads each admitted rewrite against the knowledge base's statements by meaning (semrewrite.claims); its
        # readings fuse into the rewrite's opinion after the knowledge base's evidence does.
        self.claims = claims
        self._rules_by_head: dict[str, list[Rule]] = defaultdict(list)
        for rule in kb.rules:
            self._rules_by_head[rule.head.pred].append(rule)

    def discoverer(self, memo: bool = True) -> Discoverer:
        """A fresh discoverer over this knowledge base's catalogue; share one across asks to share its memo."""
        return Discoverer(self.catalogue, self.retriever, self.scorer, k=self.k, memo=memo, admission=self.admission,
                          trace=self.trace)

    def check(self, strategy: Strategy) -> None:
        """Raise CapabilityError naming every capability `strategy`, and proofs when asked for, needs that the
        engine lacks, or when proofs are asked of a strategy that answers over merged predicates."""
        caps = self.engine.capabilities
        missing = [cap for cap, needed in strategy.needs().items() if needed and not getattr(caps, cap)]
        if self.proofs and not caps.proofs:
            missing.append("proofs")
        if missing:
            msg = f"engine {self.engine.name} cannot run {strategy.name}: it lacks {', '.join(missing)}"
            raise CapabilityError(msg)
        if self.proofs and strategy.canonical:
            msg = f"{strategy.name} answers over predicates merged into one per alias class, so it has no derivations"
            raise CapabilityError(msg)
        if self.beliefs and strategy.canonical:
            msg = f"{strategy.name} answers over predicates merged into one per alias class, so it has no lineage"
            raise CapabilityError(msg)

    def ask(self, query: Atom | str, strategy: Strategy | str = "lazy-tabled", discoverer: Discoverer | None = None) -> Outcome:
        """Answer `query` under `strategy`; pass a discoverer to share its memo across queries."""
        query = parse_atom(query) if isinstance(query, str) else query
        strategy = STRATEGIES[strategy] if isinstance(strategy, str) else strategy
        if query.pred not in self.catalogue:
            msg = f"query predicate {query.pred!r} has no gloss in {self.kb.name}"
            raise KeyError(msg)
        if self.catalogue[query.pred].arity != query.arity:
            msg = f"{query} has {query.arity} arguments, but {query.pred} takes {self.catalogue[query.pred].arity}"
            raise ValueError(msg)
        self.check(strategy)
        disc = discoverer if discoverer is not None else self.discoverer(memo=strategy.memo)
        if disc.memo != strategy.memo:
            msg = f"strategy {strategy.name} needs memo={strategy.memo}, the discoverer has memo={disc.memo}"
            raise ValueError(msg)
        before = (disc.stats.expansions, disc.stats.proposals, disc.stats.retrievals)
        out = Outcome(self.kb.name, strategy.name, self.scorer.judge.name, self.engine.name, str(query))
        started = time.perf_counter()
        run = {
            Timing.NONE: self._exact,
            Timing.EAGER: self._eager,
            Timing.SATURATE: self._saturate,
            Timing.LAZY: self._lazy,
            Timing.ROUNDS: self._rounds,
        }[strategy.timing]
        run(strategy, query, disc, out)
        if self.beliefs and out.finished:
            self._believe(query, out)
        out.seconds = time.perf_counter() - started
        out.expansions = disc.stats.expansions - before[0]
        out.proposals = disc.stats.proposals - before[1]
        out.retrievals = disc.stats.retrievals - before[2]
        return out

    def rewrite_opinions(self, keys: Iterable[Key]) -> tuple[tuple[Rewrite, ...], dict[Key, Opinion]]:
        """The rewrites with these (premise, conclusion, mapping) keys, each at its score, and the opinion that each is
        valid: its judge's opinion discounted by the judge's trust, fused by cumulative fusion (Jøsang 2016,
        Def. 12.5) with the knowledge base's evidence on it (semrewrite.evidence) discounted by the data trust, and,
        with a claim check, with the readings of the statements that bear on its claim (semrewrite.claims.weigh).
        Both read only the knowledge base, so a rewrite's opinion does not depend on which other rewrites the keys
        hold. The score cache holds every proposal discovery asked, so no judge is asked again about a proposal, and
        the check reads each claim once."""
        cat = self.catalogue
        ordered = sorted(keys)
        proposals = [Proposal(cat[p], cat[c], m) for p, c, m in ordered]
        rewrites = tuple(Rewrite(c, p, m, s) for (p, c, m), s in zip(ordered, self.scorer.score(proposals), strict=True))
        opinion = {rw.key: o for rw, o in zip(rewrites, self.scorer.opinions(proposals), strict=True)}
        if self.data_trust:
            evidence = rewrite_evidence(statements(self.kb.facts), rewrites)
            # Both sources are of one proposition, the rewrite, so the evidence takes the judge's base rate (SLN 3.1).
            opinion = {key: o.fuse(Opinion.from_evidence(*evidence[key], o.base_rate).discount(self.data_trust))
                       for key, o in opinion.items()}
        if self.claims is not None:
            readings = self.claims.readings(rewrites)
            opinion = {key: weigh(o, readings[key]) for key, o in opinion.items()}
        return rewrites, opinion

    # -- timings -------------------------------------------------------------

    def _believe(self, query: Atom, out: Outcome) -> None:
        """Each answer's opinion that the rewrites of some derivation are all valid.

        Each admitted rewrite is valid with a probability distributed as the Beta of its opinion (rewrite_opinions),
        independently of the others. The answer's probability is then a polynomial in those probabilities, read off
        its lineage over the knowledge base and every admitted rewrite with no depth bound, since a bound limits
        search and not what the admitted rewrites imply. Its mean and
        variance are exact, and its opinion is the Beta matched to those two moments (Cerutti, Kaplan, Kimmig and
        Şensoy, Machine Learning 2022), with the probability its lineage has at the rewrites' base rates as its
        base rate, as the product and coproduct of opinions take theirs (Jøsang 2016, Eqs. 7.1 and 7.3). An answer
        the knowledge base derives with no rewrite is certain. A lineage the work budget stops leaves the run at
        work_limit with no answers, as an engine's would."""
        rewrites, opinion = self.rewrite_opinions(out.admitted)
        out.rewrite_beliefs = opinion
        out.readings = self.claims.readings(rewrites) if self.claims is not None else None
        mean = {key: o.projection for key, o in opinion.items()}
        square = {key: o.projection**2 + o.variance for key, o in opinion.items()}
        base = {key: o.base_rate for key, o in opinion.items()}
        found = lineages(Program(facts=self.kb.facts, rules=self.kb.rules, rewrites=rewrites), query, mean,
                         self.beliefs, self.budget.work)
        out.work += found.work
        if not found.finished:
            out.status, out.detail = "work_limit", f"lineage stopped after {found.work:,} work"
            out.answers, out.beliefs = set(), None
            return
        missing = sorted(row for row in out.answers if row not in found.answers)
        if missing:
            msg = f"the lineage of {query} lacks the engine's answers {missing[:3]}"
            raise RuntimeError(msg)
        beliefs = {}
        for row in out.answers:
            lin = found.answers[row]
            first, second = moments(lin, mean, square)
            variance = second - first * first
            # E[f^2] - E[f]^2 cancels to a few ulps below zero where the answer is all but certain.
            if -1e-12 < variance < 0.0:
                variance = 0.0
            beliefs[row] = Opinion.from_moments(first, variance, probability(lin, base))
        out.beliefs = beliefs

    def _program(self, rewrites: Iterable[Rewrite] = (), *, kb: KB | None = None, tabled: bool = True, depth: int = 0,
                 depth_counts: DepthCounts = "steps") -> Program:
        kb = kb if kb is not None else self.kb
        return Program(facts=kb.facts, rules=kb.rules, rewrites=tuple(rewrites), tabled=tabled, depth=depth,
                       depth_counts=depth_counts, proofs=self.proofs)

    def _solve(self, program: Program, query: Atom, out: Outcome, expand=None, *, accumulate: bool = False,
               budget: Budget | None = None) -> Solution:
        if self.trace is not None:
            self.trace(Solving(self.engine.name, query, program))
        solution = self.engine.solve(program, query, budget if budget is not None else self.budget, expand)
        if self.trace is not None:
            self.trace(Solved(self.engine.name, solution))
        out.status, out.detail = solution.status, solution.detail
        out.answers = set(solution.answers) if solution.finished else set()
        out.derivations = solution.derivations
        out.work = out.work + solution.work if accumulate else solution.work
        out.tables = solution.tables
        out.proofs = solution.proofs
        return solution

    def _exact(self, strategy: Strategy, query: Atom, disc: Discoverer, out: Outcome) -> None:
        self._solve(self._program(), query, out)

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
                pending.extend(a.pred for a in rule.body)
        return list(found.values())

    def _eager(self, strategy: Strategy, query: Atom, disc: Discoverer, out: Outcome) -> None:
        rewrites = self._worklist(query.pred, disc)
        out.admitted = {rw.key for rw in rewrites}
        if not strategy.canonical:
            self._solve(self._program(rewrites), query, out)
            return
        kb, rewrites, renamed, back = canonicalise(self.kb, rewrites, query)
        self._solve(self._program(rewrites, kb=kb), renamed, out)
        # The engine answered the root predicate; put each answer back in the query's argument order.
        out.answers = {tuple(row[back[j]] for j in range(len(row))) for row in out.answers}
        if out.proofs is not None:
            out.proofs = None

    def _saturate(self, strategy: Strategy, query: Atom, disc: Discoverer, out: Outcome) -> None:
        found: dict[Key, Rewrite] = {}
        for p in self.catalogue:
            for rw in disc.rewrites(p):
                found.setdefault(rw.key, rw)
        out.admitted = set(found)
        self._solve(self._program(found.values()), query, out)

    def _lazy(self, strategy: Strategy, query: Atom, disc: Discoverer, out: Outcome) -> None:
        admitted: set[Key] = set()

        def expand(p: str) -> Iterator[Rewrite]:
            if p not in self.catalogue:
                return
            for rw in disc.rewrites(p):
                admitted.add(rw.key)
                yield rw

        self._solve(self._program(tabled=strategy.tabled, depth=strategy.depth, depth_counts=strategy.depth_counts),
                    query, out, expand)
        out.admitted = admitted

    def _rounds(self, strategy: Strategy, query: Atom, disc: Discoverer, out: Outcome) -> None:
        """Evaluate, expand every predicate the evaluation reached, add the rewrites and evaluate again,
        until a round reaches nothing new: the discover-then-reduce loop run to its fixpoint.

        The rounds share one budget: each evaluation gets the work and seconds the
        earlier ones left, so `ok` means the query as a whole stayed within it. A
        round whose expansions add no rewrite ends the loop, since evaluating the
        same program again would reach the same predicates.
        """
        added: dict[Key, Rewrite] = {}
        expanded: set[str] = set()
        started = time.perf_counter()
        while True:
            out.rounds += 1
            left = Budget(work=self.budget.work - out.work, seconds=self.budget.seconds - (time.perf_counter() - started))
            solution = self._solve(self._program(added.values()), query, out, accumulate=True, budget=left)
            if not solution.finished:
                break
            if solution.reached is None:
                msg = f"engine {self.engine.name} claims the reached capability but reported no reached predicates"
                raise CapabilityError(msg)
            fresh = sorted(p for p in solution.reached - expanded if p in self.catalogue)
            before = len(added)
            for p in fresh:
                expanded.add(p)
                for rw in disc.rewrites(p):
                    added.setdefault(rw.key, rw)
            if len(added) == before:
                break
        out.admitted = set(added)


class GroupUnionFind:
    """Union-find over predicates whose edges carry an argument mapping, joining aliases of any arity.

    union(p, q, m) records p(X) <=> q(X[m]), where X[m] is the tuple whose
    position i holds X[m[i]]. find(p) answers (root, g) with p(X) <=> root(X[g]).
    A class asked to hold one member under two mappings gains an automorphism
    of its root, root(W) <=> root(W[h]), as a symmetric binary predicate has
    the swap; the automorphisms are kept as a set closed under composition.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._label: dict[str, Mapping] = {}
        self._arity: dict[str, int] = {}
        self.automorphisms: dict[str, set[Mapping]] = defaultdict(set)

    def add(self, p: str, arity: int) -> None:
        if p not in self._parent:
            self._parent[p], self._label[p], self._arity[p] = p, identity(arity), arity

    # Time: amortised O(log n) per find with path compression alone (Tarjan and van Leeuwen, JACM 1984), n = predicates;
    # union keeps the name-order root, so the root is the class's least name; labels compose in O(arity) per step.
    def find(self, p: str) -> tuple[str, Mapping]:
        """The root of p's class and the mapping g with p(X) <=> root(X[g])."""
        path = []
        while self._parent[p] != p:
            path.append(p)
            p = self._parent[p]
        root, g = p, self._label[p]
        # From the node next to the root outwards, each label is composed with its parent's, now relative to the root.
        for node in reversed(path):
            g = compose(self._label[node], g)
            self._label[node], self._parent[node] = g, root
        return root, g

    def _close(self, root: str) -> None:
        group = self.automorphisms[root]
        while True:
            more = {compose(a, b) for a in group for b in group} - group - {identity(self._arity[root])}
            if not more:
                return
            group |= more

    def union(self, p: str, q: str, m: Mapping) -> None:
        """Join p and q as aliases: p(X) <=> q(X[m])."""
        rp, gp = self.find(p)
        rq, gq = self.find(q)
        c = compose(m, gq)  # p(X) <=> rq(X[c])
        if rp == rq:
            h = compose(inverse(gp), c)
            if h != identity(len(h)):
                self.automorphisms[rp].add(h)
                self._close(rp)
            return
        lo, hi = sorted((rp, rq))
        if lo == rp:
            label = compose(inverse(c), gp)  # rq(V) <=> rp(V[label])
        else:
            label = compose(inverse(gp), c)  # rp(W) <=> rq(W[label])
        self._parent[hi], self._label[hi] = lo, label
        moved = self.automorphisms.pop(hi, set())
        for h in moved:
            self.automorphisms[lo].add(compose(inverse(label), compose(h, label)))
        if moved:
            self._close(lo)


def canonicalise(kb: KB, rewrites: Iterable[Rewrite], query: Atom) -> tuple[KB, list[Rewrite], Atom, Mapping]:
    """Rename every alias class to one root predicate, egglog's union-find applied to predicates.

    Mutual implications (q => p under m together with p => q under inverse(m))
    and self-rewrites join classes; facts, rules, cross-class rewrites and the
    query are renamed to the class roots with their arguments re-mapped, and
    rewrites inside a class vanish except the root's automorphisms. Returns
    the renamed knowledge base, rewrites and query, and the mapping `back` that
    puts an answer row R of the root back in the query's order: row[back[j]].
    """
    rewrites = list(rewrites)
    keys = {rw.key for rw in rewrites}
    catalogue = kb.catalogue()
    uf = GroupUnionFind()
    for name, pred in catalogue.items():
        uf.add(name, pred.arity)
    for premise, conclusion, m in keys:
        if premise == conclusion or (conclusion, premise, inverse(m)) in keys:
            uf.union(conclusion, premise, m)

    def rename(a: Atom) -> Atom:
        if a.pred not in catalogue:
            return a
        root, g = uf.find(a.pred)
        return Atom(root, tuple(a.args[i] for i in g))

    facts = tuple(rename(f) for f in kb.facts)
    rules = tuple(Rule(rename(r.head), tuple(rename(b) for b in r.body)) for r in kb.rules)
    out: dict[Key, Rewrite] = {}
    for rw in rewrites:
        rp, gp = uf.find(rw.conclusion)
        rq, gq = uf.find(rw.premise)
        m = compose(inverse(gp), compose(rw.mapping, gq))
        if rp == rq and m == identity(len(m)):
            continue
        if rp == rq and m in uf.automorphisms.get(rp, set()):
            continue
        renamed = Rewrite(rp, rq, m, rw.score)
        out.setdefault(renamed.key, renamed)
    for root, group in uf.automorphisms.items():
        for h in sorted(group):
            out.setdefault((root, root, h), Rewrite(root, root, h, 1.0))
    roots = {uf.find(p)[0] for p in catalogue}
    renamed_kb = KB(name=kb.name, glosses={r: kb.glosses[r] for r in roots}, facts=facts, rules=rules)
    q = rename(query)
    _, g = uf.find(query.pred)
    return renamed_kb, list(out.values()), q, inverse(g)
