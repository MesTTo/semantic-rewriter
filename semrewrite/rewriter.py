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
(lazy). Every rewrite discovery admits is checked against what the knowledge
base states (semrewrite.claims): one a statement clearly contradicts is
withdrawn before the engine sees it. After the engine, each application of a
rewrite to a row is checked against what the knowledge base states about the
row's individuals: an answer every derivation of which applies a rewrite to a
row that a statement clearly contradicts is no answer, and a vaguer reading
weighs on that application's leaf in the answers' beliefs. It refuses a
strategy whose needs the engine's capabilities do not cover, rather than
running something else under its name.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, get_args

from semrewrite.claims import APPLICATIONS, default_check, weigh
from semrewrite.discovery import Discoverer, ScoreCache
from semrewrite.evidence import rewrite_evidence, statements
from semrewrite.kb import KB
from semrewrite.lineage import Lineages, Row, Variable, leaf, lineages, moments, probability
from semrewrite.model import Atom, Key, Mapping, Proposal, Rewrite, Rule, compose, identity, inverse, parse_atom
from semrewrite.opinion import Opinion
from semrewrite.protocols import Admission, Budget, DepthCounts, Engine, Judge, Program, Proof, Retriever, Solution
from semrewrite.trace import Excepted, Solved, Solving, Tracer

if TYPE_CHECKING:
    from semrewrite.claims import Application, ClaimCheck, Reading
    from semrewrite.judges.base import SentenceJudge

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
    answering, as discovery saw them (before any alias renaming), and
    `defeated` those the claim check withdrew, each with the readings of the
    statements that clearly contradict it; `proofs`
    holds a derivation per answer when the engine provides them; `beliefs`, when the rewriter was asked for them,
    holds each answer's opinion that some derivation's rewrites are all valid, and `rewrite_beliefs` each admitted
    rewrite's opinion that it is valid, which those answers' opinions are computed from; `readings`, with beliefs and
    a claim check, holds the statements read against each admitted rewrite (semrewrite.claims). Where the check read
    an application, `applications` holds each one read by (rewrite key, row), `applied` every row the lineage applied
    each such rewrite to, and `fallen` the engine's answers every derivation of which a defeated application carries,
    which `answers` no longer holds. `work` counts what the run would count without the exceptions, the engine's and,
    with beliefs, the lineage's; `exception_work` what the exceptions add: the lineage a run without beliefs evaluates
    to find the applications, the lineage evaluated again with the leaves, and the solve showing a derivation they
    leave.
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
    defeated: dict[Key, tuple[Reading, ...]] = field(default_factory=dict)
    rounds: int = 0
    work: int = 0
    exception_work: int = 0
    tables: int = 0
    seconds: float = 0.0
    proofs: dict[tuple[str, ...], Proof] | None = None
    beliefs: dict[tuple[str, ...], Opinion] | None = None
    rewrite_beliefs: dict[Key, Opinion] | None = None
    readings: dict[Key, tuple[Reading, ...]] | None = None
    applications: dict[tuple[Key, Row], Application] | None = None
    applied: dict[Key, frozenset[Row]] | None = None
    fallen: set[tuple[str, ...]] = field(default_factory=set)

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
        reader: SentenceJudge | None = None,
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
        # Reads each admitted rewrite, and each application of one, against the knowledge base's statements by meaning
        # (semrewrite.claims): a clear contradiction withdraws a rewrite in discovery and an application after the
        # engine, and the other readings fuse into its opinion or its leaf. Unless given one, `reader` reads where
        # given, else the run's judge where it carries the trust a reading weighs at, else the check's default reader,
        # and the run's retriever chooses what it reads; None only where the judge reads no sentences, no reader is
        # given and the knowledge base states nothing naming no individual.
        self.claims = claims if claims is not None else default_check(kb, self.scorer.judge, retriever, reader, trace)
        # A check handed in tells this rewriter's trace what it reads with, unless it has a trace of its own.
        if self.claims is not None and self.claims.trace is None:
            self.claims.trace = trace
        self._rules_by_head: dict[str, list[Rule]] = defaultdict(list)
        for rule in kb.rules:
            self._rules_by_head[rule.head.pred].append(rule)

    def discoverer(self, memo: bool = True) -> Discoverer:
        """A fresh discoverer over this knowledge base's catalogue; share one across asks to share its memo."""
        return Discoverer(self.catalogue, self.retriever, self.scorer, k=self.k, memo=memo, admission=self.admission,
                          trace=self.trace, check=self.claims)

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
        if out.finished and (self.beliefs or (self.claims is not None and self.claims.individuals)):
            self._weigh(query, out)
        out.seconds = time.perf_counter() - started
        out.expansions = disc.stats.expansions - before[0]
        out.proposals = disc.stats.proposals - before[1]
        out.retrievals = disc.stats.retrievals - before[2]
        return out

    def rewrite_opinions(self, keys: Iterable[Key]) -> tuple[tuple[Rewrite, ...], dict[Key, Opinion]]:
        """The rewrites with these (premise, conclusion, mapping) keys, each at its score, and the opinion that each is
        valid: its judge's opinion discounted by the judge's trust, fused by cumulative fusion (Jøsang 2016,
        Def. 12.5) with the knowledge base's evidence on it (semrewrite.evidence) discounted by the data trust, and
        with the readings of the statements that bear on its claim, discounted by the judge's check trust
        (semrewrite.claims.weigh).
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

    def _weigh(self, query: Atom, out: Outcome) -> None:
        """The answers the exceptions leave, and with beliefs asked each answer's opinion that the variables of some
        derivation all hold.

        Each admitted rewrite is valid with a probability distributed as the Beta of its opinion (rewrite_opinions),
        independently of the others. An application the claim check reads, a rewrite applied to a row some statement
        about the row's individuals bears on, also needs its leaf, the opinion that it applies normally there
        (semrewrite.claims.Application): a defeated application derives nothing, so an engine answer every derivation
        of which it carries is dropped, and the others weigh on the answers derived through them alone. The lineage
        over the knowledge base and every admitted rewrite, with no depth bound, since a bound limits search and not
        what the admitted rewrites imply, finds the applications; it is evaluated again with the leaves when the check
        reads any. Without beliefs it is evaluated only where a statement an application may read stands out against
        some admitted rewrite (ClaimCheck.bears), since otherwise no application reads anything, and its work counts
        in `exception_work`, as every evaluation the exceptions add does. An answer's probability is then a polynomial
        in its variables' probabilities. Its mean and variance
        are exact, and its opinion is the Beta matched to those two moments (Cerutti, Kaplan, Kimmig and Şensoy,
        Machine Learning 2022), with the probability its lineage has at the variables' base rates as its base rate, as
        the product and coproduct of opinions take theirs (Jøsang 2016, Eqs. 7.1 and 7.3); a leaf's base rate is 1,
        since no exception is assumed where none is read. An answer the knowledge base derives with no rewrite is
        certain. A lineage the work budget stops leaves the run at work_limit with no answers, as an engine's would."""
        rewrites, opinion = self.rewrite_opinions(out.admitted)
        if self.beliefs:
            out.rewrite_beliefs = opinion
            out.readings = self.claims.readings(rewrites) if self.claims is not None else None
        excepting = self.claims is not None and self.claims.individuals
        if not self.beliefs and not (excepting and self.claims is not None and self.claims.bears(rewrites)):
            return
        mean: dict[Variable, float] = {key: o.projection for key, o in opinion.items()}
        program = Program(facts=self.kb.facts, rules=self.kb.rules, rewrites=rewrites)
        found = self._lineage(program, query, mean)
        if self.beliefs:
            out.work += found.work
        else:
            out.exception_work += found.work
        if _stopped(found, out):
            return
        missing = sorted(row for row in out.answers if row not in found.answers)
        if missing:
            msg = f"the lineage of {query} lacks the engine's answers {missing[:3]}"
            raise RuntimeError(msg)
        leaves: dict[Variable, Opinion] = {}
        if excepting and self.claims is not None:
            read = self.claims.applications(found.applied)
            out.applications = read
            # A leaf that no reading disbelieves is certain, x . 1 = x, so it is left out and changes nothing.
            leaves = {leaf(key, row): app.normal for (key, row), app in read.items() if app.normal.projection < 1.0}
            self._trace_exceptions(read)
        if leaves:
            mean |= {var: o.projection for var, o in leaves.items()}
            found = self._lineage(program, query, mean)
            out.exception_work += found.work
            if _stopped(found, out):
                return
            excepted = {key for key, _ in out.applications or {}}
            out.applied = {key: frozenset(rows) for key, rows in found.applied.items() if key in excepted}
            out.fallen = {row for row in out.answers if row not in found.answers}
            out.answers -= out.fallen
            self._prove_around(query, out, rewrites)
        if self.beliefs:
            out.beliefs = answer_beliefs(found, out.answers, {**opinion, **leaves})

    def _lineage(self, program: Program, query: Atom, weights: dict[Variable, float]) -> Lineages:
        """The query's lineage at `weights`, keeping the beliefs' bound of sets per row, or one where only the answers
        are wanted."""
        return lineages(program, query, weights, self.beliefs or 1, self.budget.work)

    def _trace_exceptions(self, read: dict[tuple[Key, Row], Application]) -> None:
        """Tell the trace of each defeated application, with the statements defeating it."""
        if self.trace is None or self.claims is None:
            return
        for (key, row), app in sorted(read.items()):
            if app.defeated:
                premise, conclusion, mapping = key
                self.trace(Excepted(Rewrite(conclusion, premise, mapping), row, app.premise, app.conclusion,
                                    tuple((r.statement, r.stance.disbelief) for r in app.defeats),
                                    self.claims.bar(APPLICATIONS)))

    # Time: one walk of each proof, and where one applies a defeated application one more engine solve, unbounded, of
    # the program with each rewrite holding a defeated application guarded by a fact per row the lineage applied it to.
    def _prove_around(self, query: Atom, out: Outcome, rewrites: Iterable[Rewrite]) -> None:
        """Replace each proof that applies a defeated application with a derivation the exceptions leave: the query
        solved once more, with no depth bound, each rewrite holding a defeated application turned into its rule
        guarded by one fact per row the lineage applied it to that no defeat removes, and each guarded step read back
        as the rewrite's. An answer standing through another derivation thus shows that derivation."""
        if not out.proofs or out.applications is None or out.applied is None:
            return
        defeated = {(key, row) for (key, row), app in out.applications.items() if app.defeated}
        stale = sorted(row for row, proof in out.proofs.items() if row in out.answers and _applies(proof, defeated))
        if not stale:
            return
        names = {f.pred for f in self.kb.facts} | set(self.catalogue) | {r.head.pred for r in self.kb.rules}
        guard_of: dict[Key, str] = {}
        for i, key in enumerate(sorted({key for key, _ in defeated})):
            name = f"normal_{i}"
            while name in names:
                name += "_"
            guard_of[key] = name
        guarded: dict[Rule, Rewrite] = {}
        kept: list[Rewrite] = []
        facts = list(self.kb.facts)
        for rw in rewrites:
            if rw.key not in guard_of:
                kept.append(rw)
                continue
            rule = rw.rule()
            guarded[Rule(rule.head, (*rule.body, Atom(guard_of[rw.key], rule.head.args)))] = rw
            facts += [Atom(guard_of[rw.key], row) for row in sorted(out.applied.get(rw.key, ()))
                      if (rw.key, row) not in defeated]
        program = Program(facts=tuple(facts), rules=(*self.kb.rules, *guarded), rewrites=tuple(kept), proofs=True)
        solution = self.engine.solve(program, query, self.budget)
        out.exception_work += solution.work
        if not solution.finished or solution.proofs is None:
            msg = f"re-solving {query} around its defeated applications ended {solution.status}: {solution.detail}"
            raise RuntimeError(msg)
        for row in stale:
            if row not in solution.proofs:
                msg = f"re-solving {query} around its defeated applications lost the answer {row}, which its lineage keeps"
                raise RuntimeError(msg)
            out.proofs[row] = _unguard(solution.proofs[row], guarded)

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

    def _worklist(self, start: str, disc: Discoverer) -> tuple[list[Rewrite], set[str]]:
        """Rewrites reachable from `start`, and the predicates expanded to reach them: each expanded predicate's
        premises and rule bodies join the worklist."""
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
        return list(found.values()), seen

    def _eager(self, strategy: Strategy, query: Atom, disc: Discoverer, out: Outcome) -> None:
        rewrites, goals = self._worklist(query.pred, disc)
        out.admitted = {rw.key for rw in rewrites}
        out.defeated = _defeated(disc, goals)
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
        out.defeated = _defeated(disc, self.catalogue)
        self._solve(self._program(found.values()), query, out)

    def _lazy(self, strategy: Strategy, query: Atom, disc: Discoverer, out: Outcome) -> None:
        admitted: set[Key] = set()
        goals: set[str] = set()

        def expand(p: str) -> Iterator[Rewrite]:
            if p not in self.catalogue:
                return
            goals.add(p)
            for rw in disc.rewrites(p):
                admitted.add(rw.key)
                yield rw

        self._solve(self._program(tabled=strategy.tabled, depth=strategy.depth, depth_counts=strategy.depth_counts),
                    query, out, expand)
        out.admitted = admitted
        out.defeated = _defeated(disc, goals)

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
        out.defeated = _defeated(disc, expanded)


def _defeated(disc: Discoverer, goals: Iterable[str]) -> dict[Key, tuple[Reading, ...]]:
    """The rewrites concluding `goals` that the claim check withdrew in `disc`, each with the readings defeating it."""
    return {key: readings for key, (_, readings) in disc.defeats(goals).items()}


def _stopped(found: Lineages, out: Outcome) -> bool:
    """Whether the work budget stopped the lineage `found`, which leaves the run at work_limit with no answers, as an
    engine's stop would."""
    if found.finished:
        return False
    out.status, out.detail = "work_limit", f"lineage stopped after {found.work:,} work"
    out.answers, out.beliefs = set(), None
    return True


def answer_beliefs(found: Lineages, rows: Iterable[Row], opinions: dict[Variable, Opinion]) -> dict[Row, Opinion]:
    """Each of `rows`' opinion from its lineage in `found`, each variable at its opinion in `opinions`: the Beta
    matched to the exact mean and variance of the answer's probability, at the probability its lineage has at the
    variables' base rates (Rewriter._weigh says why)."""
    mean = {var: o.projection for var, o in opinions.items()}
    square = {var: o.projection**2 + o.variance for var, o in opinions.items()}
    base = {var: o.base_rate for var, o in opinions.items()}
    beliefs = {}
    for row in rows:
        lin = found.answers[row]
        first, second = moments(lin, mean, square)
        variance = second - first * first
        # E[f^2] - E[f]^2 cancels to a few ulps below zero where the answer is all but certain.
        if -1e-12 < variance < 0.0:
            variance = 0.0
        beliefs[row] = Opinion.from_moments(first, variance, probability(lin, base))
    return beliefs


def _applies(proof: Proof, applications: set[tuple[Key, Row]]) -> bool:
    """Whether some step of `proof` applies a rewrite to a row among `applications`."""
    if isinstance(proof.by, Rewrite) and (proof.by.key, proof.atom.args) in applications:
        return True
    return any(_applies(child, applications) for child in proof.children)


def _unguard(proof: Proof, guarded: dict[Rule, Rewrite]) -> Proof:
    """`proof` with each step of a guarded rule read back as its rewrite's step, the guard's fact dropped."""
    children = tuple(_unguard(child, guarded) for child in proof.children)
    if isinstance(proof.by, Rule) and proof.by in guarded:
        return Proof(proof.atom, guarded[proof.by], children[:-1])
    return Proof(proof.atom, proof.by, children)


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
