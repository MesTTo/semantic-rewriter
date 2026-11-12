"""Purpose: the protocols that let the semantic rewriter run on any logic engine, with
any retriever, judge and admission policy: their operations, the data they
exchange (programs, budgets, solutions, proofs) and the capabilities an engine
declares. DESIGN.md states the contracts these types carry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, get_args, runtime_checkable

from semrewrite.model import Atom, Predicate, Proposal, Rewrite, Rule

Status = Literal["ok", "work_limit", "time_limit", "error"]
"""How a run ended: finished, or stopped by the work budget, the time budget or an engine error."""

DepthCounts = Literal["steps", "rewrites"]
"""What a depth bound counts: every rule and rewrite step, or rewrite steps alone, leaving the user's rules unbounded."""

Expander = Callable[[str], Iterable[Rewrite]]
"""Given a predicate name, the rewrites that conclude it; an engine with the lazy capability calls it while it evaluates."""


@dataclass(frozen=True, slots=True)
class Budget:
    """The most a run may spend: engine work in the engine's own unit, and wall-clock seconds.

    A Prolog engine counts inferences, the Python engine the candidate rows
    its joins examine; a run that exceeds either limit ends with no answers.
    `seconds` may be math.inf, for no time limit; work is always finite, so
    every run is bounded by it.
    """

    work: int = 2_000_000
    seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What an engine supports; the rewriter refuses a strategy that needs a capability the engine lacks.

    tabling: with Program.tabled, every program terminates with its exact fixpoint answers.
    depth: honours Program.depth > 0 counting steps, the derivation-height semantics of DESIGN.md.
    rewrite_depth: honours Program.depth > 0 counting rewrite steps alone.
    lazy: calls the expander during evaluation instead of needing every rewrite up front.
    untabled: can run Program.tabled=False as plain top-down resolution.
    reached: with no depth bound, reports every predicate evaluation called, and possibly more of the program's,
        which the rounds strategy expands; under a depth bound it may report fewer.
    proofs: with Program.proofs, returns one derivation per answer.
    """

    tabling: bool = False
    depth: bool = False
    rewrite_depth: bool = False
    lazy: bool = False
    untabled: bool = False
    reached: bool = False
    proofs: bool = False


@dataclass(frozen=True, slots=True)
class Program:
    """What an engine evaluates: ground facts, user rules and the rewrites known before evaluation.

    depth 0 means no bound; depth n admits an answer only when it has a
    derivation whose counted steps, along every path from the query down to a
    fact, number at most n. With depth_counts "steps" every rule and rewrite
    step counts; with "rewrites" only rewrite steps do, so the user's own rules
    recurse as deep as their facts allow while a false rewrite still spreads
    through at most n rewrites.

    proofs asks the engine for one derivation of each answer in Solution.proofs,
    which an engine may spend more work on.
    """

    facts: tuple[Atom, ...]
    rules: tuple[Rule, ...] = ()
    rewrites: tuple[Rewrite, ...] = ()
    tabled: bool = True
    depth: int = 0
    depth_counts: DepthCounts = "steps"
    proofs: bool = False

    def __post_init__(self) -> None:
        for fact in self.facts:
            if not fact.ground():
                msg = f"fact {fact} has a variable"
                raise ValueError(msg)
        if self.depth < 0:
            msg = "depth counts steps, so it can't be negative"
            raise ValueError(msg)
        if self.depth_counts not in get_args(DepthCounts):
            msg = f"depth_counts is {self.depth_counts!r}; a depth bound counts one of {get_args(DepthCounts)}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Proof:
    """One derivation of a ground atom: a fact (by is None), or a rule or rewrite applied to sub-proofs."""

    atom: Atom
    by: Rule | Rewrite | None = None
    children: tuple[Proof, ...] = ()

    def steps(self, counts: DepthCounts = "steps") -> int:
        """The derivation height in counted steps, on the path down to a fact that has the most: every rule and
        rewrite step, or with counts "rewrites" the rewrite steps alone."""
        if self.by is None:
            return 0
        own = 1 if counts == "steps" or isinstance(self.by, Rewrite) else 0
        return own + max((c.steps(counts) for c in self.children), default=0)

    def lines(self, indent: int = 0) -> list[str]:
        """The proof as indented text, the atom first and each premise below it; a rewrite step shows its score."""
        if self.by is None:
            reason = "fact"
        elif isinstance(self.by, Rewrite):
            reason = f"by {self.by}, score {self.by.score:.3f}"
        else:
            reason = f"by {self.by}"
        out = [f"{'  ' * indent}{self.atom}    [{reason}]"]
        for c in self.children:
            out += c.lines(indent + 1)
        return out


@dataclass
class Solution:
    """What an engine returns for one query.

    answers holds one tuple per answer, the query's arguments with its
    variables filled; derivations counts answers with repeats where the engine
    can tell; reached is None when the engine lacks the capability, and proofs
    is None unless the program asked for them.
    """

    answers: set[tuple[str, ...]] = field(default_factory=set)
    status: Status = "ok"
    work: int = 0
    seconds: float = 0.0
    derivations: int = 0
    tables: int = 0
    reached: frozenset[str] | None = None
    proofs: dict[tuple[str, ...], Proof] | None = None
    detail: str = ""

    @property
    def finished(self) -> bool:
        return self.status == "ok"


@runtime_checkable
class Engine(Protocol):
    """A logic engine that evaluates a function-free Horn program under a budget.

    An implementation keeps the five rules of DESIGN.md: exact answers,
    one meaning of depth, expansion as a rule source, hard budgets, and any
    lowercase name accepted as data. `semrewrite.conformance` checks them.
    Members a caller only reads are read-only properties, so a class
    attribute, a frozen dataclass field or a property each implements them.
    """

    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> Capabilities: ...

    def solve(self, program: Program, query: Atom, budget: Budget, expand: Expander | None = None) -> Solution: ...


@runtime_checkable
class Retriever(Protocol):
    """Proposes candidate premises for a goal predicate: at most k names of the goal's arity, most similar first.

    `name` says what ranks them, as a trace prints it.
    """

    @property
    def name(self) -> str: ...

    def top_k(self, goal: Predicate, among: Mapping[str, Predicate], k: int) -> list[tuple[str, float]]: ...


@runtime_checkable
class Judge(Protocol):
    """Scores proposals: per proposal, in order, the probability that its premise sentence entails its conclusion sentence.

    `threshold` is the calibrated score at which the default admission policy
    admits, and calibration may set it; `usage` counts the model work spent,
    under units the judge names.
    """

    threshold: float

    @property
    def name(self) -> str: ...

    @property
    def usage(self) -> dict[str, int]: ...

    def score(self, proposals: Sequence[Proposal]) -> list[float]: ...


@runtime_checkable
class Admission(Protocol):
    """Decides which scored proposals become rewrites.

    When `needs_converse` is true the discoverer also scores each proposal's
    converse and passes those scores in `converse`, in the same order.
    """

    @property
    def needs_converse(self) -> bool: ...

    def admit(self, proposals: Sequence[Proposal], scores: Sequence[float],
              converse: Sequence[float] | None, threshold: float) -> list[bool]: ...
