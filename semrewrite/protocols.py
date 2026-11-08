"""Purpose: the protocols that let the semantic rewriter run on any logic engine, with
any retriever, judge and admission policy: their operations, the data they
exchange (programs, budgets, solutions, proofs) and the capabilities an engine
declares. DESIGN.md states the contracts these types carry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from semrewrite.model import Atom, Predicate, Proposal, Rewrite, Rule

Status = Literal["ok", "work_limit", "time_limit", "error"]
"""How a run ended: finished, or stopped by the work budget, the time budget or an engine error."""

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
    depth: honours Program.depth > 0 with the derivation-height semantics of DESIGN.md.
    lazy: calls the expander during evaluation instead of needing every rewrite up front.
    untabled: can run Program.tabled=False as plain top-down resolution.
    reached: reports the predicates evaluation called, which the rounds strategy needs.
    proofs: returns one derivation per answer.
    """

    tabling: bool = False
    depth: bool = False
    lazy: bool = False
    untabled: bool = False
    reached: bool = False
    proofs: bool = False


@dataclass(frozen=True, slots=True)
class Program:
    """What an engine evaluates: ground facts, user rules and the rewrites known before evaluation.

    depth 0 means no bound; depth n admits an answer only when it has a
    derivation using at most n rule or rewrite steps along every path from the
    query down to a fact.
    """

    facts: tuple[Atom, ...]
    rules: tuple[Rule, ...] = ()
    rewrites: tuple[Rewrite, ...] = ()
    tabled: bool = True
    depth: int = 0

    def __post_init__(self) -> None:
        for fact in self.facts:
            if not fact.ground():
                msg = f"fact {fact} has a variable"
                raise ValueError(msg)
        if self.depth < 0:
            msg = "depth counts steps, so it can't be negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Proof:
    """One derivation of a ground atom: a fact (by is None), or a rule or rewrite applied to sub-proofs."""

    atom: Atom
    by: Rule | Rewrite | None = None
    children: tuple[Proof, ...] = ()

    def steps(self) -> int:
        """The derivation height: rule and rewrite steps on the longest path down to a fact."""
        return 0 if self.by is None else 1 + max((c.steps() for c in self.children), default=0)

    def lines(self, indent: int = 0) -> list[str]:
        """The proof as indented text, the atom first and each premise below it."""
        reason = "fact" if self.by is None else f"by {self.by}"
        out = [f"{'  ' * indent}{self.atom}    [{reason}]"]
        for c in self.children:
            out += c.lines(indent + 1)
        return out


@dataclass
class Solution:
    """What an engine returns for one query.

    answers holds one tuple per answer, the query's arguments with its
    variables filled; derivations counts answers with repeats where the engine
    can tell; reached and proofs are None when the engine lacks the capability.
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
    """

    name: str
    capabilities: Capabilities

    def solve(self, program: Program, query: Atom, budget: Budget, expand: Expander | None = None) -> Solution: ...


@runtime_checkable
class Retriever(Protocol):
    """Proposes candidate premises for a goal predicate: at most k names of the goal's arity, most similar first."""

    def top_k(self, goal: Predicate, among: Mapping[str, Predicate], k: int) -> list[tuple[str, float]]: ...


@runtime_checkable
class Judge(Protocol):
    """Scores proposals: per proposal, in order, the probability that its premise sentence entails its conclusion sentence.

    `threshold` is the calibrated score at which the default admission policy
    admits; `usage` counts the model work spent, under units the judge names.
    """

    name: str
    threshold: float
    usage: dict[str, int]

    def score(self, proposals: Sequence[Proposal]) -> list[float]: ...


@runtime_checkable
class Admission(Protocol):
    """Decides which scored proposals become rewrites.

    When `needs_converse` is true the discoverer also scores each proposal's
    converse and passes those scores in `converse`, in the same order.
    """

    needs_converse: bool

    def admit(self, proposals: Sequence[Proposal], scores: Sequence[float],
              converse: Sequence[float] | None, threshold: float) -> list[bool]: ...
