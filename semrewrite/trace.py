"""Purpose: what a query does as it runs, as events the rewriter and discoverer emit, and TextTrace, which prints
them as they happen. `semrewrite ask --trace` passes a TextTrace to the rewriter.

The engine starts (Solving), evaluation reaches a predicate and the discoverer
expands it, retrieving candidates and asking the judge about each (Expanded),
later asks for the same predicate's rewrites come from the memo (Remembered),
and the engine stops (Solved). With lazy discovery the expansions happen
between Solving and Solved, inside evaluation; the other timings expand before
the engine starts.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO

from semrewrite.model import Atom, Predicate, Proposal, Rewrite
from semrewrite.protocols import Program, Solution


@dataclass(frozen=True, slots=True)
class Solving:
    """The engine starts evaluating `query` over `program`."""

    engine: str
    query: Atom
    program: Program


@dataclass(frozen=True, slots=True)
class Solved:
    """The engine stopped, with `solution`."""

    engine: str
    solution: Solution


@dataclass(frozen=True, slots=True)
class Expanded:
    """One expansion of `goal`: the retriever ranked `among` predicates of its arity and kept `candidates`, best first,
    with their retrieval scores; the judge scored `proposals`; `admitted` are the rewrites the admission policy took.

    `retriever` is None when there is none and every predicate of the goal's arity is a candidate.
    """

    goal: Predicate
    retriever: str | None
    among: int
    candidates: tuple[tuple[str, float], ...]
    judge: str
    threshold: float
    proposals: tuple[Proposal, ...]
    scores: tuple[float, ...]
    admitted: tuple[Rewrite, ...]


@dataclass(frozen=True, slots=True)
class Remembered:
    """An ask for the rewrites concluding `goal`, answered from the discoverer's memo without asking the judge."""

    goal: str


Event = Solving | Solved | Expanded | Remembered
Tracer = Callable[[Event], None]
"""Receives each event as it happens."""


def plural(n: int, word: str) -> str:
    """`n` and `word`, with an s unless n is 1: "1 answer", "2,296 answers"."""
    return f"{n:,} {word}{'' if n == 1 else 's'}"


class TextTrace:
    """Prints each event as a line of plain text as it arrives, to `stream`.

    An expansion prints what the retriever kept, every proposal the judge
    admitted beside the same premise under its other argument mappings, which
    shows whether the judge read the direction, and the highest score among
    the rest. Asks the memo answered, and repeated expansions without a memo,
    are counted and summarised when the engine stops, since evaluation may ask
    thousands of times.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream
        self._facts: Counter[str] = Counter()
        self._solving = False
        self._expanded: set[str] = set()
        self._remembered: Counter[str] = Counter()
        self._again: Counter[str] = Counter()

    def _print(self, line: str = "") -> None:
        # The stream is looked up at each line, so a TextTrace made before stderr is redirected writes where it now goes.
        print(line, file=self.stream if self.stream is not None else sys.stderr, flush=True)

    def __call__(self, event: Event) -> None:
        match event:
            case Solving(engine, query, program):
                self._facts = Counter(f.pred for f in program.facts)
                self._solving = True
                self._expanded.clear()
                self._remembered.clear()
                self._again.clear()
                known = plural(len(program.rewrites), "rewrite") if program.rewrites else "no rewrites yet"
                self._print(f"{engine} starts on {query}, over {plural(len(program.facts), 'fact')} and "
                            f"{plural(len(program.rules), 'rule')}, with {known}")
            case Expanded():
                self._expanded_event(event)
            case Remembered(goal):
                self._remembered[goal] += 1
            case Solved(engine, solution):
                self._solving = False
                self._print(f"{engine} finished: {solution.status}, {plural(len(solution.answers), 'answer')}, "
                            f"{solution.work:,} work")
                if self._remembered:
                    asks = ", ".join(f"{plural(n, 'more ask')} for {goal}" for goal, n in self._remembered.most_common())
                    self._print(f"  the memo answered {asks}, without asking the judge again")
                if self._again:
                    again = ", ".join(f"{goal} {plural(n, 'more time')}" for goal, n in self._again.most_common())
                    self._print(f"  with no memo, evaluation expanded {again}")

    def _expanded_event(self, e: Expanded) -> None:
        name = e.goal.name
        if name in self._expanded:
            self._again[name] += 1
            return
        self._expanded.add(name)
        stated = plural(self._facts[name], "fact") if self._facts[name] else "no fact"
        if self._solving:
            self._print(f"evaluation reached {name}, which {stated} states, and asks for the rewrites concluding it")
        else:
            self._print(f"before evaluation, discovery expands {name}, which {stated} states")
        letters = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")[: e.goal.arity]
        sentence = e.goal.sentence(*letters)
        if e.retriever is None or len(e.candidates) == e.among:
            self._print(f"  every one of the {plural(e.among, 'predicate')} of the same arity is a candidate")
        else:
            self._print(f"  the {e.retriever} retriever ranked the {plural(e.among, 'predicate')} of the same arity "
                        f'as premises of "{sentence}" and kept the {len(e.candidates):,} it ranks highest')
        self._print(f"  {e.judge} was asked {plural(len(e.proposals), 'question')}, whether each premise sentence "
                    f'entails "{sentence}", and admits a score of {e.threshold:.3f} or more:')
        self._verdicts(e)
        if e.admitted:
            self._print("  admitted as rules:")
            for rw in e.admitted:
                self._print(f"    {rw}")

    def _verdicts(self, e: Expanded) -> None:
        """Each admitted proposal with the same premise under its other mappings, then the highest of the rest."""
        admitted = {rw.key for rw in e.admitted}
        # A rank says something only when retrieval kept fewer than all; otherwise every candidate is there by name.
        ranked = e.retriever is not None and len(e.candidates) < e.among
        rank = {name: i for i, (name, _) in enumerate(e.candidates, start=1)} if ranked else {}
        scored = list(zip(e.proposals, e.scores, strict=True))
        shown: list[int] = []
        # Time: O(a·p) for a admitted and p proposals; an expansion asks p = k·n! + n! - 1.
        for i in sorted((i for i, (p, _) in enumerate(scored) if p.key in admitted), key=lambda i: -scored[i][1]):
            if i in shown:
                continue
            premise = scored[i][0].premise.name
            # The admitted proposal, then its premise under every other mapping, admitted or not.
            shown += [i, *(j for j, (p, _) in enumerate(scored) if p.premise.name == premise and j != i)]
        quoted = {i: f'"{scored[i][0].sentences()[0]}"' for i in shown}
        width = max((len(q) for q in quoted.values()), default=0)
        for i in shown:
            p, s = scored[i]
            verdict = "admit" if p.key in admitted else ""
            if p.premise.name == e.goal.name:
                where = f"{p.premise.name}, the goal itself"
            else:
                where = f"{p.premise.name}, ranked {rank[p.premise.name]}" if ranked else p.premise.name
            self._print(f"    {verdict:6} {s:.3f}  {quoted[i]:{width}}  {where}")
        rest = [i for i in range(len(scored)) if i not in shown]
        if rest:
            p, s = scored[max(rest, key=lambda i: scored[i][1])]
            count = f"{len(rest):,} more rejected" if shown else f"all {len(rest):,} rejected"
            self._print(f'    {count}, the highest at {s:.3f}: "{p.sentences()[0]}" ({p.premise.name})')
