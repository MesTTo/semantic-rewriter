"""Purpose: the Wikidata examples' three questions, each defined once for the builder and the evaluation: its subject,
the relation its answer follows from the subject, and the Wikidata properties that record the relation. Each property
is a root in Wikidata's own subproperty hierarchy (P1647), read with every property Wikidata files under it, so
Wikidata's record, not this file, says which statements an answer counts.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semrewrite.kb import KB

ROOT = Path(__file__).resolve().parents[2]
# Wikidata's subproperty links, as the judge evaluation fetched them; the builder and the evaluation read this copy.
SUBPROPERTIES = ROOT / "data" / "raw" / "wikidata" / "subproperty.json"


# Time: O(s) over the s subproperty links, once a process.
@cache
def _below() -> dict[str, tuple[str, ...]]:
    """Each property's direct subproperties."""
    below: dict[str, list[str]] = defaultdict(list)
    for row in json.loads(SUBPROPERTIES.read_text())["results"]["bindings"]:
        below[row["super"]["value"].rsplit("/", 1)[-1]].append(row["p"]["value"].rsplit("/", 1)[-1])
    return {p: tuple(subs) for p, subs in below.items()}


def property_id(pred: str) -> str | None:
    """The Wikidata property id ending a knowledge-base predicate's name, if one does.

    >>> property_id("child_p40"), property_id("parent_of")
    ('P40', None)
    """
    head, _, tail = pred.rpartition("_p")
    return f"P{tail}" if head and tail.isdigit() else None


@dataclass(frozen=True)
class Question:
    """A question about one Wikidata entity, and the answer Wikidata's own record gives it.

    The answer is what `relation` reaches from the subject, in one step or, when `transitive`, in any number of steps.
    Wikidata records the relation in each root property and in every property its hierarchy files under that root. A
    root (p, False) reads a statement p(a, b) as relation(a, b), and a root (p, True) as relation(b, a): the statement
    is made on the other entity, as child (P40) is stated on the parent and father (P22) on the child.
    """

    name: str
    # The subject's Wikidata id, and the name the question calls it.
    subject: tuple[str, str]
    relation: str
    roots: tuple[tuple[str, bool], ...]
    transitive: bool
    # Whether the knowledge base also holds the statements of every entity naming the subject in one of its own.
    mentions: bool = False

    # Time: O(u) over the u properties under the roots.
    def properties(self) -> dict[str, bool]:
        """Every property id recording the relation, with whether its statements are made on the other entity."""
        out: dict[str, bool] = {}
        for root, flipped in self.roots:
            stack = [root]
            while stack:
                pid = stack.pop()
                if pid not in out:
                    out[pid] = flipped
                    stack += _below().get(pid, ())
        return out

    # Time: O(n + a) for n statements and a entities reached: one pass to index, then a breadth-first search.
    def answer(self, statements: Iterable[tuple[str, str, str]], subject: str) -> set[str]:
        """What the relation reaches from `subject` over (entity, property id, entity) statements."""
        flipped = self.properties()
        step: dict[str, set[str]] = defaultdict(set)
        for a, pid, b in statements:
            if pid in flipped:
                if flipped[pid]:
                    step[b].add(a)
                else:
                    step[a].add(b)
        found: set[str] = set()
        frontier = [subject]
        while frontier:
            reached = []
            for node in frontier:
                for b in step[node] - found:
                    found.add(b)
                    reached.append(b)
            frontier = reached if self.transitive else []
        return found

    # Time: O(f + a) for f facts and a entities reached.
    def recorded(self, kb: KB) -> tuple[set[str], dict[str, bool]]:
        """Wikidata's own answer from the knowledge base's statements, and each predicate of the knowledge base it
        steps along, with whether that predicate's statements are made on the other entity."""
        flipped = self.properties()
        used: dict[str, bool] = {}
        statements = []
        for f in kb.facts:
            pid = property_id(f.pred)
            if pid in flipped and len(f.args) == 2:
                used[f.pred] = flipped[pid]
                statements.append((f.args[0], pid, f.args[1]))
        return self.answer(statements, self.subject[1]), dict(sorted(used.items()))

    def answers_of(self, rows: Iterable[tuple[str, ...]]) -> set[str]:
        """The answer entity in each row of the question's query: the argument that isn't the subject."""
        name = self.subject[1]
        return {row[0] if row[1] == name else row[1] for row in rows}


QUESTIONS = {
    q.name: q
    for q in (
        # Who descends from Queen Victoria: whoever is her or a descendant's child (P40, stated on the parent), or
        # has her or a descendant as parent (P8810 with father and mother under it, stated on the child).
        Question("victoria", ("Q9439", "queen_victoria"), "parent_of", (("P40", False), ("P8810", True)), transitive=True),
        # What Leonardo da Vinci created: whatever names him creator (P170) or anything Wikidata files under creator,
        # such as author or architect, stated on the work. The knowledge base also holds everything else naming him:
        # what is named after him, about him or depicting him.
        Question("leonardo", ("Q762", "leonardo_da_vinci"), "created", (("P170", True),), transitive=False,
                 mentions=True),
        # What Alphabet owns, directly or through what it owns: owner of (P1830, stated on the owner) and owned by
        # (P127, stated on what is owned), each with what Wikidata files under it, such as parent organization.
        Question("alphabet", ("Q20800404", "alphabet_inc"), "owns", (("P1830", False), ("P127", True)), transitive=True),
    )
}
