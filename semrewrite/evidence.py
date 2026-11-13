"""Purpose: the knowledge base as a second source on rewrites. A rewrite c ⇐ p claims that what p states, read in c's
argument order, c holds; the knowledge base gives evidence for that claim by how many of p's statements c's extension
holds and how many it does not. c's extension is its own statements, which are certain, and what the other admitted
rewrites concluding c derive in one step from their premises' statements, each row as probable as some rewrite deriving
it is valid.

On Alphabet's question, of the 89 rows `owns(X, Y) ⇐ parent_organization_or_unit_p749(Y, X)` derives, the 23 that
`owned_by_p127` also states count for it, through the admitted `owns(X, Y) ⇐ owned_by_p127(Y, X)`, and the other 66
against it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from semrewrite.model import Atom, Key, Rewrite

Row = tuple[str, ...]


def statement_evidence(rows: Iterable[Row], extension: Mapping[Row, float]) -> tuple[float, float]:
    """Positive and negative evidence that `rows` are rows of a relation whose extension holds each row with the
    probability `extension` gives it, 0 for a row it lacks: a row counts positive by that probability and negative by
    the rest. This is NAL's extensional evidence for an inheritance S → P, the instances of S inside P positive and
    those outside it negative (Wang, Formalization of Evidence, JAGI 2009, sec. 4), with graded membership.

    >>> statement_evidence([("a", "b"), ("c", "d")], {("a", "b"): 1.0, ("c", "d"): 0.25})
    (1.25, 0.75)
    """
    positive = negative = 0.0
    for row in rows:
        p = extension.get(row, 0.0)
        positive += p
        negative += 1.0 - p
    return positive, negative


def derived(rewrite: Rewrite, statements: Iterable[Row]) -> set[Row]:
    """The conclusion rows `rewrite` derives in one step from its premise's `statements`, through the rule the
    engines run, so a mapping and a repeated variable read as they do there."""
    rule = rewrite.rule()
    (premise,) = rule.body
    out = set()
    for row in statements:
        binding = premise.match(row)
        if binding is not None:
            out.add(tuple(binding[v] for v in rule.head.args))
    return out


# Time: building the index costs one pass over the admitted rewrites' premise statements, s of them; each rewrite's
# evidence then visits its own derived rows, and each row multiplies over the rewrites deriving it, so the whole costs
# s + sum over rows of (rewrites deriving the row)^2.
def rewrite_evidence(facts: Iterable[Atom], rewrites: Sequence[Rewrite], valid: Mapping[Key, float]
                     ) -> dict[Key, tuple[float, float]]:
    """Each rewrite's positive and negative evidence from the knowledge base's `facts`: its premise's statements
    against its conclusion's extension without itself, the conclusion's statements and the rows the other rewrites
    concluding it derive, each row held with the probability that some rewrite deriving it is valid, the rewrites
    independent and each valid with probability `valid[key]`. A rewrite whose premise states nothing has no evidence."""
    stated: dict[str, set[Row]] = defaultdict(set)
    for f in facts:
        stated[f.pred].add(f.args)
    rows = {rw.key: derived(rw, stated.get(rw.premise, ())) for rw in rewrites}
    # For each conclusion and row, the rewrites deriving it.
    makers: dict[str, dict[Row, list[Key]]] = defaultdict(lambda: defaultdict(list))
    for rw in rewrites:
        for row in rows[rw.key]:
            makers[rw.conclusion][row].append(rw.key)
    out = {}
    for rw in rewrites:
        own = stated.get(rw.conclusion, set())
        extension: dict[Row, float] = {}
        for row in rows[rw.key]:
            if row in own:
                extension[row] = 1.0
                continue
            others = [k for k in makers[rw.conclusion][row] if k != rw.key]
            extension[row] = 1.0 - math.prod(1.0 - valid[k] for k in others)
        out[rw.key] = statement_evidence(rows[rw.key], extension)
    return out
