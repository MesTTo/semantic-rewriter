"""Purpose: the knowledge base as a second source on rewrites. A rewrite c ⇐ p claims that what p states, read in c's
argument order, c holds; where the knowledge base states c, it gives evidence for that claim by how many of p's
statements, so read, it also states as c's and how many it does not.

A knowledge base stating `parent(alice, tom)`, `mum(alice, tom)` and `mum(dave, eve)` gives `parent(X, Y) ⇐ mum(X, Y)`
one statement for and one against. One stating no `parent` row gives it none, as on every Wikidata question, whose
predicates are the question's own words.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping

from semrewrite.model import Atom, Key, Rewrite

Row = tuple[str, ...]


def statement_evidence(rows: Iterable[Row], extension: Collection[Row]) -> tuple[int, int]:
    """Positive and negative evidence that `rows` are rows of the relation whose extension is `extension`: the rows
    it holds count positive and the rest negative. This is NAL's extensional evidence for an inheritance S → P, the
    instances of S inside P positive and those outside it negative (Wang, Formalization of Evidence, JAGI 2009,
    sec. 4).

    >>> statement_evidence([("a", "b"), ("c", "d")], {("a", "b")})
    (1, 1)
    """
    positive = negative = 0
    for row in rows:
        if row in extension:
            positive += 1
        else:
            negative += 1
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


def statements(facts: Iterable[Atom]) -> dict[str, set[Row]]:
    """Each predicate's rows among `facts`."""
    out: dict[str, set[Row]] = defaultdict(set)
    for f in facts:
        out[f.pred].add(f.args)
    return dict(out)


# Time: one pass over the premise's statements for each rewrite whose conclusion is stated.
def rewrite_evidence(stated: Mapping[str, Collection[Row]], rewrites: Iterable[Rewrite]
                     ) -> dict[Key, tuple[int, int]]:
    """Each rewrite's positive and negative evidence from the knowledge base's statements, `stated` by predicate: its
    premise's statements, read as its conclusion's rows, against the conclusion's own statements. That is the
    setting each judge's data trust was fitted in, Wikidata property pairs whose conclusion the knowledge base states
    (experiments/judge_eval.py), and a rewrite whose conclusion it never states gets no evidence. What the other
    admitted rewrites derive does not count: they cover only part of the relation, so a row they miss is no evidence
    against a rewrite, and a valid rewrite deriving much that they miss would read as wrong."""
    out = {}
    for rw in rewrites:
        own = stated.get(rw.conclusion, ())
        out[rw.key] = statement_evidence(derived(rw, stated.get(rw.premise, ())), own) if own else (0, 0)
    return out
