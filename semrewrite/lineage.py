"""Purpose: an answer's lineage over the rewrites it needs, and the moments of its probability: the minimal sets of
rewrite keys that some derivation of each answer uses (PosBool provenance; Green, Karvounarakis and Tannen 2007), by
semi-naive evaluation of the program's magic-sets rewrite for the query, and the probability that every rewrite of
some set is valid when each is valid independently.

A rewrite's validity is one unknown shared by every application of it, so an
answer derived through "mother" at ten generations needs that one rewrite, not
ten copies of it: the set {mother} is its lineage, and its probability is the
rewrite's. Evaluation keeps at most k sets per row, the most probable: dropping
a set from a disjunction only lowers its probability, so a belief computed from
a bounded lineage is a lower bound, and the bound is what keeps a lineage over
many chained rewrites in memory (an unbounded run over 184 Wikidata rewrites
grew past 11 GiB).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache

from semrewrite.model import Atom, Key, is_variable
from semrewrite.protocols import Program

Lineage = frozenset[frozenset[Key]]
"""An antichain of minimal rewrite-key sets: the answer holds when every rewrite of some set is valid."""

TRUE: Lineage = frozenset({frozenset()})
FALSE: Lineage = frozenset()

Row = tuple[str, ...]


def minimise(sets: Iterable[frozenset[Key]]) -> Lineage:
    """The minimal sets among `sets`, by absorption: x + xy = x."""
    kept: list[frozenset[Key]] = []
    # Time: O(m^2) subset tests for m sets, smallest first.
    for s in sorted(set(sets), key=lambda s: (len(s), sorted(s))):
        if not any(k <= s for k in kept):
            kept.append(s)
    return frozenset(kept)


@dataclass
class Lineages:
    """Each answer row's lineage, and the work the evaluation spent counting candidate rows, as the engines count it."""

    answers: dict[Row, Lineage] = field(default_factory=dict)
    work: int = 0
    rounds: int = 0
    finished: bool = True


def _magic(clauses: list[tuple[Atom, tuple[Atom, ...], Key | None]], query: Atom, sep: str):
    """The clauses rewritten for the query's bound arguments by the magic-sets transformation with left-to-right
    sideways passing (Beeri and Ramakrishnan, J. Logic Programming 10, 1991), so bottom-up evaluation derives only
    what the query reaches, as the engines' top-down evaluation does. Each clause comes back as (head, body, rewrite
    key or None, whether it derives a magic filter); a filter's rows carry no lineage."""
    derived = {h.pred for h, _, _ in clauses}

    def adorn(atom: Atom, bound: set[str]) -> str:
        return "".join("b" if not is_variable(t) or t in bound else "f" for t in atom.args)

    def magic_atom(atom: Atom, ad: str) -> Atom:
        return Atom(f"magic{sep}{atom.pred}{sep}{ad}", tuple(t for t, a in zip(atom.args, ad, strict=True) if a == "b"))

    qad = adorn(query, set())
    out = []
    seen, queue = {(query.pred, qad)}, deque([(query.pred, qad)])
    while queue:
        pred, ad = queue.popleft()
        for head, body, key in clauses:
            if head.pred != pred:
                continue
            bound = {t for t, a in zip(head.args, ad, strict=True) if a == "b" and is_variable(t)}
            prefix = [magic_atom(head, ad)]
            for b in body:
                if b.pred in derived:
                    bad = adorn(b, bound)
                    out.append((magic_atom(b, bad), tuple(prefix), None, True))
                    if (b.pred, bad) not in seen:
                        seen.add((b.pred, bad))
                        queue.append((b.pred, bad))
                    prefix.append(Atom(f"{b.pred}{sep}{bad}", b.args))
                else:
                    prefix.append(b)
                bound |= b.variables()
            out.append((Atom(f"{pred}{sep}{ad}", head.args), tuple(prefix), key, False))
    seed = magic_atom(query, qad)
    return out, seed, f"{query.pred}{sep}{qad}"


# Time: per round, each clause joins its delta rows against indexed full relations; a row re-enters the delta only when
# its lineage gains a minimal set, and each lineage holds at most k sets of at most r keys, r the rewrites.
def lineages(program: Program, query: Atom, weights: Mapping[Key, float], k: int, work: int) -> Lineages:
    """Each answer of `query` over the program's facts, rules and rewrites, with its lineage: at most `k` minimal
    rewrite sets, the most probable by `weights`, a rewrite's probability of being valid. Evaluation stops unfinished
    once it has examined `work` candidate rows."""
    if k < 1:
        msg = f"a lineage keeps at least one set per row, not {k}"
        raise ValueError(msg)
    clauses: list[tuple[Atom, tuple[Atom, ...], Key | None]] = [(r.head, r.body, None) for r in program.rules]
    for rw in program.rewrites:
        rule = rw.rule()
        clauses.append((rule.head, rule.body, rw.key))
    # Every name the rewrite makes holds `sep`, which no program predicate holds, so none can take a program's name.
    names = {f.pred for f in program.facts} | {h.pred for h, _, _ in clauses} | {a.pred for _, b, _ in clauses for a in b}
    sep = "__"
    while any(sep in name for name in names):
        sep += "x"
    stated: dict[str, dict[Row, Lineage]] = defaultdict(dict)
    for f in program.facts:
        stated[f.pred][f.args] = TRUE
    # A predicate both stated and concluded, and the query's own, reads its statements through a copy, so the
    # transformation sees it as derived and the query's answers include what is stated outright.
    for pred in ({h.pred for h, _, _ in clauses} | {query.pred}) & set(stated):
        args = tuple(f"V{i}" for i in range(len(next(iter(stated[pred])))))
        clauses.append((Atom(pred, args), (Atom(f"{pred}{sep}stated", args),), None))
        stated[f"{pred}{sep}stated"] = stated.pop(pred)
    rewritten, seed, target = _magic(clauses, query, sep)

    def weight(s: frozenset[Key]) -> float:
        return math.prod(weights[key] for key in s)

    def bounded(sets: Iterable[frozenset[Key]]) -> Lineage:
        kept = minimise(sets)
        if len(kept) <= k:
            return kept
        return frozenset(sorted(kept, key=lambda s: (-weight(s), len(s), sorted(s)))[:k])

    result = Lineages()
    used = {a.pred for _, body, _, _ in rewritten for a in body}
    rel: dict[str, dict[Row, Lineage]] = defaultdict(dict)
    for pred in used & set(stated):
        rel[pred] = stated[pred]
    rel[seed.pred][seed.args] = TRUE
    delta = {pred: dict(rows) for pred, rows in rel.items()}
    while delta:
        result.rounds += 1
        index: dict[tuple[str, tuple[int, ...]], dict[Row, list[tuple[Row, Lineage]]]] = {}

        def lookup(pred: str, positions: tuple[int, ...], key: Row,
                   index: dict[tuple[str, tuple[int, ...]], dict[Row, list[tuple[Row, Lineage]]]] = index,
                   ) -> list[tuple[Row, Lineage]]:
            if (pred, positions) not in index:
                table: dict[Row, list[tuple[Row, Lineage]]] = defaultdict(list)
                for row, lin in rel[pred].items():
                    table[tuple(row[i] for i in positions)].append((row, lin))
                index[(pred, positions)] = table
            return index[(pred, positions)].get(key, [])

        fresh: dict[str, dict[Row, Lineage]] = defaultdict(dict)
        for head, body, rewrite, is_filter in rewritten:
            for i, atom in enumerate(body):
                for row, lin in delta.get(atom.pred, {}).items():
                    binding = atom.match(row)
                    if binding is None:
                        continue
                    partial = [(binding, lin)]
                    for j, other in enumerate(body):
                        if j == i:
                            continue
                        nxt = []
                        for b, acc in partial:
                            positions = tuple(n for n, t in enumerate(other.args) if not is_variable(t) or t in b)
                            probe = tuple(b.get(other.args[n], other.args[n]) for n in positions)
                            for r2, l2 in lookup(other.pred, positions, probe):
                                result.work += 1
                                if result.work > work:
                                    result.finished = False
                                    return result
                                b2 = other.match(r2)
                                if b2 is None or any(b.get(v, val) != val for v, val in b2.items()):
                                    continue
                                nxt.append(({**b, **b2}, bounded(x | y for x in acc for y in l2)))
                        partial = nxt
                    for b, acc in partial:
                        if is_filter:
                            acc = TRUE
                        elif rewrite is not None:
                            acc = bounded(s | {rewrite} for s in acc)
                        out = tuple(b.get(t, t) for t in head.args)
                        old = rel[head.pred].get(out, FALSE)
                        new = bounded(old | acc)
                        if new != old:
                            rel[head.pred][out] = new
                            fresh[head.pred][out] = new
        delta = {pred: rows for pred, rows in fresh.items() if rows}
    result.answers = {row: lin for row, lin in rel[target].items() if query.match(row) is not None}
    return result


def _split(lin: Lineage, key: Key) -> tuple[Lineage, Lineage]:
    """The lineage given the rewrite `key` valid, and given it invalid: Shannon expansion on one variable."""
    return minimise(s - {key} for s in lin), frozenset(s for s in lin if key not in s)


def _pivot(*lins: Lineage) -> Key:
    """The rewrite occurring in the most sets, which the expansion splits on first."""
    counts: dict[Key, int] = defaultdict(int)
    for lin in lins:
        for s in lin:
            for key in s:
                counts[key] += 1
    return max(sorted(counts), key=lambda key: counts[key])


# Time: exponential in the distinct rewrites of a lineage in the worst case, memoised on each sub-lineage; with k sets
# of a few rewrites each it is small.
def probability(lin: Lineage, p: Mapping[Key, float]) -> float:
    """The probability that every rewrite of some set in `lin` is valid, each valid independently with p[key]."""

    @cache
    def f(sub: Lineage) -> float:
        if not sub:
            return 0.0
        if frozenset() in sub:
            return 1.0
        key = _pivot(sub)
        valid, invalid = _split(sub, key)
        return p[key] * f(valid) + (1 - p[key]) * f(invalid)

    return f(lin)


def moments(lin: Lineage, mean: Mapping[Key, float], square: Mapping[Key, float]) -> tuple[float, float]:
    """The exact mean and second moment of the answer's probability when each rewrite's probability of being valid is
    an independent random variable with mean[key] and E[p^2] = square[key], such as a Beta's: the answer's probability
    is multilinear in the rewrites' probabilities, so its mean is its value at their means, and its second moment
    follows the same expansion over pairs (the route of exact variance for weighted model counting, AAAI 2026)."""

    @cache
    def m1(sub: Lineage) -> float:
        return probability(sub, mean)

    @cache
    def m2(x: Lineage, y: Lineage) -> float:
        if not x or not y:
            return 0.0
        if frozenset() in x:
            return m1(y)
        if frozenset() in y:
            return m1(x)
        key = _pivot(x, y)
        x1, x0 = _split(x, key)
        y1, y0 = _split(y, key)
        both, one = square[key], mean[key] - square[key]
        neither = 1 - 2 * mean[key] + square[key]
        return both * m2(x1, y1) + one * (m2(x1, y0) + m2(x0, y1)) + neither * m2(x0, y0)

    return m1(lin), m2(lin, lin)
