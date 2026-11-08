"""Purpose: the reference evaluator: naive bottom-up evaluation of function-free Horn
programs of any arity, optionally cut at a height in counted steps. Gold answers,
gold entailment and the engine conformance kit all read answers from it, so it
stays as simple as correctness allows and shares no code with any engine.

Rules come in two kinds: counted rules, whose steps a depth bound counts, and free
rules, whose steps it doesn't. Evaluation runs in layers. A layer first applies the
free rules until they add nothing, then applies every counted rule once to what it
holds, which starts the next layer. Layer k then holds exactly the facts with a
derivation using at most k counted steps on every path down to a fact: a free step
keeps its premises' layer, and a counted step needs premises one layer down. A depth
bound n stops after layer n is closed; without one, evaluation stops when a layer's
counted rules add nothing. With every rule counted, a layer is one round applying
every rule, and layer k holds the facts of derivation height at most k.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping

from semrewrite.model import Atom, Rule, is_variable

Model = dict[str, set[tuple[str, ...]]]
"""Ground facts by predicate: pred -> set of argument tuples."""


def _bindings(body: tuple[Atom, ...], model: Mapping[str, set[tuple[str, ...]]]) -> Iterator[dict[str, str]]:
    """Every binding of the body's variables under which each body atom is in `model`."""

    def walk(i: int, binding: dict[str, str]) -> Iterator[dict[str, str]]:
        if i == len(body):
            yield binding
            return
        atom = body[i]
        for values in model.get(atom.pred, ()):
            if len(values) != atom.arity:
                continue
            extended = dict(binding)
            ok = True
            for term, value in zip(atom.args, values, strict=True):
                if is_variable(term):
                    if extended.setdefault(term, value) != value:
                        ok = False
                        break
                elif term != value:
                    ok = False
                    break
            if ok:
                yield from walk(i + 1, extended)

    yield from walk(0, {})


# Time: rounds x sum over the round's rules of the bindings their bodies admit against the whole
# model; rounds <= the number of distinct derivable facts plus the layers, and layers <= the
# depth bound when there is one. Space: the model.
def least_model(facts: Iterable[Atom], rules: Iterable[Rule], depth: int = 0, free: Iterable[Rule] = ()) -> Model:
    """The facts derivable from `facts` under `rules` and `free`.

    With depth > 0, only the facts with a derivation using at most `depth` steps by
    `rules` on every path down to a fact; steps by `free` rules don't count.
    """
    rules, free = tuple(rules), tuple(free)
    model: Model = defaultdict(set)
    for fact in facts:
        model[fact.pred].add(fact.args)
    layers = 0
    while True:
        while _round(model, free):
            pass
        if depth and layers == depth:
            break
        if not _round(model, rules):
            break
        layers += 1
    return dict(model)


def _round(model: Model, rules: tuple[Rule, ...]) -> bool:
    """Apply each rule once to the facts `model` holds, add the new facts, and say whether there were any."""
    fresh: Model = defaultdict(set)
    snapshot = {p: set(v) for p, v in model.items()}
    for rule in rules:
        for binding in _bindings(rule.body, snapshot):
            values = tuple(binding.get(t, t) for t in rule.head.args)
            if values not in model[rule.head.pred]:
                fresh[rule.head.pred].add(values)
    for pred, rows in fresh.items():
        model[pred] |= rows
    return bool(fresh)


def answers(model: Mapping[str, set[tuple[str, ...]]], query: Atom) -> set[tuple[str, ...]]:
    """The rows of the query's predicate that match its constants and repeated variables."""
    return {row for row in model.get(query.pred, ()) if query.match(row) is not None}
