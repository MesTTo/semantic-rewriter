"""Purpose: the reference evaluator: naive bottom-up evaluation of function-free Horn
programs of any arity, optionally cut at a derivation height. Gold answers,
gold entailment and the engine conformance kit all read answers from it, so it
stays as simple as correctness allows and shares no code with any engine.

Round k applies every rule to the facts known after round k-1, so after round
k the model holds exactly the facts with a derivation of height at most k,
where a fact has height 0 and a rule step adds one. A depth bound n is a stop
after round n; without one, evaluation stops at the first round adding nothing.
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


# Time: rounds x sum over rules of the bindings their bodies admit against the whole model;
# rounds <= the number of distinct derivable facts, or the depth bound. Space: the model.
def least_model(facts: Iterable[Atom], rules: Iterable[Rule], depth: int = 0) -> Model:
    """The facts derivable from `facts` under `rules`, with derivations of height at most `depth` when depth > 0."""
    rules = tuple(rules)
    model: Model = defaultdict(set)
    for fact in facts:
        model[fact.pred].add(fact.args)
    rounds = 0
    while depth == 0 or rounds < depth:
        rounds += 1
        fresh: Model = defaultdict(set)
        snapshot = {p: set(v) for p, v in model.items()}
        for rule in rules:
            for binding in _bindings(rule.body, snapshot):
                values = tuple(binding.get(t, t) for t in rule.head.args)
                if values not in model[rule.head.pred]:
                    fresh[rule.head.pred].add(values)
        if not fresh:
            break
        for pred, rows in fresh.items():
            model[pred] |= rows
    return dict(model)


def answers(model: Mapping[str, set[tuple[str, ...]]], query: Atom) -> set[tuple[str, ...]]:
    """The rows of the query's predicate that match its constants and repeated variables."""
    return {row for row in model.get(query.pred, ()) if query.match(row) is not None}
