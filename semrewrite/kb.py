"""Purpose: knowledge bases for the semantic rewriter: glossed predicates of any arity,
ground facts and Horn rules, plus the gold implications and queries a benchmark
adds, with gold answers and gold entailment read from the reference evaluator.

A rewrite counts as true when the gold program entails it, which one
canonical database decides for Horn programs: premise(c0, ..., cn-1) over fresh
constants, run to the fixpoint of the gold implications and user rules, holds
the conclusion under the rewrite's mapping exactly when the implication holds
in every model (the Herbrand-model argument behind Datalog query containment).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from semrewrite.fixpoint import Model, answers, least_model
from semrewrite.model import (
    Atom,
    Key,
    Predicate,
    Rewrite,
    Rule,
    check_symbol,
    identity,
    inverse,
    is_variable,
    keys_of,
    mappings,
)


def atom(pred: str, *args: str) -> Atom:
    """Authoring shorthand: atom("parent", "X", "tom") is Atom("parent", ("X", "tom"))."""
    return Atom(pred, tuple(args))


def canonical(arity: int) -> tuple[str, ...]:
    """Fresh constants for the canonical database of an arity; no knowledge base may use them."""
    return tuple(f"canonical_{i}" for i in range(arity))


@dataclass(frozen=True)
class KB:
    """Glosses, facts and user rules, with the gold implications, queries and trap predicates a benchmark adds.

    Facts may be written as tuples (pred, arg, ...) and gold implications as
    (premise, conclusion, mapping) triples, with a bool standing for the swap
    of a binary pair; both are normalised on construction.
    """

    name: str
    glosses: Mapping[str, str]
    facts: tuple[Atom, ...]
    rules: tuple[Rule, ...] = ()
    gold: frozenset[Key] = frozenset()
    queries: tuple[Atom, ...] = ()
    traps: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        facts = tuple(f if isinstance(f, Atom) else Atom(f[0], tuple(f[1:])) for f in self.facts)
        object.__setattr__(self, "facts", facts)
        object.__setattr__(self, "gold", keys_of(self.gold))
        object.__setattr__(self, "glosses", dict(self.glosses))
        catalogue = self.catalogue()
        for fact in facts:
            if not fact.ground():
                msg = f"fact {fact} in {self.name!r} has a variable"
                raise ValueError(msg)
        for a in (*facts, *self.queries, *(r.head for r in self.rules), *(b for r in self.rules for b in r.body)):
            if a.pred in catalogue and catalogue[a.pred].arity != a.arity:
                msg = f"{a} in {self.name!r} has {a.arity} arguments, but the gloss of {a.pred} has {catalogue[a.pred].arity}"
                raise ValueError(msg)
        for premise, conclusion, mapping in self.gold:
            for p in (premise, conclusion):
                if p in catalogue and catalogue[p].arity != len(mapping):
                    msg = f"gold implication {premise} => {conclusion} has a mapping of the wrong length"
                    raise ValueError(msg)
        missing = sorted(self.mentioned() - set(self.glosses))
        if missing:
            msg = f"knowledge base {self.name!r} has no gloss for {missing}"
            raise ValueError(msg)
        used = {c for f in facts for c in f.args}
        reserved = used & {c for n in range(1, 11) for c in canonical(n)}
        if reserved:
            msg = f"constants {sorted(reserved)} are reserved for the canonical database"
            raise ValueError(msg)

    def mentioned(self) -> set[str]:
        """Every predicate a fact, rule, gold implication or query names."""
        names = {f.pred for f in self.facts}
        for rule in self.rules:
            names.add(rule.head.pred)
            names.update(a.pred for a in rule.body)
        for premise, conclusion, _ in self.gold:
            names.update((premise, conclusion))
        names.update(q.pred for q in self.queries)
        return names

    def catalogue(self) -> dict[str, Predicate]:
        """Every glossed predicate, sorted by name."""
        return {check_symbol(n, "predicate"): Predicate(n, self.glosses[n]) for n in sorted(self.glosses)}

    def constants(self) -> set[str]:
        """Every constant a fact, rule or query mentions."""
        out = {c for f in self.facts for c in f.args}
        out |= {t for r in self.rules for a in (r.head, *r.body) for t in a.args if not is_variable(t)}
        out |= {t for q in self.queries for t in q.args if not is_variable(t)}
        return out

    def gold_rules(self) -> tuple[Rule, ...]:
        """The user rules plus one rule per gold implication."""
        return self.rules + tuple(Rewrite(c, p, m).rule() for p, c, m in sorted(self.gold))

    def closure(self, extra: Iterable[Rule] = ()) -> Model:
        """Every fact derivable from the facts under the gold rules and `extra`."""
        return least_model(self.facts, (*self.gold_rules(), *extra))

    def answers(self, query: Atom, closure: Model | None = None) -> set[tuple[str, ...]]:
        """The gold answers to `query`: rows of its predicate's closure matching its constants."""
        return answers(closure if closure is not None else self.closure(), query)

    def derives(self, premise: str) -> Model:
        """Everything the gold program derives from premise over fresh canonical constants."""
        arity = self.catalogue()[premise].arity
        return least_model((Atom(premise, canonical(arity)),), self.gold_rules())

    def entails(self, premise: str, conclusion: str, mapping: tuple[int, ...]) -> bool:
        """Whether premise under `mapping` entails conclusion in every model of the gold program."""
        c = canonical(len(mapping))
        back = inverse(mapping)
        return tuple(c[back[j]] for j in range(len(mapping))) in self.derives(premise).get(conclusion, set())

    # Time: one fixpoint per glossed predicate, over the gold program from a single fact.
    def entailed(self) -> frozenset[Key]:
        """Every non-trivial (premise, conclusion, mapping) over the glossed predicates the gold program entails."""
        catalogue = self.catalogue()
        out: set[Key] = set()
        for premise, pred in catalogue.items():
            derived = self.derives(premise)
            c = canonical(pred.arity)
            for conclusion, rows in derived.items():
                if conclusion not in catalogue or catalogue[conclusion].arity != pred.arity:
                    continue
                for mapping in mappings(pred.arity):
                    if conclusion == premise and mapping == identity(pred.arity):
                        continue
                    back = inverse(mapping)
                    if tuple(c[back[j]] for j in range(pred.arity)) in rows:
                        out.add((premise, conclusion, mapping))
        return frozenset(out)
