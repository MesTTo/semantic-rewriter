"""Purpose: the data model every part of the semantic rewriter shares, with no engine
in it: predicates with glosses, atoms, Horn rules, argument mappings, rewrites
and proposals, for predicates of any arity.

A rewrite says that a premise predicate implies a conclusion predicate of the
same arity under an argument mapping. `mapping[i]` is the conclusion argument
that fills the premise's position i, so for parent ⇐ child_of with mapping
(1, 0) the rule is parent(X0, X1) :- child_of(X1, X0), and for binary
predicates the identity (0, 1) and the swap (1, 0) are the only mappings.
"""

from __future__ import annotations

import itertools
import re
import string
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field

_SYMBOL = re.compile(r"^[a-z][A-Za-z0-9_]*$")
_VARIABLE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")
_NAMED = {"x": 0, "y": 1, "z": 2}

Mapping = tuple[int, ...]
"""A permutation: position i of the premise takes conclusion argument mapping[i]."""

Key = tuple[str, str, Mapping]
"""A rewrite's identity without its score: (premise, conclusion, mapping)."""


def is_variable(term: str) -> bool:
    """A variable is capitalised: X, Y, Person; a constant or predicate is a lowercase symbol."""
    return bool(_VARIABLE.match(term))


def check_symbol(term: str, role: str) -> str:
    """Return `term` when it is a lowercase symbol such as alice or has_mother, else raise naming `role`."""
    if not _SYMBOL.match(term):
        msg = f"{role} {term!r} must start with a lowercase letter and use only letters, digits and _"
        raise ValueError(msg)
    return term


def check_term(term: str) -> str:
    """Return `term` when it is a variable or a constant, else raise."""
    if not (is_variable(term) or _SYMBOL.match(term)):
        msg = f"term {term!r} is neither a capitalised variable nor a lowercase constant"
        raise ValueError(msg)
    return term


def identity(arity: int) -> Mapping:
    """The mapping that keeps every argument in place."""
    return tuple(range(arity))


def mappings(arity: int) -> Iterator[Mapping]:
    """Every permutation of `arity` positions, identity first; arity! of them."""
    return itertools.permutations(range(arity))


def inverse(mapping: Mapping) -> Mapping:
    """The mapping that undoes `mapping`, used to read a rewrite backwards."""
    out = [0] * len(mapping)
    for position, source in enumerate(mapping):
        out[source] = position
    return tuple(out)


def compose(outer: Mapping, inner: Mapping) -> Mapping:
    """Apply `inner` then `outer`: the mapping of a chain q ⇐(inner) r then p ⇐(outer) q is compose(outer, inner)."""
    return tuple(outer[i] for i in inner)


def gloss_positions(gloss: str) -> list[int]:
    """The argument positions a gloss mentions, in order of appearance.

    Placeholders are {x}, {y} and {z} for the first three arguments, or {0},
    {1}, ... for any position; one gloss uses one style.
    """
    fields = [f for _, f, _, _ in string.Formatter().parse(gloss) if f is not None]
    if any(f in _NAMED for f in fields) and any(f.isdigit() for f in fields):
        msg = f"gloss {gloss!r} mixes {{x}}-style and {{0}}-style placeholders"
        raise ValueError(msg)
    positions = []
    for f in fields:
        if f in _NAMED:
            positions.append(_NAMED[f])
        elif f.isdigit():
            positions.append(int(f))
        else:
            msg = f"gloss {gloss!r} has placeholder {{{f}}}; use {{x}}, {{y}}, {{z}} or {{0}}, {{1}}, ..."
            raise ValueError(msg)
    return positions


@dataclass(frozen=True, slots=True)
class Predicate:
    """A predicate with the sentence it states about its arguments.

    mother with gloss "{x} is the mother of {y}" is binary, and the fact
    mother(alice, tom) means "alice is the mother of tom". The arity is the
    number of distinct placeholders, which must be positions 0 to n-1.
    """

    name: str
    gloss: str
    arity: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.name:
            msg = "a predicate needs a name"
            raise ValueError(msg)
        positions = set(gloss_positions(self.gloss))
        if not positions or positions != set(range(len(positions))):
            msg = f"gloss of {self.name!r} must mention every argument position from 0 up: {self.gloss!r}"
            raise ValueError(msg)
        object.__setattr__(self, "arity", len(positions))

    def sentence(self, *fillers: str) -> str:
        """The gloss with argument i replaced by fillers[i], such as "A is the mother of B"."""
        if len(fillers) != self.arity:
            msg = f"{self.name} takes {self.arity} arguments, got {len(fillers)}"
            raise ValueError(msg)
        named = {k: fillers[v] for k, v in _NAMED.items() if v < self.arity}
        return self.gloss.format(*fillers, **named)


@dataclass(frozen=True, slots=True)
class Atom:
    """pred(t1, ..., tn), whose terms are variables or constants."""

    pred: str
    args: tuple[str, ...]

    def __post_init__(self) -> None:
        check_symbol(self.pred, "predicate")
        for t in self.args:
            check_term(t)

    @property
    def arity(self) -> int:
        return len(self.args)

    def variables(self) -> set[str]:
        """The variables among the arguments."""
        return {t for t in self.args if is_variable(t)}

    def ground(self) -> bool:
        """True when no argument is a variable."""
        return not self.variables()

    def match(self, values: Sequence[str]) -> dict[str, str] | None:
        """The binding that makes this atom equal `values`, or None when a constant or a repeated variable disagrees."""
        if len(values) != len(self.args):
            return None
        binding: dict[str, str] = {}
        for term, value in zip(self.args, values, strict=True):
            if is_variable(term):
                if binding.setdefault(term, value) != value:
                    return None
            elif term != value:
                return None
        return binding

    def __str__(self) -> str:
        return f"{self.pred}({', '.join(self.args)})"


def parse_atom(text: str) -> Atom:
    """Read an atom written as pred(a, B) or pred a B, as the command line and query strings use."""
    text = text.strip()
    m = re.fullmatch(r"([a-z][A-Za-z0-9_]*)\s*\((.*)\)", text)
    if m:
        args = tuple(a.strip() for a in m.group(2).split(",")) if m.group(2).strip() else ()
        return Atom(m.group(1), args)
    parts = text.split()
    if parts:
        return Atom(parts[0], tuple(parts[1:]))
    msg = f"cannot read an atom from {text!r}"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Rule:
    """A range-restricted Horn rule head :- body1, ..., bodyN with N >= 1 and no function symbols."""

    head: Atom
    body: tuple[Atom, ...]

    def __post_init__(self) -> None:
        if not self.body:
            msg = f"rule for {self.head.pred} has an empty body; write a fact instead"
            raise ValueError(msg)
        bound = set().union(*(atom.variables() for atom in self.body))
        if not self.head.variables() <= bound:
            msg = f"rule head {self.head} uses variables its body never binds"
            raise ValueError(msg)

    def __str__(self) -> str:
        return f"{self.head} :- {', '.join(map(str, self.body))}"


@dataclass(frozen=True, slots=True)
class Rewrite:
    """An admitted implication premise ⇒ conclusion under `mapping`, with the score that admitted it."""

    conclusion: str
    premise: str
    mapping: Mapping
    score: float = 1.0

    def __post_init__(self) -> None:
        check_symbol(self.conclusion, "predicate")
        check_symbol(self.premise, "predicate")
        if sorted(self.mapping) != list(range(len(self.mapping))):
            msg = f"rewrite mapping {self.mapping} is not a permutation"
            raise ValueError(msg)
        if self.premise == self.conclusion and self.mapping == identity(len(self.mapping)):
            msg = f"the identity rewrite of {self.premise} onto itself says nothing"
            raise ValueError(msg)

    @property
    def arity(self) -> int:
        return len(self.mapping)

    @property
    def key(self) -> Key:
        return (self.premise, self.conclusion, self.mapping)

    def rule(self) -> Rule:
        """The rewrite as a Horn rule: conclusion(X0, ..., Xn-1) :- premise(X[mapping[0]], ...)."""
        variables = tuple(f"X{i}" for i in range(self.arity))
        return Rule(Atom(self.conclusion, variables), (Atom(self.premise, tuple(variables[m] for m in self.mapping)),))

    def __str__(self) -> str:
        variables = [f"X{i}" for i in range(self.arity)]
        premise = ", ".join(variables[m] for m in self.mapping)
        return f"{self.conclusion}({', '.join(variables)}) ⇐ {self.premise}({premise})"


@dataclass(frozen=True, slots=True)
class Proposal:
    """The claim, put to a judge, that `premise` under `mapping` entails `conclusion`.

    With fillers A and B, Proposal(mother, parent, (0, 1)) asks whether "A is
    the mother of B" entails "A is a parent of B", and Proposal(child_of,
    parent, (1, 0)) whether "B is a child of A" entails it.
    """

    premise: Predicate
    conclusion: Predicate
    mapping: Mapping

    def __post_init__(self) -> None:
        if self.premise.arity != self.conclusion.arity or len(self.mapping) != self.conclusion.arity:
            msg = f"proposal {self.premise.name} ⇒ {self.conclusion.name} mixes arities or has a mapping of the wrong length"
            raise ValueError(msg)

    @property
    def key(self) -> Key:
        return (self.premise.name, self.conclusion.name, self.mapping)

    @property
    def swapped(self) -> bool:
        """For binary predicates, whether the premise reads its arguments the other way round."""
        return self.mapping == (1, 0)

    def sentences(self, fillers: Sequence[str] = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")) -> tuple[str, str]:
        """The premise and conclusion sentences, conclusion argument i written as fillers[i]."""
        if len(fillers) < self.conclusion.arity:
            msg = f"{len(fillers)} fillers for the {self.conclusion.arity} arguments of {self.conclusion.name}"
            raise ValueError(msg)
        premise = self.premise.sentence(*(fillers[m] for m in self.mapping))
        return premise, self.conclusion.sentence(*fillers[: self.conclusion.arity])

    def converse(self) -> Proposal:
        """The proposal read backwards, conclusion ⇒ premise, which a direction judge should score differently."""
        return Proposal(self.conclusion, self.premise, inverse(self.mapping))

    def rewrite(self, score: float) -> Rewrite:
        return Rewrite(self.conclusion.name, self.premise.name, self.mapping, score)


#: Fillers for up to ten argument positions; judges may use their own.
LETTERS: tuple[str, ...] = tuple("ABCDEFGHIJ")


def key_of(premise: str, conclusion: str, mapping: Mapping | bool) -> Key:
    """A rewrite key, accepting a bool for binary predicates (True for the swap) as authoring shorthand."""
    if isinstance(mapping, bool):
        mapping = (1, 0) if mapping else (0, 1)
    return (premise, conclusion, tuple(mapping))


def keys_of(triples: Iterable[tuple[str, str, Mapping | bool]]) -> frozenset[Key]:
    """Normalise authored (premise, conclusion, mapping-or-bool) triples to rewrite keys."""
    return frozenset(key_of(*t) for t in triples)
