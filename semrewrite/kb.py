"""Purpose: knowledge bases for the semantic rewriter: a Horn-rule DSL over binary
predicates, its native MeTTa text, a loader reading that text back from pymetta
atoms, and the semi-naive fixpoint that computes gold answers.

A knowledge base is glosses, facts and user rules. In MeTTa a fact is the
equation (= (mother alice tom) True), a user rule (= (grandparent $X $Z)
(let True (parent $X $Y) (parent $Y $Z))) and a gloss (gloss mother "{x} is
the mother of {y}"). Variables are written capitalised in the DSL (X, Y) and
become $X, $Y in MeTTa; predicates and constants are lowercase symbols.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

from metta import Expression, forms

from semrewrite.judging import Predicate

_SYMBOL = re.compile(r"^[a-z][a-z0-9_]*$")
_VARIABLE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")

#: Gold implication triples (premise, conclusion, swapped), as semrewrite.judging.Proposal.key.
Implication = tuple[str, str, bool]
Pair = tuple[str, str]


def is_variable(term: str) -> bool:
    """A DSL variable is capitalised: X, Y, Person."""
    return bool(_VARIABLE.match(term))


def _check_symbol(term: str, role: str) -> None:
    if not _SYMBOL.match(term):
        msg = f"{role} {term!r} is not a lowercase MeTTa symbol"
        raise ValueError(msg)


def _check_term(term: str) -> None:
    if not (is_variable(term) or _SYMBOL.match(term)):
        msg = f"term {term!r} is neither a capitalised variable nor a lowercase symbol"
        raise ValueError(msg)


def _metta_term(term: str) -> str:
    return f"${term}" if is_variable(term) else term


@dataclass(frozen=True, slots=True)
class Lit:
    """A binary literal pred(a, b) whose arguments are variables or constants."""

    pred: str
    a: str
    b: str

    def __post_init__(self) -> None:
        _check_symbol(self.pred, "predicate")
        _check_term(self.a)
        _check_term(self.b)

    def variables(self) -> set[str]:
        """The variables among the two arguments."""
        return {t for t in (self.a, self.b) if is_variable(t)}

    def metta(self) -> str:
        """The literal as a MeTTa call, such as (parent $X tom)."""
        return f"({self.pred} {_metta_term(self.a)} {_metta_term(self.b)})"


@dataclass(frozen=True, slots=True)
class Rule:
    """A range-restricted Horn rule head :- body1, ..., bodyN with N >= 1."""

    head: Lit
    body: tuple[Lit, ...]

    def __post_init__(self) -> None:
        if not self.body:
            msg = f"rule for {self.head.pred} has an empty body; write a fact instead"
            raise ValueError(msg)
        bound = set().union(*(lit.variables() for lit in self.body))
        if not self.head.variables() <= bound:
            msg = f"rule head {self.head} uses variables its body never binds"
            raise ValueError(msg)

    def metta_body(self, call=Lit.metta) -> str:
        """The body as a left-to-right let chain ending in its last literal.

        (let True b1 (let True b2 b3)) proves b1, then b2 under b1's bindings,
        then answers b3's value, which is True when the conjunction holds.
        """
        text = call(self.body[-1])
        for lit in reversed(self.body[:-1]):
            text = f"(let True {call(lit)} {text})"
        return text

    def metta(self) -> str:
        """The rule as a native MeTTa equation."""
        return f"(= {self.head.metta()} {self.metta_body()})"


@dataclass(frozen=True, slots=True)
class Query:
    """A question pred(a, b) with variables marking what is asked for."""

    pred: str
    a: str
    b: str

    def __post_init__(self) -> None:
        _check_symbol(self.pred, "predicate")
        _check_term(self.a)
        _check_term(self.b)

    @property
    def lit(self) -> Lit:
        """The query as a literal."""
        return Lit(self.pred, self.a, self.b)

    def metta(self) -> str:
        """A term answering one (ans a b) per proof, such as (let True (parent $X tom) (ans $X tom))."""
        a, b = _metta_term(self.a), _metta_term(self.b)
        return f"(let True ({self.pred} {a} {b}) (ans {a} {b}))"

    def __str__(self) -> str:
        return f"{self.pred}({self.a}, {self.b})"


def implication_rule(premise: str, conclusion: str, swapped: bool) -> Rule:
    """The Horn rule an implication triple states: conclusion(X, Y) :- premise(X, Y), or premise(Y, X) when swapped."""
    body = Lit(premise, "Y", "X") if swapped else Lit(premise, "X", "Y")
    return Rule(Lit(conclusion, "X", "Y"), (body,))


#: Fresh constants for the canonical database; no knowledge base may use them.
CANONICAL = ("canonical_a", "canonical_b")


@dataclass(frozen=True)
class KB:
    """Glosses, facts and user rules, with the gold implications and queries a benchmark adds."""

    name: str
    glosses: Mapping[str, str]
    facts: tuple[tuple[str, str, str], ...]
    rules: tuple[Rule, ...] = ()
    gold: frozenset[Implication] = frozenset()
    queries: tuple[Query, ...] = ()
    traps: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for name in self.glosses:
            _check_symbol(name, "predicate")
            Predicate(name, self.glosses[name])
        for pred, a, b in self.facts:
            _check_symbol(pred, "predicate")
            _check_symbol(a, "constant")
            _check_symbol(b, "constant")
        missing = sorted(self.mentioned() - set(self.glosses))
        if missing:
            msg = f"knowledge base {self.name!r} has no gloss for {missing}"
            raise ValueError(msg)

    def mentioned(self) -> set[str]:
        """Every predicate a fact, rule, gold triple or query names."""
        names = {pred for pred, _, _ in self.facts}
        for rule in self.rules:
            names.add(rule.head.pred)
            names.update(lit.pred for lit in rule.body)
        for premise, conclusion, _ in self.gold:
            names.update((premise, conclusion))
        names.update(q.pred for q in self.queries)
        return names

    def catalogue(self) -> dict[str, Predicate]:
        """Every glossed predicate as a judge-ready Predicate, sorted by name."""
        return {name: Predicate(name, self.glosses[name]) for name in sorted(self.glosses)}

    def to_metta(self) -> str:
        """The native MeTTa source: one gloss atom per predicate, then facts, then user rules."""
        lines = [f"(gloss {name} {_metta_string(self.glosses[name])})" for name in sorted(self.glosses)]
        lines += [f"(= ({p} {a} {b}) True)" for p, a, b in self.facts]
        lines += [rule.metta() for rule in self.rules]
        return "\n".join(lines) + "\n"

    def gold_rules(self) -> tuple[Rule, ...]:
        """The user rules plus one implication rule per gold triple."""
        return self.rules + tuple(implication_rule(*t) for t in sorted(self.gold))

    def closure(self, extra: Iterable[Rule] = ()) -> dict[str, set[Pair]]:
        """Every fact derivable from the facts under the gold rules and `extra`."""
        return fixpoint(self.facts, (*self.gold_rules(), *extra))

    def answers(self, query: Query, closure: Mapping[str, set[Pair]] | None = None) -> set[Pair]:
        """The gold answers to `query`: the pairs of its predicate's closure matching its constants."""
        table = (closure if closure is not None else self.closure()).get(query.pred, set())
        return {pair for pair in table if _matches(query.lit, pair)}

    def derives(self, premise: str) -> dict[str, set[Pair]]:
        """Everything the gold program derives from the single fact premise(a, b) over fresh constants."""
        a, b = CANONICAL
        return fixpoint(((premise, a, b),), self.gold_rules())

    def entails(self, premise: str, conclusion: str, swapped: bool) -> bool:
        """Whether premise(x, y), or premise(y, x) when swapped, entails conclusion(x, y) under the gold program.

        For Horn rules this is decided by one canonical database: the rewrite
        holds in every model exactly when the fixpoint of the single fact
        premise(a, b) over fresh constants a and b contains conclusion(a, b),
        or conclusion(b, a) when swapped (the Herbrand-model argument behind
        Datalog query containment). It counts what the gold implications and
        the user rules force together, so parent => ancestor holds through
        the rule ancestor(X, Y) :- parent(X, Y) with no gold triple naming it.
        """
        a, b = CANONICAL
        target = (b, a) if swapped else (a, b)
        return target in self.derives(premise).get(conclusion, set())

    # Time: |P| fixpoints, each over the gold program from one fact, for |P| glossed predicates.
    def entailed(self) -> frozenset[Implication]:
        """Every non-trivial (premise, conclusion, swapped) triple over the glossed predicates the gold program entails."""
        a, b = CANONICAL
        out: set[Implication] = set()
        for premise in self.glosses:
            derived = self.derives(premise)
            for conclusion, pairs in derived.items():
                if conclusion not in self.glosses:
                    continue
                if (a, b) in pairs and conclusion != premise:
                    out.add((premise, conclusion, False))
                if (b, a) in pairs:
                    out.add((premise, conclusion, True))
        return frozenset(out)


def _metta_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _matches(lit: Lit, pair: Pair) -> bool:
    binding: dict[str, str] = {}
    for term, value in zip((lit.a, lit.b), pair, strict=True):
        if is_variable(term):
            if binding.setdefault(term, value) != value:
                return False
        elif term != value:
            return False
    return True


def _extend(lit: Lit, pair: Pair, binding: Mapping[str, str]) -> dict[str, str] | None:
    out = dict(binding)
    for term, value in zip((lit.a, lit.b), pair, strict=True):
        if is_variable(term):
            if out.setdefault(term, value) != value:
                return None
        elif term != value:
            return None
    return out


def _ground(lit: Lit, binding: Mapping[str, str]) -> Pair:
    return (binding.get(lit.a, lit.a), binding.get(lit.b, lit.b))


def _join(
    body: tuple[Lit, ...],
    pivot: int,
    total: Mapping[str, set[Pair]],
    delta: Mapping[str, set[Pair]],
) -> Iterator[dict[str, str]]:
    """Bindings satisfying `body` with literal `pivot` drawn from delta and the rest from total."""

    def walk(i: int, binding: dict[str, str]) -> Iterator[dict[str, str]]:
        if i == len(body):
            yield binding
            return
        lit = body[i]
        source = delta if i == pivot else total
        for pair in source.get(lit.pred, ()):
            extended = _extend(lit, pair, binding)
            if extended is not None:
                yield from walk(i + 1, extended)

    yield from walk(0, {})


# Time: each round joins every rule body once per body position, so a round
# costs sum over rules of b * |delta| * |total|^(b-1) bindings for body length b;
# rounds <= the number of distinct derivable facts. Space: one set per predicate.
def fixpoint(facts: Iterable[tuple[str, str, str]], rules: Iterable[Rule]) -> dict[str, set[Pair]]:
    """Semi-naive bottom-up evaluation: each round fires rules only on facts new in the last round."""
    rules = tuple(rules)
    total: dict[str, set[Pair]] = defaultdict(set)
    for pred, a, b in facts:
        total[pred].add((a, b))
    delta: dict[str, set[Pair]] = {p: set(pairs) for p, pairs in total.items()}
    while any(delta.values()):
        fresh: dict[str, set[Pair]] = defaultdict(set)
        for rule in rules:
            for pivot in range(len(rule.body)):
                for binding in _join(rule.body, pivot, total, delta):
                    pair = _ground(rule.head, binding)
                    if pair not in total[rule.head.pred]:
                        fresh[rule.head.pred].add(pair)
        for pred, pairs in fresh.items():
            total[pred] |= pairs
        delta = fresh
    return dict(total)


def from_metta(source: str, name: str = "loaded") -> KB:
    """Read glosses, facts and Horn rules back from native MeTTa source.

    Accepts exactly the shapes KB.to_metta writes: (gloss p "text"),
    (= (p a b) True) and (= (p T U) body) whose body is a call or a
    (let True call rest) chain. Anything else raises, naming the form, since
    a rule silently skipped would make every later answer quietly incomplete.
    """
    glosses: dict[str, str] = {}
    facts: list[tuple[str, str, str]] = []
    rules: list[Rule] = []
    for form in forms(source):
        kind = _head(form)
        if kind == "gloss":
            _, pred, text = form.children
            glosses[str(pred)] = text.value
        elif kind == "=":
            _, lhs, rhs = form.children
            head = _lit(lhs)
            if str(rhs) == "True" and not head.variables():
                facts.append((head.pred, head.a, head.b))
            else:
                rules.append(Rule(head, tuple(_body(rhs))))
        else:
            msg = f"unsupported top-level form in knowledge base source: {form}"
            raise ValueError(msg)
    return KB(name=name, glosses=glosses, facts=tuple(facts), rules=tuple(rules))


def _head(form: object) -> str:
    if not isinstance(form, Expression) or not form.children:
        msg = f"expected an expression, got {form}"
        raise ValueError(msg)
    return str(form.children[0])


def _term(atom: object) -> str:
    if atom.metatype == "Variable":
        text = str(atom).lstrip("$")
        if not _VARIABLE.match(text):
            msg = f"variable {atom} must be capitalised to load"
            raise ValueError(msg)
        return text
    if atom.metatype == "Symbol":
        return str(atom)
    msg = f"argument {atom} is neither a symbol nor a variable"
    raise ValueError(msg)


def _lit(atom: object) -> Lit:
    if not isinstance(atom, Expression) or len(atom.children) != 3:
        msg = f"expected a binary call (p a b), got {atom}"
        raise ValueError(msg)
    pred, a, b = atom.children
    return Lit(str(pred), _term(a), _term(b))


def _body(atom: object) -> Iterator[Lit]:
    while isinstance(atom, Expression) and atom.children and str(atom.children[0]) == "let":
        _, pattern, call, rest = atom.children
        if str(pattern) != "True":
            msg = f"a rule body's let must bind True, got {pattern}"
            raise ValueError(msg)
        yield _lit(call)
        atom = rest
    yield _lit(atom)
