"""Purpose: read and write knowledge bases in three formats: a Datalog-like text file,
JSON, and native MeTTa. Each loader refuses a statement it does not understand,
naming it, because a rule silently skipped would make every later answer quietly
incomplete.

Text (.kb, .dl, .pl or .txt), one statement per line, % or # starting a comment:

    gloss mother "{x} is the mother of {y}"
    gloss parent "{x} is a parent of {y}"
    gloss guardian "{x} is the legal guardian of {y}"
    mother(alice, tom).
    grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
    query parent(X, tom).
    gold mother => parent.
    gold child_of => parent swap.
    trap guardian.

A gold line names an implication the benchmark counts as true, with an
optional mapping written `swap` (binary only) or as positions such as `1 0 2`;
without one it is the identity of the conclusion's arity. A trap line names a
predicate that looks related to the benchmark's but is not. A gloss is quoted
as a JSON string, so `\"`, `\\` and `\u00e9` read as in JSON and any other
character, accents and arrows included, stands for itself.

JSON (.json): {"name": ..., "predicates": {name: gloss}, "facts": [[pred, arg, ...]],
"rules": ["head :- body, ..."], "queries": ["pred(X, a)"], "gold": [[premise, conclusion, [0, 1]]], "traps": [name]}.

MeTTa (.metta): (gloss mother "{x} is the mother of {y}"), facts (= (mother alice tom) True),
rules (= (grandparent $X $Z) (let True (parent $X $Y) (parent $Y $Z))), and the benchmark's parts as
inert atoms: (query (parent $X tom)), (gold child_of parent (1 0)), (trap guardian). Reading MeTTa
needs pymetta.

Files are read and written as UTF-8, and each format holds every part of a knowledge base but its
name, so loading what a writer wrote gives the same knowledge base back.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from semrewrite.kb import KB
from semrewrite.model import Atom, Key, Predicate, Rule, is_variable, keys_of, parse_atom

_ATOM = r"[a-z][A-Za-z0-9_]*\s*\([^()]*\)"
_GLOSS = re.compile(r'^gloss\s+([a-z][A-Za-z0-9_]*)\s+("(?:[^"\\]|\\.)*")\s*\.?$')
_GOLD = re.compile(r"^gold\s+([a-z][A-Za-z0-9_]*)\s*=>\s*([a-z][A-Za-z0-9_]*)\s*(.*?)\s*\.?$")
_TRAP = re.compile(r"^trap\s+([a-z][A-Za-z0-9_]*)\s*\.?$")


def parse_rule(text: str) -> Rule:
    """Read `head :- body1, body2` into a Rule; the body must be atoms separated by commas and nothing else."""
    if ":-" not in text:
        msg = f"not a rule: {text!r}"
        raise ValueError(msg)
    head, body = text.split(":-", 1)
    if not re.fullmatch(rf"\s*{_ATOM}(?:\s*,\s*{_ATOM})*\s*", body):
        msg = f"cannot read the body of rule {text!r}; write atoms like p(X, a) separated by commas"
        raise ValueError(msg)
    return Rule(parse_atom(head), tuple(parse_atom(a) for a in re.findall(_ATOM, body)))


def _quote(gloss: str) -> str:
    """A gloss as the text and MeTTa formats quote it: a JSON string, non-ASCII left as it is."""
    return json.dumps(gloss, ensure_ascii=False)


def _strip_comment(line: str) -> str:
    """The line without its comment: a % or # outside double quotes starts one."""
    quoted = escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = quoted
        elif ch == '"':
            quoted = not quoted
        elif ch in "%#" and not quoted:
            return line[:i].strip()
    return line.strip()


def _mapping(text: str, line: str) -> tuple[int, ...] | bool | None:
    """The mapping a gold line writes: None for none, True for swap, else the positions."""
    text = text.strip()
    if not text:
        return None
    if text == "swap":
        return True
    try:
        return tuple(int(t) for t in text.split())
    except ValueError:
        msg = f"cannot read the mapping {text!r} in {line!r}; write swap or positions such as 1 0"
        raise ValueError(msg) from None


def loads_text(text: str, name: str = "kb") -> KB:
    """Read the text format."""
    glosses: dict[str, str] = {}
    facts: list[Atom] = []
    rules: list[Rule] = []
    queries: list[Atom] = []
    gold: list[tuple[str, str, tuple[int, ...] | bool | None]] = []
    traps: list[str] = []
    # Lines end at \n alone, as written; splitlines() would also break at characters a quoted gloss may not hold
    # but a comment may, such as U+2028.
    for number, raw in enumerate(text.split("\n"), 1):
        raw = raw.removesuffix("\r")
        line = _strip_comment(raw)
        if not line:
            continue
        try:
            if m := _GLOSS.match(line):
                glosses[m.group(1)] = json.loads(m.group(2))
            elif m := _GOLD.match(line):
                gold.append((m.group(1), m.group(2), _mapping(m.group(3), line)))
            elif m := _TRAP.match(line):
                traps.append(m.group(1))
            elif line.startswith("query "):
                queries.append(parse_atom(line[len("query "):].rstrip(".")))
            elif ":-" in line:
                rules.append(parse_rule(line.rstrip(".")))
            else:
                fact = parse_atom(line.rstrip("."))
                if not fact.ground():
                    msg = f"fact {fact} has a variable; a rule needs a body"
                    raise ValueError(msg)
                facts.append(fact)
        except ValueError as e:
            msg = f"line {number}: {e}"
            raise ValueError(msg) from None
    resolved = []
    for premise, conclusion, mapping in gold:
        if mapping is None:
            if conclusion not in glosses:
                msg = f"gold {premise} => {conclusion}: {conclusion} has no gloss to give its arity"
                raise ValueError(msg)
            mapping = tuple(range(Predicate(conclusion, glosses[conclusion]).arity))
        resolved.append((premise, conclusion, mapping))
    return KB(name=name, glosses=glosses, facts=tuple(facts), rules=tuple(rules), gold=keys_of(resolved),
              queries=tuple(queries), traps=frozenset(traps))


def dumps_text(kb: KB) -> str:
    """Write the text format; loads_text(dumps_text(kb)) equals kb up to the name."""
    lines = [f"gloss {n} {_quote(kb.glosses[n])}" for n in sorted(kb.glosses)]
    lines += [f"{f}." for f in kb.facts]
    lines += [f"{r}." for r in kb.rules]
    lines += [f"query {q}." for q in kb.queries]
    for premise, conclusion, mapping in sorted(kb.gold):
        tail = "" if mapping == tuple(range(len(mapping))) else " " + " ".join(map(str, mapping))
        lines.append(f"gold {premise} => {conclusion}{tail}.")
    lines += [f"trap {t}." for t in sorted(kb.traps)]
    return "\n".join(lines) + "\n"


def loads_json(text: str, name: str = "kb") -> KB:
    """Read the JSON format."""
    data = json.loads(text)
    unknown = set(data) - {"name", "predicates", "facts", "rules", "queries", "gold", "traps"}
    if unknown:
        msg = f"unknown keys {sorted(unknown)} in knowledge-base JSON"
        raise ValueError(msg)
    gold: list[Key | tuple[str, str, bool]] = []
    for row in data.get("gold", ()):
        premise, conclusion, mapping = row
        gold.append((premise, conclusion, mapping if isinstance(mapping, bool) else tuple(mapping)))
    return KB(
        name=data.get("name", name),
        glosses=data.get("predicates", {}),
        facts=tuple(Atom(f[0], tuple(f[1:])) for f in data.get("facts", ())),
        rules=tuple(parse_rule(r) for r in data.get("rules", ())),
        queries=tuple(parse_atom(q) for q in data.get("queries", ())),
        gold=keys_of(gold),
        traps=frozenset(data.get("traps", ())),
    )


def dumps_json(kb: KB) -> str:
    """Write the JSON format."""
    return json.dumps({
        "name": kb.name,
        "predicates": dict(sorted(kb.glosses.items())),
        "facts": [[f.pred, *f.args] for f in kb.facts],
        "rules": [str(r) for r in kb.rules],
        "queries": [str(q) for q in kb.queries],
        "gold": [[p, c, list(m)] for p, c, m in sorted(kb.gold)],
        "traps": sorted(kb.traps),
    }, indent=1, ensure_ascii=False) + "\n"


def _metta_term(t: str) -> str:
    return f"${t}" if is_variable(t) else t


def dumps_metta(kb: KB) -> str:
    """Write native MeTTa: gloss atoms, facts as equations, rules as let chains, then the query, gold and trap atoms."""

    def call(a: Atom) -> str:
        return "(" + " ".join([a.pred, *(_metta_term(t) for t in a.args)]) + ")"

    lines = [f"(gloss {n} {_quote(kb.glosses[n])})" for n in sorted(kb.glosses)]
    lines += [f"(= {call(f)} True)" for f in kb.facts]
    for r in kb.rules:
        body = call(r.body[-1])
        for a in reversed(r.body[:-1]):
            body = f"(let True {call(a)} {body})"
        lines.append(f"(= {call(r.head)} {body})")
    lines += [f"(query {call(q)})" for q in kb.queries]
    lines += [f"(gold {p} {c} ({' '.join(map(str, m))}))" for p, c, m in sorted(kb.gold)]
    lines += [f"(trap {t})" for t in sorted(kb.traps)]
    return "\n".join(lines) + "\n"


def loads_metta(source: str, name: str = "kb") -> KB:
    """Read native MeTTa as dumps_metta writes it; needs pymetta."""
    from metta import Atom as MeTTaAtom
    from metta import Expression, Grounded, forms

    def term(atom: MeTTaAtom) -> str:
        if atom.metatype == "Variable":
            text = str(atom).lstrip("$")
            if not is_variable(text):
                msg = f"variable {atom} must be capitalised to load"
                raise ValueError(msg)
            return text
        if atom.metatype == "Symbol":
            return str(atom)
        msg = f"argument {atom} is neither a symbol nor a variable"
        raise ValueError(msg)

    def call(atom: object) -> Atom:
        if not isinstance(atom, Expression) or not atom.children:
            msg = f"expected a call (p a ...), got {atom}"
            raise ValueError(msg)
        return Atom(str(atom.children[0]), tuple(term(c) for c in atom.children[1:]))

    def body(atom: object) -> list[Atom]:
        out = []
        while isinstance(atom, Expression) and atom.children and str(atom.children[0]) == "let":
            _, pattern, first, rest = atom.children
            if str(pattern) != "True":
                msg = f"a rule body's let must bind True, got {pattern}"
                raise ValueError(msg)
            out.append(call(first))
            atom = rest
        return [*out, call(atom)]

    glosses: dict[str, str] = {}
    facts: list[Atom] = []
    rules: list[Rule] = []
    queries: list[Atom] = []
    gold: list[Key] = []
    traps: list[str] = []
    for form in forms(source):
        if not isinstance(form, Expression) or not form.children:
            msg = f"unsupported top-level form in knowledge-base source: {form}"
            raise ValueError(msg)
        kind = str(form.children[0])
        if kind == "gloss" and len(form.children) == 3:
            _, pred, text = form.children
            if not isinstance(text, Grounded) or not isinstance(text.value, str):
                msg = f"the gloss of {pred} must be a string, got {text}"
                raise ValueError(msg)
            glosses[str(pred)] = text.value
        elif kind == "=" and len(form.children) == 3:
            _, lhs, rhs = form.children
            head = call(lhs)
            if str(rhs) == "True" and head.ground():
                facts.append(head)
            else:
                rules.append(Rule(head, tuple(body(rhs))))
        elif kind == "query" and len(form.children) == 2:
            queries.append(call(form.children[1]))
        elif kind == "gold" and len(form.children) == 4 and isinstance(form.children[3], Expression):
            _, premise, conclusion, mapping = form.children
            gold.append((str(premise), str(conclusion), tuple(int(str(p)) for p in mapping.children)))
        elif kind == "trap" and len(form.children) == 2:
            traps.append(str(form.children[1]))
        else:
            msg = f"unsupported top-level form in knowledge-base source: {form}"
            raise ValueError(msg)
    return KB(name=name, glosses=glosses, facts=tuple(facts), rules=tuple(rules), queries=tuple(queries),
              gold=keys_of(gold), traps=frozenset(traps))


_READERS = {".json": loads_json, ".metta": loads_metta}
_WRITERS = {".json": dumps_json, ".metta": dumps_metta}


def load(path: str | Path) -> KB:
    """Read a knowledge base, choosing the format by extension: .json, .metta, or the text format for anything else."""
    path = Path(path)
    return _READERS.get(path.suffix, loads_text)(path.read_text(encoding="utf-8"), name=path.stem)


def save(kb: KB, path: str | Path) -> None:
    """Write a knowledge base in the format its extension names, the text format for anything else."""
    path = Path(path)
    path.write_text(_WRITERS.get(path.suffix, dumps_text)(kb), encoding="utf-8")
