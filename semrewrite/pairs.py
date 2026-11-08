"""Purpose: labelled predicate pairs for measuring direction judges: the row
schema, its JSON-lines storage under data/pairs/, the fillers each row offers,
and the builders that derive every set from its source under data/raw/ or the
hand-written data/pairs/hard.toml.

A row says whether its premise predicate entails its conclusion predicate over
shared arguments, swapped or not, as a Proposal does. Rows that are each other's
reverse, (P, Q, s) and (Q, P, s), share a `pair` id, so direction accuracy can
compare a pair's two directions.

    python -m semrewrite.pairs fetch levyholt   # download the Levy/Holt files at the pinned commit
    python -m semrewrite.pairs fetch wikidata   # rerun the Wikidata queries, overwriting their results
    python -m semrewrite.pairs build [SET ...]  # rebuild data/pairs/<set>.jsonl from the sources, every set by default

Building the Wikidata set needs the `models` extra, for the embedding model
that picks its near-miss pairs; the other two sets need nothing.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
import tomllib
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from semrewrite.model import Predicate, Proposal

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PAIRS = ROOT / "data" / "pairs"
SETS = ("levyholt", "wikidata", "hard")
FILLER_VARIANTS = ("letters", "typed", "instances")

Fillers = tuple[str, str]


@dataclass(frozen=True, slots=True)
class PairRow:
    """One labelled implication proposal.

    `label` is gold for "premise entails conclusion" under the swap flag.
    `types` names the kinds of the conclusion's first and second argument
    ("person", "place"), which the premise shares; `instances` are example
    arguments, present only where the source gives them.
    """

    id: str
    set: str
    split: str
    pair: str
    kind: str
    premise: Predicate
    conclusion: Predicate
    swapped: bool
    label: bool
    types: Fillers
    instances: Fillers | None = None
    source: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.split not in ("dev", "test"):
            msg = f"{self.id}: split must be dev or test, not {self.split!r}"
            raise ValueError(msg)

    @property
    def proposal(self) -> Proposal:
        return Proposal(self.premise, self.conclusion, (1, 0) if self.swapped else (0, 1))

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "set": self.set,
            "split": self.split,
            "pair": self.pair,
            "kind": self.kind,
            "premise": {"name": self.premise.name, "gloss": self.premise.gloss},
            "conclusion": {"name": self.conclusion.name, "gloss": self.conclusion.gloss},
            "swapped": self.swapped,
            "label": self.label,
            "types": list(self.types),
            "instances": list(self.instances) if self.instances else None,
            "source": self.source,
            "reason": self.reason,
        }

    @classmethod
    def from_json(cls, row: dict[str, object]) -> PairRow:
        premise, conclusion = row["premise"], row["conclusion"]
        assert isinstance(premise, dict) and isinstance(conclusion, dict)
        types, instances = row["types"], row["instances"]
        return cls(
            id=str(row["id"]),
            set=str(row["set"]),
            split=str(row["split"]),
            pair=str(row["pair"]),
            kind=str(row["kind"]),
            premise=Predicate(premise["name"], premise["gloss"]),
            conclusion=Predicate(conclusion["name"], conclusion["gloss"]),
            swapped=bool(row["swapped"]),
            label=bool(row["label"]),
            types=(types[0], types[1]),  # type: ignore[index]
            instances=(instances[0], instances[1]) if instances else None,  # type: ignore[index]
            source=str(row.get("source", "")),
            reason=str(row.get("reason", "")),
        )


def typed_nouns(types: Fillers) -> Fillers:
    """Definite noun phrases for two typed arguments, told apart when their types match.

    >>> typed_nouns(("medicine", "disease"))
    ('the medicine', 'the disease')
    >>> typed_nouns(("person", "person"))
    ('the first person', 'the second person')
    """
    x, y = types
    if x == y:
        return f"the first {x}", f"the second {y}"
    return f"the {x}", f"the {y}"


def fillers_for(row: PairRow, variant: str) -> Fillers | None:
    """The argument texts a variant fills the row's sentences with; None when the row has no instances."""
    if variant == "letters":
        return ("A", "B")
    if variant == "typed":
        return typed_nouns(row.types)
    if variant == "instances":
        return row.instances
    msg = f"unknown filler variant {variant!r}; known: {FILLER_VARIANTS}"
    raise ValueError(msg)


def save(rows: Sequence[PairRow], path: pathlib.Path) -> None:
    """Write rows as JSON lines, one row per line in the order given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r.to_json(), ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def load(sets: Iterable[str] = SETS, directory: pathlib.Path = PAIRS) -> list[PairRow]:
    """Every row of the named sets, read from data/pairs/<set>.jsonl.

    A missing set raises FileNotFoundError naming the commands that make it.
    """
    rows: list[PairRow] = []
    for name in sets:
        path = directory / f"{name}.jsonl"
        if not path.exists():
            fetch = f"python -m semrewrite.pairs fetch {name}, then " if name == "levyholt" else ""
            msg = f"{path} is missing: run {fetch}python -m semrewrite.pairs build {name}"
            raise FileNotFoundError(msg)
        rows.extend(PairRow.from_json(json.loads(line)) for line in path.read_text().splitlines() if line)
    return rows


def check(rows: Sequence[PairRow]) -> None:
    """Raise unless ids are unique and each pair id holds rows that are each other's reverse."""
    ids = [r.id for r in rows]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        msg = f"duplicate row ids: {dupes[:5]}"
        raise ValueError(msg)
    by_pair: dict[str, list[PairRow]] = {}
    for r in rows:
        by_pair.setdefault(r.pair, []).append(r)
    for pair, members in by_pair.items():
        if len(members) > 2:
            msg = f"pair {pair} holds {len(members)} rows"
            raise ValueError(msg)
        if len(members) == 2:
            a, b = members
            if (a.premise, a.conclusion, a.swapped) != (b.conclusion, b.premise, b.swapped) or a.split != b.split:
                msg = f"pair {pair}: {a.id} and {b.id} are not each other's reverse in one split"
                raise ValueError(msg)


# ---------------------------------------------------------------- Levy/Holt

# Levy/Holt arguments differ between premise and hypothesis by determiner and
# number ("Some cheese" against "Cheese", "a symphony" against "symphonies"), so
# arguments are compared after dropping these and by head-noun prefix.
_DETERMINERS = re.compile(
    r"^(?:(?:the|a|an|this|that|these|those|some|any|all|both|more|most|many|several|each|every|such|"
    r"its|their|his|her|our|your|my|no)\s+)+",
    re.IGNORECASE,
)

# FIGER-style type names in the _rels files that are not already plain nouns.
_TYPE_NOUNS = {
    "location": "place",
    "chemistry": "chemical",
    "livingthing": "living thing",
    "title": "professional",
    "god": "deity",
    "biology": "biological entity",
    "education": "field of study",
    "military": "military force",
    "astral_body": "celestial body",
    "people": "people",
    "art": "artwork",
    "visual_art": "work of art",
    "finance": "financial entity",
    "geography": "geographic feature",
}

_RELS_CELL = re.compile(r"^\((?P<pred>[^)]*)\) (?P<a1>\S+)::(?P<t1>\S+) (?P<a2>\S+)::(?P<t2>\S+)$")

# The four Levy/Holt files levyholt_rows reads, as Hosseini's entgraph_eval
# holds them at the commit the experiment used, each with its SHA-256.
LEVYHOLT_REPOSITORY = "mjhosseini/entgraph_eval"
LEVYHOLT_COMMIT = "e5652b3faab2a41b1abfb4bff441b69e31925756"
LEVYHOLT_FILES = {
    "LevyHoltDS/dev_dir/dev_dir_s.txt": "c36ceb6aba81d73da8ab43d6136bb4b3d22ed19bb05d220b851d9c325f288d60",
    "LevyHoltDS/dev_dir/dev_dir_rels.txt": "7049ea89031ce2ff777f9fe398a1bf43dd03953140efa6d3060eb802c3d4ad3a",
    "LevyHoltDS/test_dir/test_dir_s.txt": "216161651ced8003a81eb37f769964384bba2d34226d86c0d1e4ad67356cf3e3",
    "LevyHoltDS/test_dir/test_dir_rels.txt": "de6164136ce4e23d6ed330d4370c05482b69d5e55601f8c32da8f6f8efc72300",
}


def fetch_levyholt(directory: pathlib.Path = RAW / "levyholt") -> None:
    """Download the Levy/Holt files from entgraph_eval at LEVYHOLT_COMMIT into directory.

    Each file is checked against its SHA-256 before it is written, and written
    whole through a temporary name, so a failed or altered download leaves no
    file that levyholt_rows would read.
    """
    import urllib.request

    directory.mkdir(parents=True, exist_ok=True)
    for path, digest in LEVYHOLT_FILES.items():
        url = f"https://raw.githubusercontent.com/{LEVYHOLT_REPOSITORY}/{LEVYHOLT_COMMIT}/{path}"
        with urllib.request.urlopen(url) as response:
            data = response.read()
        got = hashlib.sha256(data).hexdigest()
        if got != digest:
            msg = f"{url} has SHA-256 {got}, not the pinned {digest}"
            raise ValueError(msg)
        target = directory / pathlib.PurePosixPath(path).name
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(data)
        partial.replace(target)


def _normal(argument: str) -> str:
    return _DETERMINERS.sub("", argument.strip().lower().replace("_", " "))


def _same_head(a: str, b: str) -> bool:
    """Whether two words share a stem: a common prefix of at least 3 letters and 3/4 of the shorter."""
    shared = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            break
        shared += 1
    return shared >= max(3, 0.75 * min(len(a), len(b)))


def _match(a: str, b: str) -> int:
    """How surely two argument strings name one thing: 3 equal, 2 same head noun, 1 a shared word, 0 none."""
    na, nb = _normal(a), _normal(b)
    if na == nb:
        return 3
    ta, tb = na.split(), nb.split()
    if not ta or not tb:
        return 0
    if _same_head(ta[-1], tb[-1]):
        return 2
    return 1 if any(len(u) >= 3 and _same_head(u, v) for u in ta for v in tb) else 0


def _triple(cell: str, where: str) -> tuple[str, str, str]:
    parts = cell.split(",")
    if len(parts) != 3 or not all(p.strip() for p in parts):
        msg = f"{where}: expected subject,predicate,object, got {cell!r}"
        raise ValueError(msg)
    subject, predicate, obj = (p.strip() for p in parts)
    if "{" in predicate or "}" in predicate:
        msg = f"{where}: braces in predicate {predicate!r} would break its gloss"
        raise ValueError(msg)
    return subject, predicate, obj


def _swapped(hypothesis: tuple[str, str, str], premise: tuple[str, str, str], where: str) -> bool:
    """Whether the premise states the hypothesis's arguments in reverse order."""
    same = _match(hypothesis[0], premise[0]) + _match(hypothesis[2], premise[2])
    swap = _match(hypothesis[0], premise[2]) + _match(hypothesis[2], premise[0])
    if same == swap:
        msg = f"{where}: cannot tell argument order of {premise} against {hypothesis}"
        raise ValueError(msg)
    return swap > same


def _type_noun(name: str) -> str:
    return _TYPE_NOUNS.get(name, name.replace("_", " "))


def _levyholt_types(cell: str, hypothesis: tuple[str, str, str]) -> Fillers:
    """The hypothesis subject's and object's type nouns, read from its _rels cell.

    The cell's argument order usually follows the surface subject, but not
    always, so its lemmas are matched to the surface arguments; an empty cell,
    which the test split has, types both as "thing", the root type.
    """
    found = _RELS_CELL.match(cell.strip())
    if not found:
        return "thing", "thing"
    a1, a2 = found["a1"], found["a2"]
    t1, t2 = _type_noun(found["t1"]), _type_noun(found["t2"])
    subject, _, obj = hypothesis
    if _match(a1, obj) + _match(a2, subject) > _match(a1, subject) + _match(a2, obj):
        return t2, t1
    return t1, t2


def levyholt_rows(raw: pathlib.Path = RAW / "levyholt") -> list[PairRow]:
    """The directional Levy/Holt dev and test rows, from the _s sentences and _rels types.

    Rows read hypothesis TAB premise TAB label, True meaning the premise entails
    the hypothesis; row 2k+1 is row 2k reversed, which `check` confirms. A pair's
    two rows mention the same two typed entities: types and instances come from
    row 2k and are exchanged for row 2k+1 when its arguments are swapped.
    """
    rows: list[PairRow] = []
    missing = [name for name in (pathlib.PurePosixPath(p).name for p in LEVYHOLT_FILES) if not (raw / name).exists()]
    if missing:
        msg = f"{raw} lacks {', '.join(missing)}: run python -m semrewrite.pairs fetch levyholt"
        raise FileNotFoundError(msg)
    for split, stem in (("dev", "dev_dir"), ("test", "test_dir")):
        sentences = (raw / f"{stem}_s.txt").read_text().splitlines()
        rels = (raw / f"{stem}_rels.txt").read_text().splitlines()
        if len(sentences) != len(rels) or len(sentences) % 2:
            msg = f"{stem}: {len(sentences)} sentence rows against {len(rels)} relation rows, or an odd count"
            raise ValueError(msg)
        types: Fillers = ("thing", "thing")
        instances: Fillers = ("", "")
        for i, (line, rel) in enumerate(zip(sentences, rels, strict=True)):
            where = f"{stem}_s.txt:{i + 1}"
            hyp_cell, prem_cell, label = line.split("\t")
            if label not in ("True", "False"):
                msg = f"{where}: label {label!r}"
                raise ValueError(msg)
            hypothesis = _triple(hyp_cell, where)
            premise = _triple(prem_cell, where)
            swapped = _swapped(hypothesis, premise, where)
            if i % 2 == 0:
                types = _levyholt_types(rel.split("\t")[0], hypothesis)
                instances = (hypothesis[0], hypothesis[2])
            elif swapped:
                types, instances = (types[1], types[0]), (instances[1], instances[0])
            rows.append(
                PairRow(
                    id=f"levyholt:{split}:{i:04d}",
                    set="levyholt",
                    split=split,
                    pair=f"levyholt:{split}:{i // 2:04d}",
                    kind="levyholt",
                    premise=Predicate(premise[1], f"{{x}} {premise[1]} {{y}}"),
                    conclusion=Predicate(hypothesis[1], f"{{x}} {hypothesis[1]} {{y}}"),
                    swapped=swapped,
                    label=label == "True",
                    types=types,
                    instances=instances,
                    source=f"entgraph_eval@e5652b3f LevyHoltDS/{stem}/{where}",
                )
            )
    return rows


# ---------------------------------------------------------------- hand-labelled hard set

_HOLDS = {"forward": (True, False), "both": (True, True), "neither": (False, False)}


def hard_rows(source: pathlib.Path = PAIRS / "hard.toml") -> list[PairRow]:
    """The hard set's rows, all test: each relation's forward row and its reverse.

    A relation from a predicate to itself, swapped, asks whether the predicate is
    symmetric and is its own reverse, so it yields one row. Both rows of a
    relation are typed by its conclusion predicate, exchanged for the reverse
    row when swapped, so they mention the same two entities.
    """
    spec = tomllib.loads(source.read_text())
    predicates = {
        name: (Predicate(name, entry["gloss"]), (entry["types"][0], entry["types"][1]))
        for name, entry in spec["predicates"].items()
    }
    rows: list[PairRow] = []
    for n, relation in enumerate(spec["relations"]):
        premise, _ = predicates[relation["premise"]]
        conclusion, types = predicates[relation["conclusion"]]
        swapped = bool(relation.get("swapped", False))
        forward, backward = _HOLDS[relation["holds"]]
        pair = f"hard:{premise.name}>{conclusion.name}{'~' if swapped else ''}"
        common = {"set": "hard", "split": "test", "pair": pair, "kind": relation["holds"],
                  "source": f"data/pairs/hard.toml relation {n + 1}", "reason": relation["reason"]}
        rows.append(PairRow(id=pair, premise=premise, conclusion=conclusion, swapped=swapped,
                            label=forward, types=types, **common))
        if premise != conclusion or not swapped:
            reverse_types = (types[1], types[0]) if swapped else types
            rows.append(PairRow(id=pair + ":reverse", premise=conclusion, conclusion=premise, swapped=swapped,
                                label=backward, types=reverse_types, **common))
    return rows


# ---------------------------------------------------------------- Wikidata property hierarchy

# The SPARQL queries whose JSON results data/raw/wikidata/<name>.json holds;
# `python -m semrewrite.pairs fetch wikidata` reruns them. Only item-valued properties
# take part, since only they relate two entities.
SPARQL = {
    "properties": """SELECT ?p ?label ?description WHERE {
  ?p wikibase:propertyType wikibase:WikibaseItem .
  ?p rdfs:label ?label . FILTER(LANG(?label) = "en")
  OPTIONAL { ?p schema:description ?description . FILTER(LANG(?description) = "en") }
}""",
    "subproperty": """SELECT ?p ?super WHERE {
  ?p wdt:P1647 ?super .
  ?p wikibase:propertyType wikibase:WikibaseItem .
  ?super wikibase:propertyType wikibase:WikibaseItem .
}""",
    "inverse": """SELECT ?p ?inverse WHERE {
  ?p wdt:P1696 ?inverse .
  ?p wikibase:propertyType wikibase:WikibaseItem .
  ?inverse wikibase:propertyType wikibase:WikibaseItem .
}""",
    "symmetric": """SELECT ?p ?kind WHERE {
  VALUES ?kind { wd:Q18647518 wd:Q18647519 }
  ?p wdt:P31 ?kind .
  ?p wikibase:propertyType wikibase:WikibaseItem .
}""",
    "types": """SELECT ?p ?kind ?class ?classLabel WHERE {
  VALUES ?kind { wd:Q21503250 wd:Q21510865 }
  ?p wikibase:propertyType wikibase:WikibaseItem .
  ?p p:P2302 ?st . ?st ps:P2302 ?kind .
  ?st pq:P2308 ?class .
  OPTIONAL { ?class rdfs:label ?classLabel . FILTER(LANG(?classLabel) = "en") }
}""",
}
SYMMETRIC, ASYMMETRIC = "Q18647518", "Q18647519"
SUBJECT_TYPE, VALUE_TYPE = "Q21503250", "Q21510865"
DEV_SHARE = 0.3


def fetch_wikidata(directory: pathlib.Path = RAW / "wikidata") -> None:
    """Rerun every SPARQL query against the Wikidata endpoint, overwriting the raw results."""
    import urllib.parse
    import urllib.request

    directory.mkdir(parents=True, exist_ok=True)
    for name, query in SPARQL.items():
        url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query})
        request = urllib.request.Request(url, headers={
            "Accept": "application/sparql-results+json", "User-Agent": "SemanticRewriter-experiment/0.1"})
        with urllib.request.urlopen(request) as response:
            (directory / f"{name}.json").write_bytes(response.read())


def _bindings(directory: pathlib.Path, name: str) -> list[dict[str, str]]:
    data = json.loads((directory / f"{name}.json").read_text())
    return [{k: v["value"] for k, v in b.items()} for b in data["results"]["bindings"]]


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


# Function words that end a Wikidata label which predicates its subject ("part of", "followed by").
_PREPOSITIONS = frozenset(
    "of by in on at to for from with after into as about under over against between through than via per".split()
)
# Finite verbs that open a label whose subject acts ("depicts", "contains settlement").
_VERBS = frozenset(
    "advertises carries cites commemorates computes connects contains depicts emulates encodes follows grants "
    "innervates interacts overlies replaces repeals represents simulates suggests symbolizes translates "
    "underlies uses wears gained gave participated contributed destroyed".split()
)


def rule_gloss(label: str) -> str:
    """A gloss over subject {x} and value {y} read off a property label's grammar.

    >>> [rule_gloss(l) for l in ("mother", "part of", "has part(s)", "depicts", "located in protected area")]
    ['{y} is the mother of {x}', '{x} is part of {y}', '{x} has part(s) {y}', '{x} depicts {y}', '{x} is located in protected area {y}']
    """
    words = label.split()
    if words[0] in ("has", "have", "is", "are", "was", "were") or words[0] in _VERBS:
        return f"{{x}} {label} {{y}}"
    if words[-1] in _PREPOSITIONS or (words[0].endswith("ed") and any(w in _PREPOSITIONS for w in words[1:])):
        return f"{{x}} is {label} {{y}}"
    return f"{{y}} is the {label} of {{x}}"


def _reach(edges: dict[tuple[str, bool], set[tuple[str, bool]]]) -> dict[tuple[str, bool], set[tuple[str, bool]]]:
    """Every node's reachable set, itself included, by breadth-first search from each node."""
    reach = {}
    for start in edges:
        seen = {start}
        queue = deque([start])
        while queue:
            for nxt in edges[queue.popleft()]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        reach[start] = seen
    return reach


def _split(pair: str) -> str:
    share = int(hashlib.sha1(pair.encode()).hexdigest()[:8], 16) / 0x100000000
    return "dev" if share < DEV_SHARE else "test"


def _rank(key: str) -> str:
    """A stable pseudo-random order, the same on every machine."""
    return hashlib.sha1(key.encode()).hexdigest()


def wikidata_rows(
    raw: pathlib.Path = RAW / "wikidata",
    overrides: pathlib.Path = PAIRS / "wikidata.toml",
    near_model: str | None = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[PairRow]:
    """Rows over Wikidata's item-valued property hierarchy, labelled by one rule.

    Each property p has two literals, p(x, y) and p(y, x). Subproperty-of a ⊂ b
    gives a(x, y) → b(x, y) and a(y, x) → b(y, x); inverse-property p ~ q gives
    p(x, y) → q(y, x) when q is p's only inverse, since a property Wikidata lists
    with several inverses (facet of: history of topic, economy of topic) has a
    converse broader than each of them, and reading every inverse as exact would
    make history of topic equivalent to economy of topic; a symmetric property s
    gives s(x, y) ↔ s(y, x). A row
    (P, Q, swapped) is labelled True exactly when Q(x, y) is reachable from
    P(y, x) if swapped, else from P(x, y), so every candidate kind is labelled
    the same way and no two rows can disagree. Absence of a path reads as no
    entailment, which is where Wikidata's missing edges become label noise; the
    hand audit in data/pairs/wikidata.toml corrects the ones found.

    Candidates, each with its reverse:
      sub           every direct subproperty edge, the one-way pairs
      swap-order    every direct edge with the premise's arguments exchanged
      inverse       every inverse pair, swapped
      inverse-order every inverse pair, unswapped
      lifted        a subproperty edge carried through an inverse at either end
      sibling       two subproperties of one superproperty, each sibling paired
                    with the next in a stable order, so a superproperty with k
                    children gives k pairs (one when k is 2)
      near          each property's nearest unrelated property by gloss embedding
                    under near_model, kept when at least as similar as the
                    median direct edge; none when near_model is None
      self          a property against itself swapped, for properties Wikidata
                    marks symmetric, asymmetric or self-inverse; one row
    """
    spec = tomllib.loads(overrides.read_text()) if overrides.is_file() else {}
    gloss_override: dict[str, str] = spec.get("glosses", {})
    excluded: dict[str, str] = spec.get("exclude", {})

    labels = {
        _qid(r["p"]): r["label"] for r in _bindings(raw, "properties") if "DEPRECATED" not in r["label"]
    }
    labels = {p: label for p, label in labels.items() if p not in excluded}
    subs = sorted({(_qid(r["p"]), _qid(r["super"])) for r in _bindings(raw, "subproperty")})
    subs = [(a, b) for a, b in subs if a != b and a in labels and b in labels]
    inverses = sorted({tuple(sorted((_qid(r["p"]), _qid(r["inverse"])))) for r in _bindings(raw, "inverse")})
    inverses = [(p, q) for p, q in inverses if p in labels and q in labels]
    marked = {_qid(r["p"]): _qid(r["kind"]) for r in _bindings(raw, "symmetric") if _qid(r["p"]) in labels}
    symmetric = {p for p, kind in marked.items() if kind == SYMMETRIC} | {p for p, q in inverses if p == q}

    used = sorted({p for e in subs for p in e} | {p for e in inverses for p in e} | set(marked))
    glosses = {p: gloss_override.get(p, rule_gloss(labels[p])) for p in used}
    predicate = {p: Predicate(f"{p}:{labels[p]}", glosses[p]) for p in used}

    # Typed nouns: the class each type constraint names, picking the class most
    # used across all constraints when a property allows several.
    constraints = _bindings(raw, "types")
    class_label = {_qid(r["class"]): r.get("classLabel", "thing") for r in constraints}
    popularity = Counter(_qid(r["class"]) for r in constraints)
    allowed: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in constraints:
        allowed[(_qid(r["p"]), _qid(r["kind"]))].append(_qid(r["class"]))

    def noun(p: str, kind: str) -> str:
        options = allowed.get((p, kind))
        if not options:
            return "thing"
        best = min(options, key=lambda q: (-popularity[q], len(q), q))
        name = class_label[best]
        return "person" if name == "human" else name

    types = {p: (noun(p, SUBJECT_TYPE), noun(p, VALUE_TYPE)) for p in used}

    # Literal (p, flipped) stands for p(y, x) when flipped, else p(x, y).
    graph: dict[tuple[str, bool], set[tuple[str, bool]]] = {(p, f): set() for p in used for f in (False, True)}
    for a, b in subs:
        for f in (False, True):
            graph[(a, f)].add((b, f))
    partners: dict[str, set[str]] = defaultdict(set)
    for p, q in inverses:
        partners[p].add(q)
        partners[q].add(p)
    for p, others in partners.items():
        if len(others) == 1:
            (q,) = others
            for f in (False, True):
                graph[(p, f)].add((q, not f))
    for s in symmetric:
        for f in (False, True):
            graph[(s, f)].add((s, not f))
    reach = _reach(graph)

    def holds(premise: str, conclusion: str, swapped: bool) -> bool:
        return (conclusion, False) in reach[(premise, swapped)]

    candidates: dict[tuple[frozenset[str], bool], tuple[str, str, str]] = {}

    def offer(kind: str, premise: str, conclusion: str, swapped: bool) -> None:
        # P(x, y) entailing P(x, y) holds trivially and never reaches a judge.
        if premise != conclusion or swapped:
            candidates.setdefault((frozenset((premise, conclusion)), swapped), (kind, premise, conclusion))

    for a, b in subs:
        offer("sub", a, b, False)
    for a, b in subs:
        offer("swap-order", a, b, True)
    inverse_of = {}
    for p, q in inverses:
        if p != q:
            offer("inverse", p, q, True)
            offer("inverse-order", p, q, False)
            inverse_of[p], inverse_of[q] = q, p
    for a, b in subs:
        if b in inverse_of:
            offer("lifted", a, inverse_of[b], True)
        if a in inverse_of:
            offer("lifted", inverse_of[a], b, True)
        if a in inverse_of and b in inverse_of:
            offer("lifted", inverse_of[a], inverse_of[b], False)
    children: dict[str, list[str]] = defaultdict(list)
    for a, b in subs:
        children[b].append(a)
    for b, kids in sorted(children.items()):
        ring = sorted(kids, key=lambda a: _rank(f"{b}>{a}"))
        for i in range(len(ring) if len(ring) > 2 else len(ring) - 1):
            offer("sibling", ring[i], ring[(i + 1) % len(ring)], False)
    for p in sorted(set(marked) | {p for p, q in inverses if p == q}):
        offer("self", p, p, True)

    related = {p: {q for (q, _) in reach[(p, False)] | reach[(p, True)]} for p in used}
    for p in used:
        for q in used:
            if p in related[q]:
                related[p].add(q)
    for b, kids in children.items():
        for a in kids:
            related[a].update(kids)
    if near_model is not None:
        for p, q, _, _ in _near_pairs(used, predicate, related, subs, near_model):
            offer("near", p, q, False)

    label_override = {(o["premise"], o["conclusion"], bool(o.get("swapped", False))): o
                      for o in spec.get("labels", [])}
    applied: set[tuple[str, str, bool]] = set()
    rows = []
    for (_, swapped), (kind, premise, conclusion) in sorted(candidates.items(), key=lambda kv: _rank(repr(kv[1]) + repr(kv[0][1]))):
        pair = f"wikidata:{min(premise, conclusion)}~{max(premise, conclusion)}{':swapped' if swapped else ''}"
        split = _split(pair)
        pair_types = types[conclusion]
        directions = [(premise, conclusion, pair_types)]
        if premise != conclusion:
            directions.append((conclusion, premise, (pair_types[1], pair_types[0]) if swapped else pair_types))
        for p, q, t in directions:
            fixed = label_override.get((p, q, swapped))
            if fixed:
                applied.add((p, q, swapped))
            label = bool(fixed["label"]) if fixed else holds(p, q, swapped)
            rows.append(PairRow(
                id=f"{pair}:{p}>{q}", set="wikidata", split=split, pair=pair, kind=kind,
                premise=predicate[p], conclusion=predicate[q], swapped=swapped, label=label, types=t,
                source=f"wikidata {p} {q}", reason=fixed["reason"] if fixed else ""))
    unused = sorted(set(label_override) - applied)
    if unused:
        msg = f"{overrides}: label corrections for rows the build no longer makes: {unused}"
        raise ValueError(msg)
    stale = sorted(set(gloss_override) - set(used))
    if stale:
        msg = f"{overrides}: glosses for properties outside the pair set: {stale}"
        raise ValueError(msg)
    return rows


def _near_pairs(
    used: Sequence[str],
    predicate: dict[str, Predicate],
    related: dict[str, set[str]],
    subs: Sequence[tuple[str, str]],
    model: str,
) -> list[tuple[str, str, float, float]]:
    """Each property's most similar unrelated property, kept if at least as similar as the median direct edge.

    Time: one embedding pass over |used| glosses, then |used|^2 dot products.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model)
    vectors = encoder.encode([predicate[p].sentence("A", "B") for p in used], normalize_embeddings=True)
    del encoder
    index = {p: i for i, p in enumerate(used)}
    similarity = vectors @ vectors.T
    floor = float(np.median([similarity[index[a], index[b]] for a, b in subs]))
    kept = []
    for p in used:
        order = np.argsort(-similarity[index[p]])
        for j in order:
            q = used[j]
            if q != p and q not in related[p]:
                if similarity[index[p], j] >= floor:
                    kept.append((p, q, float(similarity[index[p], j]), floor))
                break
    return kept


# ---------------------------------------------------------------- command line


def build(names: Iterable[str] = SETS, directory: pathlib.Path = PAIRS) -> dict[str, int]:
    """Rebuild the named sets' JSON lines from their sources; returns the row count per set."""
    counts = {}
    for name in names:
        rows = BUILDERS[name]()
        check(rows)
        save(rows, directory / f"{name}.jsonl")
        counts[name] = len(rows)
    return counts


BUILDERS = {"levyholt": levyholt_rows, "wikidata": wikidata_rows, "hard": hard_rows}
FETCH = {"levyholt": fetch_levyholt, "wikidata": fetch_wikidata}

if __name__ == "__main__":
    if sys.argv[1:2] == ["build"] and set(sys.argv[2:]) <= set(BUILDERS):
        print(json.dumps(build(sys.argv[2:] or SETS), indent=1))
    elif len(sys.argv) == 3 and sys.argv[1] == "fetch" and sys.argv[2] in FETCH:
        FETCH[sys.argv[2]]()
    else:
        sys.exit(f"usage: python -m semrewrite.pairs build [{' '.join(BUILDERS)}] | fetch {'|'.join(FETCH)}")
