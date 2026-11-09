"""Purpose: build the Wikidata examples' knowledge bases from Wikidata's public SPARQL endpoint. For each question in
questions.py defines it writes examples/wikidata/<name>.metta: the question's own predicates, rules and query from
<name>.question.metta, every entity-valued Wikidata property as a glossed predicate, and every entity-valued statement
Wikidata holds about the question's entities. Those are the subject, Wikidata's own answer to the question, as its
endpoint computes it from the question's properties, and for a question with mentions everything naming the subject.
The build fails unless the statements it fetched reach exactly that answer from the subject, so the knowledge base
holds all of it. Beside it, <name>.ids.tsv gives each name's Wikidata id and English label, so any answer or
derivation can be checked on wikidata.org. Wikidata's data is CC0.

Run from the repository root: python examples/wikidata/build.py [NAME ...]

Names are readable: a property becomes its English label in snake case with its id, child_p40, so a derivation names
the property it used; an entity becomes its label, queen_victoria, with its id added to every entity whose label it
shares, and its id alone when it has no English label.
"""

from __future__ import annotations

import http.client
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]

from questions import QUESTIONS, Question

from semrewrite.pairs import rule_gloss

ENDPOINT = "https://query.wikidata.org/sparql"
# Wikidata asks automated clients to name themselves and a way to reach their author.
USER_AGENT = "semantic-rewriter-examples/0.1 (https://github.com/MesTTo/semantic-rewriter)"
# Entities per VALUES query: small enough that a query over full records stays inside the endpoint's 60 s limit.
BATCH = 150
# An entity-valued statement's property, as the queries below select it.
ITEM_PROPERTY = "?prop wikibase:directClaim ?p ; wikibase:propertyType wikibase:WikibaseItem ."


def sparql(query: str) -> list[dict[str, str]]:
    """The query's rows from the Wikidata endpoint, each a dict of plain values; waits out rate limits and retries
    a server error up to four times. A response the endpoint cut short, at its 60 s limit, raises ValueError."""
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    request = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=150) as response:
                rows = json.load(response)["results"]["bindings"]
            return [{k: v["value"] for k, v in row.items()} for row in rows]
        except (json.JSONDecodeError, http.client.IncompleteRead) as e:
            msg = f"the endpoint cut its response short: {e}"
            raise ValueError(msg) from e
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            time.sleep(float(e.headers.get("Retry-After", 10 * (attempt + 1))))
    msg = "unreachable"
    raise AssertionError(msg)


def _id(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


# Time: one query for the answer and, for a question with mentions, one for what names the subject.
def entities(question: Question) -> tuple[list[str], set[str]]:
    """The entities whose statements the question's knowledge base holds, and Wikidata's own answer as its endpoint
    computes it: what the question's properties reach from the subject in one step, or in one or more."""
    subject = question.subject[0]
    path = "|".join(f"^wdt:{p}" if flipped else f"wdt:{p}" for p, flipped in sorted(question.properties().items()))
    steps = "+" if question.transitive else ""
    answer = {_id(r["e"]) for r in sparql(f"SELECT DISTINCT ?e WHERE {{ wd:{subject} ({path}){steps} ?e . }}")}
    held = answer | {subject}
    if question.mentions:
        held |= {_id(r["e"]) for r in sparql(f"SELECT DISTINCT ?e WHERE {{ ?e ?p wd:{subject} . {ITEM_PROPERTY} }}")}
    return sorted(held), answer


def slug(label: str) -> str:
    """A label as a knowledge-base name: ASCII letters and digits in lower snake case, starting with a letter.

    >>> slug("Queen Victoria"), slug("Château de Chambord"), slug("3M")
    ('queen_victoria', 'chateau_de_chambord', 'n3m')
    """
    ascii_label = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", "_", ascii_label.lower()).strip("_")
    return text if text[:1].isalpha() else "n" + text


def properties() -> dict[str, tuple[str, str]]:
    """Every entity-valued property: its id, English label and English description."""
    rows = sparql('SELECT ?p ?label ?description WHERE { ?p wikibase:propertyType wikibase:WikibaseItem . '
                  '?p rdfs:label ?label . FILTER(LANG(?label) = "en") '
                  'OPTIONAL { ?p schema:description ?description . FILTER(LANG(?description) = "en") } }')
    return {_id(r["p"]): (r["label"], r.get("description", "")) for r in rows}


# Time: one query per BATCH entities, and one more for each half of a batch the endpoint cut short.
def statements(entities: list[str]) -> list[tuple[str, str, str]]:
    """Every entity-valued statement whose subject is one of `entities`, as (subject, property, value) ids."""
    out: list[tuple[str, str, str]] = []
    for start in range(0, len(entities), BATCH):
        out += _statements(entities[start:start + BATCH])
        print(f"  statements {min(start + BATCH, len(entities))}/{len(entities)}: {len(out)}", file=sys.stderr)
    return sorted(set(out))


def _statements(batch: list[str]) -> list[tuple[str, str, str]]:
    """One batch's statements, split in half while the endpoint cuts the response short."""
    values = " ".join(f"wd:{e}" for e in batch)
    try:
        rows = sparql(f"SELECT ?s ?p ?o WHERE {{ VALUES ?s {{ {values} }} ?s ?p ?o . {ITEM_PROPERTY} "
                      "FILTER(STRSTARTS(STR(?o), 'http://www.wikidata.org/entity/Q')) }")
    except ValueError:
        if len(batch) == 1:
            raise
        half = len(batch) // 2
        return _statements(batch[:half]) + _statements(batch[half:])
    return [(_id(r["s"]), _id(r["p"]), _id(r["o"])) for r in rows]


# Time: one query per 400 entities.
def labels(entities: set[str]) -> dict[str, str]:
    """Each entity's English label, where it has one."""
    out: dict[str, str] = {}
    ordered = sorted(entities)
    for start in range(0, len(ordered), 400):
        values = " ".join(f"wd:{e}" for e in ordered[start:start + 400])
        rows = sparql(f'SELECT ?e ?label WHERE {{ VALUES ?e {{ {values} }} ?e rdfs:label ?label . '
                      'FILTER(LANG(?label) = "en") }')
        out.update((_id(r["e"]), r["label"]) for r in rows)
    return out


def names(labelled: dict[str, str], entities: set[str], pinned: tuple[str, str]) -> dict[str, str]:
    """A name per entity: its label's slug, with the id added for every entity sharing that slug, else its id; the
    pinned entity takes the name the question gives it, and an entity whose slug that is gets its id added."""
    slugs = {e: slug(labelled[e]) for e in entities if e in labelled and slug(labelled[e]) != "n"}
    shared = Counter(slugs.values())
    shared[pinned[1]] += 2
    out = {e: (f"{slugs[e]}_{e.lower()}" if shared[slugs[e]] > 1 else slugs[e]) if e in slugs else e.lower()
           for e in entities}
    out[pinned[0]] = pinned[1]
    return out


def build(question: Question, catalogue: dict[str, tuple[str, str]]) -> None:
    """Fetch the question's statements and write its knowledge base and id table, once the statements are found to
    reach Wikidata's own answer exactly."""
    held, answer = entities(question)
    print(f"{question.name}: {len(held)} entities, {len(answer)} in Wikidata's answer", file=sys.stderr)
    facts = [(s, p, o) for s, p, o in statements(held) if p in catalogue]
    reached = question.answer(facts, question.subject[0])
    if reached != answer:
        msg = (f"{question.name}: the statements fetched reach {len(reached):,} entities from the subject, where "
               f"Wikidata's endpoint answers {len(answer):,}; they differ on {sorted(reached ^ answer)[:10]}")
        raise SystemExit(msg)
    mentioned = {s for s, _, _ in facts} | {o for _, _, o in facts}
    found = labels(mentioned)
    named = names(found, mentioned, question.subject)
    predicate = {p: f"{slug(label)}_{p.lower()}" for p, (label, _) in catalogue.items()}
    used = Counter(p for _, p, _ in facts)
    asked = (HERE / f"{question.name}.question.metta").read_text().rstrip()
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    lines = [
        f"; Built by examples/wikidata/build.py from Wikidata (CC0, wikidata.org) on {stamp}: every entity-valued",
        f"; property, {len(catalogue):,} of them, glossed from its English label, and the {len(facts):,} entity-valued",
        f"; statements Wikidata holds about {len(held):,} entities, in {len(used):,} of those properties. Wikidata's own",
        f"; answer to the question, as examples/wikidata/questions.py defines it, has {len(answer):,} entities.",
        f"; {question.name}.ids.tsv gives each name's Wikidata id. The question comes first.",
        "",
        asked,
        "",
        "; The catalogue: every entity-valued Wikidata property, with its id in its name.",
        *(f'(gloss {predicate[p]} {json.dumps(rule_gloss(label), ensure_ascii=False)})'
          for p, (label, _) in sorted(catalogue.items(), key=lambda kv: int(kv[0][1:]))),
        "",
        "; What Wikidata says about the question's entities.",
        *(f"(= ({predicate[p]} {named[s]} {named[o]}) True)" for s, p, o in facts),
    ]
    (HERE / f"{question.name}.metta").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (HERE / f"{question.name}.ids.tsv").open("w", encoding="utf-8") as table:
        table.write("name\twikidata\tlabel\n")
        for p in sorted(used, key=lambda p: int(p[1:])):
            table.write(f"{predicate[p]}\t{p}\t{catalogue[p][0]}\n")
        for e in sorted(mentioned, key=lambda e: int(e[1:])):
            table.write(f"{named[e]}\t{e}\t{' '.join(found.get(e, '').split())}\n")
    print(f"{question.name}: wrote {len(facts):,} statements over {len(mentioned):,} entities", file=sys.stderr)


def main(chosen: list[str]) -> None:
    unknown = sorted(set(chosen) - set(QUESTIONS))
    if unknown:
        msg = f"no question is named {', '.join(unknown)}; the questions are {', '.join(QUESTIONS)}"
        raise SystemExit(msg)
    missing = [n for n in chosen or QUESTIONS if not (HERE / f"{n}.question.metta").is_file()]
    if missing:
        msg = f"write {', '.join(f'{n}.question.metta' for n in missing)} in {HERE} before fetching its statements"
        raise SystemExit(msg)
    catalogue = properties()
    print(f"{len(catalogue):,} entity-valued properties", file=sys.stderr)
    for name in chosen or QUESTIONS:
        build(QUESTIONS[name], catalogue)


if __name__ == "__main__":
    main(sys.argv[1:])
