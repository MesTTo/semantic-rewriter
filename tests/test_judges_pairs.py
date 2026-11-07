"""Purpose: tests of the labelled predicate pairs: each committed set rebuilds
from its source unchanged, rows pair with their reverse inside one split, the
Levy/Holt rows keep the published sizes and balance, the hard set follows its
relations, and the Wikidata labelling rule decides a small hand-built
hierarchy the way its definition says.
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter, defaultdict

import pytest

from semrewrite import pairs
from semrewrite.model import Predicate

ROWS = {name: pairs.load([name]) for name in pairs.SETS}


@pytest.mark.parametrize("name", pairs.SETS)
def test_committed_set_rebuilds_from_its_source(name: str) -> None:
    """data/pairs/<set>.jsonl is exactly what the builder makes from data/raw and the hand-written sources."""
    build = {"levyholt": pairs.levyholt_rows, "wikidata": pairs.wikidata_rows, "hard": pairs.hard_rows}[name]
    assert [r.to_json() for r in build()] == [r.to_json() for r in ROWS[name]]


@pytest.mark.parametrize("name", pairs.SETS)
def test_rows_round_trip_and_pair_with_their_reverse(name: str) -> None:
    rows = ROWS[name]
    pairs.check(rows)
    assert all(pairs.PairRow.from_json(r.to_json()) == r for r in rows)
    groups: dict[str, list[pairs.PairRow]] = defaultdict(list)
    for r in rows:
        groups[r.pair].append(r)
    for members in groups.values():
        assert len({m.split for m in members}) == 1
        if len(members) == 1:
            only = members[0]
            assert only.premise == only.conclusion and only.swapped


def test_levyholt_keeps_the_directional_split() -> None:
    """630 dev and 1784 test rows, half true, every pair one-way."""
    rows = ROWS["levyholt"]
    assert Counter(r.split for r in rows) == {"dev": 630, "test": 1784}
    assert Counter((r.split, r.label) for r in rows) == {("dev", True): 315, ("dev", False): 315,
                                                         ("test", True): 892, ("test", False): 892}
    groups: dict[str, list[pairs.PairRow]] = defaultdict(list)
    for r in rows:
        groups[r.pair].append(r)
    assert all(len(m) == 2 and m[0].label != m[1].label for m in groups.values())


def test_levyholt_reads_argument_order_from_the_surface() -> None:
    """dev row 5 states its premise with the arguments exchanged and keeps both sentences' entities."""
    row = next(r for r in ROWS["levyholt"] if r.id == "levyholt:dev:0004")
    assert row.swapped and row.label
    assert row.proposal.sentences(row.instances) == ("a piedmont is common to Norway", "Norway shows a piedmont")
    reverse = next(r for r in ROWS["levyholt"] if r.id == "levyholt:dev:0005")
    assert reverse.instances == (row.instances[1], row.instances[0]) and reverse.types == (row.types[1], row.types[0])


def test_typed_nouns_tell_same_typed_arguments_apart() -> None:
    assert pairs.typed_nouns(("medicine", "disease")) == ("the medicine", "the disease")
    assert pairs.typed_nouns(("person", "person")) == ("the first person", "the second person")
    row = ROWS["hard"][0]
    assert pairs.fillers_for(row, "letters") == ("A", "B")
    assert pairs.fillers_for(row, "instances") is None
    with pytest.raises(ValueError):
        pairs.fillers_for(row, "names")


def test_hard_rows_follow_their_relations() -> None:
    """forward is true one way, both is true both ways, neither is false both ways; a symmetric check is one row."""
    rows = ROWS["hard"]
    groups: dict[str, list[pairs.PairRow]] = defaultdict(list)
    for r in rows:
        groups[r.pair].append(r)
    for members in groups.values():
        labels = {m.id.endswith(":reverse"): m.label for m in members}
        kind = members[0].kind
        if kind == "forward":
            assert labels == {False: True, True: False}
        else:
            assert set(labels.values()) == {kind == "both"}
    assert len(rows) >= 120
    assert {r.split for r in rows} == {"test"}
    mother = next(r for r in rows if r.id == "hard:mother>parent")
    assert mother.label and mother.proposal.sentences() == ("A is the mother of B", "A is a parent of B")


def _write(directory: pathlib.Path, name: str, rows: list[dict[str, str]]) -> None:
    entity = "http://www.wikidata.org/entity/"
    bindings = [{k: {"type": "uri" if v.startswith(("P", "Q")) and k != "label" else "literal",
                     "value": entity + v if v.startswith(("P", "Q")) and k != "label" else v}
                 for k, v in row.items()} for row in rows]
    (directory / f"{name}.json").write_text(json.dumps({"results": {"bindings": bindings}}))


def test_wikidata_labels_follow_reachability(tmp_path: pathlib.Path) -> None:
    """mother ⊂ parent and father ⊂ parent, parent ~ child as sole inverses, spouse symmetric,
    and employer listed as the inverse of both employee and intern.

    Reachability makes mother entail parent and not the reverse, mother(y, x)
    entail child(x, y) through the inverse, child and parent entail each other
    swapped, and spouse entail itself swapped. Employer has two inverse
    partners, so its converse is broader than each: employee(y, x) entails
    employer(x, y), but employer(y, x) does not entail employee(x, y).
    """
    _write(tmp_path, "properties", [{"p": p, "label": label} for p, label in
                                    (("P1", "mother"), ("P2", "parent"), ("P3", "child"), ("P4", "father"),
                                     ("P5", "spouse"), ("P7", "employer"), ("P8", "employee"), ("P9", "intern"))])
    _write(tmp_path, "subproperty", [{"p": "P1", "super": "P2"}, {"p": "P4", "super": "P2"}])
    _write(tmp_path, "inverse", [{"p": "P2", "inverse": "P3"}, {"p": "P3", "inverse": "P2"},
                                 {"p": "P8", "inverse": "P7"}, {"p": "P9", "inverse": "P7"}])
    _write(tmp_path, "symmetric", [{"p": "P5", "kind": pairs.SYMMETRIC}])
    _write(tmp_path, "types", [])
    rows = pairs.wikidata_rows(tmp_path, tmp_path / "none.toml", near_model=None)
    label = {(r.premise.name.split(":")[0], r.conclusion.name.split(":")[0], r.swapped): r.label for r in rows}
    assert label[("P1", "P2", False)] and not label[("P2", "P1", False)]
    assert not label[("P1", "P2", True)] and not label[("P2", "P1", True)]
    assert label[("P1", "P3", True)] and not label[("P3", "P1", True)]
    assert label[("P3", "P2", True)] and label[("P2", "P3", True)]
    assert label[("P8", "P7", True)] and not label[("P7", "P8", True)]
    assert not label[("P1", "P4", False)] and not label[("P4", "P1", False)]
    assert label[("P5", "P5", True)]
    assert all(r.premise != r.conclusion or r.swapped for r in rows)
    kinds = Counter(r.kind for r in rows)
    assert kinds["sub"] == 4 and kinds["sibling"] == 2 and kinds["self"] == 1
    mother = next(r for r in rows if r.premise.name == "P1:mother" and r.conclusion.name == "P2:parent")
    assert mother.premise == Predicate("P1:mother", "{y} is the mother of {x}")


def test_wikidata_split_keeps_pairs_together_at_the_planned_share() -> None:
    """About 30 percent of pairs are dev, and at least 200 pairs are one-way, true in exactly one direction."""
    rows = ROWS["wikidata"]
    shares = Counter(r.split for r in {r.pair: r for r in rows}.values())
    assert 0.25 < shares["dev"] / (shares["dev"] + shares["test"]) < 0.35
    labels: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        labels[r.pair].append(r.label)
    assert sum(1 for ls in labels.values() if sorted(ls) == [False, True]) >= 200


def test_rule_gloss_reads_label_grammar() -> None:
    assert pairs.rule_gloss("mother") == "{y} is the mother of {x}"
    assert pairs.rule_gloss("part of") == "{x} is part of {y}"
    assert pairs.rule_gloss("has part(s)") == "{x} has part(s) {y}"
    assert pairs.rule_gloss("depicts") == "{x} depicts {y}"
    assert pairs.rule_gloss("located in the administrative territorial entity") == (
        "{x} is located in the administrative territorial entity {y}")
