"""Purpose: tests of the labelled predicate pairs: each set rebuilds from its
source unchanged, rows pair with their reverse inside one split, rows of any
arity keep their arguments consistent, the Levy/Holt, delta-ATOMIC and Balanced
COPA rows keep the published sizes and balance, a pinned fetch writes only
files whose digest matches its pin, the hard set follows its relations, the
typical-implication builders read their sources' labels the way their
definitions say, and the Wikidata labelling rule decides a small hand-built
hierarchy the way its definition says. A downloaded set's cases skip until
that set is fetched and built.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import pathlib
from collections import Counter, defaultdict

import pytest

from semrewrite import pairs
from semrewrite.model import Predicate


def rows_of(name: str) -> list[pairs.PairRow]:
    """The set's rows from data/pairs, skipping the test when the set has not been built."""
    try:
        return pairs.load([name])
    except FileNotFoundError as missing:
        pytest.skip(str(missing))


@pytest.mark.parametrize("name", pairs.SETS)
def test_committed_set_rebuilds_from_its_source(name: str) -> None:
    """data/pairs/<set>.jsonl is exactly what the builder makes from data/raw and the hand-written sources."""
    if name == "wikidata":
        pytest.importorskip("sentence_transformers", reason="the Wikidata builder picks near-miss pairs with MiniLM")
    built = rows_of(name)
    assert [r.to_json() for r in pairs.BUILDERS[name]()] == [r.to_json() for r in built]


@pytest.mark.parametrize("name", pairs.SETS)
def test_rows_round_trip_and_pair_with_their_reverse(name: str) -> None:
    """Rows survive JSON; a pair keeps one split; and a set either pairs every row with its reverse, but for a
    predicate that is its own reverse, or, where its source labels no reverse, leaves every row alone."""
    rows = rows_of(name)
    pairs.check(rows)
    assert all(pairs.PairRow.from_json(r.to_json()) == r for r in rows)
    groups: dict[str, list[pairs.PairRow]] = defaultdict(list)
    for r in rows:
        groups[r.pair].append(r)
    for members in groups.values():
        assert len({m.split for m in members}) == 1
    if all(len(members) == 1 for members in groups.values()):
        return
    for members in groups.values():
        if len(members) == 1:
            only = members[0]
            assert only.premise == only.conclusion and only.swapped


def test_levyholt_keeps_the_directional_split() -> None:
    """630 dev and 1784 test rows, half true, every pair one-way."""
    rows = rows_of("levyholt")
    assert Counter(r.split for r in rows) == {"dev": 630, "test": 1784}
    assert Counter((r.split, r.label) for r in rows) == {("dev", True): 315, ("dev", False): 315,
                                                         ("test", True): 892, ("test", False): 892}
    groups: dict[str, list[pairs.PairRow]] = defaultdict(list)
    for r in rows:
        groups[r.pair].append(r)
    assert all(len(m) == 2 and m[0].label != m[1].label for m in groups.values())


def test_levyholt_reads_argument_order_from_the_surface() -> None:
    """dev row 5 states its premise with the arguments exchanged and keeps both sentences' entities."""
    rows = rows_of("levyholt")
    row = next(r for r in rows if r.id == "levyholt:dev:0004")
    assert row.swapped and row.label
    assert row.proposal.sentences(row.instances) == ("a piedmont is common to Norway", "Norway shows a piedmont")
    reverse = next(r for r in rows if r.id == "levyholt:dev:0005")
    assert reverse.instances == (row.instances[1], row.instances[0]) and reverse.types == (row.types[1], row.types[0])


@pytest.mark.parametrize("name", sorted(pairs.PINS))
def test_pins_are_the_files_the_builder_read(name: str) -> None:
    """Each pinned digest is the digest of the local file the set's builder reads, when that file is present."""
    present = 0
    for local, pin in pairs.PINS[name].items():
        path = pairs.RAW / name / local
        if path.exists():
            present += 1
            assert hashlib.sha256(path.read_bytes()).hexdigest() == pin.sha256, path
    if not present:
        pytest.skip(f"data/raw/{name} is not fetched: run python -m semrewrite.pairs fetch {name}")


def test_fetch_writes_only_files_matching_their_pin(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A download whose digest matches is written under its own name; one that differs raises and writes nothing."""
    served = {"https://example.org/a": b"good\n", "https://example.org/b": b"altered\n"}
    monkeypatch.setattr("urllib.request.urlopen", lambda url: io.BytesIO(served[url]))
    good = pairs.Pin("https://example.org/a", hashlib.sha256(b"good\n").hexdigest())
    monkeypatch.setitem(pairs.PINS, "fixture", {"a.txt": good})
    pairs.fetch("fixture", tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt"]
    assert (tmp_path / "a.txt").read_bytes() == b"good\n"
    bad = pairs.Pin("https://example.org/b", hashlib.sha256(b"original\n").hexdigest())
    monkeypatch.setitem(pairs.PINS, "fixture", {"b.txt": bad})
    with pytest.raises(ValueError, match="not the pinned"):
        pairs.fetch("fixture", tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt"]


@pytest.mark.parametrize("name", sorted(pairs.PINS))
def test_a_missing_download_names_the_command_that_fetches_it(name: str, tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match=f"fetch {name}"):
        pairs.BUILDERS[name](tmp_path)
    with pytest.raises(FileNotFoundError, match=f"fetch {name}, then python -m semrewrite.pairs build {name}"):
        pairs.load([name], tmp_path)


def test_typed_nouns_tell_same_typed_arguments_apart() -> None:
    assert pairs.typed_nouns(("medicine", "disease")) == ("the medicine", "the disease")
    assert pairs.typed_nouns(("person", "person")) == ("the first person", "the second person")
    assert pairs.typed_nouns(("person",)) == ("the person",)
    assert pairs.typed_nouns(()) == ()
    assert pairs.typed_nouns(("person",) * 11)[-2:] == ("the tenth person", "the person number 11")
    row = rows_of("hard")[0]
    assert pairs.fillers_for(row, "letters") == ("A", "B")
    assert pairs.fillers_for(row, "instances") is None
    with pytest.raises(ValueError):
        pairs.fillers_for(row, "names")


def test_a_row_of_any_arity_keeps_its_arguments_consistent() -> None:
    """A row about one person reads its argument in order under each variant; mixed arities, a unary swap, or types
    of the wrong number are refused."""
    runs, tired = Predicate("runs", "{x} goes for a run"), Predicate("tired", "{x} is tired")
    row = pairs.PairRow(id="r", set="fixture", split="dev", pair="r", kind="k", premise=runs, conclusion=tired,
                        swapped=False, label=True, types=("person",))
    assert row.proposal.mapping == (0,)
    assert row.proposal.sentences(pairs.fillers_for(row, "letters")) == ("A goes for a run", "A is tired")
    assert row.proposal.sentences(pairs.fillers_for(row, "typed")) == ("the person goes for a run", "the person is tired")
    for change in ({"conclusion": Predicate("parent", "{x} is a parent of {y}")}, {"swapped": True},
                   {"types": ("person", "person")}, {"instances": ("ana", "joe")}):
        with pytest.raises(ValueError):
            pairs.PairRow(**{"id": "r", "set": "fixture", "split": "dev", "pair": "r", "kind": "k", "premise": runs,
                             "conclusion": tired, "swapped": False, "label": True, "types": ("person",), **change})


def test_delta_atomic_rows_follow_their_updates(tmp_path: pathlib.Path) -> None:
    """A strengthener's row holds and a weakener's does not; the premise is the event and then the update, closed by
    full stops; a line marked impossible is left out; braces in the source read literally."""
    lines = [
        {"Premise": "PersonX yells for PersonY", "Hypothesis": "Because PersonX wanted to be nice",
         "Update": "PersonX yells a curse at them", "UpdateType": "weakener", "UpdateTypeImpossible": False,
         "AtomicRelationType": "xIntent"},
        {"Premise": "PersonX yells for PersonY", "Hypothesis": "Because PersonX wanted to be nice",
         "Update": "PersonY is about to miss the bus.", "UpdateType": "strengthener", "UpdateTypeImpossible": False,
         "AtomicRelationType": "xIntent"},
        {"Premise": "PersonX pays", "Hypothesis": "PersonX is generous", "Update": "",
         "UpdateType": "strengthener", "UpdateTypeImpossible": True, "AtomicRelationType": "xAttr"},
    ]
    for split in ("dev", "test"):
        (tmp_path / f"{split}.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    (tmp_path / "LICENSE").write_text("MIT\n")
    rows = pairs.delta_atomic_rows(tmp_path)
    assert [(r.id, r.label) for r in rows] == [("delta-atomic:dev:00001", False), ("delta-atomic:dev:00002", True),
                                               ("delta-atomic:test:00001", False), ("delta-atomic:test:00002", True)]
    weakened = rows[0]
    assert weakened.proposal.sentences(pairs.fillers_for(weakened, "letters")) == (
        "PersonX yells for PersonY. PersonX yells a curse at them.", "Because PersonX wanted to be nice")
    assert rows[1].premise.sentence() == "PersonX yells for PersonY. PersonY is about to miss the bus."
    assert weakened.kind == "xIntent" and weakened.types == () and pairs.fillers_for(weakened, "instances") is None
    pairs.check(rows)
    lines[0]["Hypothesis"] = "PersonX wanted {nothing}"
    (tmp_path / "dev.jsonl").write_text(json.dumps(lines[0]) + "\n")
    assert pairs.delta_atomic_rows(tmp_path)[0].conclusion.sentence() == "PersonX wanted {nothing}"


def test_copa_rows_read_cause_to_effect(tmp_path: pathlib.Path) -> None:
    """An effect question reads premise => alternative and a cause question alternative => premise; the labelled
    alternative holds and the other does not; train is dev and test is test."""
    header = ["label", "id", "premise", "question", "choice1", "choice2", "mirrored"]
    files = {
        "train.csv": [["1", "1", "My body cast a shadow over the grass.", "cause", "The grass was cut.",
                       "The sun was rising.", "False"],
                      ["0", "1001", "The garden looked well-groomed.", "cause", "The grass was cut.",
                       "The sun was rising.", "True"]],
        "test.csv": [["0", "502", "The man turned on the faucet.", "effect", "Water flowed from the spout.",
                      "The toilet filled with water.", "False"]],
    }
    for name, items in files.items():
        with (tmp_path / name).open("w", newline="") as handle:
            csv.writer(handle).writerows([header, *items])
    (tmp_path / "README.md").write_text("license: cc-by-4.0\n")
    rows = {r.id: r for r in pairs.copa_rows(tmp_path)}
    holds = {key: (r.split, r.premise.sentence(), r.conclusion.sentence(), r.label) for key, r in rows.items()}
    assert holds == {
        "copa:1:1": ("dev", "The grass was cut.", "My body cast a shadow over the grass.", False),
        "copa:1:2": ("dev", "The sun was rising.", "My body cast a shadow over the grass.", True),
        "copa:1001:1": ("dev", "The grass was cut.", "The garden looked well-groomed.", True),
        "copa:1001:2": ("dev", "The sun was rising.", "The garden looked well-groomed.", False),
        "copa:502:1": ("test", "The man turned on the faucet.", "Water flowed from the spout.", True),
        "copa:502:2": ("test", "The man turned on the faucet.", "The toilet filled with water.", False),
    }
    assert rows["copa:1001:1"].kind == "cause-mirrored" and rows["copa:502:1"].kind == "effect"


def test_typical_sets_keep_the_published_sizes_and_balance() -> None:
    """delta-ATOMIC keeps 3,840 dev and 4,138 test updates, about half strengtheners (Rudinger et al. 2020, Table 2);
    Balanced COPA's 1,000 dev and 500 test questions give two rows each, one holding, and every alternative of a
    mirrored question holds under one premise and fails under the other (Kavumba et al. 2019)."""
    delta_atomic = rows_of("delta-atomic")
    assert Counter(r.split for r in delta_atomic) == {"dev": 3840, "test": 4138}
    assert Counter((r.split, r.label) for r in delta_atomic) == {("dev", True): 1937, ("dev", False): 1903,
                                                               ("test", True): 2091, ("test", False): 2047}
    copa = rows_of("copa")
    assert Counter((r.split, r.label) for r in copa) == {("dev", True): 1000, ("dev", False): 1000,
                                                         ("test", True): 500, ("test", False): 500}
    mirrored = defaultdict(set)
    for r in copa:
        if r.split == "dev":
            alternative = r.premise if r.kind.startswith("cause") else r.conclusion
            mirrored[alternative.gloss].add(r.label)
    assert len(mirrored) == 999 and all(labels == {True, False} for labels in mirrored.values())


def test_hard_rows_follow_their_relations() -> None:
    """forward is true one way, both is true both ways, neither is false both ways; a symmetric check is one row."""
    rows = rows_of("hard")
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
    rows = rows_of("wikidata")
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
