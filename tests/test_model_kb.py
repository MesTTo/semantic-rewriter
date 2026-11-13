"""Purpose: engine-free tests of the data model, knowledge bases, the reference
evaluator, gold entailment, the knowledge-base formats, and alias
canonicalisation, which must leave every query's answers unchanged for
predicates of any arity.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from semrewrite import formats
from semrewrite.benchmarks import BENCHMARKS
from semrewrite.fixpoint import answers, least_model
from semrewrite.kb import KB, atom
from semrewrite.model import Atom, Predicate, Proposal, Rewrite, Rule, compose, identity, inverse, mappings, parse_atom
from semrewrite.rewriter import GroupUnionFind, canonicalise

KBS = {name: make() for name, make in BENCHMARKS.items()}


def test_predicates_take_any_arity_and_both_placeholder_styles() -> None:
    gave = Predicate("gave", "{x} gave {y} to {z}")
    lives = Predicate("lives_in", "{0} lives in {1}")
    assert (gave.arity, lives.arity) == (3, 2)
    assert gave.sentence("A", "B", "C") == "A gave B to C"
    with pytest.raises(ValueError, match="positions 0 to n-1 with none missing"):
        Predicate("bad", "{y} only")
    with pytest.raises(ValueError, match="mixes"):
        Predicate("bad", "{x} and {1}")
    raining = Predicate("raining", "it is raining")
    assert raining.arity == 0 and raining.sentence() == "it is raining"


@pytest.mark.parametrize("gloss", ["{x} is p\nof {y}", "{x} is\x07 p", "{x} is p of {y}", "{x} \ud800 p"])
def test_a_gloss_is_one_line(gloss: str) -> None:
    with pytest.raises(ValueError, match="one line of text"):
        Predicate("p", gloss)


def test_names_must_match_whole() -> None:
    """A trailing newline once slipped past the anchored patterns, making tom\\n a constant distinct from tom."""
    for bad in ("tom\n", "Tom\n", "X\n"):
        with pytest.raises(ValueError):
            Atom("p", (bad,))
    with pytest.raises(ValueError, match="predicate 'p\\\\n'"):
        Atom("p\n", ("a",))


def test_proposals_read_mappings_and_their_converse() -> None:
    parent, child = Predicate("parent", "{x} is a parent of {y}"), Predicate("child_of", "{x} is a child of {y}")
    p = Proposal(child, parent, (1, 0))
    assert p.sentences() == ("B is a child of A", "A is a parent of B")
    assert p.swapped and p.converse().key == ("parent", "child_of", (1, 0))
    assert str(p.rewrite(0.9).rule()) == "parent(X0, X1) :- child_of(X1, X0)"


@given(st.integers(1, 4).flatmap(lambda n: st.tuples(st.permutations(range(n)), st.permutations(range(n)))))
def test_mappings_form_a_group(pair: tuple[list[int], list[int]]) -> None:
    a, b = tuple(pair[0]), tuple(pair[1])
    n = len(a)
    assert compose(a, inverse(a)) == identity(n) == compose(inverse(a), a)
    assert inverse(compose(a, b)) == compose(inverse(b), inverse(a))


def test_atoms_parse_and_match() -> None:
    assert parse_atom("parent(X, tom)") == Atom("parent", ("X", "tom")) == parse_atom("parent X tom")
    assert Atom("p", ("X", "X")).match(("a", "a")) == {"X": "a"}
    assert Atom("p", ("X", "X")).match(("a", "b")) is None


def test_reference_evaluator_chains_swaps_and_cuts_at_a_depth() -> None:
    rules = (
        Rule(atom("parent", "X", "Y"), (atom("mother", "X", "Y"),)),
        Rule(atom("parent", "X", "Y"), (atom("child_of", "Y", "X"),)),
        Rule(atom("grandparent", "X", "Z"), (atom("parent", "X", "Y"), atom("parent", "Y", "Z"))),
    )
    facts = (atom("mother", "carol", "alice"), atom("child_of", "tom", "alice"))
    model = least_model(facts, rules)
    assert model["parent"] == {("carol", "alice"), ("alice", "tom")}
    assert model["grandparent"] == {("carol", "tom")}
    assert "grandparent" not in least_model(facts, rules, depth=1)
    assert least_model(facts, rules, depth=2)["grandparent"] == {("carol", "tom")}


def test_entailment_reads_gold_implications_and_user_rules_together() -> None:
    """Two swaps cancel, one survives, a symmetric conclusion takes both orders, and a user rule entails too."""
    kb = KB(
        name="entail",
        glosses={p: f"{{x}} is {p} of {{y}}" for p in ("son_of", "child", "parent", "ancestor", "spouse", "wife_of")},
        facts=(),
        rules=(Rule(atom("ancestor", "X", "Y"), (atom("parent", "X", "Y"),)),),
        gold=frozenset({("son_of", "child", False), ("child", "parent", True), ("spouse", "spouse", True), ("wife_of", "spouse", False)}),
    )
    assert kb.entails("son_of", "parent", (1, 0))
    assert not kb.entails("son_of", "parent", (0, 1))
    assert kb.entails("wife_of", "spouse", (0, 1)) and kb.entails("wife_of", "spouse", (1, 0))
    assert kb.entails("son_of", "ancestor", (1, 0))
    assert ("parent", "parent", (0, 1)) not in kb.entailed()
    assert ("spouse", "spouse", (1, 0)) in kb.entailed()


def test_entailment_works_for_ternary_predicates() -> None:
    kb = KB("t", {"gave": "{x} gave {y} to {z}", "received": "{x} received {y} from {z}"}, (),
            gold=frozenset({("gave", "received", (2, 1, 0))}))
    assert kb.entails("gave", "received", (2, 1, 0))
    assert not kb.entails("gave", "received", (0, 1, 2))
    assert kb.closure() == {}


def test_canonical_constants_are_reserved_in_rules_and_queries_too() -> None:
    """A rule naming canonical_1 made entails() claim p => q, which p(a, b) does not support."""
    glosses = {"p": "{x} p {y}", "q": "{x} q {y}"}
    for kwargs in ({"rules": (Rule(atom("q", "X", "canonical_1"), (atom("p", "X", "Y"),)),)},
                   {"queries": (atom("q", "canonical_12", "Y"),)}):
        with pytest.raises(ValueError, match="reserved for the canonical database"):
            KB("k", glosses, (atom("p", "a", "b"),), **kwargs)


def test_a_trap_must_name_a_glossed_predicate() -> None:
    with pytest.raises(ValueError, match="no gloss"):
        KB("k", {"p": "{x} p {y}"}, (atom("p", "a", "b"),), traps=frozenset({"guardian"}))


_LETTERS_NOT_BRACES = st.characters(blacklist_categories=("Cc", "Cs", "Zl", "Zp"), blacklist_characters="{}")


@st.composite
def _knowledge_bases(draw) -> KB:
    """Knowledge bases with any one-line glosses around their placeholders, arities 0 to 3, facts, a rule, gold
    implications, queries and traps."""
    names = draw(st.lists(st.sampled_from(["p", "q", "r", "has_mother", "x1"]), min_size=2, max_size=4, unique=True))
    arity = {n: draw(st.integers(0, 3)) for n in names}
    text = st.text(_LETTERS_NOT_BRACES, max_size=12)
    glosses = {n: draw(text) + "".join(f"{{{v}}}" + draw(text) for v in "xyz"[: arity[n]]) for n in names}
    consts = ("a", "b", "c")
    facts = tuple(dict.fromkeys(
        Atom(n, tuple(draw(st.sampled_from(consts)) for _ in range(arity[n])))
        for n in draw(st.lists(st.sampled_from(names), max_size=6))))
    head, body = draw(st.sampled_from(names)), draw(st.sampled_from(names))
    variables = ("X", "Y", "Z")[: arity[body]]
    head_args = tuple(draw(st.sampled_from(variables + consts)) for _ in range(arity[head]))
    rules = (Rule(Atom(head, head_args), (Atom(body, variables),)),)
    same = [(p, c) for p in names for c in names if arity[p] == arity[c] and (p != c or arity[p] > 1)]
    gold = frozenset()
    if same:
        p, c = draw(st.sampled_from(same))
        gold = frozenset({(p, c, draw(st.sampled_from([m for m in mappings(arity[p]) if p != c or m != identity(arity[p])])))})
    queries = tuple(Atom(n, tuple(draw(st.sampled_from(("X", "a"))) for _ in range(arity[n])))
                    for n in draw(st.lists(st.sampled_from(names), max_size=2)))
    traps = frozenset(draw(st.lists(st.sampled_from(names), max_size=2)))
    return KB("kb", glosses, facts, rules, gold, queries, traps)


def _parts(kb: KB) -> tuple:
    return (dict(kb.glosses), kb.facts, kb.rules, kb.queries, kb.gold, kb.traps)


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(_knowledge_bases())
def test_text_and_json_round_trip_any_knowledge_base(kb: KB) -> None:
    """Accents, arrows, quotes and backslashes in a gloss, arity 0 and traps all survive both formats."""
    assert _parts(formats.loads_text(formats.dumps_text(kb))) == _parts(kb)
    assert _parts(formats.loads_json(formats.dumps_json(kb))) == _parts(kb)


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(_knowledge_bases())
def test_metta_round_trips_any_knowledge_base(kb: KB) -> None:
    pytest.importorskip("metta")
    assert _parts(formats.loads_metta(formats.dumps_metta(kb))) == _parts(kb)


def test_files_are_utf8_whatever_the_locale(tmp_path) -> None:
    """Under an ASCII locale, with UTF-8 mode off, a file written in the locale's encoding could not hold the arrow."""
    code = (
        "import sys\n"
        "from semrewrite import formats\n"
        "from semrewrite.kb import KB, atom\n"
        "kb = KB('k', {'leads': '{x} leads {y} \\u2192 caf\\u00e9'}, (atom('leads', 'a', 'b'),))\n"
        "for suffix in ('.kb', '.json'):\n"
        "    path = sys.argv[1] + suffix\n"
        "    formats.save(kb, path)\n"
        "    assert formats.load(path).glosses == kb.glosses\n"
    )
    package_root = str(pathlib.Path(formats.__file__).resolve().parents[1])
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONPATH": package_root}
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "k")], check=True, env=env)
    for suffix in (".kb", ".json"):
        assert "→ café".encode() in (tmp_path / f"k{suffix}").read_bytes()


@pytest.mark.parametrize("name", sorted(KBS))
@pytest.mark.parametrize("dumps, loads", [(formats.dumps_text, formats.loads_text), (formats.dumps_json, formats.loads_json)])
def test_text_and_json_round_trip_every_benchmark(name: str, dumps, loads) -> None:
    kb = KBS[name]
    back = loads(dumps(kb), name)
    assert (dict(back.glosses), back.facts, back.rules, back.queries, back.gold) == (dict(kb.glosses), kb.facts, kb.rules, kb.queries, kb.gold)


@pytest.mark.parametrize("name", sorted(KBS))
def test_metta_round_trips_every_benchmark(name: str) -> None:
    pytest.importorskip("metta")
    kb = KBS[name]
    back = formats.loads_metta(formats.dumps_metta(kb), name)
    assert (dict(back.glosses), back.facts, back.rules) == (dict(kb.glosses), kb.facts, kb.rules)


def test_text_format_reads_comments_quotes_and_mappings() -> None:
    kb = formats.loads_text('''
        gloss said "{x} said \\"hi # there\\" to {y}"   % a comment
        gloss heard "{x} heard {y}"
        said(ann, bob).
        heard(X, Y) :- said(Y, X).
        query heard(bob, X).
        gold said => heard swap.
    ''')
    assert kb.glosses["said"] == '{x} said "hi # there" to {y}'
    assert kb.gold == frozenset({("said", "heard", (1, 0))})
    assert kb.answers(kb.queries[0]) == {("bob", "ann")}
    with pytest.raises(ValueError, match="line 2"):
        formats.loads_text('gloss p "{x} p {y}"\np(X, a).')
    with pytest.raises(ValueError, match="cannot read the body"):
        formats.parse_rule("p(X) :- q(X) r(X)")
    with pytest.raises(ValueError, match="line 1"):
        formats.loads_text('gloss p "{x} \\q p"')


def test_group_union_find_handles_a_class_deeper_than_the_recursion_limit() -> None:
    """Unions in descending name order build a chain as deep as the class, which a recursive find overflowed."""
    uf = GroupUnionFind()
    names = [f"n{i:04d}" for i in range(1500)]
    for n in names:
        uf.add(n, 2)
    for hi, lo in zip(reversed(names), reversed(names[:-1])):
        uf.union(hi, lo, (0, 1))
    assert uf.find("n1499") == ("n0000", (0, 1))


def test_group_union_find_joins_aliases_and_finds_automorphisms() -> None:
    uf = GroupUnionFind()
    for p in ("parent", "child_of", "mother", "mum", "spouse"):
        uf.add(p, 2)
    uf.union("parent", "child_of", (1, 0))
    uf.union("mother", "mum", (0, 1))
    assert uf.find("child_of")[0] == uf.find("parent")[0]
    assert uf.find("mum")[0] == uf.find("mother")[0]
    uf.union("spouse", "spouse", (1, 0))
    assert uf.automorphisms["spouse"] == {(1, 0)}


@st.composite
def _programs(draw: st.DrawFn) -> tuple[KB, list[Rewrite], Atom]:
    n = draw(st.integers(2, 5))
    arity = {f"p{i}": draw(st.sampled_from((1, 2, 3))) for i in range(n)}
    names = sorted(arity)
    glosses = {p: " ".join(f"{{{i}}}" for i in range(arity[p])) + f" {p}" for p in names}
    consts = st.sampled_from(("c0", "c1", "c2"))
    facts = draw(st.lists(st.sampled_from(names).flatmap(lambda p: st.tuples(st.just(p), st.lists(consts, min_size=arity[p], max_size=arity[p]))), max_size=8))
    rewrites: dict = {}
    pairs = st.tuples(st.sampled_from(names), st.sampled_from(names)).filter(lambda pq: arity[pq[0]] == arity[pq[1]])
    for p, q in draw(st.lists(pairs, max_size=8)):
        m = tuple(draw(st.permutations(range(arity[p]))))
        if p == q and m == identity(arity[p]):
            continue
        rw = Rewrite(p, q, m)
        rewrites[rw.key] = rw
        if draw(st.booleans()) and p != q:
            back = Rewrite(q, p, inverse(m))
            rewrites[back.key] = back
    query_pred = draw(st.sampled_from(names))
    args = tuple(draw(st.sampled_from(("X", "Y", "Z", "c0", "c1"))) for _ in range(arity[query_pred]))
    kb = KB("generated", glosses, tuple(Atom(p, tuple(a)) for p, a in facts))
    return kb, list(rewrites.values()), Atom(query_pred, args)


@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(_programs())
def test_canonicalisation_never_changes_answers(case: tuple[KB, list[Rewrite], Atom]) -> None:
    """Renaming alias classes to their roots, with arguments re-mapped, answers every query exactly as before."""
    kb, rewrites, query = case
    expected = answers(least_model(kb.facts, [rw.rule() for rw in rewrites]), query)
    renamed_kb, renamed, q, back = canonicalise(kb, rewrites, query)
    got = answers(least_model(renamed_kb.facts, [rw.rule() for rw in renamed]), q)
    assert {tuple(row[back[j]] for j in range(len(row))) for row in got} == expected


def test_canonicalisation_keeps_a_symmetry_found_inside_one_alias_class() -> None:
    """The falsifying example Hypothesis once found: p0 <=> p2 plus p0(y, x) => p2(x, y) makes the class symmetric."""
    kb = KB("generated", {p: f"{{x}} is {p} of {{y}}" for p in ("p0", "p1", "p2")}, (atom("p0", "c0", "c1"),))
    rewrites = [Rewrite("p2", "p0", (0, 1)), Rewrite("p2", "p0", (1, 0)), Rewrite("p0", "p2", (0, 1))]
    query = atom("p0", "X", "Y")
    renamed_kb, renamed, q, back = canonicalise(kb, rewrites, query)
    got = answers(least_model(renamed_kb.facts, [rw.rule() for rw in renamed]), q)
    assert {tuple(row[back[j]] for j in range(2)) for row in got} == {("c0", "c1"), ("c1", "c0")}
