"""Purpose: engine-free tests of the data model, knowledge bases, the reference
evaluator, gold entailment, the knowledge-base formats, and alias
canonicalisation, which must leave every query's answers unchanged for
predicates of any arity.
"""

from __future__ import annotations

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
    with pytest.raises(ValueError, match="every argument position"):
        Predicate("bad", "{y} only")
    with pytest.raises(ValueError, match="mixes"):
        Predicate("bad", "{x} and {1}")


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


@pytest.mark.parametrize("name", sorted(KBS))
@pytest.mark.parametrize("dumps, loads", [(formats.dumps_text, formats.loads_text), (formats.dumps_json, formats.loads_json)])
def test_text_and_json_round_trip_every_benchmark(name: str, dumps, loads) -> None:  # noqa: ANN001
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
        if draw(st.booleans()) and not (p == q):
            back = Rewrite(q, p, inverse(m))
            rewrites[back.key] = back
    query_pred = draw(st.sampled_from(names))
    args = tuple(draw(st.sampled_from(("X", "Y", "Z", "c0", "c1"))) for _ in range(arity[query_pred]))
    kb = KB("generated", glosses, tuple(Atom(p, tuple(a)) for p, a in facts))
    return kb, list(rewrites.values()), Atom(query_pred, args)


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
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
