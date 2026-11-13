"""Purpose: tests of the claim check (semrewrite.claims): a fact states its gloss filled with its arguments' names and a
rewrite claims its proposal as a conditional; the gate reads the statements that stand out in similarity and never a
rewrite's own predicates'; a statement the judge reads as contradicting a claim never raises an answer's P and one it
reads as neutral leaves every P unchanged, on generated knowledge bases; statements the judge reads as saying the
same count once; a judge whose disbelief is only "does not follow" counts against a claim through its negation alone;
each claim is read once; the actions example, with the readings and cosines nli-deberta-large and
bge-small gave it, withdraws the gaming match and keeps watching TV's; and the command line reports the readings and
exports the checked opinion to SNARS.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from stubs import SentenceTableJudge, TableEmbedder, YesNoTableJudge

import semrewrite.judges
from semrewrite.claims import (
    ClaimCheck,
    Stated,
    conditional,
    negated,
    sentence,
    stance,
    standing_out,
    statements,
    weigh,
)
from semrewrite.cli import main
from semrewrite.engines import make
from semrewrite.kb import KB
from semrewrite.model import Atom, Rewrite
from semrewrite.opinion import Opinion, W, average
from semrewrite.rewriter import Rewriter
from semrewrite.snars import rule_rows

NEUTRAL = Opinion(0.0, 0.0, 1.0)


def assert_close(got: Opinion, want: Opinion) -> None:
    """`got` is `want` up to rounding in the last few bits."""
    for name in ("belief", "disbelief", "uncertainty", "base_rate"):
        assert getattr(got, name) == pytest.approx(getattr(want, name), abs=1e-12), (name, got, want)


def reading(belief: float, disbelief: float) -> Opinion:
    """A judge's three-way reading with what belief and disbelief leave as uncertainty."""
    return Opinion(belief, disbelief, 1.0 - belief - disbelief)


# ---- the actions example, as nli-deberta-large and bge-small read it ----
#
# The readings and cosines were measured on 2026-11-13 with semrewrite judge --judge nli-deberta-large --trust 1 and
# BAAI/bge-small-en-v1.5; the judge is taken at nli-deberta-large's calibrated trust 0.348 and base rate 0.214.

BELIEF = "I consider gaming to be productive"
ZERO = {"watch_tv": "I watch TV", "play_games": "I play video games", "scroll_feed": "I scroll social media",
        "study": "I study for an exam", "exercise": "I go for a run", "sleep_early": "I go to bed early",
        "pay_bills": "I pay the bills", "read_book": "I read a book"}
HOLES = {"watch_tv": ("sam", "{x} watches TV"), "play_games": ("kim", "{x} plays games"),
         "scroll_feed": ("raj", "{x} scrolls social media"), "study": ("lee", "{x} studies for an exam"),
         "exercise": ("ana", "{x} goes for a run"), "sleep_early": ("joe", "{x} goes to bed early"),
         "pay_bills": ("max", "{x} pays the bills"), "read_book": ("eve", "{x} reads a book")}
# Each action's cosine to the gaming belief and to the other actions' statements, for the claim of its rewrite into
# wastes_time, in the order of ZERO and HOLES, the belief last.
ZERO_COSINES = {
    "watch_tv": (0.789, 0.611, 0.545, 0.553, 0.522, 0.478, 0.532, 0.551, 0.594),
    "play_games": (0.622, 0.821, 0.540, 0.584, 0.583, 0.474, 0.527, 0.539, 0.764),
    "study": (0.531, 0.566, 0.477, 0.820, 0.524, 0.506, 0.510, 0.579, 0.613),
}
HOLES_COSINES = {
    "watch_tv": (0.693, 0.520, 0.465, 0.512, 0.457, 0.452, 0.504, 0.536, 0.543),
    "play_games": (0.527, 0.657, 0.431, 0.520, 0.500, 0.406, 0.484, 0.533, 0.720),
    "study": (0.404, 0.413, 0.347, 0.703, 0.431, 0.372, 0.432, 0.543, 0.540),
}
# The judge's readings of a statement against a claim, where it is not neutral: most statements read as
# contradicting every claim, which is why the gate reads only those standing out (c-cc-conditional-spurious).
ZERO_READINGS = {
    ("I scroll social media.", "watch_tv"): (0.000, 0.993), ("I scroll social media.", "play_games"): (0.000, 0.999),
    ("I go for a run.", "play_games"): (0.003, 0.554), ("I go for a run.", "study"): (0.001, 0.986),
    ("I pay the bills.", "study"): (0.000, 0.964), ("I read a book.", "study"): (0.001, 0.984),
    (f"{BELIEF}.", "play_games"): (0.000, 1.000), (f"{BELIEF}.", "study"): (0.000, 1.000),
}
HOLES_READINGS = {
    ("raj scrolls social media.", "watch_tv"): (0.000, 0.999), ("raj scrolls social media.", "play_games"): (0.000, 0.999),
    ("lee studies for an exam.", "watch_tv"): (0.001, 0.846), ("lee studies for an exam.", "play_games"): (0.000, 0.993),
    ("ana goes for a run.", "watch_tv"): (0.000, 0.997), ("ana goes for a run.", "play_games"): (0.002, 0.972),
    ("max pays the bills.", "watch_tv"): (0.000, 0.946), ("max pays the bills.", "play_games"): (0.000, 0.960),
    ("eve reads a book.", "play_games"): (0.001, 0.964), (f"{BELIEF}.", "play_games"): (0.000, 1.000),
    (f"{BELIEF}.", "study"): (0.000, 1.000),
}
# The judge's readings of each rewrite, its proposal's sentences under the letters, as discovery asks them.
ZERO_REWRITES = {"watch_tv": (0.968, 0.000), "play_games": (0.943, 0.000), "study": (0.001, 0.997)}
HOLES_REWRITES = {"watch_tv": (0.755, 0.001), "play_games": (0.039, 0.003), "study": (0.001, 0.986)}


def actions(holes: bool, believed: bool) -> tuple[KB, SentenceTableJudge, TableEmbedder]:
    """The actions example, with arguments or without, with the gaming belief stated or not, and the judge and the
    embedder that read it as the measured models do."""
    glosses = {name: gloss for name, (_, gloss) in HOLES.items()} if holes else dict(ZERO)
    glosses["wastes_time"] = "{x} wastes time" if holes else "I waste time"
    facts = [Atom(name, (person,)) for name, (person, _) in HOLES.items()] if holes else [Atom(n, ()) for n in ZERO]
    if believed:
        glosses["gaming_productive"] = BELIEF
    kb = KB("actions", glosses=glosses, facts=(*facts, *([Atom("gaming_productive", ())] if believed else [])),
            queries=(Atom("wastes_time", ("X",) if holes else ()),))
    catalogue = kb.catalogue()
    stated = [sentence(catalogue[f.pred].sentence(*f.args)) for f in facts]
    letters = ("A",) if holes else ()

    def claim(action: str) -> str:
        return conditional(catalogue[action].sentence(*letters), catalogue["wastes_time"].sentence(*letters))

    table = {(catalogue[a].sentence(*letters), catalogue["wastes_time"].sentence(*letters)): reading(*bd)
             for a, bd in (HOLES_REWRITES if holes else ZERO_REWRITES).items()}
    table |= {(s, claim(a)): reading(*bd) for (s, a), bd in (HOLES_READINGS if holes else ZERO_READINGS).items()}
    cosines = {(claim(a), s): c for a, row in (HOLES_COSINES if holes else ZERO_COSINES).items()
               for s, c in zip([*stated, f"{BELIEF}."], row, strict=True)}
    # nli-deberta-large's calibrated threshold admits both matches without arguments; with them it admits watching TV
    # alone, so a threshold under playing games' 0.039 admits it, as an answer threshold then ranks the answers.
    judge = SentenceTableJudge(table, threshold=0.03 if holes else 0.659, trust=0.348, base_rate=0.214)
    return kb, judge, TableEmbedder(cosines)


def ask(holes: bool, believed: bool):
    kb, judge, embedder = actions(holes, believed)
    rewriter = Rewriter(kb, judge, make("python"), None, beliefs=16, claims=ClaimCheck(kb, judge, embedder))
    return rewriter.ask(kb.queries[0], "lazy-tabled"), rewriter


def test_the_gaming_belief_withdraws_the_gaming_match_and_keeps_watching_tv_s() -> None:
    """With holes, kim plays games and sam watches TV. Stating the belief puts it nearest the gaming match's claim,
    where the judge reads it as a contradiction, so kim's P falls below its value without the belief and below
    sam's, while nothing stands out against watching TV's claim, so sam's P stays where it was. The spurious
    contradictions the judge reads from other people's statements never weigh, since none of them stands out."""
    before, _ = ask(holes=True, believed=False)
    after, rewriter = ask(holes=True, believed=True)
    assert before.beliefs is not None and after.beliefs is not None
    assert set(before.beliefs) == set(after.beliefs) == {("sam",), ("kim",)}
    kim, sam = after.beliefs[("kim",)].projection, after.beliefs[("sam",)].projection
    assert kim < before.beliefs[("kim",)].projection
    assert kim < sam
    assert abs(sam - before.beliefs[("sam",)].projection) < 0.01
    assert after.readings is not None
    games, tv = (("play_games", "wastes_time", (0,)), ("watch_tv", "wastes_time", (0,)))
    assert [(r.statement, r.side) for r in after.readings[games]] == [(f"{BELIEF}.", "against")]
    assert after.readings[tv] == ()
    assert rewriter.claims is not None and rewriter.claims.usage["claims"] == 2


def test_without_arguments_the_answer_keeps_watching_tv_s_derivation() -> None:
    """With no holes the one answer, "I waste time", comes through either match. The belief lowers the gaming match,
    so the answer's P falls a little and stays at least the watching-TV match's alone, which does not move."""
    before, _ = ask(holes=False, believed=False)
    after, _ = ask(holes=False, believed=True)
    assert before.rewrite_beliefs is not None and after.rewrite_beliefs is not None
    assert before.beliefs is not None and after.beliefs is not None
    games, tv = ("play_games", "wastes_time", ()), ("watch_tv", "wastes_time", ())
    assert after.rewrite_beliefs[games].projection < before.rewrite_beliefs[games].projection
    assert after.rewrite_beliefs[games].projection < after.rewrite_beliefs[tv].projection
    assert after.rewrite_beliefs[tv] == before.rewrite_beliefs[tv]
    answer = after.beliefs[()].projection
    assert after.rewrite_beliefs[tv].projection <= answer < before.beliefs[()].projection


# ---- sentences, claims and the gate ----


def test_a_fact_states_its_gloss_filled_with_its_arguments_names() -> None:
    kb = KB("k", glosses={"mother": "{y} is the mother of {x}", "tv": "I watch TV", "same": "I watch TV."},
            facts=(Atom("mother", ("edward_vii", "queen_victoria")), Atom("tv", ()), Atom("same", ())))
    assert statements(kb) == {"queen victoria is the mother of edward vii.": Stated(frozenset({"mother"}), False),
                              "I watch TV.": Stated(frozenset({"tv", "same"}), True)}


def test_a_rewrite_claims_its_proposal_as_a_conditional() -> None:
    kb, judge, embedder = actions(holes=True, believed=False)
    check = ClaimCheck(kb, judge, embedder)
    assert check.claim(("play_games", "wastes_time", (0,))) == "If A plays games, A wastes time."
    kb2 = KB("k", glosses={"child": "{y} is the child of {x}", "parent": "{x} is a parent of {y}"}, facts=())
    swapped = ClaimCheck(kb2, judge, embedder).claim(("child", "parent", (1, 0)))
    assert swapped == "If A is the child of B, A is a parent of B."
    assert conditional("I play video games", "I waste time.") == "If I play video games, I waste time."
    assert negated("I play video games", "I waste time") == "It is not the case that if I play video games, I waste time."


@pytest.mark.parametrize(("cosines", "kept"), [
    ((0.5, 0.52, 0.48, 0.51, 0.49, 0.76), [5]),
    ((0.5, 0.52, 0.48, 0.51, 0.49, 0.53), []),
    ((0.1, 0.1, 0.1, 0.9), [3]),
    ((0.1, 0.9), []),
    ((), []),
])
def test_a_statement_is_read_when_its_cosine_stands_out_by_the_modified_z_score(cosines, kept) -> None:
    """3.5 robust standard deviations above the median: 0.76 among cosines near 0.5 with median deviation 0.01; a
    median deviation of 0 leaves only what lies above the median, and fewer than three cosines leave no spread."""
    assert standing_out(list(cosines)) == kept


def test_a_rewrite_s_own_predicates_statements_are_never_read() -> None:
    """"I play video games." lies nearer the gaming claim than anything and names no individual, and the judge reads
    it as a contradiction, but the knowledge base's evidence on the rewrite reads it, so the check does not."""
    kb, judge, embedder = actions(holes=False, believed=False)
    claim = "If I play video games, I waste time."
    embedder = TableEmbedder({**embedder.table, (claim, "I play video games."): 0.99})
    judge.table[("I play video games.", claim)] = reading(0.0, 0.99)
    check = ClaimCheck(kb, judge, embedder)
    (readings,) = check.readings([Rewrite("wastes_time", "play_games", ())]).values()
    assert "I play video games." not in {r.statement for r in readings}


def test_statements_naming_individuals_are_read_only_when_asked() -> None:
    """raj's statement stands out from the gaming claim and the judge reads it as a contradiction, but it names raj,
    so it bears on the rule through its rows and the check leaves it unread; asked to read every statement, it reads
    it."""
    kb, judge, embedder = actions(holes=True, believed=False)
    claim = "If A plays games, A wastes time."
    embedder = TableEmbedder({**embedder.table, (claim, "raj scrolls social media."): 0.95})
    games = Rewrite("wastes_time", "play_games", (0,))
    assert ClaimCheck(kb, judge, embedder).readings([games])[games.key] == ()
    (r,) = ClaimCheck(kb, judge, embedder, with_arguments=True).readings([games])[games.key]
    assert (r.statement, r.side) == ("raj scrolls social media.", "against")


# ---- what a reading does to answers ----


@st.composite
def worlds(draw: st.DrawFn):
    """A knowledge base of unary predicates p0 to p3 over a, b and c, the question q(X), a judge scoring each p_i => q,
    and the strength of one stated sentence's contradiction of every claim, 0 for a neutral reading."""
    facts = draw(st.sets(st.tuples(st.sampled_from(("p0", "p1", "p2", "p3")), st.sampled_from(("a", "b", "c"))),
                         min_size=1, max_size=8))
    scores = draw(st.lists(st.tuples(st.floats(0.0, 1.0), st.floats(0.0, 1.0)), min_size=4, max_size=4))
    return sorted(facts), scores, draw(st.floats(0.0, 1.0)), draw(st.sampled_from((0.3, 1.0)))


def world(facts, scores, contradiction: float, trust: float, stated: bool):
    glosses = {f"p{i}": f"{{x}} is p{i}" for i in range(4)} | {"q": "{x} is q", "filler": "Water is wet"}
    glosses |= {"stated": "The statement holds"} if stated else {}
    atoms = [Atom(p, (c,)) for p, c in facts] + [Atom("filler", ())] + ([Atom("stated", ())] if stated else [])
    kb = KB("w", glosses=glosses, facts=tuple(atoms))
    table = {}
    for i, (score, certainty) in enumerate(scores):
        # A reading with uncertainty 1 - certainty that splits the rest by the score.
        table[(f"A is p{i}", "A is q")] = Opinion(score * certainty, (1.0 - score) * certainty, 1.0 - certainty)
        claim = f"If A is p{i}, A is q."
        table[("The statement holds.", claim)] = Opinion(0.0, contradiction, 1.0 - contradiction)
    judge = SentenceTableJudge(table, threshold=0.2, trust=trust, base_rate=0.3)
    # The stated sentence stands out from every claim's other statements, whose cosines spread around 0.1.
    cosines = {(f"If A is p{i}, A is q.", "The statement holds."): 0.9 for i in range(4)}
    cosines |= {(f"If A is p{i}, A is q.", f"{c} is p{j}."): 0.1 + 0.01 * j + 0.001 * ord(c)
                for i in range(4) for j in range(4) for c in "abc"}
    return kb, judge, TableEmbedder(cosines, default=0.11)


@settings(max_examples=150, deadline=None)
@given(worlds())
def test_a_contradicting_statement_never_raises_an_answer_and_a_neutral_one_moves_none(drawn) -> None:
    """Stating a sentence the judge reads as contradicting every claim leaves the answers as they were and never
    raises one's P, since it adds evidence against a rewrite and an answer's P rises with each rewrite's; read as
    neutral, (0, 0, 1), it leaves every opinion as it was."""
    facts, scores, contradiction, trust = drawn
    out = {}
    for stated in (False, True):
        kb, judge, embedder = world(facts, scores, contradiction, trust, stated)
        rewriter = Rewriter(kb, judge, make("python"), None, beliefs=16, claims=ClaimCheck(kb, judge, embedder))
        out[stated] = rewriter.ask("q(X)", "lazy-tabled")
    without, with_ = out[False], out[True]
    assert with_.answers == without.answers
    assert with_.beliefs is not None and without.beliefs is not None
    for row, o in without.beliefs.items():
        if contradiction == 0.0:
            assert with_.beliefs[row] == o
        else:
            assert with_.beliefs[row].projection <= o.projection + 1e-12


def test_a_neutral_reading_fuses_to_no_change_and_a_contradiction_only_lowers() -> None:
    held = Opinion.from_evidence(3.0, 1.0, 0.2)
    from semrewrite.claims import Reading

    assert weigh(held, [Reading("s.", 0.9, NEUTRAL, "s.")]) == held
    lowered = weigh(held, [Reading("s.", 0.9, Opinion(0.0, 0.5, 0.5), "s.")])
    # Cumulative fusion adds the evidence: 2 units against, W d/u = 2.
    assert_close(lowered, Opinion.from_evidence(3.0, 3.0, 0.2))
    assert lowered.projection < held.projection


def test_statements_the_judge_reads_as_the_same_count_once() -> None:
    """Two wordings of one belief, each read as contradicting the claim: cumulative fusion would count it twice, so
    when the judge reads each as following from the other they are one source, averaged, and the rewrite's opinion
    is what one of them alone gives; read as different, both count."""
    first, second = "Gaming is productive.", "Playing games is a productive use of time."

    def checked(same: bool) -> Opinion:
        kb, judge, embedder = actions(holes=True, believed=False)
        kb = KB(kb.name, glosses={**kb.glosses, "g1": first[:-1], "g2": second[:-1]},
                facts=(*kb.facts, Atom("g1", ()), Atom("g2", ())), queries=kb.queries)
        claim = "If A plays games, A wastes time."
        judge.table |= {(first, claim): reading(0.0, 0.8), (second, claim): reading(0.0, 0.8)}
        if same:
            judge.table |= {(first, second): reading(0.9, 0.0), (second, first): reading(0.9, 0.0)}
        embedder = TableEmbedder({**embedder.table, (claim, first): 0.8, (claim, second): 0.79})
        check = ClaimCheck(kb, judge, embedder)
        _, opinion = Rewriter(kb, judge, make("python"), None, claims=check).rewrite_opinions(
            [("play_games", "wastes_time", (0,))]
        )
        (readings,) = check.readings([Rewrite("wastes_time", "play_games", (0,))]).values()
        assert {r.statement for r in readings} == {first, second}
        assert {r.source for r in readings} == ({first} if same else {first, second})
        return opinion[("play_games", "wastes_time", (0,))]

    one = Opinion(0.0, 0.348 * 0.8, 1.0 - 0.348 * 0.8, 0.214)
    judged = reading(*HOLES_REWRITES["play_games"])
    alone = judged.with_base_rate(0.214).discount(0.348)
    assert_close(checked(same=True), alone.fuse(average([one, one])))
    assert_close(checked(same=False), alone.fuse(one).fuse(one))
    assert checked(same=False).projection < checked(same=True).projection


def test_a_judge_that_does_not_read_contradiction_counts_against_only_through_the_negation() -> None:
    """A yes/no judge's disbelief says only that the claim does not follow, so its "no" alone weighs nothing; its "yes"
    to the claim's negation counts against the claim."""
    claim = "If A plays games, A wastes time."
    negation = "It is not the case that if A plays games, A wastes time."
    for negation_reading, lowered in ((None, False), (reading(0.9, 0.1), True)):
        kb, judge, embedder = actions(holes=True, believed=True)
        table = {**judge.table, (f"{BELIEF}.", claim): reading(0.0, 1.0)}
        if negation_reading is not None:
            table[(f"{BELIEF}.", negation)] = negation_reading
        yes_no = YesNoTableJudge(table, threshold=0.03, trust=0.348, base_rate=0.214)
        check = ClaimCheck(kb, yes_no, embedder)
        (readings,) = check.readings([Rewrite("wastes_time", "play_games", (0,))]).values()
        (r,) = readings
        assert (f"{BELIEF}.", negation) in yes_no.asked
        assert (r.opinion.disbelief > 0.0) is lowered and r.opinion.belief == 0.0
    assert_close(stance(reading(0.9, 0.0), reading(0.9, 0.0)), Opinion(0.09, 0.09, 0.82))


def test_each_claim_is_read_once() -> None:
    kb, judge, embedder = actions(holes=True, believed=True)
    check = ClaimCheck(kb, judge, embedder)
    rewriter = Rewriter(kb, judge, make("python"), None, beliefs=16, claims=check)
    rewriter.ask("wastes_time(X)", "lazy-tabled")
    asked = (len(judge.asked), check.usage["pairs"], len(embedder.asked))
    rewriter.ask("wastes_time(X)", "lazy-tabled")
    assert (len(judge.asked) - asked[0], check.usage["pairs"], len(embedder.asked)) == (2 * 0, asked[1], asked[2])
    assert check.usage == {"claims": 2, "pairs": 1, "statements": 9}


# ---- the command line and the SNARS export ----


@pytest.fixture
def actions_cli(monkeypatch: pytest.MonkeyPatch):
    from semrewrite import cli

    kb, judge, embedder = actions(holes=True, believed=True)
    monkeypatch.setattr(cli, "_judge", lambda args, kb: judge)
    monkeypatch.setattr(cli, "_embedder", lambda retriever: embedder)
    monkeypatch.setattr(cli.formats, "load", lambda path: kb)
    # The stub judge has no calibrated data trust; the knowledge base states no wastes_time, so it would weigh nothing.
    monkeypatch.setattr(semrewrite.judges, "default_data_trust", lambda name: 0.0)
    return kb


def test_ask_check_ranks_the_answers_and_explains_what_weighed(actions_cli, capsys) -> None:
    assert main(["ask", "actions.kb", "wastes_time(X)", "--retriever", "all", "--check", "--explain"]) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0].startswith("wastes_time(sam)    P 0.4")
    assert any(line.startswith("wastes_time(kim)    P 0.1") for line in lines)
    err = captured.err.splitlines()
    games = next(i for i, line in enumerate(err) if line.startswith("  wastes_time(X0) ⇐ play_games(X0)    P "))
    assert err[games + 1].startswith("    against  P ")
    assert err[games + 1].endswith(f'"{BELIEF}."  similarity 0.720')
    assert ", 1 pair reading 2 claims in " in err[-1]


def test_ask_check_writes_the_readings_as_json(actions_cli, capsys) -> None:
    assert main(["ask", "actions.kb", "wastes_time(X)", "--retriever", "all", "--check", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["answers"][0] == ["sam"]
    games = next(c for c in out["checks"] if c["rewrite"][0] == "play_games")
    assert games["claim"] == "If A plays games, A wastes time."
    (statement,) = games["statements"]
    assert (statement["statement"], statement["side"]) == (f"{BELIEF}.", "against")
    assert out["cost"]["check"]["claims"] == 2 and out["cost"]["check"]["statements"] == 9


def test_rewrites_check_lists_each_match_with_what_weighed(actions_cli, capsys) -> None:
    assert main(["rewrites", "actions.kb", "wastes_time", "--retriever", "all", "--check"]) == 0
    lines = capsys.readouterr().out.splitlines()
    games = next(i for i, line in enumerate(lines) if "⇐ play_games(X0)" in line)
    assert lines[games].startswith("admit   0.039  wastes_time(X0) ⇐ play_games(X0)    P ")
    assert lines[games + 1].strip().startswith("against  P ")


def test_the_snars_export_believes_each_validity_at_its_checked_opinion(actions_cli, capsys) -> None:
    """The exported validity of the gaming match holds its judge's evidence and the belief's against it, added, as
    cumulative fusion adds evidence (Jøsang 2016, Thm. 12.2), so every use of the rewrite in SNARS carries the
    defeat; without --check it holds the judge's alone."""
    games = Rewrite("wastes_time", "play_games", (0,))
    judged = reading(*HOLES_REWRITES["play_games"]).with_base_rate(0.214).discount(0.348)
    against = Opinion(0.0, 0.348 * 1.0, 1.0 - 0.348, 0.214)
    r, s = judged.evidence()
    expected = rule_rows(games, Opinion.from_evidence(r, s + W * against.disbelief / against.uncertainty, 0.214),
                         "semrewrite sentence stub")
    unchecked = rule_rows(games, judged, "semrewrite sentence stub")
    for flags, want in ((["--check"], expected), ([], unchecked)):
        assert main(["rewrites", "actions.kb", "wastes_time", "--retriever", "all", "--snars", *flags]) == 0
        lines = capsys.readouterr().out.splitlines()
        at = next(i for i, line in enumerate(lines) if '"rw-wastes_time-play_games-0"' in line and "believe!" in line)
        exported = lines[at].split("((evidence ")[1].split(")")[0].split()
        assert [float(x) for x in exported] == pytest.approx([float(x) for x in
                                                              want[0].split("((evidence ")[1].split(")")[0].split()])
        assert lines[at + 1] == want[1]
