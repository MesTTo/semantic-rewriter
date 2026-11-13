"""Purpose: tests of the claim check (semrewrite.claims): a fact states its gloss filled with its arguments' names and a
rewrite claims its proposal as a conditional; the gate reads the statements that stand out in similarity and never a
rewrite's own predicates'; a statement the judge reads as contradicting a claim at its threshold withdraws the rewrite,
one read more vaguely only lowers answers' P, discounted by the judge's check trust, and one read as neutral moves
nothing, on generated knowledge bases; statements the judge reads as saying the same count once; a judge whose
disbelief is only "does not follow" counts against a claim through its negation alone; each claim is read once; the
rewriter checks by default, loading nothing where the knowledge base states nothing to read, and refuses a judge that
cannot read what it states; the factory reads each judge's check trust; the actions example, with the
readings and cosines nli-deberta-large and bge-small gave it, withdraws the gaming match and keeps watching TV's; and
the command line and the trace report withdrawals and the SNARS export carries the check.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from stubs import SentenceTableJudge, TableEmbedder, TableJudge, YesNoTableJudge

import semrewrite.judges
import semrewrite.retrieve
from semrewrite.claims import (
    ClaimCheck,
    Reading,
    Stated,
    conditional,
    default_check,
    negated,
    read,
    sentence,
    stance,
    standing_out,
    statements,
    weigh,
)
from semrewrite.cli import main
from semrewrite.engines import make
from semrewrite.judges.base import SentenceJudge
from semrewrite.kb import KB
from semrewrite.model import Atom, Rewrite
from semrewrite.opinion import Opinion, W, average
from semrewrite.rewriter import Rewriter
from semrewrite.snars import rule_rows
from semrewrite.trace import Defeated, TextTrace

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
# BAAI/bge-small-en-v1.5; the judge is taken at nli-deberta-large's calibrated trust 0.348, base rate 0.214 and
# check trust 0.698, which experiments/check_trust.py fitted on delta-ATOMIC's dev updates (results/check_trust.json).

BELIEF = "I consider gaming to be productive"
CHECK_TRUST = 0.698
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
GAMES = {False: ("play_games", "wastes_time", ()), True: ("play_games", "wastes_time", (0,))}
TV = {False: ("watch_tv", "wastes_time", ()), True: ("watch_tv", "wastes_time", (0,))}


def actions(holes: bool, believed: bool, belief: tuple[float, float] = (0.0, 1.0)
            ) -> tuple[KB, SentenceTableJudge, TableEmbedder]:
    """The actions example, with arguments or without, with the gaming belief stated or not, and the judge and the
    embedder that read it as the measured models do; `belief` replaces the judge's reading of the belief against the
    gaming claim."""
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
    table[(f"{BELIEF}.", claim("play_games"))] = reading(*belief)
    cosines = {(claim(a), s): c for a, row in (HOLES_COSINES if holes else ZERO_COSINES).items()
               for s, c in zip([*stated, f"{BELIEF}."], row, strict=True)}
    # nli-deberta-large's calibrated threshold admits both matches without arguments; with them it admits watching TV
    # alone, so a threshold under playing games' 0.039 admits it, as an answer threshold then ranks the answers. The
    # check still defeats at the calibrated 0.659.
    judge = SentenceTableJudge(table, threshold=0.03 if holes else 0.659, trust=0.348, base_rate=0.214,
                               check_trust=CHECK_TRUST, calibrated_threshold=0.659)
    return kb, judge, TableEmbedder(cosines)


def ask(holes: bool, believed: bool, belief: tuple[float, float] = (0.0, 1.0)):
    kb, judge, embedder = actions(holes, believed, belief)
    rewriter = Rewriter(kb, judge, make("python"), None, beliefs=16, claims=ClaimCheck(kb, judge, embedder))
    return rewriter.ask(kb.queries[0], "lazy-tabled"), rewriter


def test_the_gaming_belief_withdraws_the_gaming_match_and_keeps_watching_tv_s() -> None:
    """With holes, kim plays games and sam watches TV. Stating the belief puts it nearest the gaming match's claim,
    where the judge reads it as a contradiction at d 1.000, past its calibrated threshold, so the match is withdrawn and
    kim is no longer an answer, while nothing stands out against watching TV's claim, so sam's opinion is what it was.
    The spurious contradictions the judge reads from other people's statements never weigh, since none stands out."""
    before, _ = ask(holes=True, believed=False)
    after, rewriter = ask(holes=True, believed=True)
    assert before.beliefs is not None and after.beliefs is not None
    assert set(before.beliefs) == {("sam",), ("kim",)} and set(after.beliefs) == {("sam",)}
    assert after.beliefs[("sam",)] == before.beliefs[("sam",)]
    assert GAMES[True] in before.admitted and GAMES[True] not in after.admitted
    (r,) = after.defeated[GAMES[True]]
    assert (r.statement, r.stance.disbelief) == (f"{BELIEF}.", 1.0)
    assert after.readings is not None and after.readings[TV[True]] == ()
    assert before.defeated == {}
    assert rewriter.claims is not None and rewriter.claims.usage["claims"] == 2


def test_without_arguments_the_answer_keeps_watching_tv_s_derivation_alone() -> None:
    """With no holes the one answer, "I waste time", comes through either match. The belief withdraws the gaming
    match, so the answer stays, with the watching-TV match's P alone, which does not move."""
    before, _ = ask(holes=False, believed=False)
    after, _ = ask(holes=False, believed=True)
    assert before.rewrite_beliefs is not None and after.rewrite_beliefs is not None
    assert before.beliefs is not None and after.beliefs is not None
    assert set(after.defeated) == {GAMES[False]} and GAMES[False] not in after.rewrite_beliefs
    assert after.rewrite_beliefs[TV[False]] == before.rewrite_beliefs[TV[False]]
    assert after.beliefs[()].projection == pytest.approx(after.rewrite_beliefs[TV[False]].projection, abs=1e-12)
    assert after.beliefs[()].projection < before.beliefs[()].projection


def test_a_vague_contradiction_weakens_the_match_without_withdrawing_it() -> None:
    """Read at d 0.5, below the judge's 0.659, the belief no longer defeats the gaming match: the match stays admitted
    and its opinion takes the reading discounted by the check trust, 2 x 0.349 / 0.651 units of evidence against."""
    after, _ = ask(holes=False, believed=True, belief=(0.0, 0.5))
    before, _ = ask(holes=False, believed=False)
    assert after.defeated == {} and GAMES[False] in after.admitted
    assert after.rewrite_beliefs is not None and before.rewrite_beliefs is not None
    against = Opinion(0.0, CHECK_TRUST * 0.5, 1.0 - CHECK_TRUST * 0.5, 0.214)
    assert_close(after.rewrite_beliefs[GAMES[False]], before.rewrite_beliefs[GAMES[False]].fuse(against))
    assert after.readings is not None
    (r,) = after.readings[GAMES[False]]
    assert_close(r.stance, reading(0.0, 0.5))
    assert_close(r.opinion, against)


def test_the_believed_example_is_the_first_person_one_with_the_belief_added() -> None:
    """examples/actions_first_person_believed.kb, which README.md's transcript reads, is the first-person example with
    the gaming belief stated as a gloss and a fact with no arguments, and nothing else changed."""
    root = Path(__file__).resolve().parents[1] / "examples"

    def stated(name: str) -> list[str]:
        return [line for line in (root / name).read_text().splitlines() if line.strip() and not line.startswith("%")]

    base, believed = stated("actions_first_person.kb"), stated("actions_first_person_believed.kb")
    assert believed == [*base[:-1], f'gloss gaming_productive "{BELIEF}".', "gaming_productive.", base[-1]]
    assert base[-1] == "query wastes_time."


def test_a_run_admitting_at_a_low_threshold_still_defeats_only_at_the_calibrated_one() -> None:
    """With holes the run admits at 0.03 to let the gaming match in, but the check defeats at the judge's calibrated
    0.659: the belief read at d 0.5 only weakens the match, kim stays an answer below where he was, and read at d 0.7
    it withdraws the match. Defeating at the run's 0.03 would let any faint reading withdraw a rewrite."""
    before, _ = ask(holes=True, believed=False)
    vague, rewriter = ask(holes=True, believed=True, belief=(0.0, 0.5))
    clear, _ = ask(holes=True, believed=True, belief=(0.0, 0.7))
    assert rewriter.claims is not None and rewriter.claims.bar == 0.659 and rewriter.scorer.threshold == 0.03
    assert vague.defeated == {} and ("kim",) in vague.answers
    assert vague.beliefs is not None and before.beliefs is not None
    assert vague.beliefs[("kim",)].projection < before.beliefs[("kim",)].projection
    assert set(clear.defeated) == {GAMES[True]} and clear.answers == {("sam",)}


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


# ---- what a reading does to rewrites and answers ----


@st.composite
def worlds(draw: st.DrawFn):
    """A knowledge base of unary predicates p0 to p3 over a, b and c, the question q(X), a judge scoring each p_i => q,
    the strength of one stated sentence's contradiction of every claim, 0 for a neutral reading, and the check trust."""
    facts = draw(st.sets(st.tuples(st.sampled_from(("p0", "p1", "p2", "p3")), st.sampled_from(("a", "b", "c"))),
                         min_size=1, max_size=8))
    scores = draw(st.lists(st.tuples(st.floats(0.0, 1.0), st.floats(0.0, 1.0)), min_size=4, max_size=4))
    return sorted(facts), scores, draw(st.floats(0.0, 1.0)), draw(st.sampled_from((0.3, 1.0)))


FILLERS = ("Water is wet", "Grass is green", "Snow is cold")


def world(facts, scores, contradiction: float, trust: float, stated: bool):
    glosses = {f"p{i}": f"{{x}} is p{i}" for i in range(4)} | {"q": "{x} is q"}
    glosses |= {f"f{k}": text for k, text in enumerate(FILLERS)}
    glosses |= {"stated": "The statement holds"} if stated else {}
    atoms = [Atom(p, (c,)) for p, c in facts] + [Atom(f"f{k}", ()) for k in range(len(FILLERS))]
    kb = KB("w", glosses=glosses, facts=(*atoms, *([Atom("stated", ())] if stated else [])))
    table = {}
    for i, (score, certainty) in enumerate(scores):
        # A reading with uncertainty 1 - certainty that splits the rest by the score.
        table[(f"A is p{i}", "A is q")] = Opinion(score * certainty, (1.0 - score) * certainty, 1.0 - certainty)
        claim = f"If A is p{i}, A is q."
        table[("The statement holds.", claim)] = Opinion(0.0, contradiction, 1.0 - contradiction)
    judge = SentenceTableJudge(table, threshold=0.2, trust=trust, base_rate=0.3, check_trust=trust)
    # The stated sentence stands out from every claim's other statements: three fillers naming no one at 0.10, 0.11
    # and 0.12, so there is always a spread to stand out from, and the facts around 0.2, which the judge reads as
    # neutral where they are read at all.
    cosines = {(f"If A is p{i}, A is q.", "The statement holds."): 0.9 for i in range(4)}
    cosines |= {(f"If A is p{i}, A is q.", f"{text}."): 0.1 + 0.01 * k
                for i in range(4) for k, text in enumerate(FILLERS)}
    cosines |= {(f"If A is p{i}, A is q.", f"{c} is p{j}."): 0.1 + 0.01 * j + 0.001 * ord(c)
                for i in range(4) for j in range(4) for c in "abc"}
    return kb, judge, TableEmbedder(cosines)


@settings(max_examples=150, deadline=None)
@given(worlds())
def test_a_contradicting_statement_never_raises_an_answer_and_a_neutral_one_moves_none(drawn) -> None:
    """Stating a sentence the judge reads as contradicting every claim never adds an answer nor raises one's P. At or
    above the judge's threshold it withdraws every rewrite, and since no fact states q, no answer is left; below it,
    the answers stay and each P can only fall, since the reading adds evidence against a rewrite and an answer's P
    rises with each rewrite's. Read as neutral, (0, 0, 1), it leaves every opinion as it was."""
    facts, scores, contradiction, trust = drawn
    out = {}
    for stated in (False, True):
        kb, judge, embedder = world(facts, scores, contradiction, trust, stated)
        rewriter = Rewriter(kb, judge, make("python"), None, beliefs=16, claims=ClaimCheck(kb, judge, embedder))
        out[stated] = rewriter.ask("q(X)", "lazy-tabled")
    without, with_ = out[False], out[True]
    assert with_.beliefs is not None and without.beliefs is not None
    if contradiction >= 0.2:
        assert with_.answers == set() and with_.admitted == set()
        assert set(with_.defeated) == without.admitted
        return
    assert with_.answers == without.answers and with_.defeated == {}
    for row, o in without.beliefs.items():
        if contradiction == 0.0:
            assert with_.beliefs[row] == o
        else:
            assert with_.beliefs[row].projection <= o.projection + 1e-12


def test_a_neutral_reading_fuses_to_no_change_and_a_contradiction_only_lowers() -> None:
    held = Opinion.from_evidence(3.0, 1.0, 0.2)
    assert weigh(held, [Reading("s.", 0.9, NEUTRAL, NEUTRAL, "s.")]) == held
    lowered = weigh(held, [Reading("s.", 0.9, reading(0.0, 0.5), Opinion(0.0, 0.5, 0.5), "s.")])
    # Cumulative fusion adds the evidence: 2 units against, W d/u = 2.
    assert_close(lowered, Opinion.from_evidence(3.0, 3.0, 0.2))
    assert lowered.projection < held.projection


def test_statements_the_judge_reads_as_the_same_count_once() -> None:
    """Two wordings of one belief, each read as contradicting the claim, below the judge's threshold: cumulative
    fusion would count it twice, so when the judge reads each as following from the other they are one source,
    averaged, and the rewrite's opinion is what one of them alone gives; read as different, both count."""
    first, second = "Gaming is productive.", "Playing games is a productive use of time."

    def checked(same: bool) -> Opinion:
        kb, judge, embedder = actions(holes=False, believed=False)
        kb = KB(kb.name, glosses={**kb.glosses, "g1": first[:-1], "g2": second[:-1]},
                facts=(*kb.facts, Atom("g1", ()), Atom("g2", ())), queries=kb.queries)
        claim = "If I play video games, I waste time."
        judge.table |= {(first, claim): reading(0.0, 0.5), (second, claim): reading(0.0, 0.5)}
        if same:
            judge.table |= {(first, second): reading(0.9, 0.0), (second, first): reading(0.9, 0.0)}
        embedder = TableEmbedder({**embedder.table, (claim, first): 0.9, (claim, second): 0.89})
        check = ClaimCheck(kb, judge, embedder)
        _, opinion = Rewriter(kb, judge, make("python"), None, claims=check).rewrite_opinions([GAMES[False]])
        (readings,) = check.readings([Rewrite("wastes_time", "play_games", ())]).values()
        assert {r.statement for r in readings} == {first, second}
        assert {r.source for r in readings} == ({first} if same else {first, second})
        assert check.defeats([Rewrite("wastes_time", "play_games", ())]) == {}
        return opinion[GAMES[False]]

    one = Opinion(0.0, CHECK_TRUST * 0.5, 1.0 - CHECK_TRUST * 0.5, 0.214)
    alone = reading(*ZERO_REWRITES["play_games"]).with_base_rate(0.214).discount(0.348)
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
        yes_no = YesNoTableJudge(table, threshold=0.03, trust=0.348, base_rate=0.214, check_trust=CHECK_TRUST)
        check = ClaimCheck(kb, yes_no, embedder)
        (readings,) = check.readings([Rewrite("wastes_time", "play_games", (0,))]).values()
        (r,) = readings
        assert (f"{BELIEF}.", negation) in yes_no.asked
        assert (r.opinion.disbelief > 0.0) is lowered and r.opinion.belief == 0.0
        assert bool(check.defeats([Rewrite("wastes_time", "play_games", (0,))])) is lowered
    assert_close(stance(reading(0.9, 0.0), reading(0.9, 0.0)), Opinion(0.09, 0.09, 0.82))


def test_read_asks_the_claim_and_for_a_yes_no_judge_its_negation() -> None:
    """read() is the check's reading of a statement against a claim, which the trust fit reuses: one pair per question
    for a judge that reads contradiction, the claim and its negation for one that does not."""
    claim, negation = "If A plays games, A wastes time.", "It is not the case that if A plays games, A wastes time."
    table = {("s.", claim): reading(0.2, 0.7), ("s.", negation): reading(0.6, 0.1)}
    nli, yes_no = SentenceTableJudge(table), YesNoTableJudge(table)
    assert read(nli, [("s.", "A plays games", "A wastes time")]) == [reading(0.2, 0.7)]
    assert nli.asked == [("s.", claim)]
    (o,) = read(yes_no, [("s.", "A plays games", "A wastes time")])
    assert_close(o, stance(reading(0.2, 0.7), reading(0.6, 0.1)))
    assert yes_no.asked == [("s.", claim), ("s.", negation)]
    assert read(nli, []) == []


def test_each_claim_is_read_once() -> None:
    kb, judge, embedder = actions(holes=True, believed=True)
    check = ClaimCheck(kb, judge, embedder)
    rewriter = Rewriter(kb, judge, make("python"), None, beliefs=16, claims=check)
    rewriter.ask("wastes_time(X)", "lazy-tabled")
    asked = (len(judge.asked), check.usage["pairs"], len(embedder.asked))
    rewriter.ask("wastes_time(X)", "lazy-tabled")
    assert (len(judge.asked) - asked[0], check.usage["pairs"], len(embedder.asked)) == (0, asked[1], asked[2])
    assert check.usage == {"claims": 2, "pairs": 1, "statements": 9}


# ---- the check as the rewriter's behaviour ----


def test_the_rewriter_checks_by_default_and_loads_nothing_where_nothing_is_stated(monkeypatch) -> None:
    """With no check given, the rewriter builds one with its judge. With holes and no belief every statement names
    someone, so it reads nothing, asks no pair and never loads the embedding model; with the belief stated it loads
    the default model once a claim has something to read, and withdraws the gaming match."""
    loaded: list[TableEmbedder] = []

    def model() -> TableEmbedder:
        loaded.append(actions(holes=True, believed=True)[2])
        return loaded[-1]

    monkeypatch.setattr(semrewrite.retrieve, "EmbeddingRetriever", model)
    kb, judge, _ = actions(holes=True, believed=False)
    rewriter = Rewriter(kb, judge, make("python"), None)
    assert isinstance(rewriter.claims, ClaimCheck) and not rewriter.claims.reads
    out = rewriter.ask("wastes_time(X)")
    assert out.answers == {("sam",), ("kim",)} and out.defeated == {} and loaded == []
    assert rewriter.claims.usage["pairs"] == 0
    kb, judge, _ = actions(holes=True, believed=True)
    rewriter = Rewriter(kb, judge, make("python"), None)
    out = rewriter.ask("wastes_time(X)")
    assert out.answers == {("sam",)} and set(out.defeated) == {GAMES[True]} and len(loaded) == 1


def test_a_judge_reading_no_sentences_cannot_check_what_the_knowledge_base_states() -> None:
    """A judge of predicate names, as the oracle is, has no check where every statement names someone, and is refused
    where the knowledge base states a sentence naming no one, which the check would have to read."""
    holes, _, _ = actions(holes=True, believed=False)
    names = TableJudge({("watch_tv", "wastes_time", (0,)): 0.9})
    rewriter = Rewriter(holes, names, make("python"), None)
    assert rewriter.claims is None and rewriter.ask("wastes_time(X)").answers == {("sam",)}
    believed, _, _ = actions(holes=True, believed=True)
    with pytest.raises(ValueError, match="I consider gaming to be productive.*reads no sentences"):
        Rewriter(believed, names, make("python"), None)
    assert default_check(holes, names) is None


def test_a_judge_with_no_check_trust_is_refused_only_when_something_is_read() -> None:
    kb, judge, embedder = actions(holes=True, believed=False)
    judge.check_trust = None
    assert Rewriter(kb, judge, make("python"), None, claims=ClaimCheck(kb, judge, embedder)).ask(
        "wastes_time(X)").answers == {("sam",), ("kim",)}
    kb, judge, embedder = actions(holes=True, believed=True)
    judge.check_trust = None
    with pytest.raises(LookupError, match="no calibrated check trust for 'sentence stub'"):
        Rewriter(kb, judge, make("python"), None, claims=ClaimCheck(kb, judge, embedder)).ask("wastes_time(X)")
    with pytest.raises(ValueError, match="a check trust is a probability"):
        ClaimCheck(kb, judge, embedder, trust=1.5)


def test_the_factory_reads_each_judge_s_check_trust_and_leaves_none_where_none_was_fitted(monkeypatch, tmp_path) -> None:
    class Plain(SentenceJudge):
        def _score_sentences(self, pairs):
            return [0.5] * len(pairs)

    table = {"sets": {"wikidata": {"base_rate": 0.25}},
             "judges": {"fitted": {"default": 0.6, "trust": 0.4, "data_trust": 0.1, "check_trust": 0.7},
                        "unfitted": {"default": 0.6, "trust": 0.4, "data_trust": 0.1}}}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(table))
    monkeypatch.setattr(semrewrite.judges, "CALIBRATION", path)
    for name in ("fitted", "unfitted"):
        monkeypatch.setitem(semrewrite.judges.REGISTRY, name, (Plain, {}))
    fitted, unfitted = semrewrite.judges.make("fitted"), semrewrite.judges.make("unfitted")
    assert isinstance(fitted, SentenceJudge) and isinstance(unfitted, SentenceJudge)
    assert (fitted.threshold, fitted.trust, fitted.base_rate, fitted.check_trust) == (0.6, 0.4, 0.25, 0.7)
    overridden = semrewrite.judges.make("fitted", threshold=0.03)
    assert isinstance(overridden, SentenceJudge)
    assert (overridden.threshold, overridden.calibrated_threshold) == (0.03, 0.6)
    assert unfitted.check_trust is None
    given = semrewrite.judges.make("unfitted", check_trust=0.2)
    assert isinstance(given, SentenceJudge) and given.check_trust == 0.2
    with pytest.raises(LookupError, match="no calibrated check_trust for 'unfitted'"):
        semrewrite.judges.default_check_trust("unfitted", Path(path))


def test_the_trace_names_each_withdrawn_rewrite_and_the_statement_that_withdrew_it() -> None:
    stream = io.StringIO()
    kb, judge, embedder = actions(holes=True, believed=True)
    Rewriter(kb, judge, make("python"), None, claims=ClaimCheck(kb, judge, embedder),
             trace=TextTrace(stream)).ask("wastes_time(X)")
    lines = stream.getvalue().splitlines()
    assert ('  withdrawn: wastes_time(X0) ⇐ play_games(X0), since "I consider gaming to be productive." contradicts '
            '"If A plays games, A wastes time." (d 1.000, threshold 0.659)') in lines
    event = Defeated(Rewrite("wastes_time", "play_games", (0,)), "c", (("s.", 0.9),), 0.5)
    assert event.statements == (("s.", 0.9),)


# ---- the command line and the SNARS export ----


@pytest.fixture
def actions_cli(monkeypatch: pytest.MonkeyPatch):
    """Runs the command line on an actions example the fixture's caller picks, read by the stub judge and embedder."""
    from semrewrite import cli

    def use(holes: bool = True, believed: bool = True, belief: tuple[float, float] = (0.0, 1.0)) -> KB:
        kb, judge, embedder = actions(holes, believed, belief)
        monkeypatch.setattr(cli, "_judge", lambda args, kb: judge)
        monkeypatch.setattr(semrewrite.retrieve, "EmbeddingRetriever", lambda: embedder)
        monkeypatch.setattr(cli.formats, "load", lambda path: kb)
        # The stub judge has no calibrated data trust; the knowledge base states no wastes_time, so it would weigh
        # nothing.
        monkeypatch.setattr(semrewrite.judges, "default_data_trust", lambda name: 0.0)
        return kb

    return use


def test_ask_withdraws_kim_and_explains_what_withdrew_the_gaming_match(actions_cli, capsys) -> None:
    actions_cli()
    assert main(["ask", "actions.kb", "wastes_time(X)", "--retriever", "all", "--beliefs", "--explain"]) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0].startswith("wastes_time(sam)    P 0.4")
    assert not any(line.startswith("wastes_time(kim)") for line in lines)
    err = captured.err.splitlines()
    games = err.index("  wastes_time(X0) ⇐ play_games(X0)")
    assert err[games - 1] == "rewrites withdrawn by what the knowledge base states (1):"
    assert err[games + 1] == f'    contradicts  d 1.000 >= 0.659  "{BELIEF}."  similarity 0.720'
    assert ", 1 pair reading 2 claims in " in err[-1] and ", 1 rewrite withdrawn, " in err[-1]


def test_ask_writes_the_withdrawn_rewrites_and_readings_as_json(actions_cli, capsys) -> None:
    actions_cli()
    assert main(["ask", "actions.kb", "wastes_time(X)", "--retriever", "all", "--beliefs", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["answers"] == [["sam"]]
    (games,) = out["defeated"]
    assert games["rewrite"] == ["play_games", "wastes_time", [0]]
    assert games["claim"] == "If A plays games, A wastes time."
    (statement,) = games["statements"]
    assert (statement["statement"], statement["stance"]["disbelief"]) == (f"{BELIEF}.", 1.0)
    assert [c["rewrite"][0] for c in out["checks"]] == ["watch_tv"]
    assert out["cost"]["check"]["claims"] == 2 and out["cost"]["check"]["statements"] == 9


def test_ask_shows_no_check_where_the_knowledge_base_states_nothing_to_read(actions_cli, capsys) -> None:
    actions_cli(believed=False)
    assert main(["ask", "actions.kb", "wastes_time(X)", "--retriever", "all", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["answers"] == [["kim"], ["sam"]] and "defeated" not in out and "check" not in out["cost"]


def test_rewrites_lists_the_withdrawn_match_with_the_statement_that_withdrew_it(actions_cli, capsys) -> None:
    actions_cli()
    assert main(["rewrites", "actions.kb", "wastes_time", "--retriever", "all", "--all"]) == 0
    lines = capsys.readouterr().out.splitlines()
    games = lines.index("defeat  0.039  wastes_time(X0) ⇐ play_games(X0)")
    assert lines[games + 1] == f'          contradicts  d 1.000 >= 0.659  "{BELIEF}."  similarity 0.720'
    assert lines[0] == "admit   0.755  wastes_time(X0) ⇐ watch_tv(X0)"
    assert not any(line.startswith("reject") and "play_games" in line for line in lines)


def test_the_snars_export_carries_the_check(actions_cli, capsys) -> None:
    """Read at d 0.5, under the threshold, the belief weakens the gaming match, whose exported validity holds the
    judge's evidence and the belief's against it, added, as cumulative fusion adds evidence (Jøsang 2016, Thm. 12.2),
    so every use of the rewrite in SNARS carries it; read at d 1.000 the belief withdraws the match, which is then not
    exported at all, while watching TV's still is."""
    games = Rewrite("wastes_time", "play_games", ())
    judged = reading(*ZERO_REWRITES["play_games"]).with_base_rate(0.214).discount(0.348)
    against = Opinion(0.0, CHECK_TRUST * 0.5, 1.0 - CHECK_TRUST * 0.5, 0.214)
    r, s = judged.evidence()
    want = rule_rows(games, Opinion.from_evidence(r, s + W * against.disbelief / against.uncertainty, 0.214),
                     "semrewrite sentence stub")
    actions_cli(holes=False, belief=(0.0, 0.5))
    assert main(["rewrites", "actions.kb", "wastes_time", "--retriever", "all", "--snars"]) == 0
    lines = capsys.readouterr().out.splitlines()
    at = next(i for i, line in enumerate(lines) if '"rw-wastes_time-play_games-"' in line and "believe!" in line)
    exported = lines[at].split("((evidence ")[1].split(")")[0].split()
    assert [float(x) for x in exported] == pytest.approx(
        [float(x) for x in want[0].split("((evidence ")[1].split(")")[0].split()])
    assert lines[at + 1] == want[1]
    actions_cli(holes=False)
    assert main(["rewrites", "actions.kb", "wastes_time", "--retriever", "all", "--snars"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert not any("play_games" in line for line in lines)
    assert any('"rw-wastes_time-watch_tv-"' in line for line in lines)
