"""Purpose: tests of per-application exceptions (semrewrite.claims.Application, the rewriter's exception step): on the
actions example with per-person statements, as nli-deberta-large and bge-small read it, "kim considers gaming to be
productive." defeats kim's application of the gaming match and no other, so kim is no answer where gaming was kim's
only derivation, under every strategy the engine runs, and keeps watching TV's where kim also watches TV, with the
proof shown going through it, while amy's gaming answer and the rewrite's validity stay as they were; an application
reads only the statements naming its row's individuals that stand out against its rewrite's claim, never its
rewrite's own predicates' rows; a vaguer reading weighs on the leaf at the judge's exception trust; a statement the
claims read, as with every statement read, is not read again for an application; a run with nothing to weigh
evaluates no lineage, and the exceptions' own work counts apart from the run's; a run judge with no exception trust
hands applications to the default reader, and a reader given without one is refused once there is something to read;
the trace and the command line report the defeat; and the SNARS export writes the leaves, which SNARS, run on PeTTa,
counts as the lineage does.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from stubs import SentenceTableJudge, TableEmbedder

from semrewrite.claims import ClaimCheck, conditional, given, sentence
from semrewrite.cli import main
from semrewrite.engines import make
from semrewrite.kb import KB
from semrewrite.lineage import leaf
from semrewrite.model import Atom, Rewrite
from semrewrite.opinion import Opinion
from semrewrite.rewriter import STRATEGIES, Rewriter, Timing
from semrewrite.snars import believed, exception_rows, statement
from semrewrite.trace import TextTrace

# The actions example with holes, examples/actions.kb, and three more facts: amy plays games too, kim also watches TV,
# and kim considers gaming to be productive.
ACTIONS = {"watch_tv": ("sam", "{x} watches TV"), "play_games": ("kim", "{x} plays games"),
           "scroll_feed": ("raj", "{x} scrolls social media"), "study": ("lee", "{x} studies for an exam"),
           "exercise": ("ana", "{x} goes for a run"), "sleep_early": ("joe", "{x} goes to bed early"),
           "pay_bills": ("max", "{x} pays the bills"), "read_book": ("eve", "{x} reads a book")}
BELIEF = "kim considers gaming to be productive."
GAMES, TV = ("play_games", "wastes_time", (0,)), ("watch_tv", "wastes_time", (0,))
# Cosines of each rewrite's claim to each statement, as BAAI/bge-small-en-v1.5 gave them on 2026-11-16
# (measured for this fixture, a-cc-pa-tests): the belief stands out against the gaming claim at z 4.4 among the
# statements not of the rewrite's own predicates, and against the TV claim reaches z 0.98 only.
COSINES = {
    conditional("A plays games", "A wastes time"): {
        "sam watches TV.": 0.527, "kim plays games.": 0.657, "amy plays games.": 0.68, "kim watches TV.": 0.474,
        "raj scrolls social media.": 0.431, "lee studies for an exam.": 0.52, "ana goes for a run.": 0.5,
        "joe goes to bed early.": 0.406, "max pays the bills.": 0.484, "eve reads a book.": 0.533, BELIEF: 0.676},
    conditional("A watches TV", "A wastes time"): {
        "sam watches TV.": 0.693, "kim plays games.": 0.52, "amy plays games.": 0.516, "kim watches TV.": 0.661,
        "raj scrolls social media.": 0.465, "lee studies for an exam.": 0.512, "ana goes for a run.": 0.457,
        "joe goes to bed early.": 0.452, "max pays the bills.": 0.504, "eve reads a book.": 0.536, BELIEF: 0.547},
}
# nli-deberta-large's readings, measured the same day: the rewrites under the letters, as discovery asks them, and the
# belief with kim's gaming premise against kim's conclusion, as an application reads it.
REWRITES = {("A watches TV", "A wastes time"): (0.756, 0.001), ("A plays games", "A wastes time"): (0.039, 0.003),
            ("A scrolls social media", "A wastes time"): (0.003, 0.002)}
DEFEAT = (0.001, 0.985)
EXCEPTION_TRUST = 0.670


def reading(belief: float, disbelief: float) -> Opinion:
    """A judge's three-way reading with what belief and disbelief leave as uncertainty."""
    return Opinion(belief, disbelief, 1.0 - belief - disbelief)


def people(kim_watches: bool = False, belief: tuple[float, float] = DEFEAT, believed: bool = True,
           exception_trust: float | None = EXCEPTION_TRUST) -> tuple[KB, SentenceTableJudge, TableEmbedder]:
    """The per-person actions example, the judge reading it as nli-deberta-large did, at threshold 0.03 so the gaming
    match is admitted, its calibrated threshold 0.659 the bar, and the embedder giving bge-small's cosines; `belief`
    replaces the belief's reading against kim's gaming application."""
    glosses = {name: gloss for name, (_, gloss) in ACTIONS.items()}
    glosses |= {"wastes_time": "{x} wastes time", "gaming_productive": "{x} considers gaming to be productive"}
    facts = [Atom(name, (person,)) for name, (person, _) in ACTIONS.items()] + [Atom("play_games", ("amy",))]
    facts += [Atom("watch_tv", ("kim",))] if kim_watches else []
    facts += [Atom("gaming_productive", ("kim",))] if believed else []
    kb = KB("people", glosses=glosses, facts=tuple(facts), queries=(Atom("wastes_time", ("X",)),))
    table = {pair: reading(*bd) for pair, bd in REWRITES.items()}
    table[(given("kim plays games", BELIEF), sentence("kim wastes time"))] = reading(*belief)
    judge = SentenceTableJudge(table, threshold=0.03, trust=0.348, base_rate=0.214, check_trust=0.698,
                               exception_trust=exception_trust, calibrated_threshold=0.659, name="nli stub")
    cosines = {(claim, s): c for claim, row in COSINES.items() for s, c in row.items()}
    return kb, judge, TableEmbedder(cosines, default=0.45)


def ask(kb: KB, judge: SentenceTableJudge, embedder: TableEmbedder, *, beliefs: int = 16, proofs: bool = False,
        trace=None):
    rewriter = Rewriter(kb, judge, make("python"), None, beliefs=beliefs, proofs=proofs, trace=trace,
                        claims=ClaimCheck(kb, judge, embedder))
    return rewriter, rewriter.ask("wastes_time(X)", "lazy-tabled")


@pytest.mark.parametrize("beliefs", [16, 0])
def test_a_statement_about_one_person_defeats_that_persons_application_alone(beliefs: int) -> None:
    """kim's only derivation applies the gaming match to kim, which the belief clearly contradicts, so kim is no
    answer, with or without beliefs; amy's application of the same match keeps its answer, at the same P as with
    nothing stated about kim, and the rewrite's own opinion is untouched."""
    kb, judge, embedder = people()
    _, out = ask(kb, judge, embedder, beliefs=beliefs)
    assert out.answers == {("sam",), ("amy",)} and out.fallen == {("kim",)}
    assert out.applications is not None and set(out.applications) == {(GAMES, ("kim",))}
    assert out.applications[(GAMES, ("kim",))].defeated
    if beliefs:
        _, plain = ask(*people(believed=False), beliefs=beliefs)
        assert out.beliefs is not None and plain.beliefs is not None
        assert out.beliefs[("amy",)] == plain.beliefs[("amy",)] and out.beliefs[("sam",)] == plain.beliefs[("sam",)]
        assert out.rewrite_beliefs is not None and plain.rewrite_beliefs is not None
        assert out.rewrite_beliefs[GAMES] == plain.rewrite_beliefs[GAMES]


# Every strategy that rewrites and that the python engine can run.
RUNNABLE = sorted(name for name, s in STRATEGIES.items() if s.timing is not Timing.NONE
                  and all(getattr(make("python").capabilities, c) for c, need in s.needs().items() if need))


@pytest.mark.parametrize("strategy", RUNNABLE)
def test_every_strategy_drops_the_answer_of_the_defeated_application_alone(strategy: str) -> None:
    """The exception step reads the lineage over the knowledge base and the admitted rewrites, not the program the
    engine ran, so under every strategy, alias-canonical's merged predicates included, kim falls and amy stays."""
    kb, judge, embedder = people()
    rewriter = Rewriter(kb, judge, make("python"), None, claims=ClaimCheck(kb, judge, embedder))
    out = rewriter.ask("wastes_time(X)", strategy)
    assert out.answers == {("sam",), ("amy",)} and out.fallen == {("kim",)}


def test_another_derivation_keeps_the_answer_and_the_proof_shown_is_one_the_exception_allows() -> None:
    """Where kim also watches TV, kim stays an answer through watching TV alone, at that match's P, and the proof
    shown applies it, though the engine's first proof went through the defeated gaming application."""
    kb, judge, embedder = people(kim_watches=True)
    _, out = ask(kb, judge, embedder, proofs=True)
    assert ("kim",) in out.answers and out.fallen == set()
    assert out.beliefs is not None and out.rewrite_beliefs is not None
    assert out.beliefs[("kim",)].projection == pytest.approx(out.rewrite_beliefs[TV].projection, abs=1e-12)
    assert out.beliefs[("kim",)] == out.beliefs[("sam",)]
    assert out.proofs is not None
    proof = out.proofs[("kim",)]
    assert isinstance(proof.by, Rewrite) and proof.by.key == TV
    assert [c.atom for c in proof.children] == [Atom("watch_tv", ("kim",))]


def test_an_application_reads_only_the_statements_naming_its_individuals_that_stand_out() -> None:
    """kim's gaming application has two statements about kim outside its own predicates, and reads the one standing
    out against the gaming claim; kim's TV application keeps neither; amy's has none; and no application reads its
    rewrite's own predicates' rows."""
    kb, judge, embedder = people(kim_watches=True)
    check = ClaimCheck(kb, judge, embedder)
    read = check.applications({GAMES: [("kim",), ("amy",)], TV: [("kim",), ("sam",)]})
    assert set(read) == {(GAMES, ("kim",))}
    assert [r.statement for r in read[(GAMES, ("kim",))].readings] == [BELIEF]
    assert judge.asked == [(given("kim plays games", BELIEF), sentence("kim wastes time"))]
    assert check.usage == {"claims": 0, "pairs": 1, "statements": 11, "applications": 1}
    assert check.applications({GAMES: [("kim",)]}) == {(GAMES, ("kim",)): read[(GAMES, ("kim",))]}
    assert len(judge.asked) == 1


def test_a_statement_the_claims_read_is_not_read_again_for_an_application() -> None:
    """With every statement read against the claims, the belief is read against the gaming claim and defeats the
    rewrite there, as before applications were read, and no application reads anything, so its evidence counts
    once."""
    kb, judge, embedder = people()
    judge.table[(BELIEF, conditional("A plays games", "A wastes time"))] = reading(*DEFEAT)
    check = ClaimCheck(kb, judge, embedder, with_arguments=True)
    assert not check.individuals
    assert check.applications({GAMES: [("kim",), ("amy",)]}) == {} and "applications" not in check.usage
    assert set(check.defeats([Rewrite("wastes_time", "play_games", (0,))])) == {GAMES}


def test_a_vaguer_reading_weighs_on_the_leaf_at_the_exception_trust() -> None:
    """Read at d 0.4, under the bar, the belief only weighs: kim's leaf projects 1 - 0.670 x 0.4, and kim's answer is
    the gaming match's P times it."""
    kb, judge, embedder = people(belief=(0.1, 0.4))
    _, out = ask(kb, judge, embedder)
    assert out.applications is not None
    app = out.applications[(GAMES, ("kim",))]
    assert not app.defeated
    assert app.normal.projection == pytest.approx(1 - EXCEPTION_TRUST * 0.4, abs=1e-12)
    assert out.beliefs is not None and out.rewrite_beliefs is not None
    assert out.beliefs[("kim",)].projection == pytest.approx(
        out.rewrite_beliefs[GAMES].projection * (1 - EXCEPTION_TRUST * 0.4), abs=1e-12)
    assert out.beliefs[("amy",)].projection == pytest.approx(out.rewrite_beliefs[GAMES].projection, abs=1e-12)


def test_a_neutral_reading_leaves_every_answer_as_it_was() -> None:
    """Read as neutral, the belief gives kim's application a certain leaf, which the lineage leaves out."""
    kb, judge, embedder = people(belief=(0.0, 0.0))
    _, out = ask(kb, judge, embedder)
    _, plain = ask(*people(believed=False))
    assert out.answers == plain.answers and out.beliefs == plain.beliefs and out.fallen == set()


def test_a_judge_without_an_exception_trust_hands_applications_to_the_default_reader(monkeypatch) -> None:
    """jev's case: the run's judge has no exception trust, so the default reader reads kim's application, the
    substitution is recorded and traced with why, and the defeat stands as before; the process holds that reader once,
    for every later check and whatever else asks for it; a reader given without one is refused, naming the experiment
    that fits it; and with nothing to read nothing is loaded."""
    import semrewrite.claims
    import semrewrite.judges

    loaded: list[SentenceTableJudge] = []

    def default(name: str, **_: object) -> SentenceTableJudge:
        loaded.append(people()[1])
        loaded[-1].name = name
        return loaded[-1]

    monkeypatch.setattr(semrewrite.judges, "make", default)
    monkeypatch.setattr(semrewrite.claims, "_SHARED", {})
    _, out = ask(*people(believed=False, exception_trust=None))
    assert out.answers == {("sam",), ("kim",), ("amy",)} and loaded == []
    kb, judge, embedder = people(exception_trust=None)
    stream = io.StringIO()
    rewriter, out = ask(kb, judge, embedder, trace=TextTrace(stream))
    assert out.answers == {("sam",), ("amy",)} and len(loaded) == 1
    assert rewriter.claims is not None
    assert rewriter.claims.substituted == {"exception_trust": ("nli-deberta-large", "nli stub has no exception trust")}
    assert ("the claim check reads statements about applications' individuals with nli-deberta-large: nli stub has "
            "no exception trust") in stream.getvalue()
    assert semrewrite.claims.default_reader() is loaded[0] and len(loaded) == 1
    check = ClaimCheck(kb, judge, embedder, reader=judge)
    with pytest.raises(LookupError, match="exception trust .* experiments/exception_trust.py"):
        Rewriter(kb, judge, make("python"), None, beliefs=16, claims=check).ask("wastes_time(X)")


@pytest.mark.parametrize("beliefs", [16, 0])
def test_the_exceptions_work_is_counted_apart_and_only_where_something_may_be_read(beliefs: int) -> None:
    """With no statement about an individual a rewrite's applications can bind, a run without beliefs evaluates no
    lineage and loads no model, and every run counts in `work` what it counted before the exceptions: the engine's, and
    with beliefs the lineage's. With the belief stated, the lineages and the solve the exceptions add count in
    `exception_work`, and `work` is the run's without the belief."""
    kb, judge, embedder = people(believed=False)
    check = ClaimCheck(kb, judge, embedder)
    out = Rewriter(kb, judge, make("python"), None, beliefs=beliefs, claims=check).ask("wastes_time(X)")
    assert out.exception_work == 0 and embedder.asked == []
    assert (out.applications is None) == (beliefs == 0)
    kb, judge, embedder = people()
    excepted = Rewriter(kb, judge, make("python"), None, beliefs=beliefs, proofs=True,
                        claims=ClaimCheck(kb, judge, embedder)).ask("wastes_time(X)")
    assert excepted.fallen == {("kim",)} and excepted.exception_work > 0 and excepted.work == out.work


def test_without_the_models_extra_a_run_with_a_statement_to_weigh_names_the_extra(monkeypatch, capsys) -> None:
    """An install without the models extra, such as Jev's extra alone: where no application has a statement about its
    individuals the check loads nothing and the run answers; where one has, the gate needs the embedding model, and a
    run judge with no exception trust needs nli-deberta-large, so the run stops with an error saying what for and
    naming the extra and the command that installs it, which the command line prints as it prints every missing
    library's."""
    import semrewrite.claims
    import semrewrite.judges
    import semrewrite.retrieve
    from semrewrite import cli

    def missing(*_: object, **__: object) -> None:
        raise ModuleNotFoundError("No module named 'sentence_transformers'", name="sentence_transformers")

    monkeypatch.setattr(semrewrite.retrieve, "EmbeddingRetriever", missing)
    monkeypatch.setattr(semrewrite.judges, "make", missing)
    monkeypatch.setattr(semrewrite.claims, "_SHARED", {})
    kb, judge, _ = people(believed=False, exception_trust=None)
    out = Rewriter(kb, judge, make("python"), None, claims=ClaimCheck(kb, judge)).ask("wastes_time(X)", "lazy-tabled")
    assert out.answers == {("sam",), ("kim",), ("amy",)} and out.applications is None
    kb, judge, embedder = people(exception_trust=None)
    extra = r"which needs sentence_transformers, part of the models extra: pip install 'semantic-rewriter\[models\]'"
    with pytest.raises(ModuleNotFoundError, match=r"chooses the statements it reads by an embedding model's cosines, "
                       + extra):
        Rewriter(kb, judge, make("python"), None, claims=ClaimCheck(kb, judge)).ask("wastes_time(X)", "lazy-tabled")
    with pytest.raises(ModuleNotFoundError, match="reads with nli-deberta-large where nli stub has no exception "
                       "trust, " + extra):
        Rewriter(kb, judge, make("python"), None, claims=ClaimCheck(kb, judge, embedder)).ask("wastes_time(X)",
                                                                                                "lazy-tabled")
    monkeypatch.setattr(cli, "_judge", lambda args, kb: judge)
    monkeypatch.setattr(cli.formats, "load", lambda path: kb)
    monkeypatch.setattr(semrewrite.judges, "default_data_trust", lambda name: 0.0)
    assert main(["ask", "people.kb", "wastes_time(X)", "--retriever", "all"]) == 1
    assert capsys.readouterr().err == ("semrewrite: sentence_transformers is not installed; "
                                       "pip install 'semantic-rewriter[models]'\n")


def test_the_trace_reports_each_defeated_application() -> None:
    stream = io.StringIO()
    ask(*people(), trace=TextTrace(stream))
    assert ('  excepted: wastes_time(X0) ⇐ play_games(X0) for (kim), since "kim considers gaming to be productive." '
            'with "kim plays games." contradicts "kim wastes time." (d 0.985, threshold 0.659)') in stream.getvalue()


def test_the_command_line_shows_the_defeated_application_and_the_answer_it_left(monkeypatch, capsys) -> None:
    import semrewrite.judges
    import semrewrite.retrieve
    from semrewrite import cli

    kb, judge, embedder = people()
    monkeypatch.setattr(cli, "_judge", lambda args, kb: judge)
    monkeypatch.setattr(semrewrite.retrieve, "EmbeddingRetriever", lambda: embedder)
    monkeypatch.setattr(cli.formats, "load", lambda path: kb)
    # The stub judge has no calibrated data trust; the knowledge base states no wastes_time, so it would weigh nothing.
    monkeypatch.setattr(semrewrite.judges, "default_data_trust", lambda name: 0.0)
    assert main(["ask", "people.kb", "wastes_time(X)", "--retriever", "all", "--beliefs", "--explain"]) == 0
    printed = capsys.readouterr()
    assert [line.split("(")[1].split(")")[0] for line in printed.out.splitlines() if line.startswith("wastes_time")] \
        == ["sam", "amy"]
    assert ("applications defeated by what the knowledge base states about their individuals (1):\n"
            '  wastes_time(kim) ⇐ play_games(kim), given "kim plays games."\n'
            '    contradicts  d 0.985 >= 0.659  "kim considers gaming to be productive."  similarity 0.676') in printed.err
    assert "answers a defeat leaves no derivation for (1): wastes_time(kim)" in printed.err
    assert re.search(r"1 application read, 1 defeated, [1-9][0-9,]* work on exceptions", printed.err)
    assert main(["ask", "people.kb", "wastes_time(X)", "--retriever", "all", "--json"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["answers"] == [["amy"], ["sam"]] and got["fallen"] == [["kim"]] and got["cost"]["exception_work"] > 0
    (exception,) = got["exceptions"]
    assert exception["rewrite"] == ["play_games", "wastes_time", [0]] and exception["row"] == ["kim"]
    assert exception["defeated"] and [s["statement"] for s in exception["statements"]] == [BELIEF]


@pytest.mark.parametrize(("belief", "listed"), [((0.5, 0.3), True), ((0.9, 0.0), False)])
def test_explain_lists_a_weighed_application_only_where_it_weighs_on_an_answer(monkeypatch, capsys, belief, listed) -> None:
    """A belief read against kim's gaming application below the bar weighs on kim's P, so --explain with beliefs lists
    the application; one that disbelieves nothing leaves the leaf certain and out of the lineage, so it is only counted.
    Without beliefs nothing weighs, and only the cost line counts what was read."""
    import semrewrite.judges
    import semrewrite.retrieve
    from semrewrite import cli

    kb, judge, embedder = people(kim_watches=True, belief=belief)
    monkeypatch.setattr(cli, "_judge", lambda args, kb: judge)
    monkeypatch.setattr(semrewrite.retrieve, "EmbeddingRetriever", lambda: embedder)
    monkeypatch.setattr(cli.formats, "load", lambda path: kb)
    monkeypatch.setattr(semrewrite.judges, "default_data_trust", lambda name: 0.0)
    assert main(["ask", "people.kb", "wastes_time(X)", "--retriever", "all", "--beliefs", "--explain"]) == 0
    err = capsys.readouterr().err
    assert ("applications weighed by what the knowledge base states about their individuals (1):\n"
            "  wastes_time(kim) ⇐ play_games(kim)    normal") in err if listed else "applications weighed" not in err
    assert ("1 other application read weighs on no answer" in err) is not listed
    assert "1 application read, 0 defeated" in err
    assert main(["ask", "people.kb", "wastes_time(X)", "--retriever", "all", "--explain"]) == 0
    err = capsys.readouterr().err
    assert "applications weighed" not in err and "other application" not in err and "1 application read" in err


def test_a_rewrite_with_leaves_becomes_two_certain_rows_and_a_normal_statement_per_row() -> None:
    """Rows nothing was read about take the certain leaf, a defeated row none, and the rewrite's validity is written
    as rule_rows writes it."""
    rewrite = Rewrite("wastes_time", "play_games", (0,))
    lines = exception_rows(rewrite, Opinion(0.6, 0.1, 0.3, 0.214), [("kim",), ("amy",), ("lee",)],
                           {("kim",): Opinion(0.0, 1.0, 0.0, 1.0), ("lee",): Opinion(0.2, 0.3, 0.5, 1.0)})
    assert lines[0].startswith('!(believe! (sh (tag P d s ()) "valid" ')
    assert lines[1].startswith("(sn rw rw-wastes_time-play_games-0-applies certain-rule ")
    assert lines[2].startswith("(sn rw rw-wastes_time-play_games-0-normal certain-rule ")
    normal = [line for line in lines if '"normal"' in line and line.startswith("!(believe!")]
    assert [re.findall(r'"(amy|kim|lee)"', line) for line in normal] == [["amy"], ["lee"]]
    assert "(evidence 2000.0 0.0) (rate 1.0)" in normal[0] and "(evidence 0.8 1.2) (rate 1.0)" in normal[1]


_NEEDS_SNARS = pytest.mark.skipif(
    not (os.environ.get("PETTA_PATH") and os.environ.get("METTABASE_PATH") and shutil.which("swipl")),
    reason="the SNARS check needs PETTA_PATH, METTABASE_PATH and swipl",
)


def _snars(lines: list[str]) -> str:
    """PeTTa's standard output on SNARS loaded and `lines`, run as tests/test_snars.py runs it."""
    petta, mettabase = Path(os.environ["PETTA_PATH"]), Path(os.environ["METTABASE_PATH"])
    path = [str(mettabase / "src"), str(mettabase / "lib"), str(petta / "python"), os.environ.get("PYTHONPATH", "")]
    result = subprocess.run(
        [shutil.which("swipl") or "swipl", "--stack_limit=8g", "-q", "-s", petta / "src" / "main.pl", "--",
         "/dev/stdin", "silent"],
        input="\n".join([f'!(import! &self "{mettabase / "lib" / "mettabase" / "snars"}")', *lines, ""]),
        cwd=mettabase, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in path if p)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@_NEEDS_SNARS
def test_snars_counts_an_application_s_leaf_as_the_lineage_does() -> None:
    """amy's answer through the gaming rewrite, whose application to amy carries a leaf, fuses to the product of the
    fact's, the validity's and the leaf's P as SNARS reads them back, the conjunction the lineage counts, and no
    disbelief; kim's application is defeated, so kim has no proof at all."""
    # The fact is not dogmatic, as in tests/test_snars.py: SNARS floors uncertainty at 0.001, which moves a dogmatic
    # fact's P from what its evidence reads back as, while a leaf nothing was read about, at base rate 1, stays at 1.
    fact, validity, normal = Opinion(0.8, 0.0, 0.2), Opinion(0.6, 0.1, 0.3, 0.214), Opinion(0.2, 0.3, 0.5, 1.0)
    rewrite = Rewrite("wastes_time", "play_games", (0,))
    world = [believed(Atom("play_games", ("amy",)), fact, "kb"), believed(Atom("play_games", ("kim",)), fact, "kb"),
             *exception_rows(rewrite, validity, [("amy",), ("kim",)],
                             {("amy",): normal, ("kim",): Opinion(0.0, 1.0, 0.0, 1.0)})]
    out = _snars([*world, f"!(let $r (collapse (sn prove {statement('wastes_time', ('amy',))} 3)) (Proved $r))",
                  f"!(let $r (collapse (sn prove {statement('wastes_time', ('kim',))} 3)) (Proved $r))"])
    proved = re.findall(r"\(Proved \((.*?)\)\)\s*$", out, re.MULTILINE)
    assert len(proved) == 2 and proved[1].strip() == "", out
    proofs = re.findall(r"\(: \(Opinion (\S+) (\S+) (\S+) (\S+)\) (\(.*?\)) \(sh ", proved[0])
    assert len(proofs) == 1, out
    fused = _snars([*world, f"!(Fused (sn fuse-proofs ({proved[0]})))"])
    found = re.search(r"\(Fused \(Opinion (\S+) (\S+) (\S+) (\S+)\)\)", fused)
    assert found, fused
    b, d, u, _ = (float(x) for x in found.groups())
    # The P SNARS reads back from the evidence written, (r + 2a) / (r + s + 2) (J 3.11).
    def back(o: Opinion) -> float:
        r, s = o.evidence()
        return (r + 2.0 * o.base_rate) / (r + s + 2.0)
    assert b == pytest.approx(back(fact) * back(validity) * back(normal), abs=1e-9)
    assert d == 0.0 and b + u == pytest.approx(1.0, abs=1e-12)
    assert leaf(rewrite.key, ("amy",)) == ("play_games", "wastes_time", (0,), ("amy",))


@_NEEDS_SNARS
def test_snars_conditions_on_a_validity_two_applications_share_and_one_application_s_leaf() -> None:
    """owns(a, b2) applies owns(X0, X1) <= owned_by(X1, X0) at both links, joined by a certain transitive rule, and
    only the second application carries a leaf: SNARS counts the derivation from its distinct leaves, the two facts,
    the validity once and the leaf, which is the lineage's one set {validity, leaf} with the facts' read-back P."""
    from semrewrite.snars import certain_rule

    fact, validity, normal = Opinion(0.8, 0.0, 0.2), Opinion(0.7, 0.1, 0.2, 0.214), Opinion(0.3, 0.2, 0.5, 1.0)
    owns = Rewrite("owns", "owned_by", (1, 0))
    transitive = certain_rule("rule-0", statement("owns", ("X", "Y")), statement("owns", ("Y", "Z")),
                              statement("owns", ("X", "Z")))
    world = [believed(Atom("owned_by", ("b1", "a")), fact, "kb"), believed(Atom("owned_by", ("b2", "b1")), fact, "kb"),
             *exception_rows(owns, validity, [("a", "b1"), ("b1", "b2")], {("b1", "b2"): normal}), transitive]
    out = _snars([*world, f"!(let $r (collapse (sn prove {statement('owns', ('a', 'b2'))} 4)) (Proved $r))"])
    (proved,) = re.findall(r"\(Proved \((.*?)\)\)\s*$", out, re.MULTILINE)
    fused = _snars([*world, f"!(Fused (sn fuse-proofs ({proved})))"])
    found = re.search(r"\(Fused \(Opinion (\S+) (\S+) (\S+) (\S+)\)\)", fused)
    assert found, fused
    b, d, _, _ = (float(x) for x in found.groups())

    def back(o: Opinion) -> float:
        r, s = o.evidence()
        return (r + 2.0 * o.base_rate) / (r + s + 2.0)

    assert b == pytest.approx(back(fact) ** 2 * back(validity) * back(normal), abs=1e-9) and d == 0.0
    # Counted twice, as step-by-step composition counts it, the validity would put the answer far lower.
    assert b - back(fact) ** 2 * back(validity) ** 2 * back(normal) > 0.05


def test_the_per_person_examples_are_the_actions_one_with_three_facts_added() -> None:
    """examples/actions_people.kb is examples/actions.kb with amy also playing games and kim also watching TV, and
    examples/actions_people_believed.kb is that with the belief about kim stated as a gloss and a fact naming kim,
    nothing else changed; this test's knowledge base states the same."""
    root = Path(__file__).resolve().parents[1] / "examples"

    def stated(name: str) -> list[str]:
        return [line for line in (root / name).read_text().splitlines() if line.strip() and not line.startswith("%")]

    base, people_, believed = stated("actions.kb"), stated("actions_people.kb"), stated("actions_people_believed.kb")
    at = base.index("read_book(eve).") + 1
    assert people_ == [*base[:at], "play_games(amy).", "watch_tv(kim).", *base[at:]]
    at = people_.index("watch_tv(kim).") + 1
    assert believed == [*people_[:at], 'gloss gaming_productive "{x} considers gaming to be productive".',
                        "gaming_productive(kim).", *people_[at:]]
    from semrewrite import formats

    kb, _, _ = people(kim_watches=True)
    assert set(formats.load(root / "actions_people_believed.kb").facts) == set(kb.facts)


def test_the_command_line_names_a_substituted_reader_and_takes_an_explicit_one(monkeypatch, capsys) -> None:
    """A run judge with no exception trust hands kim's application to the default reader, which ask --explain and
    ask --json name with why; --reader names the reader, and one without an exception trust is refused, naming the
    experiment that fits it."""
    import semrewrite.claims
    import semrewrite.judges
    import semrewrite.retrieve
    from semrewrite import cli

    kb, judge, embedder = people(exception_trust=None)
    reader = people()[1]
    reader.name = "nli-deberta-large"
    monkeypatch.setattr(cli, "_judge", lambda args, kb: judge)
    monkeypatch.setattr(semrewrite.retrieve, "EmbeddingRetriever", lambda: embedder)
    monkeypatch.setattr(cli.formats, "load", lambda path: kb)
    monkeypatch.setattr(semrewrite.judges, "default_data_trust", lambda name: 0.0)
    monkeypatch.setattr(semrewrite.judges, "make", lambda name, **_: reader if name == "nli-deberta-large" else judge)
    monkeypatch.setattr(semrewrite.claims, "_SHARED", {})
    assert main(["ask", "people.kb", "wastes_time(X)", "--retriever", "all", "--explain"]) == 0
    err = capsys.readouterr().err
    assert ("statements about applications' individuals read by nli-deberta-large: nli stub has no exception trust"
            in err)
    assert main(["ask", "people.kb", "wastes_time(X)", "--retriever", "all", "--json"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["readers"] == {"exception_trust": {"judge": "nli-deberta-large", "why": "nli stub has no exception trust"}}
    assert got["fallen"] == [["kim"]]
    assert main(["ask", "people.kb", "wastes_time(X)", "--retriever", "all", "--reader", "stub"]) == 1
    assert "no calibrated exception trust for 'nli stub'" in capsys.readouterr().err
