"""Purpose: tests of the trace: an expansion emits the retrieval, the judge's scores and the admitted rewrites, a
repeated ask emits a memo event, a solve emits the engine's start and stop around the expansions it calls, and
TextTrace prints each admitted proposal beside its premise read the other way, the highest of the rest, retrieval
ranks only when retrieval kept fewer than all, and a summary of the memo's answers; `semrewrite ask --trace` prints
the trace on standard error and leaves standard output to the answers.
"""

from __future__ import annotations

import io

from stubs import TableJudge

from semrewrite import formats
from semrewrite.cli import main
from semrewrite.discovery import Discoverer, ScoreCache
from semrewrite.engines.python import PythonEngine
from semrewrite.kb import KB
from semrewrite.model import Atom, Predicate
from semrewrite.protocols import Program, Solution
from semrewrite.rewriter import Rewriter
from semrewrite.trace import Expanded, Remembered, Solved, Solving, TextTrace

CATALOGUE = {p.name: p for p in (
    Predicate("owns", "{x} owns {y}"),
    Predicate("owned_by", "{x} is owned by {y}"),
    Predicate("beforehand_owned_by", "{x} is beforehand owned by {y}"),
    Predicate("developer", "{x} is the developer of {y}"),
    Predicate("parent", "{x} is the parent of {y}"),
)}
SCORES = {
    ("owned_by", "owns", (1, 0)): 0.99,
    ("owned_by", "owns", (0, 1)): 0.02,
    ("beforehand_owned_by", "owns", (1, 0)): 0.62,
}


class StubRanking:
    """Ranks the catalogue in a fixed order, as a retriever that keeps the k best would."""

    name = "stub ranking"

    def top_k(self, goal: Predicate, among, k: int) -> list[tuple[str, float]]:
        order = ["owned_by", "beforehand_owned_by", "developer", "parent"]
        return [(n, 1.0 - i / 10) for i, n in enumerate(n for n in order if n in among)][:k]


def test_an_expansion_emits_what_was_retrieved_judged_and_admitted_and_a_repeat_comes_from_the_memo() -> None:
    events: list = []
    disc = Discoverer(CATALOGUE, StubRanking(), ScoreCache(TableJudge(SCORES, threshold=0.65)), k=2, trace=events.append)
    first = disc.rewrites("owns")
    again = disc.rewrites("owns")
    assert first == again and [rw.key for rw in first] == [("owned_by", "owns", (1, 0))]
    expanded, remembered = events
    assert isinstance(expanded, Expanded) and remembered == Remembered("owns")
    assert expanded.among == 4 and expanded.candidates == (("owned_by", 1.0), ("beforehand_owned_by", 0.9))
    assert expanded.retriever == "stub ranking" and expanded.judge == "stub" and expanded.threshold == 0.65
    # Two candidates under both mappings, and the goal against itself with its arguments swapped.
    assert len(expanded.proposals) == len(expanded.scores) == 5
    assert expanded.admitted == first


def test_a_lazy_solve_emits_its_start_and_stop_around_the_expansions_evaluation_calls() -> None:
    kb = KB("owning", glosses={n: p.gloss for n, p in CATALOGUE.items()},
            facts=(Atom("owned_by", ("youtube", "google")),))
    events: list = []
    rewriter = Rewriter(kb, TableJudge(SCORES, threshold=0.65), PythonEngine(), StubRanking(), k=2, trace=events.append)
    out = rewriter.ask("owns(google, X)", "lazy-tabled")
    assert out.answers == {("google", "youtube")}
    assert isinstance(events[0], Solving) and isinstance(events[-1], Solved)
    assert events[0].query == Atom("owns", ("google", "X")) and events[-1].solution.answers == out.answers
    assert next(e.goal.name for e in events if isinstance(e, Expanded)) == "owns"


def test_the_text_trace_prints_each_admission_beside_its_premise_read_the_other_way() -> None:
    events: list = []
    disc = Discoverer(CATALOGUE, StubRanking(), ScoreCache(TableJudge(SCORES, threshold=0.65)), k=2, trace=events.append)
    disc.rewrites("owns")
    disc.rewrites("owns")
    stream = io.StringIO()
    trace = TextTrace(stream)
    program = Program(facts=(Atom("owned_by", ("youtube", "google")),))
    trace(Solving("python", Atom("owns", ("google", "X")), program))
    for e in events:
        trace(e)
    trace(Solved("python", Solution(answers={("google", "youtube")}, work=3)))
    assert stream.getvalue().splitlines() == [
        "python starts on owns(google, X), over 1 fact and 0 rules, with no rewrites yet",
        "evaluation reached owns, which no fact states, and asks for the rewrites concluding it",
        ('  the stub ranking retriever ranked the 4 predicates of the same arity as premises of "A owns B" and kept '
         "the 2 it ranks highest"),
        ('  stub was asked 5 questions, whether each premise sentence entails "A owns B", and admits a score of '
         "0.650 or more:"),
        '    admit  0.990  "B is owned by A"  owned_by, ranked 1',
        '           0.020  "A is owned by B"  owned_by, ranked 1',
        '    3 more rejected, the highest at 0.620: "B is beforehand owned by A" (beforehand_owned_by)',
        "  admitted as rules:",
        "    owns(X0, X1) ⇐ owned_by(X1, X0)",
        "python finished: ok, 1 answer, 3 work",
        "  the memo answered 1 more ask for owns, without asking the judge again",
    ]


def test_a_candidate_has_no_rank_when_every_predicate_is_one() -> None:
    events: list = []
    Discoverer(CATALOGUE, None, ScoreCache(TableJudge(SCORES, threshold=0.65)), trace=events.append).rewrites("owns")
    stream = io.StringIO()
    TextTrace(stream)(events[0])
    lines = stream.getvalue().splitlines()
    assert lines[0] == "before evaluation, discovery expands owns, which no fact states"
    assert lines[1] == "  every one of the 4 predicates of the same arity is a candidate"
    assert '    admit  0.990  "B is owned by A"  owned_by' in lines


def test_ask_trace_prints_the_trace_on_stderr_and_the_answers_on_stdout(capsys) -> None:
    kb = "examples/parent_of_tom.kb"
    assert formats.load(kb).gold
    status = main(["ask", kb, "grandparent(X, tom)", "--judge", "oracle", "--retriever", "all", "--trace"])
    out, err = capsys.readouterr()
    assert status == 0
    assert out.splitlines() == ["grandparent(carol, tom)", "grandparent(dave, tom)"]
    assert err.startswith("python starts on grandparent(X, tom), over ")
    assert "evaluation reached parent, which no fact states, and asks for the rewrites concluding it" in err
    assert "    parent(X0, X1) ⇐ child_of(X1, X0)" in err.splitlines()
    assert "python finished: ok, 2 answers" in err
