"""Purpose: tests of answer opinions: each answer's opinion has the exact mean and variance of the probability that
the rewrites of some derivation are all valid, each rewrite valid as its judge's trusted opinion of it says, a rewrite
reused counting once and independent derivations combining; a judge trusted less leaves more uncertainty; an answer
the knowledge base states needs no rewrite and is certain; the command line ranks answers by projected probability
and drops those under --answer-threshold; and a strategy answering over merged predicates refuses opinions as it
refuses proofs.
"""

from __future__ import annotations

import pytest
from stubs import TableJudge

from semrewrite.cli import main
from semrewrite.engines import available, make
from semrewrite.kb import KB
from semrewrite.model import Atom, Rule
from semrewrite.opinion import W
from semrewrite.rewriter import CapabilityError, Rewriter

GLOSSES = {"mum": "{x} is the mum of {y}", "dad": "{x} is the dad of {y}", "guardian": "{x} is the guardian of {y}",
           "parent": "{x} is a parent of {y}", "grandparent": "{x} is a grandparent of {y}"}
KB_ = KB("tom", glosses=GLOSSES,
         facts=(Atom("mum", ("alice", "tom")), Atom("dad", ("bob", "tom")), Atom("guardian", ("erin", "tom")),
                Atom("mum", ("carol", "alice")), Atom("dad", ("carol", "alice"))),
         rules=(Rule(Atom("grandparent", ("X", "Z")), (Atom("parent", ("X", "Y")), Atom("parent", ("Y", "Z")))),))
SCORES = {("mum", "parent", (0, 1)): 0.9, ("dad", "parent", (0, 1)): 0.8, ("guardian", "parent", (0, 1)): 0.3}


def _rewriter(engine_name: str, trust: float = 1.0) -> Rewriter:
    judge = TableJudge(SCORES, threshold=0.2, trust=trust)
    return Rewriter(KB_, judge, make(engine_name), None, beliefs=16)


@pytest.mark.parametrize("engine_name", available())
def test_a_fully_trusted_judge_s_scores_are_the_answers_probabilities(engine_name: str) -> None:
    """At complete trust each rewrite is valid with its score and no uncertainty, so each answer's opinion is the
    dogmatic one of its probability: one rewrite gives that rewrite's score, and mum reused counts once."""
    rewriter = _rewriter(engine_name)
    parents = rewriter.ask("parent(X, tom)", "lazy-tabled")
    assert parents.beliefs is not None
    assert {row: o.projection for row, o in parents.beliefs.items()} == pytest.approx(
        {("alice", "tom"): 0.9, ("bob", "tom"): 0.8, ("erin", "tom"): 0.3})
    assert all(o.uncertainty == pytest.approx(0.0, abs=1e-9) for o in parents.beliefs.values())
    grand = rewriter.ask("grandparent(X, tom)", "lazy-tabled")
    # carol is alice's mum and dad: two derivations, through mum twice or through dad and then mum; mum reused counts
    # once, so the probability is P(mum valid) = 0.9 whatever dad is.
    assert grand.beliefs is not None
    assert {row: o.projection for row, o in grand.beliefs.items()} == pytest.approx({("carol", "tom"): 0.9})


@pytest.mark.parametrize("engine_name", available())
def test_trusting_the_judge_half_leaves_half_the_mass_uncertain(engine_name: str) -> None:
    """Trust 0.5 discounts mum's (0.9, 0.1, 0) to (0.45, 0.05, 0.5): an answer through mum alone has that opinion."""
    parents = _rewriter(engine_name, trust=0.5).ask("parent(X, tom)", "lazy-tabled")
    assert parents.beliefs is not None
    alice = parents.beliefs[("alice", "tom")]
    assert (alice.belief, alice.disbelief, alice.uncertainty, alice.base_rate) == pytest.approx((0.45, 0.05, 0.5, 0.5))


@pytest.mark.parametrize("engine_name", available())
def test_two_independent_derivations_give_the_exact_mean_and_variance(engine_name: str) -> None:
    """carol is alice's parent through mum or through dad. Each rewrite is valid with a Beta-distributed probability
    of mean m and variance m (1 - m) u / (W + u); the answer holds unless both fail, so its mean is
    1 - (1 - m1)(1 - m2) and its second moment 1 - 2 (1 - m1)(1 - m2) + E[(1 - p1)^2] E[(1 - p2)^2]."""
    parents = _rewriter(engine_name, trust=0.5).ask("parent(X, alice)", "lazy-tabled")
    assert parents.beliefs is not None
    carol = parents.beliefs[("carol", "alice")]
    u = 0.5
    m1, m2 = 0.5 + 0.5 * (0.9 - 0.5), 0.5 + 0.5 * (0.8 - 0.5)
    v1, v2 = m1 * (1 - m1) * u / (W + u), m2 * (1 - m2) * u / (W + u)
    mean = 1 - (1 - m1) * (1 - m2)
    second = 1 - 2 * (1 - m1) * (1 - m2) + ((1 - m1) ** 2 + v1) * ((1 - m2) ** 2 + v2)
    assert carol.projection == pytest.approx(mean)
    assert carol.variance == pytest.approx(second - mean**2)
    # Its base rate is the coproduct's, a1 + a2 - a1 a2 (Jøsang 2016, Eq. 7.3).
    assert carol.base_rate == pytest.approx(0.75)


@pytest.mark.parametrize("engine_name", available())
def test_an_answer_the_knowledge_base_states_is_certain(engine_name: str) -> None:
    mums = _rewriter(engine_name, trust=0.5).ask("mum(X, tom)", "lazy-tabled")
    assert mums.beliefs is not None and set(mums.beliefs) == {("alice", "tom")}
    alice = mums.beliefs[("alice", "tom")]
    assert (alice.belief, alice.disbelief, alice.uncertainty, alice.base_rate) == pytest.approx((1.0, 0.0, 0.0, 1.0))


def test_beliefs_are_refused_where_answers_are_over_merged_predicates() -> None:
    with pytest.raises(CapabilityError, match="no lineage"):
        _rewriter("python").ask("parent(X, tom)", "alias-canonical")


def test_beliefs_rank_on_the_command_line(monkeypatch, capsys) -> None:
    from semrewrite import cli, judges

    monkeypatch.setattr(cli, "_judge", lambda args, kb: TableJudge(SCORES, threshold=0.2))
    monkeypatch.setattr(cli.formats, "load", lambda path: KB_)
    # The stub judge has no calibrated data trust; at 0 its opinions stand alone.
    monkeypatch.setattr(judges, "default_data_trust", lambda name: 0.0)
    assert main(["ask", "tom.kb", "parent(X, tom)", "--retriever", "all", "--beliefs", "--answer-threshold", "0.5"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["parent(alice, tom)    P 0.900  b 0.900  d 0.100  u 0.000",
                     "parent(bob, tom)    P 0.800  b 0.800  d 0.200  u 0.000"]


def test_rewrites_export_as_narsese_with_their_opinions(monkeypatch, capsys) -> None:
    """At judge trust 0.5, mum's (0.9, 0.1, 0) reads (0.45, 0.05, 0.5): f = 0.9 and, with ONA's k = 1, evidence
    r + s = W (b + d)/u = 2 gives c = 2/3."""
    from semrewrite import cli, judges

    monkeypatch.setattr(cli, "_judge", lambda args, kb: TableJudge(SCORES, threshold=0.2, trust=0.5))
    monkeypatch.setattr(cli.formats, "load", lambda path: KB_)
    monkeypatch.setattr(judges, "default_data_trust", lambda name: 0.0)
    assert main(["rewrites", "tom.kb", "parent", "--retriever", "all", "--narsese"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "<<($X0 * $X1) --> mum> ==> <($X0 * $X1) --> parent>>. %0.90;0.67%"
    assert len(lines) == 3 and all(line.endswith("%") for line in lines)
