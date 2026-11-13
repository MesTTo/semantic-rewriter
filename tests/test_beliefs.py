"""Purpose: tests of answer beliefs: each answer's belief is the probability that the rewrites of some derivation are
all valid at their scores, a rewrite reused counting once and independent derivations combining; the command line
ranks answers by belief and drops those under --min-belief; and a strategy answering over merged predicates refuses
beliefs as it refuses proofs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from semrewrite.cli import main
from semrewrite.engines import available, make
from semrewrite.kb import KB
from semrewrite.model import Atom, Proposal, Rule
from semrewrite.rewriter import CapabilityError, Rewriter

GLOSSES = {"mum": "{x} is the mum of {y}", "dad": "{x} is the dad of {y}", "guardian": "{x} is the guardian of {y}",
           "parent": "{x} is a parent of {y}", "grandparent": "{x} is a grandparent of {y}"}
KB_ = KB("tom", glosses=GLOSSES,
         facts=(Atom("mum", ("alice", "tom")), Atom("dad", ("bob", "tom")), Atom("guardian", ("erin", "tom")),
                Atom("mum", ("carol", "alice")), Atom("dad", ("carol", "alice"))),
         rules=(Rule(Atom("grandparent", ("X", "Z")), (Atom("parent", ("X", "Y")), Atom("parent", ("Y", "Z")))),))
SCORES = {("mum", "parent", (0, 1)): 0.9, ("dad", "parent", (0, 1)): 0.8, ("guardian", "parent", (0, 1)): 0.3}


@dataclass
class StubJudge:
    """Scores from a fixed table keyed by proposal key; 0 for anything else."""

    table: dict
    name: str = "stub"
    threshold: float = 0.2
    usage: dict[str, int] = field(default_factory=lambda: {"pairs": 0})

    def score(self, proposals: Sequence[Proposal]) -> list[float]:
        return [self.table.get(p.key, 0.0) for p in proposals]


@pytest.mark.parametrize("engine_name", available())
def test_each_answer_s_belief_is_the_probability_some_derivation_s_rewrites_all_hold(engine_name: str) -> None:
    rewriter = Rewriter(KB_, StubJudge(SCORES), make(engine_name), None, beliefs=16)
    parents = rewriter.ask("parent(X, tom)", "lazy-tabled")
    assert parents.beliefs == pytest.approx({("alice", "tom"): 0.9, ("bob", "tom"): 0.8, ("erin", "tom"): 0.3})
    grand = rewriter.ask("grandparent(X, tom)", "lazy-tabled")
    # carol is alice's mum and dad: two derivations, through mum twice or through dad and then mum; mum reused counts
    # once, so the belief is P(mum valid) = 0.9 whatever dad is.
    assert grand.beliefs == pytest.approx({("carol", "tom"): 0.9})


def test_beliefs_are_refused_where_answers_are_over_merged_predicates() -> None:
    rewriter = Rewriter(KB_, StubJudge(SCORES), make("python"), None, beliefs=16)
    with pytest.raises(CapabilityError, match="no lineage"):
        rewriter.ask("parent(X, tom)", "alias-canonical")


def test_beliefs_rank_on_the_command_line(monkeypatch, capsys) -> None:
    from semrewrite import cli

    monkeypatch.setattr(cli, "_judge", lambda args, kb: StubJudge(SCORES))
    monkeypatch.setattr(cli.formats, "load", lambda path: KB_)
    assert main(["ask", "tom.kb", "parent(X, tom)", "--retriever", "all", "--beliefs", "--min-belief", "0.5"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["parent(alice, tom)    belief 0.900", "parent(bob, tom)    belief 0.800"]
