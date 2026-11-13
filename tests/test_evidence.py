"""Purpose: tests of the knowledge base as a second source on rewrites: evidence counts each row by its membership
probability; a rewrite reads its premise's statements into its conclusion's rows exactly as the engine derives them;
each rewrite's evidence comes from the conclusion's statements and the other rewrites, never from itself; the
rewriter fuses that evidence, discounted by the data trust, with the judge's opinion, and a data trust of 0 changes
nothing; and the data trust fitted is the likeliest on a fine grid.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from stubs import TableJudge

from semrewrite.engines.python import PythonEngine
from semrewrite.evidence import derived, rewrite_evidence, statement_evidence
from semrewrite.judges.metrics import fit_data_trust
from semrewrite.kb import KB
from semrewrite.model import Atom, Rewrite
from semrewrite.opinion import Opinion
from semrewrite.protocols import Budget, Program
from semrewrite.rewriter import Rewriter

CONSTANTS = ("a", "b", "c")


@given(st.lists(st.tuples(st.tuples(st.sampled_from(CONSTANTS), st.sampled_from(CONSTANTS)), st.floats(0, 1)),
                max_size=12))
def test_each_row_counts_for_by_its_membership_and_against_by_the_rest(rows: list[tuple[tuple[str, str], float]]
                                                                       ) -> None:
    extension = dict(rows)
    premise = [row for row, _ in rows] + [("z", "z")]
    positive, negative = statement_evidence(premise, extension)
    assert positive + negative == pytest.approx(len(premise))
    assert positive == pytest.approx(sum(extension.get(row, 0.0) for row in premise))


@settings(max_examples=100)
@given(st.permutations(range(3)), st.sets(st.tuples(*[st.sampled_from(CONSTANTS)] * 3), max_size=10))
def test_a_rewrite_reads_its_premise_s_statements_as_the_engine_derives_them(mapping: list[int], facts: set) -> None:
    rewrite = Rewrite("q", "p", tuple(mapping))
    program = Program(facts=tuple(Atom("p", row) for row in sorted(facts)), rewrites=(rewrite,))
    solved = PythonEngine().solve(program, Atom("q", ("X", "Y", "Z")), Budget(work=1_000_000, seconds=math.inf))
    assert derived(rewrite, facts) == solved.answers


def test_a_rewrite_s_evidence_comes_from_the_conclusion_s_statements_and_the_other_rewrites() -> None:
    """mum derives parent(alice, tom) and parent(carol, alice); dad derives parent(carol, alice) too, and the
    knowledge base states parent(alice, tom). So mum's evidence is 1 (stated) + 0.8 (through dad) for, 0.2 against;
    dad's is 0.9 (through mum) for its one row; guardian's row nobody else derives counts against it."""
    facts = (Atom("mum", ("alice", "tom")), Atom("mum", ("carol", "alice")), Atom("dad", ("carol", "alice")),
             Atom("guardian", ("erin", "tom")), Atom("parent", ("alice", "tom")))
    mum, dad, guardian = (Rewrite("parent", p, (0, 1)) for p in ("mum", "dad", "guardian"))
    got = rewrite_evidence(facts, [mum, dad, guardian], {mum.key: 0.9, dad.key: 0.8, guardian.key: 0.3})
    assert got[mum.key] == pytest.approx((1.8, 0.2))
    assert got[dad.key] == pytest.approx((0.9, 0.1))
    assert got[guardian.key] == pytest.approx((0.0, 1.0))


GLOSSES = {"mum": "{x} is the mum of {y}", "dad": "{x} is the dad of {y}", "parent": "{x} is a parent of {y}"}
KB_ = KB("tom", glosses=GLOSSES, facts=(Atom("mum", ("alice", "tom")), Atom("mum", ("carol", "alice")),
                                         Atom("dad", ("carol", "alice")), Atom("dad", ("bob", "tom"))))
SCORES = {("mum", "parent", (0, 1)): 0.9, ("dad", "parent", (0, 1)): 0.8}


def _alice(data_trust: float) -> Opinion:
    rewriter = Rewriter(KB_, TableJudge(SCORES, threshold=0.2, trust=0.5), PythonEngine(), None, beliefs=16,
                        data_trust=data_trust)
    out = rewriter.ask("parent(X, tom)", "lazy-tabled")
    assert out.beliefs is not None
    return out.beliefs[("alice", "tom")]


def test_the_rewriter_fuses_the_knowledge_base_s_evidence_at_its_data_trust() -> None:
    """alice is tom's parent through mum alone, so her opinion is mum's: the judge's (0.9, 0.1, 0) at trust 0.5,
    fused with mum's evidence, 1 row of 2 also derived by dad (projection 0.5 + 0.5 (0.8 - 0.5) = 0.65), discounted
    by the data trust."""
    judged = Opinion(0.9, 0.1, 0.0).discount(0.5)
    data = Opinion.from_evidence(0.65, 1.35)
    for t in (0.25, 1.0):
        expected = judged.fuse(data.discount(t))
        got = _alice(t)
        assert (got.belief, got.disbelief, got.uncertainty) == pytest.approx(
            (expected.belief, expected.disbelief, expected.uncertainty))
    alone = _alice(0.0)
    assert (alone.belief, alone.disbelief, alone.uncertainty) == pytest.approx(
        (judged.belief, judged.disbelief, judged.uncertainty))


def test_a_data_trust_outside_zero_to_one_is_refused() -> None:
    with pytest.raises(ValueError, match="data trust"):
        Rewriter(KB_, TableJudge(SCORES), PythonEngine(), None, data_trust=1.5)


opinions_ = st.tuples(st.floats(0, 1), st.floats(0, 1), st.floats(0.01, 1)).map(
    lambda m: Opinion(m[0] / sum(m), m[1] / sum(m), m[2] / sum(m)))


def _log_likelihood(judged: list[Opinion], data: list[Opinion], labels: list[bool], t: float) -> float:
    total = 0.0
    for j, d, y in zip(judged, data, labels, strict=True):
        p = j.fuse(d.discount(t)).projection
        q = p if y else 1.0 - p
        total += math.log(q) if q > 0.0 else -math.inf
    return total


@settings(max_examples=100, deadline=None)
@given(st.lists(st.tuples(opinions_, opinions_, st.booleans()), min_size=1, max_size=12))
def test_the_fitted_data_trust_is_the_likeliest_on_a_fine_grid(rows: list[tuple[Opinion, Opinion, bool]]) -> None:
    judged, data, labels = ([r[i] for r in rows] for i in range(3))
    t = fit_data_trust(judged, data, labels)
    assert 0.0 <= t <= 1.0
    best = max(_log_likelihood(judged, data, labels, g / 400) for g in range(401))
    assert _log_likelihood(judged, data, labels, t) >= best - 1e-6
