"""Purpose: tests of the direction judges behind semrewrite.judges.make: the
factory's threshold and trust contract, the shared checks on opinions, the
evaluation metrics and the trust fit against brute-force definitions, and each
model judge on the mother/parent case. Jev runs only from the experiment's
answer cache, in a checkout that has data/raw/jev, with a request budget of
zero, so these tests send no request.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
from itertools import product

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from semrewrite import judges, pairs
from semrewrite.judges import JEV_ANSWERS, NAMES, default_threshold, default_trust, make
from semrewrite.judges.base import SentenceJudge, sentence_pairs
from semrewrite.judges.metrics import at_threshold, calibrate, direction, fit_trust, probability_scores, roc_auc
from semrewrite.judges.oracle import OracleJudge
from semrewrite.model import Predicate, Proposal
from semrewrite.opinion import Opinion
from semrewrite.protocols import Judge

MOTHER = Predicate("mother", "{x} is the mother of {y}")
PARENT = Predicate("parent", "{x} is a parent of {y}")
CHILD = Predicate("child_of", "{x} is a child of {y}")
FORWARD, BACKWARD = Proposal(MOTHER, PARENT, (0, 1)), Proposal(PARENT, MOTHER, (0, 1))
SYMMETRIC = ("cosine-bge", "cosine-qwen3")


class Constant(SentenceJudge):
    """A model-free judge scoring each pair by a fixed rule, for testing the shared base."""

    def __init__(self, rule) -> None:
        super().__init__("constant", 0.5)
        self.rule = rule
        self.seen: list[tuple[str, str]] = []

    def _score_sentences(self, pairs):
        self.seen.extend(pairs)
        return [self.rule(p, c) for p, c in pairs]


def test_base_scores_in_order_and_counts_usage() -> None:
    judge = Constant(lambda p, c: 0.9 if "mother" in p else 0.1)
    assert judge.score([]) == [] and judge.usage == {"pairs": 0, "calls": 0}
    assert judge.score([FORWARD, BACKWARD, FORWARD]) == [0.9, 0.1, 0.9]
    assert judge.usage == {"pairs": 3, "calls": 1}
    # A probability is the dogmatic opinion of it, so the score is the opinion's belief.
    (opinion,) = judge.opinions([FORWARD])
    assert (opinion.belief, opinion.disbelief, opinion.uncertainty) == pytest.approx((0.9, 0.1, 0.0))
    assert judge.usage == {"pairs": 4, "calls": 2}
    assert judge.trust == 1.0
    assert isinstance(judge, Judge)


def test_base_refuses_scores_that_are_not_probabilities() -> None:
    for bad in (1.5, -0.1, float("nan")):
        with pytest.raises(RuntimeError):
            Constant(lambda p, c, b=bad: b).score([FORWARD])


def test_fillers_reach_the_sentences_per_proposal() -> None:
    judge = Constant(lambda p, c: 0.5)
    judge.score([FORWARD, Proposal(CHILD, PARENT, (1, 0))], fillers=[("the first person", "the second person"), ("X", "Y")])
    assert judge.seen == [
        ("the first person is the mother of the second person", "the first person is a parent of the second person"),
        ("Y is a child of X", "X is a parent of Y"),
    ]
    with pytest.raises(ValueError):
        sentence_pairs([FORWARD], [])


def test_factory_reads_calibrated_thresholds_and_trusts(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    table = tmp_path / "calibration.json"
    table.write_text(json.dumps({"judges": {"jev": {"default": 0.42, "trust": 0.8}, "cosine-qwen3": {"default": 0.9}}}))
    monkeypatch.setattr(judges, "CALIBRATION", table)
    assert default_threshold("jev", table) == 0.42
    assert make("jev", budget=0).threshold == 0.42
    assert make("jev", threshold=0.7, budget=0).threshold == 0.7
    assert default_trust("jev", table) == make("jev", budget=0).trust == 0.8
    assert make("jev", trust=1.0, budget=0).trust == 1.0
    with pytest.raises(LookupError):
        default_threshold("cosine-bge", table)
    with pytest.raises(LookupError, match="no calibrated trust"):
        default_trust("cosine-qwen3", table)
    with pytest.raises(ValueError):
        make("no-such-judge", threshold=0.5)


def test_committed_calibration_covers_every_judge() -> None:
    for name in NAMES:
        assert 0.0 <= default_threshold(name) <= 1.0
        assert 0.0 <= default_trust(name) <= 1.0


def test_factory_builds_the_oracle_from_gold() -> None:
    oracle = make("oracle", gold=[("mother", "parent", False)])
    assert isinstance(oracle, OracleJudge)
    assert oracle.opinions([FORWARD, BACKWARD]) == [Opinion(1.0, 0.0, 0.0), Opinion(0.0, 1.0, 0.0)]
    assert oracle.trust == 1.0
    with pytest.raises(ValueError):
        make("oracle")


@pytest.fixture(scope="module", params=NAMES)
def judge(request: pytest.FixtureRequest):
    """Each registered judge in turn, one model resident at a time; Jev answers only from its cache.

    A judge whose library is not installed (the models or jev extra) is skipped.
    """
    pytest.importorskip("typesafe_sdk" if request.param == "jev" else "sentence_transformers")
    if request.param == "jev" and not JEV_ANSWERS.is_file():
        pytest.skip(f"no Jev answer cache at {JEV_ANSWERS}")
    options = {"budget": 0} if request.param == "jev" else {}
    built = make(request.param, threshold=0.5, **options)
    yield built
    built.close()


def test_every_judge_satisfies_the_protocol_and_counts(judge) -> None:
    assert isinstance(judge, Judge)
    before = judge.usage["pairs"]
    opinions = judge.opinions([FORWARD, BACKWARD, Proposal(CHILD, PARENT, (1, 0))])
    assert len(opinions) == 3 and all(isinstance(o, Opinion) for o in opinions)
    assert judge.usage["pairs"] == before + 3
    # Only the NLI judges say more than a probability: their neutral class is uncertainty.
    if judge.name.startswith("nli-"):
        assert all(o.uncertainty > 0.0 for o in opinions)
    else:
        assert all(o.uncertainty == 0.0 for o in opinions)


def test_mother_entails_parent_and_not_the_reverse(judge) -> None:
    """Every directional judge scores mother => parent above its threshold and above parent => mother;
    a symmetric judge scores the two alike, which is why it cannot tell direction."""
    forward, backward = judge.score([FORWARD, BACKWARD])
    if judge.name in SYMMETRIC:
        assert forward == pytest.approx(backward, abs=1e-6)
    else:
        assert forward > backward
        assert forward >= 0.5


@given(st.lists(st.tuples(st.sampled_from([0.0, 0.25, 0.5, 0.75, 1.0]), st.booleans()), min_size=1, max_size=30))
def test_roc_auc_is_the_pairwise_win_rate(rows: list[tuple[float, bool]]) -> None:
    scores, labels = [s for s, _ in rows], [y for _, y in rows]
    pos = [s for s, y in rows if y]
    neg = [s for s, y in rows if not y]
    auc = roc_auc(scores, labels)
    if not pos or not neg:
        assert auc is None
    else:
        wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in product(pos, neg))
        assert auc == pytest.approx(wins / (len(pos) * len(neg)))


@given(st.lists(st.tuples(st.floats(0, 1), st.floats(0, 1)), max_size=30))
def test_direction_counts_wins_ties_half(pairs_: list[tuple[float, float]]) -> None:
    result = direction([t for t, _ in pairs_], [f for _, f in pairs_])
    assert result["pairs"] == len(pairs_)
    if pairs_:
        expected = sum(1.0 if t > f else 0.5 if t == f else 0.0 for t, f in pairs_) / len(pairs_)
        assert result["accuracy"] == pytest.approx(expected)
        assert result["ties"] == pytest.approx(sum(t == f for t, f in pairs_) / len(pairs_))


@settings(max_examples=200)
@given(st.lists(st.tuples(st.sampled_from([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]), st.booleans()), min_size=1, max_size=25))
def test_calibrated_threshold_reaches_the_best_balanced_accuracy(rows: list[tuple[float, bool]]) -> None:
    """No threshold admits rows with better balanced accuracy, or plain accuracy when one class is absent."""
    scores, labels = [s for s, _ in rows], [y for _, y in rows]
    threshold = calibrate(scores, labels)
    key = "balanced_accuracy" if any(labels) and not all(labels) else "accuracy"
    best = max(at_threshold(scores, labels, t)[key] for t in [*scores, 1.5])
    assert at_threshold(scores, labels, threshold)[key] == pytest.approx(best)


def test_at_threshold_handles_empty_admissions() -> None:
    result = at_threshold([0.1, 0.2], [True, False], 0.5)
    assert result["admitted"] == 0 and result["precision"] is None and result["recall"] == 0.0


def _log_likelihood(opinions: list[Opinion], labels: list[bool], t: float) -> float:
    total = 0.0
    for o, y in zip(opinions, labels, strict=True):
        p = o.discount(t).projection
        total += math.log(p if y else 1.0 - p) if 0.0 < p < 1.0 else (0.0 if p == float(y) else -math.inf)
    return total


opinions_ = st.tuples(st.floats(0, 1), st.floats(0, 1), st.floats(0, 1)).filter(lambda m: sum(m) > 0.01).map(
    lambda m: Opinion(m[0] / sum(m), m[1] / sum(m), m[2] / sum(m)))


@settings(max_examples=200)
@given(st.lists(st.tuples(opinions_, st.booleans()), min_size=1, max_size=20))
def test_the_fitted_trust_is_the_likeliest_on_a_fine_grid(rows: list[tuple[Opinion, bool]]) -> None:
    """No trust on a grid of 1001 points gives the labels a higher likelihood than the fitted one."""
    opinions, labels = [o for o, _ in rows], [y for _, y in rows]
    t = fit_trust(opinions, labels)
    assert 0.0 <= t <= 1.0
    best = max(_log_likelihood(opinions, labels, g / 1000) for g in range(1001))
    assert _log_likelihood(opinions, labels, t) >= best - 1e-9


def test_a_trusted_probability_shrinks_toward_the_base_rate() -> None:
    """A judge right 7 times in 10 at probability 0.9 is trusted half: its 0.9 projects to 0.5 + 0.5 (0.9 - 0.5),
    the 0.7 it earned, worth W t / (1 - t) = 2 observations."""
    opinions, labels = [Opinion(0.9, 0.1, 0.0)] * 10, [True] * 7 + [False] * 3
    t = fit_trust(opinions, labels)
    assert t == pytest.approx(0.5)
    assert opinions[0].discount(t).projection == pytest.approx(0.7)
    assert fit_trust(opinions, [True] * 5 + [False] * 5) == pytest.approx(0.0, abs=1e-12)
    assert probability_scores([0.7] * 10, labels)["brier"] == pytest.approx(0.21)


def test_evaluation_compares_only_one_way_pairs() -> None:
    """A one-way pair yields its true and false direction's scores; an equivalence or a lone row yields none."""
    spec = importlib.util.spec_from_file_location("judge_eval", pathlib.Path(__file__).parents[1] / "experiments" / "judge_eval.py")
    judge_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge_eval)
    row = lambda i, p, c, label, pair: pairs.PairRow(
        id=i, set="hard", split="test", pair=pair, kind="k", premise=p, conclusion=c, swapped=False,
        label=label, types=("person", "person"))
    spouse = Predicate("spouse", "{x} is the spouse of {y}")
    married = Predicate("married", "{x} is married to {y}")
    rows = [row("r1", PARENT, MOTHER, False, "p1"), row("f1", MOTHER, PARENT, True, "p1"),
            row("a", spouse, married, True, "p2"), row("b", married, spouse, True, "p2")]
    scores = {"r1": 0.7, "f1": 0.2, "a": 0.9, "b": 0.8}
    assert judge_eval.one_way(rows, scores) == ([0.2], [0.7])
    opinions = {key: Opinion(s, 1.0 - s, 0.0) for key, s in scores.items()}
    measured = judge_eval.measure(rows, opinions, default=0.5, own=None, trust=1.0)
    assert set(measured) == {"test"}
    assert measured["test"]["direction"] == {"pairs": 1, "accuracy": 0.0, "ties": 0.0}
    assert measured["test"]["at_default"]["admitted"] == 3
    # At complete trust each row holds with its score: Brier ((0.7 - 0)^2 + (0.2 - 1)^2 + (0.9 - 1)^2 + (0.8 - 1)^2) / 4.
    assert measured["test"]["at_default_trust"]["brier"] == pytest.approx((0.49 + 0.64 + 0.01 + 0.04) / 4)
