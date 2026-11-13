"""Purpose: tests of the direction judges behind semrewrite.judges.make: the
factory's threshold, trust and base-rate contract, the shared checks on opinions,
the evaluation metrics, the base rate and the trust fit against brute-force
definitions, each model judge on the mother/parent case, and each typicality
judge on the defeasible case that birds typically fly and penguins do not. Jev
runs only from the experiment's answer cache, in a checkout that has
data/raw/jev, with a request budget of zero, so these tests send no request.
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
from semrewrite.judges import (
    JEV_ANSWERS,
    NAMES,
    calibration_set,
    default_base_rate,
    default_threshold,
    default_trust,
    make,
)
from semrewrite.judges.base import SentenceJudge, sentence_pairs
from semrewrite.judges.metrics import (
    at_threshold,
    calibrate,
    direction,
    fit_trust,
    probability_scores,
    roc_auc,
)
from semrewrite.judges.oracle import OracleJudge
from semrewrite.judges.typical import TypicalJudge
from semrewrite.model import Predicate, Proposal
from semrewrite.opinion import Opinion
from semrewrite.protocols import Judge, calibrated

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


def test_sentences_written_directly_are_judged_as_written_and_counted() -> None:
    """A request's text against a statement it may support reaches the model unchanged, one pair per opinion, and
    counts as proposals do."""
    judge = Constant(lambda p, c: 0.8 if "emails" in p else 0.2)
    pairs = [("Raw logs include customer emails.", "The data contains personal information."),
             ("The build is green.", "The data contains personal information.")]
    assert judge.sentence_opinions([]) == [] and judge.usage == {"pairs": 0, "calls": 0}
    got = judge.sentence_opinions(pairs)
    assert judge.seen == pairs and judge.usage == {"pairs": 2, "calls": 1}
    assert [o.belief for o in got] == pytest.approx([0.8, 0.2])


def test_calibration_sets_the_base_rate_then_discounts_by_trust() -> None:
    """At trust 0.5 and base rate 0.2, (0.8, 0.2, 0) becomes (0.4, 0.1, 0.5) at 0.2, projecting to 0.5."""
    judge = Constant(lambda p, c: 0.8)
    judge.trust, judge.base_rate = 0.5, 0.2
    (o,) = calibrated(judge, judge.sentence_opinions([("a", "b")]))
    assert (o.belief, o.disbelief, o.uncertainty, o.base_rate) == pytest.approx((0.4, 0.1, 0.5, 0.2))
    assert o.projection == pytest.approx(0.5)


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
    """A judge's threshold and trust are its entry's, and its base rate is that of the set its entry names, stored once
    for the set: Wikidata's where the entry names none."""
    table = tmp_path / "calibration.json"
    table.write_text(json.dumps({"sets": {"wikidata": {"base_rate": 0.2}, "delta-atomic": {"base_rate": 0.5}},
                                 "judges": {"jev": {"default": 0.42, "trust": 0.8},
                                            "cosine-qwen3": {"default": 0.9},
                                            "typical-qwen3-4b": {"set": "delta-atomic", "default": 0.3, "trust": 0.6},
                                            "nli-deberta-base": {"set": "copa", "default": 0.5, "trust": 0.1}}}))
    monkeypatch.setattr(judges, "CALIBRATION", table)
    assert default_threshold("jev", table) == 0.42
    assert make("jev", budget=0).threshold == 0.42
    assert make("jev", threshold=0.7, budget=0).threshold == 0.7
    assert default_trust("jev", table) == make("jev", budget=0).trust == 0.8
    assert make("jev", trust=1.0, budget=0).trust == 1.0
    assert calibration_set("jev", table) == "wikidata" and calibration_set("typical-qwen3-4b", table) == "delta-atomic"
    assert default_base_rate("jev", table) == make("jev", budget=0).base_rate == 0.2
    assert default_base_rate("typical-qwen3-4b", table) == 0.5
    assert make("jev", base_rate=0.5, budget=0).base_rate == 0.5
    with pytest.raises(LookupError):
        default_threshold("cosine-bge", table)
    with pytest.raises(LookupError, match="no calibrated trust"):
        default_trust("cosine-qwen3", table)
    with pytest.raises(LookupError, match="no calibrated base_rate for set 'copa'"):
        default_base_rate("nli-deberta-base", table)
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(LookupError, match="no calibrated base_rate for set 'wikidata'"):
        default_base_rate("jev", empty)
    with pytest.raises(ValueError):
        make("no-such-judge", threshold=0.5)


def test_committed_calibration_covers_every_judge() -> None:
    """Every registered judge has a calibrated threshold, trust and base rate, its defaults fitted on the set its kind
    of question names."""
    for name in NAMES:
        assert 0.0 <= default_threshold(name) <= 1.0
        assert 0.0 <= default_trust(name) <= 1.0
        assert 0.0 < default_base_rate(name) < 1.0
        assert calibration_set(name) == judges.REGISTRY[name][0].calibration_set


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
    # A judge whose model says more than a probability gives its own opinions: an NLI judge's neutral class, a
    # typicality judge's Unsure, is uncertainty. Every other judge's probability is a dogmatic opinion.
    if type(judge)._opinions is not SentenceJudge._opinions:
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


def test_a_typicality_judge_reads_what_typically_follows(judge) -> None:
    """Birds typically fly and penguins typically do not, the defeasible inference (Reiter 1980) that no entailment
    judge is asked: a typicality judge believes the first and disbelieves the second."""
    if not isinstance(judge, TypicalJudge):
        pytest.skip(f"{judge.name} is asked what is entailed, not what typically follows")
    bird, penguin, flies = (Predicate(n, g) for n, g in (("bird", "{x} is a bird"), ("penguin", "{x} is a penguin"),
                                                         ("flies", "{x} can fly")))
    typical, exception = judge.opinions([Proposal(bird, flies, (0,)), Proposal(penguin, flies, (0,))])
    assert typical.belief > 0.5 and typical.belief > exception.belief
    assert exception.disbelief > 0.5


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


def _judge_eval():
    """experiments/judge_eval.py as a module, loaded from its path, since experiments/ is not a package."""
    spec = importlib.util.spec_from_file_location("judge_eval", pathlib.Path(__file__).parents[1] / "experiments" / "judge_eval.py")
    judge_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge_eval)
    return judge_eval


def test_evaluation_compares_only_one_way_pairs() -> None:
    """A one-way pair yields its true and false direction's scores; an equivalence or a lone row yields none."""
    judge_eval = _judge_eval()
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
    measured = judge_eval.measure(rows, opinions, default=0.5, own=None, calibration=(1.0, 0.5))
    assert set(measured) == {"test"}
    assert measured["test"]["direction"] == {"pairs": 1, "accuracy": 0.0, "ties": 0.0}
    assert measured["test"]["at_default"]["admitted"] == 3
    # At complete trust each row holds with its score: Brier ((0.7 - 0)^2 + (0.2 - 1)^2 + (0.9 - 1)^2 + (0.8 - 1)^2) / 4.
    assert measured["test"]["at_default_trust"]["brier"] == pytest.approx((0.49 + 0.64 + 0.01 + 0.04) / 4)


def test_evaluation_takes_a_set_s_base_rate_from_its_dev_rows_and_fits_trust_there() -> None:
    """One of four dev rows holds, so the set's base rate is 1/4. A judge sure of the true row and silent on the
    others is right wherever it speaks, so it is trusted fully, and its silence projects to 1/4, not one half."""
    judge_eval = _judge_eval()
    rows = [pairs.PairRow(id=str(i), set="hard", split="dev", pair=str(i), kind="k", premise=MOTHER, conclusion=PARENT,
                          swapped=False, label=i == 0, types=("person", "person")) for i in range(4)]
    assert judge_eval.base_rates(rows) == {"hard": 0.25}
    opinions = {"0": Opinion(1.0, 0.0, 0.0), **{str(i): Opinion(0.0, 0.0, 1.0) for i in range(1, 4)}}
    ((trust, rate),) = judge_eval.calibrations_for(rows, opinions, {"hard": 0.25}).values()
    assert (trust, rate) == (pytest.approx(1.0), 0.25)
    assert judge_eval.calibrated(opinions["1"], (trust, rate)).projection == pytest.approx(0.25)


def test_the_judge_command_writes_each_pair_s_calibrated_opinion_in_order(monkeypatch, capsys) -> None:
    """Two pairs in, two JSON lines out in the same order, at the trust and base rate given on the command line."""
    import io

    from semrewrite.cli import main

    def factory(name, **overrides):
        judge = Constant(lambda p, c: 0.8 if "emails" in p else 0.2)
        judge.trust, judge.base_rate = overrides.get("trust", 1.0), overrides.get("base_rate", 0.5)
        return judge

    monkeypatch.setattr(judges, "make", factory)
    lines = [json.dumps({"premise": p, "conclusion": "The data is personal."}) for p in ("Logs hold emails.", "All green.")]
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(lines) + "\n\n"))
    assert main(["judge", "--trust", "0.5", "--base-rate", "0.2"]) == 0
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["premise"] for r in out] == ["Logs hold emails.", "All green."]
    assert (out[0]["belief"], out[0]["uncertainty"], out[0]["base_rate"]) == pytest.approx((0.4, 0.5, 0.2))
    assert out[1]["projection"] == pytest.approx(0.1 + 0.2 * 0.5)


def test_the_judge_command_refuses_a_line_without_two_sentences_naming_it(monkeypatch, capsys) -> None:
    import io

    from semrewrite.cli import main

    monkeypatch.setattr(judges, "make", lambda name, **o: Constant(lambda p, c: 0.5))
    for text, said in (('{"premise": "a"}\n', "line 1 is not a JSON object"), ('\n{"premise": "a", "conclusion"\n',
                                                                                     "line 2 is not JSON")):
        monkeypatch.setattr("sys.stdin", io.StringIO(text))
        assert main(["judge"]) == 1
        assert said in capsys.readouterr().err


def test_the_calibration_file_holds_each_set_s_base_rate_once() -> None:
    """The file holds the base rate of each set some judge is calibrated on, once, Wikidata's for an entry naming no
    set, and no base rate of its own or in any entry."""
    judge_eval = _judge_eval()
    entries = {"nli": {"trust": 0.3}, "typical": {"set": "delta-atomic", "trust": 0.6}, "cosine": {"trust": 0.0}}
    written = judge_eval.calibration_file(entries, {"wikidata": 0.214, "delta-atomic": 0.504, "copa": 0.5})
    assert written["sets"] == {"delta-atomic": {"base_rate": 0.504}, "wikidata": {"base_rate": 0.214}}
    assert list(written["judges"]) == ["cosine", "nli", "typical"]
    assert "base_rate" not in written and not any("base_rate" in entry for entry in written["judges"].values())


def test_evaluation_scores_test_rows_at_the_set_s_own_trust_beside_the_default() -> None:
    """A silent opinion of a false test row projects to the base rate of the calibration it is read at: one half
    under a default (1, 1/2), so log loss log 2, and 1/4 under the set's own (1, 1/4), so log loss log 4/3. Dev rows
    get no own-trust score, since that trust was fitted on them."""
    judge_eval = _judge_eval()
    rows = [pairs.PairRow(id=split, set="hard", split=split, pair=split, kind="k", premise=MOTHER, conclusion=PARENT,
                          swapped=False, label=False, types=("person", "person")) for split in ("dev", "test")]
    silent = {r.id: Opinion(0.0, 0.0, 1.0) for r in rows}
    measured = judge_eval.measure(rows, silent, default=None, own=None, calibration=(1.0, 0.5),
                                  own_calibration=(1.0, 0.25))
    assert measured["test"]["at_default_trust"]["log_loss"] == pytest.approx(math.log(2.0))
    assert measured["test"]["at_own_trust"]["log_loss"] == pytest.approx(math.log(4.0 / 3.0))
    # One false row: no ranking to score.
    assert measured["test"]["at_own_trust"]["auc"] is None
    assert "at_own_trust" not in measured["dev"]
