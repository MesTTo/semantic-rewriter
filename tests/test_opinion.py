"""Purpose: check semrewrite.opinion against the numbers Jøsang's book and SLN print, against the properties the two
state, and against the implementations it must agree with: ONA's Narsese parser reads every exported line back, and
SNARS's truth.metta, run on PeTTa, computes every operation it shares to within 1e-9.

The ONA check needs ONA_PATH, an OpenNARS-for-Applications checkout with its NAR binary built. The SNARS check needs
PETTA_PATH, a PeTTa checkout, METTABASE_PATH, the mettabase checkout whose lib/mettabase/snars it loads, and swipl on
the PATH; it runs PeTTa with mettabase as the working directory, which SNARS's kernel needs, and writes nothing.
Each is skipped, with the reason, when what it needs is missing. The SNARS check includes the product's base-rate
corners and fusion of unequal base rates, which truth.metta at mettabase c9bbbe8 gets wrong (sn kdiv drops the cross
terms, sn fuse-op keeps the first base rate) and its commit 10cdbc4 corrects; it fails on the former.
"""

from __future__ import annotations

import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from semrewrite.opinion import Opinion, W, narsese_implication

UNIT = st.floats(min_value=0.0, max_value=1.0)
# Evidence from none to a million units, which leaves u between 2e-6 and 1.
AMOUNTS = st.one_of(st.just(0.0), st.floats(min_value=1e-6, max_value=1e6))
# Base rates whose complement 1 - a is exact, so that complementing twice returns a itself.
DYADIC = st.integers(min_value=0, max_value=1024).map(lambda i: i / 1024)


@st.composite
def evidence_opinions(draw: st.DrawFn, base_rates: st.SearchStrategy[float] = UNIT) -> Opinion:
    return Opinion.from_evidence(draw(AMOUNTS), draw(AMOUNTS), draw(base_rates))


@st.composite
def dogmatic_opinions(draw: st.DrawFn, base_rates: st.SearchStrategy[float] = UNIT) -> Opinion:
    belief = draw(UNIT)
    return Opinion(belief, 1.0 - belief, 0.0, draw(base_rates))


@st.composite
def simplex_opinions(draw: st.DrawFn, base_rates: st.SearchStrategy[float] = UNIT) -> Opinion:
    """Any point of the opinion triangle, the corners and the dogmatic edge included."""
    x, y, z = draw(UNIT), draw(UNIT), draw(UNIT)
    assume(x + y + z > 0.0)
    total = x + y + z
    return Opinion(x / total, y / total, z / total, draw(base_rates))


def opinions(base_rates: st.SearchStrategy[float] = UNIT) -> st.SearchStrategy[Opinion]:
    return st.one_of(evidence_opinions(base_rates), dogmatic_opinions(base_rates), simplex_opinions(base_rates))


def assert_close(got: Opinion, want: Opinion, tolerance: float = 1e-12) -> None:
    for name in ("belief", "disbelief", "uncertainty", "base_rate"):
        assert math.isclose(getattr(got, name), getattr(want, name), abs_tol=tolerance), (name, got, want)


def assert_printed(got: Opinion, printed: tuple[float, float, float, float]) -> None:
    """`got` rounds to the book's two-place figures: each lies within half a unit of the last place."""
    values = (got.belief, got.disbelief, got.uncertainty, got.base_rate)
    assert all(abs(g - p) <= 0.005 + 1e-12 for g, p in zip(values, printed, strict=True)), (got, printed)


# ---- the book's and SLN's worked examples ----


def test_running_example_of_the_book() -> None:
    # J Fig 3.1 and 3.2: Beta(3.8, 1.2) = Beta^e(2, 1, 0.9) is the opinion (0.4, 0.2, 0.4, 0.9), projecting 0.76.
    opinion = Opinion.from_evidence(2, 1, 0.9)
    assert opinion == Opinion(0.4, 0.2, 0.4, 0.9)
    assert math.isclose(opinion.projection, 0.76)
    alpha, beta = opinion.beta()
    assert math.isclose(alpha, 3.8) and math.isclose(beta, 1.2)
    assert_close(Opinion.from_beta(3.8, 1.2, 0.9), opinion)
    # J 3.10 against the Beta's own variance alpha beta / ((alpha + beta)^2 (alpha + beta + 1)).
    assert math.isclose(opinion.variance, 3.8 * 1.2 / (5.0**2 * 6.0))


def test_vacuous_opinion_is_the_uniform_beta() -> None:
    # J 3.4.3: W = 2 makes (0, 0, 1, 1/2) Beta(1, 1), whose variance is 1/12.
    vacuous = Opinion.from_evidence(0, 0)
    assert vacuous == Opinion(0.0, 0.0, 1.0, 0.5)
    assert vacuous.beta() == (1.0, 1.0)
    assert math.isclose(vacuous.variance, 1 / 12)
    assert vacuous.to_nal() == (None, 0.0)


def test_product_example_of_the_book() -> None:
    # J Fig 7.2: (0.75, 0.15, 0.10, 0.50) and (0.10, 0.00, 0.90, 0.20) multiply to (0.15, 0.15, 0.70, 0.10), P 0.22.
    product = Opinion(0.75, 0.15, 0.10, 0.50).conjoin(Opinion(0.10, 0.00, 0.90, 0.20))
    assert_printed(product, (0.15, 0.15, 0.70, 0.10))
    assert math.isclose(product.projection, 0.80 * 0.28)


def test_coproduct_example_of_the_book() -> None:
    # J Fig 7.3: (0.75, 0.15, 0.10, 0.50) or (0.35, 0.00, 0.65, 0.20) is (0.84, 0.06, 0.10, 0.60), P 0.90.
    coproduct = Opinion(0.75, 0.15, 0.10, 0.50).disjoin(Opinion(0.35, 0.00, 0.65, 0.20))
    assert_printed(coproduct, (0.84, 0.06, 0.10, 0.60))
    assert math.isclose(coproduct.projection, 0.80 + 0.48 - 0.80 * 0.48)


def test_products_at_the_base_rate_corners_keep_the_most_uncertainty() -> None:
    # J 7.1 is 0/0 at a_x = a_y = 1 and J 7.3 at a_x = a_y = 0; the cross terms go to u (J 7.1.3).
    half = Opinion(0.5, 0.0, 0.5, 1.0)
    assert_close(half.conjoin(half), Opinion(0.25, 0.0, 0.75, 1.0))
    assert half.conjoin(half).projection == half.projection * half.projection
    true = Opinion(1.0, 0.0, 0.0, 1.0)
    assert true.conjoin(true) == true
    assert_close(half.complement().disjoin(half.complement()), Opinion(0.0, 0.25, 0.75, 0.0))
    # Next to the corner the quotients keep their precision: J 7.1 there, with no mass lost.
    near = Opinion(0.5, 0.0, 0.5, 1.0 - 2.0**-40)
    assert_close(half.conjoin(near), Opinion(0.5, 0.0, 0.5, near.base_rate))


def test_conflict_examples_of_the_book() -> None:
    # J 4.8: equal projections 0.77 do not conflict however the masses differ; J 4.64: 0.56 * 0.2 * 0.9 = 0.10.
    assert math.isclose(Opinion(0.05, 0.15, 0.80, 0.90).conflict(Opinion(0.68, 0.22, 0.10, 0.90)), 0.0, abs_tol=1e-12)
    conflict = Opinion(0.05, 0.15, 0.80, 0.10).conflict(Opinion(0.68, 0.22, 0.10, 0.10))
    assert math.isclose(conflict, 0.56 * 0.2 * 0.9)


def test_discount_example_of_the_book() -> None:
    # J 14.8: A trusts B as (0.20, 0.40, 0.40, 0.75), P = 0.5; B's (0.45, 0.35, 0.20, 0.25) reaches A as
    # (0.225, 0.175, 0.600, 0.250), P = 0.375.
    trust = Opinion(0.20, 0.40, 0.40, 0.75)
    assert math.isclose(trust.projection, 0.5)
    relayed = Opinion(0.45, 0.35, 0.20, 0.25).discount(trust.projection)
    assert_close(relayed, Opinion(0.225, 0.175, 0.6, 0.25))
    assert math.isclose(relayed.projection, 0.375)


def test_fusion_example_of_sln() -> None:
    # SLN 3.1: two judgments of (2, 0) fuse to (4, 0), b = 2/3, u = 1/3, which NAL at k = 2 reads as f = 1, c = 4/6.
    one = Opinion.from_evidence(2, 0)
    assert one == Opinion(0.5, 0.0, 0.5, 0.5)
    fused = one.fuse(one)
    assert_close(fused, Opinion(2 / 3, 0.0, 1 / 3, 0.5))
    f, c = fused.to_nal(k=2)
    assert f == 1.0 and math.isclose(c, 4 / 6)


def test_ona_truth_converts_through_evidence() -> None:
    # SLN section 6: ONA's %1.0;0.9% at k = 1 is w = 9, the opinion (0.818, 0, 0.182), not the shortcut (0.9, 0, 0.1).
    assert_close(Opinion.from_nal(1.0, 0.9, k=1), Opinion(9 / 11, 0.0, 2 / 11))
    assert_close(Opinion.from_nal(1.0, 0.9, k=1), Opinion.from_evidence(9, 0))


def test_from_moments_clamps_where_matching_leaves_negative_belief() -> None:
    # Mean 0.1 and variance 0.05 match u = 2.5; at a = 0.9 the largest u keeping b >= 0 is m/a = 1/9.
    opinion = Opinion.from_moments(0.1, 0.05, 0.9)
    assert_close(opinion, Opinion(0.0, 8 / 9, 1 / 9, 0.9))
    assert math.isclose(opinion.projection, 0.1)
    assert opinion.variance < 0.05


@pytest.mark.parametrize(
    ("mean", "base_rate", "want"),
    [
        (1.0, 1.0, Opinion(1.0, 0.0, 0.0, 1.0)),  # a fact the knowledge base states, as the rewriter answers it
        (0.0, 0.0, Opinion(0.0, 1.0, 0.0, 0.0)),
        (1.0, 0.5, Opinion(1.0, 0.0, 0.0, 0.5)),
        (0.3, 0.9, Opinion(0.3, 0.7, 0.0, 0.9)),
    ],
)
def test_from_moments_reads_variance_zero_as_a_point_mass(mean: float, base_rate: float, want: Opinion) -> None:
    assert Opinion.from_moments(mean, 0.0, base_rate) == want


@pytest.mark.parametrize(
    "opinion", [Opinion(0.5, 0.0, 0.5, 1.0), Opinion(0.0, 0.0, 1.0, 1.0), Opinion(0.0, 0.7, 0.3, 0.0)]
)
def test_from_moments_cannot_tell_apart_opinions_projecting_their_base_rate(opinion: Opinion) -> None:
    # P = a in {0, 1} has variance 0 whatever u is, so the moments name a whole family, and its point mass comes back.
    assert opinion.variance == 0.0
    a = opinion.base_rate
    assert Opinion.from_moments(opinion.projection, 0.0, a) == Opinion(a, 1.0 - a, 0.0, a)


# ---- properties ----


@given(AMOUNTS, AMOUNTS, UNIT)
def test_evidence_round_trip(positive: float, negative: float, base_rate: float) -> None:
    opinion = Opinion.from_evidence(positive, negative, base_rate)
    assert opinion.uncertainty > 0.0
    r, s = opinion.evidence()
    assert math.isclose(r, positive, rel_tol=1e-12) and math.isclose(s, negative, rel_tol=1e-12)
    assert_close(Opinion.from_evidence(r, s, base_rate), opinion)


@given(UNIT, st.floats(min_value=0.0, max_value=1e6), UNIT)
def test_probability_is_weighted_evidence(probability: float, weight: float, base_rate: float) -> None:
    opinion = Opinion.from_probability(probability, weight, base_rate)
    r, s = opinion.evidence()
    assert math.isclose(r, probability * weight, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(s, (1.0 - probability) * weight, rel_tol=1e-12, abs_tol=1e-12)


@given(evidence_opinions(), evidence_opinions())
def test_fusion_adds_evidence(x: Opinion, y: Opinion) -> None:
    (rx, sx), (ry, sy) = x.evidence(), y.evidence()
    r, s = x.fuse(y).evidence()
    assert math.isclose(r, rx + ry, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(s, sx + sy, rel_tol=1e-12, abs_tol=1e-12)


@given(opinions(), opinions())
def test_fusion_commutes(x: Opinion, y: Opinion) -> None:
    assert_close(x.fuse(y), y.fuse(x))


@given(opinions(), opinions(), opinions())
def test_fusion_associates(x: Opinion, y: Opinion, z: Opinion) -> None:
    # J 12.15 averages two dogmatic opinions with weight 1/2 each, so at most one of three may be dogmatic. A fused
    # u = u1 u2/k below the smallest float rounds to 0 and turns dogmatic, as u1 = u2 = 5e-324 do, so the others
    # keep u at 1e-150 or more, evidence below 1e150 units.
    assume(sum(o.uncertainty == 0.0 for o in (x, y, z)) <= 1)
    assume(all(o.uncertainty == 0.0 or o.uncertainty >= 1e-150 for o in (x, y, z)))
    left, right = x.fuse(y).fuse(z), x.fuse(y.fuse(z))
    for name in ("belief", "disbelief", "uncertainty"):
        assert math.isclose(getattr(left, name), getattr(right, name), abs_tol=1e-12), (name, left, right)
    # Three vacuous opinions carry no evidence to weigh their base rates by, and J 12.14 takes the plain mean of two.
    if not all(o.uncertainty == 1.0 for o in (x, y, z)):
        assert math.isclose(left.base_rate, right.base_rate, abs_tol=1e-12)


@given(evidence_opinions(), UNIT)
def test_fused_base_rate_is_the_evidence_weighted_mean(x: Opinion, base_rate: float) -> None:
    y = Opinion.from_evidence(3.0, 1.0, base_rate)
    (rx, sx), (ry, sy) = x.evidence(), y.evidence()
    want = (x.base_rate * (rx + sx) + base_rate * (ry + sy)) / (rx + sx + ry + sy)
    assert math.isclose(x.fuse(y).base_rate, want, abs_tol=1e-12)


@given(opinions(), dogmatic_opinions())
def test_a_dogmatic_opinion_outweighs_evidence(x: Opinion, dogma: Opinion) -> None:
    assume(x.uncertainty > 0.0)
    assert x.fuse(dogma) == dogma
    assert dogma.fuse(x) == dogma


@given(opinions())
def test_complement_is_an_involution(opinion: Opinion) -> None:
    twice = opinion.complement().complement()
    masses = (twice.belief, twice.disbelief, twice.uncertainty)
    assert masses == (opinion.belief, opinion.disbelief, opinion.uncertainty)
    assert math.isclose(twice.base_rate, opinion.base_rate, abs_tol=1e-15)
    assert math.isclose(opinion.complement().projection, 1.0 - opinion.projection, abs_tol=1e-12)


@given(opinions(DYADIC), opinions(DYADIC))
def test_conjoin_and_disjoin_are_de_morgan_duals(x: Opinion, y: Opinion) -> None:
    assert_close(x.conjoin(y), x.complement().disjoin(y.complement()).complement())
    assert_close(x.disjoin(y), x.complement().conjoin(y.complement()).complement())


@given(opinions(), opinions())
def test_products_project_as_probabilities_do(x: Opinion, y: Opinion) -> None:
    # J 7.2 and J 7.4, which the corner rule keeps: there a = 1 makes P = b + u = 1 - d (and a = 0 makes P = b).
    px, py = x.projection, y.projection
    assert math.isclose(x.conjoin(y).projection, px * py, abs_tol=1e-12)
    assert_close(x.conjoin(y), y.conjoin(x))
    assert math.isclose(x.disjoin(y).projection, px + py - px * py, abs_tol=1e-12)
    assert_close(x.disjoin(y), y.disjoin(x))


@given(opinions())
def test_full_trust_keeps_and_no_trust_empties(opinion: Opinion) -> None:
    assert opinion.discount(1.0) == opinion
    assert opinion.discount(0.0) == Opinion(0.0, 0.0, 1.0, opinion.base_rate)


@given(opinions(), UNIT)
def test_discount_keeps_frequency_and_scales_confidence(opinion: Opinion, trust: float) -> None:
    # SLN 3.3: in (f, c) at k = W, discounting by p is f' = f, c' = p c.
    f, c = opinion.to_nal()
    discounted = opinion.discount(trust)
    discounted_f, discounted_c = discounted.to_nal()
    assert math.isclose(discounted_c, trust * c, abs_tol=1e-12)
    # f survives where p b and p d are 0 or normal floats: a subnormal product, such as 5e-324 times 0.75, keeps too
    # few bits to hold their ratio, and both underflowing leaves no frequency at all.
    masses = (discounted.belief, discounted.disbelief)
    if f is not None and sum(masses) > 0.0 and all(m == 0.0 or m >= sys.float_info.min for m in masses):
        assert discounted_f is not None and math.isclose(discounted_f, f, abs_tol=1e-12)


@given(AMOUNTS, AMOUNTS, UNIT)
def test_from_moments_recovers_the_beta(positive: float, negative: float, base_rate: float) -> None:
    alpha, beta = positive + base_rate * W, negative + (1.0 - base_rate) * W
    assume(alpha > 0.0 and beta > 0.0)
    total = alpha + beta
    mean, variance = alpha / total, alpha * beta / (total**2 * (total + 1.0))
    # Subnormal moments, as a base rate of 2e-309 gives, keep too few bits to recover the Beta from.
    assume(min(mean, 1.0 - mean, variance) >= sys.float_info.min)
    # The mean carries 1 - m only to within an ulp of 1, so beta, which is proportional to 1 - m, comes back to a
    # relative ulp/(1 - m), and alpha likewise to ulp/m: 1.2e-9 for 1 - m = 7.8e-8.
    tolerance = 1e-9 + 1e-14 / min(mean, 1.0 - mean)
    got_alpha, got_beta = Opinion.from_moments(mean, variance, base_rate).beta()
    assert math.isclose(got_alpha, alpha, rel_tol=tolerance) and math.isclose(got_beta, beta, rel_tol=tolerance)


@given(opinions())
def test_from_moments_inverts_projection_and_variance(opinion: Opinion) -> None:
    # Off the family from_moments cannot tell apart, u > 0 with P = a in {0, 1}, such as (0.5, 0, 0.5, 1). Floats
    # also round P to 0 or 1 for a within an ulp of either, as (0.5, 0, 0.5, 0.9999999999999999) projects 1.0.
    p = opinion.projection
    assume(0.0 < p < 1.0 or opinion.uncertainty == 0.0)
    # Moments below the normal range keep too few bits to invert, as P = 2e-315 from a = 2e-311 does.
    a = opinion.base_rate
    assume(all(x == 0.0 or x >= sys.float_info.min for x in (p, 1.0 - p, opinion.variance, a, 1.0 - a)))
    assert_close(Opinion.from_moments(opinion.projection, opinion.variance, opinion.base_rate), opinion, 1e-9)


@given(evidence_opinions())
def test_projection_and_variance_are_the_betas(opinion: Opinion) -> None:
    # J 3.9 and J 3.10: the Beta's mean alpha/(alpha + beta) and variance.
    alpha, beta = opinion.beta()
    assume(alpha > 0.0 and beta > 0.0)
    total = alpha + beta
    assert math.isclose(opinion.projection, alpha / total, abs_tol=1e-12)
    assert math.isclose(opinion.variance, alpha * beta / (total**2 * (total + 1.0)), rel_tol=1e-9, abs_tol=1e-15)


@given(opinions(), st.sampled_from([1.0, 2.0]))
def test_nal_round_trip(opinion: Opinion, k: float) -> None:
    f, c = opinion.to_nal(k)
    assert_close(Opinion.from_nal(f, c, k, opinion.base_rate), opinion, 1e-9)


@given(UNIT, UNIT, st.sampled_from([1.0, 2.0]))
def test_nal_round_trip_from_nal(f: float, c: float, k: float) -> None:
    # A subnormal confidence such as 5e-324 underflows to no evidence at all once divided by W(1 - c) + k c.
    assume(c >= sys.float_info.min)
    got_f, got_c = Opinion.from_nal(f, c, k).to_nal(k)
    assert got_f is not None and math.isclose(got_f, f, abs_tol=1e-9) and math.isclose(got_c, c, abs_tol=1e-9)


@given(opinions())
def test_nal_at_k_equal_w_is_the_clean_form(opinion: Opinion) -> None:
    # SLN T1 with k = W: c = b + d, f = b/(b + d); and T2: at a = 1/2 the expectation c(f - 1/2) + 1/2 is P.
    f, c = opinion.to_nal(W)
    assert math.isclose(c, opinion.belief + opinion.disbelief, abs_tol=1e-12)
    if f is not None:
        assert math.isclose(f * (opinion.belief + opinion.disbelief), opinion.belief, abs_tol=1e-12)
        half = Opinion(opinion.belief, opinion.disbelief, opinion.uncertainty, 0.5)
        assert math.isclose(c * (f - 0.5) + 0.5, half.projection, abs_tol=1e-12)


@given(opinions(), opinions())
def test_conflict_is_symmetric_bounded_and_nil_on_itself(x: Opinion, y: Opinion) -> None:
    assert x.conflict(x) == 0.0
    assert x.conflict(y) == y.conflict(x)
    assert 0.0 <= x.conflict(y) <= 1.0


# ---- refusals ----


@pytest.mark.parametrize(
    "masses",
    [
        (0.5, 0.5, 0.5, 0.5),  # sums to 1.5
        (-0.1, 0.6, 0.5, 0.5),
        (1.2, -0.1, -0.1, 0.5),
        (math.nan, 0.5, 0.5, 0.5),
        (0.2, 0.3, 0.5, 1.5),
        (0.2, 0.3, 0.5, math.nan),
    ],
)
def test_constructor_refuses_what_is_not_an_opinion(masses: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        Opinion(*masses)


def test_dogmatic_opinions_are_allowed_but_are_no_evidence() -> None:
    dogma = Opinion(1.0, 0.0, 0.0)
    assert dogma.variance == 0.0 and dogma.to_nal() == (1.0, 1.0)
    assert Opinion.from_nal(1.0, 1.0) == dogma
    assert Opinion.from_moments(1.0, 0.0, 0.5) == dogma
    with pytest.raises(ValueError, match="infinite evidence"):
        dogma.evidence()
    with pytest.raises(ValueError, match="infinite evidence"):
        dogma.beta()


@pytest.mark.parametrize(
    "build",
    [
        lambda: Opinion.from_evidence(-1.0, 0.0),
        lambda: Opinion.from_evidence(math.inf, 0.0),
        lambda: Opinion.from_evidence(math.nan, 0.0),
        lambda: Opinion.from_probability(1.5, 1.0),
        lambda: Opinion.from_probability(0.5, -1.0),
        lambda: Opinion.from_beta(0.5, 3.0, 0.5),  # alpha below a W = 1: negative evidence
        lambda: Opinion.from_moments(1.2, 0.0, 0.5),
        lambda: Opinion.from_moments(0.5, -0.01, 0.5),
        lambda: Opinion.from_moments(0.5, 0.3, 0.5),  # above m(1 - m) = 0.25
        lambda: Opinion.from_moments(0.5, 0.01, 1.5),
        lambda: Opinion.from_nal(0.5, 1.5),
        lambda: Opinion.from_nal(None, 0.5),
        lambda: Opinion.from_nal(1.5, 0.5),
        lambda: Opinion.from_nal(0.5, 0.5, k=0.0),
        lambda: Opinion(0.2, 0.3, 0.5).to_nal(k=-1.0),
        lambda: Opinion(0.2, 0.3, 0.5).discount(1.5),
    ],
)
def test_operations_refuse_what_they_cannot_answer(build: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        build()


def test_from_moments_accepts_the_widest_beta_its_base_rate_allows() -> None:
    # Variance m(1 - m) is the Bernoulli limit: no strength at all, so the uncertainty-maximised opinion of mean m.
    assert_close(Opinion.from_moments(0.25, 0.25 * 0.75, 0.5), Opinion(0.0, 0.5, 0.5, 0.5))


# ---- export to ONA ----

OWNS_LINE = "<<($X1 * $X0) --> owned_by_p127> ==> <($X0 * $X1) --> owns>>. %0.90;0.91%"


def test_exports_a_rewrite_as_an_implication_between_products() -> None:
    line = narsese_implication("owned_by_p127", ("X1", "X0"), "owns", ("X0", "X1"), Opinion.from_evidence(9, 1))
    assert line == OWNS_LINE


@pytest.mark.parametrize(
    ("premise_args", "conclusion_args", "want"),
    [
        ((), (), "<q ==> p>"),
        (("X0",), ("X0",), "<<(* $X0) --> q> ==> <(* $X0) --> p>>"),
        (("X2", "X0", "X1"), ("X0", "X1", "X2"), "<<($X2 * ($X0 * $X1)) --> q> ==> <($X0 * ($X1 * $X2)) --> p>>"),
        (("X0", "acme"), ("acme", "X0"), "<<($X0 * acme) --> q> ==> <(acme * $X0) --> p>>"),
    ],
)
def test_exports_every_arity_as_ona_writes_it(
    premise_args: tuple[str, ...], conclusion_args: tuple[str, ...], want: str
) -> None:
    # SLN section 6: k = 1 for ONA, so 3 units for and 1 against is f = 0.75, c = 4/5.
    line = narsese_implication("q", premise_args, "p", conclusion_args, Opinion.from_evidence(3, 1))
    assert line == f"{want}. %0.75;0.80%"


def test_export_caps_confidence_at_onas_maximum() -> None:
    assert narsese_implication("q", ("X0",), "p", ("X0",), Opinion(1.0, 0.0, 0.0)).endswith("%1.00;0.99%")
    assert narsese_implication("q", ("X0",), "p", ("X0",), Opinion.from_evidence(500, 0)).endswith("%1.00;0.99%")


@pytest.mark.parametrize(
    ("premise", "premise_args", "conclusion", "conclusion_args", "opinion"),
    [
        ("q", ("X0",), "p", ("X0",), Opinion.from_evidence(0, 0)),  # vacuous: no frequency
        ("q", ("A", "B", "C", "D", "E"), "p", ("A", "B", "C", "D", "E"), Opinion.from_evidence(1, 0)),  # 5 terms
        ("q" * 32, ("X0",), "p", ("X0",), Opinion.from_evidence(1, 0)),  # longer than ONA's 31 characters
        ("Q", ("X0",), "p", ("X0",), Opinion.from_evidence(1, 0)),  # not a predicate symbol
        ("q", ("x-y",), "p", ("X0",), Opinion.from_evidence(1, 0)),  # not a term
    ],
)
def test_export_refuses_what_ona_cannot_read(
    premise: str, premise_args: tuple[str, ...], conclusion: str, conclusion_args: tuple[str, ...], opinion: Opinion
) -> None:
    with pytest.raises(ValueError):
        narsese_implication(premise, premise_args, conclusion, conclusion_args, opinion)


def _same_up_to_variable_names(ours: str, theirs: str) -> bool:
    """Whether the two terms differ only by a one-to-one renaming of their $ variables."""
    variable = re.compile(r"\$\w+")
    if variable.sub("$", ours) != variable.sub("$", theirs):
        return False
    pairs = set(zip(variable.findall(ours), variable.findall(theirs), strict=True))
    return len(pairs) == len({o for o, _ in pairs}) == len({t for _, t in pairs})


def _export_cases(rng: random.Random) -> list[str]:
    """The lines ONA must read back: every arity ONA holds, constants, and random permutations and opinions."""
    lines = [OWNS_LINE, narsese_implication("q", ("X0",), "p", ("X0",), Opinion(1.0, 0.0, 0.0))]
    for arity in range(5):
        for _ in range(8):
            variables = [f"X{i}" for i in range(arity)]
            terms = [v if rng.random() < 0.8 else rng.choice(["acme", "q42", "zed"]) for v in variables]
            premise_terms = rng.sample(terms, arity)
            opinion = Opinion.from_evidence(rng.uniform(0.01, 300.0), rng.uniform(0.0, 300.0), rng.random())
            lines.append(narsese_implication(f"pre_{arity}", premise_terms, f"con_{arity}", terms, opinion))
    return lines


@pytest.mark.skipif(not os.environ.get("ONA_PATH"), reason="ONA_PATH names no OpenNARS-for-Applications checkout")
def test_ona_reads_every_exported_line_back() -> None:
    nar = Path(os.environ["ONA_PATH"]) / "NAR"
    if not nar.is_file():
        pytest.skip(f"{nar} is not built")
    lines = _export_cases(random.Random(20261113))
    # ONA prints an input implication once as an event and again, when its implication table takes it (Memory.c),
    # so each line goes to an emptied memory (*reset) after a comment, which ONA echoes as "Comment: line i", and
    # the first Input line after that comment is the line as ONA read it.
    script = "".join(f"*reset\n//line {i}\n{line}\n" for i, line in enumerate(lines))
    result = subprocess.run([nar, "shell"], input=script, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    segments = re.split(r"^Comment: line \d+$", result.stdout, flags=re.MULTILINE)[1:]
    assert len(segments) == len(lines), result.stdout
    echo = re.compile(r"^Input: (.*)\. Priority=\S+ Stamp=\S+ Truth: frequency=(\S+), confidence=(\S+)$")
    for line, segment in zip(lines, segments, strict=True):
        read = next((m.groups() for m in map(echo.match, segment.splitlines()) if m), None)
        assert read, (line, segment)
        term, frequency, confidence = read
        ours, truth = line.split(". %")
        f, c = (float(x) for x in truth.rstrip("%").split(";"))
        assert _same_up_to_variable_names(ours, term), (line, term)
        assert (float(frequency), float(confidence)) == (f, c), (line, frequency, confidence)
    print(f"ONA read back all {len(lines)} exported lines")


# ---- the SNARS oracle ----

_OPINION = re.compile(r"^\(Opinion (\S+) (\S+) (\S+) (\S+)\)$")
_EVIDENCE = re.compile(r"^\(Ev (\S+) (\S+) (\S+)\)$")
_NUMBER = re.compile(r"^(\S+)$")


def _metta(opinion: Opinion) -> str:
    # Python's shortest round-trip repr, such as 1e-05, reads back in PeTTa as the same double.
    return f"(Opinion {opinion.belief!r} {opinion.disbelief!r} {opinion.uncertainty!r} {opinion.base_rate!r})"


def _fields(opinion: Opinion) -> tuple[float, ...]:
    return (opinion.belief, opinion.disbelief, opinion.uncertainty, opinion.base_rate)


class _Oracle:
    """A batch of SNARS calls and the values this module gives for them, compared after one PeTTa run."""

    def __init__(self) -> None:
        self.setup: list[str] = []
        self.cases: list[tuple[str, str, re.Pattern[str], tuple[float, ...]]] = []

    def add(self, name: str, call: str, shape: re.Pattern[str], want: tuple[float, ...]) -> None:
        """Ask SNARS `call`, whose answer must match `shape` and equal `want` within 1e-9."""
        self.cases.append((name, call, shape, want))

    def program(self, snars: Path) -> str:
        imports = [f'!(import! &self "{snars / module}")' for module in ("kernel", "config", "truth")]
        calls = [f"!(let $r {call} (Case {i} $r))" for i, (_, call, _, _) in enumerate(self.cases)]
        return "\n".join([*imports, *self.setup, *calls]) + "\n"


def _sample_evidence(rng: random.Random) -> float:
    # Up to 450 units each side: two opinions fused hold under 1998 units, where SNARS's u-min = 0.001 never binds.
    return 0.0 if rng.random() < 0.15 else 10 ** rng.uniform(-6.0, math.log10(450.0))


def _sample_rate(rng: random.Random) -> float:
    return rng.choice([0.0, 1.0, 0.5, 0.001, 0.999]) if rng.random() < 0.3 else rng.random()


def _sample_opinion(rng: random.Random, dogmatic: bool) -> Opinion:
    if dogmatic and rng.random() < 0.25:
        belief = rng.choice([0.0, 1.0, rng.random()])
        return Opinion(belief, 1.0 - belief, 0.0, _sample_rate(rng))
    return Opinion.from_evidence(_sample_evidence(rng), _sample_evidence(rng), _sample_rate(rng))


# The base-rate corners of the product and coproduct and their neighbours, which the first calls take.
_CORNERS = ((1.0, 1.0), (1.0, 1.0 - 1e-9), (0.0, 0.0), (0.0, 1e-9), (1.0, 0.0), (0.0, 1.0))


def _oracle_cases(rng: random.Random, per_operator: int) -> _Oracle:
    oracle = _Oracle()
    for i in range(per_operator):
        x, y = _sample_opinion(rng, dogmatic=True), _sample_opinion(rng, dogmatic=True)
        if i < len(_CORNERS):
            x = Opinion(x.belief, x.disbelief, x.uncertainty, _CORNERS[i][0])
            y = Opinion(y.belief, y.disbelief, y.uncertainty, _CORNERS[i][1])
        trust = rng.choice([0.0, 1.0]) if rng.random() < 0.2 else rng.random()
        oracle.add("project", f"(sn project {_metta(x)})", _NUMBER, (x.projection,))
        oracle.add("complement", f"(sn complement {_metta(x)})", _OPINION, _fields(x.complement()))
        discounted = _fields(x.discount(trust))
        oracle.add("discount-op", f"(sn discount-op {_metta(x)} {trust!r})", _OPINION, discounted)
        oracle.add("and-op", f"(sn and-op {_metta(x)} {_metta(y)})", _OPINION, _fields(x.conjoin(y)))
        oracle.add("or-op", f"(sn or-op {_metta(x)} {_metta(y)})", _OPINION, _fields(x.disjoin(y)))
        # op-to-ev, fuse-op and the cells hold evidence, which a dogmatic opinion is not.
        x, y = _sample_opinion(rng, dogmatic=False), _sample_opinion(rng, dogmatic=False)
        r, s = x.evidence()
        oracle.add("op-to-ev", f"(sn op-to-ev {_metta(x)})", _EVIDENCE, (r, s, x.base_rate))
        same = Opinion(y.belief, y.disbelief, y.uncertainty, x.base_rate)
        oracle.add("fuse-op", f"(sn fuse-op {_metta(x)} {_metta(same)})", _OPINION, _fields(x.fuse(same)))
        oracle.add("fuse-op, unequal a", f"(sn fuse-op {_metta(x)} {_metta(y)})", _OPINION, _fields(x.fuse(y)))
        cells = []
        for side in ("x", "y"):
            positive, negative, rate = _sample_evidence(rng), _sample_evidence(rng), _sample_rate(rng)
            oracle.setup.append(f"!(sn_ev_add cell-{i}-{side} {positive!r} {negative!r} {rate!r} 1.0)")
            cells.append(Opinion.from_evidence(positive, negative, rate))
        oracle.add("opinion view", f"(sn opinion cell-{i}-x)", _OPINION, _fields(cells[0]))
        oracle.add("dc", f"(sn dc cell-{i}-x cell-{i}-y)", _NUMBER, (cells[0].conflict(cells[1]),))
    return oracle


@pytest.mark.skipif(
    not (os.environ.get("PETTA_PATH") and os.environ.get("METTABASE_PATH") and shutil.which("swipl")),
    reason="the SNARS oracle needs PETTA_PATH, METTABASE_PATH and swipl",
)
def test_operators_agree_with_snars_truth_metta() -> None:
    petta, mettabase = Path(os.environ["PETTA_PATH"]), Path(os.environ["METTABASE_PATH"])
    oracle = _oracle_cases(random.Random(20261113), per_operator=60)
    # SNARS's kernel consults ./lib/mettabase/snars/kernel.pl, so PeTTa runs in mettabase; the program comes on stdin.
    result = subprocess.run(
        [shutil.which("swipl") or "swipl", "-q", "-s", petta / "src" / "main.pl", "--", "/dev/stdin", "silent"],
        input=oracle.program(mettabase / "lib" / "mettabase" / "snars"),
        cwd=mettabase,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    answers = dict(re.findall(r"^\(Case (\d+) (.*)\)$", result.stdout, flags=re.MULTILINE))
    assert len(answers) == len(oracle.cases), result.stdout + result.stderr
    worst: dict[str, tuple[int, int, float]] = {}
    disagreements = []
    for i, (name, call, shape, want) in enumerate(oracle.cases):
        match = shape.match(answers[str(i)])
        assert match, (call, answers[str(i)])
        got = tuple(float(g) for g in match.groups())
        error = max(abs(g - w) for g, w in zip(got, want, strict=True))
        if error > 1e-9:
            disagreements.append((name, call, got, want))
        count, off, largest = worst.get(name, (0, 0, 0.0))
        worst[name] = (count + 1, off + (error > 1e-9), max(largest, error))
    print(f"\nSNARS truth.metta at {mettabase} against semrewrite.opinion, {len(oracle.cases)} calls:")
    for name, (count, off, largest) in sorted(worst.items()):
        print(f"  {name:20} {count:4} calls, {off:3} beyond 1e-9, largest difference {largest:.3g}")
    assert not disagreements, disagreements[:5]
