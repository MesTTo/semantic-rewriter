"""Purpose: subjective-logic binomial opinions as the Subjective Logic Narsese specification (SLN) defines
them and SNARS's truth.metta computes them: their evidence, Beta and NAL readings, the operators the graded rewriter
combines them with.

An opinion (b, d, u, a) about a proposition holds belief b, disbelief d and uncertainty u, which sum to 1, and a base
rate a, the probability of the proposition before any evidence (J Def 3.1). r units of evidence for the proposition
and s against it are the opinion b = r/(r+s+W), d = s/(r+s+W), u = W/(r+s+W) (J 3.11), so

    >>> Opinion.from_evidence(2, 1, base_rate=0.9)
    Opinion(belief=0.4, disbelief=0.2, uncertainty=0.4, base_rate=0.9)

is the book's running example, which projects P = b + a u = 0.76 and is the Beta distribution Beta(3.8, 1.2).
W = 2 is the weight of the non-informative prior: under it a vacuous opinion at a = 1/2 is the uniform Beta(1, 1).
SLN T1 fixes NAL's evidential horizon k to the same 2, which makes NAL confidence c = b + d; ONA and OpenNARS use
k = 1, so (f, c) for them is to_nal(k=1) (SLN section 6).

Finite evidence always gives u > 0, and every constructor from evidence keeps it so, as SLN T1 requires of the
evidence arena. The constructor also accepts dogmatic opinions (u = 0, as J Def 3.1 allows): the rewriter answers a
query the knowledge base derives outright as (1, 0, 0), a fact of SLN T1's structural layer rather than evidence.

Equation numbers (J x.y) are those of A. Jøsang, Subjective Logic, Springer 2016. Where SNARS's truth.metta (at
mettabase c9bbbe8) computes otherwise, this module follows SLN and the book:

* SNARS's `sn opinion` view and `sn fuse-op` floor u at u-min = 0.001, which binds past 1998 units of evidence, and
  at c9bbbe8 do so without renormalising b and d, so b + d + u exceeds 1. This module never floors u; below that
  much evidence the views agree to the bit and fusion to within 1e-15.
* `sn fuse-op` keeps its first argument's base rate; fuse takes the evidence-weighted mean when the base rates
  differ (J Def 12.5, SLN 3.1). SNARS cites J 12.12 for it, which is the book's cinema example; cumulative fusion is
  J Def 12.5, Eq 12.14, and Thm 12.2, Eq 12.17.
* `sn and-op` and `sn or-op` replace a quotient whose denominator is below 1e-6 by 0 (`sn kdiv`), which near
  a_x = a_y = 1 (product) or a_x = a_y = 0 (coproduct) loses the cross terms and leaves b + d + u < 1. Here the
  denominators keep their precision up to the corner, and at the corner, where J 7.1 and J 7.3 are 0/0, the cross
  terms go to uncertainty (conjoin).
* `sn op-to-ev` divides by u and so fails on dogmatic opinions; fuse handles them by J Def 12.5's two cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The non-informative prior weight (J 3.11, SLN T1): with W = 2 a vacuous opinion at a = 1/2 is Beta(1, 1).
W = 2.0

# The slack granted to rounding: b + d + u may miss 1 by this much, a mass or base rate may stray this far outside
# [0, 1] before the constructor moves it back, and evidence may fall this far below 0 relative to the Beta
# parameters it came from. Every operation here loses a few ulps, so the bound only rejects wrong inputs.
_TOLERANCE = 1e-9



@dataclass(frozen=True, slots=True)
class Opinion:
    """A binomial opinion (b, d, u, a) about one proposition (J Def 3.1).

    b, d and u each lie in [0, 1] and sum to 1 within 1e-9; the base rate a lies in [0, 1]. A value outside [0, 1]
    by no more than 1e-9 is rounding, such as the 1.0000000000000002 a product of opinions can compute for u = 1,
    and is stored as the end of the interval it overshot.
    """

    belief: float
    disbelief: float
    uncertainty: float
    base_rate: float = 0.5

    def __post_init__(self) -> None:
        for field in ("belief", "disbelief", "uncertainty", "base_rate"):
            value = getattr(self, field)
            if -_TOLERANCE <= value < 0.0 or 1.0 < value <= 1.0 + _TOLERANCE:
                object.__setattr__(self, field, min(max(value, 0.0), 1.0))
            _check_unit(field.replace("_", " "), getattr(self, field))
        masses = (self.belief, self.disbelief, self.uncertainty)
        if abs(math.fsum(masses) - 1.0) > _TOLERANCE:
            msg = f"belief, disbelief and uncertainty must sum to 1 (J 3.1), got {math.fsum(masses)!r} from {masses}"
            raise ValueError(msg)

    @classmethod
    def from_evidence(cls, positive: float, negative: float, base_rate: float = 0.5) -> Opinion:
        """The opinion of `positive` units of evidence for the proposition and `negative` against it (J 3.11).

        No evidence is the vacuous opinion (0, 0, 1, a); any finite evidence leaves u > 0.
        """
        if not (math.isfinite(positive) and math.isfinite(negative) and positive >= 0.0 and negative >= 0.0):
            msg = f"evidence must be finite and non-negative, got r={positive!r}, s={negative!r}"
            raise ValueError(msg)
        # J 3.11; SNARS's `sn opinion` divides by the same (r + s) + 2.0, so the two agree to the bit.
        total = positive + negative + W
        return cls(positive / total, negative / total, W / total, base_rate)

    def evidence(self) -> tuple[float, float]:
        """(r, s), the evidence for and against the proposition (J 3.11 inverted, SNARS `sn op-to-ev`).

        Raises ValueError for a dogmatic opinion, which is infinite evidence.
        """
        if self.uncertainty == 0.0:
            msg = f"a dogmatic opinion (u = 0) is infinite evidence (J 3.11): {self}"
            raise ValueError(msg)
        # J 3.11: r = W b/u, s = W d/u, in SNARS's order of operations.
        return W * self.belief / self.uncertainty, W * self.disbelief / self.uncertainty

    @classmethod
    def from_probability(cls, probability: float, weight: float, base_rate: float = 0.5) -> Opinion:
        """A source's `probability` that the proposition holds, read as `weight` units of evidence split by it.

        A judge that says 0.9 with weight 10 counts as 9 units for and 1 against: from_evidence(9, 1).
        """
        _check_unit("probability", probability)
        if not (math.isfinite(weight) and weight >= 0.0):
            msg = f"weight must be finite and non-negative, got {weight!r}"
            raise ValueError(msg)
        return cls.from_evidence(probability * weight, (1.0 - probability) * weight, base_rate)

    @property
    def projection(self) -> float:
        """The projected probability P = b + a u (J 3.2, SNARS `sn project`), the mean of the opinion's Beta.

        With a = 1/2 and k = W it is NAL's expectation c(f - 1/2) + 1/2 (SLN T2).
        """
        return self.belief + self.base_rate * self.uncertainty

    @property
    def variance(self) -> float:
        """The variance of the opinion's Beta, P(1 - P) u/(W + u) (J 3.10); 0 for a dogmatic opinion.

        Cerutti, Kaplan, Kimmig and Sensoy (Machine Learning, 2022; arXiv 2102.10865, section 3) use the same
        formula; with E[p] = P, E[p^2] = P^2 + variance.
        """
        p = self.projection
        return p * (1.0 - p) * self.uncertainty / (W + self.uncertainty)

    def beta(self) -> tuple[float, float]:
        """(alpha, beta) of the Beta distribution the opinion is (J 3.7): alpha = r + a W, beta = s + (1 - a) W.

        Raises ValueError for a dogmatic opinion, whose Beta is a point mass with infinite parameters.
        """
        r, s = self.evidence()
        return r + self.base_rate * W, s + (1.0 - self.base_rate) * W

    @classmethod
    def from_beta(cls, alpha: float, beta: float, base_rate: float) -> Opinion:
        """The opinion whose Beta is Beta(alpha, beta) at `base_rate`, the exact inverse of beta() (J 3.7).

        J 3.7 gives r = alpha - a W and s = beta - (1 - a) W, so a Beta with alpha < a W or beta < (1 - a) W is
        wider than the prior the base rate fixes, and no opinion has it: that raises ValueError.
        """
        r = _clear_rounding(alpha - base_rate * W, alpha)
        s = _clear_rounding(beta - (1.0 - base_rate) * W, beta)
        if not (r >= 0.0 and s >= 0.0):
            msg = (
                f"Beta({alpha!r}, {beta!r}) is wider than the prior at base rate {base_rate!r}: "
                f"an opinion needs alpha >= a W = {base_rate * W!r} and beta >= (1 - a) W = {(1.0 - base_rate) * W!r}"
            )
            raise ValueError(msg)
        return cls.from_evidence(r, s, base_rate)

    @classmethod
    def from_moments(cls, mean: float, variance: float, base_rate: float) -> Opinion:
        """The opinion with projection `mean` whose Beta has `variance` where one does, at `base_rate`.

        Moment matching gives a Beta of strength alpha + beta = m(1 - m)/v - 1, so u = W v/(m(1 - m) - v) (J 3.10
        solved for u). Where that u would make b = m - a u or d = 1 - m - (1 - a) u negative, u falls to the largest
        value keeping both non-negative, m/a or (1 - m)/(1 - a), the uncertainty-maximised opinion of mean m (J 3.25
        to 3.27): this is Cerutti et al.'s strength max{m(1 - m)/v - 1, W a/m, W (1 - a)/(1 - m)} (Machine Learning,
        2022, section 3), each bound with a zero denominator dropped. So from_moments(0.1, 0.05, 0.9) is
        (0, 0.889, 0.111, 0.9), whose variance is below 0.05. Variance 0 is a point mass, of infinite strength
        whatever the mean, so it gives the dogmatic (m, 1 - m, 0, a): a fact the knowledge base states, mean 1 and
        variance 0, is (1, 0, 0, a) even at a = 1. Variance m(1 - m) > 0, two points, has no strength at all.

        It inverts (projection, variance) for every opinion except those with u > 0 that project 0 or 1, which have
        P = a in {0, 1}, such as (0.5, 0, 0.5, 1) and the vacuous (0, 0, 1, 1): all of them have variance 0, and this
        returns the point mass (a, 1 - a, 0, a).
        """
        _check_unit("mean", mean)
        _check_unit("base rate", base_rate)
        spread = mean * (1.0 - mean)
        if not 0.0 <= variance <= spread:
            msg = f"a distribution on [0, 1] with mean {mean!r} has variance in [0, {spread!r}], got {variance!r}"
            raise ValueError(msg)
        if variance == 0.0:
            u = 0.0
        else:
            # 0 < v <= m(1 - m) puts m inside (0, 1), where a base rate in [0, 1] leaves at least one bound.
            bounds = [W * variance / (spread - variance)] if variance < spread else []
            if base_rate > 0.0:
                bounds.append(mean / base_rate)
            if base_rate < 1.0:
                bounds.append((1.0 - mean) / (1.0 - base_rate))
            u = min(bounds)
        # The bounds keep b and d non-negative up to rounding, such as m - a (m/a), which the constructor clears.
        return cls(mean - base_rate * u, 1.0 - mean - (1.0 - base_rate) * u, u, base_rate)

    def complement(self) -> Opinion:
        """The opinion about the proposition's negation, (d, b, u, 1 - a) (SLN 3.2, SNARS `sn complement`)."""
        return Opinion(self.disbelief, self.belief, self.uncertainty, 1.0 - self.base_rate)

    def fuse(self, other: Opinion) -> Opinion:
        """Cumulative fusion: the opinion of both sources' evidence together (J Def 12.5, SNARS `sn fuse-op`).

        It assumes the sources saw disjoint evidence. The evidence adds (J Thm 12.2, Eq 12.17), so two opinions of
        (2, 0) fuse to (4, 0), b = 2/3, u = 1/3 (SLN 3.1). Opinions about one proposition should share a base rate
        (SLN 3.1); when they don't, the fused base rate is their evidence-weighted mean, which is J 12.14's, and the
        plain mean when neither holds evidence. A dogmatic opinion outweighs any finite evidence (J 12.14 with
        u = 0), and two dogmatic opinions average with weight 1/2 each (J 12.15 with the book's gamma = 1/2), which
        makes fusing three opinions associative only while at most one is dogmatic.
        """
        u1, u2 = self.uncertainty, other.uncertainty
        if u1 == 0.0 and u2 == 0.0:
            rate = _mean_rate(self.base_rate, 1.0, other.base_rate, 1.0)
            return Opinion((self.belief + other.belief) / 2.0, (self.disbelief + other.disbelief) / 2.0, 0.0, rate)
        # J 12.14: b = (b1 u2 + b2 u1)/k, u = u1 u2/k with k = u1 + u2 - u1 u2, which is evidence addition by Thm 12.2.
        # It is written with the shares u2/k and u1/k, which lie in [0, 1], so it never overflows as r = W b/u can,
        # and u = u1 (u2/k) underflows only where u1 u2/k itself lies below the smallest float; k is a sum of
        # non-negative terms, 0 only when both are dogmatic, the case above.
        k = u1 + u2 * (1.0 - u1)
        share1, share2 = u2 / k, u1 / k
        # J 12.14's base rate weighs a1 by u2 (1 - u1) and a2 by u1 (1 - u2), in proportion to the evidence r + s.
        rate = _mean_rate(self.base_rate, share1 * (1.0 - u1), other.base_rate, share2 * (1.0 - u2))
        return Opinion(
            self.belief * share1 + other.belief * share2,
            self.disbelief * share1 + other.disbelief * share2,
            u1 * share1,
            rate,
        )

    def with_base_rate(self, base_rate: float) -> Opinion:
        """The same belief, disbelief and uncertainty at `base_rate`: the prior the opinion projects to where it says
        nothing, which belongs to the population its proposition is drawn from (J 3.5), as a judge's calibration sets
        it for the proposals it judges."""
        return Opinion(self.belief, self.disbelief, self.uncertainty, base_rate)

    def discount(self, trust: float) -> Opinion:
        """The opinion as relayed by a source trusted with probability `trust` (J Def 14.6, SNARS `sn discount-op`).

        J 14.6 scales belief and disbelief by the trust p and gives the rest to uncertainty: b' = p b, d' = p d,
        u' = 1 - p(b + d), a' = a. Its p is the projected probability of the trust opinion, so a caller holding one
        passes trust_opinion.projection. The book's opinion (0.45, 0.35, 0.20, 0.25) relayed by a source trusted as
        (0.20, 0.40, 0.40, 0.75), P = 0.5, becomes (0.225, 0.175, 0.6, 0.25) (J 14.8). In NAL terms it keeps f and
        scales c by p, which is also temporal projection with p = lambda^dt (SLN 3.3).
        """
        _check_unit("trust", trust)
        # u' = 1 - p(b + d) is (1 - p) + p u because b + d + u = 1; this form is exact at p = 0 (vacuous) and at
        # p = 1 (the opinion unchanged), and stays in [0, 1] where 1 - p(b + d) can round below 0.
        return Opinion(
            trust * self.belief,
            trust * self.disbelief,
            (1.0 - trust) + trust * self.uncertainty,
            self.base_rate,
        )

    def conjoin(self, other: Opinion) -> Opinion:
        """The opinion that both independent propositions hold: binomial multiplication (J Def 7.1, `sn and-op`).

        It projects P(x) P(y) (J 7.2). The book's (0.75, 0.15, 0.10, 0.50) and (0.10, 0.00, 0.90, 0.20) multiply to
        (0.15, 0.15, 0.70, 0.10) to two places. J 7.1 shares the cross terms b_x u_y + u_x b_y between belief and
        uncertainty by quotients over 1 - a_x a_y, which are 0/0 when both base rates are 1; there P(x) P(y) = b + u
        no longer fixes the split, and the cross terms all go to uncertainty, since J 7.1.3 designs the product to
        keep the most uncertainty and b_x b_y is the least belief it allows. So (0.5, 0, 0.5, 1) with itself is
        (0.25, 0, 0.75, 1), and a fact at base rate 1, (1, 0, 0, 1), with itself stays (1, 0, 0, 1).
        """
        belief, disbelief, uncertainty, rate, _ = _multiply(self, 1.0 - self.base_rate, other, 1.0 - other.base_rate)
        return Opinion(belief, disbelief, uncertainty, rate)

    def disjoin(self, other: Opinion) -> Opinion:
        """The opinion that either independent proposition holds: comultiplication (J Def 7.2, Eq 7.3, `sn or-op`).

        It projects P(x) + P(y) - P(x) P(y) (J 7.4). The book's (0.75, 0.15, 0.10, 0.50) and (0.35, 0.00, 0.65,
        0.20) give (0.84, 0.06, 0.10, 0.60) to two places. Where both base rates are 0, J 7.3's quotients are 0/0
        and the cross terms d_x u_y + u_x d_y go to uncertainty, the dual of conjoin's corner.
        """
        # J 7.3 is J 7.1 under the duality b <-> d, a <-> 1 - a (J 7.1.3, de Morgan): x or y is the complement of
        # (not x) and (not y). Passing each original a as its complement's complement keeps the coproduct's base
        # rate a_x + a_y - a_x a_y, which _multiply returns as its denominator, free of 1 - (1 - a) rounding.
        belief, disbelief, uncertainty, _, rate = _multiply(
            self.complement(), self.base_rate, other.complement(), other.base_rate
        )
        return Opinion(disbelief, belief, uncertainty, rate)

    def conflict(self, other: Opinion) -> float:
        """The degree of conflict |P1 - P2| (1 - u1)(1 - u2) (J Def 4.20, Eqs 4.61 to 4.63; SLN 3.8; SNARS `sn dc`).

        It is large only when both opinions are confident and project far apart: the book's (0.05, 0.15, 0.80, 0.10)
        and (0.68, 0.22, 0.10, 0.10) project 0.13 and 0.69 yet conflict only 0.56 * 0.2 * 0.9 = 0.10 (J 4.64).
        """
        return abs(self.projection - other.projection) * ((1.0 - self.uncertainty) * (1.0 - other.uncertainty))

    def to_nal(self, k: float = W) -> tuple[float | None, float]:
        """NAL truth (f, c) at evidential horizon k (SLN T1): f = r/(r + s) and c = (r + s)/(r + s + k).

        Written in opinion terms, f = b/(b + d) and c = W(b + d)/(W(b + d) + k u), which needs no division by u:
        a dogmatic opinion has c = 1. f is None for a vacuous opinion, which has no evidence to take a frequency of.
        With k = W this is T1's clean form c = b + d. ONA and OpenNARS take k = 1 (SLN section 6).
        """
        _check_horizon(k)
        mass = self.belief + self.disbelief
        return (self.belief / mass if mass > 0.0 else None), W * mass / (W * mass + k * self.uncertainty)

    @classmethod
    def from_nal(cls, f: float | None, c: float, k: float = W, base_rate: float = 0.5) -> Opinion:
        """The opinion of NAL truth (f, c) at evidential horizon k, inverting to_nal (SLN T1 and section 6).

        u = W(1 - c)/(W(1 - c) + k c), b = f(1 - u), d = (1 - f)(1 - u): the evidence w = k c/(1 - c) mapped by
        J 3.11, so ONA's %1.0;0.9% at k = 1 is w = 9 and the opinion (9/11, 0, 2/11), where reading b = f c off it
        would give (0.9, 0, 0.1) and halve the uncertainty (SLN section 6). c = 1 gives a dogmatic opinion; f may be
        None only when c = 0, as to_nal returns for a vacuous opinion.
        """
        _check_horizon(k)
        _check_unit("confidence", c)
        if f is None:
            if c != 0.0:
                msg = f"a frequency is needed whenever the confidence is above 0, got c={c!r}"
                raise ValueError(msg)
            f = 0.0
        else:
            _check_unit("frequency", f)
        # 1 - u is its own fraction k c/D rather than a subtraction, which would lose a confidence below 1e-16.
        denominator = W * (1.0 - c) + k * c
        certainty = k * c / denominator
        return cls(f * certainty, (1.0 - f) * certainty, W * (1.0 - c) / denominator, base_rate)


def _check_unit(name: str, value: float) -> None:
    """Raise unless `value` lies in [0, 1], naming it `name`; NaN fails the comparison and raises too."""
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must lie in [0, 1], got {value!r}"
        raise ValueError(msg)


def _clear_rounding(value: float, scale: float) -> float:
    """`value`, or 0 when it lies below 0 by no more than rounding of numbers the size of `scale` could leave."""
    return 0.0 if -_TOLERANCE * max(1.0, abs(scale)) <= value < 0.0 else value


def _check_horizon(k: float) -> None:
    """Raise unless the evidential horizon k is positive and finite (SLN T1 needs k > 0)."""
    if not (math.isfinite(k) and k > 0.0):
        msg = f"the evidential horizon k must be positive and finite, got {k!r}"
        raise ValueError(msg)


def _mean_rate(rate1: float, weight1: float, rate2: float, weight2: float) -> float:
    """The fused base rate: rate1 and rate2 averaged by weight, equally when both weights are 0 (J 12.14, 12.15).

    Equal rates come back unchanged, where the weighted mean of two equal numbers can round off them.
    """
    if rate1 == rate2:
        return rate1
    if weight1 + weight2 == 0.0:
        weight1 = weight2 = 1.0
    mean = (rate1 * weight1 + rate2 * weight2) / (weight1 + weight2)
    # A convex combination lies between its points; rounding can step just past one, and past 1.
    return min(max(mean, min(rate1, rate2)), max(rate1, rate2))


def _multiply(
    x: Opinion, x_rate_complement: float, y: Opinion, y_rate_complement: float
) -> tuple[float, float, float, float, float]:
    """J 7.1's product of independent opinions x and y as (b, d, u, a, 1 - a), given 1 - a of each argument.

    b = b_x b_y + q_b and u = u_x u_y + q_u, where the quotients q_b and q_u over 1 - a_x a_y share the cross terms
    b_x u_y + u_x b_y: their numerators sum to the cross terms times 1 - a_x a_y. That denominator is computed as
    (1 - a_x) + a_x (1 - a_y), a sum of non-negative terms that keeps its precision where a_x a_y is near 1 and is 0
    only when both base rates are 1; the same sum is the product's 1 - a. Time: a fixed 30 or so float operations.
    """
    ax, ay = x.base_rate, y.base_rate
    belief_by_uncertainty = x.belief * y.uncertainty
    uncertainty_by_belief = x.uncertainty * y.belief
    scale = max(x_rate_complement, y_rate_complement)
    if scale == 0.0:
        # Both base rates are 1: J 7.1's quotients are 0/0 and P(x) P(y) = b + u leaves the split open. J 7.1.3
        # designs the product to keep the most uncertainty, so the cross terms all go to u.
        belief_share, uncertainty_share = 0.0, belief_by_uncertainty + uncertainty_by_belief
    else:
        # The quotients and their shared denominator are linear in the two complements, so dividing both by the
        # larger leaves the quotients unchanged and keeps their precision where the complements are tiny: the
        # coproduct of base rates 0 and 5e-324 would otherwise divide subnormals and return u = 1 for u = 2/3.
        xc, yc = x_rate_complement / scale, y_rate_complement / scale
        denominator = xc + ax * yc
        belief_share = (xc * ay * belief_by_uncertainty + ax * yc * uncertainty_by_belief) / denominator
        uncertainty_share = (yc * belief_by_uncertainty + xc * uncertainty_by_belief) / denominator
    # d_x + d_y - d_x d_y written d_x + d_y (1 - d_x), which cannot round above 1.
    disbelief = x.disbelief + y.disbelief * (1.0 - x.disbelief)
    belief = x.belief * y.belief + belief_share
    uncertainty = x.uncertainty * y.uncertainty + uncertainty_share
    return belief, disbelief, uncertainty, ax * ay, x_rate_complement + ax * y_rate_complement
