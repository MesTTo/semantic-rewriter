"""Purpose: write admitted rewrites as SNARS rules, the opinion-typed MeTTa S-expressions that mettabase's subjective
backward chainer reads and GoalChainer reasons with. The rewrite owns(X0, X1) <= owned_by_p127(X1, X0), held at the
opinion (0.957, 0.016, 0.027, 0.5), is the row

    (: (Opinion 0.957 0.016 0.027 0.5) rw-owns-owned_by_p127-10 (-> (owned_by_p127 $X1 $X0) (owns $X0 $X1)))

the rule form sbc.metta documents, `(: <Opinion> <name> (-> <premise> <conclusion>))`. In a space beside facts such
as `(: (Opinion 0.9 0.0 0.1 0.5) wd1 (owned_by_p127 youtube google))`, the chainer applies it by subjective-logic
deduction: `(sbc (space &kb) (fromNumber 2) (: $o $proof (owns google youtube)))` proves the goal through the proof
`(rw-owns-owned_by_p127-10 wd1)`.
"""

from __future__ import annotations

from semrewrite.model import Rewrite, is_variable
from semrewrite.opinion import Opinion


def rule_name(rewrite: Rewrite) -> str:
    """The row's name, which the chainer's proof trees cite: rw-, the conclusion, the premise and the mapping's digits."""
    return f"rw-{rewrite.conclusion}-{rewrite.premise}-{''.join(str(m) for m in rewrite.mapping)}"


def _number(x: float) -> str:
    """A float as PeTTa reads a number, to twelve significant digits, positional where Python would write an
    exponent: arithmetic noise such as 0.5 * 0.1 = 0.04999999999999999 reads 0.05, and an opinion's masses still sum
    to 1 within 1e-12.

    >>> _number(0.957), _number(1.0), _number(2.5e-05), _number(0.5 * 0.1)
    ('0.957', '1.0', '0.000025', '0.05')
    """
    rounded = float(f"{x:.12g}")
    text = repr(rounded)
    if "e" in text:
        text = f"{rounded:.20f}".rstrip("0")
        text = text + "0" if text.endswith(".") else text
    return text


def _term(term: str) -> str:
    """A rule argument in MeTTa: a variable X0 is $X0, a constant stays itself."""
    return f"${term}" if is_variable(term) else term


def rule_row(rewrite: Rewrite, opinion: Opinion) -> str:
    """The rewrite as a SNARS rule row holding `opinion`, its variables those of the rule the engines run."""
    rule = rewrite.rule()
    (premise,) = rule.body

    def atom(pred: str, args: tuple[str, ...]) -> str:
        return f"({' '.join([pred, *(_term(a) for a in args)])})" if args else f"({pred})"

    truth = " ".join(_number(x) for x in (opinion.belief, opinion.disbelief, opinion.uncertainty, opinion.base_rate))
    return (f"(: (Opinion {truth}) {rule_name(rewrite)} "
            f"(-> {atom(premise.pred, premise.args)} {atom(rule.head.pred, rule.head.args)}))")
