"""Purpose: write admitted rewrites into SNARS's live world, the MeTTa S-expressions mettabase's reasoner and so
GoalChainer reason with. A rewrite becomes two lines: its validity, a statement believed with the rewrite's opinion as
evidence, and the rewrite itself, a row of SNARS's rule kit whose second premise is that statement. The rewrite
parent(X0, X1) <= mum(X0, X1), held at the opinion (0.9, 0.05, 0.05, 0.2), is

    !(believe! (sh (tag P d s ()) "valid" (args ((arg s (sh-atom (tag C c NoRoles ()) "rw-parent-mum-01"))))) ((evidence 36.0 2.0) (rate 0.2) (source "semrewrite")))
    (sn rw rw-parent-mum-01 certain-rule (sh (tag P d so ()) "mum" (args ((arg s $X0) (arg o $X1)))) (sh (tag P d s ()) "valid" (args ((arg s (sh-atom (tag C c NoRoles ()) "rw-parent-mum-01"))))) (sh (tag P d so ()) "parent" (args ((arg s $X0) (arg o $X1)))))

The row is of SNARS's certain-rule class: the rule "premise and validity imply conclusion" is certain, and the rewrite's
doubt lives only in its validity, as in a Bayesian network over the rewrites' validities. `sn prove` gives a
conclusion the product of the premise's and the validity's probabilities as belief and no disbelief, so a doubted
rewrite leaves its conclusion uncertain rather than disbelieved, and where a derivation fails the conclusion keeps its
base rate, the marginal of its object concept. A knowledge base seeds those marginals, each of a question's N
candidate answers at 1/N (`marginal`), as SNARS's open world reads them (SLN 8.2). Every application of the rewrite
shares its one validity statement, as the rewriter's answer beliefs do: SNARS revises them all when evidence about the
rewrite arrives, a proof that uses the statement more than once is conditioned on it so its doubt counts once, and
its proof stamps hold the statement's sid, so two proofs through the rewrite are never fused as independent.
"""

from __future__ import annotations

from collections.abc import Sequence

from semrewrite.model import Atom, Rewrite, is_variable
from semrewrite.opinion import Opinion, W

# The order SNARS's canonical form sorts argument roles in (mettabase src/mettabase/snars/tnf.py, ROLE_PRECEDENCE),
# and the role each argument position takes here: subject, object, indirect object, then specifiers.
_PRECEDENCE = "scioxtjr"
_POSITIONS = ("s", "o", "i")
# SNARS floors an opinion's uncertainty at this (mettabase lib/mettabase/snars/config.metta, `sn config u-min`), so a
# dogmatic opinion is written as the most evidence SNARS can hold.
_U_MIN = 0.001


def rule_name(rewrite: Rewrite) -> str:
    """The row's name, which SNARS's proof trees cite: rw-, the conclusion, the premise and the mapping's digits."""
    return f"rw-{rewrite.conclusion}-{rewrite.premise}-{''.join(str(m) for m in rewrite.mapping)}"


def _number(x: float) -> str:
    """A float as PeTTa reads a number, to twelve significant digits, positional where Python would write an
    exponent: arithmetic noise such as 0.5 * 0.1 = 0.04999999999999999 reads 0.05.

    >>> _number(0.957), _number(1.0), _number(2.5e-05), _number(0.5 * 0.1)
    ('0.957', '1.0', '0.000025', '0.05')
    """
    rounded = float(f"{x:.12g}")
    text = repr(rounded)
    if "e" in text:
        text = f"{rounded:.20f}".rstrip("0")
        text = text + "0" if text.endswith(".") else text
    return text


def statement(pred: str, args: Sequence[str]) -> str:
    """pred(args) as the canonical tree SNARS stores, each argument a variable, $X0, or a concept. An argument's role
    is s, o or i by its position, then x, and the arguments stand in SNARS's role order, so gave(ann, book, bob)
    lists bob, the indirect object, before the book:

    >>> statement("gave", ("ann", "book", "bob"))
    '(sh (tag P d sio ()) "gave" (args ((arg s (sh-atom (tag C c NoRoles ()) "ann")) (arg i (sh-atom (tag C c NoRoles ()) "bob")) (arg o (sh-atom (tag C c NoRoles ()) "book")))))'
    """
    roles = [*_POSITIONS[: len(args)], *("x" for _ in args[len(_POSITIONS):])]
    # sorted is stable, so specifiers keep their positions' order.
    pairs = sorted(zip(roles, args, strict=True), key=lambda ra: _PRECEDENCE.index(ra[0]))
    items = " ".join(f"(arg {role} {_term(arg)})" for role, arg in pairs)
    return f'(sh (tag P d {"".join(r for r, _ in pairs) or "NoRoles"} ()) "{pred}" (args ({items})))'


def _term(arg: str) -> str:
    """An argument in a SNARS tree: a variable X0 is $X0, a constant the concept named after it."""
    return f"${arg}" if is_variable(arg) else f'(sh-atom (tag C c NoRoles ()) "{arg}")'


def _evidence(opinion: Opinion) -> tuple[float, float]:
    """The evidence SNARS reads back as `opinion` (J 3.11), a dogmatic opinion's at SNARS's floor on uncertainty."""
    if opinion.uncertainty > 0.0:
        return opinion.evidence()
    return W * opinion.belief / _U_MIN, W * opinion.disbelief / _U_MIN


def believed(fact: Atom, opinion: Opinion, source: str) -> str:
    """The line believing `fact` into SNARS's world at `opinion`, as from `source`."""
    return _believe(statement(fact.pred, fact.args), opinion, source)


def _believe(tree: str, opinion: Opinion, source: str) -> str:
    """The line believing the statement `tree` at `opinion`, as from `source`."""
    r, s = _evidence(opinion)
    return (f"!(believe! {tree} ((evidence {_number(r)} {_number(s)}) (rate {_number(opinion.base_rate)}) "
            f'(source "{source}")))')


def rule_rows(rewrite: Rewrite, opinion: Opinion, source: str = "semrewrite") -> tuple[str, str]:
    """The two lines putting `rewrite`, held at `opinion` by `source`, into SNARS's world: its validity believed at the
    opinion, and the rule-kit row deriving its conclusion from its premise and its validity. Its variables are those of
    the rule the engines run."""
    rule = rewrite.rule()
    (premise,) = rule.body
    # The rewrite's name is a concept here, not one of the rewriter's constants, so the statement is written directly.
    validity = statement("valid", (rule_name(rewrite),))
    row = certain_rule(rule_name(rewrite), statement(premise.pred, premise.args), validity,
                       statement(rule.head.pred, rule.head.args))
    return _believe(validity, opinion, source), row


def certain_rule(name: str, first: str, second: str, conclusion: str) -> str:
    """The rule-kit row named `name` deriving the statement tree `conclusion` from the trees `first` and `second` by
    SNARS's certain-rule class: a rule that holds whenever both premises do, its conclusion's belief their
    probabilities' product, with no disbelief.

    >>> certain_rule("rule-0", statement("p", ("X",)), statement("q", ("X",)), statement("r", ("X",)))
    '(sn rw rule-0 certain-rule (sh (tag P d s ()) "p" (args ((arg s $X)))) (sh (tag P d s ()) "q" (args ((arg s $X)))) (sh (tag P d s ()) "r" (args ((arg s $X)))))'
    """
    return f"(sn rw {name} certain-rule {first} {second} {conclusion})"


def marginal(concept: str, rate: float) -> str:
    """The line seeding `concept`'s marginal, the base rate SNARS reads for a conclusion whose object it is, at `rate`:
    SLN 8.2's prior, the chance a random individual of the universe of discourse is that concept, so 1/N for each of
    a question's N candidate answers.

    >>> marginal("tom", 0.25)
    '!(sn marginal! (sh-atom (tag C c NoRoles ()) "tom") 0.25)'
    """
    return f"!(sn marginal! {_term(concept)} {_number(rate)})"
