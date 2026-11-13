"""Purpose: tests of the SNARS export: a rewrite becomes its validity, believed at its opinion, and a certain-rule row
deriving its conclusion from its premise and that validity; a candidate's marginal is a line of its own; statements are
written as SNARS's canonical trees at any arity, with numbers PeTTa reads; the command line prints a rewriter's admitted
rewrites that way; and SNARS itself, run on PeTTa, proves a conclusion through the exported lines at the certain rule's
truth, and counts a validity a proof uses twice once.

The SNARS check needs PETTA_PATH, a PeTTa checkout that loads mettabase's SNARS (mettabase's main loads on PeTTa
d8d4692, GoalChainer's mettabase line, whose imports are file-relative, on PeTTa's import-overhaul), METTABASE_PATH, the
mettabase checkout, and swipl on the PATH; it imports SNARS by absolute path, which both resolve, and runs PeTTa with
mettabase as the working directory, mettabase's and PeTTa's Python packages on PYTHONPATH as mettabase's
scripts/use-petta-env.sh puts them, and the program on standard input, so it writes nothing, and is skipped, with the
reason, when what it needs is missing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from stubs import TableJudge

from semrewrite.cli import main
from semrewrite.kb import KB
from semrewrite.model import Atom, Rewrite
from semrewrite.opinion import Opinion
from semrewrite.snars import believed, certain_rule, marginal, rule_rows, statement

MUM = Rewrite("parent", "mum", (0, 1))
VALID_MUM = '(sh (tag P d s ()) "valid" (args ((arg s (sh-atom (tag C c NoRoles ()) "rw-parent-mum-01")))))'


def test_a_rewrite_becomes_its_validity_belief_and_a_rule_kit_row() -> None:
    """The example semrewrite.snars gives: (0.9, 0.05, 0.05) is 36 units of evidence for and 2 against."""
    belief, row = rule_rows(MUM, Opinion(0.9, 0.05, 0.05, 0.2))
    assert belief == f'!(believe! {VALID_MUM} ((evidence 36.0 2.0) (rate 0.2) (source "semrewrite")))'
    assert row == ('(sn rw rw-parent-mum-01 certain-rule (sh (tag P d so ()) "mum" (args ((arg s $X0) (arg o $X1)))) '
                   f'{VALID_MUM} (sh (tag P d so ()) "parent" (args ((arg s $X0) (arg o $X1)))))')


def test_statements_take_snars_canonical_role_order_at_any_arity() -> None:
    """Positions take s, o, i, then x, and SNARS sorts roles s, c, i, o, x, so a third argument comes second."""
    assert statement("p", ()) == '(sh (tag P d NoRoles ()) "p" (args ()))'
    assert statement("p", ("X0",)) == '(sh (tag P d s ()) "p" (args ((arg s $X0))))'
    assert statement("p", ("X0", "X1", "X2", "X3", "X4")) == (
        '(sh (tag P d sioxx ()) "p" (args ((arg s $X0) (arg i $X2) (arg o $X1) (arg x $X3) (arg x $X4))))')
    _, row = rule_rows(Rewrite("between", "located", (2, 0, 1)), Opinion(0.25, 0.7, 0.05))
    assert row.endswith('(sh (tag P d sio ()) "between" (args ((arg s $X0) (arg i $X2) (arg o $X1)))))')
    assert '"located" (args ((arg s $X2) (arg i $X1) (arg o $X0))))' in row


def test_a_question_rule_is_a_certain_rule_row_and_a_candidate_marginal_a_line() -> None:
    """A two-premise rule is the certain-rule row over its three trees, and a candidate's prior is sn marginal! on its
    concept, written as PeTTa reads a number."""
    row = certain_rule("rule-0", statement("owns", ("X", "Y")), statement("owns", ("Y", "Z")), statement("owns", ("X", "Z")))
    assert row.startswith('(sn rw rule-0 certain-rule (sh (tag P d so ()) "owns" (args ((arg s $X) (arg o $Y)))) ')
    assert row.endswith('(sh (tag P d so ()) "owns" (args ((arg s $X) (arg o $Z)))))')
    assert marginal("tom", 1 / 8) == '!(sn marginal! (sh-atom (tag C c NoRoles ()) "tom") 0.125)'


def test_a_dogmatic_opinion_is_written_at_snars_floor_on_uncertainty() -> None:
    """SNARS holds u at 0.001 or more, so a certain opinion is the 2,000 units of evidence it reads back nearest."""
    assert "(evidence 2000.0 0.0)" in believed(Atom("mum", ("alice", "tom")), Opinion(1.0, 0.0, 0.0), "kb")
    belief, _ = rule_rows(MUM, Opinion(0.25, 0.75 - 2.5e-05, 2.5e-05, 0.1))
    assert "e-" not in belief and "e+" not in belief


GLOSSES = {"mum": "{x} is the mum of {y}", "dad": "{x} is the dad of {y}", "parent": "{x} is a parent of {y}"}
KB_ = KB("tom", glosses=GLOSSES, facts=(Atom("mum", ("alice", "tom")), Atom("dad", ("bob", "tom"))))
SCORES = {("mum", "parent", (0, 1)): 0.9, ("dad", "parent", (0, 1)): 0.8}


def test_the_command_line_prints_each_admitted_rewrite_as_its_two_lines(monkeypatch, capsys) -> None:
    """At judge trust 0.5, mum's (0.9, 0.1, 0) reads (0.45, 0.05, 0.5), 1.8 units of evidence for and 0.2 against,
    and comes before dad's; the judge is named as the evidence's source."""
    from semrewrite import cli

    monkeypatch.setattr(cli, "_judge", lambda args, kb: TableJudge(SCORES, threshold=0.2, trust=0.5))
    monkeypatch.setattr(cli.formats, "load", lambda path: KB_)
    assert main(["rewrites", "tom.kb", "parent", "--retriever", "all", "--snars"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f'!(believe! {VALID_MUM} ((evidence 1.8 0.2) (rate 0.5) (source "semrewrite stub")))'
    assert lines[1].startswith("(sn rw rw-parent-mum-01 certain-rule ")
    assert lines[2].startswith("!(believe! ") and '"rw-parent-dad-01"' in lines[2]
    assert lines[3].startswith("(sn rw rw-parent-dad-01 certain-rule ")
    assert len(lines) == 4


_NEEDS_SNARS = pytest.mark.skipif(
    not (os.environ.get("PETTA_PATH") and os.environ.get("METTABASE_PATH") and shutil.which("swipl")),
    reason="the SNARS check needs PETTA_PATH, METTABASE_PATH and swipl",
)
_PROOF = re.compile(r"\(: \(Opinion (\S+) (\S+) (\S+) (\S+)\) (\(.*?\)) \(sh ")


def _snars(lines: list[str]) -> str:
    """PeTTa's standard output on SNARS loaded and `lines`, run in mettabase as its scripts run it."""
    petta, mettabase = Path(os.environ["PETTA_PATH"]), Path(os.environ["METTABASE_PATH"])
    path = [str(mettabase / "src"), str(mettabase / "lib"), str(petta / "python"), os.environ.get("PYTHONPATH", "")]
    result = subprocess.run(
        [shutil.which("swipl") or "swipl", "--stack_limit=8g", "-q", "-s", petta / "src" / "main.pl", "--",
         "/dev/stdin", "silent"],
        input="\n".join([f'!(import! &self "{mettabase / "lib" / "mettabase" / "snars"}")', *lines, ""]),
        cwd=mettabase, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in path if p)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@_NEEDS_SNARS
def test_snars_proves_a_conclusion_through_the_exported_lines_at_the_certain_rules_truth() -> None:
    """SNARS's sn prove reaches parent(alice, tom) from the fact mum(alice, tom) and the rewrite's validity, through
    the exported row, at the certain rule's truth: belief the product of the fact's and the validity's P as SNARS reads
    them back from the evidence written, no disbelief, at tom's seeded marginal."""
    fact, validity = Opinion(0.8, 0.0, 0.2), Opinion(0.45, 0.05, 0.5)
    out = _snars([believed(Atom("mum", ("alice", "tom")), fact, "kb"), *rule_rows(MUM, validity), marginal("tom", 0.25),
                  f"!(let $r (collapse (sn prove {statement('parent', ('alice', 'tom'))} 3)) (Proved $r))"])
    proved = re.search(r"\(: \(Opinion (\S+) (\S+) (\S+) (\S+)\) \(rw-parent-mum-01 (s[0-9a-f]+) (s[0-9a-f]+)\) "
                       r'\(sh \(tag P d so \(\)\) "parent"', out)
    assert proved, out
    b, d, u, a = (float(x) for x in proved.groups()[:4])
    # The fact's and the validity's sids are the proof's leaves, so a second proof through the rewrite shares one.
    assert proved.group(5) != proved.group(6)
    belief = fact.projection * validity.projection
    assert (b, d, u, a) == pytest.approx((belief, 0.0, 1.0 - belief, 0.25), abs=1e-9)


@_NEEDS_SNARS
def test_snars_counts_a_validity_a_proof_uses_twice_once() -> None:
    """owns(a, b2) through the rewrite owns(X0, X1) <= owned_by(X1, X0) at both links and a certain transitive rule
    uses the rewrite's validity twice, and SNARS conditions on it: its P is the validity's P times the answer's P with
    the rewrite valid, plus the rest times its P with the rewrite invalid, when each link concludes the candidates'
    prior m. Counted twice, as step-by-step composition counts it, the answer would sit far lower."""
    fact, validity, m = Opinion(0.8, 0.0, 0.2), Opinion(0.7, 0.1, 0.2, 0.214), 0.25
    owns = Rewrite("owns", "owned_by", (1, 0))
    transitive = certain_rule("rule-0", statement("owns", ("X", "Y")), statement("owns", ("Y", "Z")),
                              statement("owns", ("X", "Z")))
    out = _snars([*(marginal(b, m) for b in ("b1", "b2")), believed(Atom("owned_by", ("b1", "a")), fact, "kb"),
                  believed(Atom("owned_by", ("b2", "b1")), fact, "kb"), *rule_rows(owns, validity), transitive,
                  f"!(let $r (collapse (sn prove {statement('owns', ('a', 'b2'))} 2)) (Proved $r))"])
    (proof,) = _PROOF.findall(out)
    b, _, u, a = (float(x) for x in proof[:4])

    def answer(link: float) -> float:
        """owns(a, b2)'s P when each link has belief `link`: the transitive rule's belief is the links' P squared."""
        p = link + m * (1.0 - link)
        return p * p + m * (1.0 - p * p)

    conditioned = validity.projection * answer(fact.projection) + (1.0 - validity.projection) * answer(0.0)
    assert b + a * u == pytest.approx(conditioned, abs=1e-9)
    assert b + a * u - answer(fact.projection * validity.projection) > 0.05
