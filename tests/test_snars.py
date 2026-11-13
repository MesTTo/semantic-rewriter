"""Purpose: tests of the SNARS export: a rewrite becomes the rule row sbc.metta documents, its variables in the order
of the rule the engines run, at any arity, with numbers PeTTa reads; the command line prints a rewriter's admitted
rewrites that way; and mettabase's own chainer, run on PeTTa, proves a conclusion through an exported row by
subjective-logic deduction.

The SNARS check needs PETTA_PATH, a PeTTa checkout that loads mettabase's SNARS (its fixtures pass on PeTTa d8d4692),
METTABASE_PATH, the mettabase checkout, and swipl on the PATH; it runs PeTTa with mettabase as the working directory
and the program on standard input, so it writes nothing, and is skipped, with the reason, when what it needs is
missing.
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
from semrewrite.snars import rule_row

OWNS = Rewrite("owns", "owned_by_p127", (1, 0))


def test_a_rewrite_is_a_rule_row_with_its_opinion() -> None:
    row = rule_row(OWNS, Opinion(0.957, 0.016, 0.027))
    assert row == ("(: (Opinion 0.957 0.016 0.027 0.5) rw-owns-owned_by_p127-10 "
                   "(-> (owned_by_p127 $X1 $X0) (owns $X0 $X1)))")


def test_variables_follow_the_rule_at_any_arity_and_numbers_have_no_exponent() -> None:
    row = rule_row(Rewrite("between", "located", (2, 0, 1)), Opinion(0.25, 0.75 - 2.5e-05, 2.5e-05, 0.1))
    assert row.endswith("rw-between-located-201 (-> (located $X2 $X0 $X1) (between $X0 $X1 $X2)))")
    assert "e-" not in row and "0.000025" in row
    assert rule_row(Rewrite("p", "q", ()), Opinion(1.0, 0.0, 0.0)).endswith("(-> (q) (p)))")


GLOSSES = {"mum": "{x} is the mum of {y}", "dad": "{x} is the dad of {y}", "parent": "{x} is a parent of {y}"}
KB_ = KB("tom", glosses=GLOSSES, facts=(Atom("mum", ("alice", "tom")), Atom("dad", ("bob", "tom"))))
SCORES = {("mum", "parent", (0, 1)): 0.9, ("dad", "parent", (0, 1)): 0.8}


def test_the_command_line_prints_the_admitted_rewrites_as_rows(monkeypatch, capsys) -> None:
    """At judge trust 0.5, mum's (0.9, 0.1, 0) reads (0.45, 0.05, 0.5), the stronger row first."""
    from semrewrite import cli, judges

    monkeypatch.setattr(cli, "_judge", lambda args, kb: TableJudge(SCORES, threshold=0.2, trust=0.5))
    monkeypatch.setattr(cli.formats, "load", lambda path: KB_)
    monkeypatch.setattr(judges, "default_data_trust", lambda name: 0.0)
    assert main(["rewrites", "tom.kb", "parent", "--retriever", "all", "--snars"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "(: (Opinion 0.45 0.05 0.5 0.5) rw-parent-mum-01 (-> (mum $X0 $X1) (parent $X0 $X1)))",
        "(: (Opinion 0.4 0.1 0.5 0.5) rw-parent-dad-01 (-> (dad $X0 $X1) (parent $X0 $X1)))",
    ]


@pytest.mark.skipif(
    not (os.environ.get("PETTA_PATH") and os.environ.get("METTABASE_PATH") and shutil.which("swipl")),
    reason="the SNARS check needs PETTA_PATH, METTABASE_PATH and swipl",
)
def test_snars_proves_a_conclusion_through_an_exported_row() -> None:
    petta, mettabase = Path(os.environ["PETTA_PATH"]), Path(os.environ["METTABASE_PATH"])
    program = "\n".join([
        "!(import! &self lib/mettabase/snars)",
        "!(bind! &kb (new-space))",
        "!(add-atom &kb (: (Opinion 0.9 0.0 0.1 0.5) f1 (mum alice tom)))",
        f"!(add-atom &kb {rule_row(Rewrite('parent', 'mum', (0, 1)), Opinion(0.45, 0.05, 0.5))})",
        "!(let $r (collapse (sbc (space &kb) (fromNumber 1) (: $o $p (parent $x tom)))) (Proved $r))",
        "",
    ])
    result = subprocess.run(
        [shutil.which("swipl") or "swipl", "--stack_limit=8g", "-q", "-s", petta / "src" / "main.pl", "--",
         "/dev/stdin", "silent"],
        input=program, cwd=mettabase, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    proved = re.search(r"\(: \(Opinion (\S+) (\S+) (\S+) (\S+)\) \(rw-parent-mum-01 f1\) \(parent alice tom\)\)",
                       result.stdout)
    assert proved, result.stdout + result.stderr
    b, d, u, a = (float(x) for x in proved.groups())
    # Deduction through a rule held at P = 0.7 from a fact at P = 0.95 projects between the rule's base rate and 1,
    # as an opinion whose masses sum to 1.
    assert abs(b + d + u - 1.0) < 1e-9
    assert 0.5 < b + a * u < 1.0
