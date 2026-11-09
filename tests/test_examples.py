"""Purpose: tests of the MeTTa examples: with the oracle judge, which answers from each knowledge base's gold lines,
every engine answers each query examples/README.md asks with the answers it describes, each with a valid derivation;
and the walkthrough script fills exactly the README's generated blocks. Jev's own transcripts need a key, so the
walkthrough regenerates them and these tests guard the knowledge bases.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

pytest.importorskip("metta", reason="the examples are MeTTa files, which loading needs pymetta for")

from semrewrite import conformance as kit
from semrewrite import formats
from semrewrite.engines import available, make
from semrewrite.judges.oracle import OracleJudge
from semrewrite.model import Atom, Rewrite, parse_atom
from semrewrite.protocols import Budget, Program
from semrewrite.retrieve import AllCandidates
from semrewrite.rewriter import STRATEGIES, Rewriter

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

# Each query the walkthrough asks, with the answers its prose describes.
EXPECTED = {
    ("swanson.metta", "worth_testing_against(X, raynauds)"): {("fish_oil", "raynauds"), ("omega_3", "raynauds")},
    ("swanson.metta", "lowers(omega_3, B)"): {("omega_3", "blood_viscosity"), ("omega_3", "platelet_aggregation")},
    ("swanson.metta", "aggravates(B, raynauds)"): {("blood_viscosity", "raynauds"), ("platelet_aggregation", "raynauds")},
    ("family_tree.metta", "descendant_of(X, tom)"): {(p, "tom") for p in ("bob", "liz", "ann", "pat", "jim")},
    ("family_tree.metta", "descendant_of(jim, Y)"): {("jim", p) for p in ("pat", "bob", "tom", "pam")},
    ("family_tree.metta", "mother_of(X, bob)"): {("pam", "bob")},
    ("family_tree.metta", "mother_of(X, jim)"): set(),
}


@pytest.mark.parametrize("engine_name", available())
@pytest.mark.parametrize(("name", "query"), sorted(EXPECTED))
def test_each_example_answers_as_the_walkthrough_says(engine_name: str, name: str, query: str) -> None:
    kb = formats.load(EXAMPLES / name)
    rewriter = Rewriter(kb, OracleJudge(kb.entailed()), make(engine_name), AllCandidates(),
                        budget=Budget(work=5_000_000, seconds=60), proofs=True)
    out = rewriter.ask(query, "rewrite-depth-tabled")
    assert out.finished, (out.status, out.detail)
    assert out.answers == EXPECTED[(name, query)] == kb.answers(parse_atom(query), kb.closure())
    s = STRATEGIES["rewrite-depth-tabled"]
    program = Program(facts=kb.facts, rules=kb.rules, rewrites=tuple(Rewrite(c, p, m) for p, c, m in out.admitted),
                      depth=s.depth, depth_counts=s.depth_counts)
    for row, proof in (out.proofs or {}).items():
        assert kit.proof_problem(proof, Atom(parse_atom(query).pred, row), program) is None, proof.lines()


def test_the_walkthrough_fills_exactly_the_readmes_generated_blocks() -> None:
    spec = importlib.util.spec_from_file_location("walkthrough", EXAMPLES / "walkthrough.py")
    assert spec is not None and spec.loader is not None
    walkthrough = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(walkthrough)
    readme = (EXAMPLES / "README.md").read_text()
    begun = re.findall(r"<!-- BEGIN GENERATED: (\S+) -->", readme)
    ended = re.findall(r"<!-- END GENERATED: (\S+) -->", readme)
    assert begun == ended and len(set(begun)) == len(begun)
    assert set(begun) == set(walkthrough.BLOCKS)
    for argv in (a for spec in walkthrough.BLOCKS.values() if isinstance(spec, list) for a in spec):
        assert f"$ semrewrite {walkthrough.shlex.join(argv)}\n" in readme, argv
