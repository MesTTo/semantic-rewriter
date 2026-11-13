"""Purpose: tests of the MeTTa examples: with the oracle judge, which answers from each knowledge base's gold lines,
every engine answers each query examples/README.md asks with the answers it describes, each with a valid derivation;
on each Wikidata knowledge base, Wikidata's own answer is the one the README describes, and with exactly the rewrites
Wikidata's subproperty hierarchy names every engine returns it; and the walkthrough script fills exactly the README's
generated blocks in README.md and examples/README.md. Jev's own transcripts need a key, so the walkthrough regenerates
them and these tests guard the knowledge bases.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import re
import sys
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
WIKIDATA = EXAMPLES / "wikidata"

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
# Wikidata's own answer to each question as the README describes it: how many entities, and some it names.
WIKIDATA_ANSWERS = {
    "victoria": (1164, {"frederik_x_of_denmark", "charles_iii_q43274", "harald_v_of_norway", "felipe_vi_of_spain"}),
    "leonardo": (854, {"mona_lisa", "the_last_supper_q128910"}),
    "alphabet": (3225, {"google_q95", "youtube_q866", "google_deepmind"}),
}


@pytest.mark.parametrize("engine_name", available())
@pytest.mark.parametrize(("name", "query"), sorted(EXPECTED))
def test_each_example_answers_as_the_walkthrough_says(engine_name: str, name: str, query: str) -> None:
    kb = formats.load(EXAMPLES / name)
    rewriter = Rewriter(kb, OracleJudge(kb.entailed()), make(engine_name), AllCandidates(),
                        budget=Budget(work=5_000_000, seconds=math.inf), proofs=True)
    out = rewriter.ask(query, "rewrite-depth-tabled")
    assert out.finished, (out.status, out.detail)
    assert out.answers == EXPECTED[(name, query)] == kb.answers(parse_atom(query), kb.closure())
    s = STRATEGIES["rewrite-depth-tabled"]
    program = Program(facts=kb.facts, rules=kb.rules, rewrites=tuple(Rewrite(c, p, m) for p, c, m in out.admitted),
                      depth=s.depth, depth_counts=s.depth_counts)
    for row, proof in (out.proofs or {}).items():
        assert kit.proof_problem(proof, Atom(parse_atom(query).pred, row), program) is None, proof.lines()


def _wikidata_questions():
    """examples/wikidata/questions.py, registered under the name the builder and the evaluation import it by."""
    spec = importlib.util.spec_from_file_location("questions", WIKIDATA / "questions.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_wikidata_answer_follows_the_hierarchy_in_each_property_s_direction() -> None:
    """Father (P22) and mother (P25) are filed under parent (P8810) and stated on the child, child (P40) on the
    parent, and spouse (P26) is no parent property; a one-step question stops after the first step."""
    questions = _wikidata_questions()
    statements = [("b", "P22", "a"), ("a", "P40", "c"), ("c", "P40", "d"), ("e", "P26", "a"), ("d", "P25", "x")]
    descent = questions.Question("t", ("Q1", "a"), "parent_of", (("P40", False), ("P8810", True)), transitive=True)
    assert descent.properties() == {"P40": False, "P8810": True, "P22": True, "P25": True}
    assert descent.answer(statements, "a") == {"b", "c", "d"}
    assert dataclasses.replace(descent, transitive=False).answer(statements, "a") == {"b", "c"}


@pytest.mark.parametrize("engine_name", available())
@pytest.mark.parametrize("name", sorted(WIKIDATA_ANSWERS))
def test_each_wikidata_question_returns_wikidata_s_answer_given_its_hierarchy_s_rewrites(engine_name: str,
                                                                                         name: str) -> None:
    """With the oracle admitting exactly one rewrite per property holding the relation's statements, into the
    question's own predicate in that property's direction, every engine answers with Wikidata's own answer."""
    question = _wikidata_questions().QUESTIONS[name]
    kb = formats.load(WIKIDATA / f"{name}.metta")
    truth, used = question.recorded(kb)
    count, named = WIKIDATA_ANSWERS[name]
    assert len(truth) == count and named <= truth
    judge = OracleJudge.of((p, question.relation, flipped) for p, flipped in used.items())
    rewriter = Rewriter(kb, judge, make(engine_name), AllCandidates(), budget=Budget(work=50_000_000, seconds=math.inf))
    out = rewriter.ask(kb.queries[0], "rewrite-depth-tabled")
    assert out.finished, (out.status, out.detail)
    assert question.answers_of(out.answers) == truth
    assert {(p, c) for p, c, _ in out.admitted} == {(p, question.relation) for p in used}


def test_the_walkthrough_fills_exactly_the_readmes_generated_blocks() -> None:
    spec = importlib.util.spec_from_file_location("walkthrough", EXAMPLES / "walkthrough.py")
    assert spec is not None and spec.loader is not None
    walkthrough = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(walkthrough)
    assert walkthrough.DOCUMENTS == (ROOT / "README.md", EXAMPLES / "README.md")
    texts = [document.read_text() for document in walkthrough.DOCUMENTS]
    blocks: set[str] = set()
    for text in texts:
        begun = re.findall(r"<!-- BEGIN GENERATED: (\S+) -->", text)
        ended = re.findall(r"<!-- END GENERATED: (\S+) -->", text)
        assert begun == ended and len(set(begun)) == len(begun)
        blocks |= set(begun)
    assert blocks == set(walkthrough.BLOCKS)
    for argv in (a for spec in walkthrough.BLOCKS.values() if isinstance(spec, list) for a in spec):
        assert any(f"$ {walkthrough.command_line(argv)}\n" in text for text in texts), argv
