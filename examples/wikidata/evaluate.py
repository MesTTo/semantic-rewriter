"""Purpose: answer each Wikidata question several ways and score every answer set against Wikidata's own record: read
literally, with no rewriting; and with an embedding-similarity judge and with Jev admitting rewrites, allowed one
rewrite on any path and, on the questions DEPTHS names, chains of up to three as well. Every run uses the PyMeTTa
engine with tables, and the same cascade retrieves the candidates each judge sees: the cross-encoder
nli-deberta-large ranks all 1,790 properties and the judge decides on the top K. It prints a markdown table per
question, then the rewrites each judge admitted into the question's own predicates and some of the answers each run
got wrong.

Run from the repository root: python examples/wikidata/evaluate.py [NAME ...]

Wikidata's own answer to each question is computed from the knowledge base's own statements, through the properties
questions.py names and every property Wikidata files under them in its own subproperty hierarchy; the builder checked
it equals the answer Wikidata's endpoint computes. An answer only some other property reaches is counted as not in
Wikidata's answer, and some are listed. Jev's questions not yet in its answer cache need a TypeSafe key.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]

from questions import QUESTIONS, Question, property_id

from semrewrite import formats
from semrewrite.engines import make as make_engine
from semrewrite.judges import make as make_judge
from semrewrite.protocols import Budget
from semrewrite.retrieve import JudgeRetriever
from semrewrite.rewriter import STRATEGIES, Rewriter

if TYPE_CHECKING:
    from semrewrite.discovery import Discoverer
    from semrewrite.protocols import Engine
    from semrewrite.rewriter import Outcome

STRATEGY = STRATEGIES["rewrite-depth-tabled"]
# Candidates each expansion shows the admitting judge, from the cascade's ranking of every property.
K = 50
# Each judge with its threshold, None for its calibrated default; Jev's is the one RESULTS.md calibrated for answer F1.
JUDGES = (("cosine-bge", None), ("jev", 0.65))
# The most rewrites any path may take, per question. Chains of three are the bound RESULTS.md calibrated on the
# benchmarks; over 1,790 properties they cost Jev about 25 times the questions one rewrite does, so one question runs
# them, enough to show what chaining does here.
DEPTHS = {"victoria": (1, 3)}
# Enough for every finishing run here; a run admitting rewrites that join everyone to everyone spends it instead.
WORK = 50_000_000
STOPPED = {"work_limit": "the work budget stopped it", "time_limit": "the time budget stopped it"}


def _row(label: str, depth: str, question: Question, truth: set[str], out: Outcome) -> str:
    """One run's table row; a run a budget stopped returns no answers, so it misses all of Wikidata's."""
    if out.status == "error":
        msg = f"{label}: {out.detail}"
        raise RuntimeError(msg)
    got = question.answers_of(out.answers)
    answers = f"{len(got):,}" if out.finished else f"none, {STOPPED[out.status]}"
    return (f"| {label} | {depth} | {answers} | {len(got & truth):,} | {len(got - truth):,} | {len(truth - got):,} | "
            f"{out.proposals:,} | {len(out.admitted)} | {out.seconds:.1f} |")


def _notes(label: str, question: Question, truth: set[str], out: Outcome, disc: Discoverer,
           shown: dict[str, frozenset]) -> list[str]:
    """What a run admitted into the question's own predicates, unless the same judge's earlier run admitted exactly
    that, how many rewrites it chained beyond them, and some answers it got wrong."""
    judge = label.split(",")[0]
    scores = {r.key: r.score for c in {c for _, c, _ in out.admitted} for r in disc.rewrites(c)}
    own = frozenset(k for k in out.admitted if property_id(k[1]) is None)
    lines = ["", f"{label}: {out.status}, {out.work:,} inferences."]
    if shown.get(judge) == own:
        lines.append("Into the question's own predicates it admitted the same rewrites as above.")
    else:
        shown[judge] = own
        lines += ["Rewrites it admitted into the question's own predicates:", ""]
        lines += [f"- `{c}` ⇐ `{p}` {'' if m == (0, 1) else '(arguments swapped) '}{scores.get((p, c, m), 0):.2f}"
                  for p, c, m in sorted(own, key=lambda k: (k[1], -scores.get(k, 0), k[0]))]
    if len(out.admitted) > len(own):
        lines.append(f"\nIt chained {len(out.admitted) - len(own)} more rewrites into the properties those premises name.")
    got = question.answers_of(out.answers)
    if got - truth:
        lines.append(f"\nSome answers not in Wikidata's: {', '.join(f'`{e}`' for e in sorted(got - truth)[:8])}")
    if out.finished and truth - got:
        lines.append(f"\nSome of Wikidata's answers it missed: {', '.join(f'`{e}`' for e in sorted(truth - got)[:8])}")
    return lines


# Time: one literal read and a judged run per judge and depth, each a cascade ranking per expansion and one solve.
def evaluate(name: str, retriever: JudgeRetriever, engine: Engine) -> str:
    """One question's table and the rewrites each run admitted, as markdown; every run shares the retriever, whose
    cache scores each candidate once across runs and questions."""
    question = QUESTIONS[name]
    kb = formats.load(HERE / f"{name}.metta")
    query = kb.queries[0]
    truth, used = question.recorded(kb)
    budget = Budget(work=WORK, seconds=math.inf)
    exact = Rewriter(kb, make_judge("oracle", gold=frozenset()), engine, None, budget=budget).ask(query, "exact")
    catalogue = sum(property_id(p) is not None for p in kb.catalogue())
    held = len({f.pred for f in kb.facts})
    rows = [f"### {name}: `{query}`", "",
            f"The knowledge base holds {len(kb.facts):,} statements, in {held} of the {catalogue:,} properties it glosses.",
            f"Wikidata's own answer, through {', '.join(f'`{p}`' for p in used)}: {len(truth):,} entities.", "",
            ("| judge | most rewrites on a path | answers | in Wikidata's answer | not in it | missed | judge questions "
             "| rewrites admitted | seconds |"),
            "|---|---|---|---|---|---|---|---|---|",
            _row("none, read literally", "none", question, truth, exact)]
    notes: list[str] = []
    shown: dict[str, frozenset] = {}
    for depth in DEPTHS.get(name, (1,)):
        for judge_name, threshold in JUDGES:
            judge = make_judge(judge_name, threshold=threshold)
            rewriter = Rewriter(kb, judge, engine, retriever, k=K, budget=budget)
            disc = rewriter.discoverer()
            out = rewriter.ask(query, replace(STRATEGY, depth=depth), disc)
            rows.append(_row(judge_name, str(depth), question, truth, out))
            notes += _notes(f"{judge_name}, at most {depth} rewrite{'s' if depth > 1 else ''} on a path", question,
                            truth, out, disc, shown)
            close = getattr(judge, "close", None)
            if callable(close):
                close()
    return "\n".join(rows + notes)


def report(names: list[str]) -> str:
    """The evaluation of each named question, every question when none is named, as markdown."""
    engine = make_engine("pymetta")
    with JudgeRetriever(make_judge("nli-deberta-large")) as retriever:
        return "\n\n".join(evaluate(name, retriever, engine) for name in names or list(QUESTIONS))


if __name__ == "__main__":
    print(report(sys.argv[1:]))
