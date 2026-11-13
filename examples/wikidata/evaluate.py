"""Purpose: answer each Wikidata question several ways and score every answer set against Wikidata's own record: read
literally, with no rewriting; and with an embedding-similarity judge and with Jev admitting rewrites, allowed one
rewrite on any path and, on the questions DEPTHS names, chains of up to three as well. Every run uses the PyMeTTa
engine with tables, and the same cascade retrieves the candidates each judge sees: the cross-encoder
nli-deberta-large ranks all 1,790 properties and the judge decides on the top K. It prints a markdown table per
question, then the rewrites each judge admitted into the question's own predicates and some of the answers each run
got wrong; summary() puts the runs allowed one rewrite on a path in one table across the questions; beliefs() ranks
each question's answers by the opinion each gets from its lineage, with Jev admitting at two rewrite thresholds and
nli-deberta-large at its own, and scores the ranking against Wikidata's answer.

Run from the repository root: python examples/wikidata/evaluate.py [NAME ...]; with --beliefs first it
prints the beliefs table and writes results/wikidata_beliefs.json, which RESULTS.md's table reads.

Wikidata's own answer to each question is computed from the knowledge base's own statements, through the properties
questions.py names and every property Wikidata files under them in its own subproperty hierarchy; the builder checked
it equals the answer Wikidata's endpoint computes. An answer only some other property reaches is counted as not in
Wikidata's answer, and some are listed. Jev's questions not yet in its answer cache need a TypeSafe key.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import sys
from dataclasses import replace
from functools import cache
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
from semrewrite.judges import default_data_trust
from semrewrite.judges import default_threshold as make_threshold
from semrewrite.judges import make as make_judge
from semrewrite.lineage import SETS
from semrewrite.protocols import Budget
from semrewrite.retrieve import JudgeRetriever
from semrewrite.rewriter import STRATEGIES, Rewriter

if TYPE_CHECKING:
    from semrewrite.discovery import Discoverer
    from semrewrite.kb import KB
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
# The judges and rewrite thresholds whose answers get opinions, None for the judge's calibrated default: Jev at the
# threshold RESULTS.md calibrated for answer F1 and at a low one that admits more true rewrites with the false, leaving
# the answer threshold to sort their answers; and nli-deberta-large, whose opinions the knowledge base's evidence is
# fused with at a data trust above 0, where Jev's is 0.
BELIEF_RUNS = (("jev", 0.65), ("jev", 0.1), ("nli-deberta-large", None))


@cache
def _tools() -> tuple[Engine, JudgeRetriever]:
    """One engine and one cascade retriever a process: every run shares the retriever, whose cache scores each
    candidate once across runs and questions, and the retriever releases its model when the process ends."""
    retriever = JudgeRetriever(make_judge("nli-deberta-large"))
    atexit.register(retriever.close)
    return make_engine("pymetta"), retriever


@cache
def _question(name: str) -> tuple[Question, KB, set[str], dict[str, bool]]:
    """The question, its knowledge base, Wikidata's own answer, and the predicates that answer steps along."""
    question = QUESTIONS[name]
    kb = formats.load(HERE / f"{name}.metta")
    truth, used = question.recorded(kb)
    return question, kb, truth, used


@cache
def _literal(name: str) -> Outcome:
    """The question read literally, with no rewriting: one solve, made once a process."""
    _, kb, _, _ = _question(name)
    engine, _ = _tools()
    budget = Budget(work=WORK, seconds=math.inf)
    return Rewriter(kb, make_judge("oracle", gold=frozenset()), engine, None, budget=budget).ask(kb.queries[0], "exact")


# Time: a cascade ranking per expansion and one solve.
@cache
def _judged(name: str, judge_name: str, depth: int) -> tuple[Outcome, Discoverer]:
    """The question answered with `judge_name` at its threshold in JUDGES and at most `depth` rewrites on a path, made
    once a process, with its discoverer, whose memo holds the scores of the rewrites it admitted."""
    _, kb, _, _ = _question(name)
    engine, retriever = _tools()
    budget = Budget(work=WORK, seconds=math.inf)
    judge = make_judge(judge_name, threshold=dict(JUDGES)[judge_name])
    rewriter = Rewriter(kb, judge, engine, retriever, k=K, budget=budget)
    disc = rewriter.discoverer()
    out = rewriter.ask(kb.queries[0], replace(STRATEGY, depth=depth), disc)
    close = getattr(judge, "close", None)
    if callable(close):
        close()
    return out, disc


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


def evaluate(name: str) -> str:
    """One question's table and the rewrites each run admitted, as markdown."""
    question, kb, truth, used = _question(name)
    exact = _literal(name)
    catalogue = sum(property_id(p) is not None for p in kb.catalogue())
    held = len({f.pred for f in kb.facts})
    rows = [f"### {name}: `{kb.queries[0]}`", "",
            f"The knowledge base holds {len(kb.facts):,} statements, in {held} of the {catalogue:,} properties it glosses.",
            f"Wikidata's own answer, through {', '.join(f'`{p}`' for p in used)}: {len(truth):,} entities.", "",
            ("| judge | most rewrites on a path | answers | in Wikidata's answer | not in it | missed | judge questions "
             "| rewrites admitted | seconds |"),
            "|---|---|---|---|---|---|---|---|---|",
            _row("none, read literally", "none", question, truth, exact)]
    notes: list[str] = []
    shown: dict[str, frozenset] = {}
    for depth in DEPTHS.get(name, (1,)):
        for judge_name, _ in JUDGES:
            out, disc = _judged(name, judge_name, depth)
            rows.append(_row(judge_name, str(depth), question, truth, out))
            notes += _notes(f"{judge_name}, at most {depth} rewrite{'s' if depth > 1 else ''} on a path", question,
                            truth, out, disc, shown)
    return "\n".join(rows + notes)


def report(names: list[str]) -> str:
    """The evaluation of each named question, every question when none is named, as markdown."""
    return "\n\n".join(evaluate(name) for name in names or list(QUESTIONS))


def _cell(question: Question, truth: set[str], out: Outcome) -> str:
    """A run's answers as one table cell: how many, and how many of them Wikidata's answer holds."""
    if not out.finished:
        return f"none, {STOPPED[out.status]}"
    got = question.answers_of(out.answers)
    if not got:
        return "0"
    more = len(got - truth)
    if not more:
        return f"{len(got):,}, all in Wikidata's answer"
    return f"{len(got):,}: {len(got & truth):,} in Wikidata's answer, {more:,} not"


def summary(names: list[str]) -> str:
    """One markdown table of each named question, every question when none is named: Wikidata's own answer, then the
    literal read and each judge allowed one rewrite on a path."""
    head = ["question", "Wikidata's answer", "read literally", *(f"judged by {judge}" for judge, _ in JUDGES)]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name in names or list(QUESTIONS):
        question, kb, truth, _ = _question(name)
        cells = [f"{name}: `{kb.queries[0]}`", f"{len(truth):,}", _cell(question, truth, _literal(name)),
                 *(_cell(question, truth, _judged(name, judge, 1)[0]) for judge, _ in JUDGES)]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


# Time: one cascade ranking and one solve, then the lineage of every answer; the second rewriter shares the first's
# discoverer, so only the lineage runs again.
@cache
def _believed(name: str, judge_name: str, threshold: float | None) -> tuple[Outcome, Outcome]:
    """The question answered with `judge_name` admitting at `threshold`, its calibrated default when None, one rewrite
    on a path, each answer with its opinion: from the judge's opinions alone, and fused with the knowledge base's
    evidence at the judge's calibrated data trust."""
    _, kb, _, _ = _question(name)
    engine, retriever = _tools()
    budget = Budget(work=WORK, seconds=math.inf)
    judge = make_judge(judge_name) if threshold is None else make_judge(judge_name, threshold=threshold)
    alone = Rewriter(kb, judge, engine, retriever, k=K, budget=budget, beliefs=SETS)
    disc = alone.discoverer()
    strategy = replace(STRATEGY, depth=1)
    first = alone.ask(kb.queries[0], strategy, disc)
    fused = Rewriter(kb, alone.scorer, engine, retriever, k=K, budget=budget, beliefs=SETS,
                     data_trust=default_data_trust(judge_name))
    second = fused.ask(kb.queries[0], strategy, disc)
    judge.close()
    return first, second


def ranking(question: Question, truth: set[str], out: Outcome) -> dict[str, float | None]:
    """How well a run's answers ranked by projected probability sort Wikidata's answer from the rest: average
    precision against Wikidata's answer, the best F1 over answer thresholds with the threshold reaching it, the F1 of
    every answer, and the lowest projection inside Wikidata's answer and highest outside it."""
    if not out.finished or out.beliefs is None:
        msg = f"{question.name}: {out.status} {out.detail}"
        raise RuntimeError(msg)
    scored = {question.answers_of([row]).pop(): o.projection for row, o in out.beliefs.items()}
    ranked = sorted(scored.items(), key=lambda es: (-es[1], es[0]))
    hits, precision_sum, best = 0, 0.0, (0.0, None)
    for k, (entity, p) in enumerate(ranked, start=1):
        if entity in truth:
            hits += 1
            precision_sum += hits / k
        # F1 is read only where the next answer's projection differs, since a threshold keeps tied answers together.
        if k == len(ranked) or ranked[k][1] != p:
            f1 = 2 * hits / (k + len(truth))
            if f1 > best[0]:
                best = (f1, p)
    inside = [p for e, p in ranked if e in truth]
    outside = [p for e, p in ranked if e not in truth]
    return {"ap": precision_sum / len(truth), "best_f1": best[0], "at": best[1],
            "f1": 2 * len(set(scored) & truth) / (len(scored) + len(truth)),
            "lowest_inside": min(inside, default=None), "highest_outside": max(outside, default=None)}


def belief_rows(names: list[str]) -> list[dict[str, object]]:
    """Each named question's runs, every question's when none is named, for each judge and threshold BELIEF_RUNS
    names: its answers, how many Wikidata's answer holds, and how well their projected probabilities rank them, from
    the judge's opinions alone and with the knowledge base's evidence fused in at its data trust."""
    rows: list[dict[str, object]] = []
    for name in names or list(QUESTIONS):
        question, _, truth, _ = _question(name)
        for judge_name, chosen in BELIEF_RUNS:
            alone, fused = _believed(name, judge_name, chosen)
            got = question.answers_of(alone.answers)
            threshold = make_threshold(judge_name) if chosen is None else chosen
            rows.append({"question": name, "judge": judge_name, "threshold": threshold,
                         "data_trust": default_data_trust(judge_name), "answers": len(got), "outside": len(got - truth),
                         "truth": len(truth), "rewrites": len(alone.admitted),
                         "alone": ranking(question, truth, alone), "fused": ranking(question, truth, fused)})
    return rows


def beliefs(rows: list[dict[str, object]]) -> str:
    """belief_rows' runs as one markdown table."""
    def number(x: object) -> str:
        return "none" if x is None else f"{x:.3f}"

    head = ["question", "judge", "rewrite threshold", "answers", "not in Wikidata's", "F1 of all answers",
            "average precision by P", "best F1 by P (at P)", "lowest P inside", "highest P outside",
            "average precision with the knowledge base's evidence (data trust)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        m, f = r["alone"], r["fused"]
        assert isinstance(m, dict) and isinstance(f, dict)
        lines.append("| " + " | ".join([
            str(r["question"]), str(r["judge"]), f"{r['threshold']:.3g}", f"{r['answers']:,}", f"{r['outside']:,}",
            number(m["f1"]), number(m["ap"]), f"{number(m['best_f1'])} ({number(m['at'])})",
            number(m["lowest_inside"]), number(m["highest_outside"]), f"{number(f['ap'])} ({r['data_trust']:.2f})"])
            + " |")
    return "\n".join(lines)


def placed(name: str, threshold: float, shown: int = 5) -> str:
    """Where the answers outside Wikidata's land among the answers of Jev admitting at `threshold`, ranked by projected
    probability: the `shown` highest outside Wikidata's answer and the `shown` lowest inside it, with each one's rank
    and opinion, as a markdown table."""
    question, _, truth, _ = _question(name)
    out, _ = _believed(name, "jev", threshold)
    if not out.finished or out.beliefs is None:
        msg = f"{name}: {out.status} {out.detail}"
        raise RuntimeError(msg)
    opinions = {question.answers_of([row]).pop(): o for row, o in out.beliefs.items()}
    ranked = sorted(opinions, key=lambda e: (-opinions[e].projection, e))
    rank = {e: i for i, e in enumerate(ranked, start=1)}
    chosen = [e for e in ranked if e not in truth][:shown] + [e for e in ranked if e in truth][-shown:]
    rows = ["| answer | rank | in Wikidata's answer | P | b | d | u |", "|---|---|---|---|---|---|---|"]
    for e in sorted(chosen, key=rank.__getitem__):
        o = opinions[e]
        rows.append(f"| `{e}` | {rank[e]:,} of {len(ranked):,} | {'yes' if e in truth else 'no'} | {o.projection:.3f} "
                    f"| {o.belief:.3f} | {o.disbelief:.3f} | {o.uncertainty:.3f} |")
    return "\n".join(rows)


# The answer-belief runs' measurements, which RESULTS.md's table reads.
BELIEFS_JSON = ROOT / "results" / "wikidata_beliefs.json"


if __name__ == "__main__":
    if sys.argv[1:2] == ["--beliefs"]:
        measured = belief_rows(sys.argv[2:])
        BELIEFS_JSON.write_text(json.dumps(measured, indent=1) + "\n")
        print(beliefs(measured))
    else:
        print(report(sys.argv[1:]))
