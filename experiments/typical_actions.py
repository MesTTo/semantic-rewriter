"""Purpose: measure the actions example, examples/actions.kb, under a judge asked what is entailed and a judge
asked what typically follows: every action's opinion that its doer wastes time, as the rewriter weighs it at the
judge's calibrated trust, base rate and data trust; the ROC AUC of those opinions' probabilities against the
example's gold lines; and the answers `semrewrite ask examples/actions.kb 'wastes_time(X)' --retriever all
--beliefs` returns at the judge's calibrated threshold, with the rewrites it admitted and the average precision of
the answers ranked by P against the gold lines' answers, written to results/typical_actions.json.

    .venv/bin/python experiments/typical_actions.py [--judges nli-deberta-large,typical-qwen3-4b]

The judges load one at a time on the GPU.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semrewrite import formats
from semrewrite.engines import make as make_engine
from semrewrite.judges import default_data_trust, make
from semrewrite.judges.metrics import roc_auc
from semrewrite.lineage import SETS
from semrewrite.model import parse_atom
from semrewrite.retrieve import AllCandidates
from semrewrite.rewriter import Rewriter

KB = ROOT / "examples" / "actions.kb"
QUERY = "wastes_time(X)"
RESULTS = ROOT / "results" / "typical_actions.json"
JUDGES = ("nli-deberta-large", "typical-qwen3-4b")
# What a run reads from the repository: the package the rewriter and the judges come from, calibration.json
# included, the example and this script.
INPUTS = ("semrewrite", "examples/actions.kb", "experiments/typical_actions.py")


def average_precision(ranked: Sequence[str], relevant: set[str]) -> float | None:
    """The mean, over the relevant items, of the precision of the ranking down to each one; None when none is relevant.

    >>> average_precision(["kim", "eve", "raj"], {"kim", "raj"})
    0.8333333333333333
    """
    hits, total = 0, 0.0
    for rank, item in enumerate(ranked, 1):
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant) if relevant else None


# Time: one judge question per action, then one lazy-tabled ask over 8 facts, whose discovery asks about every
# action as a premise of each predicate it reaches, at most 8 x 9 questions, the first 8 answered from the cache.
def measure(name: str) -> dict[str, Any]:
    """One judge on the actions example: each action's opinion that its doer wastes time, as the rewriter weighs it,
    their ROC AUC against the gold lines, and the answers with their opinions at the judge's calibrated threshold,
    the rewrites admitted on the way and the answers' average precision by P."""
    kb = formats.load(KB)
    goal = parse_atom(QUERY).pred
    doer = {f.pred: f.args[0] for f in kb.facts}
    judge = make(name)
    try:
        rw = Rewriter(kb, judge, make_engine("python"), AllCandidates(), beliefs=SETS,
                      data_trust=default_data_trust(name))
        keys = [(action, goal, (0,)) for action in sorted(doer)]
        rewrites, opinion = rw.rewrite_opinions(keys)
        rows: list[dict[str, Any]] = []
        for key, rewrite in zip(keys, rewrites, strict=True):
            o = opinion[key]
            rows.append({"action": key[0], "sentence": rw.catalogue[key[0]].sentence("A"), "doer": doer[key[0]],
                         "gold": key in kb.gold, "belief": o.belief, "disbelief": o.disbelief,
                         "uncertainty": o.uncertainty, "projection": o.projection, "score": rewrite.score,
                         "admitted": rewrite.score >= judge.threshold})
        out = rw.ask(QUERY)
        beliefs = out.beliefs or {}
        answers = sorted(out.answers, key=lambda a: (-beliefs[a].projection, a))
        admitted, weighed = rw.rewrite_opinions(out.admitted)
        gold_answers = {doer[p] for p, _, _ in kb.gold}
        return {
            "threshold": judge.threshold, "trust": judge.trust, "base_rate": judge.base_rate,
            "data_trust": rw.data_trust,
            "auc": roc_auc([r["projection"] for r in rows], [r["gold"] for r in rows]),
            "actions": sorted(rows, key=lambda r: (-r["projection"], r["action"])),
            "status": out.status,
            "answers": [{"answer": a[0], "projection": beliefs[a].projection, "belief": beliefs[a].belief,
                         "disbelief": beliefs[a].disbelief, "uncertainty": beliefs[a].uncertainty} for a in answers],
            "gold_answers": sorted(gold_answers),
            "average_precision": average_precision([a[0] for a in answers], gold_answers),
            "rewrites": [{"premise": r.premise, "conclusion": r.conclusion, "score": r.score,
                          "projection": weighed[r.key].projection}
                         for r in sorted(admitted, key=lambda r: (-weighed[r.key].projection, r.premise, r.conclusion))],
        }
    finally:
        judge.close()


def provenance() -> dict[str, Any]:
    """What a run is computed from, taken before the judges load: the code revision and the inputs that differ
    from it."""
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", *INPUTS],
                           capture_output=True, text=True, check=False)
    return {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": "python experiments/typical_actions.py " + " ".join(sys.argv[1:]),
            "commit": head.stdout.strip() or None,
            "inputs_uncommitted": sorted(line[3:] for line in dirty.stdout.splitlines())}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--judges", default=",".join(JUDGES), help="comma-separated judge names")
    args = parser.parse_args(argv)
    results = {
        "provenance": provenance(),
        "kb": str(KB.relative_to(ROOT)), "query": QUERY,
        "judges": {name: measure(name) for name in args.judges.split(",") if name},
    }
    RESULTS.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
