"""Purpose: measure direction-aware admission against plain admission: calibrate
each judge's slack on Wikidata dev pairs, compare the two policies on the test
rows of every pair set that has a direction to read (Levy/Holt, Wikidata and
the hard set), and compare them inside the rewriter (lazy-tabled, cold) on all four
benchmarks, writing results/direction_policy.<tag>.json (one file per run).

Run: .venv/bin/python experiments/direction_policy.py --tag nli [--judges nli-deberta-base,...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from impl_eval import aggregate, run_kb

from semrewrite import pairs
from semrewrite.admission import Directional, Threshold, balanced_accuracy, calibrate_slack
from semrewrite.benchmarks import BENCHMARKS
from semrewrite.engines import make as make_engine
from semrewrite.judges import make
from semrewrite.model import Proposal
from semrewrite.protocols import Budget, Judge
from semrewrite.retrieve import EmbeddingRetriever

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_JUDGES = "nli-deberta-base,nli-deberta-large,nli-deberta-large-wanli,reranker-qwen3,jev"
# The pair sets whose rows have a direction to read, which the policies are compared on; the typical-implication sets
# label no reverse, so there is no direction for a converse check to protect.
DIRECTION_SETS = ("levyholt", "wikidata", "hard")


def policy_metrics(admitted: list[bool], labels: list[bool]) -> dict:
    """Precision, recall and balanced accuracy of one admission policy on labelled rows."""
    tp = sum(a and y for a, y in zip(admitted, labels, strict=True))
    n_admitted, positives = sum(admitted), sum(labels)
    return {"admitted": n_admitted, "precision": round(tp / n_admitted, 4) if n_admitted else 1.0,
            "recall": round(tp / positives, 4) if positives else 1.0,
            "balanced_accuracy": round(balanced_accuracy(admitted, labels), 4)}


def beliefs(judge: Judge, proposals: list[Proposal]) -> list[float]:
    """Each proposal's score: the belief of the judge's opinion of it."""
    return [o.belief for o in judge.opinions(proposals)]


def main() -> None:
    """Run both policies for every judge named and write the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judges", default=DEFAULT_JUDGES)
    parser.add_argument("--tag", required=True, help="names the output file, so runs over different judges sit side by side")
    parser.add_argument("--engine", default="pymetta")
    args = parser.parse_args()
    rows = pairs.load(DIRECTION_SETS)
    kbs = [make_kb() for make_kb in BENCHMARKS.values()]
    retriever = EmbeddingRetriever()
    engine = make_engine(args.engine)
    budget = Budget(work=500_000, seconds=120)  # the main judge runs' budget
    out = []
    for name in args.judges.split(","):
        base = make(name)
        dev = [r for r in rows if r.set == "wikidata" and r.split == "dev"]
        props = [r.proposal for r in dev]
        forward, backward = beliefs(base, props), beliefs(base, [p.converse() for p in props])
        slack, curve = calibrate_slack(forward, backward, [r.label for r in dev], base.threshold)
        sets = {}
        for set_name in DIRECTION_SETS:
            test = [r for r in rows if r.set == set_name and r.split == "test"]
            props = [r.proposal for r in test]
            f, b = beliefs(base, props), beliefs(base, [p.converse() for p in props])
            labels = [r.label for r in test]
            sets[set_name] = {
                "plain": policy_metrics([x >= base.threshold for x in f], labels),
                "directional": policy_metrics([x >= base.threshold and x + slack >= y for x, y in zip(f, b, strict=True)], labels),
            }
        rewriter_rows, discovery = [], []
        for policy, admission in (("plain", Threshold()), ("directional", Directional(slack))):
            label = f"{name}:{policy}"
            for kb in kbs:
                kb_rows, kb_discovery = run_kb(kb, label, base, retriever, budget, ["cold"], ["lazy-tabled"], engine, admission)
                for r in kb_rows:
                    r["judge"] = label
                kb_discovery["judge"] = label
                rewriter_rows += kb_rows
                discovery.append(kb_discovery)
        out.append({"judge": name, "threshold": base.threshold, "slack": slack, "dev_curve": curve, "pair_sets": sets,
                    "rewriter": [a for a in aggregate(rewriter_rows) if a["kb"] == "ALL"], "discovery": discovery})
        print(f"  {name}: slack {slack}", flush=True)
        close = getattr(base, "close", None)
        if callable(close):
            close()
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"direction_policy.{args.tag}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
