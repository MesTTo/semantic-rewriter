"""Purpose: measure direction-aware admission against plain admission: calibrate
each judge's slack on Wikidata dev pairs, compare the two policies on every test
pair set, and compare them inside the rewriter (lazy-tabled, cold) on all four
benchmarks, writing results/direction_policy.<tag>.json (one file per run).

Run: .venv/bin/python experiments/direction_policy.py --tag nli [--judges nli-deberta-base,...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semrewrite import pairs
from semrewrite.benchmarks import BENCHMARKS
from semrewrite.directional import DirectionalJudge, balanced_accuracy, calibrate_slack, reverse
from semrewrite.engine import Budget
from semrewrite.judges import make
from semrewrite.retrieve import Retriever

from impl_eval import aggregate, run_kb

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_JUDGES = "nli-deberta-base,nli-deberta-large,nli-deberta-large-wanli,reranker-qwen3,jev"


def policy_metrics(admitted: list[bool], labels: list[bool]) -> dict:
    """Precision, recall and balanced accuracy of one admission policy on labelled rows."""
    tp = sum(a and y for a, y in zip(admitted, labels, strict=True))
    n_admitted, positives = sum(admitted), sum(labels)
    return {"admitted": n_admitted, "precision": round(tp / n_admitted, 4) if n_admitted else 1.0,
            "recall": round(tp / positives, 4) if positives else 1.0,
            "balanced_accuracy": round(balanced_accuracy(admitted, labels), 4)}


def main() -> None:
    """Run both policies for every judge named and write the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judges", default=DEFAULT_JUDGES)
    parser.add_argument("--tag", required=True, help="names the output file, so runs over different judges sit side by side")
    args = parser.parse_args()
    rows = pairs.load()
    kbs = [make_kb() for make_kb in BENCHMARKS.values()]
    retriever = Retriever()
    budget = Budget(inferences=500_000, seconds=120)  # the main judge runs' budget
    out = []
    for name in args.judges.split(","):
        base = make(name)
        dev = [r for r in rows if r.set == "wikidata" and r.split == "dev"]
        props = [r.proposal for r in dev]
        forward, backward = base.score(props), base.score([reverse(p) for p in props])
        slack, curve = calibrate_slack(forward, backward, [r.label for r in dev], base.threshold)
        sets = {}
        for set_name in ("levyholt", "wikidata", "hard"):
            test = [r for r in rows if r.set == set_name and r.split == "test"]
            props = [r.proposal for r in test]
            f, b = base.score(props), base.score([reverse(p) for p in props])
            labels = [r.label for r in test]
            sets[set_name] = {
                "plain": policy_metrics([x >= base.threshold for x in f], labels),
                "directional": policy_metrics([x >= base.threshold and x + slack >= y for x, y in zip(f, b, strict=True)], labels),
            }
        rewriter_rows, discovery = [], []
        for policy, judge in (("plain", base), ("directional", DirectionalJudge(base, slack))):
            label = f"{name}:{policy}"
            for kb in kbs:
                kb_rows, kb_discovery = run_kb(kb, label, judge, retriever, budget, ["cold"], ["lazy-tabled"])
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
