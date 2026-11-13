"""Purpose: render the experiment's results files into the tables RESULTS.md
shows, replacing each block between `<!-- BEGIN GENERATED: name -->` and
`<!-- END GENERATED: name -->`, so every number in the report's tables comes
from a file under results/ by one command.

Run: .venv/bin/python experiments/report.py
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from impl_eval import aggregate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "RESULTS.md"

STRATEGY_ORDER = ["exact", "naive", "memo-untabled", "nomemo-tabled", "depth-bounded", "depth-tabled",
                  "rewrite-depth-tabled", "lazy-tabled", "rounds", "eager-worklist", "alias-canonical", "saturate"]


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.is_file():
        msg = f"{path} is missing; run the experiment that writes it first"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text())


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def _load_judges() -> dict:
    """The real-judge runs, written as one file pair per process (implementations.judges-*), merged.

    Aggregates are recomputed from the saved rows with the runner's own
    aggregate(), so every table reads one definition of each metric whichever
    version of the runner wrote the summary.
    """
    paths = sorted(RESULTS.glob("implementations.judges-*.json"))
    if not paths:
        msg = f"no implementations.judges-*.json under {RESULTS}; run experiments/impl_eval.py with real judges first"
        raise FileNotFoundError(msg)
    merged: dict = {"aggregates": [], "discovery": [], "sweep": []}
    rows: list[dict] = []
    for path in paths:
        part = json.loads(path.read_text())
        merged["discovery"] += part["discovery"]
        merged["sweep"] += part.get("sweep", [])
        rows += [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
    merged["aggregates"] = aggregate(rows)
    return merged


def _load_oracle() -> dict:
    """The oracle run's summary with its aggregates recomputed from its saved rows."""
    summary = _load("implementations.oracle.json")
    rows = [json.loads(line) for line in (RESULTS / "implementations.oracle.jsonl").read_text().splitlines()]
    summary["aggregates"] = aggregate(rows)
    return summary


def _aggregates(summary: dict, judge: str, kb: str = "ALL") -> dict[tuple[str, str], dict]:
    return {(a["strategy"], a["mode"]): a for a in summary["aggregates"] if a["judge"] == judge and a["kb"] == kb}


def strategies_table() -> str:
    """Every strategy with the oracle judge over all four benchmarks, cold and warm.

    Wall time is left out: per-query seconds grew over the life of the run's
    process, so they measure when a strategy ran more than what it costs;
    engine work is read from counted inferences instead.
    """
    agg = _aggregates(_load_oracle(), "oracle")
    rows = []
    for s in STRATEGY_ORDER:
        cold, warm = agg.get((s, "cold")), agg.get((s, "warm"))
        if cold is None:
            continue
        rows.append([
            s, cold["precision"], cold["recall"], cold["finished_rate"], cold["proposals_per_query"],
            warm["proposals_per_query"] if warm else "n/a", cold["expansions_per_query"],
            int(cold["work_per_query"]), cold["tables_per_query"],
        ])
    return _table(["strategy", "precision", "recall", "finished", "proposals/query cold", "proposals/query warm",
                   "expansions/query", "inferences/query", "tables/query"], rows)


def divergence_table() -> str:
    """The untabled strategies' work at rising budgets on mother(X, Y), against lazy-tabled."""
    rows = [[d["strategy"], d["budget"], d["status"], d["work"], d["expansions"], d["proposals"], d["answers"], d["seconds"]]
            for d in _load("implementations.oracle.json")["divergence"]]
    return _table(["strategy", "budget (inferences)", "status", "inferences", "expansions", "proposals", "answers", "seconds"], rows)


def isolation_table() -> str:
    """Whether runs sharing one process agree with the same runs in fresh processes."""
    iso = _load("implementations.oracle.json")["isolation"]
    return _table(["knowledge base", "runs compared", "mismatches"], [[iso["kb"], iso["checked"], len(iso["mismatches"])]])


def retrieval_table() -> str:
    """Retrieval recall of gold-closure implications at k."""
    rows = [[r["kb"], r["k"], r["closure_pairs"], r["retrieved"], r["recall"]] for r in _load("implementations.oracle.json")["retrieval_recall"]]
    return _table(["knowledge base", "k", "gold implications", "premise retrieved", "recall"], rows)


def judges_in_kb_table() -> str:
    """Each real judge inside the rewriter: rewrite precision over the catalogue and lazy-tabled answer quality."""
    summary = _load_judges()
    by_judge: dict[str, list[dict]] = {}
    for d in summary["discovery"]:
        by_judge.setdefault(d["judge"], []).append(d)
    rows = []
    for judge, ds in by_judge.items():
        admitted = sum(d["admitted"] for d in ds)
        kinds: dict[str, int] = {}
        for d in ds:
            for k, v in d["kinds"].items():
                kinds[k] = kinds.get(k, 0) + v
        true = kinds.get("true", 0)
        lazy = _aggregates(summary, judge).get(("lazy-tabled", "cold"), {})
        sweep = next((s for s in summary["sweep"] if s["judge"] == judge and s["strategy"] == "lazy-tabled"), {})
        rows.append([
            judge, admitted, true / admitted if admitted else 1.0, kinds.get("reversed", 0), kinds.get("swap", 0),
            kinds.get("trap", 0), kinds.get("unrelated", 0), lazy.get("finished_rate", "n/a"), lazy.get("f1", "n/a"),
            sweep.get("f1_held_out_pooled", "n/a"),
        ])
    return _table(["judge", "rewrites admitted", "rewrite precision", "reversed", "wrong argument order", "trap",
                   "unrelated", "finished (default threshold)", "answer F1 (default threshold)",
                   "answer F1 (held-out threshold)"], rows)


def judges_strategies_table() -> str:
    """Answer precision and recall of the main strategies under each real judge."""
    summary = _load_judges()
    judges = sorted({a["judge"] for a in summary["aggregates"]})
    rows = []
    for judge in judges:
        agg = _aggregates(summary, judge)
        for s in STRATEGY_ORDER:
            a = agg.get((s, "cold"))
            if a:
                rows.append([judge, s, a["finished_rate"], a["precision"], a["recall"], a["f1"],
                             a["precision_finished"], a["recall_finished"], a["proposals_per_query"]])
    return _table(["judge", "strategy", "finished", "precision", "recall", "F1", "precision (finished)",
                   "recall (finished)", "proposals/query"], rows)


def pairs_table() -> str:
    """The direction judges on the labelled pair sets, from the judge thread's results/judges.json."""
    judges = _load("judges.json")["judges"]
    rows = []
    for name, j in judges.items():
        variant = j["variants"]["letters"]
        for pair_set in ("levyholt", "wikidata", "hard"):
            test = variant["sets"][pair_set]["test"]
            at = test["at_default"]
            rows.append([name, pair_set, test["rows"], test["direction"]["pairs"], test["direction"]["accuracy"],
                         test["auc"], at["threshold"], at["precision"], at["recall"], at["balanced_accuracy"]])
    return _table(["judge", "pair set", "rows", "one-way pairs", "direction accuracy", "ROC AUC", "threshold",
                   "precision", "recall", "balanced accuracy"], rows)


def trust_table() -> str:
    """Each judge's trust and data trust as judge_eval.py fitted them on Wikidata dev pairs, with Wikidata test pairs'
    ROC AUC and the log loss and Brier score of its trusted opinions, and on the test pairs the knowledge base has
    evidence on, its log loss alone and fused with that evidence."""
    judges = _load("judges.json")["judges"]
    rows = []
    for name, j in sorted(judges.items(), key=lambda nj: -nj[1]["variants"]["letters"]["trust"]["default"]):
        variant = j["variants"]["letters"]
        test = variant["sets"]["wikidata"]["test"]
        data = variant["data"]
        rows.append([name, variant["trust"]["default"], test["auc"], test["at_default_trust"]["log_loss"],
                     test["at_default_trust"]["brier"], data["trust"], data["test"]["judge"]["rows"],
                     data["test"]["judge"]["log_loss"], data["test"]["fused"]["log_loss"]])
    return _table(["judge", "trust", "test ROC AUC", "test log loss", "test Brier", "data trust", "test pairs with data",
                   "their log loss, judge alone", "fused with the data"], rows)


def beliefs_table() -> str:
    """The Wikidata questions' answers ranked by projected probability, from examples/wikidata/evaluate.py --beliefs,
    rendered by that script's own table."""
    import importlib.util

    rows = _load("wikidata_beliefs.json")
    spec = importlib.util.spec_from_file_location("wikidata_evaluate", ROOT / "examples" / "wikidata" / "evaluate.py")
    assert spec is not None and spec.loader is not None
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    return evaluate.beliefs(rows)


def direction_policy_table() -> str:
    """Plain against direction-aware admission per judge: the calibrated slack, Wikidata test pairs, and lazy-tabled answers."""
    rows = []
    for path in sorted(RESULTS.glob("direction_policy.*.json")):
        for entry in json.loads(path.read_text()):
            wd = entry["pair_sets"]["wikidata"]
            rewriter = {a["judge"].split(":")[1]: a for a in entry["rewriter"]}
            for policy in ("plain", "directional"):
                a = rewriter[policy]
                rows.append([entry["judge"], policy, entry["slack"] if policy == "directional" else "",
                             wd[policy]["precision"], wd[policy]["recall"], wd[policy]["balanced_accuracy"],
                             a["precision"], a["recall"], a["f1"], a["false_rewrite_rate"]])
    return _table(["judge", "policy", "slack", "Wikidata precision", "Wikidata recall", "Wikidata balanced accuracy",
                   "answer precision", "answer recall", "answer F1", "false rewrites"], rows)


def sweep_table() -> str:
    """Each swept strategy's answer F1 at the judge's pair-calibrated default against a threshold calibrated on the other benchmarks."""
    rows = []
    for s in _load_judges()["sweep"]:
        folds = ", ".join(f"{h['kb']} {h['threshold']:.3f}" for h in s["held_out"])
        best = max(s["curve"], key=lambda c: c["f1"])
        rows.append([s["judge"], s["strategy"], s["default_threshold"], s["f1_default_pooled"], folds,
                     s["f1_held_out_pooled"], s["precision_held_out_pooled"], s["recall_held_out_pooled"],
                     s["finished_held_out_pooled"], best["threshold"], best["f1"]])
    return _table(["judge", "strategy", "default threshold", "F1 at default", "held-out threshold per benchmark",
                   "F1 held-out", "precision held-out", "recall held-out", "finished held-out",
                   "best threshold in hindsight", "F1 in hindsight"], rows)


def blowup_table() -> str:
    """A permissive judge's closure against the oracle's on the same hard queries, with no budget in the way."""
    rows = [[r["query"], r["judge"], r["strategy"], r["status"], r["work"], r["tables"], r["rewrites"],
             f"{r['true']}/{r['answers']}", r["gold"]] for r in _load("blowup.json")]
    return _table(["query", "judge", "strategy", "status", "inferences", "tables", "rewrites", "true/answers", "gold"], rows)


NAMED_CASES = (
    "hard:mother>parent", "hard:mother>parent:reverse", "hard:legal_guardian>parent", "hard:stepmother>parent",
    "hard:mother_in_law>mother", "hard:employs>manages", "hard:employs>manages:reverse",
    "hard:causes>correlated_with", "hard:causes>prevents",
)


def named_cases_table() -> str:
    """Each judge's score on the cases the design discussion named; a star marks a score at or over the judge's threshold."""
    data = _load("judges.json")
    names = list(data["judges"])
    thresholds = {n: data["judges"][n]["variants"]["letters"]["thresholds"]["default"] for n in names}
    rows_by_id = {r["id"]: r for r in data["hard_rows"]}
    rows = []
    for case in NAMED_CASES:
        r = rows_by_id[case]
        cells = []
        for n in names:
            s = r["scores"][n]["letters"]
            cells.append(f"{s:.2f}{'*' if s >= thresholds[n] else ''}")
        rows.append([f"{r['premise']} ⇒ {r['conclusion']}", "yes" if r["label"] else "no", *cells])
    return _table(["implication", "holds", *names], rows)


GENERATORS: dict[str, Callable[[], str]] = {
    "named-cases": named_cases_table,
    "direction-policy": direction_policy_table,
    "blowup": blowup_table,
    "sweep": sweep_table,
    "strategies": strategies_table,
    "divergence": divergence_table,
    "isolation": isolation_table,
    "retrieval": retrieval_table,
    "judges-in-kb": judges_in_kb_table,
    "judges-strategies": judges_strategies_table,
    "pairs": pairs_table,
    "trust": trust_table,
    "wikidata-beliefs": beliefs_table,
}


def main() -> None:
    """Replace every generated block in RESULTS.md that a generator names.

    A table whose results file is missing is left as it was and named on
    stderr; with --complete a missing file is an error, since a finished
    report must not keep a stale table.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Fill the generated tables in RESULTS.md from the files under results/.")
    parser.add_argument("--complete", action="store_true", help="fail when a results file a table needs is missing")
    complete = parser.parse_args().complete
    text = REPORT.read_text()
    missing = []
    for name, generate in GENERATORS.items():
        pattern = re.compile(rf"(<!-- BEGIN GENERATED: {re.escape(name)} -->\n)(?:.*?\n)?(<!-- END GENERATED: {re.escape(name)} -->)", re.DOTALL)
        if not pattern.search(text):
            continue
        try:
            body = generate()
        except FileNotFoundError as e:
            if complete:
                raise
            missing.append(f"{name}: {e}")
            continue
        text = pattern.sub(lambda m, body=body: m.group(1) + body + "\n" + m.group(2), text)
    REPORT.write_text(text)
    print(f"rendered {REPORT}")
    for line in missing:
        print(f"not rendered, input missing: {line}", file=sys.stderr)


if __name__ == "__main__":
    main()
