"""Purpose: score every direction judge on every labelled pair set under each
filler variant, calibrate each judge's threshold on dev rows, and write the
measurements to results/judges.json and the thresholds the judge factory reads
to semrewrite/judges/thresholds.json.

    .venv/bin/python experiments/judge_eval.py                       # every judge
    .venv/bin/python experiments/judge_eval.py --judges jev,cosine-bge  # these, merged into the existing files

GPU judges load one at a time and are closed before the next loads. Jev
answers come from data/raw/jev/answers.jsonl where cached, so a rerun spends no
requests; new questions go out in batches that count against the request
budget logged in data/raw/jev/requests.jsonl.

Per judge, filler variant and pair set the results give ROC AUC and direction
accuracy on dev and test, accuracy, balanced accuracy, precision and recall at
the engine's default threshold, and on test at the threshold calibrated on that
set's own dev rows. Thresholds maximise balanced accuracy, so the class ratio
the pair builder produced does not move them. The default threshold is
calibrated on Wikidata dev rows, since Wikidata properties are knowledge-base
relations like the engine's catalogue; the hard set has no dev rows and is read
at the default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semrewrite import pairs  # noqa: E402
from semrewrite.judges import JEV_REQUESTS, NAMES, REGISTRY, THRESHOLDS, make  # noqa: E402
from semrewrite.judges.metrics import at_threshold, calibrate, direction, roc_auc  # noqa: E402

RESULTS = ROOT / "results" / "judges.json"
DEFAULT_SET = "wikidata"
ENGINE_VARIANT = "letters"
SPLITS = ("dev", "test")
# What the results are a function of; other files in the tree belong to other work.
INPUTS = ("semrewrite/judging.py", "semrewrite/judges", "semrewrite/pairs.py", "experiments/judge_eval.py", "data")


def score_rows(judge: Any, rows: Sequence[pairs.PairRow], variant: str) -> dict[str, float]:
    """The judge's score for every row offering this filler variant, keyed by row id, in one call."""
    offered = [(r, pairs.fillers_for(r, variant)) for r in rows]
    offered = [(r, f) for r, f in offered if f is not None]
    scores = judge.score([r.proposal for r, _ in offered], fillers=[f for _, f in offered])
    return {r.id: s for (r, _), s in zip(offered, scores, strict=True)}


def one_way(rows: Sequence[pairs.PairRow], scores: dict[str, float]) -> tuple[list[float], list[float]]:
    """The scores of each one-way pair's true and false direction, over pairs with both rows scored."""
    members: dict[str, list[pairs.PairRow]] = defaultdict(list)
    for r in rows:
        if r.id in scores:
            members[r.pair].append(r)
    true_scores, false_scores = [], []
    for group in members.values():
        if len(group) == 2 and group[0].label != group[1].label:
            yes, no = (group[0], group[1]) if group[0].label else (group[1], group[0])
            true_scores.append(scores[yes.id])
            false_scores.append(scores[no.id])
    return true_scores, false_scores


def measure(rows: Sequence[pairs.PairRow], scores: dict[str, float], default: float | None, own: float | None) -> dict[str, Any]:
    """One pair set's measurements under one judge and variant, by split."""
    out: dict[str, Any] = {}
    for split in SPLITS:
        part = [r for r in rows if r.split == split and r.id in scores]
        if not part:
            continue
        s = [scores[r.id] for r in part]
        y = [r.label for r in part]
        entry: dict[str, Any] = {
            "rows": len(part),
            "positives": sum(y),
            "auc": roc_auc(s, y),
            "direction": direction(*one_way(part, scores)),
        }
        if default is not None:
            entry["at_default"] = at_threshold(s, y, default)
        if own is not None and split == "test":
            entry["at_own_dev"] = at_threshold(s, y, own)
        out[split] = entry
    return out


def thresholds_for(rows: Sequence[pairs.PairRow], scores: dict[str, float]) -> dict[str, float]:
    """Each pair set's dev-calibrated threshold under these scores, for sets with scored dev rows."""
    out = {}
    for name in pairs.SETS:
        dev = [r for r in rows if r.set == name and r.split == "dev" and r.id in scores]
        if dev:
            out[name] = calibrate([scores[r.id] for r in dev], [r.label for r in dev])
    return out


def run_judge(name: str, rows: Sequence[pairs.PairRow], dump: pathlib.Path | None) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Score one judge on every row and variant, then release it; returns its results and scores by variant."""
    import torch

    gpu = torch.cuda.is_available()
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    judge = make(name, threshold=0.5)  # provisional; the calibrated value is written below
    seconds: dict[str, float] = {"load": round(time.monotonic() - started, 2)}
    by_variant: dict[str, dict[str, float]] = {}
    try:
        for variant in pairs.FILLER_VARIANTS:
            started = time.monotonic()
            by_variant[variant] = score_rows(judge, rows, variant)
            seconds[variant] = round(time.monotonic() - started, 2)
            print(f"{name} {variant}: {len(by_variant[variant])} rows in {seconds[variant]} s", flush=True)
    finally:
        usage = dict(judge.usage)
        model = getattr(judge, "model_name", None) or getattr(judge, "model", None)
        judge.close()
    if dump is not None:
        dump.mkdir(parents=True, exist_ok=True)
        with (dump / f"{name}.jsonl").open("w") as handle:
            for variant, scores in by_variant.items():
                for row_id, score in scores.items():
                    handle.write(json.dumps({"id": row_id, "variant": variant, "score": score}) + "\n")

    variants: dict[str, Any] = {}
    calibrated: dict[str, dict[str, float]] = {}
    for variant, scores in by_variant.items():
        own = thresholds_for(rows, scores)
        calibrated[variant] = own
        default = own.get(DEFAULT_SET)
        variants[variant] = {
            "thresholds": {"default": default, "by_set": own},
            "sets": {
                name_: measure([r for r in rows if r.set == name_], scores, default, own.get(name_))
                for name_ in pairs.SETS
            },
        }
    result = {
        "judge": REGISTRY[name][0].__name__,
        "model": model,
        "usage": usage,
        "seconds": seconds,
        "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if gpu else None,
        "variants": variants,
    }
    return result, {"scores": by_variant, "thresholds": calibrated}


def hard_table(rows: Sequence[pairs.PairRow], scored: dict[str, dict[str, dict[str, float]]], previous: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every hard-set row with each judge's score under each variant, so each tricky pair can be read."""
    kept = {entry["id"]: entry for entry in previous}
    table = []
    for r in rows:
        if r.set != "hard":
            continue
        premise, conclusion = r.proposal.sentences()
        entry = kept.get(r.id, {}).get("scores", {})
        for judge, by_variant in scored.items():
            entry[judge] = {v: round(s[r.id], 4) for v, s in by_variant.items() if r.id in s}
        table.append({
            "id": r.id, "premise": premise, "conclusion": conclusion, "label": r.label,
            "kind": r.kind, "reason": r.reason, "scores": dict(sorted(entry.items())),
        })
    return table


def provenance() -> dict[str, Any]:
    """What these results were computed from: the code revision, each pair file's digest, and the
    Jev request ledger's totals, which a rerun answered from the cache does not add to."""
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", *INPUTS],
                           capture_output=True, text=True, check=False)
    ledger = [json.loads(line) for line in JEV_REQUESTS.read_text().splitlines() if line] if JEV_REQUESTS.is_file() else []
    return {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": "python experiments/judge_eval.py " + " ".join(sys.argv[1:]),
        "commit": head.stdout.strip() or None,
        "inputs_uncommitted": sorted(line[3:] for line in dirty.stdout.splitlines()),
        "pairs": {
            name: hashlib.sha256((pairs.PAIRS / f"{name}.jsonl").read_bytes()).hexdigest() for name in pairs.SETS
        },
        "jev_ledger": {
            "requests": len(ledger),
            "questions": sum(row["questions"] for row in ledger),
            "input_tokens": sum(row.get("input_tokens") or 0 for row in ledger),
            "models": sorted({row["model"] for row in ledger if "model" in row}),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--judges", default=",".join(NAMES), help="comma-separated judge names")
    parser.add_argument("--scores", type=pathlib.Path, help="directory to dump every row's score per judge")
    args = parser.parse_args(argv)
    names = [n for n in args.judges.split(",") if n]
    unknown = sorted(set(names) - set(NAMES))
    if unknown:
        parser.error(f"unknown judges {unknown}; known: {', '.join(NAMES)}")

    rows = pairs.load()
    pairs.check(rows)
    results = json.loads(RESULTS.read_text()) if RESULTS.is_file() else {}
    table = json.loads(THRESHOLDS.read_text()) if THRESHOLDS.is_file() else {}
    judged: dict[str, Any] = {k: v for k, v in results.get("judges", {}).items() if k in NAMES}
    thresholds: dict[str, Any] = {k: v for k, v in table.get("judges", {}).items() if k in NAMES}
    scored: dict[str, dict[str, dict[str, float]]] = {}
    for name in names:
        judged[name], details = run_judge(name, rows, args.scores)
        scored[name] = details["scores"]
        engine = details["thresholds"][ENGINE_VARIANT]
        thresholds[name] = {"default": engine[DEFAULT_SET], "by_set": engine}

    results = {
        "provenance": provenance(),
        "calibration": {
            "rule": "maximum dev balanced accuracy of admitting scores at or above the threshold, middle of the optimal plateau",
            "default": f"{DEFAULT_SET} dev rows under each variant's fillers",
            "engine_variant": ENGINE_VARIANT,
        },
        "judges": dict(sorted(judged.items())),
        "hard_rows": hard_table(rows, scored, results.get("hard_rows", [])),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1, sort_keys=False) + "\n")
    THRESHOLDS.write_text(json.dumps({
        "about": f"written by experiments/judge_eval.py: default calibrated on {DEFAULT_SET} dev rows with "
                 f"{ENGINE_VARIANT} fillers; by_set on each set's dev rows",
        "judges": dict(sorted(thresholds.items())),
    }, indent=1) + "\n")
    print(f"wrote {RESULTS.relative_to(ROOT)} and {THRESHOLDS.relative_to(ROOT)} for {', '.join(names)}")


if __name__ == "__main__":
    main()
