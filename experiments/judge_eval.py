"""Purpose: judge every labelled pair set with every direction judge under each
filler variant, calibrate each judge's threshold and trust on dev rows, and
write the measurements to results/judges.json and the thresholds and trusts the
judge factory reads to semrewrite/judges/calibration.json.

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

A judge's trust is fitted on the same dev rows: the trust maximising the
labels' likelihood when each row holds with the projection of the judge's
opinion discounted by it. The results give the log loss and Brier score of
those projections on dev and test at the default trust.

The knowledge base is a second source on a pair whose two properties both have
statements in the Wikidata examples' knowledge bases: the premise's statements
read through the pair's mapping, against the conclusion's (semrewrite.evidence).
Each judge's data trust is fitted on those dev rows jointly with its opinions,
fused with them, and the results give the test scores of the judge alone and
fused on the same rows.
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

from examples.wikidata.questions import property_id
from semrewrite import formats, pairs
from semrewrite.evidence import rewrite_evidence
from semrewrite.judges import CALIBRATION, JEV_REQUESTS, NAMES, REGISTRY, make
from semrewrite.judges.metrics import (
    at_threshold,
    calibrate,
    direction,
    fit_data_trust,
    fit_trust,
    probability_scores,
    roc_auc,
)
from semrewrite.model import Rewrite
from semrewrite.opinion import Opinion

RESULTS = ROOT / "results" / "judges.json"
DEFAULT_SET = "wikidata"
ENGINE_VARIANT = "letters"
SPLITS = ("dev", "test")
# The knowledge bases whose statements are the second source on Wikidata pairs.
EXAMPLES = tuple(ROOT / "examples" / "wikidata" / f"{name}.metta" for name in ("victoria", "leonardo", "alphabet"))
# What the results are a function of; other files in the tree belong to other work.
INPUTS = ("semrewrite/model.py", "semrewrite/judges", "semrewrite/pairs.py", "semrewrite/evidence.py",
          "semrewrite/opinion.py", "experiments/judge_eval.py", "data", "examples/wikidata")


def judge_rows(judge: Any, rows: Sequence[pairs.PairRow], variant: str) -> dict[str, Opinion]:
    """The judge's opinion of every row offering this filler variant, keyed by row id, in one call."""
    offered = [(r, pairs.fillers_for(r, variant)) for r in rows]
    offered = [(r, f) for r, f in offered if f is not None]
    opinions = judge.opinions([r.proposal for r, _ in offered], fillers=[f for _, f in offered])
    return {r.id: o for (r, _), o in zip(offered, opinions, strict=True)}


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


def measure(rows: Sequence[pairs.PairRow], opinions: dict[str, Opinion], default: float | None, own: float | None,
            calibration: tuple[float, float] | None) -> dict[str, Any]:
    """One pair set's measurements under one judge and variant, by split; a row's score is its opinion's belief, and
    its probability its opinion's projection at the calibrated base rate after discounting by the calibrated trust."""
    scores = {key: o.belief for key, o in opinions.items()}
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
        if calibration is not None:
            entry["at_default_trust"] = probability_scores([calibrated(opinions[r.id], calibration).projection
                                                            for r in part], y)
        out[split] = entry
    return out


def calibrated(o: Opinion, calibration: tuple[float, float]) -> Opinion:
    """A judge's opinion at a calibrated (trust, base rate), as the score cache gives it to the rewriter."""
    trust, rate = calibration
    return o.with_base_rate(rate).discount(trust)


def thresholds_for(rows: Sequence[pairs.PairRow], scores: dict[str, float]) -> dict[str, float]:
    """Each pair set's dev-calibrated threshold under these scores, for sets with scored dev rows."""
    out = {}
    for name in pairs.SETS:
        dev = [r for r in rows if r.set == name and r.split == "dev" and r.id in scores]
        if dev:
            out[name] = calibrate([scores[r.id] for r in dev], [r.label for r in dev])
    return out


def base_rates(rows: Sequence[pairs.PairRow]) -> dict[str, float]:
    """Each pair set's base rate: the share of its dev rows that hold, the prior every judge's opinions project to
    where they say nothing."""
    out = {}
    for name in pairs.SETS:
        dev = [r.label for r in rows if r.set == name and r.split == "dev"]
        if dev:
            out[name] = sum(dev) / len(dev)
    return out


def calibrations_for(rows: Sequence[pairs.PairRow], opinions: dict[str, Opinion], rates: dict[str, float]
                     ) -> dict[str, tuple[float, float]]:
    """Each pair set's (trust, base rate) under these opinions, for sets with judged dev rows: the trust fitted on
    the dev rows to the opinions at the set's base rate."""
    out = {}
    for name in pairs.SETS:
        dev = [r for r in rows if r.set == name and r.split == "dev" and r.id in opinions]
        if dev:
            at = [opinions[r.id].with_base_rate(rates[name]) for r in dev]
            out[name] = (fit_trust(at, [r.label for r in dev]), rates[name])
    return out


# Time: one pass over the examples' f statements, then per row the premise's statements.
def data_opinions(rows: Sequence[pairs.PairRow]) -> dict[str, Opinion]:
    """The knowledge base's opinion of each Wikidata pair whose two properties both have statements in the examples'
    knowledge bases: the premise's statements read through the pair's mapping, as evidence for and against their
    being the conclusion's (semrewrite.evidence)."""
    statements: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for path in EXAMPLES:
        for f in formats.load(path).facts:
            pid = property_id(f.pred)
            if pid is not None:
                statements[pid].add(f.args)
    out = {}
    for r in rows:
        if r.set != "wikidata":
            continue
        # The pair as a rewrite between placeholder predicates holding its properties' statements. Its evidence is
        # (0, 0) exactly when either property has no statements, which leaves the pair without data.
        stated = {name: statements.get(p.name.split(":")[0], set())
                  for name, p in (("conclusion", r.conclusion), ("premise", r.premise))}
        rewrite = Rewrite("conclusion", "premise", r.proposal.mapping)
        evidence = rewrite_evidence(stated, [rewrite])[rewrite.key]
        if sum(evidence):
            out[r.id] = Opinion.from_evidence(*evidence)
    return out


def data_measure(rows: Sequence[pairs.PairRow], opinions: dict[str, Opinion], data: dict[str, Opinion],
                 calibration: tuple[float, float] | None) -> dict[str, Any]:
    """The data trust fitted on dev rows with data jointly with the judge's opinions at its calibration, and on each
    split the scores of the judge alone and fused with the data at that trust, over the same rows. The data's
    opinions take the judge's base rate, being of the same proposition, as the rewriter gives them."""
    if calibration is None:
        return {}
    part = {split: [r for r in rows if r.split == split and r.id in data and r.id in opinions] for split in SPLITS}
    if not part["dev"]:
        return {}
    rate = calibration[1]
    alone = {r.id: calibrated(opinions[r.id], calibration) for split in SPLITS for r in part[split]}
    source = {r.id: data[r.id].with_base_rate(rate) for split in SPLITS for r in part[split]}
    t = fit_data_trust([alone[r.id] for r in part["dev"]], [source[r.id] for r in part["dev"]],
                       [r.label for r in part["dev"]])
    out: dict[str, Any] = {"trust": t}
    for split in SPLITS:
        y = [r.label for r in part[split]]
        if y:
            out[split] = {
                "judge": probability_scores([alone[r.id].projection for r in part[split]], y),
                "fused": probability_scores([alone[r.id].fuse(source[r.id].discount(t)).projection
                                             for r in part[split]], y),
            }
    return out


def run_judge(name: str, rows: Sequence[pairs.PairRow], data: dict[str, Opinion], rates: dict[str, float],
              dump: pathlib.Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Judge every row and variant with one judge, then release it; returns its results, and its scores, thresholds
    and trusts by variant; `data` is the knowledge base's opinion of the rows it has evidence on."""
    import torch

    gpu = torch.cuda.is_available()
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    judge = make(name, threshold=0.5, trust=1.0, base_rate=0.5)  # provisional; the calibrated values are written below
    seconds: dict[str, float] = {"load": round(time.monotonic() - started, 2)}
    by_variant: dict[str, dict[str, Opinion]] = {}
    try:
        for variant in pairs.FILLER_VARIANTS:
            started = time.monotonic()
            by_variant[variant] = judge_rows(judge, rows, variant)
            seconds[variant] = round(time.monotonic() - started, 2)
            print(f"{name} {variant}: {len(by_variant[variant])} rows in {seconds[variant]} s", flush=True)
    finally:
        usage = dict(judge.usage)
        model = getattr(judge, "model_name", None) or getattr(judge, "model", None)
        judge.close()
    if dump is not None:
        dump.mkdir(parents=True, exist_ok=True)
        with (dump / f"{name}.jsonl").open("w") as handle:
            for variant, opinions in by_variant.items():
                for row_id, o in opinions.items():
                    handle.write(json.dumps({"id": row_id, "variant": variant, "score": o.belief,
                                             "opinion": [o.belief, o.disbelief, o.uncertainty]}) + "\n")

    variants: dict[str, Any] = {}
    thresholds: dict[str, dict[str, float]] = {}
    calibrations: dict[str, dict[str, tuple[float, float]]] = {}
    data_trusted: dict[str, float | None] = {}
    scored: dict[str, dict[str, float]] = {}
    for variant, opinions in by_variant.items():
        scores = {key: o.belief for key, o in opinions.items()}
        own = thresholds_for(rows, scores)
        fits = calibrations_for(rows, opinions, rates)
        thresholds[variant] = own
        calibrations[variant] = fits
        scored[variant] = scores
        default = own.get(DEFAULT_SET)
        fit = fits.get(DEFAULT_SET)
        with_data = data_measure([r for r in rows if r.set == DEFAULT_SET], opinions, data, fit)
        data_trusted[variant] = with_data.get("trust")
        variants[variant] = {
            "thresholds": {"default": default, "by_set": own},
            "trust": {"default": fit and fit[0], "by_set": {k: t for k, (t, _) in fits.items()}},
            "data": with_data,
            "sets": {
                name_: measure([r for r in rows if r.set == name_], opinions, default, own.get(name_), fit)
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
    return result, {"scores": scored, "thresholds": thresholds, "calibrations": calibrations,
                    "data_trusts": data_trusted}


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
    data = data_opinions(rows)
    rates = base_rates(rows)
    results = json.loads(RESULTS.read_text()) if RESULTS.is_file() else {}
    table = json.loads(CALIBRATION.read_text()) if CALIBRATION.is_file() else {}
    judged: dict[str, Any] = {k: v for k, v in results.get("judges", {}).items() if k in NAMES}
    calibration: dict[str, Any] = {k: v for k, v in table.get("judges", {}).items() if k in NAMES}
    scored: dict[str, dict[str, dict[str, float]]] = {}
    for name in names:
        judged[name], details = run_judge(name, rows, data, rates, args.scores)
        scored[name] = details["scores"]
        engine = details["thresholds"][ENGINE_VARIANT]
        trust, _ = details["calibrations"][ENGINE_VARIANT][DEFAULT_SET]
        calibration[name] = {"default": engine[DEFAULT_SET], "by_set": engine,
                             "trust": trust,
                             "data_trust": details["data_trusts"][ENGINE_VARIANT]}

    results = {
        "provenance": provenance(),
        "calibration": {
            "rule": "maximum dev balanced accuracy of admitting scores at or above the threshold, middle of the optimal plateau",
            "default": f"{DEFAULT_SET} dev rows under each variant's fillers",
            "engine_variant": ENGINE_VARIANT,
            "trust": "maximum dev likelihood of the labels under each opinion's projection at its set's base rate "
                     "after discounting by the trust (Josang 2016, Eq. 14.6)",
            "base_rates": rates,
            "base_rate": "each set's share of dev rows that hold: the projection of an opinion that says nothing",
            "data_trust": f"maximum likelihood on {DEFAULT_SET} dev rows with statements on both sides in "
                          f"{', '.join(p.name for p in EXAMPLES)}, of the judge's trusted opinion fused with the "
                          "knowledge base's evidence discounted by the data trust",
        },
        "judges": dict(sorted(judged.items())),
        "hard_rows": hard_table(rows, scored, results.get("hard_rows", [])),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1, sort_keys=False) + "\n")
    CALIBRATION.write_text(json.dumps({
        "about": f"written by experiments/judge_eval.py: the default threshold, and the trust and base rate, "
                 f"calibrated on {DEFAULT_SET} dev rows with {ENGINE_VARIANT} fillers; by_set thresholds on each "
                 "set's dev rows",
        "base_rate": rates[DEFAULT_SET],
        "judges": dict(sorted(calibration.items())),
    }, indent=1) + "\n")
    print(f"wrote {RESULTS.relative_to(ROOT)} and {CALIBRATION.relative_to(ROOT)} for {', '.join(names)}")


if __name__ == "__main__":
    main()
