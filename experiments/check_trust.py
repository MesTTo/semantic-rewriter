"""Purpose: fit the trust the claim check puts in each sentence judge's reading of a statement against a rewrite's
claim, on delta-ATOMIC (Rudinger et al., Findings of EMNLP 2020). Each update is a statement, and its inference,
written as the claim its premise brings, "If PersonX ..., ...", is read against it as semrewrite.claims.read reads a
statement against a rewrite's claim. A strengthener's claim holds and a weakener's does not. The trust maximising the
dev labels' likelihood when each stance projects at the set's base rate after discounting by the trust
(semrewrite.judges.metrics.fit_trust, as experiments/judge_eval.py fits a judge's trust on predicate pairs) is the
judge's check trust.

On the test updates it reports log loss and Brier score at that trust, at the judge's predicate-pair trust and at
none, which leaves the base rate, and how often a stance reaches the judge's threshold as a contradiction, so the check
would withdraw the rewrite: for weakeners, which it should, for strengtheners, which it should not, and for other
premises' updates read against a claim, which bear on it not at all. It writes the measurement to
results/check_trust.json and each judge's check_trust into semrewrite/judges/calibration.json, leaving every other
value there as it was; the set's base rate stays in the file's sets table, once.

Run from the repository root, with a GPU for the model judges: python experiments/check_trust.py [--judges a,b]
[--stances DIR]. Jev is left out by default: it needs a TypeSafe key and spends its budget on every pair.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semrewrite import pairs
from semrewrite.claims import read, sentence
from semrewrite.judges import CALIBRATION, CHECK_SET, NAMES
from semrewrite.judges import make as make_judge
from semrewrite.judges.base import SentenceJudge
from semrewrite.judges.metrics import fit_trust, probability_scores, roc_auc
from semrewrite.opinion import Opinion

OUT = ROOT / "results" / "check_trust.json"
# What a run reads, whose uncommitted changes its provenance names.
INPUTS = ("experiments/check_trust.py", "semrewrite", "data/raw/delta-atomic")
# Other premises' updates read against a test claim, standing for statements that bear on it not at all.
OTHERS = 2000
SEED = 20261114


def provenance() -> dict[str, Any]:
    """What a run is computed from, taken before the judges load: the code revision and the inputs that differ
    from it."""
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", *INPUTS],
                           capture_output=True, text=True, check=False)
    return {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": "python experiments/check_trust.py " + " ".join(sys.argv[1:]),
            "commit": head.stdout.strip() or None,
            "inputs_uncommitted": sorted(line[3:] for line in dirty.stdout.splitlines())}


def asked(items: Sequence[pairs.DefeasibleItem]) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Each item's (update, premise, inference), the update read as a statement against the claim "If premise,
    inference.", and OTHERS test claims each read against the update of a test item with another premise, drawn with
    a fixed seed."""
    own = [(sentence(i.update), i.premise, i.hypothesis) for i in items]
    test = [i for i in items if i.split == "test"]
    rng = random.Random(SEED)
    others: list[tuple[str, str, str]] = []
    while len(others) < OTHERS:
        claim, statement = rng.sample(test, 2)
        if claim.premise != statement.premise:
            others.append((sentence(statement.update), claim.premise, claim.hypothesis))
    return own, others


def stances(judge: SentenceJudge, questions: Sequence[tuple[str, str, str]], cache: Path | None) -> list[Opinion]:
    """The judge's stance on each question, read from `cache` when an earlier run of this judge wrote the same
    questions there, else read by the judge and written to `cache`."""
    if cache is not None and cache.is_file():
        rows = [json.loads(line) for line in cache.read_text().splitlines()]
        if [tuple(r["question"]) for r in rows] == [tuple(q) for q in questions]:
            return [Opinion(r["belief"], r["disbelief"], r["uncertainty"]) for r in rows]
    found = read(judge, questions)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("".join(json.dumps({"question": list(q), "belief": o.belief, "disbelief": o.disbelief,
                                             "uncertainty": o.uncertainty}) + "\n" for q, o in zip(questions, found,
                                                                                                   strict=True)))
    return found


def scored(opinions: Sequence[Opinion], labels: Sequence[bool], rate: float, trust: float) -> dict[str, float]:
    """Log loss and Brier score of the stances' projections at `rate`, discounted by `trust`."""
    return probability_scores([o.with_base_rate(rate).discount(trust).projection for o in opinions], labels)


def defeat_rate(opinions: Sequence[Opinion], bar: float) -> float | None:
    """The share of stances whose disbelief reaches `bar`, which the check reads as a clear contradiction."""
    return sum(o.disbelief >= bar for o in opinions) / len(opinions) if opinions else None


def measure(name: str, items: Sequence[pairs.DefeasibleItem], rate: float, cache: Path | None) -> dict[str, Any]:
    """Judge `name`'s check trust fitted on the dev items, and its scores and defeat rates on dev and test."""
    judge = make_judge(name)
    if not isinstance(judge, SentenceJudge):
        msg = f"judge {name} does not read sentences, so the claim check cannot use it"
        raise ValueError(msg)  # noqa: TRY004 - a judge name it cannot use, as the command line refuses it
    own, others = asked(items)
    started = time.perf_counter()
    try:
        found = stances(judge, [*own, *others], None if cache is None else cache / f"{name}.jsonl")
    finally:
        judge.close()
    seconds = time.perf_counter() - started
    read_own, read_others = found[:len(own)], found[len(own):]
    split = {s: [(o, i.strengthens) for o, i in zip(read_own, items, strict=True) if i.split == s]
             for s in ("dev", "test")}
    dev_opinions, dev_labels = [o for o, _ in split["dev"]], [y for _, y in split["dev"]]
    trust = fit_trust([o.with_base_rate(rate) for o in dev_opinions], dev_labels)
    bar = judge.threshold
    out: dict[str, Any] = {"check_trust": trust, "trust": judge.trust, "threshold": bar,
                           "reads_contradiction": judge.reads_contradiction, "seconds": round(seconds, 1)}
    for s, rows in split.items():
        opinions, labels = [o for o, _ in rows], [y for _, y in rows]
        weakeners = [o for o, y in rows if not y]
        strengtheners = [o for o, y in rows if y]
        out[s] = {
            "rows": len(rows),
            "at_check_trust": scored(opinions, labels, rate, trust),
            "at_trust": scored(opinions, labels, rate, judge.trust),
            "at_base_rate": scored(opinions, labels, rate, 0.0),
            "auc": roc_auc([o.belief - o.disbelief for o in opinions], labels),
            "defeat": {"weakener": defeat_rate(weakeners, bar), "strengthener": defeat_rate(strengtheners, bar)},
        }
    out["test"]["defeat"]["other"] = defeat_rate(read_others, bar)
    return out


def store(trusts: dict[str, float], rate: float, path: Path = CALIBRATION) -> None:
    """Write each judge's check trust into the calibration file, every other value kept, and CHECK_SET's base rate in
    its sets table once, refusing a different one already there."""
    table = json.loads(path.read_text())
    held = table.setdefault("sets", {}).setdefault(CHECK_SET, {})
    if "base_rate" in held and abs(held["base_rate"] - rate) > 1e-12:
        msg = f"{path} holds {CHECK_SET}'s base rate {held['base_rate']}, but its dev items give {rate}"
        raise ValueError(msg)
    held["base_rate"] = rate
    for name, trust in trusts.items():
        table.setdefault("judges", {}).setdefault(name, {})["check_trust"] = trust
    path.write_text(json.dumps(table, indent=1) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--judges", default=",".join(n for n in NAMES if n != "jev"), help="comma-separated judge names")
    parser.add_argument("--stances", type=Path, help="directory to keep each judge's stances in, and reuse them from")
    args = parser.parse_args(argv)
    names = [n for n in args.judges.split(",") if n]
    unknown = sorted(set(names) - set(NAMES))
    if unknown:
        parser.error(f"unknown judges {unknown}; known: {', '.join(NAMES)}")
    state = provenance()
    items = pairs.delta_atomic_items()
    dev = [i for i in items if i.split == "dev"]
    rate = sum(i.strengthens for i in dev) / len(dev)
    previous = json.loads(OUT.read_text()) if OUT.is_file() else {}
    measured = {k: v for k, v in previous.get("judges", {}).items() if k in NAMES}
    for name in names:
        measured[name] = measure(name, items, rate, args.stances)
        print(f"{name}: check trust {measured[name]['check_trust']:.4f}, test log loss "
              f"{measured[name]['test']['at_check_trust']['log_loss']:.4f}, weakeners defeated "
              f"{measured[name]['test']['defeat']['weakener']:.3f}", flush=True)
    results = {
        "provenance": state,
        "set": CHECK_SET,
        "base_rate": rate,
        "rule": "each update read against its inference written as the claim If premise, inference. as the claim "
                "check reads a statement against a rewrite's claim; a strengthener's claim holds, a weakener's does "
                "not; the check trust maximises the dev labels' likelihood under each stance's projection at the "
                "set's base rate after discounting by it (Josang 2016, Eq. 14.6); a stance whose disbelief reaches "
                "the judge's threshold defeats",
        "others": {"rows": OTHERS, "seed": SEED},
        "judges": dict(sorted(measured.items())),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1) + "\n")
    store({name: measured[name]["check_trust"] for name in names}, rate)
    print(f"wrote {OUT.relative_to(ROOT)} and {CALIBRATION.relative_to(ROOT)} for {', '.join(names)}")


if __name__ == "__main__":
    main()
