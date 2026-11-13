"""Purpose: measure the claim check's relevance gate on held-out statements, delta-ATOMIC's updates (Rudinger et al.,
Findings of EMNLP 2020), each written to weaken or to strengthen one typical inference about PersonX. Each inference
is written as the claim its rewrite would make, "If P, H.", and read against a knowledge base holding its own two
updates and updates written for other premises, which bear on it at most by chance. For knowledge bases of several
sizes and each gate spread on a grid it counts the claim's own updates the gate reads and the other updates it reads
per claim, and with the judge it reads them as the check does (semrewrite.claims.read): the share of other premises'
updates read as contradicting a claim before the gate, over a random sample of all such pairs, and among those the
gate keeps; the share of own updates read the way they were written; and the defeats, readings whose disbelief
reaches the judge's threshold, with which the check withdraws a rewrite: weakeners defeating their claim, and other
premises' updates and strengtheners defeating it wrongly. The spread is chosen on the dev split by a false-alarm
budget, counted in spurious contradictions and again in spurious defeats, as a detector's
threshold is under the Neyman-Pearson criterion: the smallest spread, so the one reading most of a claim's own
updates, whose spurious contradictions, other premises' updates kept and read as contradicting the claim, stay at or
under ALPHA per claim at every size; the test split then shows whether the budget holds on statements nothing was
chosen on.

Run from the repository root with delta-ATOMIC's files in data/raw/delta-atomic, where
`python -m semrewrite.pairs fetch delta-atomic` pins them, or in the directory --raw names:
python experiments/claim_gate.py [--judge NAME]. It writes the judge's measurement into results/claim_gate.json,
keeping other judges' there, and RESULTS.md's tables read it. A judge that does not read contradiction is asked each
claim's negation too, as the check asks it. With --choose it only chooses each stored judge's spreads again.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semrewrite.claims import BATCH, conditional, read, robust_z, sentence
from semrewrite.judges import make
from semrewrite.judges.base import SentenceJudge
from semrewrite.opinion import Opinion, W
from semrewrite.retrieve import EmbeddingRetriever

RAW = ROOT / "data" / "raw" / "delta-atomic"
OUT = ROOT / "results" / "claim_gate.json"
# Statements in a knowledge base, a claim's own two updates among them; None is every update of the split.
SIZES = (8, 32, 128, 512, None)
# Gate spreads in robust standard deviations, from 2 to 6; past the last, the gate reads nothing.
SPREADS = tuple(2.0 + 0.25 * i for i in range(17))
# Other premises' updates read against a random claim, standing for every pair the gate could keep.
BEFORE = 2000
# The false-alarm budget: spurious contradictions a gate may let through per claim it reads, one per hundred claims,
# so that a run reading a hundred rewrites expects at most one valid rewrite lowered by a statement that does not bear
# on it.
ALPHA = 0.01
SEED = 20261114


def items(raw: Path, split: str) -> list[tuple[str, str, dict[str, str]]]:
    """The split's inferences, each its premise, its hypothesis and its updates by type, in file order; an update its
    writers found impossible is left out, as Rudinger et al. leave it out."""
    found: dict[tuple[str, str], dict[str, str]] = {}
    for line in (raw / f"{split}.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row["UpdateTypeImpossible"]:
            found.setdefault((row["Premise"], row["Hypothesis"]), {})[row["UpdateType"]] = sentence(row["Update"])
    return [(p, h, u) for (p, h), u in found.items() if set(u) == {"weakener", "strengthener"}]


def right_and_wrong(kind: str, o: Opinion) -> tuple[float, float]:
    """A calibrated reading's evidence pointing the right way and the wrong way, in units of evidence (J 3.11): a
    weakener rightly counts against its claim, a strengthener for it, and another premise's update not at all."""
    r, s = W * o.belief / o.uncertainty, W * o.disbelief / o.uncertainty
    if kind == "weakener":
        return s, r
    if kind == "strengthener":
        return r, s
    return 0.0, r + s


# Time: k x N cosines for k claims and N updates in batches, then per claim and size a median and a median deviation,
# O(k N log N) at the largest size; one judge call over every pair the loosest spread keeps, the own pairs and the
# sample, twice that many for a judge that does not read contradiction.
def measure(raw: Path, split: str, embedder: EmbeddingRetriever, judge: SentenceJudge) -> dict:
    """The gate's counts and the judge's readings on one split, by size and spread."""
    found = items(raw, split)
    claims = [conditional(p, h) for p, h, _ in found]
    statements = sorted({u for _, _, us in found for u in us.values()})
    index = {s: j for j, s in enumerate(statements)}
    premises: dict[int, set[str]] = defaultdict(set)
    for p, _, us in found:
        for u in us.values():
            premises[index[u]].add(p)
    # For each size and claim, the candidates whose z-score reaches the loosest spread, and each own update's z.
    near: dict[tuple[int | None, int], list[tuple[int, float, str]]] = {}
    started = time.perf_counter()
    for start in range(0, len(claims), BATCH):
        for i, row in enumerate(embedder.cosines(claims[start:start + BATCH], statements), start):
            p, _, us = found[i]
            own = {index[u]: kind for kind, u in us.items()}
            others = [j for j in range(len(statements)) if p not in premises[j]]
            local = random.Random(f"{SEED}:{split}:{i}")
            for size in SIZES:
                pool = others if size is None else local.sample(others, min(size - len(own), len(others)))
                candidates = [*own, *pool]
                near[(size, i)] = [(j, z, own.get(j, "other"))
                                   for j, z in zip(candidates, robust_z([row[j] for j in candidates]), strict=True)
                                   if z >= SPREADS[0] or j in own]
    embedding = time.perf_counter() - started
    rng = random.Random(f"{SEED}:{split}")
    before: list[tuple[int, int]] = []
    while len(before) < BEFORE:
        i, j = rng.randrange(len(found)), rng.randrange(len(statements))
        if found[i][0] not in premises[j]:
            before.append((i, j))
    asked = sorted({(i, j) for (_, i), js in near.items() for j, _, _ in js} | set(before))
    started = time.perf_counter()
    raw_readings = dict(zip(asked, read(judge, [(statements[j], found[i][0], found[i][1]) for i, j in asked]),
                            strict=True))
    trust = judge.check_trust if judge.check_trust is not None else judge.trust
    readings = {pair: o.with_base_rate(judge.base_rate).discount(trust) for pair, o in raw_readings.items()}
    judging = time.perf_counter() - started

    def contradicts(pair: tuple[int, int]) -> bool:
        return raw_readings[pair].disbelief > 0.5

    def defeats(pair: tuple[int, int]) -> bool:
        return raw_readings[pair].disbelief >= judge.threshold

    def as_written(pair: tuple[int, int], kind: str) -> bool:
        return contradicts(pair) if kind == "weakener" else raw_readings[pair].belief > 0.5

    owns = [(i, j, kind) for (size, i), js in near.items() if size is None for j, _, kind in js if kind != "other"]
    out: dict = {
        "claims": len(claims), "statements": len(statements), "judge": judge.name, "trust": judge.trust,
        "check_trust": judge.check_trust, "threshold": judge.threshold,
        "seconds": {"embedding": round(embedding, 1), "judging": round(judging, 1)}, "pairs_judged": len(asked),
        "before": {"other_pairs": len(before), "other_contradiction": sum(map(contradicts, before)) / len(before),
                   "other_defeat": sum(map(defeats, before)) / len(before),
                   "own_pairs": len(owns), "own_as_written": sum(as_written((i, j), k) for i, j, k in owns) / len(owns),
                   **{f"{kind}_as_written": sum(as_written((i, j), k) for i, j, k in owns if k == kind)
                      / sum(k == kind for _, _, k in owns) for kind in ("weakener", "strengthener")},
                   **{f"{kind}_defeat": sum(defeats((i, j)) for i, j, k in owns if k == kind)
                      / sum(k == kind for _, _, k in owns) for kind in ("weakener", "strengthener")}},
        "sizes": {}}
    for size in SIZES:
        rows = []
        for spread in SPREADS:
            kept = [(i, j, kind) for (s, i), js in near.items() if s == size for j, z, kind in js if z >= spread]
            other = [(i, j) for i, j, kind in kept if kind == "other"]
            own = [(i, j, kind) for i, j, kind in kept if kind != "other"]
            net = sum(r - w for r, w in (right_and_wrong(kind, readings[(i, j)]) for i, j, kind in kept))
            spurious = sum(map(contradicts, other))
            rows.append({"spread": spread, "own_recall": len(own) / (2 * len(claims)),
                         "others_per_claim": len(other) / len(claims), "spurious_per_claim": spurious / len(claims),
                         "spurious_defeats_per_claim": sum(map(defeats, other)) / len(claims),
                         "weakener_defeats_per_claim": sum(defeats((i, j)) for i, j, k in own if k == "weakener")
                         / len(claims),
                         "strengthener_defeats_per_claim": sum(defeats((i, j)) for i, j, k in own if k == "strengthener")
                         / len(claims),
                         "net_per_claim": net / len(claims),
                         "other_contradiction": spurious / len(other) if other else None,
                         "own_as_written": sum(as_written((i, j), k) for i, j, k in own) / len(own) if own else None})
        out["sizes"]["all" if size is None else str(size)] = rows
    return out


def choose(dev: dict, counted: str = "spurious_per_claim") -> float | None:
    """The smallest spread whose spurious readings per claim, contradictions by default or `counted` such as
    "spurious_defeats_per_claim", stay at or under ALPHA at every size on dev; None when even the largest on the grid
    lets more through."""
    for k, spread in enumerate(SPREADS):
        if all(rows[k][counted] <= ALPHA for rows in dev["sizes"].values()):
            return spread
    return None


def chosen(measured: dict) -> dict[str, float | None]:
    """A judge's spread chosen on its dev updates by spurious contradictions, and by spurious defeats, which withdraw a
    rewrite."""
    return {"chosen_spread": choose(measured["dev"]),
            "chosen_spread_defeats": choose(measured["dev"], "spurious_defeats_per_claim")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the claim check's relevance gate on delta-ATOMIC.")
    parser.add_argument("--raw", type=Path, default=RAW, help="the directory holding delta-ATOMIC's dev and test files")
    parser.add_argument("--judge", default="nli-deberta-large", help="a judge that reads sentences")
    parser.add_argument("--choose", action="store_true",
                        help="only choose each measured judge's spreads again from results/claim_gate.json")
    args = parser.parse_args()
    if args.choose:
        stored = json.loads(OUT.read_text())
        OUT.write_text(json.dumps({j: {**m, **chosen(m)} for j, m in stored.items()}, indent=1, sort_keys=True) + "\n")
        print(json.dumps({j: chosen(m) for j, m in stored.items()}))
        return
    judge = make(args.judge)
    if not isinstance(judge, SentenceJudge):
        msg = f"{args.judge} does not read sentences, so the check cannot use it"
        raise SystemExit(msg)
    embedder = EmbeddingRetriever()
    try:
        measured: dict[str, object] = {split: measure(args.raw, split, embedder, judge) for split in ("dev", "test")}
    finally:
        judge.close()
        embedder.close()
    measured |= chosen(measured)
    kept = json.loads(OUT.read_text()) if OUT.is_file() else {}
    OUT.write_text(json.dumps({**kept, args.judge: measured}, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"judge": args.judge, **chosen(measured)}))


if __name__ == "__main__":
    main()
