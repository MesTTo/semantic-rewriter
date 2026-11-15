"""Purpose: fit the trust the claim check puts in each sentence judge's reading of a statement about an application's
individuals, on delta-ATOMIC (Rudinger et al., Findings of EMNLP 2020) with a person's name filled in for PersonX and
PersonY, and measure that reading against the one it replaced. Each update is a statement about one person, and its
premise and inference are an application's premise and conclusion, "kim plays games." and "kim wastes time.". The
taken form reads the update with the premise against the inference (semrewrite.claims.read_given), as the check reads
an application's statement; the rejected form, the named conditional, reads it against "If premise, inference."
(semrewrite.claims.read). A strengthener's inference holds and a weakener's does not. The trust maximising the dev
labels' likelihood when each stance projects at the set's base rate after discounting by it
(semrewrite.judges.metrics.fit_trust, as experiments/check_trust.py fits the check trust) is the form's trust, and the
taken form's is the judge's exception trust.

On the test updates it reports, for each form, log loss and AUC, and the defeats, stances whose disbelief reaches the
judge's threshold: weakeners, which a defeat should remove, and strengtheners and other premises' updates about the
same person, which it should not, with a defeat's precision, its weakeners over all its own updates, and recall, the
weakeners it reaches. It then measures the relevance gate as an application meets it: every update of a split names
its own item's people, each claim is written with letters as the check writes it, and for knowledge bases of several
sizes it counts the updates standing out at the deployed spread, a claim's own and others, and how often each form
reads a kept other, filled with the claim's names, as a defeat. It writes results/exception_trust.json, each judge's
with the provenance of the run that measured it, since the file keeps the judges earlier runs measured, and each
judge's exception_trust into semrewrite/judges/calibration.json, every other value there kept.

Run from the repository root, with a GPU for the model judges: python experiments/exception_trust.py [--judges a,b]
[--stances DIR].
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from check_trust import defeat_rate, precision_recall, provenance, scored, stances, store
from claim_gate import SIZES

from semrewrite import pairs
from semrewrite.claims import BATCH, SPREAD, conditional, read, read_given, robust_z, sentence
from semrewrite.judges import CALIBRATION, CHECK_SET, NAMES
from semrewrite.judges import make as make_judge
from semrewrite.judges.base import LETTERS, SentenceJudge
from semrewrite.judges.metrics import fit_trust, roc_auc
from semrewrite.opinion import Opinion
from semrewrite.retrieve import EmbeddingRetriever

OUT = ROOT / "results" / "exception_trust.json"
INPUTS = ("experiments/exception_trust.py", "experiments/check_trust.py", "semrewrite", "data/raw/delta-atomic")
# The people PersonX and PersonY become, two distinct per item, drawn with a fixed seed; lowercase, as the check writes
# a constant such as kim.
PEOPLE = ("kim", "sam", "lee", "ana", "joe", "max", "eve", "raj", "amy", "tom", "liz", "ben", "mia", "dan", "zoe", "ian")
# Other premises' updates read against a test claim, about the claim's own people, standing for statements about an
# application's individuals that bear on it not at all.
OTHERS = 2000
SEED = 20261116
Reader = Callable[[SentenceJudge, Sequence[tuple[str, str, str]]], list[Opinion]]
# The forms an update is read in: the one the check takes for an application's statements, and the one it rejected.
FORMS: dict[str, Reader] = {"premise and statement": read_given, "named conditional": read}
TAKEN = "premise and statement"

Item = pairs.DefeasibleItem
Names = tuple[str, str]


def fill(text: str, names: Names) -> str:
    """`text` with PersonX and PersonY named: fill("PersonX calls PersonY", ("kim", "sam")) is "kim calls sam"."""
    return text.replace("PersonX", names[0]).replace("PersonY", names[1])


def people(items: Sequence[Item]) -> dict[str, Names]:
    """Each item's two people, by item id, drawn with SEED in file order."""
    rng = random.Random(SEED)
    named: dict[str, Names] = {}
    for i in items:
        x, y = rng.sample(PEOPLE, 2)
        named[i.id] = (x, y)
    return named


def question(statement: Item, claim: Item, names: Names) -> tuple[str, str, str]:
    """`statement`'s update read against `claim`'s premise and inference, everything about the people `names`."""
    return sentence(fill(statement.update, names)), fill(claim.premise, names), fill(claim.hypothesis, names)


def asked(items: Sequence[Item], named: dict[str, Names]) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Each item's own question, and OTHERS test claims each read against the update of a test item with another
    premise, filled with the claim's people, drawn with a fixed seed."""
    own = [question(i, i, named[i.id]) for i in items]
    test = [i for i in items if i.split == "test"]
    rng = random.Random(SEED)
    others: list[tuple[str, str, str]] = []
    while len(others) < OTHERS:
        claim, statement = rng.sample(test, 2)
        if claim.premise != statement.premise:
            others.append(question(statement, claim, named[claim.id]))
    return own, others


# Time: k x N cosines for k claims and N updates in batches, then per claim and size a median and a median deviation,
# O(k N log N) at the largest size; one judge call per form over the other updates the gate keeps at any size.
def gate(split: str, items: Sequence[Item], named: dict[str, Names], own: dict[str, dict[str, Opinion]],
         judge: SentenceJudge, embedder: EmbeddingRetriever, cache: Path | None) -> dict[str, Any]:
    """The gate as an application meets it on one split: each item's claim with letters against the split's updates,
    each naming its own item's people, by knowledge base size, the claim's own update among them; for each form, how
    often a kept other read about the claim's people is a defeat, and the claim's own update kept and defeating, its
    stance taken from `own`, each form's stance by item id."""
    found = [i for i in items if i.split == split]
    letters = (LETTERS[0], LETTERS[1])
    claims = [conditional(fill(i.premise, letters), fill(i.hypothesis, letters)) for i in found]
    texts = [sentence(fill(i.update, named[i.id])) for i in found]
    kept: dict[int | None, list[tuple[int, int]]] = defaultdict(list)
    own_kept: dict[int | None, list[int]] = defaultdict(list)
    drawn: dict[int | None, int] = defaultdict(int)
    started = time.perf_counter()
    for start in range(0, len(claims), BATCH):
        for c, row in enumerate(embedder.cosines(claims[start:start + BATCH], texts), start):
            others = [j for j in range(len(found)) if found[j].premise != found[c].premise]
            local = random.Random(f"{SEED}:{split}:{c}")
            for size in SIZES:
                pool = others if size is None else local.sample(others, min(size - 1, len(others)))
                drawn[size] += len(pool)
                candidates = [c, *pool]
                for j, z in zip(candidates, robust_z([row[j] for j in candidates]), strict=True):
                    if z < SPREAD:
                        continue
                    if j == c:
                        own_kept[size].append(c)
                    else:
                        kept[size].append((c, j))
    embedding = time.perf_counter() - started
    pairs_kept = sorted({p for rows in kept.values() for p in rows})
    bar = judge.threshold
    out: dict[str, Any] = {"claims": len(claims), "statements": len(texts), "embedding_seconds": round(embedding, 1)}
    for form, reader in FORMS.items():
        questions = [question(found[j], found[c], named[found[c].id]) for c, j in pairs_kept]
        where = None if cache is None else cache / f"{judge.name}.gate.{split}.{form.replace(' ', '-')}.jsonl"
        read_kept = dict(zip(pairs_kept, stances(judge, questions, where, reader), strict=True))
        rows = {}
        for size in SIZES:
            others = kept[size]
            defeats = sum(read_kept[p].disbelief >= bar for p in others)
            mine = [found[c] for c in own_kept[size]]
            rows["all" if size is None else str(size)] = {
                "own_kept": {kind: sum(i.strengthens == (kind == "strengthener") for i in mine) / len(claims)
                             for kind in ("weakener", "strengthener")},
                "own_defeats": {kind: sum(i.strengthens == (kind == "strengthener")
                                          and own[form][i.id].disbelief >= bar for i in mine) / len(claims)
                                for kind in ("weakener", "strengthener")},
                "others_kept_per_claim": len(others) / len(claims),
                "others_kept_per_statement": len(others) / drawn[size],
                "kept_other_defeat_rate": defeats / len(others) if others else None,
            }
        out[form] = rows
    return out


def measure(name: str, items: Sequence[Item], rate: float, cache: Path | None) -> dict[str, Any]:
    """Judge `name`'s trust fitted on the dev items for each form, its scores and defeats on dev and test, and the
    gate on each split."""
    judge = make_judge(name)
    if not isinstance(judge, SentenceJudge):
        msg = f"judge {name} does not read sentences, so the claim check cannot use it"
        raise ValueError(msg)  # noqa: TRY004 - a judge name it cannot use, as the command line refuses it
    named = people(items)
    own, others = asked(items, named)
    labels = {s: [i.strengthens for i in items if i.split == s] for s in ("dev", "test")}
    bar = judge.threshold
    out: dict[str, Any] = {"trust": judge.trust, "check_trust": judge.check_trust, "threshold": bar,
                           "reads_contradiction": judge.reads_contradiction, "forms": {}}
    embedder = EmbeddingRetriever()
    by_item: dict[str, dict[str, Opinion]] = {}
    started = time.perf_counter()
    try:
        for form, reader in FORMS.items():
            where = None if cache is None else cache / f"{name}.{form.replace(' ', '-')}.jsonl"
            found = stances(judge, [*own, *others], where, reader)
            read_own, read_others = found[:len(own)], found[len(own):]
            by_item[form] = {i.id: o for i, o in zip(items, read_own, strict=True)}
            split = {s: [o for o, i in zip(read_own, items, strict=True) if i.split == s] for s in ("dev", "test")}
            trust = fit_trust([o.with_base_rate(rate) for o in split["dev"]], labels["dev"])
            result: dict[str, Any] = {"trust": trust}
            for s in ("dev", "test"):
                opinions, ys = split[s], labels[s]
                result[s] = {
                    "rows": len(opinions),
                    "at_trust": scored(opinions, ys, rate, trust),
                    "at_base_rate": scored(opinions, ys, rate, 0.0),
                    "auc": roc_auc([o.belief - o.disbelief for o in opinions], ys),
                    "defeat": {"weakener": defeat_rate([o for o, y in zip(opinions, ys, strict=True) if not y], bar),
                               "strengthener": defeat_rate([o for o, y in zip(opinions, ys, strict=True) if y], bar)},
                    **precision_recall(opinions, ys, bar),
                }
            result["test"]["defeat"]["other_same_person"] = defeat_rate(read_others, bar)
            out["forms"][form] = result
        out["gate"] = {s: gate(s, items, named, by_item, judge, embedder, cache) for s in ("dev", "test")}
    finally:
        judge.close()
        embedder.close()
    out["seconds"] = round(time.perf_counter() - started, 1)
    out["exception_trust"] = out["forms"][TAKEN]["trust"]
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--judges", default="nli-deberta-large,typical-qwen3-4b", help="comma-separated judge names")
    parser.add_argument("--stances", type=Path, help="directory to keep each judge's stances in, and reuse them from")
    args = parser.parse_args(argv)
    names = [n for n in args.judges.split(",") if n]
    unknown = sorted(set(names) - set(NAMES))
    if unknown:
        parser.error(f"unknown judges {unknown}; known: {', '.join(NAMES)}")
    state = provenance(INPUTS)
    items = pairs.delta_atomic_items()
    dev = [i for i in items if i.split == "dev"]
    rate = sum(i.strengthens for i in dev) / len(dev)
    previous = json.loads(OUT.read_text()) if OUT.is_file() else {}
    measured = {k: v for k, v in previous.get("judges", {}).items() if k in NAMES}
    # The file keeps the judges earlier runs measured, so each judge carries the provenance of the run measuring it.
    for name in names:
        measured[name] = {**measure(name, items, rate, args.stances), "provenance": state}
        taken = measured[name]["forms"][TAKEN]
        print(f"{name}: exception trust {taken['trust']:.4f}, test log loss {taken['test']['at_trust']['log_loss']:.4f}, "
              f"defeat precision {taken['test']['precision']}, recall {taken['test']['recall']:.3f}", flush=True)
    results = {
        "set": CHECK_SET,
        "base_rate": rate,
        "people": {"names": list(PEOPLE), "seed": SEED},
        "rule": "PersonX and PersonY named per item; the taken form reads the update with the premise against the "
                "inference, the rejected named conditional reads it against If premise, inference.; a strengthener's "
                "inference holds, a weakener's does not; each form's trust maximises the dev labels' likelihood under "
                "each stance's projection at the set's base rate after discounting by it; a stance whose disbelief "
                "reaches the judge's threshold defeats; the gate writes claims with letters and statements with names, "
                "at the deployed spread",
        "spread": SPREAD,
        "others": {"rows": OTHERS, "seed": SEED},
        "judges": dict(sorted(measured.items())),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1) + "\n")
    store({name: measured[name]["exception_trust"] for name in names}, rate, CALIBRATION, "exception_trust")
    print(f"wrote {OUT.relative_to(ROOT)} and {CALIBRATION.relative_to(ROOT)} for {', '.join(names)}")


if __name__ == "__main__":
    main()
