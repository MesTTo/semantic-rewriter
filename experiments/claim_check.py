"""Purpose: show what the claim check does to the actions example: examples/actions.kb, eight people each
doing one thing and asked who wastes time, examples/actions_first_person.kb, the same actions stated in the first
person with no arguments and asked whether I waste time, and examples/actions_people.kb, the eight people with amy
also playing games and kim also watching TV. Each is asked as the rewriter always asks, checking every rewrite it
admits and every application its answers' lineage makes, without and with a belief stated: "I consider gaming to be
productive" in the first two, with no arguments, and "kim considers gaming to be productive" in the third, naming kim.
Each runs at the judge's calibrated rewrite threshold and at LOW, and it records each answer's opinion, the answers no
derivation is left for, each admitted rewrite's opinion with the statements read against it, each rewrite the check
withdrew with the statements contradicting it, each application read with its leaf and statements, the judge's check
and exception trusts and the check's cost in judge pairs and seconds.

Run from the repository root: python experiments/claim_check.py [--judge NAME ...]. It writes
results/claim_check_actions.json, which RESULTS.md's table reads, keeping the runs of judges it was not asked for.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semrewrite import formats
from semrewrite.claims import APPLICATIONS, CLAIMS, ClaimCheck
from semrewrite.engines import make as make_engine
from semrewrite.judges import default_data_trust, default_threshold
from semrewrite.judges import make as make_judge
from semrewrite.kb import KB
from semrewrite.lineage import SETS
from semrewrite.model import Atom, Proposal
from semrewrite.protocols import Budget
from semrewrite.retrieve import AllCandidates, EmbeddingRetriever
from semrewrite.rewriter import Rewriter

OUT = ROOT / "results" / "claim_check_actions.json"
# Each form's knowledge base and the belief stated in it: a gloss and the fact of that gloss.
GENERAL = (Atom("gaming_productive", ()), "I consider gaming to be productive")
FORMS = {"with holes": (ROOT / "examples" / "actions.kb", GENERAL),
         "first person": (ROOT / "examples" / "actions_first_person.kb", GENERAL),
         "per person": (ROOT / "examples" / "actions_people.kb",
                        (Atom("gaming_productive", ("kim",)), "{x} considers gaming to be productive"))}
# A rewrite threshold under the 0.039 nli-deberta-large believes "A plays games" => "A wastes time": at its calibrated
# 0.659 that judge admits watching TV's match alone, so kim is no answer and the belief has nothing to withdraw.
LOW = 0.03


def believing(kb: KB, belief: tuple[Atom, str]) -> KB:
    """`kb` with `belief` stated, as a user would add it: its gloss and its fact."""
    fact, gloss = belief
    return KB(kb.name, glosses={**kb.glosses, fact.pred: gloss}, facts=(*kb.facts, fact), rules=kb.rules,
              gold=kb.gold, queries=kb.queries, traps=kb.traps)


def run(judge, form: str, believed: bool, threshold: float, embedder: EmbeddingRetriever) -> dict:
    """One ask of `form`, `judge` admitting at `threshold`: its answers, its matches with the statements read against
    each, and the matches the check withdrew with the statements contradicting each. The check is the rewriter's
    default one, given the run's embedding model so the runs share it."""
    path, belief = FORMS[form]
    kb = formats.load(path)
    kb = believing(kb, belief) if believed else kb
    judge.threshold = threshold
    check = ClaimCheck(kb, judge, embedder)
    # No time limit, the work budget still bounding every run, as the conformance kit's must-finish runs have, so the
    # machine's load cannot decide a measurement: typical-qwen3-4b, its blocks streamed from disk under load, spent
    # the default 60 s expanding a predicate at 19:03 on 17 November (i-cc-pa-actions-time-budget).
    rewriter = Rewriter(kb, judge, make_engine("python"), AllCandidates(), beliefs=SETS,
                        data_trust=default_data_trust(judge.name), claims=check, budget=Budget(seconds=math.inf))
    out = rewriter.ask(kb.queries[0], "lazy-tabled")
    if not out.finished or out.beliefs is None or out.rewrite_beliefs is None or out.readings is None:
        msg = f"{form}: {out.status} {out.detail}"
        raise RuntimeError(msg)

    def opinion(o) -> dict[str, float]:
        return {"P": o.projection, "b": o.belief, "d": o.disbelief, "u": o.uncertainty}

    def score(premise: str, conclusion: str, m) -> float:
        return rewriter.scorer.score([Proposal(rewriter.catalogue[premise], rewriter.catalogue[conclusion], m)])[0]

    def statement(r) -> dict:
        return {"statement": r.statement, "similarity": r.similarity, "side": r.side, **opinion(r.opinion),
                "stance": opinion(r.stance)}

    return {
        "judge": judge.name, "form": form, "believed": believed, "threshold": threshold, "check_trust": check.reading(CLAIMS)[1],
        "exception_trust": check.reading(APPLICATIONS)[1],
        "answers": {",".join(row) or "()": opinion(o) for row, o in sorted(out.beliefs.items())},
        "fallen": [",".join(row) for row in sorted(out.fallen)],
        "matches": {f"{premise} => {conclusion}": {
            **opinion(out.rewrite_beliefs[(premise, conclusion, m)]), "mapping": list(m),
            "score": score(premise, conclusion, m), "claim": check.claim((premise, conclusion, m)),
            "read": [statement(r) for r in out.readings[(premise, conclusion, m)]]}
            for premise, conclusion, m in sorted(out.admitted)},
        "withdrawn": {f"{premise} => {conclusion}": {
            "mapping": list(m), "score": score(premise, conclusion, m), "claim": check.claim((premise, conclusion, m)),
            "contradicted_by": [statement(r) for r in out.defeated[(premise, conclusion, m)]]}
            for premise, conclusion, m in sorted(out.defeated)},
        "applications": {f"{premise} => {conclusion} ({','.join(row)})": {
            "defeated": app.defeated, "normal": opinion(app.normal), "premise": app.premise,
            "conclusion": app.conclusion, "read": [statement(r) for r in app.readings]}
            for ((premise, conclusion, _), row), app in sorted((out.applications or {}).items())},
        "cost": {"claims": check.usage["claims"], "applications": check.usage.get("applications", 0),
                 "pairs": check.usage["pairs"], "statements": check.usage["statements"], "seconds": check.seconds},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the actions example, without and with a belief about gaming.")
    parser.add_argument("--judge", nargs="+", default=["nli-deberta-large"])
    args = parser.parse_args()
    kept = [r for r in json.loads(OUT.read_text()) if r["judge"] not in args.judge] if OUT.is_file() else []
    embedder = EmbeddingRetriever()
    runs = []
    try:
        for judge_name in args.judge:
            judge = make_judge(judge_name)
            try:
                for form in FORMS:
                    for threshold in (default_threshold(judge_name), LOW):
                        for believed in (False, True):
                            runs.append(run(judge, form, believed, threshold, embedder))
                            print(json.dumps({k: runs[-1][k] for k in ("judge", "form", "believed", "threshold")}
                                             | {"answers": runs[-1]["answers"]}), flush=True)
            finally:
                judge.close()
    finally:
        embedder.close()
    OUT.write_text(json.dumps(kept + runs, indent=1) + "\n")


if __name__ == "__main__":
    main()
