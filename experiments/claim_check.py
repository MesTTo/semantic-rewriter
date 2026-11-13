"""Purpose: show what the claim check does to the actions example: examples/actions.kb, eight people each
doing one thing and asked who wastes time, and examples/actions_first_person.kb, the same actions stated in the first
person with no arguments and asked whether I waste time. Each is asked with the check on, without and with "I consider
gaming to be productive" stated, at the judge's calibrated rewrite threshold and at LOW, and it records each answer's
opinion, each admitted rewrite's opinion, the statements read against each and the check's cost in judge pairs and
seconds.

Run from the repository root: python experiments/claim_check.py [--judge NAME ...]. It writes
results/claim_check_actions.json, which RESULTS.md's table reads, keeping the runs of judges it was not asked for.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semrewrite import formats
from semrewrite.claims import ClaimCheck
from semrewrite.engines import make as make_engine
from semrewrite.judges import default_data_trust, default_threshold
from semrewrite.judges import make as make_judge
from semrewrite.kb import KB
from semrewrite.lineage import SETS
from semrewrite.model import Atom, Proposal
from semrewrite.retrieve import AllCandidates, EmbeddingRetriever
from semrewrite.rewriter import Rewriter

OUT = ROOT / "results" / "claim_check_actions.json"
FORMS = {"with holes": ROOT / "examples" / "actions.kb", "first person": ROOT / "examples" / "actions_first_person.kb"}
BELIEF = ("gaming_productive", "I consider gaming to be productive")
# A rewrite threshold under the 0.039 nli-deberta-large believes "A plays games" => "A wastes time": at its calibrated
# 0.659 that judge admits watching TV's match alone, so kim is no answer and the belief has nothing to withdraw.
LOW = 0.03


def believing(kb: KB) -> KB:
    """`kb` with the gaming belief stated, as a user would add it: a gloss and a fact with no arguments."""
    name, gloss = BELIEF
    return KB(kb.name, glosses={**kb.glosses, name: gloss}, facts=(*kb.facts, Atom(name, ())), rules=kb.rules,
              gold=kb.gold, queries=kb.queries, traps=kb.traps)


def run(judge, form: str, believed: bool, threshold: float, embedder: EmbeddingRetriever) -> dict:
    """One ask of `form` with the check on, `judge` admitting at `threshold`: its answers, its matches and the
    statements read against each."""
    kb = formats.load(FORMS[form])
    kb = believing(kb) if believed else kb
    judge.threshold = threshold
    check = ClaimCheck(kb, judge, embedder)
    rewriter = Rewriter(kb, judge, make_engine("python"), AllCandidates(), beliefs=SETS,
                        data_trust=default_data_trust(judge.name), claims=check)
    out = rewriter.ask(kb.queries[0], "lazy-tabled")
    if not out.finished or out.beliefs is None or out.rewrite_beliefs is None or out.readings is None:
        msg = f"{form}: {out.status} {out.detail}"
        raise RuntimeError(msg)

    def opinion(o) -> dict[str, float]:
        return {"P": o.projection, "b": o.belief, "d": o.disbelief, "u": o.uncertainty}

    return {
        "judge": judge.name, "form": form, "believed": believed, "threshold": threshold,
        "answers": {",".join(row) or "()": opinion(o) for row, o in sorted(out.beliefs.items())},
        "matches": {f"{premise} => {conclusion}": {
            **opinion(out.rewrite_beliefs[(premise, conclusion, m)]), "mapping": list(m),
            "score": rewriter.scorer.score([Proposal(rewriter.catalogue[premise], rewriter.catalogue[conclusion], m)])[0],
            "claim": check.claim((premise, conclusion, m)),
            "read": [{"statement": r.statement, "similarity": r.similarity, "side": r.side, **opinion(r.opinion)}
                     for r in out.readings[(premise, conclusion, m)]]}
            for premise, conclusion, m in sorted(out.admitted)},
        "cost": {"claims": check.usage["claims"], "pairs": check.usage["pairs"], "statements": check.usage["statements"],
                 "seconds": check.seconds},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the actions example with the claim check on.")
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
