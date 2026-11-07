"""Purpose: measure how big a permissive judge's closure gets when nothing cuts it
off: one hard query per benchmark under nli-deberta-base at its default
threshold and under the oracle, with the budget lifted to 200M inferences, and
write the work, tables and answer quality to results/blowup.json.

Tables guarantee the closure is finite; this
measures what finite costs once a judge admits rewrites between most predicate
pairs.

Run: .venv/bin/python experiments/blowup.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from semrewrite.benchmarks import BENCHMARKS
from semrewrite.discovery import ScoreCache
from semrewrite.engine import STRATEGIES, Budget, Rewriter
from semrewrite.judges import make
from semrewrite.judging import OracleJudge
from semrewrite.kb import Query, implication_closure
from semrewrite.retrieve import Retriever

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("family", Query("parent", "X", "Y")),
    ("family", Query("ancestor", "X", "leo")),
    ("work", Query("superior_of", "X", "finn")),
    ("geography", Query("inside", "X", "palir")),
    ("medicine", Query("brings_about", "X", "numbness")),
)


def main() -> None:
    """Run every case under both judges with lazy-tabled and eager-worklist, and write the rows."""
    retriever = Retriever()
    permissive = make("nli-deberta-base")
    rows = []
    for kb_name, query in CASES:
        kb = BENCHMARKS[kb_name]()
        gold = kb.answers(query)
        judges = (("nli-deberta-base", ScoreCache(permissive)), ("oracle", ScoreCache(OracleJudge.of(implication_closure(kb.gold)))))
        for judge_name, scorer in judges:
            rewriter = Rewriter(kb, scorer, retriever, k=8, budget=Budget(inferences=200_000_000, seconds=900))
            for strategy in ("lazy-tabled", "eager-worklist"):
                started = time.perf_counter()
                o = rewriter.ask(STRATEGIES[strategy], query)
                rows.append({"kb": kb_name, "query": str(query), "judge": judge_name, "strategy": strategy, "status": o.status,
                             "inferences": o.inferences, "tables": o.tables, "rewrites": o.rewrites, "answers": len(o.answers),
                             "true": len(o.answers & gold), "gold": len(gold), "seconds": round(time.perf_counter() - started, 2)})
                print(json.dumps(rows[-1]), flush=True)
    permissive.close()
    (ROOT / "results" / "blowup.json").write_text(json.dumps(rows, indent=1))
    print("wrote results/blowup.json", flush=True)


if __name__ == "__main__":
    main()
