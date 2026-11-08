"""Purpose: measure how big a permissive judge's closure gets when nothing cuts it
off: one hard query per benchmark under nli-deberta-base at its default
threshold and under the oracle, with the budget lifted to 200M inferences, and
write the work, tables and answer quality to results/blowup.json.

Tables guarantee the closure is finite (DESIGN.md, Termination); this measures
what finite costs once a judge admits rewrites between most predicate pairs.

Run: .venv/bin/python experiments/blowup.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from semrewrite.benchmarks import BENCHMARKS
from semrewrite.discovery import ScoreCache
from semrewrite.engines import make as make_engine
from semrewrite.judges import make
from semrewrite.judges.oracle import OracleJudge
from semrewrite.kb import atom
from semrewrite.protocols import Budget
from semrewrite.retrieve import EmbeddingRetriever
from semrewrite.rewriter import Rewriter

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("family", atom("parent", "X", "Y")),
    ("family", atom("ancestor", "X", "leo")),
    ("work", atom("superior_of", "X", "finn")),
    ("geography", atom("inside", "X", "palir")),
    ("medicine", atom("brings_about", "X", "numbness")),
)


def main() -> None:
    """Run every case under both judges with lazy-tabled and eager-worklist, and write the rows."""
    retriever = EmbeddingRetriever()
    engine = make_engine("pymetta")
    permissive = make("nli-deberta-base")
    rows = []
    for kb_name, query in CASES:
        kb = BENCHMARKS[kb_name]()
        gold = kb.answers(query)
        judges = (("nli-deberta-base", ScoreCache(permissive)), ("oracle", ScoreCache(OracleJudge(kb.entailed()))))
        for judge_name, scorer in judges:
            rewriter = Rewriter(kb, scorer, engine, retriever, k=8, budget=Budget(work=200_000_000, seconds=900))
            for strategy in ("lazy-tabled", "eager-worklist"):
                started = time.perf_counter()
                o = rewriter.ask(query, strategy)
                rows.append({"kb": kb_name, "query": str(query), "judge": judge_name, "strategy": strategy, "status": o.status,
                             "work": o.work, "tables": o.tables, "rewrites": len(o.admitted), "answers": len(o.answers),
                             "true": len(o.answers & gold), "gold": len(gold), "seconds": round(time.perf_counter() - started, 2)})
                print(json.dumps(rows[-1]), flush=True)
    permissive.close()
    (ROOT / "results" / "blowup.json").write_text(json.dumps(rows, indent=1))
    print("wrote results/blowup.json", flush=True)


if __name__ == "__main__":
    main()
