"""Purpose: the parent-of-Tom example end to end: a knowledge base that knows Tom's
mum and dad, asked for Tom's parents and grandparents, with the rewrites a judge
discovers, the answers with their derivations, and what happens without tables.

Run: python examples/parent_of_tom.py [--judge NAME] [--engine python|pymetta]
NAME is any judge semrewrite.judges.make takes (default nli-deberta-large), or
oracle for the gold lines in parent_of_tom.kb, which needs no model at all.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from semrewrite import Budget, Rewriter, formats
from semrewrite.engines import make as make_engine
from semrewrite.judges.oracle import OracleJudge
from semrewrite.retrieve import AllCandidates
from semrewrite.rewriter import STRATEGIES

KB_PATH = Path(__file__).with_name("parent_of_tom.kb")


def main() -> None:
    """Answer the knowledge base's queries under three strategies, print derivations, then the admitted rewrites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", default="nli-deberta-large")
    parser.add_argument("--engine", default="python")
    args = parser.parse_args()
    kb = formats.load(KB_PATH)
    if args.judge == "oracle":
        judge = OracleJudge(kb.entailed())
    else:
        from semrewrite.judges import make

        judge = make(args.judge)
    engine = make_engine(args.engine)
    rewriter = Rewriter(kb, judge, engine, AllCandidates(), budget=Budget(work=500_000, seconds=30))
    caps = engine.capabilities
    for query in kb.queries:
        print(f"\n{query}")
        for name in ("exact", "lazy-tabled", "naive"):
            if not all(getattr(caps, c) for c, need in STRATEGIES[name].needs().items() if need):
                print(f"  {name:12s} (the {engine.name} engine cannot run it)")
                continue
            o = rewriter.ask(query, name)
            answers = ", ".join(sorted(row[0] for row in o.answers)) or "nothing"
            print(f"  {name:12s} {o.status:11s} answers: {answers:20s} judge questions: {o.proposals}")
            if name == "lazy-tabled" and o.proofs:
                for row in sorted(o.proofs):
                    print("\n".join("      " + line for line in o.proofs[row].lines()))
    disc = rewriter.discoverer()
    print("\nrewrites the judge admits for each predicate, with scores:")
    for goal in sorted(kb.glosses):
        for rw in disc.rewrites(goal):
            print(f"  {rw}   {rw.score:.3f}")
    close = getattr(judge, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
