"""Purpose: the parent-of-Tom example run end to end: a knowledge base that knows
Tom's mum and dad, asked for Tom's parents, with the rewrites a judge discovers,
the answers, and what happens to the alias loop with and without tables.

Run: .venv/bin/python examples/parent_of_tom.py [judge]
`judge` is any name semrewrite.judges.make takes (default nli-deberta-large),
or oracle for the gold implications.
"""

from __future__ import annotations

import sys

from semrewrite.engine import STRATEGIES, Budget, Rewriter
from semrewrite.judging import OracleJudge
from semrewrite.kb import Query, from_metta, implication_closure

SOURCE = """
(gloss mum "{x} is the mum of {y}")
(gloss mother "{x} is the mother of {y}")
(gloss dad "{x} is the dad of {y}")
(gloss parent "{x} is a parent of {y}")
(gloss child_of "{x} is a child of {y}")
(gloss guardian "{x} is the legal guardian of {y}")
(gloss grandparent "{x} is a grandparent of {y}")
(= (mum alice tom) True)
(= (dad bob tom) True)
(= (mother carol alice) True)
(= (child_of bob dave) True)
(= (guardian erin tom) True)
(= (grandparent $X $Z) (let True (parent $X $Y) (parent $Y $Z)))
"""

GOLD = {("mum", "mother", False), ("mother", "mum", False), ("mother", "parent", False),
        ("dad", "parent", False), ("child_of", "parent", True)}


def main(judge_name: str) -> None:
    """Load the MeTTa source, discover rewrites with `judge_name`, and answer the parent and grandparent questions."""
    kb = from_metta(SOURCE, "parent-of-tom")
    if judge_name == "oracle":
        judge = OracleJudge.of(implication_closure(GOLD))
    else:
        from semrewrite.judges import make

        judge = make(judge_name)
    rewriter = Rewriter(kb, judge, None, budget=Budget(inferences=500_000, seconds=30))
    for query in (Query("parent", "X", "tom"), Query("grandparent", "X", "tom")):
        print(f"\n{query}")
        for name in ("exact", "lazy-tabled", "naive"):
            o = rewriter.ask(STRATEGIES[name], query)
            answers = ", ".join(sorted(a for a, _ in o.answers)) or "nothing"
            print(f"  {name:12s} {o.status:15s} answers: {answers:20s} judge proposals: {o.proposals}")
    disc = rewriter.discoverer()
    print("\nrewrites the judge admits for each predicate (premise => conclusion, score):")
    for goal in sorted(kb.glosses):
        for rw in disc.rewrites(goal):
            order = "(y, x)" if rw.swapped else "(x, y)"
            print(f"  {rw.premise}{order} => {goal}(x, y)   {rw.score:.3f}")
    close = getattr(judge, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nli-deberta-large")
