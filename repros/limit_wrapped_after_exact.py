"""Purpose: reproduce, deterministically, an inference limit reaching the caller as a plain
EngineError on pymetta 0.9.4, through this repository's engine.

Run: .venv/bin/python repros/limit_wrapped_after_exact.py
The exact strategy runs once, then the naive strategy runs out of budget inside
its `impliers` op. Expected: InferenceLimitError. Observed: EngineError
"Python InferenceLimitError in (impliers mother)" whose cause is a janus
PrologError and whose remedy reads "report the ball with the message it
carries; this engine did". Run alone, naive raises InferenceLimitError.
"""
from metta import parse
from metta.errors import EngineError, InferenceLimitError

from semrewrite.engine import STRATEGIES, Budget, Rewriter
from semrewrite.judging import OracleJudge
from semrewrite.kb import KB, Lit, Query, Rule

KB_TINY = KB(
    name="tiny",
    glosses={
        "mother": "{x} is the mother of {y}", "mum": "{x} is the mum of {y}",
        "father": "{x} is the father of {y}", "parent": "{x} is a parent of {y}",
        "child_of": "{x} is a child of {y}", "guardian": "{x} is the legal guardian of {y}",
        "grandparent": "{x} is a grandparent of {y}",
    },
    facts=(("mother", "alice", "tom"), ("father", "bob", "tom"), ("mum", "carol", "alice"),
           ("child_of", "bob", "dave"), ("guardian", "erin", "tom")),
    rules=(Rule(Lit("grandparent", "X", "Z"), (Lit("parent", "X", "Y"), Lit("parent", "Y", "Z"))),),
    gold=frozenset({("mother", "parent", False), ("father", "parent", False), ("mum", "mother", False),
                    ("mother", "mum", False), ("child_of", "parent", True)}),
)


def main() -> None:
    """Run exact, then catch what naive's budget exhaustion raises and print its class."""
    rewriter = Rewriter(KB_TINY, OracleJudge.of(KB_TINY.gold), None, budget=Budget(inferences=300_000, seconds=20))
    rewriter.ask(STRATEGIES["exact"], Query("parent", "X", "tom"))
    seen: list[BaseException] = []

    def capture(self, m, term, out, accumulate=False):
        try:
            m.eval(parse(term), inferences=self.budget.inferences, timeout=self.budget.seconds)
        except EngineError as e:
            seen.append(e)

    Rewriter._evaluate = capture
    rewriter.ask(STRATEGIES["naive"], Query("parent", "X", "tom"))
    for e in seen:
        print(type(e).__name__, "isinstance(InferenceLimitError) =", isinstance(e, InferenceLimitError), "|", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
