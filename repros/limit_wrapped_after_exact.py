"""Purpose: reproduce, deterministically, an inference limit reaching the caller as a plain
EngineError on pymetta 0.9.4, through this repository's PyMeTTa engine.

Run: python repros/limit_wrapped_after_exact.py
The exact strategy runs once, then the naive strategy runs out of budget inside
its `impliers` op. Expected: InferenceLimitError. Observed: EngineError
"Python InferenceLimitError in (impliers mother)" whose cause is a janus
PrologError. Run alone, naive raises InferenceLimitError. The engine reads such
an error as a work limit by its own inference count, so this script records
what pymetta raised before the engine reclassifies it.
"""

from metta import MeTTa
from metta.errors import EngineError, InferenceLimitError

from semrewrite import Budget, Rewriter, atom
from semrewrite.engines import make
from semrewrite.judges.oracle import OracleJudge
from semrewrite.kb import KB
from semrewrite.model import Rule

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
    rules=(Rule(atom("grandparent", "X", "Z"), (atom("parent", "X", "Y"), atom("parent", "Y", "Z"))),),
    gold=frozenset({("mother", "parent", False), ("father", "parent", False), ("mum", "mother", False),
                    ("mother", "mum", False), ("child_of", "parent", True)}),
)


def main() -> None:
    """Run exact, then record what naive's budget exhaustion raises inside pymetta and print its class."""
    seen: list[BaseException] = []
    original = MeTTa.eval

    def recording(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except EngineError as e:
            seen.append(e)
            raise

    MeTTa.eval = recording
    rewriter = Rewriter(KB_TINY, OracleJudge.of(KB_TINY.gold), make("pymetta"), None, budget=Budget(work=300_000, seconds=20))
    rewriter.ask(atom("parent", "X", "tom"), "exact")
    rewriter.ask(atom("parent", "X", "tom"), "naive")
    for e in seen:
        print(type(e).__name__, "isinstance(InferenceLimitError) =", isinstance(e, InferenceLimitError), "|", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
