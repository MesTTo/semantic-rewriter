"""Purpose: reproduce pymetta 0.9.4's inference limit letting an evaluation through a function of several equations
count far more inferences, by stats(), than the limit before it stops.

Run: .venv/bin/python repros/nondeterministic_eval_overruns_limit.py
Three endless loops, none of which yields an answer, are stopped by limits of 10,000, 100,000 and 300,000 inferences,
and stats() counts each stopped evaluation. Expected: about the limit each time, since pymetta's control.pl bounds an
evaluation's spend by its budget plus one answer's cost. Observed: the loop of one equation counts 309 inferences
past every limit; the loop of two equations and the loop through a let over a three-equation call count 13 to 33
percent past it, more as the limit grows.
"""
from metta import MeTTa, parse
from metta.errors import InferenceLimitError

LOOPS = {
    "one equation": "(= (w $d) (w (+ $d 1)))",
    "two equations": "(= (w $d) (w (+ $d 1))) (= (w $d) stop)",
    "let over three": "(= (pick) a) (= (pick) b) (= (pick) c) (= (w $d) (let $x (pick) (w (+ $d 1))))",
}
LIMITS = (10_000, 100_000, 300_000)


def counted(loop: str, limit: int) -> int:
    """Inferences stats() counts for (w 0) stopped by the inference limit, in a fresh context."""
    with MeTTa() as m:
        m.self.run(loop)
        goal = parse("(w 0)")
        with m.stats() as stats:
            try:
                m.eval(goal, inferences=limit)
            except InferenceLimitError:
                pass
            else:
                raise AssertionError("the loop finished")
        return stats.inferences


if __name__ == "__main__":
    counted(LOOPS["one equation"], LIMITS[0])  # a process's first evaluation also counts one-time setup
    for name, loop in LOOPS.items():
        counts = [counted(loop, limit) for limit in LIMITS]
        shown = (f"limit {limit}: {count} ({count - limit:+})" for limit, count in zip(LIMITS, counts, strict=True))
        print(f"{name:14} " + ", ".join(shown))
