"""Purpose: reproduce run() spending far longer than eval() to exhaust the same
inference budget on pymetta 0.9.4.

Both evaluate the same untabled alias cycle, which never terminates, so each
call ends at its inference limit; the work is the same and only the door differs.

Run: .venv/bin/python repros/run_budget_slower_than_eval.py
Expected: comparable seconds per budget. Observed: run() many times slower and growing faster than the budget.
"""
import time

from metta import MeTTa
from metta.errors import InferenceLimitError

PROGRAM = "(= (mother alice tom) True) (= (mother $x $y) (mum $x $y)) (= (mum $x $y) (mother $x $y))"


def seconds(door: str, budget: int) -> float:
    """Seconds one door takes to reach `budget` inferences on the alias cycle."""
    with MeTTa() as m:
        m.self.run(PROGRAM)
        started = time.perf_counter()
        try:
            if door == "run":
                m.run("!(let True (mother $x tom) $x)", inferences=budget)
            else:
                m.eval("(let True (mother $x tom) $x)", inferences=budget)
        except InferenceLimitError:
            pass
        return time.perf_counter() - started


if __name__ == "__main__":
    for budget in (10_000, 25_000, 50_000, 100_000):
        e, r = seconds("eval", budget), seconds("run", budget)
        print(f"budget {budget:>7}: eval {e:7.3f} s   run {r:7.3f} s   ratio {r / e:6.1f}", flush=True)
