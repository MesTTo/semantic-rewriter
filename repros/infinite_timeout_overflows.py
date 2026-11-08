"""Purpose: reproduce pymetta 0.9.4 failing an evaluation given timeout=float("inf"): MeTTa.eval
takes `timeout: float | None`, rejects 0 with a ValueError, but passes an infinite timeout on to
SWI-Prolog, whose arithmetic raises float_overflow, so a one-fact lookup ends in an EngineError.

Run: .venv/bin/python repros/infinite_timeout_overflows.py
Expected: timeout=inf answers like timeout=None (no limit), or is refused like timeout=0 is.
Observed: timeout=None and timeout=60 answer [b]; timeout=inf raises EngineError with
"Arithmetic: evaluation error: `float_overflow'".
"""
import math

from metta import MeTTa
from metta.errors import EngineError


def lookup(timeout: float | None) -> str:
    """The answers to a one-fact lookup under the given timeout, or the error it raises."""
    with MeTTa() as m:
        m.self.run("(= (fact p a b) True)")
        try:
            return str([str(x) for x in m.eval("(let True (fact p a $y) $y)", inferences=100_000, timeout=timeout)])
        except (EngineError, ValueError) as e:
            return f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"


if __name__ == "__main__":
    for timeout in (None, 60.0, math.inf, 0.0):
        print(f"timeout={timeout!r:6}: {lookup(timeout)}")
