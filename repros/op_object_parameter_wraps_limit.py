"""Purpose: reproduce pymetta 0.9.4 raising an inference limit inside a Python op as an EngineError, and charging more
than twice the inferences a call, when the op's parameter is annotated object.

Run: .venv/bin/python repros/op_object_parameter_wraps_limit.py
An endless untabled loop through an op is evaluated under inferences=300_000 in three fresh contexts of one process,
with the op's parameter unannotated and then annotated object, and a chain of 2,000 calls through each op is counted.
Expected: InferenceLimitError in every context, and the same inferences a call either way.
Observed: unannotated, InferenceLimitError in all three contexts and 33 inferences a call. Annotated, InferenceLimitError
in the first context, then EngineError("Python InferenceLimitError in (next n)") in the second and third, and 77
inferences a call.
"""
from metta import MeTTa, S, parse
from metta.errors import EngineError, InferenceLimitError
from metta.vocabularies import EffectClass


def unannotated(n):
    """The next node of an endless chain."""
    yield S["n"]


def annotated(n: object):
    """The same op, its parameter annotated."""
    yield S["n"]


def limit_error(op) -> str:
    """Which exception ends an endless loop through op under 300,000 inferences, in a fresh context."""
    with MeTTa() as m:
        m.op(op, name="next", effect=EffectClass.nondeterministicReadOnly)
        m.self.run("(= (loop $n) (let $m (next $n) (loop $m)))")
        try:
            m.eval(parse("(loop n)"), inferences=300_000)
        except InferenceLimitError:
            return "InferenceLimitError"
        except EngineError as e:
            return f"EngineError({str(e).splitlines()[0]!r})"
        return "no error"


def per_call(op) -> float:
    """Inferences a call of op costs, over a chain of 2,000 calls."""
    with MeTTa() as m:
        m.op(op, name="next", effect=EffectClass.nondeterministicReadOnly)
        m.self.run("(= (chain $n $d) (if (> $d 0) (let $m (next $n) (chain $m (- $d 1))) $n))")
        with m.stats() as base:
            m.eval(parse("(chain n 0)"))
        with m.stats() as stats:
            m.eval(parse("(chain n 2000)"))
        return (stats.inferences - base.inferences) / 2000


if __name__ == "__main__":
    for op in (unannotated, annotated):
        errors = ", ".join(limit_error(op) for _ in range(3))
        print(f"{op.__name__}: {errors}; {per_call(op):.0f} inferences a call")
