"""Purpose: reproduce pymetta 0.9.4 making every call of a function name cost more inferences, in every later
context of the process, for each earlier context that tabled a function of that name.

Run: .venv/bin/python repros/tabled_contexts_raise_inference_cost.py
An untabled evaluation of chain, a function that calls itself ten times through a Python op, is counted in a
fresh context, then again after each batch of 100 contexts that import lib_tabling, table a function also
named chain and call it. Expected: the same count every time, since each context is closed before the next.
Observed: 7,264 inferences fresh, then 9,383, 11,679 and 13,979, about 23 more per earlier tabled context.
Tabling a function of another name leaves the count unchanged, as do contexts that only import the library,
and abolish_all_tables after each tabled context does not undo the growth.
"""
from metta import MeTTa, S
from metta.vocabularies import EffectClass


def step(n: object):
    """The successor of a chain node, as the rewriter's discovery op answers a goal."""
    yield S[f"n{int(str(n)[1:]) + 1}"]


def measured() -> int:
    """Inferences one untabled ten-step chain costs in a fresh context."""
    with MeTTa() as m:
        m.op(step, name="next", effect=EffectClass.nondeterministicReadOnly)
        m.self.run("(= (chain $n $d) (if (> $d 0) (let $m (next $n) (chain $m (- $d 1))) $n))")
        with m.stats() as stats:
            m.eval("(chain n0 10)")
        return stats.inferences


def tabled_context() -> None:
    with MeTTa() as m:
        m.self.run("!(import! (context-space) (library lib_tabling))")
        m.self.run("(= (chain $n $d) (fact $n))\n(= (fact n0) True)\n!(tabled (chain $n $d))")
        m.eval("(chain $n 3)")


if __name__ == "__main__":
    print(f"fresh process: {measured()} inferences")
    for batch in range(1, 4):
        for _ in range(100):
            tabled_context()
        print(f"after {batch * 100} tabled contexts: {measured()} inferences")
