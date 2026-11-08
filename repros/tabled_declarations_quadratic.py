"""Purpose: reproduce pymetta 0.9.4 taking time quadratic in n to declare n tabled functions that call each other,
and near linear once each function also has a clause calling a Python op.

Run: .venv/bin/python repros/tabled_declarations_quadratic.py
For n = 30, 60, 120 and 240, a fresh context defines a chain of n binary functions, each with four facts and each
calling the one before it, and declares every function tabled; the declarations are timed, once as is and once with
an extra clause on each function that calls an op yielding nothing. Expected: time linear in n either way.
Observed on a Ryzen 9 9950X over two runs: as is, 0.06 to 0.08 s at n = 30 and 2.5 to 2.6 s at n = 240, a
declaration taking 2 to 11 ms as n grows; with the op clause, 0.04 to 0.05 s and 0.56 to 0.69 s, 1.2 to 2.9 ms.
"""
import time

from metta import MeTTa
from metta.vocabularies import EffectClass


def nothing(p):
    """An op that answers nothing."""
    yield from ()


def declarations(n: int, op_clause: bool) -> float:
    """Seconds to declare tabled every function of a chain of n, in a fresh context."""
    lines = []
    for i in range(n):
        lines += [f"(= (p{i} c{j} c{j + 1}) True)" for j in range(4)]
        if i:
            lines.append(f"(= (p{i} $X0 $X1) (p{i - 1} $X0 $X1))")
        if op_clause:
            lines.append(f'(= (p{i} $X0 $X1) (nothing "p{i}"))')
    with MeTTa() as m:
        m.op(nothing, name="nothing", effect=EffectClass.nondeterministicReadOnly)
        m.self.run("!(import! (context-space) (library lib_tabling))")
        m.self.run("\n".join(lines))
        started = time.perf_counter()
        for i in range(n):
            assert m.self.run(f"!(tabled (p{i} $X0 $X1))") == [[True]]
        return time.perf_counter() - started


if __name__ == "__main__":
    for n in (30, 60, 120, 240):
        plain, with_op = declarations(n, False), declarations(n, True)
        print(f"n {n:3}: {plain:.2f} s as is ({plain / n * 1000:.1f} ms each), {with_op:.2f} s with the op clause "
              f"({with_op / n * 1000:.1f} ms each)")
