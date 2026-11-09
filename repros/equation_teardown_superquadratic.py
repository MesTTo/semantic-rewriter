"""Purpose: reproduce pymetta 0.9.4 taking time growing faster than quadratically in n to drop n ground equations of
one function when a context closes, against near nothing for the same facts held as atoms.

Run: .venv/bin/python repros/equation_teardown_superquadratic.py
For n = 1,250, 2,500 and 5,000, a fresh context loads n random ground equations (= (fact p a b) True) of one function
and is closed, the close timed; then the same n facts are loaded as atoms (fact p a b) and the context closed.
Expected: closing takes time linear in n either way. Observed on a Ryzen 9 9950X under load, over two runs: the
equations took 0.49 s to drop at n = 1,250, 2.1 to 2.7 s at 2,500, 10.9 to 18.8 s at 5,000 and, in the first run,
105.9 s at 10,000, so doubling n multiplied the time by 4 to 9; Space.clear() on 5,000 of them took 23.3 s, so
emptying the space first does not avoid it. The atoms took 0.00 to 0.06 s.
"""
import random
import time

from metta import MeTTa


def close_seconds(lines: str) -> float:
    """Seconds to close a fresh context after loading `lines`."""
    m = MeTTa()
    m.__enter__()
    m.self.run(lines)
    started = time.perf_counter()
    m.__exit__(None, None, None)
    return time.perf_counter() - started


def main() -> None:
    rng = random.Random(1)
    for n in (1250, 2500, 5000):
        facts = [(f"p{rng.randrange(300)}", f"e{rng.randrange(n // 5)}", f"e{rng.randrange(n // 5)}") for _ in range(n)]
        equations = close_seconds("\n".join(f"(= (fact {p} {a} {b}) True)" for p, a, b in facts))
        atoms = close_seconds("\n".join(f"(fact {p} {a} {b})" for p, a, b in facts))
        print(f"n={n}: closing after {n} equations of one function took {equations:.2f} s, after {n} atoms {atoms:.2f} s",
              flush=True)


if __name__ == "__main__":
    main()
