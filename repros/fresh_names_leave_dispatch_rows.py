"""Purpose: reproduce pymetta 0.9.4 keeping a row of its dispatch cache, and about 0.9 KiB, for every function name a
MeTTa() context defined, after the context closes, without lib_tabling.

Run: .venv/bin/python repros/fresh_names_leave_dispatch_rows.py [same]
2,000 contexts are opened and closed one after another; each defines 10 functions of 5 equations and evaluates each
once. The names are new in every context, or the same in all of them with `same`. Expected: flat resident memory and
a dispatch cache that stops growing, since each context is closed before the next opens. Observed over two runs:
with new names, 8.3 to 9.4 KiB kept a context and 20,000 rows of spaces:metta_dispatch_value_cache/4 after 2,000
contexts, one for each name defined; with the same names, nothing kept and 10 rows.
"""
import gc
import sys

from metta import MeTTa

NAMES = 10


def rss_kib() -> int:
    with open("/proc/self/status") as f:
        return next(int(line.split()[1]) for line in f if line.startswith("VmRSS:"))


def cache_rows() -> int:
    """Rows of pymetta's process-wide dispatch cache."""
    with MeTTa() as m:
        return int(m.runtime.once("aggregate_all(count, spaces:metta_dispatch_value_cache(_, _, _, _), N)")["N"])


def context(names: list[str]) -> None:
    """One context: five equations for each name, each name's function evaluated once."""
    with MeTTa() as m:
        m.self.run(" ".join(f"(= ({n} a{j}) b{j})" for n in names for j in range(5)))
        for n in names:
            assert len(m.eval(f"({n} $x)")) == 5


if __name__ == "__main__":
    same = sys.argv[1:] == ["same"]

    def names(i: int) -> list[str]:
        return [f"f{k}" if same else f"f{i}_{k}" for k in range(NAMES)]

    for i in range(1, 101):
        context(names(i))
    gc.collect()
    start = rss_kib()
    for i in range(101, 2001):
        context(names(i))
        if i % 500 == 0:
            gc.collect()
            print(f"{i} contexts: {(rss_kib() - start) / (i - 100):.1f} KiB kept a context since the 100th, "
                  f"{cache_rows()} dispatch cache rows")
