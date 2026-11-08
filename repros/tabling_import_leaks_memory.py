"""Purpose: reproduce pymetta 0.9.4 keeping about 25 KB for every MeTTa() context that imports lib_tabling, after the
context closes.

Run: .venv/bin/python repros/tabling_import_leaks_memory.py [plain]
2,000 contexts are opened and closed one after another; each imports lib_tabling (unless `plain` is given), defines
50 equations and evaluates a query over them. Expected: the same peak resident memory after 500 contexts as after
2,000, since each context is closed before the next opens. Observed over two runs: with the import, 97 to 98 MiB
after 500 contexts, 120 after 1,500 and 133 to 140 after 2,000; without it, 98 to 100 MiB throughout. Nothing needs
to be tabled, the names do not matter, and abolish_all_tables before closing does not free it.
"""
import resource
import sys

from metta import MeTTa


def context(importing: bool) -> None:
    """One context: the tabling library imported or not, 50 equations, one query over them."""
    with MeTTa() as m:
        if importing:
            m.self.run("!(import! (context-space) (library lib_tabling))")
        m.self.run(" ".join(f"(= (f a{j}) b{j})" for j in range(50)))
        assert len(m.eval("(f $x)")) == 50


if __name__ == "__main__":
    importing = sys.argv[1:] != ["plain"]
    for i in range(1, 2001):
        context(importing)
        if i % 500 == 0:
            print(f"{i} contexts: max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB")
