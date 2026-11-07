"""Purpose: reproduce symbol-use state crossing pymetta 0.9.4 contexts.

A first context uses `parent` before anything defines it, then closes. A second,
fresh MeTTa() defines `parent` before any use of its own, yet pymetta warns
"parent is defined or imported after already being used; earlier expressions
treat it as a plain symbol". The llms.txt guide says two contexts are
independent engines' worth of state in one process.

Run: .venv/bin/python repros/symbol_use_leaks_across_contexts.py MODE
MODE is none, rule (define a rule calling parent), eval (evaluate a call to
undefined parent) or tabled (table a rule calling parent).
Expected: no warning in any mode.
"""
import sys

from metta import MeTTa, parse


def main(mode: str) -> None:
    """Use `parent` undefined in a first context the way MODE says, then define and query it in a second."""
    if mode != "none":
        with MeTTa() as first:
            if mode == "tabled":
                first.self.run("!(import! (context-space) (library lib_tabling))")
            first.self.run("(= (grandparent $X $Z) (let True (parent $X $Y) (parent $Y $Z)))")
            if mode == "tabled":
                first.self.run("!(tabled (grandparent $X $Y))")
            if mode == "eval":
                first.eval(parse("(let True (parent $X tom) (ans $X tom))"))
    with MeTTa() as second:
        print(f"[{mode}] defining parent in a fresh context", flush=True)
        second.self.run("(= (mother alice tom) True) (= (parent $X $Y) (mother $X $Y))")
        print(f"[{mode}] answers:", second.eval(parse("(let True (parent $X tom) (ans $X tom))")), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval")
