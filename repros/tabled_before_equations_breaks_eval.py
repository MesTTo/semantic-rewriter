"""Purpose: reproduce pymetta 0.9.4 raising an internal type error when a function is declared tabled before its
equations are loaded, and the same error in every later context that tables a function of that name.

Run: .venv/bin/python repros/tabled_before_equations_breaks_eval.py
Three contexts in one process load the same two functions, p0 and p1 calling p0, and table both: the first loads the
equations before the declarations, the second declares first, the third loads the equations first again.
Expected: the answer c0 in all three. Observed: [(ans c0)] in the first; in the second, the query raises
EngineError("Type error: `metta_function_call' expected, found `[p1,_,_]' (a list)"); in the third, a fresh context
written as the first was, declaring p1 tabled raises the same error. Functions of other names still table.
"""
from metta import MeTTa, parse
from metta.errors import EngineError

EQUATIONS = "(= (p0 c0 c1) True)\n(= (p1 $X0 $X1) (p0 $X0 $X1))"
DECLARATIONS = ("!(tabled (p0 $X0 $X1))", "!(tabled (p1 $X0 $X1))")


def context(declare_first: bool) -> str:
    """Load the two functions and their tabling declarations in the order given, then ask p1 for what reaches c1."""
    with MeTTa() as m:
        m.self.run("!(import! (context-space) (library lib_tabling))")
        try:
            if declare_first:
                for d in DECLARATIONS:
                    m.self.run(d)
                m.self.run(EQUATIONS)
            else:
                m.self.run(EQUATIONS)
                for d in DECLARATIONS:
                    m.self.run(d)
            return str(m.eval(parse("(let True (p1 $x c1) (ans $x))")))
        except EngineError as e:
            return f"EngineError({str(e).splitlines()[0]!r})"


if __name__ == "__main__":
    for label, declare_first in (("equations first", False), ("declared first", True), ("equations first", False)):
        print(f"{label}: {context(declare_first)}")
