"""Purpose: reproduce pymetta 0.9.4 losing an answer in untabled evaluation when a constant is spelled
like a builtin (max), while tabled evaluation and a direct lookup of the same fact find it.

Run: .venv/bin/python repros/builtin_named_constant_untabled.py
The program states daughter_of(NAME, hope) and lets a Python op say that parent(x, y) follows from
daughter_of(y, x); the query asks for parent(X, NAME) within three steps. Expected: X = hope for every
NAME, tabled or not. Observed: untabled evaluation answers nothing when NAME is max or min and answers
hope for tom; tabled evaluation answers hope for all three.
"""
from metta import Expression, MeTTa, S
from metta.vocabularies import EffectClass

CLAUSES = """(= (holds $p $x $y $d) (fact $p $x $y))
(= (holds $p $x $y $d) (if (> $d 0) (let (rw $q $s) (impliers $p) (via $q $s $x $y (- $d 1))) (empty)))
(= (via $q same $x $y $d) (holds $q $x $y $d))
(= (via $q swap $x $y $d) (holds $q $y $x $d))"""


def impliers(p: object):
    """parent(x, y) follows from daughter_of(y, x)."""
    if str(p) == "parent":
        yield Expression((S.rw, S["daughter_of"], S["swap"]))


def answers(name: str, tabled: bool) -> tuple[list[str], list[str]]:
    """The rewrite query's answers and a direct lookup of the fact, for one spelling of the constant."""
    with MeTTa() as m:
        m.op(impliers, name="impliers", effect=EffectClass.nondeterministicReadOnly)
        if tabled:
            m.self.run("!(import! (context-space) (library lib_tabling))")
        m.self.run(f"(= (fact daughter_of {name} hope) True)\n{CLAUSES}" + ("\n!(tabled (holds $p $x $y $d))" if tabled else ""))
        via_rewrite = m.eval(f"(let True (holds parent $X {name} 3) $X)", inferences=1_000_000, timeout=30)
        direct = m.eval(f"(let True (fact daughter_of {name} $Y) $Y)")
        return sorted(map(str, via_rewrite)), sorted(map(str, direct))


if __name__ == "__main__":
    with MeTTa() as m:
        builtins = set(m.self.builtins())
    for name in ("tom", "max", "min"):
        for tabled in (False, True):
            got, direct = answers(name, tabled)
            print(f"{name:3} (builtin: {name in builtins!s:5}) tabled={tabled!s:5}: parent(X, {name}) -> {got}; daughter_of({name}, Y) -> {direct}")
