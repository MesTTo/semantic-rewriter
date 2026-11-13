"""Purpose: compare how SNARS ranks a Wikidata question's answers with how the rewriter ranks them, and find on a
constructed chain what makes the two part. The rewriter weighs each answer by its exact lineage, the question's own
rules certain and a rewrite used twice in a derivation counting once. SNARS, given the knowledge base's statements,
the question's own rules and the admitted rewrites as semrewrite.snars writes them, proves every answer by modus
ponens at each step and fuses each answer's evidence-disjoint proofs (mettabase's sn prove and sn fuse-proofs).
SNARS does all of its own proving and fusing; this script only routes its proofs to the answers they conclude.

Run from the repository root with PETTA_PATH and METTABASE_PATH set, as tests/test_snars.py needs them:

    python experiments/snars_ranking.py leonardo alphabet chain

The questions write results/snars_ranking.json and print each ranking's average precision against Wikidata's own
answer and how far the two rankings agree. `chain` writes results/snars_chain.json: SNARS's P along a chain that uses
one rewrite at every link, with SNARS's rule rows at their structural truth or certain, and the rewrite's validity
at several opinions.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semrewrite.judges import default_data_trust
from semrewrite.judges import make as make_judge
from semrewrite.model import Atom, Rewrite, Rule, is_variable
from semrewrite.opinion import Opinion
from semrewrite.protocols import Budget
from semrewrite.rewriter import Rewriter
from semrewrite.snars import believed, rule_rows, statement

RESULTS = ROOT / "results" / "snars_ranking.json"
CHAIN = ROOT / "results" / "snars_chain.json"
# Each question's judge, rewrite threshold and SNARS proof depth: Jev at the low threshold whose answers RESULTS.md
# ranks, one rewrite on a path as there. Leonardo's answers each need one rewrite, so depth 1 reaches them all;
# Alphabet's chain ownership through its transitive rule, and depth 3 reaches 2,894 of its 2,903 in about three
# minutes, where depth 2 reached 376.
RUNS = {"leonardo": ("jev", 0.1, 1), "alphabet": ("jev", 0.1, 3)}
# The chain's arms. SNARS's backward chainer applies every rule row at one structural truth, (sn struct-tv), belief
# 0.9; these two lines, run once SNARS is loaded, make it certain instead. The rewrite's validity is certain, or an
# opinion at Wikidata's base rate whose P, 0.95 or 0.24, lies near the top or the bottom of the rewriter's P on
# Leonardo's question.
CERTAIN_RULES = ("!(match &self (= (sn struct-tv) $o) (remove-atom &self (= (sn struct-tv) $o)))",
                 "!(add-atom &self (= (sn struct-tv) (Opinion 1.0 0.0 0.0 0.5)))")
VALIDITIES = {"certain": Opinion(1.0, 0.0, 0.0), "P 0.95": Opinion(0.94, 0.0, 0.06, 0.214),
              "P 0.24": Opinion(0.2, 0.6, 0.2, 0.214)}


def _evaluate():
    """examples/wikidata/evaluate.py as a module, for its questions, tools and strategy."""
    spec = importlib.util.spec_from_file_location("wikidata_evaluate", ROOT / "examples" / "wikidata" / "evaluate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rule_lines(index: int, rule: Rule) -> list[str]:
    """A question's own rule in SNARS's world: a two-premise rule as a rule-kit row, a one-premise rule over distinct
    variables as a rewrite certainly valid."""
    if len(rule.body) == 2:
        first, second = rule.body
        return [(f"(sn rw rule-{index} ded-shape {statement(first.pred, first.args)} "
                 f"{statement(second.pred, second.args)} {statement(rule.head.pred, rule.head.args)})")]
    (premise,) = rule.body
    if len(set(premise.args)) == len(premise.args) and set(rule.head.args) <= set(premise.args):
        mapping = tuple(premise.args.index(v) for v in rule.head.args)
        return list(rule_rows(Rewrite(rule.head.pred, premise.pred, mapping), Opinion(1.0, 0.0, 0.0), "question"))
    msg = f"SNARS's rule kit has no form for {rule}"
    raise ValueError(msg)


# --- reading PeTTa's output ----------------------------------------------------------------------------------------
_TOKEN = re.compile(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+')


def _parse(text: str) -> list:
    """Every top-level S-expression in `text`, lists as Python lists and strings kept with their quotes."""
    stack: list[list] = [[]]
    for token in _TOKEN.findall(text):
        if token == "(":
            stack.append([])
        elif token == ")":
            if len(stack) > 1:
                done = stack.pop()
                stack[-1].append(done)
        else:
            stack[-1].append(token)
    return stack[0]


def _show(expr) -> str:
    """An S-expression back as MeTTa text."""
    return f"({' '.join(_show(e) for e in expr)})" if isinstance(expr, list) else expr


def _proofs(exprs: list) -> list[list]:
    """Every (: (Opinion b d u a) proof conclusion) triple among `exprs`, at any depth."""
    out = []
    for e in exprs:
        if isinstance(e, list):
            if len(e) == 4 and e[0] == ":" and isinstance(e[1], list) and e[1][:1] == ["Opinion"]:
                out.append(e)
            else:
                out.extend(_proofs(e))
    return out


def _filler(tree: list, role: str) -> str:
    """The concept filling `role` in a statement's tree."""
    (_, _, _, args) = tree
    for _, r, child in args[1]:
        if r == role:
            return json.loads(child[2])
    msg = f"no {role} argument in {_show(tree)}"
    raise ValueError(msg)


def _petta(program: str) -> str:
    """PeTTa's standard output on `program`, run as tests/test_snars.py runs it."""
    petta, mettabase = Path(os.environ["PETTA_PATH"]), Path(os.environ["METTABASE_PATH"])
    path = [str(mettabase / "src"), str(mettabase / "lib"), str(petta / "python"), os.environ.get("PYTHONPATH", "")]
    result = subprocess.run(
        [shutil.which("swipl") or "swipl", "--stack_limit=16g", "-q", "-s", petta / "src" / "main.pl", "--",
         "/dev/stdin", "silent"],
        input=program, cwd=mettabase, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in path if p)},
    )
    if result.returncode != 0:
        msg = f"PeTTa exited {result.returncode}: {result.stderr[-2000:]}"
        raise RuntimeError(msg)
    return result.stdout


def _import_snars() -> str:
    """The line importing METTABASE_PATH's SNARS by absolute path, which PeTTa resolves whether mettabase's imports are
    relative to the repository (its main, on PeTTa d8d4692) or to each file (GoalChainer's line, on import-overhaul)."""
    return f'!(import! &self "{Path(os.environ["METTABASE_PATH"]) / "lib" / "mettabase" / "snars"}")'


def _applications(proof) -> int:
    """The rule applications in a proof term: a leaf, a statement's sid, has none."""
    return 1 + sum(_applications(arg) for arg in proof[1:]) if isinstance(proof, list) else 0


# --- the comparison ------------------------------------------------------------------------------------------------
def _average_precision(scores: dict[str, float], truth: set[str]) -> float:
    """Average precision of the answers ranked by score, highest first, ties broken by name, against `truth`."""
    hits, total = 0, 0.0
    for k, (answer, _) in enumerate(sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])), start=1):
        if answer in truth:
            hits += 1
            total += hits / k
    return total / len(truth)


# Time: n^2 / 2 comparisons, n = the answers both rank.
def _kendall(a: dict[str, float], b: dict[str, float]) -> tuple[float, int, int]:
    """Kendall's tau-b between two scorings of the same answers, with the pairs they order alike and oppositely."""
    keys = sorted(a)
    concordant = discordant = ties_a = ties_b = 0
    for i, x in enumerate(keys):
        for y in keys[i + 1:]:
            da, db = a[x] - a[y], b[x] - b[y]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return ((concordant - discordant) / denominator if denominator else 1.0), concordant, discordant


# Time: one PeTTa run proving every answer at once, and one fusing each answer's proofs.
def _prove(world: list[str], query: Atom, depth: int) -> tuple[dict[str, list[list]], dict[str, float], dict]:
    """SNARS's proofs of `query` to `depth` in the world the lines `world` write, by the answer filling the query's
    variable; each answer's P once SNARS fuses its proofs; and the seconds each run took."""
    role = "o" if is_variable(query.args[1]) else "s"
    program = [_import_snars(), *world, f"!(sn prove {statement(query.pred, query.args)} {depth})"]
    started = time.perf_counter()
    by_answer: dict[str, list[list]] = {}
    for proof in _proofs(_parse(_petta("\n".join(program) + "\n"))):
        by_answer.setdefault(_filler(proof[3], role), []).append(proof)
    proving = time.perf_counter() - started

    fusing = [_import_snars()]
    fusing += [f'!(Fused {json.dumps(a)} (sn fuse-proofs ({" ".join(_show(p) for p in ps)})))'
               for a, ps in sorted(by_answer.items())]
    started = time.perf_counter()
    fused = {}
    for e in _parse(_petta("\n".join(fusing) + "\n")):
        if isinstance(e, list) and e[:1] == ["Fused"]:
            b, _, u, a = (float(x) for x in e[2][1:])
            fused[json.loads(e[1])] = b + a * u
    return by_answer, fused, {"prove": round(proving, 1), "fuse": round(time.perf_counter() - started, 1)}


# Time: the rewriter's run, and _prove's two PeTTa runs.
def compare(name: str, judge_name: str, threshold: float, depth: int) -> dict[str, object]:
    """The question answered by the rewriter and by SNARS from the same rewrites, both rankings scored against
    Wikidata's answer, and how far they agree."""
    evaluate = _evaluate()
    question, kb, truth, _ = evaluate._question(name)
    engine, retriever = evaluate._tools()
    judge = make_judge(judge_name, threshold=threshold)
    rewriter = Rewriter(kb, judge, engine, retriever, k=evaluate.K, budget=Budget(work=evaluate.WORK, seconds=math.inf),
                        beliefs=evaluate.SETS, data_trust=default_data_trust(judge_name))
    out = rewriter.ask(kb.queries[0], replace(evaluate.STRATEGY, depth=1), rewriter.discoverer())
    rewrites, opinion = rewriter.rewrite_opinions(out.admitted)
    judge.close()
    if out.beliefs is None:
        msg = f"{name}: {out.status} {out.detail}"
        raise RuntimeError(msg)
    lineage = {question.answers_of([row]).pop(): o.projection for row, o in out.beliefs.items()}

    # SNARS's world: the statements the rewrites and rules read, certain; the rules; the rewrites.
    used = {r.premise for r in rewrites} | {b.pred for rule in kb.rules for b in rule.body}
    facts = [f for f in kb.facts if f.pred in used]
    world = [believed(f, Opinion(1.0, 0.0, 0.0), "kb") for f in facts]
    world += [line for i, rule in enumerate(kb.rules) for line in _rule_lines(i, rule)]
    world += [line for r in rewrites for line in rule_rows(r, opinion[r.key], f"semrewrite {judge_name}")]
    by_answer, snars, seconds = _prove(world, kb.queries[0], depth)

    both = set(lineage) & set(snars)
    # An answer's shortest proof, counted in rule applications, and where each ranking puts the answers outside
    # Wikidata's own.
    steps = {a: min(_applications(p[2]) for p in ps) for a, ps in by_answer.items()}
    rank = {name_: {a: k for k, (a, _) in enumerate(sorted(((a, s[a]) for a in both), key=lambda kv: (-kv[1], kv[0])),
                                                   start=1)}
            for name_, s in (("rewriter", lineage), ("snars", snars))}
    tau, concordant, discordant = _kendall({a: lineage[a] for a in both}, {a: snars[a] for a in both})
    multi = sum(1 for a in both if len(by_answer[a]) > 1)
    return {
        "question": name, "judge": judge_name, "threshold": threshold, "depth": depth,
        "rewrites": len(rewrites), "facts": len(facts), "proofs": sum(len(ps) for ps in by_answer.values()),
        "answers": {"rewriter": len(lineage), "snars": len(snars), "both": len(both),
                    "rewriter_only": sorted(set(lineage) - set(snars))[:20],
                    "snars_only": sorted(set(snars) - set(lineage))[:20], "with_several_proofs": multi},
        "average_precision": {"rewriter": _average_precision({a: lineage[a] for a in both}, truth & both),
                              "snars": _average_precision({a: snars[a] for a in both}, truth & both)},
        "kendall_tau_b": tau, "pairs_concordant": concordant, "pairs_discordant": discordant,
        "p_range": {"rewriter": [min(lineage[a] for a in both), max(lineage[a] for a in both)],
                    "snars": [min(snars[a] for a in both), max(snars[a] for a in both)]},
        "seconds": seconds,
        "outside": {a: {"rewriter": rank["rewriter"][a], "snars": rank["snars"][a]} for a in sorted(both - truth)},
        "per_answer": {a: [round(lineage[a], 6), round(snars[a], 6), len(by_answer[a]), steps[a]] for a in sorted(both)},
    }


# Time: 2 x 3 arms, each _prove's two PeTTa runs over 2 x links statements and rules.
def chain(links: int = 4, depth: int = 3) -> dict[str, Any]:
    """SNARS's P for each answer along a chain a -> b1 -> ... -> b<links> of owned_by statements, which a owns through
    the rewrite owns(X,Y) <= owned_by(Y,X) at every link and the transitive rule between links, in each arm: the rule
    rows at SNARS's structural truth or certain, and the rewrite's validity at each of VALIDITIES."""
    nodes = ["a", *(f"b{i}" for i in range(1, links + 1))]
    facts = [believed(Atom("owned_by", (child, parent)), Opinion(1.0, 0.0, 0.0), "kb")
             for parent, child in itertools.pairwise(nodes)]
    transitive = _rule_lines(0, Rule(Atom("owns", ("X", "Z")), (Atom("owns", ("X", "Y")), Atom("owns", ("Y", "Z")))))
    rewrite = Rewrite("owns", "owned_by", (1, 0))
    arms = []
    for rules, swap in (("SNARS's structural truth", ()), ("certain", CERTAIN_RULES)):
        for validity, opinion in VALIDITIES.items():
            world = [*swap, *facts, *transitive, *rule_rows(rewrite, opinion, "chain")]
            by_answer, fused, _ = _prove(world, Atom("owns", ("a", "X")), depth)
            arms.append({"rules": rules, "validity": validity,
                         "answers": {a: {"applications": min(_applications(p[2]) for p in ps), "P": round(fused[a], 4)}
                                     for a, ps in sorted(by_answer.items())}})
    return {"links": links, "depth": depth,
            "validities": {v: [o.belief, o.disbelief, o.uncertainty, o.base_rate] for v, o in VALIDITIES.items()},
            "arms": arms}


if __name__ == "__main__":
    # Each argument names a question, optionally with its proof depth after a colon (alphabet:4), or the chain.
    args = sys.argv[1:] or [*RUNS, "chain"]
    if "chain" in args:
        result = chain()
        CHAIN.write_text(json.dumps(result, indent=1) + "\n")
        for arm in result["arms"]:
            print(json.dumps({"rules": arm["rules"], "validity": arm["validity"],
                              "P": {a: [x["applications"], x["P"]] for a, x in arm["answers"].items()}}))
    runs = [(a.split(":")[0], int(a.split(":")[1]) if ":" in a else None) for a in args if a != "chain"]
    if runs:
        measured = [compare(n, *RUNS[n][:2], RUNS[n][2] if depth is None else depth) for n, depth in runs]
        RESULTS.write_text(json.dumps(measured, indent=1) + "\n")
        for m in measured:
            print(json.dumps({k: m[k] for k in ("question", "depth", "average_precision", "kendall_tau_b",
                                                 "pairs_discordant", "p_range", "seconds", "outside")}))
