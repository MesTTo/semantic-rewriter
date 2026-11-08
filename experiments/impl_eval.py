"""Purpose: the implementation experiment: run every rewriter strategy with each
judge on every benchmark knowledge base, score answers against the gold answers
and admitted rewrites against what the gold program entails, and write
results/implementations.jsonl (one row per run) and results/implementations.json
(aggregates, retrieval recall, the divergence sweep, the isolation check and,
with --sweep, each named strategy's threshold sweep per judge).

Run: .venv/bin/python experiments/impl_eval.py [--judges oracle,nli-deberta-base,...] [--sweep lazy-tabled,depth-bounded]

Modes: `cold` gives each query a fresh discoverer, so its model work is the
query's alone; `warm` shares one discoverer across a knowledge base's queries
under one strategy, so repeated predicates cost nothing after their first
expansion. Strategies that need no memo run only cold.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from semrewrite.benchmarks import BENCHMARKS
from semrewrite.discovery import ScoreCache
from semrewrite.engines import make as make_engine
from semrewrite.judges.oracle import OracleJudge
from semrewrite.kb import KB, atom
from semrewrite.model import Atom, Key, Proposal, inverse, mappings
from semrewrite.protocols import Admission, Budget, Engine, Judge, Retriever
from semrewrite.retrieve import EmbeddingRetriever
from semrewrite.rewriter import STRATEGIES, Outcome, Rewriter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
K = 8


def make_judge(name: str, kb: KB) -> Judge:
    """The oracle judge admits exactly what the gold program entails; every other judge comes from the factory."""
    if name == "oracle":
        return OracleJudge(kb.entailed())
    from semrewrite.judges import make

    return make(name)


def classify_rewrite(key: Key, kb: KB) -> str:
    """true when the gold program entails it, else why it is false: reversed, swap (wrong argument order), trap or unrelated."""
    premise, conclusion, mapping = key
    if kb.entails(premise, conclusion, mapping):
        return "true"
    if kb.entails(conclusion, premise, inverse(mapping)):
        return "reversed"
    if any(kb.entails(premise, conclusion, m) for m in mappings(len(mapping)) if m != mapping):
        return "swap"
    if premise in kb.traps:
        return "trap"
    return "unrelated"


def row(outcome: Outcome, mode: str, gold: set, kb: KB) -> dict:
    """One run as a flat record: answer counts against gold, cost, termination and admitted-rewrite kinds."""
    kinds: dict[str, int] = defaultdict(int)
    for key in outcome.admitted:
        kinds[classify_rewrite(key, kb)] += 1
    tp = len(outcome.answers & gold)
    return {
        "kb": outcome.kb, "judge": outcome.judge, "strategy": outcome.strategy, "mode": mode, "query": outcome.query,
        "gold": len(gold), "answers": len(outcome.answers), "tp": tp, "fp": len(outcome.answers) - tp, "fn": len(gold) - tp,
        "exact": outcome.answers == gold, "status": outcome.status, "finished": outcome.finished,
        "derivations": outcome.derivations, "expansions": outcome.expansions, "proposals": outcome.proposals,
        "retrievals": outcome.retrievals, "rewrites": len(outcome.admitted), "rounds": outcome.rounds,
        "work": outcome.work, "tables": outcome.tables, "seconds": round(outcome.seconds, 4),
        "rewrite_kinds": dict(kinds),
    }


def warm_scores(rewriter: Rewriter) -> int:
    """Score every proposal any expansion could ask, in one judge call, so later runs read the cache.

    Returns the number of proposals the judge computed. Strategy counts come
    from each discoverer's own stats, so this prefetch adds nothing to them.
    """
    disc = rewriter.discoverer(memo=True)
    proposals: list[Proposal] = []
    for goal in rewriter.catalogue:
        proposals += disc.proposals(goal)
    before = rewriter.scorer.computed
    rewriter.scorer.score(proposals)
    return rewriter.scorer.computed - before


def retrieval_recall(kb: KB, retriever: Retriever, k: int = K) -> dict:
    """Share of entailed implications between distinct predicates whose premise is among the conclusion's k retrieved candidates."""
    catalogue = kb.catalogue()
    hits = total = 0
    for premise, conclusion, _ in kb.entailed():
        if premise == conclusion:
            continue
        total += 1
        names = {n for n, _ in retriever.top_k(catalogue[conclusion], catalogue, k)}
        hits += premise in names
    return {"kb": kb.name, "k": k, "closure_pairs": total, "retrieved": hits, "recall": hits / total if total else 1.0}


def loop_prediction(kb: KB, admitted: set[Key], query: Atom) -> str:
    """What untabled rewriting does on `query` under these admitted rewrites, by the tested characterization.

    tests/test_termination.py checks on generated catalogues that untabled
    rewriting never stops when the query predicate lies on a cycle of rewrites,
    and stops correctly when nothing reachable from it is cyclic; anything else
    is left undecided.
    """
    rewrites: dict[str, set[str]] = defaultdict(set)
    for premise, conclusion, _ in admitted:
        rewrites[conclusion].add(premise)
    calls: dict[str, set[str]] = defaultdict(set, {p: set(qs) for p, qs in rewrites.items()})
    for rule in kb.rules:
        calls[rule.head.pred].update(lit.pred for lit in rule.body)
    start = query.pred
    frontier, seen = list(rewrites.get(start, ())), set()
    while frontier:
        q = frontier.pop()
        if q == start:
            return "loops"
        if q not in seen:
            seen.add(q)
            frontier.extend(rewrites.get(q, ()))
    state: dict[str, int] = {}

    def acyclic(p: str) -> bool:
        state[p] = 1
        for q in calls.get(p, ()):
            if state.get(q) == 1 or (q not in state and not acyclic(q)):
                return False
        state[p] = 2
        return True

    return "terminates" if acyclic(start) else "undecided"


def run_kb(
    kb: KB, judge_name: str, judge: Judge, retriever: Retriever, budget: Budget, modes: Iterable[str], strategies: Iterable[str],
    engine: Engine | None = None, admission: Admission | None = None,
) -> tuple[list[dict], dict]:
    """Every strategy on every query of `kb` under one judge; returns rows and the judge's rewrite scores over the catalogue."""
    rewriter = Rewriter(kb, ScoreCache(judge), engine or make_engine("pymetta"), retriever, k=K, budget=budget, admission=admission)
    computed = warm_scores(rewriter)
    entailed = kb.entailed()
    answers_closure = kb.closure()
    rows: list[dict] = []
    for name in strategies:
        strategy = STRATEGIES[name]
        for mode in modes:
            if mode == "warm" and not strategy.memo:
                continue
            shared = rewriter.discoverer(memo=strategy.memo) if mode == "warm" else None
            for query in kb.queries:
                outcome = rewriter.ask(query, strategy, shared)
                rows.append(row(outcome, mode, kb.answers(query, answers_closure), kb))
            print(f"  {kb.name} {judge_name} {name} {mode}: done", flush=True)
    full = rewriter.discoverer(memo=True)
    admitted = {rw.key for goal in rewriter.catalogue for rw in full.rewrites(goal)}
    kinds: dict[str, int] = defaultdict(int)
    for key in admitted:
        kinds[classify_rewrite(key, kb)] += 1
    true = kinds.get("true", 0)
    loops: dict[str, int] = defaultdict(int)
    for query in kb.queries:
        loops[loop_prediction(kb, admitted, query)] += 1
    discovery = {
        "kb": kb.name, "judge": judge_name, "admitted": len(admitted), "kinds": dict(kinds),
        "precision": true / len(admitted) if admitted else 1.0,
        "recall": true / len(entailed) if entailed else 1.0,
        "judge_computed": computed, "judge_usage": dict(getattr(judge, "usage", {})),
        "untabled_prediction": dict(loops),
    }
    return rows, discovery


def aggregate(rows: list[dict]) -> list[dict]:
    """Micro precision and recall, termination and mean cost per knowledge base, judge, strategy and mode."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["kb"], r["judge"], r["strategy"], r["mode"])].append(r)
        groups[("ALL", r["judge"], r["strategy"], r["mode"])].append(r)
    out = []
    for (kb, judge, strategy, mode), rs in sorted(groups.items()):
        tp, fp, fn = (sum(r[f] for r in rs) for f in ("tp", "fp", "fn"))
        n = len(rs)
        p = tp / (tp + fp) if tp + fp else 1.0
        rc = tp / (tp + fn) if tp + fn else 1.0
        # A run that exhausts its budget answers nothing, so all-runs recall mixes
        # missing rewrites with truncated work; the finished-only figures separate them.
        done = [r for r in rs if r["finished"]]
        dtp, dfp, dfn = (sum(r[f] for r in done) for f in ("tp", "fp", "fn"))
        kinds: dict[str, int] = defaultdict(int)
        for r in rs:
            for kind, c in r["rewrite_kinds"].items():
                kinds[kind] += c
        admitted = sum(kinds.values())
        out.append({
            "kb": kb, "judge": judge, "strategy": strategy, "mode": mode, "queries": n,
            "precision": round(p, 4), "recall": round(rc, 4), "f1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0,
            "exact_rate": round(sum(r["exact"] for r in rs) / n, 4),
            "finished_rate": round(sum(r["finished"] for r in rs) / n, 4),
            "precision_finished": round(dtp / (dtp + dfp), 4) if dtp + dfp else 1.0,
            "recall_finished": round(dtp / (dtp + dfn), 4) if dtp + dfn else 1.0,
            "proposals_per_query": round(sum(r["proposals"] for r in rs) / n, 2),
            "expansions_per_query": round(sum(r["expansions"] for r in rs) / n, 2),
            "retrievals_per_query": round(sum(r["retrievals"] for r in rs) / n, 2),
            "work_per_query": round(sum(r["work"] for r in rs) / n, 1),
            "tables_per_query": round(sum(r["tables"] for r in rs) / n, 2),
            "seconds_per_query": round(sum(r["seconds"] for r in rs) / n, 4),
            "false_rewrite_rate": round(1 - kinds.get("true", 0) / admitted, 4) if admitted else 0.0,
            "rewrite_kinds": dict(kinds),
        })
    return out


def divergence(kb: KB, budgets: Iterable[int], engine: Engine) -> list[dict]:
    """Untabled strategies on a query through an alias cycle at rising budgets: work grows with the budget, answers never come."""
    judge = OracleJudge(kb.entailed())
    query = atom("mother", "X", "Y")
    out = []
    for budget in budgets:
        rewriter = Rewriter(kb, ScoreCache(judge), engine, None, budget=Budget(work=budget, seconds=600))
        for name in ("naive", "memo-untabled", "lazy-tabled"):
            o = rewriter.ask(query, name)
            out.append({"kb": kb.name, "query": str(query), "strategy": name, "budget": budget, "status": o.status,
                        "work": o.work, "expansions": o.expansions, "proposals": o.proposals,
                        "answers": len(o.answers), "seconds": round(o.seconds, 3)})
    return out


def run_one(kb_name: str, strategy: str, query_index: int, engine_name: str) -> dict:
    """One oracle-judge run in this process, for the isolation check's subprocesses."""
    kb = BENCHMARKS[kb_name]()
    rewriter = Rewriter(kb, OracleJudge(kb.entailed()), make_engine(engine_name), None, budget=Budget(work=300_000, seconds=60))
    o = rewriter.ask(kb.queries[query_index], strategy)
    return {"answers": sorted(map(list, o.answers)), "status": o.status}


def isolation_check(kb_name: str, engine_name: str) -> dict:
    """Compare each oracle run done in one shared process with the same run in a fresh process.

    pymetta 0.9.4 lets state cross contexts (repros/symbol_use_leaks_across_contexts.py),
    so the shared-process results stand only if a fresh process gives the same
    answers and status for every strategy and query.
    """
    kb = BENCHMARKS[kb_name]()
    rewriter = Rewriter(kb, OracleJudge(kb.entailed()), make_engine(engine_name), None, budget=Budget(work=300_000, seconds=60))
    caps = rewriter.engine.capabilities
    mismatches, checked = [], 0
    for name, strategy in STRATEGIES.items():
        if not all(getattr(caps, c) for c, need in strategy.needs().items() if need):
            continue
        for i, query in enumerate(kb.queries):
            shared = rewriter.ask(query, strategy)
            fresh = json.loads(subprocess.run(
                [sys.executable, __file__, "--engine", engine_name, "--one", kb_name, name, str(i)],
                capture_output=True, text=True, check=True,
            ).stdout.strip().splitlines()[-1])
            checked += 1
            same = sorted(map(list, shared.answers)) == fresh["answers"] and shared.status == fresh["status"]
            if not same:
                mismatches.append({"strategy": name, "query": str(query), "shared": shared.status, "fresh": fresh["status"]})
    return {"kb": kb_name, "checked": checked, "mismatches": mismatches}


def sweep(kbs: list[KB], judge_name: str, judge: Judge, retriever: Retriever, budget: Budget,
          strategies: Sequence[str], engine: Engine,
          quantiles: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995)) -> list[dict]:
    """Each strategy's answer quality at a grid of thresholds, and a threshold chosen without the benchmark it is scored on.

    The grid is quantiles of the judge's own scores over every proposal the
    four benchmarks ask, so it adapts to how the judge spreads its scores; it
    sits in the upper half because true implications are a small share of the
    proposals, and below the median nearly every proposal is admitted. For
    each strategy and benchmark the threshold maximising answer F1 on the other
    three is chosen, then scored on the held-out one: leave-one-benchmark-out
    calibration for the rewriter's objective, beside the pair-set default. All
    strategies share one grid and one score cache per benchmark, so they are
    compared on the same admissions. Each point also counts the runs that
    finished inside the budget, since an unfinished run returns no answers and
    so loses recall without any wrong rewrite. Returns one result per strategy.

    Cost: thresholds x strategies x queries rewriter runs, each within budget.
    """
    rewriters = [Rewriter(kb, ScoreCache(judge), engine, retriever, k=K, budget=budget) for kb in kbs]
    for r in rewriters:
        warm_scores(r)
    pooled = sorted(s for r in rewriters for s in r.scorer.scores())
    grid = sorted({pooled[round(q * (len(pooled) - 1))] for q in quantiles} | {judge.threshold})
    default = judge.threshold
    counts: dict[tuple[str, str, float], tuple[int, int, int, int]] = {}
    for rewriter in rewriters:
        closure = rewriter.kb.closure()
        golds = [(query, rewriter.kb.answers(query, closure)) for query in rewriter.kb.queries]
        for t in grid:
            judge.threshold = t
            for strategy in strategies:
                tp = fp = fn = finished = 0
                for query, gold in golds:
                    o = rewriter.ask(query, strategy)
                    tp += len(o.answers & gold)
                    fp += len(o.answers - gold)
                    fn += len(gold - o.answers)
                    finished += o.finished
                counts[(strategy, rewriter.kb.name, t)] = (tp, fp, fn, finished)
    judge.threshold = default
    queries = sum(len(r.kb.queries) for r in rewriters)

    def f1(tp: int, fp: int, fn: int) -> float:
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    names = [kb.name for kb in kbs]
    results = []
    for strategy in strategies:
        def total(kbs_: list[str], t: float, strategy: str = strategy) -> tuple[int, int, int]:
            tp_, fp_, fn_ = (sum(counts[(strategy, n, t)][i] for n in kbs_) for i in range(3))
            return tp_, fp_, fn_

        held_out = []
        for name in names:
            others = [n for n in names if n != name]
            best = max(grid, key=lambda t: (f1(*total(others, t)), -t))
            held_out.append({"kb": name, "threshold": best, "f1_held_out": round(f1(*counts[(strategy, name, best)][:3]), 4),
                             "f1_at_default": round(f1(*counts[(strategy, name, default)][:3]), 4),
                             "finished_held_out": counts[(strategy, name, best)][3]})
        curve = []
        for t in grid:
            tp, fp, fn = total(names, t)
            curve.append({"threshold": t, "f1": round(f1(tp, fp, fn), 4),
                          "precision": round(tp / max(1, tp + fp), 4), "recall": round(tp / max(1, tp + fn), 4),
                          "finished": round(sum(counts[(strategy, n, t)][3] for n in names) / queries, 4)})
        tp, fp, fn = (sum(counts[(strategy, h["kb"], h["threshold"])][i] for h in held_out) for i in range(3))
        results.append({"judge": judge_name, "strategy": strategy, "default_threshold": default, "curve": curve,
                        "held_out": held_out, "f1_held_out_pooled": round(f1(tp, fp, fn), 4),
                        "precision_held_out_pooled": round(tp / max(1, tp + fp), 4),
                        "recall_held_out_pooled": round(tp / max(1, tp + fn), 4),
                        "finished_held_out_pooled": round(sum(h["finished_held_out"] for h in held_out) / queries, 4),
                        "f1_default_pooled": round(f1(*total(names, default)), 4)})
    return results


def main() -> None:
    """Parse arguments, run the requested parts and write the results files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judges", default="oracle")
    parser.add_argument("--engine", default="pymetta", help="engine to evaluate on; the experiment in RESULTS.md used pymetta")
    parser.add_argument("--kbs", default=",".join(BENCHMARKS))
    parser.add_argument("--modes", default="cold,warm")
    parser.add_argument("--strategies", default=None,
                        help="strategies to run; default all, or none with --reuse-rows")
    parser.add_argument("--budget", type=int, default=2_000_000)
    parser.add_argument("--skip-extras", action="store_true", help="skip the divergence sweep and the isolation check")
    parser.add_argument("--sweep", default="", help="comma-separated strategies whose threshold to sweep for each real judge")
    parser.add_argument("--reuse-rows", action="store_true",
                        help="keep the rows and extras saved under the tag, recomputing only the strategies, sweeps and extras this run names")
    parser.add_argument("--divergence-budgets", default="10000,100000,1000000,3000000",
                        help="inference budgets for the divergence sweep; time grows faster than the budget")
    parser.add_argument("--tag", default="", help="suffix for the output files, so judge runs can be split across processes")
    parser.add_argument("--one", nargs=3, metavar=("KB", "STRATEGY", "QUERY"))
    args = parser.parse_args()
    if args.one:
        kb_name, strategy, index = args.one
        print(json.dumps(run_one(kb_name, strategy, int(index), args.engine)))
        return
    RESULTS.mkdir(exist_ok=True)
    kbs = [BENCHMARKS[name]() for name in args.kbs.split(",")]
    retriever = EmbeddingRetriever()
    engine = make_engine(args.engine)
    budget = Budget(work=args.budget, seconds=120)
    modes = args.modes.split(",")
    rows: list[dict] = []
    discovery: list[dict] = []
    sweeps: list[dict] = []
    started = time.perf_counter()
    tag = f".{args.tag}" if args.tag else ""
    rows_path = RESULTS / f"implementations{tag}.jsonl"
    summary_path = RESULTS / f"implementations{tag}.json"
    previous = json.loads(summary_path.read_text()) if args.reuse_rows else {}
    saved = [json.loads(line) for line in rows_path.read_text().splitlines()] if args.reuse_rows else []
    if args.strategies is not None:
        strategies = [s for s in args.strategies.split(",") if s]
    else:
        strategies = [] if args.reuse_rows else list(STRATEGIES)
    swept = [s for s in args.sweep.split(",") if s]
    for judge_name in args.judges.split(","):
        judge: Judge | None = None
        for kb in kbs:
            judge = make_judge(judge_name, kb) if judge_name == "oracle" or judge is None else judge
            kb_rows, kb_discovery = run_kb(kb, judge_name, judge, retriever, budget, modes, strategies, engine)
            rows += kb_rows
            discovery.append(kb_discovery)
        if swept and judge_name != "oracle":
            sweeps += sweep(kbs, judge_name, judge, retriever, budget, swept, engine)
            print(f"  sweep {judge_name}: done", flush=True)
        close = getattr(judge, "close", None)
        if callable(close):
            close()
    rows = [r for r in saved if r["strategy"] not in strategies] + rows
    with rows_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    summary = {
        "judges": args.judges.split(","), "engine": args.engine, "kbs": args.kbs.split(","), "modes": modes, "k": K, "budget_work": args.budget,
        "retriever": retriever.model_name, "seconds": round(time.perf_counter() - started, 1),
        "aggregates": aggregate(rows), "discovery": discovery,
        "retrieval_recall": [retrieval_recall(kb, retriever) for kb in kbs],
    }
    kept_sweeps = [s for s in previous.get("sweep", []) if s["strategy"] not in swept]
    if kept_sweeps or sweeps:
        summary["sweep"] = kept_sweeps + sweeps
    if not args.skip_extras:
        budgets = [int(b) for b in args.divergence_budgets.split(",")]
        summary["divergence"] = divergence(BENCHMARKS["family"](), budgets, engine)
        summary["isolation"] = isolation_check("family", args.engine)
    else:
        summary |= {key: previous[key] for key in ("divergence", "isolation") if key in previous}
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(f"wrote {len(rows)} rows in {summary['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
