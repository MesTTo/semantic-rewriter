"""Purpose: the implementation experiment: run every rewriter strategy with each
judge on every benchmark knowledge base, score answers against the gold answers
and admitted rewrites against the gold implication closure, and write
results/implementations.jsonl (one row per run) and results/implementations.json
(aggregates, retrieval recall, the divergence sweep and the isolation check).

Run: .venv/bin/python experiments/impl_eval.py [--judges oracle,nli-deberta-base,...]

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
from collections.abc import Iterable
from pathlib import Path

from semrewrite.benchmarks import BENCHMARKS
from semrewrite.discovery import ScoreCache
from semrewrite.engine import STRATEGIES, Budget, Outcome, Rewriter
from semrewrite.judging import Judge, OracleJudge, Proposal
from semrewrite.kb import KB, Query, implication_closure
from semrewrite.retrieve import Retriever

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
K = 8


def make_judge(name: str, kb: KB) -> Judge:
    """The oracle judge knows the knowledge base's gold closure; every other judge comes from the factory."""
    if name == "oracle":
        return OracleJudge.of(implication_closure(kb.gold))
    from semrewrite.judges import make

    return make(name)


def classify_rewrite(key: tuple[str, str, bool], closure: frozenset, traps: frozenset[str]) -> str:
    """true, or why an admitted rewrite is false: reversed, swap (wrong argument order), trap or unrelated."""
    premise, conclusion, swapped = key
    if key in closure:
        return "true"
    if (conclusion, premise, swapped) in closure:
        return "reversed"
    if (premise, conclusion, not swapped) in closure:
        return "swap"
    if premise in traps:
        return "trap"
    return "unrelated"


def row(outcome: Outcome, mode: str, gold: set, closure: frozenset, traps: frozenset[str]) -> dict:
    """One run as a flat record: answer counts against gold, cost, termination and admitted-rewrite kinds."""
    kinds: dict[str, int] = defaultdict(int)
    for key in outcome.admitted:
        kinds[classify_rewrite(key, closure, traps)] += 1
    tp = len(outcome.answers & gold)
    return {
        "kb": outcome.kb, "judge": outcome.judge, "strategy": outcome.strategy, "mode": mode, "query": outcome.query,
        "gold": len(gold), "answers": len(outcome.answers), "tp": tp, "fp": len(outcome.answers) - tp, "fn": len(gold) - tp,
        "exact": outcome.answers == gold, "status": outcome.status, "terminated": outcome.terminated,
        "derivations": outcome.derivations, "expansions": outcome.expansions, "proposals": outcome.proposals,
        "retrievals": outcome.retrievals, "rewrites": outcome.rewrites, "rounds": outcome.rounds,
        "inferences": outcome.inferences, "tables": outcome.tables, "seconds": round(outcome.seconds, 4),
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
        target = rewriter.catalogue[goal]
        for q in disc.candidates(goal):
            proposals += [Proposal(rewriter.catalogue[q], target, s) for s in (False, True)]
        proposals.append(Proposal(target, target, True))
    before = rewriter.scorer.computed
    rewriter.scorer.score(proposals)
    return rewriter.scorer.computed - before


def retrieval_recall(kb: KB, retriever: Retriever, k: int = K) -> dict:
    """Share of gold-closure implications whose premise is among the conclusion's k retrieved candidates."""
    closure = implication_closure(kb.gold)
    catalogue = kb.catalogue()
    hits = total = 0
    for premise, conclusion, _ in closure:
        if premise == conclusion:
            continue
        total += 1
        names = {n for n, _ in retriever.top_k(catalogue[conclusion], catalogue, k)}
        hits += premise in names
    return {"kb": kb.name, "k": k, "closure_pairs": total, "retrieved": hits, "recall": hits / total if total else 1.0}


def run_kb(kb: KB, judge_name: str, judge: Judge, retriever: Retriever, budget: Budget, modes: Iterable[str]) -> tuple[list[dict], dict]:
    """Every strategy on every query of `kb` under one judge; returns rows and the judge's rewrite scores over the catalogue."""
    rewriter = Rewriter(kb, ScoreCache(judge), retriever, k=K, budget=budget)
    computed = warm_scores(rewriter)
    closure = implication_closure(kb.gold)
    answers_closure = kb.closure()
    rows: list[dict] = []
    for name, strategy in STRATEGIES.items():
        for mode in modes:
            if mode == "warm" and not strategy.memo:
                continue
            shared = rewriter.discoverer(memo=strategy.memo) if mode == "warm" else None
            for query in kb.queries:
                outcome = rewriter.ask(strategy, query, shared)
                rows.append(row(outcome, mode, kb.answers(query, answers_closure), closure, kb.traps))
            print(f"  {kb.name} {judge_name} {name} {mode}: done", flush=True)
    full = rewriter.discoverer(memo=True)
    admitted = {rw.key for goal in rewriter.catalogue for rw in full.rewrites(goal)}
    kinds: dict[str, int] = defaultdict(int)
    for key in admitted:
        kinds[classify_rewrite(key, closure, kb.traps)] += 1
    true = kinds.get("true", 0)
    discovery = {
        "kb": kb.name, "judge": judge_name, "admitted": len(admitted), "kinds": dict(kinds),
        "precision": true / len(admitted) if admitted else 1.0,
        "recall": true / len([c for c in closure if c[0] != c[1] or c[2]]) if closure else 1.0,
        "judge_computed": computed, "judge_usage": dict(getattr(judge, "usage", {})),
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
        kinds: dict[str, int] = defaultdict(int)
        for r in rs:
            for kind, c in r["rewrite_kinds"].items():
                kinds[kind] += c
        admitted = sum(kinds.values())
        out.append({
            "kb": kb, "judge": judge, "strategy": strategy, "mode": mode, "queries": n,
            "precision": round(p, 4), "recall": round(rc, 4), "f1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0,
            "exact_rate": round(sum(r["exact"] for r in rs) / n, 4),
            "terminated_rate": round(sum(r["terminated"] for r in rs) / n, 4),
            "proposals_per_query": round(sum(r["proposals"] for r in rs) / n, 2),
            "expansions_per_query": round(sum(r["expansions"] for r in rs) / n, 2),
            "retrievals_per_query": round(sum(r["retrievals"] for r in rs) / n, 2),
            "inferences_per_query": round(sum(r["inferences"] for r in rs) / n, 1),
            "tables_per_query": round(sum(r["tables"] for r in rs) / n, 2),
            "seconds_per_query": round(sum(r["seconds"] for r in rs) / n, 4),
            "false_rewrite_rate": round(1 - kinds.get("true", 0) / admitted, 4) if admitted else 0.0,
            "rewrite_kinds": dict(kinds),
        })
    return out


def divergence(kb: KB, budgets: Iterable[int]) -> list[dict]:
    """Untabled strategies on a query through an alias cycle at rising budgets: work grows with the budget, answers never come."""
    judge = OracleJudge.of(implication_closure(kb.gold))
    query = Query("mother", "X", "Y")
    out = []
    for budget in budgets:
        rewriter = Rewriter(kb, ScoreCache(judge), None, budget=Budget(inferences=budget, seconds=600))
        for name in ("naive", "memo-untabled", "lazy-tabled"):
            o = rewriter.ask(STRATEGIES[name], query)
            out.append({"kb": kb.name, "query": str(query), "strategy": name, "budget": budget, "status": o.status,
                        "inferences": o.inferences, "expansions": o.expansions, "proposals": o.proposals,
                        "answers": len(o.answers), "seconds": round(o.seconds, 3)})
    return out


def run_one(kb_name: str, strategy: str, query_index: int) -> dict:
    """One oracle-judge run in this process, for the isolation check's subprocesses."""
    kb = BENCHMARKS[kb_name]()
    rewriter = Rewriter(kb, OracleJudge.of(implication_closure(kb.gold)), None, budget=Budget(inferences=300_000, seconds=60))
    o = rewriter.ask(STRATEGIES[strategy], kb.queries[query_index])
    return {"answers": sorted(map(list, o.answers)), "status": o.status}


def isolation_check(kb_name: str) -> dict:
    """Compare each oracle run done in one shared process with the same run in a fresh process.

    pymetta 0.9.4 lets state cross contexts (repros/symbol_use_leaks_across_contexts.py), so the
    shared-process results stand only if a fresh process gives the same
    answers and status for every strategy and query.
    """
    kb = BENCHMARKS[kb_name]()
    rewriter = Rewriter(kb, OracleJudge.of(implication_closure(kb.gold)), None, budget=Budget(inferences=300_000, seconds=60))
    mismatches, checked = [], 0
    for name in STRATEGIES:
        for i, query in enumerate(kb.queries):
            shared = rewriter.ask(STRATEGIES[name], query)
            fresh = json.loads(subprocess.run(
                [sys.executable, __file__, "--one", kb_name, name, str(i)],
                capture_output=True, text=True, check=True,
            ).stdout.strip().splitlines()[-1])
            checked += 1
            same = sorted(map(list, shared.answers)) == fresh["answers"] and shared.status == fresh["status"]
            if not same:
                mismatches.append({"strategy": name, "query": str(query), "shared": shared.status, "fresh": fresh["status"]})
    return {"kb": kb_name, "checked": checked, "mismatches": mismatches}


def main() -> None:
    """Parse arguments, run the requested parts and write the results files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judges", default="oracle")
    parser.add_argument("--kbs", default=",".join(BENCHMARKS))
    parser.add_argument("--modes", default="cold,warm")
    parser.add_argument("--budget", type=int, default=2_000_000)
    parser.add_argument("--skip-extras", action="store_true", help="skip the divergence sweep and the isolation check")
    parser.add_argument("--tag", default="", help="suffix for the output files, so judge runs can be split across processes")
    parser.add_argument("--one", nargs=3, metavar=("KB", "STRATEGY", "QUERY"))
    args = parser.parse_args()
    if args.one:
        kb_name, strategy, index = args.one
        print(json.dumps(run_one(kb_name, strategy, int(index))))
        return
    RESULTS.mkdir(exist_ok=True)
    kbs = [BENCHMARKS[name]() for name in args.kbs.split(",")]
    retriever = Retriever()
    budget = Budget(inferences=args.budget, seconds=120)
    modes = args.modes.split(",")
    rows: list[dict] = []
    discovery: list[dict] = []
    started = time.perf_counter()
    for judge_name in args.judges.split(","):
        judge: Judge | None = None
        for kb in kbs:
            judge = make_judge(judge_name, kb) if judge_name == "oracle" or judge is None else judge
            kb_rows, kb_discovery = run_kb(kb, judge_name, judge, retriever, budget, modes)
            rows += kb_rows
            discovery.append(kb_discovery)
        close = getattr(judge, "close", None)
        if callable(close):
            close()
    tag = f".{args.tag}" if args.tag else ""
    with (RESULTS / f"implementations{tag}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    summary = {
        "judges": args.judges.split(","), "kbs": args.kbs.split(","), "modes": modes, "k": K, "budget_inferences": args.budget,
        "retriever": retriever.model_name, "seconds": round(time.perf_counter() - started, 1),
        "aggregates": aggregate(rows), "discovery": discovery,
        "retrieval_recall": [retrieval_recall(kb, retriever) for kb in kbs],
    }
    if not args.skip_extras:
        summary["divergence"] = divergence(BENCHMARKS["family"](), (10_000, 100_000, 1_000_000, 10_000_000))
        summary["isolation"] = isolation_check("family")
    (RESULTS / f"implementations{tag}.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(f"wrote {len(rows)} rows in {summary['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
