"""Purpose: render the experiment's results files into the tables RESULTS.md
shows, replacing each block between `<!-- BEGIN GENERATED: name -->` and
`<!-- END GENERATED: name -->`, so every number in the report's tables comes
from a file under results/ by one command.

Run: .venv/bin/python experiments/report.py
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from impl_eval import aggregate

from semrewrite.trace import plural

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "RESULTS.md"

STRATEGY_ORDER = ["exact", "naive", "memo-untabled", "nomemo-tabled", "depth-bounded", "depth-tabled",
                  "rewrite-depth-tabled", "lazy-tabled", "rounds", "eager-worklist", "alias-canonical", "saturate"]


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.is_file():
        msg = f"{path} is missing; run the experiment that writes it first"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text())


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def _load_judges() -> dict:
    """The real-judge runs, written as one file pair per process (implementations.judges-*), merged.

    Aggregates are recomputed from the saved rows with the runner's own
    aggregate(), so every table reads one definition of each metric whichever
    version of the runner wrote the summary.
    """
    paths = sorted(RESULTS.glob("implementations.judges-*.json"))
    if not paths:
        msg = f"no implementations.judges-*.json under {RESULTS}; run experiments/impl_eval.py with real judges first"
        raise FileNotFoundError(msg)
    merged: dict = {"aggregates": [], "discovery": [], "sweep": []}
    rows: list[dict] = []
    for path in paths:
        part = json.loads(path.read_text())
        merged["discovery"] += part["discovery"]
        merged["sweep"] += part.get("sweep", [])
        rows += [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
    merged["aggregates"] = aggregate(rows)
    return merged


def _load_oracle() -> dict:
    """The oracle run's summary with its aggregates recomputed from its saved rows."""
    summary = _load("implementations.oracle.json")
    rows = [json.loads(line) for line in (RESULTS / "implementations.oracle.jsonl").read_text().splitlines()]
    summary["aggregates"] = aggregate(rows)
    return summary


def _aggregates(summary: dict, judge: str, kb: str = "ALL") -> dict[tuple[str, str], dict]:
    return {(a["strategy"], a["mode"]): a for a in summary["aggregates"] if a["judge"] == judge and a["kb"] == kb}


def strategies_table() -> str:
    """Every strategy with the oracle judge over all four benchmarks, cold and warm.

    Wall time is left out: per-query seconds grew over the life of the run's
    process, so they measure when a strategy ran more than what it costs;
    engine work is read from counted inferences instead.
    """
    agg = _aggregates(_load_oracle(), "oracle")
    rows = []
    for s in STRATEGY_ORDER:
        cold, warm = agg.get((s, "cold")), agg.get((s, "warm"))
        if cold is None:
            continue
        rows.append([
            s, cold["precision"], cold["recall"], cold["finished_rate"], cold["proposals_per_query"],
            warm["proposals_per_query"] if warm else "n/a", cold["expansions_per_query"],
            int(cold["work_per_query"]), cold["tables_per_query"],
        ])
    return _table(["strategy", "precision", "recall", "finished", "proposals/query cold", "proposals/query warm",
                   "expansions/query", "inferences/query", "tables/query"], rows)


def divergence_table() -> str:
    """The untabled strategies' work at rising budgets on mother(X, Y), against lazy-tabled."""
    rows = [[d["strategy"], d["budget"], d["status"], d["work"], d["expansions"], d["proposals"], d["answers"], d["seconds"]]
            for d in _load("implementations.oracle.json")["divergence"]]
    return _table(["strategy", "budget (inferences)", "status", "inferences", "expansions", "proposals", "answers", "seconds"], rows)


def isolation_table() -> str:
    """Whether runs sharing one process agree with the same runs in fresh processes."""
    iso = _load("implementations.oracle.json")["isolation"]
    return _table(["knowledge base", "runs compared", "mismatches"], [[iso["kb"], iso["checked"], len(iso["mismatches"])]])


def retrieval_table() -> str:
    """Retrieval recall of gold-closure implications at k."""
    rows = [[r["kb"], r["k"], r["closure_pairs"], r["retrieved"], r["recall"]] for r in _load("implementations.oracle.json")["retrieval_recall"]]
    return _table(["knowledge base", "k", "gold implications", "premise retrieved", "recall"], rows)


def judges_in_kb_table() -> str:
    """Each real judge inside the rewriter: rewrite precision over the catalogue and lazy-tabled answer quality."""
    summary = _load_judges()
    by_judge: dict[str, list[dict]] = {}
    for d in summary["discovery"]:
        by_judge.setdefault(d["judge"], []).append(d)
    rows = []
    for judge, ds in by_judge.items():
        admitted = sum(d["admitted"] for d in ds)
        kinds: dict[str, int] = {}
        for d in ds:
            for k, v in d["kinds"].items():
                kinds[k] = kinds.get(k, 0) + v
        true = kinds.get("true", 0)
        lazy = _aggregates(summary, judge).get(("lazy-tabled", "cold"), {})
        sweep = next((s for s in summary["sweep"] if s["judge"] == judge and s["strategy"] == "lazy-tabled"), {})
        rows.append([
            judge, admitted, true / admitted if admitted else 1.0, kinds.get("reversed", 0), kinds.get("swap", 0),
            kinds.get("trap", 0), kinds.get("unrelated", 0), lazy.get("finished_rate", "n/a"), lazy.get("f1", "n/a"),
            sweep.get("f1_held_out_pooled", "n/a"),
        ])
    return _table(["judge", "rewrites admitted", "rewrite precision", "reversed", "wrong argument order", "trap",
                   "unrelated", "finished (default threshold)", "answer F1 (default threshold)",
                   "answer F1 (held-out threshold)"], rows)


def judges_strategies_table() -> str:
    """Answer precision and recall of the main strategies under each real judge."""
    summary = _load_judges()
    judges = sorted({a["judge"] for a in summary["aggregates"]})
    rows = []
    for judge in judges:
        agg = _aggregates(summary, judge)
        for s in STRATEGY_ORDER:
            a = agg.get((s, "cold"))
            if a:
                rows.append([judge, s, a["finished_rate"], a["precision"], a["recall"], a["f1"],
                             a["precision_finished"], a["recall_finished"], a["proposals_per_query"]])
    return _table(["judge", "strategy", "finished", "precision", "recall", "F1", "precision (finished)",
                   "recall (finished)", "proposals/query"], rows)


def pairs_table() -> str:
    """The direction judges on the labelled pair sets, from the judge thread's results/judges.json."""
    judges = _load("judges.json")["judges"]
    rows = []
    for name, j in judges.items():
        variant = j["variants"]["letters"]
        for pair_set in ("levyholt", "wikidata", "hard"):
            test = variant["sets"][pair_set]["test"]
            at = test["at_default"]
            rows.append([name, pair_set, test["rows"], test["direction"]["pairs"], test["direction"]["accuracy"],
                         test["auc"], at["threshold"], at["precision"], at["recall"], at["balanced_accuracy"]])
    return _table(["judge", "pair set", "rows", "one-way pairs", "direction accuracy", "ROC AUC", "threshold",
                   "precision", "recall", "balanced accuracy"], rows)


# Each pair set's name in prose.
SET_NAMES = {"levyholt": "Levy/Holt", "wikidata": "Wikidata", "hard": "the hard set", "delta-atomic": "delta-ATOMIC",
             "copa": "Balanced COPA"}
# The pair sets of implications that typically hold, in the order the tables show them.
TYPICAL_SETS = ("delta-atomic", "copa")


def trust_table() -> str:
    """Each judge's trust as judge_eval.py fitted it on the dev pairs of its set, the kind of question it is asked,
    with that set's test pairs' ROC AUC and the log loss and Brier score of its calibrated opinions, and the base rate
    each set's opinions take; and its data trust, fitted on the Wikidata pairs the knowledge base has evidence on, with
    its log loss on those test pairs alone and fused with that evidence."""
    measured = _load("judges.json")
    judges = measured["judges"]
    rates = measured["calibration"]["base_rates"]
    rows, homes = [], set()
    for name, j in sorted(judges.items(), key=lambda nj: -nj[1]["variants"]["letters"]["trust"]["default"]):
        home = j.get("set", "wikidata")
        homes.add(home)
        variant = j["variants"]["letters"]
        test = variant["sets"][home]["test"]
        data = variant["data"]
        rows.append([name, SET_NAMES[home], variant["trust"]["default"], test["auc"],
                     test["at_default_trust"]["log_loss"], test["at_default_trust"]["brier"], data["trust"],
                     data["test"]["judge"]["rows"], data["test"]["judge"]["log_loss"],
                     data["test"]["fused"]["log_loss"]])
    said = " and ".join(f"{rates[home]:.3f} for {SET_NAMES[home]}'s" for home in sorted(homes, key=list(SET_NAMES).index))
    return (f"Each judge's opinions are taken at the base rate of its set, the share of that set's dev pairs that "
            f"hold: {said}.\n\n"
            + _table(["judge", "set", "trust", "test ROC AUC", "test log loss", "test Brier", "data trust",
                      "Wikidata test pairs with data", "their log loss, judge alone", "fused with the data"], rows))


def typical_table() -> str:
    """Every judge measured on the typical-implication sets' test rows, best log loss first in each set: the ROC AUC
    of its beliefs, as the other tables rank, and of its probabilities, which read disbelief too, at the trust fitted
    on that set's own dev rows and at its base rate, with their log loss and Brier score, from results/judges.json;
    with the scores of answering every test row with the base rate, and the judges the results hold no measurement
    of."""
    measured = _load("judges.json")
    judges = measured["judges"]
    rates = measured["calibration"]["base_rates"]
    lines, rows, absent = [], [], set()
    for pair_set in TYPICAL_SETS:
        title = SET_NAMES[pair_set]
        part = []
        for name, j in judges.items():
            variant = j["variants"]["letters"]
            test = variant["sets"].get(pair_set, {}).get("test")
            if test is None:
                absent.add(name)
                continue
            own = test["at_own_trust"]
            part.append([name, title, test["rows"], variant["trust"]["by_set"][pair_set], test["auc"], own["auc"],
                         own["log_loss"], own["brier"]])
            share, rate = test["positives"] / test["rows"], rates[pair_set]
        rows += sorted(part, key=lambda r: r[6])
        if part:
            # Answering every test row with the dev base rate a: log loss -(s log a + (1 - s) log(1 - a)) and Brier
            # s (1 - a)^2 + (1 - s) a^2, s the share of test rows that hold.
            loss = -(share * math.log(rate) + (1 - share) * math.log(1 - rate))
            brier = share * (1 - rate) ** 2 + (1 - share) * rate**2
            lines.append(f"{title}'s dev rows hold at the base rate {rate:.3f}; answering every test row with it "
                         f"scores log loss {loss:.3f} and Brier {brier:.3f}.")
    text = " ".join(lines) + "\n\n" + _table(["judge", "pair set", "test rows", "trust", "ROC AUC of beliefs",
                                              "ROC AUC of P", "log loss", "Brier"], rows)
    if absent:
        text += f"\n\nNot measured on these sets: {', '.join(sorted(absent))}."
    return text


def typical_actions_table() -> str:
    """The actions example under each judge experiments/typical_actions.py ran, side by side: each action's opinion
    that its doer wastes time as the rewriter weighs it, whether the example's gold lines say it does, and the
    answers each judge's admitted rewrites give, ranked by P, with their average precision against the gold lines'."""
    measured = _load("typical_actions.json")
    judges = measured["judges"]
    names = list(judges)
    by_action = {name: {r["action"]: r for r in j["actions"]} for name, j in judges.items()}
    first = judges[names[0]]["actions"]
    rows = []
    for r in first:
        cells = [r["sentence"].replace("A ", "", 1), "yes" if r["gold"] else "no"]
        for name in names:
            o = by_action[name][r["action"]]
            cells += [f"{o['projection']:.2f}", f"{o['belief']:.2f} / {o['disbelief']:.2f} / {o['uncertainty']:.2f}"
                      + (" *" if o["admitted"] else "")]
        rows.append(cells)
    header = ["action", "wastes time"]
    for name in names:
        header += [f"{name} P", f"{name} b / d / u"]
    text = _table(header, rows)
    for name in names:
        j = judges[name]
        answers = ", ".join(f"{a['answer']} ({a['projection']:.2f})" for a in j["answers"]) or "none"
        text += (f"\n\n{name}: ROC AUC {j['auc']:.3f} against the gold lines. At its threshold {j['threshold']:.3f}, "
                 f"trust {j['trust']:.3f} and base rate {j['base_rate']:.3f}, it admits "
                 f"{plural(len(j['rewrites']), 'rewrite')}, and the rewriter answers {answers}, highest P first, "
                 "where the gold lines give "
                 f"{', '.join(j['gold_answers'])}: average precision {j['average_precision']:.3f}.")
    return text


def beliefs_table() -> str:
    """The Wikidata questions' answers ranked by projected probability, from examples/wikidata/evaluate.py --beliefs,
    rendered by that script's own table."""
    import importlib.util

    rows = _load("wikidata_beliefs.json")
    spec = importlib.util.spec_from_file_location("wikidata_evaluate", ROOT / "examples" / "wikidata" / "evaluate.py")
    assert spec is not None and spec.loader is not None
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    return evaluate.beliefs(rows)


def direction_policy_table() -> str:
    """Plain against direction-aware admission per judge: the calibrated slack, Wikidata test pairs, and lazy-tabled answers."""
    rows = []
    for path in sorted(RESULTS.glob("direction_policy.*.json")):
        for entry in json.loads(path.read_text()):
            wd = entry["pair_sets"]["wikidata"]
            rewriter = {a["judge"].split(":")[1]: a for a in entry["rewriter"]}
            for policy in ("plain", "directional"):
                a = rewriter[policy]
                rows.append([entry["judge"], policy, entry["slack"] if policy == "directional" else "",
                             wd[policy]["precision"], wd[policy]["recall"], wd[policy]["balanced_accuracy"],
                             a["precision"], a["recall"], a["f1"], a["false_rewrite_rate"]])
    return _table(["judge", "policy", "slack", "Wikidata precision", "Wikidata recall", "Wikidata balanced accuracy",
                   "answer precision", "answer recall", "answer F1", "false rewrites"], rows)


def sweep_table() -> str:
    """Each swept strategy's answer F1 at the judge's pair-calibrated default against a threshold calibrated on the other benchmarks."""
    rows = []
    for s in _load_judges()["sweep"]:
        folds = ", ".join(f"{h['kb']} {h['threshold']:.3f}" for h in s["held_out"])
        best = max(s["curve"], key=lambda c: c["f1"])
        rows.append([s["judge"], s["strategy"], s["default_threshold"], s["f1_default_pooled"], folds,
                     s["f1_held_out_pooled"], s["precision_held_out_pooled"], s["recall_held_out_pooled"],
                     s["finished_held_out_pooled"], best["threshold"], best["f1"]])
    return _table(["judge", "strategy", "default threshold", "F1 at default", "held-out threshold per benchmark",
                   "F1 held-out", "precision held-out", "recall held-out", "finished held-out",
                   "best threshold in hindsight", "F1 in hindsight"], rows)


def blowup_table() -> str:
    """A permissive judge's closure against the oracle's on the same hard queries, with no budget in the way."""
    rows = [[r["query"], r["judge"], r["strategy"], r["status"], r["work"], r["tables"], r["rewrites"],
             f"{r['true']}/{r['answers']}", r["gold"]] for r in _load("blowup.json")]
    return _table(["query", "judge", "strategy", "status", "inferences", "tables", "rewrites", "true/answers", "gold"], rows)


NAMED_CASES = (
    "hard:mother>parent", "hard:mother>parent:reverse", "hard:legal_guardian>parent", "hard:stepmother>parent",
    "hard:mother_in_law>mother", "hard:employs>manages", "hard:employs>manages:reverse",
    "hard:causes>correlated_with", "hard:causes>prevents",
)


def named_cases_table() -> str:
    """Each judge's score on the cases the design discussion named; a star marks a score at or over the judge's threshold."""
    data = _load("judges.json")
    names = list(data["judges"])
    thresholds = {n: data["judges"][n]["variants"]["letters"]["thresholds"]["default"] for n in names}
    rows_by_id = {r["id"]: r for r in data["hard_rows"]}
    rows = []
    for case in NAMED_CASES:
        r = rows_by_id[case]
        cells = []
        for n in names:
            s = r["scores"][n]["letters"]
            cells.append(f"{s:.2f}{'*' if s >= thresholds[n] else ''}")
        rows.append([f"{r['premise']} ⇒ {r['conclusion']}", "yes" if r["label"] else "no", *cells])
    return _table(["implication", "holds", *names], rows)


def snars_table() -> str:
    """How SNARS, proving and fusing every answer itself from the exported rewrites, ranks each question's answers
    beside the rewriter's lineage, from experiments/snars_ranking.py."""
    rows = []
    for m in _load("snars_ranking.json"):
        answers, ap, p = m["answers"], m["average_precision"], m["p_range"]
        outside = "; ".join(f"{a}: {r['rewriter']:,} and {r['snars']:,}" for a, r in m["outside"].items()) or "none"
        rows.append([m["question"], m["depth"], f"{answers['both']:,} of {answers['rewriter']:,}",
                     answers["with_several_proofs"], ap["rewriter"], ap["snars"], m["kendall_tau_b"],
                     f"{p['rewriter'][0]:.3f} to {p['rewriter'][1]:.3f}", f"{p['snars'][0]:.3f} to {p['snars'][1]:.3f}",
                     outside])
    return _table(["question", "SNARS depth", "answers both rank", "with several proofs",
                   "average precision over them, rewriter", "and SNARS", "Kendall's tau-b", "P, rewriter",
                   "P, SNARS", "answers outside Wikidata's: rank by each"], rows)


def snars_chain_table() -> str:
    """SNARS's P along experiments/snars_ranking.py's chain, by its rule rows' truth and the rewrite's validity, one
    column per number of rule applications an answer's proofs take."""
    m = _load("snars_chain.json")
    steps = sorted({x["applications"] for arm in m["arms"] for x in arm["answers"].values()})
    rows = [[arm["rules"], arm["validity"],
             *(next(x["P"] for x in arm["answers"].values() if x["applications"] == s) for s in steps)]
            for arm in m["arms"]]
    return _table(["rule rows", "rewrite's validity", *(f"P at {s} rule application{'s' * (s > 1)}" for s in steps)],
                  rows)


def claim_actions_table() -> str:
    """The actions example with the claim check on, from experiments/claim_check.py: for each judge, form and rewrite
    threshold, the gaming and watching-TV matches and the answers through them, each one's P without the gaming belief
    and with it, what the check read against each match, and the check's cost with the belief stated."""
    runs = {(r["judge"], r["form"], r["threshold"], r["believed"]): r for r in _load("claim_check_actions.json")}
    rows = []
    for judge, form, threshold, believed in sorted(runs):
        if believed:
            continue
        before, after = runs[(judge, form, threshold, False)], runs[(judge, form, threshold, True)]
        items = [(f"match {name.replace(' => ', ' ⇒ ')}", "matches", name) for name in before["matches"]
                 if name.endswith("=> wastes_time") and name.split(" => ")[0] in ("play_games", "watch_tv")]
        items += [(f"answer wastes_time({'' if name == '()' else name})", "answers", name) for name in before["answers"]]
        for label, kind, name in items:
            read = after[kind][name].get("read")
            weighed = ("nothing" if not read else "; ".join(
                f"{x['side']} \"{x['statement']}\" (b {x['b']:.3f}, d {x['d']:.3f})" for x in read)) if kind == "matches" else ""
            pairs = after["cost"]["pairs"]
            rows.append([judge, form, f"{threshold:.3g}", label, before[kind][name]["P"], after[kind][name]["P"],
                         weighed, f"{pairs} pair{'s' * (pairs != 1)}, {after['cost']['seconds']:.2f} s"])
    return _table(["judge", "form", "rewrite threshold", "match or answer", "P without the belief", "P with it",
                   "read against it, with the belief stated", "check's cost"], rows)


def claim_readings_table() -> str:
    """How each judge reads delta-ATOMIC's updates against inferences written as conditionals, with no gate, from
    experiments/claim_gate.py: other premises' updates read as contradicting the claim, and a claim's own weakener read
    as contradicting it and own strengthener as entailing it."""
    rows = [[judge, split, m[split]["before"]["other_contradiction"], m[split]["before"]["weakener_as_written"],
             m[split]["before"]["strengthener_as_written"]]
            for judge, m in sorted(_load("claim_gate.json").items()) for split in ("dev", "test")]
    return _table(["judge", "split", "other premises' updates read as contradicting a claim",
                   "own weakeners read as contradicting it", "own strengtheners read as entailing it"], rows)


def claim_gate_table() -> str:
    """The claim check's gate on delta-ATOMIC's updates, from experiments/claim_gate.py: by judge and knowledge-base
    size, at the spread chosen on nli-deberta-large's dev updates, the share of a claim's own updates the gate reads,
    the other premises' updates it reads per claim and the spurious contradictions among them per claim, on dev and
    test, and the spurious contradictions per claim at the rule of thumb 3.5 beside them."""
    measured = _load("claim_gate.json")
    chosen = measured["nli-deberta-large"]["chosen_spread"]
    if chosen is None:
        return "No spread on the grid kept nli-deberta-large's spurious contradictions within the budget on dev."
    rows = []
    for judge, m in sorted(measured.items()):
        for size in sorted(m["dev"]["sizes"], key=lambda s: (s == "all", int(s) if s.isdigit() else 0)):
            at = {(split, spread): next(r for r in m[split]["sizes"][size] if r["spread"] == spread)
                  for split in ("dev", "test") for spread in (chosen, 3.5)}
            rows.append([judge, size, at[("test", chosen)]["own_recall"], at[("test", chosen)]["others_per_claim"],
                         f"{at[('dev', chosen)]['spurious_per_claim']:.4f}",
                         f"{at[('test', chosen)]['spurious_per_claim']:.4f}",
                         f"{at[('dev', 3.5)]['spurious_per_claim']:.4f}",
                         f"{at[('test', 3.5)]['spurious_per_claim']:.4f}"])
    return (f"The gate's spread is {chosen:g}, chosen on nli-deberta-large's dev updates.\n\n"
            + _table(["judge", "statements in the knowledge base", "own updates read, test",
                      "other updates read per claim, test", "spurious contradictions per claim, dev", "test",
                      "at 3.5, dev", "at 3.5, test"], rows))


GENERATORS: dict[str, Callable[[], str]] = {
    "claim-check-actions": claim_actions_table,
    "claim-readings": claim_readings_table,
    "claim-gate": claim_gate_table,
    "named-cases": named_cases_table,
    "direction-policy": direction_policy_table,
    "blowup": blowup_table,
    "sweep": sweep_table,
    "strategies": strategies_table,
    "divergence": divergence_table,
    "isolation": isolation_table,
    "retrieval": retrieval_table,
    "judges-in-kb": judges_in_kb_table,
    "judges-strategies": judges_strategies_table,
    "pairs": pairs_table,
    "trust": trust_table,
    "typical": typical_table,
    "typical-actions": typical_actions_table,
    "wikidata-beliefs": beliefs_table,
    "snars-ranking": snars_table,
    "snars-chain": snars_chain_table,
}


def main() -> None:
    """Replace every generated block in RESULTS.md that a generator names.

    A table whose results file is missing is left as it was and named on
    stderr; with --complete a missing file is an error, since a finished
    report must not keep a stale table.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Fill the generated tables in RESULTS.md from the files under results/.")
    parser.add_argument("--complete", action="store_true", help="fail when a results file a table needs is missing")
    complete = parser.parse_args().complete
    text = REPORT.read_text()
    missing = []
    for name, generate in GENERATORS.items():
        pattern = re.compile(rf"(<!-- BEGIN GENERATED: {re.escape(name)} -->\n)(?:.*?\n)?(<!-- END GENERATED: {re.escape(name)} -->)", re.DOTALL)
        if not pattern.search(text):
            continue
        try:
            body = generate()
        except FileNotFoundError as e:
            if complete:
                raise
            missing.append(f"{name}: {e}")
            continue
        text = pattern.sub(lambda m, body=body: m.group(1) + body + "\n" + m.group(2), text)
    REPORT.write_text(text)
    print(f"rendered {REPORT}")
    for line in missing:
        print(f"not rendered, input missing: {line}", file=sys.stderr)


if __name__ == "__main__":
    main()
