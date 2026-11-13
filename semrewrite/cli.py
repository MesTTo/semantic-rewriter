"""Purpose: the semrewrite command line: answer a query over a knowledge base file,
show the rewrites discovery admits for a predicate, list engines and strategies,
run the engine conformance kit, and convert between knowledge-base formats.

    semrewrite ask family.kb "parent(X, tom)" --judge nli-deberta-large --explain
    semrewrite rewrites family.kb parent --judge jev --all
    semrewrite engines
    semrewrite conformance --engine pymetta
    semrewrite convert family.metta family.kb

Exit status: 0 when the query finished, 2 when a budget stopped it, 1 on an error.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, replace

from semrewrite import formats
from semrewrite.admission import Directional, Threshold
from semrewrite.engines import REGISTRY as ENGINES
from semrewrite.engines import available
from semrewrite.engines import make as make_engine
from semrewrite.kb import KB
from semrewrite.lineage import SETS
from semrewrite.model import Rewrite, parse_atom
from semrewrite.opinion import Opinion
from semrewrite.protocols import Budget
from semrewrite.rewriter import STRATEGIES, CapabilityError, Rewriter
from semrewrite.snars import rule_rows
from semrewrite.trace import TextTrace, plural

# The pyproject.toml extra that installs each optional library, by import name;
# tests/test_packaging.py checks it against the extras.
EXTRAS = {
    "metta": "pymetta",
    "torch": "models",
    "sentence_transformers": "models",
    "transformers": "models",
    "numpy": "models",
    "typesafe_sdk": "jev",
    "hypothesis": "conformance",
}


def _judge(args: argparse.Namespace, kb: KB):
    from semrewrite.judges import make

    if args.judge == "oracle":
        gold = kb.entailed() if kb.gold else None
        if not gold:
            msg = "the oracle judge needs gold implications in the knowledge base (gold lines)"
            raise SystemExit(msg)
        return make("oracle", gold=gold, **({"threshold": args.threshold} if args.threshold is not None else {}))
    return make(args.judge, **({"threshold": args.threshold} if args.threshold is not None else {}))


def _retriever(args: argparse.Namespace):
    from semrewrite.retrieve import AllCandidates, EmbeddingRetriever, JudgeRetriever

    if args.retriever == "judge":
        from semrewrite.judges import make

        return JudgeRetriever(make(args.prefilter))
    return AllCandidates() if args.retriever == "all" else EmbeddingRetriever()


def _rewriter(args: argparse.Namespace, kb: KB) -> Rewriter:
    from semrewrite.judges import default_data_trust

    admission = Directional(args.directional) if args.directional is not None else Threshold()
    beliefs = getattr(args, "beliefs", False)
    # Answer beliefs and the SNARS export both read each rewrite's opinion, which the data trust enters.
    weighed = beliefs or getattr(args, "snars", False)
    return Rewriter(kb, _judge(args, kb), make_engine(args.engine), _retriever(args), k=args.k,
                    budget=Budget(work=args.work, seconds=args.seconds), admission=admission,
                    proofs=getattr(args, "explain", False), trace=TextTrace() if getattr(args, "trace", False) else None,
                    beliefs=SETS if beliefs else 0, data_trust=default_data_trust(args.judge) if weighed else 0.0)


def _opinion_text(o: Opinion) -> str:
    """An answer's opinion as the command prints it: projected probability, then belief, disbelief and uncertainty."""
    return f"P {o.projection:.3f}  b {o.belief:.3f}  d {o.disbelief:.3f}  u {o.uncertainty:.3f}"


def _opinion_json(o: Opinion) -> dict[str, float]:
    """An answer's opinion as --json writes it."""
    return {"projection": round(o.projection, 6), "belief": round(o.belief, 6), "disbelief": round(o.disbelief, 6),
            "uncertainty": round(o.uncertainty, 6), "base_rate": round(o.base_rate, 6)}


def cmd_ask(args: argparse.Namespace) -> int:
    kb = formats.load(args.kb)
    query = parse_atom(args.query)
    rw = _rewriter(args, kb)
    strategy = STRATEGIES[args.strategy]
    out = rw.ask(query, strategy if args.depth is None else replace(strategy, depth=args.depth))
    rows = sorted(out.answers)
    beliefs = out.beliefs
    if beliefs is not None:
        # Ranked by projected probability, highest first; --answer-threshold drops the answers below it.
        rows = sorted((r for r in rows if beliefs[r].projection >= args.answer_threshold),
                      key=lambda r: (-beliefs[r].projection, r))
    if args.json:
        print(json.dumps({
            "query": str(query), "status": out.status, "detail": out.detail,
            "answers": [list(a) for a in rows],
            **({"beliefs": [_opinion_json(beliefs[r]) for r in rows]} if beliefs is not None else {}),
            "rewrites": sorted([p, c, list(m)] for p, c, m in out.admitted),
            "cost": {"work": out.work, "seconds": round(out.seconds, 3), "expansions": out.expansions,
                     "proposals": out.proposals, "tables": out.tables},
        }, indent=1))
    else:
        for row in rows:
            belief = "" if beliefs is None else f"    {_opinion_text(beliefs[row])}"
            print(f"{query.pred}({', '.join(row)}){belief}")
            if args.explain and out.proofs and row in out.proofs:
                print("\n".join("    " + line for line in out.proofs[row].lines()))
        if args.explain:
            print(f"\nrewrites admitted ({len(out.admitted)}):", file=sys.stderr)
            for p, c, m in sorted(out.admitted, key=lambda k: (k[1], k[0], k[2])):
                print(f"  {Rewrite(c, p, m)}", file=sys.stderr)
        print(f"{out.status}: {plural(len(out.answers), 'answer')}, {out.work:,} work, {out.proposals:,} judge questions, "
              f"{out.seconds:.2f} s", file=sys.stderr)
    return 0 if out.finished else (2 if out.status in ("work_limit", "time_limit") else 1)


def cmd_rewrites(args: argparse.Namespace) -> int:
    kb = formats.load(args.kb)
    rw = _rewriter(args, kb)
    disc = rw.discoverer()
    if args.predicate not in disc.catalogue:
        print(f"{args.predicate} has no gloss in {kb.name}", file=sys.stderr)
        return 1
    admitted = {r.key: r for r in disc.rewrites(args.predicate)}
    if args.snars:
        return _snars(rw, admitted)
    proposals = disc.proposals(args.predicate) if args.all else []
    scores = rw.scorer.score(proposals) if proposals else []
    for r in sorted(admitted.values(), key=lambda r: -r.score):
        print(f"admit   {r.score:.3f}  {r}")
    for p, s in sorted(zip(proposals, scores, strict=True), key=lambda ps: -ps[1]):
        if p.key not in admitted:
            print(f"reject  {s:.3f}  {p.rewrite(s)}")
    print(f"{disc.stats.proposals:,} judge questions at threshold {rw.scorer.threshold:.3f}", file=sys.stderr)
    return 0


def _snars(rw: Rewriter, admitted: dict) -> int:
    """Print each admitted rewrite as the two lines putting it into SNARS's world, its validity believed at its opinion
    and its rule-kit row, strongest first: the MeTTa S-expressions mettabase's reasoner and GoalChainer reason with
    (semrewrite.snars). The judge is named as the evidence's source."""
    rewrites, opinion = rw.rewrite_opinions(admitted)
    source = f"semrewrite {rw.scorer.judge.name}"
    for r in sorted(rewrites, key=lambda r: (-opinion[r.key].projection, r.key)):
        print(*rule_rows(r, opinion[r.key], source), sep="\n")
    return 0


def cmd_engines(args: argparse.Namespace) -> int:
    ready = set(available())
    for name in sorted(ENGINES):
        if name not in ready:
            print(f"{name:10} not installed")
            continue
        caps = make_engine(name).capabilities
        print(f"{name:10} " + ", ".join(f.name for f in fields(caps) if getattr(caps, f.name)))
    return 0


def cmd_strategies(args: argparse.Namespace) -> int:
    print(f"{'strategy':21} {'timing':9} {'memo':5} {'tables':7} {'depth':11} aliases")
    for s in STRATEGIES.values():
        depth = f"{s.depth} {s.depth_counts}" if s.depth else ""
        print(f"{s.name:21} {s.timing.value:9} {'on' if s.memo else 'off':5} {'on' if s.tabled else 'off':7} "
              f"{depth:11} {'merged' if s.canonical else ''}")
    return 0


def cmd_conformance(args: argparse.Namespace) -> int:
    from semrewrite import conformance

    report = conformance.run(make_engine(args.engine), max_examples=args.examples)
    print(report.summary() if hasattr(report, "summary") else report)
    return 0 if getattr(report, "passed", True) else 1


def cmd_convert(args: argparse.Namespace) -> int:
    formats.save(formats.load(args.source), args.target)
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    """Judge sentence pairs read from standard input, one JSON object per line with a premise and a conclusion, and
    write each pair's opinion as a JSON line in the same order, calibrated as answers weigh opinions: at the judge's
    base rate, discounted by its trust, either of which --base-rate and --trust replace for another population of
    pairs. A line not holding a premise and a conclusion that are both strings is refused, naming the line."""
    from semrewrite.judges import make
    from semrewrite.judges.base import SentenceJudge
    from semrewrite.protocols import calibrated

    pairs: list[tuple[str, str]] = []
    for number, line in enumerate(sys.stdin, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            msg = f"line {number} is not JSON: {e}"
            raise ValueError(msg) from e
        if not (isinstance(row, dict) and all(isinstance(row.get(k), str) for k in ("premise", "conclusion"))):
            msg = f"line {number} is not a JSON object holding a premise and a conclusion, both strings"
            raise ValueError(msg)  # bad input, which main() reports as an error, not a programming fault
        pairs.append((row["premise"], row["conclusion"]))
    overrides = {k: v for k, v in (("trust", args.trust), ("base_rate", args.base_rate)) if v is not None}
    judge = make(args.judge, **overrides)
    try:
        if not isinstance(judge, SentenceJudge):
            msg = f"judge {args.judge} does not read sentences"
            raise ValueError(msg)  # noqa: TRY004 - a judge name it cannot use, which main() reports as others
        opinions = calibrated(judge, judge.sentence_opinions(pairs))
    finally:
        judge.close()
    for (premise, conclusion), o in zip(pairs, opinions, strict=True):
        print(json.dumps({"premise": premise, "conclusion": conclusion, **_opinion_json(o)}))
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="semrewrite", description="Semantic rewriting over logic engines.")
    sub = top.add_subparsers(dest="command", required=True)

    def rewriting(p: argparse.ArgumentParser) -> None:
        p.add_argument("kb", help="knowledge base: .kb text, .json or .metta")
        p.add_argument("--judge", default="nli-deberta-large", help="judge name, or oracle to use the file's gold lines")
        p.add_argument("--threshold", type=float, help="override the judge's calibrated threshold")
        p.add_argument("--directional", type=float, metavar="SLACK", help="direction-aware admission with this slack")
        p.add_argument("--retriever", choices=("embedding", "all", "judge"), default="embedding",
                       help="candidates by gloss embedding, every predicate, or ranked by the --prefilter judge")
        p.add_argument("--prefilter", default="nli-deberta-large", metavar="JUDGE",
                       help="the judge that ranks candidates for --retriever judge")
        p.add_argument("--k", type=int, default=8, help="candidates retrieved per expansion")
        p.add_argument("--engine", default="python", choices=sorted(ENGINES))
        p.add_argument("--work", type=int, default=2_000_000, help="work budget in the engine's unit")
        p.add_argument("--seconds", type=float, default=60.0, help="time budget")

    ask = sub.add_parser("ask", help="answer a query")
    rewriting(ask)
    ask.add_argument("query", help='such as "parent(X, tom)"')
    ask.add_argument("--strategy", default="lazy-tabled", choices=sorted(STRATEGIES))
    ask.add_argument("--depth", type=int, help="the most counted steps on any path, counted as the strategy counts "
                     "them, in place of its own bound; 0 removes the bound")
    ask.add_argument("--explain", action="store_true", help="print each answer's derivation and the rewrites admitted")
    ask.add_argument("--trace", action="store_true",
                     help="print each step on standard error as it happens: the engine starting and stopping, "
                          "evaluation reaching a predicate, the candidates retrieved, the judge's verdicts and the "
                          "rewrites admitted")
    ask.add_argument("--beliefs", action="store_true",
                     help="give each answer an opinion that some derivation's rewrites are all valid: its projected "
                          "probability P with belief b, disbelief d and uncertainty u; rank the answers by P")
    ask.add_argument("--answer-threshold", type=float, default=0.0, metavar="P",
                     help="with --beliefs, print only the answers whose projected probability is at least P")
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(run=cmd_ask)

    rewrites = sub.add_parser("rewrites", help="show the rewrites discovery admits for a predicate")
    rewriting(rewrites)
    rewrites.add_argument("predicate")
    rewrites.add_argument("--all", action="store_true", help="also show the rejected proposals with their scores")
    rewrites.add_argument("--snars", action="store_true",
                          help="print each admitted rewrite as two lines of SNARS's world: its validity believed at "
                               "its opinion, and a rule-kit row deriving its conclusion from its premise and validity")
    rewrites.set_defaults(run=cmd_rewrites)

    sub.add_parser("engines", help="list engines and their capabilities").set_defaults(run=cmd_engines)
    sub.add_parser("strategies", help="list strategies and their knobs").set_defaults(run=cmd_strategies)

    conf = sub.add_parser("conformance", help="check an engine against the protocol")
    conf.add_argument("--engine", default="python", choices=sorted(ENGINES))
    conf.add_argument("--examples", type=int, default=200)
    conf.set_defaults(run=cmd_conformance)

    convert = sub.add_parser("convert", help="convert a knowledge base between formats")
    convert.add_argument("source")
    convert.add_argument("target")
    convert.set_defaults(run=cmd_convert)

    judge = sub.add_parser("judge", help="judge sentence pairs read as JSON lines on standard input")
    judge.add_argument("--judge", default="nli-deberta-large", help="judge name")
    judge.add_argument("--trust", type=float, help="the trust to discount by, in place of the calibrated one")
    judge.add_argument("--base-rate", type=float, help="the base rate to set, in place of the calibrated one")
    judge.set_defaults(run=cmd_judge)
    return top


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.run(args)
    except (LookupError, ValueError, CapabilityError) as e:
        # LookupError covers an unknown name and a judge with no calibrated threshold; a KeyError's message is its repr.
        print(f"semrewrite: {e.args[0] if isinstance(e, KeyError) and e.args else e}", file=sys.stderr)
        return 1
    except ImportError as e:
        # A judge, engine or the conformance kit needs a library its extra installs.
        extra = EXTRAS.get((e.name or "").partition(".")[0])
        if extra is None:
            raise
        print(f"semrewrite: {e.name} is not installed; pip install 'semantic-rewriter[{extra}]'", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
