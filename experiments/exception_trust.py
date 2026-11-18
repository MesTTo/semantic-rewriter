"""Purpose: compare gated and ungated claim readings and fit the trust each judge puts in statements about an application's
individuals, on delta-ATOMIC (Rudinger et al., Findings of EMNLP 2020) with a person's name filled in for PersonX and
PersonY, and measure that reading against the one it replaced. Each update is a statement about one person, and its
premise and inference are an application's premise and conclusion, "kim plays games." and "kim wastes time.". The
taken form reads the update with the premise against the inference (semrewrite.claims.read_given), as the check reads
an application's statement; the rejected form, the named conditional, reads it against "If premise, inference."
(semrewrite.claims.read). A strengthener's inference holds and a weakener's does not. The trust maximising the dev
labels' likelihood when each stance projects at the set's base rate after discounting by it
(semrewrite.judges.metrics.fit_trust, as experiments/check_trust.py fits the check trust) is the form's trust, and the
taken form's is the judge's exception trust.

On the test updates it reports, for each form, log loss and AUC, and the defeats, stances whose disbelief reaches the
judge's threshold: weakeners, which a defeat should remove, and strengtheners and other premises' updates about the
same person, which it should not, with a defeat's precision, its weakeners over all its own updates, and recall, the
weakeners it reaches. It then measures the relevance gate as an application meets it: every update of a split names
its own item's people, each claim is written with letters as the check writes it, and for knowledge bases of several
sizes it counts the updates standing out at the deployed spread, a claim's own and others, and how often each form
reads a kept other, filled with the claim's names, as a defeat. It writes results/exception_trust.json, each judge's
with the provenance of the run that measured it, since the file keeps the judges earlier runs measured, and each
judge's exception_trust into semrewrite/judges/calibration.json, every other value there kept.

Run from the repository root, with a GPU for the model judges: python experiments/exception_trust.py [--judges a,b]
[--stances DIR] [--no-gate]. --no-gate measures the two forms alone, for a judge too slow to read the gate as well: the
gate reads the kept pairs of every knowledge-base size, 45,322 for a judge to the forms' 19,956.

--compare-gate delta measures both policies on test without refitting calibration. Each labelled update is used once
as the own statement, with nested contexts of up to 31 updates from other premises. The statement is written and
read in the same form, with either the application's individuals or generic role placeholders. A claim's own premise
and conclusion are not candidate statements, as in the release check. Use --compare-gate actions or wikidata to
measure the same policies through inference. The Wikidata runs replay DeBERTa admission under both readers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import closing, nullcontext
from dataclasses import replace
from itertools import islice, repeat
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from check_trust import defeat_rate, precision_recall, provenance, scored, stances, store
from claim_gate import SIZES

from semrewrite import formats, pairs
from semrewrite.claims import (
    APPLICATIONS,
    BATCH,
    DEFAULT_READER,
    SPREAD,
    UNGATED_BATCH,
    ClaimCheck,
    Reading,
    conditional,
    denied,
    given,
    read,
    read_given,
    robust_z,
    sentence,
    stance,
    standing_out,
    statements,
)
from semrewrite.discovery import Discoverer, ScoreCache
from semrewrite.engines import make as make_engine
from semrewrite.judges import (
    CALIBRATION,
    CHECK_SET,
    NAMES,
    default_base_rate,
    default_check_trust,
    default_data_trust,
    default_exception_trust,
    default_threshold,
)
from semrewrite.judges import make as make_judge
from semrewrite.judges.base import LETTERS, SentenceJudge
from semrewrite.judges.metrics import fit_trust, roc_auc
from semrewrite.kb import KB
from semrewrite.lineage import SETS
from semrewrite.model import Atom, Key, Predicate, Proposal, Rewrite, Rule, mappings
from semrewrite.opinion import Opinion
from semrewrite.protocols import Budget, Engine, Retriever
from semrewrite.retrieve import AllCandidates, EmbeddingRetriever, JudgeRetriever
from semrewrite.rewriter import Outcome, Rewriter, Strategy

OUT = ROOT / "results" / "exception_trust.json"
COMPARISON_OUT = ROOT / "results" / "read_without_gate.json"
INPUTS = ("experiments/exception_trust.py", "experiments/check_trust.py", "semrewrite", "data/raw/delta-atomic")
CONTEXT_SIZES = (1, 2, 4, 8, 16, 32)
SCOPES = ("application", "general")
# Each checkpoint has up to this many contexts, with repeated questions judged once inside the checkpoint.
CHECKPOINT = 64
EXAMPLE_INPUTS = (*INPUTS, "examples/actions_people.kb", "examples/actions_people_believed.kb", "examples/wikidata",
                  "data/raw/wikidata", "experiments/claim_check.py")
# The people PersonX and PersonY become, two distinct per item, drawn with a fixed seed; lowercase, as the check writes
# a constant such as kim.
PEOPLE = ("kim", "sam", "lee", "ana", "joe", "max", "eve", "raj", "amy", "tom", "liz", "ben", "mia", "dan", "zoe", "ian")
# Other premises' updates read against a test claim, about the claim's own people, standing for statements about an
# application's individuals that bear on it not at all.
OTHERS = 2000
SEED = 20261116
Reader = Callable[[SentenceJudge, Sequence[tuple[str, str, str]]], list[Opinion]]
# The forms an update is read in: the one the check takes for an application's statements, and the one it rejected.
FORMS: dict[str, Reader] = {"premise and statement": read_given, "named conditional": read}
TAKEN = "premise and statement"

Item = pairs.DefeasibleItem
Names = tuple[str, str]


def fill(text: str, names: Names) -> str:
    """`text` with PersonX and PersonY named: fill("PersonX calls PersonY", ("kim", "sam")) is "kim calls sam"."""
    return text.replace("PersonX", names[0]).replace("PersonY", names[1])


def people(items: Sequence[Item]) -> dict[str, Names]:
    """Each item's two people, by item id, drawn with SEED in file order."""
    rng = random.Random(SEED)
    named: dict[str, Names] = {}
    for i in items:
        x, y = rng.sample(PEOPLE, 2)
        named[i.id] = (x, y)
    return named


def question(statement: Item, claim: Item, names: Names) -> tuple[str, str, str]:
    """`statement`'s update read against `claim`'s premise and inference, everything about the people `names`."""
    return sentence(fill(statement.update, names)), fill(claim.premise, names), fill(claim.hypothesis, names)


def asked(items: Sequence[Item], named: dict[str, Names]) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Each item's own question, and OTHERS test claims each read against the update of a test item with another
    premise, filled with the claim's people, drawn with a fixed seed."""
    own = [question(i, i, named[i.id]) for i in items]
    test = [i for i in items if i.split == "test"]
    rng = random.Random(SEED)
    others: list[tuple[str, str, str]] = []
    while len(others) < OTHERS:
        claim, statement = rng.sample(test, 2)
        if claim.premise != statement.premise:
            others.append(question(statement, claim, named[claim.id]))
    return own, others


# Time: k x N cosines for k claims and N updates in batches, then per claim and size a median and a median deviation,
# O(k N log N) at the largest size; one judge call per form over the other updates the gate keeps at any size.
def gate(split: str, items: Sequence[Item], named: dict[str, Names], own: dict[str, dict[str, Opinion]],
         judge: SentenceJudge, embedder: EmbeddingRetriever, cache: Path | None) -> dict[str, Any]:
    """The gate as an application meets it on one split: each item's claim with letters against the split's updates,
    each naming its own item's people, by knowledge base size, the claim's own update among them; for each form, how
    often a kept other read about the claim's people is a defeat, and the claim's own update kept and defeating, its
    stance taken from `own`, each form's stance by item id."""
    found = [i for i in items if i.split == split]
    letters = (LETTERS[0], LETTERS[1])
    claims = [conditional(fill(i.premise, letters), fill(i.hypothesis, letters)) for i in found]
    texts = [sentence(fill(i.update, named[i.id])) for i in found]
    kept: dict[int | None, list[tuple[int, int]]] = defaultdict(list)
    own_kept: dict[int | None, list[int]] = defaultdict(list)
    drawn: dict[int | None, int] = defaultdict(int)
    started = time.perf_counter()
    for start in range(0, len(claims), BATCH):
        for c, row in enumerate(embedder.cosines(claims[start:start + BATCH], texts), start):
            others = [j for j in range(len(found)) if found[j].premise != found[c].premise]
            local = random.Random(f"{SEED}:{split}:{c}")
            for size in SIZES:
                pool = others if size is None else local.sample(others, min(size - 1, len(others)))
                drawn[size] += len(pool)
                candidates = [c, *pool]
                for j, z in zip(candidates, robust_z([row[j] for j in candidates]), strict=True):
                    if z < SPREAD:
                        continue
                    if j == c:
                        own_kept[size].append(c)
                    else:
                        kept[size].append((c, j))
    embedding = time.perf_counter() - started
    pairs_kept = sorted({p for rows in kept.values() for p in rows})
    bar = judge.threshold
    out: dict[str, Any] = {"claims": len(claims), "statements": len(texts), "embedding_seconds": round(embedding, 1)}
    for form, reader in FORMS.items():
        questions = [question(found[j], found[c], named[found[c].id]) for c, j in pairs_kept]
        where = None if cache is None else cache / f"{judge.name}.gate.{split}.{form.replace(' ', '-')}.jsonl"
        read_kept = dict(zip(pairs_kept, stances(judge, questions, where, reader), strict=True))
        rows = {}
        for size in SIZES:
            others = kept[size]
            defeats = sum(read_kept[p].disbelief >= bar for p in others)
            mine = [found[c] for c in own_kept[size]]
            rows["all" if size is None else str(size)] = {
                "own_kept": {kind: sum(i.strengthens == (kind == "strengthener") for i in mine) / len(claims)
                             for kind in ("weakener", "strengthener")},
                "own_defeats": {kind: sum(i.strengthens == (kind == "strengthener")
                                          and own[form][i.id].disbelief >= bar for i in mine) / len(claims)
                                for kind in ("weakener", "strengthener")},
                "others_kept_per_claim": len(others) / len(claims),
                "others_kept_per_statement": len(others) / drawn[size],
                "kept_other_defeat_rate": defeats / len(others) if others else None,
            }
        out[form] = rows
    return out


# Time: one shuffled index per distinct claim, with one scan to find its first 31 eligible updates.
def contexts(items: Sequence[Item]) -> list[tuple[Item, Names, list[Item]]]:
    """One own update and a shared, nested draw of distractors for each test item.

    Items with the same claim use the same names and random stream, so their different own updates are compared
    in the same background when the draws are eligible. Texts must be distinct in both scopes.
    """
    test = [i for i in items if i.split == "test"]
    named = people(test)
    same_premise: dict[str, set[str]] = defaultdict(set)
    for item in test:
        same_premise[item.premise].add(sentence(item.update))
    out = []
    for (premise, hypothesis), claim_items in _claim_groups(test).items():
        names = named[claim_items[0].id]
        used = same_premise[premise] | {sentence(premise), sentence(hypothesis)}
        named_used = {fill(text, names) for text in used}
        rng = random.Random(f"{SEED}:{premise}:{hypothesis}")
        others = []
        for j in rng.sample(range(len(test)), len(test)):
            other = test[j]
            text = sentence(other.update)
            named_text = fill(text, names)
            if text in used or named_text in named_used:
                continue
            others.append(other)
            used.add(text)
            named_used.add(named_text)
            if len(others) == CONTEXT_SIZES[-1] - 1:
                break
        if len(others) != CONTEXT_SIZES[-1] - 1:
            msg = f"{claim_items[0].id} has only {len(others)} eligible distractors"
            raise ValueError(msg)
        out.extend((claim, names, [claim, *others]) for claim in claim_items)
    return out


def _claim_groups(items: Sequence[Item]) -> dict[tuple[str, str], list[Item]]:
    groups: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for item in items:
        groups[item.premise, item.hypothesis].append(item)
    return groups


def comparison_questions(scope: str, item: Item, names: Names, pool: Sequence[Item]) -> list[tuple[str, str, str]]:
    """Write the same measured questions for the model run and its cached replay."""
    if scope == "application":
        return [question(other, item, names) for other in pool]
    return [(sentence(other.update), item.premise, item.hypothesis) for other in pool]


# Time: 32 cosines needed per claim, computed in blocks of BATCH claims over their union of statements.
def comparison_inputs(items: Sequence[Item], embedder: EmbeddingRetriever) -> dict[str, list]:
    """Prepare exactly the questions and release gate choices both arms will share."""
    out: dict[str, list] = {scope: [] for scope in SCOPES}
    letters = (LETTERS[0], LETTERS[1])
    draws = contexts(items)
    for scope, rows in out.items():
        for start in range(0, len(draws), BATCH):
            claims = []
            asked = []
            for item, names, pool in draws[start:start + BATCH]:
                if scope == "application":
                    claims.append(conditional(fill(item.premise, letters), fill(item.hypothesis, letters)))
                else:
                    claims.append(conditional(item.premise, item.hypothesis))
                asked.append(comparison_questions(scope, item, names, pool))
            texts = list(dict.fromkeys(s for questions in asked for s, _, _ in questions))
            positions = {s: i for i, s in enumerate(texts)}
            for (item, _, _), questions, cosines in zip(draws[start:start + BATCH], asked,
                                                      embedder.cosines(claims, texts), strict=True):
                sims = [cosines[positions[s]] for s, _, _ in questions]
                rows.append((item.strengthens, questions, [standing_out(sims[:h]) for h in CONTEXT_SIZES]))
        print(f"prepared release gate for {len(rows):,} {scope} contexts", flush=True)
    return out


def record_comparison(counts: dict[str, int], kept: Sequence[int], defeats: Sequence[bool], strengthens: bool,
                      stances_per_statement: int, *, ungated: bool) -> None:
    """Count defeat events and cost for a fresh check over a single context.

    Source grouping asks every distinct ordered pair. The ungated check skips this work after a hard defeat.
    """
    other = sum(defeats[j] for j in kept if j)
    counts["own_strengthener_defeats" if strengthens else "own_weakener_defeats"] += (0 in kept and defeats[0])
    counts["other_defeats"] += other
    counts["contexts_with_other_defeat"] += bool(other)
    counts["stance_pairs"] += len(kept) * stances_per_statement
    groups_sources = not ungated or not any(defeats[j] for j in kept)
    sources = len(kept) * (len(kept) - 1) if groups_sources else 0
    counts["source_pairs"] += sources
    counts["judge_calls"] += bool(kept)
    counts["judge_calls"] += (sources + UNGATED_BATCH - 1) // UNGATED_BATCH if ungated else bool(sources)


# Time: one stance per distinct question per checkpoint, then O(sum(h)) counted events per context.
def compare_delta(judge: SentenceJudge, inputs: dict[str, list], cache: Path | None) -> dict:
    """Read the largest context once, then measure both policies on each nested prefix."""
    bar = judge.calibrated_threshold
    if bar is None:
        msg = f"{judge.name} has no calibrated threshold"
        raise ValueError(msg)
    counts = ("own_weakener_defeats", "own_strengthener_defeats", "other_defeats",
              "contexts_with_other_defeat", "stance_pairs", "source_pairs", "judge_calls")
    out = {}
    for scope, rows in inputs.items():
        started = time.perf_counter()
        reader = read_given if scope == "application" else read
        metrics = {policy: {str(h): dict.fromkeys(counts, 0) for h in CONTEXT_SIZES}
                   for policy in ("gated", "ungated")}
        for start in range(0, len(rows), CHECKPOINT):
            batch = rows[start:start + CHECKPOINT]
            asked = list(dict.fromkeys(q for _, questions, _ in batch for q in questions))
            where = None if cache is None else cache / f"{judge.name}.no-gate.{scope}.{start}.jsonl"
            stance = dict(zip(asked, stances(judge, asked, where, reader), strict=True))
            for strengthens, questions, gated in batch:
                defeats = [stance[q].disbelief >= bar for q in questions]
                for h, kept in zip(CONTEXT_SIZES, gated, strict=True):
                    for policy, selected in (("gated", kept), ("ungated", range(h))):
                        record_comparison(metrics[policy][str(h)], selected, defeats, strengthens,
                                          1 if judge.reads_contradiction else 2, ungated=policy == "ungated")
            if start % (CHECKPOINT * 4) == 0 or start + CHECKPOINT >= len(rows):
                print(f"{judge.name}: {scope} {min(start + CHECKPOINT, len(rows)):,}/{len(rows):,}", flush=True)
        strengths = sum(strengthens for strengthens, _, _ in rows)
        out[scope] = {"weakener_contexts": len(rows) - strengths, "strengthener_contexts": strengths,
                      "seconds": round(time.perf_counter() - started, 1), **metrics}
    return out


def compare_gate(names: Sequence[str], cache: Path | None, part: str, *, short_circuit: bool = False,
                 samples: Mapping[str, int] | None = None) -> None:
    """Write the paired test or example arm without changing any previous calibration."""
    state = provenance(INPUTS if part == "delta" else EXAMPLE_INPUTS)
    out = json.loads(COMPARISON_OUT.read_text()) if COMPARISON_OUT.is_file() else {}
    out.update({"set": CHECK_SET, "split": "test", "sizes": CONTEXT_SIZES, "seed": SEED, "spread": SPREAD})
    if part == "wikidata":
        wikidata = wikidata_cases()
    with EmbeddingRetriever() as embedder:
        if part == "delta":
            inputs = comparison_inputs(pairs.delta_atomic_items(), embedder)
        elif part == "actions":
            actions = [formats.load(file) for file in sorted((ROOT / "examples").glob("actions_people*.kb"))]
            for kb in actions:
                catalogue = kb.catalogue()
                prepare_cosines(kb, embedder, [(p, c) for c in catalogue.values() for p in catalogue.values()
                                               if p.arity == c.arity])
        else:
            for _, kb, _, admission, _ in wikidata:
                embedder.cosines((), [*statements(kb), *(conditional(*pair) for pair in admission.table)])
    for name in names:
        measured = out.setdefault("judges", {}).setdefault(name, {})
        judge = make_judge(name)
        if not isinstance(judge, SentenceJudge):
            msg = f"{name} does not read sentences"
            raise TypeError(msg)
        with judge:
            if part == "delta":
                result = compare_delta(judge, inputs, cache)
            elif part == "actions":
                result = compare_actions(judge, actions, embedder)
            else:
                ordered = {}
                for case in compare_wikidata(judge, wikidata, cache, embedder, short_circuit=short_circuit,
                                             samples=samples):
                    ordered[case["case"]] = case
                    measured.update({"threshold": judge.calibrated_threshold, "check_trust": judge.check_trust,
                                     "exception_trust": judge.exception_trust, part: list(ordered.values()),
                                     f"{part}_provenance": state})
                    save_comparison(out, name)
                result = list(ordered.values())
            measured.update({"threshold": judge.calibrated_threshold, "check_trust": judge.check_trust,
                             "exception_trust": judge.exception_trust, part: result, f"{part}_provenance": state})
        save_comparison(out, name)


def save_comparison(out: dict, name: str) -> None:
    pending = COMPARISON_OUT.with_suffix(".pending")
    pending.write_text(json.dumps(out, indent=1) + "\n")
    os.replace(pending, COMPARISON_OUT)
    print(f"wrote {COMPARISON_OUT.relative_to(ROOT)} for {name}", flush=True)


def prepare_cosines(kb: KB, embedder: EmbeddingRetriever, predicates: Sequence[tuple[Predicate, Predicate]]) -> None:
    """Cache the vectors needed by the actions gate so the reader and embedder need not share GPU memory."""
    claims = [conditional(*Proposal(p, c, mapping).sentences(LETTERS))
              for p, c in predicates for mapping in mappings(c.arity)]
    embedder.cosines((), [*statements(kb), *claims])


class RecordedAdmissions(SentenceJudge):
    """Replay the release's candidate scores while a different model reads the statements."""

    reads_contradiction = True

    def __init__(self, judge: SentenceJudge, table: dict[tuple[str, str], Opinion]) -> None:
        super().__init__(judge.name, judge.threshold, judge.fillers)
        self.trust = judge.trust
        self.base_rate = judge.base_rate
        self.table = table

    def _opinions(self, pairs: Sequence[tuple[str, str]]) -> list[Opinion]:
        return [self.table[pair] for pair in pairs]


class RecordedRanking:
    """The same top-k candidates for both policies and both readers."""

    name = DEFAULT_READER

    def __init__(self, rankings: dict[str, list[tuple[str, float]]]) -> None:
        self.rankings = rankings

    def top_k(self, goal: Predicate, among: Mapping[str, Predicate], k: int) -> list[tuple[str, float]]:
        return self.rankings[goal.name][:k]


class SavedReader(SentenceJudge):
    """Keep exact sentence opinions across interrupted Wikidata runs."""

    def __init__(self, judge: SentenceJudge, db: sqlite3.Connection,
                 priority: sqlite3.Connection | None = None) -> None:
        super().__init__(judge.name, judge.threshold, judge.fillers)
        self.base_rate = judge.base_rate
        self.check_trust = judge.check_trust
        self.exception_trust = judge.exception_trust
        self.calibrated_threshold = judge.calibrated_threshold
        self.judge = judge
        self.db = db
        self.priority = priority
        db.execute("CREATE TABLE IF NOT EXISTS readings (p TEXT, h TEXT, b REAL, d REAL, u REAL, PRIMARY KEY (p, h))")

    # Time: one indexed database lookup per pair, one model evaluation and insert per distinct missing pair.
    def _opinions(self, pairs: Sequence[tuple[str, str]]) -> list[Opinion]:
        cached: dict[tuple[str, str], Opinion] = {}
        missing = []
        for pair in dict.fromkeys(pairs):
            row = self.db.execute("SELECT b, d, u FROM readings WHERE p = ? AND h = ?", pair).fetchone()
            if row is None:
                missing.append(pair)
            else:
                cached[pair] = Opinion(*row)
        fresh = self.judge.sentence_opinions(missing)
        with self.db:
            self.db.executemany("INSERT INTO readings VALUES (?, ?, ?, ?, ?)",
                                ((*pair, o.belief, o.disbelief, o.uncertainty)
                                 for pair, o in zip(missing, fresh, strict=True)))
        cached.update(zip(missing, fresh, strict=True))
        return [cached[pair] for pair in pairs]

    def order(self, questions: Iterable[tuple[str, str, str]]) -> Iterable[tuple[str, str, str]]:
        """Use DeBERTa as a scheduling hint only. Every candidate remains available to Qwen."""
        if self.priority is None:
            return questions

        def rank(question: tuple[str, str, str]) -> float:
            text, premise, conclusion = question
            assert self.priority is not None
            found = self.priority.execute("SELECT d FROM readings WHERE p=? AND h=?",
                                          (given(premise, text), sentence(conclusion))).fetchone()
            return -found[0] if found is not None else 0.0

        return sorted(questions, key=rank)


class SavedContradictionReader(SavedReader):
    """DeBERTa's cached opinions already contain the contradiction score."""

    reads_contradiction = True


class SourcePairCounts:
    """Count the new ordered statement pairs a source-reading batch would reserve."""

    def __init__(self) -> None:
        self._seen: dict[int, int] = {}

    # Time: h bitset operations per group of h statement indices. Space: one bit per possible pair of those indices.
    def reserve(self, groups: Sequence[Sequence[int]]) -> int:
        count = 0
        for group in groups:
            mask = sum(1 << j for j in group)
            for j in group:
                peers = mask ^ (1 << j)
                before = self._seen.get(j, 0)
                count += (peers & ~before).bit_count()
                self._seen[j] = peers | before
        return count


class WikidataAnswerCheck(ClaimCheck):
    """Read all stances for answer membership and account for source-grouping work."""

    def __init__(self, kb: KB, admission: SentenceJudge, reader: SentenceJudge, embedder: EmbeddingRetriever | None,
                 *, short_circuit: bool = False) -> None:
        trust = reader.exception_trust
        if trust is None or trust >= 1.0:
            msg = f"Wikidata answer accounting needs {reader.name}'s exception trust to be less than one"
            raise ValueError(msg)
        super().__init__(kb, admission, embedder, reader=reader, spread=None)
        self.source_counts = SourcePairCounts()
        self.accounted_pairs = 0
        self.accounted_calls = 0
        self.short_circuit = short_circuit

    def _paraphrases(self, kept: Sequence[Sequence[int]], reader: SentenceJudge) -> None:
        pairs = self.source_counts.reserve(kept)
        self.accounted_pairs += pairs
        self.accounted_calls += (pairs + UNGATED_BATCH - 1) // UNGATED_BATCH

    def _sources(self, texts: Sequence[str], reader: SentenceJudge) -> dict[str, str]:
        # Every surviving application keeps positive weight under the calibrated trust.
        return {text: text for text in texts}

    def _store_application(self, key: Key, row: tuple[str, ...], claim: tuple[str, str], readings: tuple[Reading, ...],
                           bar: float) -> None:
        # Hard defeats still carry their readings. A surviving application needs no probability leaf.
        super()._store_application(key, row, claim, tuple(r for r in readings if r.stance.disbelief >= bar), bar)

    def _read_all(self, contexts, *, application: bool) -> None:
        if not self.short_circuit or not application:
            super()._read_all(contexts, application=application)
            return
        reader, trust = self._reader_for(APPLICATIONS, "without the similarity gate")
        bar = self.bar(APPLICATIONS)
        assert trust is not None
        todo = [(key, row, list(js), self.instance(key, row)) for key, row, js in contexts]
        count = sum(len(js) for _, _, js, _ in todo)
        self.usage["applications"] = self.usage.get("applications", 0) + len(todo)
        self.usage["pairs"] += count * (1 if reader.reads_contradiction else 2)
        self.judge_calls += (count + UNGATED_BATCH - 1) // UNGATED_BATCH
        questions = (zip((self._sentences[j] for j in js), repeat(premise), repeat(conclusion))
                     for _, _, js, (premise, conclusion) in todo)
        if isinstance(reader, SavedReader):
            questions = (iter(reader.order(group)) for group in questions)
        defeated = first_defeats(reader, questions, bar)
        for i, (key, row, js, claim) in enumerate(todo):
            if i in defeated:
                text, mine = defeated[i]
                readings = (Reading(text, None, mine, mine.with_base_rate(reader.base_rate).discount(trust), text),)
            else:
                self._paraphrases([js], reader)
                readings = ()
            self._store_application(key, row, claim, readings, bar)


def first_defeats(reader: SentenceJudge, questions: Iterable[Iterable[tuple[str, str, str]]],
                  bar: float) -> dict[int, tuple[str, Opinion]]:
    """Find one calibrated defeat per application, in batches, retaining the exact answer membership."""
    pending = {i: iter(group) for i, group in enumerate(questions)}
    defeated: dict[int, tuple[str, Opinion]] = {}
    width = 1
    while pending:
        next_pending = {}
        batch = []
        for i, stream in pending.items():
            selected = list(islice(stream, width))
            if len(selected) == width:
                next_pending[i] = stream
            batch.extend((i, text, given(premise, text), conclusion) for text, premise, conclusion in selected)
        for start in range(0, len(batch), UNGATED_BATCH):
            chunk = [item for item in batch[start:start + UNGATED_BATCH] if item[0] not in defeated]
            if reader.reads_contradiction:
                scored = zip(chunk, reader.sentence_opinions([(p, sentence(c)) for _, _, p, c in chunk]), strict=True)
            else:
                negative = reader.sentence_opinions([(p, denied(c)) for _, _, p, c in chunk])
                candidates = [(item, n) for item, n in zip(chunk, negative, strict=True) if n.belief >= bar]
                positive = reader.sentence_opinions([(p, sentence(c)) for (_, _, p, c), _ in candidates])
                scored = ((item, stance(o, n)) for (item, n), o in zip(candidates, positive, strict=True))
            for (i, text, _, _), opinion in scored:
                if opinion.disbelief >= bar and i not in defeated:
                    defeated[i] = text, opinion
        pending = {i: stream for i, stream in next_pending.items() if i not in defeated}
        width = min(width * 2, 16)
    return defeated


class WikidataMembershipRewriter(Rewriter):
    """Find all applications with one derivation per row and no probability calculation."""

    def _witnessed(self, query: Atom, out: Outcome, rewrites: tuple[Rewrite, ...]) -> bool:
        # The ordinary answer-only shortcut visits just one witness per answer. The comparison reads every
        # application in the query closure before deciding which answers survive.
        return False


def example_run(kb: KB, admission: SentenceJudge, reader: SentenceJudge, engine: Engine, retriever: Retriever,
                embedder: EmbeddingRetriever, strategy: Strategy | str, *, ungated: bool, work: int,
                k: int = 8, account_sources: bool = False, short_circuit: bool = False) -> dict:
    """Answer a fixture and record its stance calls and source-grouping work."""
    check = (WikidataAnswerCheck(kb, admission, reader, embedder, short_circuit=short_circuit) if account_sources
             else ClaimCheck(kb, admission, embedder, reader=reader, spread=None if ungated else SPREAD))
    rewriter_type = WikidataMembershipRewriter if account_sources else Rewriter
    rewriter = rewriter_type(kb, admission, engine, retriever, k=k, beliefs=0 if account_sources else SETS,
                             budget=Budget(work=work, seconds=math.inf), data_trust=default_data_trust(admission.name),
                             claims=check)
    policy = "ungated" if ungated else "gated"
    print(f"{reader.name}: {kb.name}, {policy}, starting", flush=True)
    measured_before = reader.usage.copy()
    live_reader = reader.judge if isinstance(reader, SavedReader) else reader
    physical_before = live_reader.usage.copy()
    result = rewriter.ask(kb.queries[0], strategy)
    projections = result.beliefs
    if not result.finished or (not account_sources and projections is None):
        msg = f"{kb.name}: {result.status} {result.detail}"
        raise RuntimeError(msg)
    applications = result.applications or {}
    accounted_pairs = check.accounted_pairs if isinstance(check, WikidataAnswerCheck) else 0
    accounted_calls = check.accounted_calls if isinstance(check, WikidataAnswerCheck) else 0
    pairs = check.usage["pairs"] + accounted_pairs
    calls = check.judge_calls + accounted_calls
    print(f"{reader.name}: {kb.name}, {policy}, {len(result.answers):,} answers, "
          f"{pairs:,} check pairs in {calls:,} calls ({accounted_pairs:,} source pairs counted)", flush=True)
    return {"answers": [[list(row), projections[row].projection if projections is not None else None]
                        for row in sorted(result.answers)],
            "rewrites_defeated": len(result.defeated), "rewrites_admitted": len(result.admitted),
            "applications_read": len(applications), "applications_defeated": sum(a.defeated for a in applications.values()),
            "check_pairs": pairs, "reader_calls": calls, "admission_pairs": rewriter.scorer.computed,
            "accounted_source_pairs": accounted_pairs, "accounted_source_calls": accounted_calls,
            **({"queried_pairs": reader.usage["pairs"] - measured_before["pairs"],
                "queried_calls": reader.usage["calls"] - measured_before["calls"],
                "physical_pairs": live_reader.usage["pairs"] - physical_before["pairs"],
                "physical_calls": live_reader.usage["calls"] - physical_before["calls"]}
               if short_circuit else {}),
            "seconds": round(result.seconds, 2)}


def compare_actions(judge: SentenceJudge, knowledge_bases: Sequence[KB], embedder: EmbeddingRetriever) -> list[dict]:
    """Ask both people fixtures at their release evaluation's two admission bars."""
    from claim_check import LOW

    assert judge.calibrated_threshold is not None
    calibrated = judge.calibrated_threshold
    results = []
    engine = make_engine("python")
    retriever = AllCandidates()
    for kb in knowledge_bases:
        for threshold in (calibrated, LOW):
            judge.threshold = threshold
            results.append({"case": kb.name, "admission": judge.name, "admission_bar": threshold,
                            **{policy: example_run(kb, judge, judge, engine, retriever, embedder, "lazy-tabled",
                                                   ungated=policy == "ungated", work=50_000_000)
                               for policy in ("gated", "ungated")}})
    judge.threshold = calibrated
    return results


WikidataCase = tuple[str, KB, set[str], RecordedAdmissions, RecordedRanking]


def wikidata_cases() -> list[WikidataCase]:
    """Prepare the release's DeBERTa candidate cascade and calibrated admission before loading either reader."""
    from examples.wikidata.evaluate import QUESTIONS, K, _question

    cases = []
    judge = make_judge(DEFAULT_READER)
    assert isinstance(judge, SentenceJudge)
    with judge:
        retriever = JudgeRetriever(judge)
        for name in QUESTIONS:
            _, kb, truth, _ = _question(name)
            catalogue = kb.catalogue()
            goals = {q.pred for q in kb.queries}
            # Depth one traverses the KB's own rules at no cost, but cannot expand a second rewrite.
            pending = list(goals)
            while pending:
                pred = pending.pop()
                next_goals = {atom.pred for rule in kb.rules if rule.head.pred == pred for atom in rule.body} - goals
                pending.extend(sorted(next_goals))
                goals.update(next_goals)
            disc = Discoverer(catalogue, retriever, ScoreCache(judge), k=K)
            ranks = {g: disc.candidates(g) for g in sorted(goals)}
            opinions = {}
            for goal, rank in ranks.items():
                proposals = disc.proposals(goal, [candidate for candidate, _ in rank])
                textual = [proposal.sentences(judge.fillers) for proposal in proposals]
                opinions.update(zip(textual, judge.sentence_opinions(textual), strict=True))
            cases.append((name, kb, truth, RecordedAdmissions(judge, opinions), RecordedRanking(ranks)))
    return cases


SAMPLE_BLOCK = 4
WIKIDATA_SAMPLE_CASES = ("alphabet", "leonardo")


# Time: one scan of the stated edges, then one graph search and witness check per pass that disqualifies an edge.
def sample_relation(kb: KB, admission: SentenceJudge, reader: SentenceJudge, retrieval: Retriever,
                    chosen: Sequence, k: int) -> dict:
    """Find whether the sample has a surviving proof in the checked relation graph."""
    started = time.perf_counter()
    query = kb.queries[0]
    pred, subject = query.pred, query.args[0]
    transitive = Rule(Atom(pred, ("X", "Z")), (Atom(pred, ("X", "Y")), Atom(pred, ("Y", "Z"))))
    if len(query.args) != 2 or kb.rules not in ((), (transitive,)):
        raise ValueError("a sampled relation must be binary with at most its transitive rule")
    scorer = ScoreCache(admission)
    disc = Discoverer(kb.catalogue(), retrieval, scorer, k=k)
    rewrites = disc.rewrites(pred)
    if any(rule.premise == pred for rule in rewrites):
        raise ValueError("a sampled relation requires rewrites from stated predicates")
    by_pred = defaultdict(list)
    for rewrite in rewrites:
        by_pred[rewrite.premise].append(rewrite)
    edges: dict[str, dict[str, list[Key | None]]] = defaultdict(lambda: defaultdict(list))
    for fact in kb.facts:
        if fact.pred == pred:
            left, right = fact.args
            edges[left][right].append(None)
        for rewrite in by_pred[fact.pred]:
            left, right = (fact.args[i] for i in rewrite.mapping)
            edges[left][right].append(rewrite.key)
    check = WikidataAnswerCheck(kb, admission, reader, None, short_circuit=True)
    if check.reads:
        raise ValueError("a sampled relation requires admission independent of general statements")
    disabled: set[tuple[Key, tuple[str, ...]]] = set()
    kept = set()
    pending = {row[1] for row, _ in chosen}
    live = reader.judge if isinstance(reader, SavedReader) else reader
    physical_before = live.usage.copy()
    while pending:
        # One breadth-first search gives a candidate proof for every remaining sampled answer.
        parents: dict[str, tuple[str, Key | None]] = {}
        frontier = [subject]
        for left in frontier:
            for right, sources in edges[left].items():
                if right in parents:
                    continue
                choices = [key for key in sources if key is None or (key, (left, right)) not in disabled]
                if choices:
                    parents[right] = left, None if None in choices else choices[0]
                    if kb.rules and right != subject:
                        frontier.append(right)
        candidate = {}
        wanted = defaultdict(set)
        for answer in sorted(pending & parents.keys()):
            proof = set()
            at = answer
            while True:
                left, key = parents[at]
                if key is not None:
                    row = left, at
                    proof.add((key, row))
                    wanted[key].add(row)
                at = left
                if at == subject:
                    break
            candidate[answer] = proof
        applications = check.applications(wanted)
        new = {key for key, app in applications.items() if app.defeated}
        disabled.update(new)
        kept.update(answer for answer, proof in candidate.items() if not proof & new)
        pending = {answer for answer, proof in candidate.items() if proof & new}
    return {"answers": [[[subject, answer], None] for answer in sorted(kept)],
            "physical_pairs": live.usage["pairs"] - physical_before["pairs"],
            "physical_calls": live.usage["calls"] - physical_before["calls"], "seconds": round(time.perf_counter() - started, 2)}


def sample_permutation(answers: Sequence) -> list:
    """A seeded uniform permutation whose prefixes are the requested samples."""
    return random.Random(SEED).sample(list(answers), len(answers))


def add_sample_block(total: dict, block: dict) -> dict:
    """Accumulate the answers, new model work and timings of independent sample blocks."""
    if not total:
        return block
    return {key: total[key] + value for key, value in block.items()}


class CachedDeberta(SentenceJudge):
    """Replay DeBERTa's saved model opinions, with an error on any cache miss."""

    reads_contradiction = True

    def __init__(self, db: sqlite3.Connection) -> None:
        name = DEFAULT_READER
        super().__init__(name, default_threshold(name))
        self.calibrated_threshold = self.threshold
        self.base_rate = default_base_rate(name)
        self.check_trust = default_check_trust(name)
        self.exception_trust = default_exception_trust(name)
        self.db = db

    def _opinions(self, pairs: Sequence[tuple[str, str]]) -> list[Opinion]:
        opinions = []
        for pair in pairs:
            row = self.db.execute("SELECT b, d, u FROM readings WHERE p = ? AND h = ?", pair).fetchone()
            if row is None:
                raise ValueError(f"missing engine stance in the cached DeBERTa audit: {pair!r}")
            opinions.append(Opinion(*row))
        return opinions


def verify_wikidata_witnesses(cache: Path) -> None:
    """Compare kept row identities for every gated answer using the full DeBERTa run and its cache."""
    from examples.wikidata.evaluate import K

    out = json.loads(COMPARISON_OUT.read_text())
    exact = {case["case"]: case for case in out["judges"][DEFAULT_READER]["wikidata"] if "sample" not in case}
    if any(name not in exact for name in WIKIDATA_SAMPLE_CASES):
        raise ValueError("the witness audit needs full DeBERTa runs for Alphabet and Leonardo")
    path = cache / f"{DEFAULT_READER}.wikidata-readings.sqlite"
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        cases = wikidata_cases()
        reader = CachedDeberta(db)
        verdict = {}
        for name, kb, _, admission, retrieval in cases:
            if name not in WIKIDATA_SAMPLE_CASES:
                continue
            baseline = exact[name]
            if (baseline["admission"], baseline["admission_bar"]) != (admission.name, admission.threshold):
                raise ValueError(f"{name}: DeBERTa witness audit requires the full run's admission calibration")
            chosen = baseline["gated"]["answers"]
            observed = sample_relation(kb, admission, reader, retrieval, chosen, K)
            found = {tuple(row) for row, _ in observed["answers"]}
            gated = {tuple(row) for row, _ in chosen}
            expected = {tuple(row) for row, _ in baseline["ungated"]["answers"]} & gated
            if found != expected:
                raise ValueError(f"{name}: witness search misses {sorted(expected - found)} and adds {sorted(found - expected)}")
            verdict[name] = {"reader": reader.name, "compared": len(gated), "kept": len(found), "lost": len(gated - found)}
            print(f"{name}: witness search and full engine agree on all {len(gated):,} answers", flush=True)
    out["wikidata_witness_audit"] = verdict
    save_comparison(out, "the cached DeBERTa witness audit")


def compare_wikidata(judge: SentenceJudge, cases: Sequence[WikidataCase], cache: Path | None,
                     embedder: EmbeddingRetriever, *, short_circuit: bool = False,
                     samples: Mapping[str, int] | None = None) -> Iterator[dict]:
    """Ask the release's depth-one Wikidata questions under the two reading policies."""
    from examples.wikidata.evaluate import QUESTIONS, STRATEGY, WORK, K

    engine = make_engine("pymetta")
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    where = ":memory:" if cache is None else str(cache / f"{judge.name}.wikidata-readings.sqlite")
    prior = None if cache is None else cache / f"{DEFAULT_READER}.wikidata-readings.sqlite"
    more = (closing(sqlite3.connect(prior.resolve().as_uri() + "?mode=ro", uri=True))
            if short_circuit and not judge.reads_contradiction and prior is not None and prior.is_file()
            else nullcontext(None))
    with closing(sqlite3.connect(where)) as db, more as priority:
        saved = SavedContradictionReader if judge.reads_contradiction else SavedReader
        reader = saved(judge, db, priority) if cache is not None else judge
        for name, kb, truth, admission, retrieval in cases:
            strategy = replace(STRATEGY, depth=1)
            gated = example_run(kb, admission, reader, engine, retrieval, embedder, strategy,
                                ungated=False, work=WORK, k=K)
            sample = {}
            if samples and name in samples:
                population = len(gated["answers"])
                chosen = sample_permutation(gated["answers"])[:samples[name]]
                gated["answers"] = chosen
                ungated: dict = {}
                for start in range(0, max(len(chosen), 1), SAMPLE_BLOCK):
                    block = sample_relation(kb, admission, reader, retrieval, chosen[start:start + SAMPLE_BLOCK], K)
                    ungated = add_sample_block(ungated, block)
                sample = {"sample": {"population": population, "size": len(chosen), "seed": SEED}}
            else:
                ungated = example_run(kb, admission, reader, engine, retrieval, embedder, strategy,
                                      ungated=True, work=WORK, k=K, account_sources=True, short_circuit=short_circuit)
            runs = {"gated": gated, "ungated": ungated}
            for outcome in runs.values():
                entities = QUESTIONS[name].answers_of(tuple(row) for row, _ in outcome["answers"])
                outcome["reference_found"] = sorted(entities & truth)
            yield {"case": name, "admission": admission.name, "admission_bar": admission.threshold,
                   "reference_answers": sorted(truth), **sample, **runs}


def measure(name: str, items: Sequence[Item], rate: float, cache: Path | None, *, gated: bool = True) -> dict[str, Any]:
    """Judge `name`'s trust fitted on the dev items for each form, its scores and defeats on dev and test, and, unless
    `gated` is False, the gate on each split."""
    judge = make_judge(name)
    if not isinstance(judge, SentenceJudge):
        msg = f"judge {name} does not read sentences, so the claim check cannot use it"
        raise ValueError(msg)  # noqa: TRY004 - a judge name it cannot use, as the command line refuses it
    named = people(items)
    own, others = asked(items, named)
    labels = {s: [i.strengthens for i in items if i.split == s] for s in ("dev", "test")}
    bar = judge.threshold
    out: dict[str, Any] = {"trust": judge.trust, "check_trust": judge.check_trust, "threshold": bar,
                           "reads_contradiction": judge.reads_contradiction, "forms": {}}
    embedder = EmbeddingRetriever()
    by_item: dict[str, dict[str, Opinion]] = {}
    started = time.perf_counter()
    try:
        for form, reader in FORMS.items():
            where = None if cache is None else cache / f"{name}.{form.replace(' ', '-')}.jsonl"
            found = stances(judge, [*own, *others], where, reader)
            read_own, read_others = found[:len(own)], found[len(own):]
            by_item[form] = {i.id: o for i, o in zip(items, read_own, strict=True)}
            split = {s: [o for o, i in zip(read_own, items, strict=True) if i.split == s] for s in ("dev", "test")}
            trust = fit_trust([o.with_base_rate(rate) for o in split["dev"]], labels["dev"])
            result: dict[str, Any] = {"trust": trust}
            for s in ("dev", "test"):
                opinions, ys = split[s], labels[s]
                result[s] = {
                    "rows": len(opinions),
                    "at_trust": scored(opinions, ys, rate, trust),
                    "at_base_rate": scored(opinions, ys, rate, 0.0),
                    "auc": roc_auc([o.belief - o.disbelief for o in opinions], ys),
                    "defeat": {"weakener": defeat_rate([o for o, y in zip(opinions, ys, strict=True) if not y], bar),
                               "strengthener": defeat_rate([o for o, y in zip(opinions, ys, strict=True) if y], bar)},
                    **precision_recall(opinions, ys, bar),
                }
            result["test"]["defeat"]["other_same_person"] = defeat_rate(read_others, bar)
            out["forms"][form] = result
        if gated:
            out["gate"] = {s: gate(s, items, named, by_item, judge, embedder, cache) for s in ("dev", "test")}
    finally:
        judge.close()
        embedder.close()
    out["seconds"] = round(time.perf_counter() - started, 1)
    out["exception_trust"] = out["forms"][TAKEN]["trust"]
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--judges", default="nli-deberta-large,typical-qwen3-4b", help="comma-separated judge names")
    parser.add_argument("--stances", type=Path, help="directory to keep each judge's stances in, and reuse them from")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--no-gate", action="store_true", help="measure the two calibration forms without the gate")
    modes.add_argument("--compare-gate", choices=("delta", "actions", "wikidata"),
                       help="compare gated and ungated at release calibration")
    modes.add_argument("--verify-wikidata-witnesses", action="store_true",
                       help="check witness answers against the full DeBERTa engine run and saved stances")
    parser.add_argument("--short-circuit", action="store_true", help="short-circuit hard defeats for Wikidata answers")
    parser.add_argument("--wikidata-sample", metavar="CASE=SIZE", action="append", default=[],
                        help="fixed size for a sampled Leonardo or Alphabet comparison; may be repeated")
    args = parser.parse_args(argv)
    samples: dict[str, int] = {}
    for spec in args.wikidata_sample:
        name, _, size = spec.partition("=")
        if name not in WIKIDATA_SAMPLE_CASES or name in samples or not size.isdecimal() or int(size) < 1:
            parser.error("--wikidata-sample requires a unique alphabet=SIZE or leonardo=SIZE with SIZE >= 1")
        samples[name] = int(size)
    if (args.short_circuit or samples) and args.compare_gate != "wikidata":
        parser.error("--short-circuit and --wikidata-sample require --compare-gate wikidata")
    names = [n for n in args.judges.split(",") if n]
    unknown = sorted(set(names) - set(NAMES))
    if unknown:
        parser.error(f"unknown judges {unknown}; known: {', '.join(NAMES)}")
    if args.verify_wikidata_witnesses:
        if args.stances is None:
            parser.error("--verify-wikidata-witnesses needs --stances from the full DeBERTa run")
        verify_wikidata_witnesses(args.stances)
        return
    if args.compare_gate:
        compare_gate(names, args.stances, args.compare_gate, short_circuit=args.short_circuit,
                     samples=samples)
        return
    state = provenance(INPUTS)
    items = pairs.delta_atomic_items()
    dev = [i for i in items if i.split == "dev"]
    rate = sum(i.strengthens for i in dev) / len(dev)
    previous = json.loads(OUT.read_text()) if OUT.is_file() else {}
    measured = {k: v for k, v in previous.get("judges", {}).items() if k in NAMES}
    # The file keeps the judges earlier runs measured, so each judge carries the provenance of the run measuring it.
    for name in names:
        measured[name] = {**measure(name, items, rate, args.stances, gated=not args.no_gate), "provenance": state}
        taken = measured[name]["forms"][TAKEN]
        print(f"{name}: exception trust {taken['trust']:.4f}, test log loss {taken['test']['at_trust']['log_loss']:.4f}, "
              f"defeat precision {taken['test']['precision']}, recall {taken['test']['recall']:.3f}", flush=True)
    results = {
        "set": CHECK_SET,
        "base_rate": rate,
        "people": {"names": list(PEOPLE), "seed": SEED},
        "rule": "PersonX and PersonY named per item; the taken form reads the update with the premise against the "
                "inference, the rejected named conditional reads it against If premise, inference.; a strengthener's "
                "inference holds, a weakener's does not; each form's trust maximises the dev labels' likelihood under "
                "each stance's projection at the set's base rate after discounting by it; a stance whose disbelief "
                "reaches the judge's threshold defeats; the gate writes claims with letters and statements with names, "
                "at the deployed spread",
        "spread": SPREAD,
        "others": {"rows": OTHERS, "seed": SEED},
        "judges": dict(sorted(measured.items())),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1) + "\n")
    store({name: measured[name]["exception_trust"] for name in names}, rate, CALIBRATION, "exception_trust")
    print(f"wrote {OUT.relative_to(ROOT)} and {CALIBRATION.relative_to(ROOT)} for {', '.join(names)}")


if __name__ == "__main__":
    main()
