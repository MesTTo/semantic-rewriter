"""Purpose: the claim check, which weighs each rewrite the rewriter admits against what the knowledge base states, read
by meaning. A rewrite claims the conditional its proposal states, "If A plays games, A wastes time."; each fact is a
statement, its predicate's gloss filled with its arguments' names as words, "kim plays games." The run's judge reads
against a claim the statements that name no individual and stand out in similarity to it: what it reads as following
from a statement counts for the claim, what it reads as contradicted counts against, and statements it reads as
saying the same are one source. The sources' opinions fuse with the rewrite's own, so an answer moves through its lineage: with "I
consider gaming to be productive." stated, the match from playing games loses belief, and an answer that also comes
through watching TV keeps that derivation.

A rewrite is checked rather than an answer because an answer's sentence does not say which match it came through:
nli-deberta-large reads that belief as contradicting "kim wastes time." and "sam wastes time." alike, though only
kim's answer comes through playing games, and as contradicting "If A plays games, A wastes time." but not "If A
watches TV, A wastes time.". DESIGN.md, under "The claim check", gives the measurements and the prior art.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from semrewrite.model import Key, Predicate, Proposal, Rewrite
from semrewrite.opinion import Opinion, average
from semrewrite.protocols import calibrated

if TYPE_CHECKING:
    from semrewrite.judges.base import SentenceJudge
    from semrewrite.kb import KB

# How far, in robust standard deviations of its cosines to the knowledge base's statements, a statement's cosine to a
# claim must lie above their median for the claim to be read against it: the modified z-score 0.6745 (x - median)/MAD
# of Iglewicz and Hoaglin (1993), whose rule of thumb for an outlier is 3.5. This value is the smallest spread that kept
# nli-deberta-large's spurious contradictions to one per hundred claims on delta-ATOMIC's dev updates, at every
# knowledge base size from 8 statements to all 3,748 (experiments/claim_gate.py, results/claim_gate.json).
SPREAD = 3.75


# Claims whose cosines one call of the embedder computes: 32 rows over the 23,116 statements of Victoria's question
# are 740,000 floats, about 24 MB as Python lists.
BATCH = 32


def words(constant: str) -> str:
    """A constant's name as the words it spells: queen_victoria reads "queen victoria"."""
    return constant.replace("_", " ")


def _bare(text: str) -> str:
    """`text` without the space and the punctuation that end it."""
    return text.strip().rstrip(".!?;:, ")


def sentence(text: str) -> str:
    """`text` as one sentence ending in a full stop: "I watch TV" reads "I watch TV.", and "I watch TV." is kept."""
    return f"{_bare(text)}."


@dataclass(frozen=True, slots=True)
class Stated:
    """A sentence the knowledge base states: the predicates whose facts state it, and whether one of those facts
    names no individual, having no arguments."""

    predicates: frozenset[str]
    general: bool


def statements(kb: KB) -> dict[str, Stated]:
    """Each sentence the knowledge base states. A fact is its predicate's gloss filled with its arguments' names as
    words, so play_games(kim) under "{x} plays games" states "kim plays games.", and a fact with no arguments states
    its gloss, naming no individual."""
    catalogue = kb.catalogue()
    preds: dict[str, set[str]] = {}
    general: dict[str, bool] = {}
    for fact in kb.facts:
        text = sentence(catalogue[fact.pred].sentence(*map(words, fact.args)))
        preds.setdefault(text, set()).add(fact.pred)
        general[text] = general.get(text, False) or not fact.args
    return {text: Stated(frozenset(ps), general[text]) for text, ps in preds.items()}


def conditional(premise: str, conclusion: str) -> str:
    """The claim that `premise` brings `conclusion`: "If A plays games, A wastes time."."""
    return f"If {_bare(premise)}, {sentence(conclusion)}"


def negated(premise: str, conclusion: str) -> str:
    """The claim's negation, which a judge whose disbelief is only that a conclusion does not follow is asked about:
    "It is not the case that if A plays games, A wastes time."."""
    return f"It is not the case that if {_bare(premise)}, {sentence(conclusion)}"


def stance(reading: Opinion, negation: Opinion | None) -> Opinion:
    """The judge's opinion of a claim given a statement. A judge that reads contradiction gives it in one reading,
    `reading`, whose disbelief is the contradiction. Any other judge's disbelief says only that the claim does not
    follow, so it is also asked whether the claim's negation follows, `negation`; the two answers are read as
    independent: the statement counts for the claim when the judge affirms the claim and denies its negation, against
    it in the converse case, and not at all when the judge affirms or denies both, which no evidence explains.

    >>> stance(Opinion(0.75, 0.0, 0.25), Opinion(0.5, 0.5, 0.0))
    Opinion(belief=0.375, disbelief=0.125, uncertainty=0.5, base_rate=0.5)
    """
    if negation is None:
        return reading
    support, refute = reading.belief, negation.belief
    belief, disbelief = support * (1.0 - refute), refute * (1.0 - support)
    return Opinion(belief, disbelief, 1.0 - belief - disbelief)


@dataclass(frozen=True, slots=True)
class Reading:
    """One statement's weight on a claim: the statement, its cosine to the claim, the judge's opinion of the claim given
    the statement, calibrated as answers weigh the judge's opinions, and its source, the statement nearest the claim
    among those the judge reads as saying the same as it."""

    statement: str
    similarity: float
    opinion: Opinion
    source: str

    @property
    def side(self) -> str:
        """"for" when the reading believes the claim more than it disbelieves it, "against" in the converse case, else
        "neither"."""
        if self.opinion.belief > self.opinion.disbelief:
            return "for"
        return "against" if self.opinion.disbelief > self.opinion.belief else "neither"


# Time: one averaging fusion per source and one cumulative fusion per source, linear in len(readings).
def weigh(opinion: Opinion, readings: Iterable[Reading]) -> Opinion:
    """`opinion` with the readings' evidence: each source's readings averaged, since its statements say the same and
    saw the same evidence, then each source fused with `opinion` by cumulative fusion as an independent source
    (Jøsang 2016, Def. 12.5). The readings take `opinion`'s base rate, being about the same proposition (SLN 3.1). A
    reading the judge reads as neutral, (0, 0, 1), fuses to no change, and one with no belief only lowers P."""
    sources: dict[str, list[Opinion]] = {}
    for r in readings:
        sources.setdefault(r.source, []).append(r.opinion.with_base_rate(opinion.base_rate))
    out = opinion
    for source in sorted(sources):
        held = [o for o in sources[source] if o.belief or o.disbelief]
        if held:
            out = out.fuse(average(held))
    return out


class Embedder(Protocol):
    """What the check needs of an embedding model: each anchor's cosine to each sentence, a row per anchor, as
    semrewrite.retrieve.EmbeddingRetriever gives them."""

    def cosines(self, anchors: Sequence[str], sentences: Sequence[str]) -> list[list[float]]: ...


# Time: two sorts of n cosines, for the median and the median deviation, O(n log n).
def robust_z(similarities: Sequence[float]) -> list[float]:
    """Each cosine's modified z-score, 0.6745 (x - median)/MAD, its distance above the cosines' median in robust
    standard deviations (Iglewicz and Hoaglin 1993). Fewer than three cosines leave no spread to measure, so each
    scores minus infinity; where the median deviation is 0, a cosine above the median scores infinity and the rest
    minus infinity.

    >>> [round(z, 2) for z in robust_z([0.50, 0.52, 0.48, 0.51, 0.49, 0.76])]
    [-0.22, 0.67, -1.12, 0.22, -0.67, 11.47]
    """
    if len(similarities) < 3:
        return [-math.inf] * len(similarities)
    median = statistics.median(similarities)
    deviation = statistics.median(abs(x - median) for x in similarities)
    if deviation == 0.0:
        return [math.inf if x > median else -math.inf for x in similarities]
    return [0.6745 * (x - median) / deviation for x in similarities]


def standing_out(similarities: Sequence[float], spread: float = SPREAD) -> list[int]:
    """The positions of the cosines whose modified z-score reaches `spread`: the statements nearer a claim than the
    knowledge base's statements at large are.

    >>> standing_out([0.50, 0.52, 0.48, 0.51, 0.49, 0.76])
    [5]
    """
    return [i for i, z in enumerate(robust_z(similarities)) if z >= spread]


class ClaimCheck:
    """Weighs rewrites against one knowledge base's statements with one sentence judge, whose opinions are calibrated
    as answers weigh them (protocols.calibrated), and one embedding model, whose cosines choose the statements each
    claim is read against.

    A claim is read against the statements that name no individual, unless `with_arguments` reads every statement. A
    fact with arguments states something about the individuals it names, and bears on a rule through its rows, which
    the knowledge base's evidence on the rewrite reads (semrewrite.evidence), as an assertion about individuals bears
    on a terminological axiom through its instances (Baader et al., The Description Logic Handbook, 2003); read by
    meaning against a rule written with letters, the judge reads it as noise. On Alphabet's Wikidata question,
    reading every statement asked 39,429 judge pairs, moved 54 of 62 rewrites and lowered the lowest P inside
    Wikidata's answer from 0.237 to 0.081, with no gain in average precision (RESULTS.md). Every statement still sets
    the background a statement must stand out from. A rewrite's own predicates' statements are left out of both: the
    knowledge base's evidence on the rewrite reads them as rows, and the judge would read a statement of the premise
    against the rule as its own opinion of the rule again. Each claim is read once; `usage` counts the claims read,
    the judge pairs asked and the statements embedded, and `seconds` the time spent."""

    def __init__(self, kb: KB, judge: SentenceJudge, embedder: Embedder, spread: float = SPREAD,
                 with_arguments: bool = False) -> None:
        self.catalogue: Mapping[str, Predicate] = kb.catalogue()
        self.judge = judge
        self.embedder = embedder
        self.spread = spread
        self.with_arguments = with_arguments
        stated = statements(kb)
        self._sentences = sorted(stated)
        self._stated_by = [stated[text].predicates for text in self._sentences]
        self._readable = [with_arguments or stated[text].general for text in self._sentences]
        self._read: dict[Key, tuple[Reading, ...]] = {}
        self._same: dict[tuple[str, str], float] = {}
        self.usage: dict[str, int] = {"claims": 0, "pairs": 0, "statements": len(self._sentences)}
        self.seconds = 0.0

    def claim(self, key: Key) -> str:
        """The rewrite's claim, its proposal's premise and conclusion sentences under the judge's fillers."""
        premise, conclusion, mapping = key
        return conditional(*Proposal(self.catalogue[premise], self.catalogue[conclusion], mapping).sentences(
            self.judge.fillers))

    def readings(self, rewrites: Iterable[Rewrite]) -> dict[Key, tuple[Reading, ...]]:
        """Each rewrite's readings, nearest the claim first, reading in one batch the claims not read before."""
        keys = sorted({rw.key for rw in rewrites})
        fresh = [key for key in keys if key not in self._read]
        if fresh:
            started = time.perf_counter()
            self._read_claims(fresh)
            self.seconds += time.perf_counter() - started
        return {key: self._read[key] for key in keys}

    # Time: per call, k claims' cosines to N statements from the embedder, in batches of BATCH claims, a median and a
    # median deviation per claim, O(k N log N), one judge call over the m kept pairs (2m for a judge that does not
    # read contradiction) and one over the ordered pairs among each claim's kept statements not judged before.
    def _read_claims(self, keys: Sequence[Key]) -> None:
        self.usage["claims"] += len(keys)
        if not any(self._readable):
            self._read.update({key: () for key in keys})
            return
        claims = [self.claim(key) for key in keys]
        # Each claim's kept statements with their cosines, nearest first; only these outlive their batch's rows.
        near: list[dict[int, float]] = []
        for start in range(0, len(keys), BATCH):
            rows = self.embedder.cosines(claims[start:start + BATCH], self._sentences)
            for (premise, conclusion, _), row in zip(keys[start:start + BATCH], rows, strict=True):
                positions = [j for j, preds in enumerate(self._stated_by)
                             if premise not in preds and conclusion not in preds]
                chosen = (j for j in (positions[i] for i in standing_out([row[j] for j in positions], self.spread))
                          if self._readable[j])
                near.append(dict(sorted(((j, row[j]) for j in chosen), key=lambda jc: (-jc[1], jc[0]))))
        kept = [list(js) for js in near]
        contradiction = self.judge.reads_contradiction
        pairs = [(self._sentences[j], claim) for claim, js in zip(claims, kept, strict=True) for j in js]
        if not contradiction:
            negations = [negated(*Proposal(self.catalogue[p], self.catalogue[c], m).sentences(self.judge.fillers))
                         for p, c, m in keys]
            pairs += [(self._sentences[j], negation) for negation, js in zip(negations, kept, strict=True) for j in js]
        opinions = self._judge(pairs)
        n = sum(len(js) for js in kept)
        stances = [stance(o, None if contradiction else opinions[n + i]) for i, o in enumerate(opinions[:n])]
        weighed = calibrated(self.judge, stances)
        self._paraphrases(kept)
        at = 0
        for key, js in zip(keys, near, strict=True):
            sources = self._sources([self._sentences[j] for j in js])
            readings = []
            for j, similarity in js.items():
                text = self._sentences[j]
                readings.append(Reading(text, float(similarity), weighed[at], sources[text]))
                at += 1
            self._read[key] = tuple(readings)

    def _judge(self, pairs: Sequence[tuple[str, str]]) -> list[Opinion]:
        """The judge's own opinions of `pairs`, counted in `usage`."""
        self.usage["pairs"] += len(pairs)
        return self.judge.sentence_opinions(pairs) if pairs else []

    def _paraphrases(self, kept: Sequence[Sequence[int]]) -> None:
        """Judge, both ways, each pair of statements one claim keeps that has not been judged, so _sources can tell
        which say the same."""
        ordered = sorted({(self._sentences[a], self._sentences[b]) for js in kept for a in js for b in js if a != b
                          and (self._sentences[a], self._sentences[b]) not in self._same})
        for pair, o in zip(ordered, self._judge(ordered), strict=True):
            self._same[pair] = o.belief

    def _sources(self, texts: Sequence[str]) -> dict[str, str]:
        """Each of `texts`, nearest the claim first, mapped to its source: two statements say the same when the judge
        reads each as following from the other at its admission threshold, and a statement joins the source of the
        first earlier text it says the same as, so a chain of them is one source."""
        source: dict[str, str] = {}
        for text in texts:
            source[text] = text
            for earlier in texts:
                if earlier == text:
                    break
                if min(self._same[(earlier, text)], self._same[(text, earlier)]) >= self.judge.threshold:
                    source[text] = source[earlier]
                    break
        return source
