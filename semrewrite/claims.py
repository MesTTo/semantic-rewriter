"""Purpose: the claim check, which weighs each rewrite the rewriter admits, and each application of one, against what
the knowledge base states, read by meaning. A rewrite claims the conditional its proposal states, "If A plays games, A
wastes time."; each fact is a statement, its predicate's gloss filled with its arguments' names as words, "kim plays
games." The run's judge reads against a claim the statements that name no individual and stand out in similarity to
it. A statement it reads as contradicting the claim at the threshold it was calibrated to admit at defeats the
rewrite, which is withdrawn before the engine uses it, as a fact added in ASP blocks a default: with "I consider gaming
to be productive." stated, the match from playing games is withdrawn and the one from watching TV stays. A vaguer
reading only weighs on the rewrite's opinion: what the judge reads as following from a statement counts for the claim,
what it reads as contradicted counts against, each discounted by the judge's check trust, and statements it reads as
saying the same are one source.

A statement naming individuals bears on the applications that bind them, not on the rule: with "kim considers gaming
to be productive." stated, the match from playing games stays for every gamer but kim. An application, the rewrite
applied to the row it concludes, reads the statements that name one of the row's individuals and stand out against the
rewrite's claim, as delta-NLI reads an update (Rudinger et al., Findings of EMNLP 2020): the premise and the statement
together, "kim plays games. kim considers gaming to be productive.", against the conclusion, "kim wastes time.". A
clear contradiction defeats that application alone, and a vaguer reading weighs on its exception, at the judge's
exception trust, fitted on delta-ATOMIC's updates read so (Application).

A rewrite is checked rather than an answer because an answer's sentence does not say which match it came through:
nli-deberta-large reads that belief as contradicting "kim wastes time." and "sam wastes time." alike, though only
kim's answer comes through playing games, and as contradicting "If A plays games, A wastes time." but not "If A
watches TV, A wastes time.". The rewriter checks every rewrite it admits and every application the lineage of its
answers makes (semrewrite.rewriter). The embedding model loads only once a claim has a statement it could be read
against: one naming no individual for a rewrite's claim, one naming an individual of the row for an application's.
DESIGN.md, under "The claim check", gives the measurements and the prior art.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from semrewrite.model import LETTERS, Key, Predicate, Proposal, Rewrite
from semrewrite.opinion import Opinion, average

if TYPE_CHECKING:
    from semrewrite.judges.base import SentenceJudge
    from semrewrite.kb import KB
    from semrewrite.trace import Tracer

# How far, in robust standard deviations of its cosines to the knowledge base's statements, a statement's cosine to a
# claim must lie above their median for the claim to be read against it: the modified z-score 0.6745 (x - median)/MAD
# of Iglewicz and Hoaglin (1993), whose rule of thumb for an outlier is 3.5. This value is the smallest spread that kept
# nli-deberta-large's spurious contradictions to one per hundred claims on delta-ATOMIC's dev updates, at every
# knowledge base size from 8 statements to all 3,748 (experiments/claim_gate.py, results/claim_gate.json). The spurious
# defeats, contradictions read at the judge's threshold, which withdraw a rewrite, stay within that budget at it on the
# test updates for nli-deberta-large (0.0059 a claim at most) and typical-qwen3-4b (0.0088).
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
    """A sentence the knowledge base states: the predicates whose facts state it, whether one of those facts names no
    individual, having no arguments, and the individuals the others name, its facts' arguments."""

    predicates: frozenset[str]
    general: bool
    individuals: frozenset[str]


def statements(kb: KB) -> dict[str, Stated]:
    """Each sentence the knowledge base states. A fact is its predicate's gloss filled with its arguments' names as
    words, so play_games(kim) under "{x} plays games" states "kim plays games.", naming kim, and a fact with no
    arguments states its gloss, naming no individual."""
    catalogue = kb.catalogue()
    preds: dict[str, set[str]] = {}
    general: dict[str, bool] = {}
    names: dict[str, set[str]] = {}
    for fact in kb.facts:
        text = sentence(catalogue[fact.pred].sentence(*map(words, fact.args)))
        preds.setdefault(text, set()).add(fact.pred)
        general[text] = general.get(text, False) or not fact.args
        names.setdefault(text, set()).update(fact.args)
    return {text: Stated(frozenset(ps), general[text], frozenset(names[text])) for text, ps in preds.items()}


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


# Time: one judge call over n pairs, 2n for a judge that does not read contradiction, n = len(asked).
def read(judge: SentenceJudge, asked: Sequence[tuple[str, str, str]]) -> list[Opinion]:
    """The judge's own stance on each claim given a statement, as the check reads it: for each (statement, premise,
    conclusion), its opinion that the statement bears out "If premise, conclusion.", the claim's negation asked as
    well where the judge does not read contradiction (stance). The stances are the judge's own; the check discounts
    them by the judge's check trust before they weigh on a rewrite, and experiments/check_trust.py fits that trust on
    stances read so.

    >>> from semrewrite.judges.base import SentenceJudge
    >>> class Asked(SentenceJudge):
    ...     def _score_sentences(self, pairs):
    ...         return [0.25 if claim.startswith("It is not") else 0.75 for _, claim in pairs]
    >>> read(Asked("asked", 0.5), [("kim plays chess.", "A plays games", "A thinks")])
    [Opinion(belief=0.5625, disbelief=0.0625, uncertainty=0.375, base_rate=0.5)]
    """
    return _stances(judge, [(s, conditional(p, c)) for s, p, c in asked], [(s, negated(p, c)) for s, p, c in asked])


def given(premise: str, statement: str) -> str:
    """What an application's statement is read from: the application's premise, which the knowledge base holds, and
    the statement: "kim plays games. kim considers gaming to be productive."."""
    return f"{sentence(premise)} {sentence(statement)}"


def denied(conclusion: str) -> str:
    """The conclusion's negation, which a judge whose disbelief is only that a conclusion does not follow is asked
    about: "It is not the case that kim wastes time."."""
    return f"It is not the case that {_bare(conclusion)}."


# Time: one judge call over n pairs, 2n for a judge that does not read contradiction, n = len(asked).
def read_given(judge: SentenceJudge, asked: Sequence[tuple[str, str, str]]) -> list[Opinion]:
    """The judge's own stance on each conclusion given its premise and a statement, as delta-NLI reads an update
    (Rudinger et al. 2020): for each (statement, premise, conclusion), its opinion that "premise. statement." bears out
    "conclusion.", the conclusion's negation asked as well where the judge does not read contradiction (stance). An
    application's statements are read so, not against its conditional, because its premise is a row the knowledge base
    holds and an exception is the conclusion failing where the premise holds, Reiter's test for blocking one instance
    of a default; nli-deberta-large reads a named conditional as contradicted by statements that only set another
    scene for the same person, "kim reads a book." at d 0.97 against "If kim plays games, kim wastes time.", and
    reads this form at d 0.06.

    >>> from semrewrite.judges.base import SentenceJudge
    >>> class Asked(SentenceJudge):
    ...     def _score_sentences(self, pairs):
    ...         return [0.25 if claim.startswith("It is not") else 0.75 for _, claim in pairs]
    >>> read_given(Asked("asked", 0.5), [("kim plays chess.", "kim plays games", "kim thinks")])
    [Opinion(belief=0.5625, disbelief=0.0625, uncertainty=0.375, base_rate=0.5)]
    """
    return _stances(judge, [(given(p, s), sentence(c)) for s, p, c in asked],
                    [(given(p, s), denied(c)) for s, p, c in asked])


def _stances(judge: SentenceJudge, pairs: Sequence[tuple[str, str]], negations: Sequence[tuple[str, str]]) -> list[Opinion]:
    """The judge's stance on each pair, in one call: its reading alone where it reads contradiction, else combined
    with its reading of the matching negation (stance)."""
    asked = [*pairs] if judge.reads_contradiction else [*pairs, *negations]
    opinions = judge.sentence_opinions(asked) if asked else []
    n = len(pairs)
    return [stance(o, None if judge.reads_contradiction else opinions[n + i]) for i, o in enumerate(opinions[:n])]


@dataclass(frozen=True, slots=True)
class Reading:
    """One statement's weight on a claim: the statement, its cosine to the claim, the judge's own stance on the claim
    given the statement (read), the opinion it weighs on the rewrite with, that stance discounted by the judge's check
    trust, and its source, the statement nearest the claim among those the judge reads as saying the same as it."""

    statement: str
    similarity: float
    stance: Opinion
    opinion: Opinion
    source: str

    @property
    def side(self) -> str:
        """"for" when the reading believes the claim more than it disbelieves it, "against" in the converse case, else
        "neither"."""
        if self.opinion.belief > self.opinion.disbelief:
            return "for"
        return "against" if self.opinion.disbelief > self.opinion.belief else "neither"


@dataclass(frozen=True, slots=True)
class Application:
    """A rewrite, by its key, applied to the row it concludes, whose arguments bind every variable the rewrite has,
    read against the statements naming the row's individuals: its premise and conclusion as the row fills them, the
    readings nearest the rewrite's claim first, `defeats`, the readings whose stance disbelieves the conclusion at or
    past the judge's calibrated threshold, and `normal`, the leaf, the opinion that the rewrite applies normally to the
    row, McCarthy's not ab. A defeated application's leaf is certainly false, (0, 1, 0). Any other's is its readings
    fused as a rewrite's are (weigh), from no evidence, at base rate 1, since circumscription assumes no exception
    where nothing is read: its projection is 1 minus their disbelief, so a neutral reading leaves it certain and
    support only narrows it."""

    key: Key
    row: tuple[str, ...]
    premise: str
    conclusion: str
    readings: tuple[Reading, ...]
    defeats: tuple[Reading, ...]
    normal: Opinion

    @property
    def defeated(self) -> bool:
        """Whether a statement clearly contradicts the conclusion where the premise holds, so the application derives
        nothing."""
        return bool(self.defeats)


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


@runtime_checkable
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


# The judge that reads what the run's judge carries no calibrated trust for: it runs locally, and both of its trusts
# were fitted on delta-ATOMIC (experiments/check_trust.py, experiments/exception_trust.py).
DEFAULT_READER = "nli-deberta-large"
# The two kinds of reading, each named by the trust its reader must carry: a statement naming no individual against a
# rewrite's claim, and a statement about an application's individuals, with its premise, against its conclusion.
CLAIMS, APPLICATIONS = "check_trust", "exception_trust"
_KINDS = {CLAIMS: "check trust", APPLICATIONS: "exception trust"}
# The default reader, loaded once a process on first use and shared by every check, as one embedding model serves
# every run of the Wikidata evaluation; the process's end releases it.
_SHARED: dict[str, SentenceJudge] = {}


def default_reader() -> SentenceJudge:
    """DEFAULT_READER, made on the first call in a process and returned to every later one, so every check reading
    with it, and whatever else in the process ranks with that model, such as the Wikidata evaluation's cascade, holds
    one copy of it."""
    if DEFAULT_READER not in _SHARED:
        from semrewrite.judges import make
        from semrewrite.judges.base import SentenceJudge

        loaded = make(DEFAULT_READER)
        if not isinstance(loaded, SentenceJudge):
            msg = f"{DEFAULT_READER} reads no sentences"
            raise TypeError(msg)
        _SHARED[DEFAULT_READER] = loaded
    return _SHARED[DEFAULT_READER]


class ClaimCheck:
    """Weighs rewrites, and applications of them, against one knowledge base's statements, read by a sentence judge,
    and one embedding model whose cosines choose the statements each claim is read against.

    A claim is read against the statements that name no individual, unless `with_arguments` reads every statement. A
    fact with arguments states something about the individuals it names, and bears on a rule through its rows, which
    the knowledge base's evidence on the rewrite reads (semrewrite.evidence), as an assertion about individuals bears
    on a terminological axiom through its instances (Baader et al., The Description Logic Handbook, 2003); read by
    meaning against a rule written with letters, the judge reads it as noise. On Alphabet's Wikidata question,
    reading every statement asked 39,429 judge pairs and withdrew 31 of 62 rewrites, which left the question no
    answer (RESULTS.md). Every statement still sets the background a statement must stand out from. A rewrite's own
    predicates' statements are left out of both: the knowledge base's evidence on the rewrite reads them as rows, and
    the judge would read a statement of the premise against the rule as its own opinion of the rule again.

    A stance whose disbelief reaches its reader's calibrated threshold, the score at which that judge was calibrated
    to admit a rewrite, is a clear contradiction: it defeats the rewrite (defeats), whose discoverer then withdraws it.
    Every other reading weighs on the rewrite's opinion at `trust`, its reader's check trust unless given, fitted on
    delta-ATOMIC's updates read against their inferences (experiments/check_trust.py).

    A statement that names individuals is read instead for each application binding one of them (applications): the
    rewrite applied to a row, the statements that name one of the row's individuals among those standing out against
    the rewrite's claim, read with the row's premise against its conclusion (read_given). The gate is the claim's own,
    over every statement, so an individual with a handful of statements is never weighed against that handful alone. A
    clear contradiction defeats that application; every other reading weighs on its leaf at `exception_trust`, its
    reader's exception trust unless given, fitted on delta-ATOMIC's updates read in that form with names filled in
    (experiments/exception_trust.py). Each statement is read in one place, so its evidence counts once: with
    `with_arguments` the claims read the statements naming individuals, and no application reads any.

    Each kind of reading has its reader (reading): `reader` where one is given, else `judge`, the run's, where it is a
    sentence judge carrying the trust that reading weighs at, else DEFAULT_READER, loaded once a process on the first
    statement any check hands it, with why recorded in `substituted` and told to `trace`. A given reader without that
    trust is refused once it has something to read. The embedding model is `embedder`, or the default one, loaded
    when a claim first has a statement to be read against. Each claim and each application is read once; `usage`
    counts the claims read, the judge pairs asked, the statements embedded and, once one is read, the applications,
    and `seconds` the time spent."""

    def __init__(self, kb: KB, judge: object, embedder: Embedder | None = None, *, reader: SentenceJudge | None = None,
                 trust: float | None = None, exception_trust: float | None = None, spread: float = SPREAD,
                 with_arguments: bool = False, trace: Tracer | None = None) -> None:
        self.catalogue: Mapping[str, Predicate] = kb.catalogue()
        self.judge = judge
        self.reader = reader
        self._given = {CLAIMS: trust, APPLICATIONS: exception_trust}
        for kind, value in self._given.items():
            if value is not None and not 0.0 <= value <= 1.0:
                msg = f"a {_KINDS[kind]} is a probability, not {value}"
                raise ValueError(msg)
        self.spread = spread
        self.with_arguments = with_arguments
        self.trace = trace
        self._embedder = embedder
        self._readers: dict[str, SentenceJudge] = {}
        self.substituted: dict[str, tuple[str, str]] = {}
        stated = statements(kb)
        self._sentences = sorted(stated)
        self._stated_by = [stated[text].predicates for text in self._sentences]
        self._readable = [with_arguments or stated[text].general for text in self._sentences]
        # The statements naming each individual that no claim reads, so an application finds its own without reading
        # the rest, and a statement the claims read is not read again.
        self._about: dict[str, list[int]] = {}
        for j, text in enumerate(self._sentences):
            for name in () if self._readable[j] else stated[text].individuals:
                self._about.setdefault(name, []).append(j)
        # Each claim's statements standing out against it, with their cosines, nearest first; each application read;
        # and each reader's reading of each ordered pair of statements, whether the first says the second.
        self._kept: dict[Key, dict[int, float]] = {}
        self._read: dict[Key, tuple[Reading, ...]] = {}
        self._applied: dict[tuple[Key, tuple[str, ...]], Application | None] = {}
        self._same: dict[str, dict[tuple[str, str], float]] = {}
        # "applications" joins the counts once an application is read, so a run reading none counts what it did before.
        self.usage: dict[str, int] = {"claims": 0, "pairs": 0, "statements": len(self._sentences)}
        self.seconds = 0.0

    def reading(self, kind: str) -> tuple[SentenceJudge, float | None]:
        """The judge reading `kind`, CLAIMS or APPLICATIONS, and the trust its readings weigh at: the one given, else
        that judge's own, None where it has none, which a reading then refuses."""
        if kind not in self._readers:
            self._readers[kind] = self._choose(kind)
        reader = self._readers[kind]
        given = self._given[kind]
        return reader, given if given is not None else getattr(reader, kind)

    def _choose(self, kind: str) -> SentenceJudge:
        from semrewrite.judges.base import SentenceJudge

        if self.reader is not None:
            return self.reader
        name = getattr(self.judge, "name", type(self.judge).__name__)
        if isinstance(self.judge, SentenceJudge):
            if self._given[kind] is not None or getattr(self.judge, kind) is not None:
                return self.judge
            why = f"{name} has no {_KINDS[kind]}"
        else:
            why = f"{name} reads no sentences"
        try:
            reader = default_reader()
        except ModuleNotFoundError as error:
            raise _needs_models(error, f"reads with {DEFAULT_READER} where {why}") from error
        self.substituted[kind] = (DEFAULT_READER, why)
        if self.trace is not None:
            from semrewrite.trace import ReadBy

            self.trace(ReadBy(kind, DEFAULT_READER, why))
        return reader

    def bar(self, kind: str = CLAIMS) -> float:
        """The disbelief at which a stance of `kind` defeats what it is read against: the threshold its reader was
        calibrated to admit at, the reader's run threshold where it has none. A run admitting at 0.03 to let weak
        matches in does not let a reading as faint as d 0.078 withdraw one, as it did with typical-qwen3-4b while the
        bar followed the run's threshold."""
        reader, _ = self.reading(kind)
        return reader.calibrated_threshold if reader.calibrated_threshold is not None else reader.threshold

    @property
    def reads(self) -> bool:
        """Whether the knowledge base states anything this check reads against a rewrite's claim."""
        return any(self._readable)

    @property
    def individuals(self) -> bool:
        """Whether the knowledge base states anything naming an individual that no claim reads, which an application
        binding that individual may read."""
        return bool(self._about)

    @property
    def embedder(self) -> Embedder:
        """The embedding model choosing statements: the one given, else the default, loaded on first use."""
        if self._embedder is None:
            try:
                from semrewrite.retrieve import EmbeddingRetriever

                self._embedder = EmbeddingRetriever()
            except ModuleNotFoundError as error:
                raise _needs_models(error, "chooses the statements it reads by an embedding model's cosines") from error
        return self._embedder

    def sentences(self, key: Key) -> tuple[str, str]:
        """The rewrite's proposal's premise and conclusion sentences under the run's judge's fillers, the letters."""
        premise, conclusion, mapping = key
        fillers = getattr(self.judge, "fillers", LETTERS)
        return Proposal(self.catalogue[premise], self.catalogue[conclusion], mapping).sentences(fillers)

    def claim(self, key: Key) -> str:
        """The rewrite's claim, the conditional of its proposal's sentences."""
        return conditional(*self.sentences(key))

    def instance(self, key: Key, row: Sequence[str]) -> tuple[str, str]:
        """The premise and conclusion sentences of the rewrite applied to `row`, its arguments' names as words:
        ("kim plays games", "kim wastes time")."""
        premise, conclusion, mapping = key
        names = tuple(map(words, row))
        return Proposal(self.catalogue[premise], self.catalogue[conclusion], mapping).sentences(names)

    def readings(self, rewrites: Iterable[Rewrite]) -> dict[Key, tuple[Reading, ...]]:
        """Each rewrite's readings, nearest the claim first, reading in one batch the claims not read before."""
        keys = sorted({rw.key for rw in rewrites})
        fresh = [key for key in keys if key not in self._read]
        if fresh:
            started = time.perf_counter()
            self._read_claims(fresh)
            self.seconds += time.perf_counter() - started
        return {key: self._read[key] for key in keys}

    def defeats(self, rewrites: Iterable[Rewrite]) -> dict[Key, tuple[Reading, ...]]:
        """Each of `rewrites` that a statement clearly contradicts, with the readings that do: those whose stance has
        disbelief at or above the bar, the score its reader was calibrated to admit a rewrite at, so the judge refutes
        the claim as surely as it must affirm one. Stated beliefs are premises, firmer than any match, so one such
        statement defeats the rewrite however strongly the judge admitted it."""
        read = self.readings(rewrites)
        if not any(read.values()):
            return {}
        bar = self.bar(CLAIMS)
        return {key: found for key, readings in read.items()
                if (found := tuple(r for r in readings if r.stance.disbelief >= bar))}

    def applications(self, applied: Mapping[Key, Iterable[Sequence[str]]]) -> dict[tuple[Key, tuple[str, ...]], Application]:
        """Each application among `applied`, rows by rewrite key, that has a statement to read: the statements that
        name one of its row's individuals, other than its rewrite's own predicates' rows, among those standing out
        against its rewrite's claim, read with its premise against its conclusion (read_given), in one batch for the
        applications not read before. An application with no such statement has no exception, as circumscription has
        none it was not given; where no application has a statement about its individuals at all, nothing is
        embedded and no model loads."""
        wanted = sorted({(key, tuple(row)) for key, rows in applied.items() for row in rows})
        fresh = [a for a in wanted if a not in self._applied]
        if fresh and self.individuals:
            started = time.perf_counter()
            self._read_applications(fresh)
            self.seconds += time.perf_counter() - started
        self._applied.update({a: None for a in fresh if a not in self._applied})
        return {a: found for a in wanted if (found := self._applied[a]) is not None}

    # Time: k claims' cosines to N statements from the embedder, in batches of BATCH claims, and a median and a median
    # deviation per claim, O(k N log N), for the k of `keys` not seen before.
    def _standing(self, keys: Sequence[Key]) -> None:
        """Each claim's statements standing out against it among every statement but its rewrite's own predicates',
        with their cosines, nearest first, found for the claims not seen before."""
        fresh = [key for key in keys if key not in self._kept]
        claims = [self.claim(key) for key in fresh]
        for start in range(0, len(fresh), BATCH):
            rows = self.embedder.cosines(claims[start:start + BATCH], self._sentences)
            for key, row in zip(fresh[start:start + BATCH], rows, strict=True):
                premise, conclusion, _ = key
                positions = [j for j, preds in enumerate(self._stated_by)
                             if premise not in preds and conclusion not in preds]
                chosen = [positions[i] for i in standing_out([row[j] for j in positions], self.spread)]
                self._kept[key] = dict(sorted(((j, row[j]) for j in chosen), key=lambda jc: (-jc[1], jc[0])))

    def _reader_for(self, kind: str, what: str) -> tuple[SentenceJudge, float]:
        """`kind`'s reader and trust once there is something to read, refusing a reader without the trust."""
        reader, trust = self.reading(kind)
        if trust is None:
            script = "experiments/check_trust.py" if kind == CLAIMS else "experiments/exception_trust.py"
            msg = (f"no calibrated {_KINDS[kind]} for {reader.name!r}, which the claim check needs to weigh what it "
                   f"reads {what}; run {script} or pass {kind}=")
            raise LookupError(msg)
        return reader, trust

    # Time: the gate of each claim not seen before (_standing), one judge call over the m kept pairs (2m for a judge
    # that does not read contradiction) and one over the ordered pairs among each claim's kept statements not judged
    # before.
    def _read_claims(self, keys: Sequence[Key]) -> None:
        self.usage["claims"] += len(keys)
        if not self.reads:
            self._read.update({key: () for key in keys})
            return
        self._standing(keys)
        near = [{j: c for j, c in self._kept[key].items() if self._readable[j]} for key in keys]
        asked = [(self._sentences[j], *self.sentences(key)) for key, js in zip(keys, near, strict=True) for j in js]
        if not asked:
            self._read.update({key: () for key in keys})
            return
        reader, trust = self._reader_for(CLAIMS, "against a rewrite's claim")
        self.usage["pairs"] += len(asked) * (1 if reader.reads_contradiction else 2)
        stances = read(reader, asked)
        self._paraphrases([list(js) for js in near], reader)
        at = 0
        for key, js in zip(keys, near, strict=True):
            sources = self._sources([self._sentences[j] for j in js], reader)
            readings = []
            for j, similarity in js.items():
                text, mine = self._sentences[j], stances[at]
                weighed = mine.with_base_rate(reader.base_rate).discount(trust)
                readings.append(Reading(text, float(similarity), mine, weighed, sources[text]))
                at += 1
            self._read[key] = tuple(readings)

    # Time: per application, the statements naming its individuals; the gate of each rewrite one of whose applications
    # has any, not seen before (_standing); a scan of its kept statements per such application; one judge call over
    # the m statements the applications keep (2m for a judge that does not read contradiction), and one over the
    # ordered pairs among each application's statements not judged before.
    def _read_applications(self, fresh: Sequence[tuple[Key, tuple[str, ...]]]) -> None:
        about: dict[tuple[Key, tuple[str, ...]], set[int]] = {}
        for key, row in fresh:
            premise, conclusion, _ = key
            found = {j for name in set(row) for j in self._about.get(name, ())
                     if premise not in self._stated_by[j] and conclusion not in self._stated_by[j]}
            if found:
                about[(key, row)] = found
        self._standing(sorted({key for key, _ in about}))
        todo: list[tuple[Key, tuple[str, ...], dict[int, float]]] = []
        for (key, row), found in about.items():
            near = {j: c for j, c in self._kept[key].items() if j in found}
            if near:
                todo.append((key, row, near))
        if not todo:
            return
        asked = [(self._sentences[j], *self.instance(key, row)) for key, row, near in todo for j in near]
        reader, trust = self._reader_for(APPLICATIONS, "about an application's individuals")
        self.usage["applications"] = self.usage.get("applications", 0) + len(todo)
        self.usage["pairs"] += len(asked) * (1 if reader.reads_contradiction else 2)
        stances = read_given(reader, asked)
        self._paraphrases([list(near) for _, _, near in todo], reader)
        bar = self.bar(APPLICATIONS)
        at = 0
        for key, row, near in todo:
            sources = self._sources([self._sentences[j] for j in near], reader)
            readings = []
            for j, similarity in near.items():
                text, mine = self._sentences[j], stances[at]
                weighed = mine.with_base_rate(reader.base_rate).discount(trust)
                readings.append(Reading(text, float(similarity), mine, weighed, sources[text]))
                at += 1
            defeats = tuple(r for r in readings if r.stance.disbelief >= bar)
            normal = Opinion(0.0, 1.0, 0.0, 1.0) if defeats else weigh(Opinion(0.0, 0.0, 1.0, 1.0), readings)
            self._applied[(key, row)] = Application(key, row, *self.instance(key, row), tuple(readings), defeats, normal)

    def _paraphrases(self, kept: Sequence[Sequence[int]], reader: SentenceJudge) -> None:
        """Have `reader` judge, both ways, each pair of statements one claim keeps that it has not judged, so _sources
        can tell which say the same."""
        same = self._same.setdefault(reader.name, {})
        ordered = sorted({(self._sentences[a], self._sentences[b]) for js in kept for a in js for b in js if a != b
                          and (self._sentences[a], self._sentences[b]) not in same})
        if not ordered:
            return
        self.usage["pairs"] += len(ordered)
        for pair, o in zip(ordered, reader.sentence_opinions(ordered), strict=True):
            same[pair] = o.belief

    def _sources(self, texts: Sequence[str], reader: SentenceJudge) -> dict[str, str]:
        """Each of `texts`, nearest the claim first, mapped to its source: two statements say the same when `reader`
        reads each as following from the other at its admission threshold, and a statement joins the source of the
        first earlier text it says the same as, so a chain of them is one source."""
        same = self._same.get(reader.name, {})
        source: dict[str, str] = {}
        for text in texts:
            source[text] = text
            for earlier in texts:
                if earlier == text:
                    break
                if min(same[(earlier, text)], same[(text, earlier)]) >= reader.threshold:
                    source[text] = source[earlier]
                    break
        return source


def _needs_models(error: ModuleNotFoundError, what: str) -> ModuleNotFoundError:
    """The error saying the claim check needs the models extra for what it `what`s, naming the extra and the command
    that installs it, as the command line names every missing library's extra."""
    msg = (f"the claim check {what}, which needs {error.name}, part of the models extra: "
           "pip install 'semantic-rewriter[models]'")
    return ModuleNotFoundError(msg, name=error.name)


def default_check(kb: KB, judge: object, retriever: object = None, reader: SentenceJudge | None = None,
                  trace: Tracer | None = None) -> ClaimCheck | None:
    """The check a rewriter runs unless given one: `reader` reads the statements where one is given, else `judge`, the
    run's, wherever it carries the trust a reading weighs at, else DEFAULT_READER (ClaimCheck.reading), and the run's
    retriever chooses them when it embeds, else the default embedding model does, loaded only once a claim has a
    statement to be read against. A judge that reads no sentences, as the oracle, which judges by predicate name, is
    given no check unless a reader is: with it there is none where the knowledge base states nothing naming no
    individual, its facts naming individuals staying rows, and a ValueError where it states something.
    """
    from semrewrite.judges.base import SentenceJudge

    embedder = retriever if isinstance(retriever, Embedder) else None
    if isinstance(judge, SentenceJudge) or reader is not None:
        return ClaimCheck(kb, judge, embedder, reader=reader, trace=trace)
    general = sorted(text for text, stated in statements(kb).items() if stated.general)
    if general:
        name = getattr(judge, "name", type(judge).__name__)
        msg = (f"{kb.name} states {len(general)} sentences naming no individual, such as {general[0]!r}, which the "
               f"claim check reads against each rewrite with the run's judge, and {name} reads no sentences")
        raise ValueError(msg)
    return None
