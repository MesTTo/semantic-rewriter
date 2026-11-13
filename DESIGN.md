# Semantic rewriting: design

A logic program answers `parent(X, tom)` only from clauses whose head says `parent`. When the facts say `mother(alice, tom)`, `dad(bob, tom)` and `child_of(tom, erin)` instead, the query finds nothing, although a person reading the facts sees three answers. A semantic rewriter closes that gap. When evaluation first reaches a predicate, it asks a semantic model which other predicates imply it, admits the implications it trusts as ordinary rules, such as `parent(X, Y) :- mother(X, Y)` and `parent(X, Y) :- child_of(Y, X)`, and lets the engine carry on.

This document describes the idea independently of any engine. It states what the rewriter guarantees and under which conditions, the protocols a logic engine, a retriever and a judge must implement to take part, and the conformance tests that decide whether an engine implements them. The reference implementation in this repository runs on two engines: upstream PyMeTTa (MeTTa on SWI-Prolog) and a small pure-Python engine. `RESULTS.md` holds the experiment behind the defaults chosen here.

## The idea in one example

The knowledge base has these facts and glosses, and the query is `parent(X, tom)`:

```
mother(alice, tom).     % {x} is the mother of {y}
dad(bob, tom).          % {x} is the dad of {y}
child_of(tom, erin).    % {x} is a child of {y}
guardian(gus, tom).     % {x} is the legal guardian of {y}
parent                  % {x} is a parent of {y}   (no facts)
```

1. Evaluation reaches the goal `parent(X, tom)` and finds no clause for `parent`.
2. The rewriter retrieves the predicates whose glosses read most like "A is a parent of B": `mother`, `dad`, `child_of` and `guardian`.
3. For each candidate it asks a judge two questions, one per argument order. Does "A is the mother of B" entail "A is a parent of B"? Does "B is the mother of A" entail it? It also asks whether `parent` is symmetric.
4. It admits each implication whose score clears the judge's threshold: `mother ⇒ parent`, `dad ⇒ parent` and `child_of` with its arguments swapped. A good judge rejects `guardian ⇒ parent`, because a legal guardian need not be a parent, and rejects `parent ⇒ mother`, because the implication runs only one way.
5. Each admitted implication becomes a rule for `parent`. Evaluation continues with ordinary resolution and returns `alice`, `bob` and `erin`.

The same step runs again whenever evaluation reaches another predicate for the first time, so a chain such as `mum ⇒ mother ⇒ parent` is found one link at a time, and only the predicates the query actually needs are ever expanded.

## Vocabulary

A predicate has a name, an arity and a gloss, a sentence template with one placeholder per argument. `{x}`, `{y}` and `{z}` name the first three arguments; `{0}`, `{1}` and so on name any argument by position.

A rewrite is an implication from a premise predicate to a conclusion predicate of the same arity, together with an argument mapping. For binary predicates the mapping is either the identity (`mother(x, y) ⇒ parent(x, y)`) or the swap (`child_of(y, x) ⇒ parent(x, y)`). For arity n it is any permutation of the n positions. Read as a rule, the rewrite `parent ⇐ child_of, swap` is `parent(X, Y) :- child_of(Y, X)`.

A proposal is a rewrite before judgement. A judge returns an opinion of each proposal in subjective logic's terms (Jøsang, *Subjective Logic*, 2016): a belief, a disbelief and an uncertainty that sum to 1, and a base rate, the probability assumed without evidence. The opinion's belief is the proposal's score, read as the probability that the premise sentence entails the conclusion sentence, or, for a typicality judge, that the conclusion typically follows from it. A judge that gives only a probability p gives the dogmatic opinion (p, 1 − p, 0).

The catalogue is the set of predicates the knowledge base declares, with their glosses. Discovery only ever proposes rewrites between catalogue predicates, which is what keeps the rewrite set finite.

## What the rewriter guarantees

### Termination

Every query terminates when these four conditions hold:

1. The catalogue is finite and does not change while a query runs.
2. Rewrites only rename a predicate and permute its arguments; user rules have no function symbols, so every argument is a constant or a variable.
3. Discovery is memoised: each predicate is expanded at most once per memo.
4. Evaluation is tabled (SLG resolution or an equivalent fixpoint), or a depth bound limits the number of rule and rewrite steps below the query.

Under conditions 1 to 3 the admitted rewrites form a finite set, and the program has the bounded term-size property: there are finitely many distinct goals and answers. SLG resolution terminates on such programs (Verbaeten, De Schreye and Sagonas, arXiv cs/0003045; Riguzzi and Swift, arXiv 1110.0631, Theorem 3), and a bottom-up fixpoint terminates because it can only add finitely many facts. A depth bound gives a finite search tree directly.

Without tables or a depth bound, a top-down engine loops exactly when the query reaches a cycle of rewrites, such as `mother ⇐ mum ⇐ mother`, because renaming a goal consumes nothing. The experiment's property tests confirm both statements on 400 generated catalogues per property.

Each condition is needed. A rewrite that builds a bigger term breaks the bounded term-size property; discovery that invents predicates never freezes the catalogue; and discovery that extracts facts from open text can bring in new constants without bound.

### Soundness and its limit

Every answer follows from the facts, the user rules and the admitted rewrites. Whether it is true depends on whether the admitted rewrites are true. A false rewrite becomes a rule applied everywhere, so its error spreads to every query that reaches its conclusion and compounds along chains. Admission precision, the share of admitted rewrites that are true, therefore matters more than anything else the judge does.

### Completeness and its limits

The rewriter finds an answer when a chain of admitted rewrites and rules connects the query to facts. It misses answers for three reasons: retrieval did not propose a needed premise (the k nearest glosses missed it), the judge scored a true implication below its threshold, or a depth bound cut the chain.

### Cost

An expansion of a binary predicate with k candidates asks the judge 2k + 1 questions: each candidate in both argument orders, plus whether the predicate is symmetric. For arity n it asks k·n! + n! − 1: each candidate under each of the n! argument mappings, plus the predicate onto itself under each mapping other than the identity. Direction-aware admission doubles that. With a memo, the number of expansions per query is at most the number of catalogue predicates the query reaches, and a memo shared across queries brings repeated predicates to zero further questions. Engine work has no such bound: a judge that admits rewrites between most pairs of predicates makes nearly every predicate reach nearly every other, and the fixpoint, while finite, grows large. In the experiment one permissive judge needed 1,559 times the engine work of a perfect judge on the same query.

## Architecture

```
            query
              │
              ▼
   ┌─────────────────────┐   solve(program, query, budget, expand)   ┌──────────────────┐
   │      Rewriter       │ ─────────────────────────────────────────▶ │      Engine      │
   │ strategy, budgets,  │ ◀───────────────────────────────────────── │ any backend that │
   │ outcome, provenance │      answers, status, work, reached        │ passes the kit   │
   └─────────────────────┘                                            └──────────────────┘
              │  expand(P): rewrites concluding P                              │
              ▼                                                               │
   ┌─────────────────────┐   candidates   ┌───────────┐                      │
   │     Discoverer      │ ─────────────▶ │ Retriever │                      │
   │ memo per predicate  │                └───────────┘                      │
   │                     │   scores       ┌───────────┐   ┌───────────┐      │
   │                     │ ─────────────▶ │   Judge   │ ─▶│ Admission │      │
   └─────────────────────┘                └───────────┘   └───────────┘      │
              ▲                                                               │
              └──────────── the engine calls expand(P) during evaluation ────┘
                            (lazy timing), or the rewriter calls it first
                            and hands the engine every rewrite (eager timing)
```

The rewriter owns the strategy: when discovery runs, whether its results are memoised, whether the engine tables, the depth bound, and whether alias classes are merged. The engine owns evaluation. The discoverer owns the rewrite memo. Retriever, judge and admission policy are replaceable parts with small contracts.

Both the rewriter and the discoverer report what they do to an optional tracer (`semrewrite.trace`): the rewriter as the engine starts and stops a solve, the discoverer once per expansion, with the candidates the retriever kept, every proposal the judge scored and the rewrites admitted, and once per ask its memo answers. With lazy timing the expansions arrive between a solve's start and stop, because the engine calls expand inside evaluation. `semrewrite ask --trace` prints them as they arrive; the engine needs nothing for it.

## Protocols

Each protocol below gives its operations, the contract an implementation must keep, and the reference implementations. The signatures are Python, but nothing in them depends on Python: a Prolog, Rust or SQL implementation keeps the same contract. A member that callers only read is declared as a read-only property, so a class attribute, a frozen dataclass field or a property implements it.

### Retriever

```python
class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def top_k(self, goal: Predicate, among: Mapping[str, Predicate], k: int) -> list[tuple[str, float]]: ...
```

It returns at most k candidate premise names from `among`, which never contains the goal itself, most similar first, with their similarity. It must be deterministic for the same inputs and must only return names of the goal's arity. `name` says what ranks them, as a trace prints it.

Reference implementations: `EmbeddingRetriever` embeds each gloss with its placeholders filled by fixed letters and ranks by cosine similarity (BAAI/bge-small-en-v1.5 by default); `AllCandidates` returns every predicate of the same arity, which is right for small catalogues; `JudgeRetriever` scores every predicate of the same arity as a premise of the goal with a judge, under every argument mapping, and keeps the k best. With a local cross-encoder as its judge, `JudgeRetriever` is the first stage of a cascade for a large catalogue: it reads every predicate, and the admitting judge, which may be slower or paid for per question, sees only k of them. Embedding similarity is symmetric and lexical, which a large catalogue punishes. Among Wikidata's 1,790 entity-valued properties, bge-small ranked `author` 327th and `notable work` 794th as premises of "{x} created {y}", and ranked `child organization or unit` 916th for "{x} owns {y}"; nli-deberta-large ranked the first two 26th and 2nd.

### Judge

```python
class Judge(Protocol):
    threshold: float                      # calibration may set it
    trust: float                          # calibration may set it
    base_rate: float                      # calibration may set it
    @property
    def name(self) -> str: ...
    @property
    def usage(self) -> dict[str, int]: ...
    def opinions(self, proposals: Sequence[Proposal]) -> list[Opinion]: ...
    def close(self) -> None: ...         # releases a model's memory, if it holds one
```

It returns one opinion per proposal, in input order, whose belief, the score, estimates the probability that the premise sentence entails the conclusion sentence. An NLI cross-encoder gives its entailment, contradiction and neutral probabilities as belief, disbelief and uncertainty, since neutral is SNLI's class for a conclusion that might be true given the premise; the other reference judges give a probability. A judge must read direction: it must be able to score `mother ⇒ parent` high and `parent ⇒ mother` low. Embedding similarity cannot do this by construction, since the two sentences of a pair are the same distance apart in both directions. `usage` counts the model work spent, under units the judge names, such as pairs and calls.

A typicality judge is asked a weaker question: whether the conclusion typically follows from the premise, true in the normal case though not in every case, as a bird typically flies and a penguin does not. That is the question a knowledge base of actions needs, where nothing entails that someone who watches TV wastes time, and where an entailment judge answers neutral to almost every such pair. A rewrite it admits holds typically, so the answers through it are what typically follows, defeasible as default rules are, and its opinions are calibrated on labelled typical implications rather than on entailments.

`threshold` is the score at or above which the default admission policy admits a proposal. It is calibrated, never guessed (see Calibration).

`trust` is the probability that the judge's opinions are right. Where answers are weighed, each of the judge's opinions is discounted by it (Jøsang 2016, Eq. 14.6): belief and disbelief are scaled by the trust and the rest becomes uncertainty, so a probability p from a judge trusted at t projects to a + t(p − a), its score drawn toward the base rate a. `base_rate` is that a, the probability an opinion projects to where it says nothing: the rate at which the proposals a judge is asked about hold, not one half. A labelled set's base rate belongs to its proposals, and a judge's defaults come from the labelled set of the kind of question it judges: the judges of what must follow from the Wikidata pairs, base rate 0.214, where an NLI judge's neutral answer projects, and the typicality judge from δ-ATOMIC's, base rate 0.504. Trust is calibrated with the threshold on that set, at its base rate, and `semrewrite/judges/calibration.json` names each judge's set beside its defaults and holds each set's base rate once. `semrewrite.protocols.calibrated(judge, opinions)` gives a judge's opinions at both, as answers weigh them, and a sentence judge's `sentence_opinions(pairs)` judges sentences written directly, such as a request's text against a statement it may support.

Reference implementations: three NLI cross-encoders (DeBERTa-v3 base and large, and a DeBERTa-v3-large checkpoint trained on WANLI among other sets), a yes/no reranker (Qwen3-Reranker-0.6B), Jev (TypeSafe's hosted System One model, asked one yes/no question per proposal, with its answers cached on disk), a typicality judge (Qwen3-4B-Instruct-2507 asked whether the conclusion typically follows, its probabilities of answering Yes, No and Unsure as belief, disbelief and uncertainty), two cosine judges kept as the similarity baseline, and an oracle judge that admits exactly a given set of implications, used by tests and to separate engine behaviour from judge error.

### Admission policy

```python
class Admission(Protocol):
    @property
    def needs_converse(self) -> bool: ...
    def admit(self, proposals: Sequence[Proposal], scores: Sequence[float],
              converse: Sequence[float] | None, threshold: float) -> list[bool]: ...
```

It decides which scored proposals become rewrites, given the judge's threshold. The default policy admits a proposal whose score reaches the threshold. The direction-aware policy sets `needs_converse`, so the discoverer also scores each proposal's converse, and admits a proposal only when it reaches the threshold and `score + slack >= converse score`, with the slack calibrated per judge. A true equivalence such as `mum ⇔ mother` then survives in both directions, and a one-way implication read backwards is dropped. It costs one extra judge question per candidate.

A proposal of a predicate onto itself with swapped arguments, such as `spouse(y, x) ⇒ spouse(x, y)`, states that the predicate is symmetric; admitting it makes the engine try both argument orders.

### Discoverer

```python
class Discoverer:
    def rewrites(self, goal: str) -> tuple[Rewrite, ...]: ...
```

It turns a goal predicate into its admitted rewrites: retrieve k candidates, build every proposal (each candidate under each argument mapping, plus the goal's own symmetry), score them, admit, and sort strongest first. With its memo on, it expands each predicate at most once and returns the same tuple afterwards; this is condition 3 of the termination theorem. Its statistics count expansions, proposals and retrievals, which is how judge cost per query is measured.

The discoverer is engine-independent and lives in the rewriter's process. An engine never sees retriever, judge or admission policy; it sees only the rewrites.

### Answer beliefs

Asked for beliefs (`ask --beliefs`), the rewriter gives every answer of a finished run an opinion that it holds. A rewrite's validity is one unknown shared by every application of it: if "B is owned by A ⇒ A owns B" is wrong, it is wrong for every company. So an answer holds when every rewrite of some derivation is valid, and its lineage is the minimal sets of rewrites its derivations use (PosBool provenance; Green, Karvounarakis and Tannen, PODS 2007). A descent of ten generations through `mother ⇒ parent` needs that one rewrite, not ten copies of it.

`semrewrite.lineage` computes each answer's lineage over the knowledge base and every rewrite admitted while answering, with no depth bound, since a bound limits search and not what the admitted rewrites imply. It evaluates the magic-sets transformation of the program for the query (Beeri and Ramakrishnan, 1991) bottom-up, so it derives only what the query reaches, as the engines' top-down evaluation does, and it checks that every answer the engine returned has a lineage.

Each admitted rewrite is valid with a probability distributed as the Beta of its opinion, independently of the other rewrites. That opinion is its judge's discounted by the judge's trust, fused by cumulative fusion (Jøsang 2016, Def. 12.5) with the knowledge base's evidence on it discounted by the data trust, and with the readings of the statements that bear on its claim, discounted by the judge's check trust (The claim check, below). The knowledge base gives evidence only where it states the rewrite's conclusion: of the premise's statements read as the conclusion's rows, those it also states count for the rewrite and the rest against, NAL's extensional evidence (Wang, JAGI 2009). What other admitted rewrites derive does not count: they cover only part of the relation, so a row they miss is no evidence against a rewrite. The answer's probability is then a polynomial in those probabilities. Its mean is the weighted model count of its lineage, computed by Shannon expansion on one rewrite at a time, and its second moment comes from the same expansion over pairs, both exactly. The answer's opinion is the Beta matched to those two moments (Cerutti, Kaplan, Kimmig and Şensoy, Machine Learning, 2022), with as base rate the probability its lineage has at the rewrites' base rates, as the product and coproduct of opinions take theirs (Jøsang 2016, Eqs. 7.1 and 7.3). An answer the knowledge base derives with no rewrite is certain. Only the matching is an approximation, and where the matched Beta would leave belief or disbelief negative its strength is raised to the least that keeps both at zero or above, as Cerutti et al. do.

A lineage keeps at most k sets per answer, the most probable, 16 by default. Dropping a set from a disjunction only lowers its probability, so a bounded lineage gives a lower bound, and the bound is what keeps lineages over many chained rewrites in memory: over 184 rewrites admitted for Queen Victoria's descendants an unbounded run passed 11 GiB. The lineage counts its work against the run's budget, and a run whose lineage the budget stops ends at `work_limit` with no answers.

### The claim check

The rewriter checks every rewrite it admits against what the knowledge base states, read by meaning (`semrewrite.claims`), so that a stated belief can withdraw a match it contradicts, as adding a fact can withdraw a conclusion in answer-set programming. Its example is a knowledge base of actions asked what wastes time: once it states "I consider gaming to be productive", playing games should no longer come back as wasting time, while watching TV still does. The check is the rewriter's behaviour, not an option. Where the knowledge base states nothing that names no individual, as none of the Wikidata questions' knowledge bases does, it reads nothing, asks the judge nothing and loads no embedding model, so the rewriter answers as it would without it. A judge that reads no sentences, as the oracle, which judges by predicate name, cannot read a statement, so a knowledge base that states one is refused with it.

A rewrite claims the conditional its proposal states, its premise and conclusion sentences under the judge's own fillers: `wastes_time ⇐ play_games` claims "If A plays games, A wastes time.", and with no arguments "If I play video games, I waste time.". Each fact is a statement: its predicate's gloss filled with its arguments' names as words, an underscore read as a space, so `play_games(kim)` states "kim plays games." and a fact with no arguments states its gloss. The check reads rewrites rather than answers because an answer's sentence does not say which match it came through. nli-deberta-large reads the gaming belief as contradicting "kim wastes time." and "sam wastes time." alike (d 1.000 for both, though sam watches TV), and without arguments the one answer "I waste time." comes through both matches. It reads the same belief as contradicting "If A plays games, A wastes time." (d 1.000) and as neutral on "If A watches TV, A wastes time." (u 0.998). Answers then move through their lineage, as answer beliefs compute it: kim's falls with the gaming match, sam's stays, and "I waste time" keeps watching TV's derivation.

**Which statements are read.** nli-deberta-large reads most statements about other people or other activities as contradicting a claim: "raj scrolls social media." against "If A watches TV, A wastes time." gives d 0.999. SNLI's annotators wrote hypotheses about the scene a premise describes (Bowman et al. 2015), so the model reads two unrelated sentences as one scene that cannot hold both. Reading each pair the other way round as well, since contradiction is symmetric (Minervini and Riedel 2018), does not filter these: the model reads the reverse as contradiction too. So a claim is read only against the statements that stand out in similarity to it. The run's embedding model, the one its retriever loads (bge-small by default), gives the claim's cosine to every statement, and a statement is read when its modified z-score among them, 0.6745 (x − median)/MAD (Iglewicz and Hoaglin 1993), is 3.75 or more: when it is nearer the claim than the knowledge base's statements at large are. How many statements that is comes from the knowledge base, not from a frozen k. The value 3.75 is the smallest spread that kept nli-deberta-large's spurious contradictions, statements written for another inference that it reads as contradicting a claim, to one per hundred claims on delta-ATOMIC's dev updates at every size of knowledge base from 8 statements to all 3,748; RESULTS.md gives the measurement. In the actions example the gaming belief stands out from the gaming match's claim at 4.8 without arguments and 5.8 with them, and no other statement reaches 2.5 against any claim. A rewrite's own predicates' statements are never read: the knowledge base's evidence on the rewrite (semrewrite.evidence) reads them as rows, and the judge would read "kim plays games." against "If A plays games, A wastes time." as its own opinion of the rewrite a second time.

Only statements that name no individual, facts with no arguments, are read; every statement still sets the background they must stand out from. A fact with arguments states something about the individuals it names, and bears on a rule through its rows, which the knowledge base's evidence reads, as an assertion about individuals bears on a terminological axiom through its instances (Baader et al., The Description Logic Handbook, 2003). Read by meaning against a rule written with letters, the judge reads it as noise: on the three Wikidata questions, whose statements all name entities, reading every statement asked 39,429 judge pairs on Alphabet's alone and withdrew 31 of its 62 rewrites, which left that question no answer, and 13 of Leonardo's 59 and 5 of Victoria's 117, without raising any question's average precision (RESULTS.md). The gaming belief names nothing, as a statement without holes does, so it is read.

**How a reading counts.** The run's judge reads each chosen statement against the claim (`semrewrite.claims.read`). A stated fact is certain, so the judge's stance is an opinion about the claim: belief counts for it and disbelief against. That holds for a judge whose disbelief is a contradiction, as an NLI model's contradiction class is (`reads_contradiction`). A judge that gives how likely a conclusion follows has disbelief that it does not follow, which says nothing against the claim, so the check also asks it whether the claim's negation follows, "It is not the case that if A plays games, A wastes time."; a statement then counts for the claim when the judge affirms the claim and denies its negation, against it in the converse case, and not at all when it affirms or denies both.

The stance weighs on the rewrite at the judge's base rate, discounted by the judge's check trust. That trust is fitted for this reading, not borrowed from predicate pairs. delta-ATOMIC (Rudinger et al. 2020) labels exactly this reading: each update is a statement, its inference written as the claim its premise brings, "If PersonX is best friends, Before, PersonX needed meet someone." in the source's words, and the claim holds after a strengthener and fails after a weakener. `experiments/check_trust.py` reads each update against its claim as the check does and fits the trust maximising the dev labels' likelihood when each stance projects at the set's base rate, 0.504, after discounting, as each judge's trust is fitted on predicate pairs; `calibration.json` holds it as `check_trust` beside the set's base rate. nli-deberta-large's check trust is 0.698, where its trust on predicate pairs is 0.348, and on the test updates its stances score log loss 0.642 at that trust, 0.656 at the predicate-pair trust and 0.693 at none; typical-qwen3-4b's is 0.738 (0.626, 0.631 and 0.693). RESULTS.md gives every judge's.

**Fusion, and where independence fails.** Each reading fuses with the rewrite's opinion by cumulative fusion (Jøsang 2016, Def. 12.5), after the knowledge base's evidence, each statement an independent source. Independence fails where two statements say the same in different words: "Gaming is productive." and "Playing games is a productive use of time." are one belief, and cumulative fusion would count it twice. So statements the judge reads as following from each other, both ways at its admission threshold, are one source, and their readings are averaged by averaging fusion, the opinion of their mean evidence (Jøsang, Information Fusion, 2010), before the source fuses with the rewrite. Statements that overlap without saying the same, or that the judge misreads for one shared reason, stay correlated and count too much; the limits below say so. Readings fuse into the rewrite, not the answer, so a rewrite's opinion is one unknown shared by all its uses, as the lineage assumes, and `semrewrite rewrites --snars` exports each rewrite's validity at its checked opinion, so every use of the rewrite in SNARS carries the readings too.

**Premises and defeat.** In ASPIC+ terms (Modgil and Prakken, Argument & Computation 2014), the knowledge base's facts are axioms, which no argument attacks, and the admitted rewrites are defeasible rules. A stated belief firmer than any match should defeat the match it clearly contradicts, as a default yields to a fact inconsistent with it in default logic (Reiter 1980) and in answer-set programming. Fusion cannot do that: cumulative fusion adds a clear contradiction from a judge trusted at t as W t/(1 − t) units of evidence against the rule, which lowers its P at any trust below 1 and never withdraws it, so the answers through it stay. So the check decides a clear contradiction as the judge decides an admission: a stance whose disbelief reaches the judge's calibrated threshold, the score at which it was calibrated to admit a rewrite, defeats the rewrite, and the discoverer withdraws it before the engine sees it (`ClaimCheck.defeats`, `Discoverer.rewrites`). A vaguer stance only weighs on the rewrite's opinion. A run's own threshold does not move that bar. While it did, typical-qwen3-4b asked in the first person at 0.03, a threshold low enough to admit weak matches, withdrew three matches with no belief stated, on readings as faint as "I watch TV." against "If I scroll social media, I waste time." at d 0.078; and the rates below, like the gate's budget, are measured at the calibrated bar. With "I consider gaming to be productive" stated, both nli-deberta-large and typical-qwen3-4b withdraw the gaming match, so kim is no answer and "I waste time" keeps only watching TV's derivation, at the watching-TV match's P. On delta-ATOMIC's test updates, nli-deberta-large's stance reaches its threshold, 0.659, for 32% of weakeners, 7% of strengtheners and 27% of other premises' updates; typical-qwen3-4b's reaches its 0.269 for 21%, 5% and 4%. Other premises' updates are what the gate exists for: past it, spurious defeats stay within the gate's budget of one per hundred claims for both judges, at most 0.0059 a claim for nli-deberta-large and 0.0088 for typical-qwen3-4b on the test updates (RESULTS.md).

**Cost.** Each claim is read once per check, when discovery admits its rewrite: one row of cosines over the knowledge base's N statements, a median and a median deviation, O(N log N), a judge pair per statement chosen (two for a judge that does not read contradiction), and the k(k − 1) ordered pairs among a claim's k chosen statements that the sources need, each pair judged once per check. The statements are embedded once, and over a knowledge base whose statements all name individuals the check asks nothing and loads no embedding model.

**A global step.** Per-rewrite fusion is the special case of ConCoRD's factor graph (Mitchell et al., EMNLP 2022) where every stated fact is clamped true, as its section 4.3 clamps a retrieved context sentence, and no two claims constrain each other. A global step decides otherwise where an answer, not a rule, is contradicted. Suppose kim and lee both play games, and the knowledge base states `spends_time_well(kim)`, "kim spends time well.". The fact names kim, so the check leaves it to the rows, and as a row it is no statement of `wastes_time`, so the knowledge base's evidence on the rewrite does not read it either: both answers keep their P. In a factor graph with a variable for the rewrite's validity, a factor tying each answer to its derivations, and the statement clamped against kim's answer, MaxSAT (as BeliefBank and ConCoRD solve theirs) or clingo's weak constraints would trade the contradiction off against the rewrite's weight: either the rewrite gives way, lowering lee's answer too, or, with an abnormality atom per application (`ab(play_games, kim)`, as default logic writes exceptions), kim's application alone. Answers contradicting each other, such as `parent(alice, tom)` and `child_of(alice, tom)` from two rewrites the judge admits, are the same kind of factor. Building that step is its own approach, `a-cc-global-step`: it needs answer sentences read against statements, which the same-scene readings above make unreliable without an entity anchor, and a solver over the lineage's rewrite variables.

### Engine

The engine is the protocol that lets the rewriter run on anything. An engine evaluates a function-free Horn program with a budget, and may call back for rewrites while it evaluates.

```python
@dataclass(frozen=True)
class Program:
    facts: tuple[Atom, ...]        # ground atoms
    rules: tuple[Rule, ...]        # user rules: head :- body atoms, variables shared by name
    rewrites: tuple[Rewrite, ...]  # rewrites known before evaluation (eager timing)
    tabled: bool = True            # resolve cyclic programs to their fixpoint
    depth: int = 0                 # 0: no bound; n: at most n rule or rewrite steps below the query
    depth_counts: str = "steps"    # "steps": every rule and rewrite step counts; "rewrites": only rewrite steps
    proofs: bool = False           # return one derivation per answer

class Engine(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def capabilities(self) -> Capabilities: ...
    def solve(self, program: Program, query: Atom, budget: Budget,
              expand: Callable[[str], Iterable[Rewrite]] | None = None) -> Solution: ...
```

`Capabilities` declares what the engine supports:

| capability | meaning |
|---|---|
| `tabling` | with `tabled=True`, every program terminates with its exact fixpoint answers |
| `depth` | honours `depth > 0` with the semantics below |
| `rewrite_depth` | honours `depth > 0` with `depth_counts = "rewrites"`, counting rewrite steps alone |
| `lazy` | calls `expand` during evaluation instead of needing every rewrite up front |
| `untabled` | can run `tabled=False` as plain top-down resolution (only useful for experiments) |
| `reached` | with no depth bound, reports every predicate evaluation called, and possibly more of the program's; the rounds strategy expands what it reports, so a predicate left out loses the answers its rewrites give, while an extra one only costs an expansion. Under a depth bound it may report fewer |
| `proofs` | with `proofs=True`, returns a derivation for each answer |

`Budget` gives the most a run may spend: work in the engine's own unit (inferences for a Prolog engine, candidate rows examined for the Python engine) and wall-clock seconds, which may be infinite. `Solution` carries the answers (one tuple of constants per answer, the query's arguments with its variables filled), a status (`ok`, `work_limit`, `time_limit` or `error`), the work spent, the predicates reached when the engine reports them, and proofs when the program asked for them. Proofs are asked for because they can cost an engine work: on PyMeTTa, where tables keep each answer's shortest proof, they cost 35 to 43 percent more inferences over the four benchmarks' queries (lazy-tabled 1,278,407 against 1,719,877, eager-worklist 1,031,206 against 1,471,025, with the oracle judge).

An engine must keep this contract:

1. Answers are exact. With `tabling` and `tabled=True`, the answers are those of the least model of facts, rules and rewrites, restricted to the query. The rewrites are the program's rewrites plus whatever `expand` returned.
2. Depth has one meaning. With `depth = n`, an answer is returned exactly when it has a derivation whose counted steps, along every path from the query down to a fact, number at most n. Every rule and rewrite step counts, or with `depth_counts = "rewrites"` only rewrite steps do, so the knowledge base's own rules recurse as deep as their facts allow. A fact needs zero steps. Tables then only change the cost, never the answers.
3. Expansion is a rule source. When `expand` is given and the engine is `lazy`, it calls `expand(P)` before it resolves a goal on P that it has not expanded in this call to `solve`, and treats each returned rewrite as a rule for P. It may call `expand` again for the same predicate; the discoverer's memo makes that free.
4. Budgets are hard. The engine stops when it exceeds the work or the seconds of its budget and reports which limit it hit, and a status of `ok` means the run stayed within both. A budget-stopped run returns no answers; partial answers would read as complete ones.
5. Names are data. Any constant or predicate name matching `[a-z][A-Za-z0-9_]*` must work, including names the engine's own language reserves. An engine renames them internally if it must. (PyMeTTa needed this: a person called `max` collided with the builtin `max` and lost answers until the backend renamed it.)

Two patterns implement the lazy capability, and either is enough:

- A meta-interpreter. Predicates become data, `holds(P, Args)`, and one clause asks the discovery callback for rewrites of P and follows each: `holds(P, Args) :- expand(P, Q, Map), permute(Map, Args, QArgs), holds(Q, QArgs)`. Tabling `holds` gives SLG resolution with semantic goal expansion. This is how the PyMeTTa backend works, through a Python op that MeTTa calls. Any Prolog with foreign predicates (SWI through janus, Scryer, Trealla), any MeTTa implementation with grounded operations, and any engine with external atoms (HEX programs) can do the same. Measure what one call of the callback costs on the engine at hand: in PyMeTTa, how the op's parameter is annotated changed its cost per call by more than half and decided whether the first run of a process stopped at its budget (RESULTS.md, under pymetta). Measure what the engine's data costs too. The PyMeTTa meta-interpreter holds its facts as atoms in the space, which `holds` reads with `match`, because pymetta 0.9.4 takes time growing faster than quadratically to drop one function's equations when a context closes: 10,000 ground equations took 106 s against 0.06 s for the same facts as atoms (`repros/equation_teardown_superquadratic.py`). With facts as equations of one function, a solve over 10,000 facts spent 75 of its 77 s closing the context; with atoms, a solve over the 23,116 statements of the Victoria example in examples/README.md takes about 3 s.
- Demand-driven bottom-up evaluation. The engine computes which predicates the query depends on, calls `expand` for each new one, adds the returned rewrites as rules and extends the dependency set, then runs a semi-naive fixpoint over the relevant predicates. Round k of the fixpoint derives exactly the facts of derivation height k, so a depth bound is a round limit. A bound on rewrites alone runs the fixpoint in levels: each level first closes under the user rules, then one round of rewrites starts the next level, so level k holds the facts that need at most k rewrites on any path, a generalisation of Dijkstra's algorithm to such costs (Knuth, 1977). This is how the pure-Python backend works, and it carries over to Datalog engines and to SQL with recursive common table expressions.

An engine without the lazy capability still runs every strategy except the lazy ones: the rewriter computes the rewrites first and passes them in `program.rewrites`.

A top-down engine gives proofs by making every clause answer a proof term instead of true: a fact answers a leaf, a rewrite step names the rewrite and wraps its premise's proof, and a rule step names the rule and wraps its premises' proofs with the values of the variables only its body has. Tables then need one change. A rewrite cycle such as `mother ⇔ mum` gives each answer infinitely many proof terms, one per trip round the cycle, so a table of terms never completes. Answer subsumption fixes this: SWI-Prolog's moded tabling (`lattice`) joins every new answer for the same arguments with the old one, so each table keeps one proof per answer. Which proof it keeps is the join's choice. Keeping the first one found terminates, but a table's first proof can embed another call's first proof of the same atom, so a derivation could go round a cycle and back (`lowers ⇐ is_reduced_by ⇐ lowers`). PyMeTTa's proof terms therefore carry their number of steps, and the join keeps the proof with fewer. Step counts are natural numbers, so the join stops improving and the fixpoint is each answer's shortest derivation, which never repeats an atom on a path: Knuth's argument for superior functions, since a step's size is one more than its premises' sizes together (Knuth, 1977). PyMeTTa declares its proof tables through `(cache name (plain private (lattice keep)))` rows; a plain table of proof terms ran out of table space on a two-predicate cycle. A bottom-up engine records, for each fact, the step that first derived it; round order then makes every recorded proof well founded, and of the least height.

### Strategies

A strategy is a setting of five knobs, chosen per query:

| knob | settings | what it decides |
|---|---|---|
| discovery timing | none, lazy, eager worklist, rounds, saturate | when rewrites are found relative to evaluation |
| memo | on, off | whether a predicate is expanded once or on every call |
| tables | on, off | termination on cyclic rewrites, and whether answers are proved once |
| depth bound | none, n steps, n rewrites | how many rule and rewrite steps, or rewrite steps alone, a derivation may use |
| alias classes | merged, kept | whether mutually implying predicates are renamed to one before evaluation |

The named strategies are settings of these knobs. `lazy-tabled` (lazy, memo, tables) is semantic goal expansion inside SLG resolution. `depth-tabled` adds a depth bound of 3 and keys each table on the depth left. `rewrite-depth-tabled` bounds rewrites alone at 3, so user rules pass the depth left on unchanged. `eager-worklist` expands everything reachable from the query before evaluation and hands the engine native rules. `alias-canonical` does the same and then merges alias classes with a union-find that tracks argument swaps. `rounds` evaluates, expands the predicates that evaluation reached, and evaluates again until nothing new is reached. `saturate` expands the whole catalogue first. `naive` and `memo-untabled` drop tables and exist to show the loop.

The rewriter refuses a strategy the engine's capabilities cannot run, rather than running something else under its name.

### Calibration

A judge's threshold and a direction-aware policy's slack are frozen guesses unless they are fitted to data, so the reference implementation fits them:

- On labelled predicate pairs, a threshold maximising balanced accuracy, and the slack that maximises balanced accuracy of the direction-aware policy.
- On the same labelled pairs, the base rate, the share of the dev pairs that hold, and each judge's trust: the trust maximising the labels' likelihood when each pair holds with the projection of the judge's opinion at the base rate discounted by it. The projection is linear in the trust, so the log-likelihood is concave and its maximum is found by bisecting its derivative. The defaults the factory reads, threshold and trust, come from the set of the kind of question the judge is asked: the Wikidata pairs for a judge of what must follow, δ-ATOMIC's for the typicality judge. Every other set with dev pairs gets its own base rate and trust beside them, and its test pairs are scored at those too, since a set asking another question calibrates a judge differently.
- On δ-ATOMIC's dev updates read against their inferences as the claim check reads a statement against a rewrite's claim, each judge's check trust, fitted as its trust is, at that set's base rate (The claim check).
- On the labelled Wikidata pairs whose two properties both have statements in the examples' knowledge bases, each judge's data trust: the trust in the knowledge base's evidence that maximises the labels' likelihood once that evidence, discounted by it, is fused with the judge's trusted opinion. Fusion weighs each opinion by the other's uncertainty, so this likelihood need not be concave; a grid finds its best cell and golden-section search refines it.
- On knowledge bases with gold answers, the threshold maximising answer F1 of a given strategy, chosen leave-one-knowledge-base-out: for each knowledge base, the threshold that is best on the others is scored on it.

The second fit is the one the rewriter needs. Balanced accuracy favours recall, while the rewriter needs precision because it applies every admitted rewrite everywhere. In the experiment, Jev's pair-calibrated threshold was 0.165, and its answer-calibrated threshold for lazy-tabled evaluation was 0.65 on every held-out fold.

## Conformance

An engine implements the protocol when it passes the conformance kit, `semrewrite.conformance`, which runs property tests against any object with a `solve` method:

| property | needs capability |
|---|---|
| with tables and no depth bound, answers equal a reference fixpoint on generated programs with rewrite cycles, arities 0 to 3, and up to three user rules of one to three body atoms with constants and repeated variables | `tabling` |
| with a depth bound, answers equal the reference fixpoint restricted to derivations of at most that many steps, with tables on or off | `depth` |
| with a bound on rewrites, answers equal the reference fixpoint restricted to derivations of at most that many rewrites on any path, and recursion through user rules answers completely, with tables on or off | `rewrite_depth` |
| lazy expansion gives the same answers as passing every rewrite up front | `lazy` |
| a budget too small for the program ends the run with a limit status and no answers, and an `ok` run stayed within its budget | always |
| constants and predicates spelled like the engine's reserved words keep their answers, with every rewrite up front and through the callback | `tabling`, `depth` or `rewrite_depth` |
| each answer has one proof whose steps are program rules or rewrites, whose leaves are program facts, and whose height in counted steps respects the depth bound | `proofs` |
| without tables, a query on a rewrite cycle never finishes, and one that reaches no cycle finishes with the fixpoint | `untabled` |
| evaluating, adding the rewrites of every predicate a run reached and evaluating again, until a run reaches nothing new, gives the reference answers | `reached` and `tabling` |

A run with tables, or with a depth bound of 1 step, must finish with exactly the reference answers; it gets the default work budget and no time limit, so a loaded machine cannot fail an engine, while the work limit still stops one that does not terminate. An untabled run deeper than that may instead stop at its budget with no answers, because top-down search without tables enumerates every derivation, and through a join their number grows faster than any budget the kit could promise; so may an untabled run bounding rewrites alone, whose user rules may recurse forever. A run that reports a limit must actually have reached it. The reference evaluator is a naive bottom-up fixpoint with no dependencies, so the kit checks an engine against something other than itself. Run it with `semrewrite conformance --engine NAME` or from pytest.

## Choosing settings

These defaults come from the experiment in `RESULTS.md`: four generated knowledge bases with 75 predicates and 58 queries, seven judges, and leave-one-out threshold calibration.

- Use tables and a memo. Every tabled strategy terminated on every query, and on every generated catalogue it returned exactly the fixpoint's answers; without tables, 56 of 58 queries ran until the budget stopped them.
- Use a judge that reads direction. Jev scored 0.82 to 0.89 direction accuracy on two external pair sets and the NLI cross-encoders 0.77 to 0.87, while cosine similarity sat at chance.
- Calibrate the threshold for answer F1 on held-out knowledge bases, not on pair sets.
- With an imperfect judge, add a depth bound, and calibrate what it counts beside the threshold. A short chain limits how far each false rewrite spreads. At held-out thresholds, tabled evaluation bounded at 3 steps beat unbounded tabled evaluation for six of the seven judges, nli-deberta-large's answer F1 rising from 0.68 to 0.91: every query finished, and calibration could choose lower thresholds, admitting more true rewrites, without the false ones taking over. The exception was Jev, which admits few false rewrites: bounded at 3 rewrites alone, so the knowledge base's own rules recursed freely, it scored 0.92, its best, against 0.83 with the step bound and 0.89 with none. With every other judge the rewrite bound scored below the step bound, because a false rewrite's answers then travel as far as the rules recurse. The step bound also cuts recursion through user rules deeper than three steps, as the limits below say
- On a catalogue that holds every relation the data uses, bound rewrites at one a path. Each relation a question needs is then itself a candidate for the question's own predicate, so a longer chain adds no answer a direct rewrite misses, and it lets one misreading feed the next. Over Wikidata's 1,790 properties (examples/README.md), Jev found all 1,164 of Queen Victoria's descendants that Wikidata records, and nothing else, with one rewrite a path; with chains of three it added 27 armed forces, military units and sports clubs, and asked 5,454 questions against 202.
- Ask the judge both ways when it is cheap. Direction-aware admission lowered every judge's false-rewrite rate.
- With a judge whose trust is high, admit at a low threshold and rank the answers by their beliefs instead of cutting at a high one. With Jev at 0.1 rather than 0.65, every answer outside Wikidata's record ranked below every answer in it on Victoria's and Alphabet's questions, and Alphabet's F1 rose from 0.938 to 0.947 (RESULTS.md).

## Limits and non-goals

- Function symbols, arithmetic and negation are outside the protocol, because they break the bounded term-size property or monotonicity that the guarantees rest on.
- Discovery never invents predicates and never reads facts from text. Both would unfreeze the catalogue or the constants.
- The rewriter admits rewrites and then reasons exactly. Answer beliefs come after evaluation, from each answer's lineage, and never steer unification, search or the stopping condition, as soft-unification systems' proof scores do.
- Answer beliefs assume the admitted rewrites are valid independently of one another and of the knowledge base, which is taken as true. Two rewrites a judge misreads for one shared reason, such as two glosses that both mention ownership, are correlated: the belief of an answer needing both is then too low, and of an answer reachable through either too high.
- Trust only shrinks a judge's opinions toward the base rate; it cannot recalibrate them. A judge that ranks pairs well but states miscalibrated probabilities gets little trust: reranker-qwen3 ranks Wikidata test pairs at ROC AUC 0.67, and its trust fits to 0.08.
- A question asked in its own words states none of its predicates, so the knowledge base gives no evidence on its rewrites and the judge's opinions stand alone. What the other admitted rewrites derive could corroborate a rewrite, but reading it needs each rewrite's coverage of the relation: counted as if the others covered all of it, a valid rewrite deriving much that they miss reads as wrong, as `creator` did for Leonardo, 818 of the 858 works it states counted against it.
- Beliefs need a lineage over named predicates, so a strategy that answers over merged alias classes refuses them, as it refuses proofs.
- A typicality judge's rewrites hold typically, and typical implications do not chain as entailments do: studying for an exam typically means reading a book, which may typically waste time, yet studying does not. Discovery chains admitted rewrites as rules, so an answer through such a chain gets a lower belief but not zero; default logic would block it by the more specific premise, which the rewriter does not read.
- Admission decides each rewrite on its own. Choosing the rewrite set jointly under transitivity, as entailment-graph methods do, needs every pair's score before admitting anything, so it cannot run goal by goal.
- The claim check reads a statement only where it stands out in similarity to a claim among the knowledge base's statements. A statement bearing on a claim through background knowledge rarely does when many statements read alike: among delta-ATOMIC's updates, which all describe a person called PersonX, the gate reads 2.1% of an inference's own updates in a knowledge base of 8 statements and 0.15% among all 4,037 of the test split. With fewer than three statements to compare, nothing stands out and the check reads none.
- The check reads only statements with no arguments, so a general belief written about a constant, such as `productive(gaming)`, is left to the knowledge base's evidence like any fact; written with no arguments, "I consider gaming to be productive", it is read.
- The check trust is fitted on delta-ATOMIC, whose statements are updates about a person called PersonX and whose claims are typical inferences, not a knowledge base's beliefs read against rewrites between predicates; no labelled set of the latter exists. It shrinks a judge's readings toward the base rate and cannot fix what the judge misreads: nli-deberta-large reads delta-ATOMIC's updates as their writers meant them 18% of the time, 33% of the weakeners as contradicting their inference and 3% of the strengtheners as entailing it.
- The check defeats at the judge's calibrated threshold, whatever threshold a run admits at, so a judge calibrated low defeats on faint readings: nli-deberta-large-wanli's 0.054 lets its spurious defeats past the gate reach 0.0107 a claim on the full test set, just over the budget. A clear contradiction defeats however strongly the judge admitted the rewrite, and a statement that defeats wrongly costs every answer through the rewrite, which is why the gate's false-alarm budget is counted in defeats too.
- A statement about one answer, such as `spends_time_well(kim)`, bears on that answer's use of a rewrite, which the check does not read; it needs the global step under The claim check.
- The rewrite set grows while a query runs and is kept only in the memo. Maintaining answers incrementally as facts change is left to the engine.
- A depth bound counting steps counts the knowledge base's own rule steps as well as rewrite steps, so besides limiting how far a false rewrite spreads, it cuts recursion through user rules deeper than the bound, such as an ancestor rule over more generations than the bound allows. A bound on rewrites alone (`rewrite-depth-tabled`) keeps that recursion complete. It guarantees termination only with tables: without them a recursive user rule can loop, as it would in Prolog.

## Prior art

- Query rewriting under ontologies compiles subclass and subproperty axioms into a union of queries over the original vocabulary (Calvanese et al., the DL-Lite family, Journal of Automated Reasoning 2007), and Ontop does it over relational databases (Calvanese et al., Semantic Web 2016). Semantic rewriting is the same compilation with the axioms found by a model at query time instead of written by hand.
- HEX programs let an answer-set program call external sources through external atoms during evaluation (Eiter, Ianni, Schindlauer and Tompits, IJCAI 2005). The `expand` callback is an external atom that returns rules instead of tuples.
- NLProlog lets Prolog unify symbols whose embeddings are similar and scores each proof (Weber et al., ACL 2019). The rewriter admits implications instead, so evaluation stays exact and terminates, and it asks a judge that reads direction instead of a symmetric similarity.
- Entailment graphs learn directional implications between predicates from text and choose them jointly under transitivity (Berant, Dagan and Goldberger, ACL 2011; Hosseini et al., TACL 2018). Their typed relation data, with the Levy/Holt directional pairs, is how the experiment measured the judges.
- Default logic draws conclusions that hold typically and yield to more specific knowledge: birds fly, but a penguin is a bird that does not (Reiter, Artificial Intelligence 13, 1980). The typicality judge is asked such a default's question about two sentences. δ-NLI labels human-written updates that strengthen or defeat a typical inference about an everyday event (Rudinger et al., Findings of EMNLP 2020), and COPA asks which of two alternatives is the more plausible cause or effect (Roemmele, Bejan and Gordon, AAAI Spring Symposium 2011), balanced so that each alternative is right under one premise and wrong under another (Kavumba et al., COIN 2019). Their δ-ATOMIC and Balanced COPA pairs are how the experiment measured what typically follows.
- SLG resolution and its termination on programs with bounded term size (Verbaeten, De Schreye and Sagonas, arXiv cs/0003045; Riguzzi and Swift, arXiv 1110.0631) give the termination theorem.
- Knuth's generalisation of Dijkstra's algorithm finds least-cost derivations in grammars whose costs combine by superior functions, processing facts in cost order (Information Processing Letters 6(1), 1977, doi:10.1016/0020-0190(77)90002-3). A bound on rewrites alone is such a cost, the most rewrites on any path, with rule steps free, so evaluating it level by level finds each fact at its least cost.
- FEVER (Thorne, Vlachos, Christodoulopoulos and Mittal, NAACL 2018, arXiv 1803.05355) verifies a claim against sentences retrieved for it: retrieval first, then entailment over the evidence retrieved. The claim check does the same for a rewrite's claim over the knowledge base's statements, its gate retrieving and the run's judge reading.
- BeliefBank (Kassner, Tafjord, Schütze and Clark, EMNLP 2021, arXiv 2109.14723) keeps a model's answers as beliefs with constraints between them, revises them by weighted MaxSAT, and feeds relevant beliefs back as context. ConCoRD (Mitchell et al., EMNLP 2022, arXiv 2211.11875) has an NLI model estimate those constraints, between answers and between an answer and a context statement, and solves the factor graph by MaxSAT; its section 4.3 clamps a retrieved context sentence true to update answers at test time, the case of a belief stated after the fact, which the claim check answers rule by rule. ConCoRD drops relations the NLI model gives low confidence; the check's gate drops statements by relevance instead, since an NLI model's spurious contradictions are confident.
- Defeasible inference (Rudinger et al., Findings of EMNLP 2020) pairs an inference that typically holds with updates written to weaken or strengthen it. Its delta-ATOMIC split is the held-out set the check's gate was chosen on and each judge's check trust fitted on.
- ASPIC+ (Modgil and Prakken, Argument & Computation 5(1), 2014) builds arguments from axioms, which cannot be attacked, and ordinary premises, by strict and defeasible rules, and attacks them by undermining, rebutting and undercutting; the claim check treats facts as axioms and rewrites as defeasible rules, and lets a stated fact the judge reads as clearly contradicting a rule defeat it.
- Contradiction is symmetric (Minervini and Riedel, CoNLL 2018, arXiv 1808.08609), so a model's reading of a pair in one direction alone is not one it holds; the check tried requiring both directions as a relevance filter, and nli-deberta-large reads unrelated pairs as contradiction both ways.
