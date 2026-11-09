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

A proposal is a rewrite before judgement. A judge returns a score in [0, 1] for each proposal, read as the probability that the premise sentence entails the conclusion sentence.

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

## Protocols

Each protocol below gives its operations, the contract an implementation must keep, and the reference implementations. The signatures are Python, but nothing in them depends on Python: a Prolog, Rust or SQL implementation keeps the same contract. A member that callers only read is declared as a read-only property, so a class attribute, a frozen dataclass field or a property implements it.

### Retriever

```python
class Retriever(Protocol):
    def top_k(self, goal: Predicate, among: Mapping[str, Predicate], k: int) -> list[tuple[str, float]]: ...
```

It returns at most k candidate premise names from `among`, which never contains the goal itself, most similar first, with their similarity. It must be deterministic for the same inputs and must only return names of the goal's arity.

Reference implementations: `EmbeddingRetriever` embeds each gloss with its placeholders filled by fixed letters and ranks by cosine similarity (BAAI/bge-small-en-v1.5 by default); `AllCandidates` returns every predicate of the same arity, which is right for small catalogues; `JudgeRetriever` scores every predicate of the same arity as a premise of the goal with a judge, under every argument mapping, and keeps the k best. With a local cross-encoder as its judge, `JudgeRetriever` is the first stage of a cascade for a large catalogue: it reads every predicate, and the admitting judge, which may be slower or paid for per question, sees only k of them. Embedding similarity is symmetric and lexical, which a large catalogue punishes. Among Wikidata's 1,790 entity-valued properties, bge-small ranked `author` 327th and `notable work` 794th as premises of "{x} created {y}", and ranked `child organization or unit` 916th for "{x} owns {y}"; nli-deberta-large ranked the first two 26th and 2nd.

### Judge

```python
class Judge(Protocol):
    threshold: float                      # calibration may set it
    @property
    def name(self) -> str: ...
    @property
    def usage(self) -> dict[str, int]: ...
    def score(self, proposals: Sequence[Proposal]) -> list[float]: ...
```

It returns one score in [0, 1] per proposal, in input order, where the score estimates the probability that the premise sentence entails the conclusion sentence. A judge must read direction: it must be able to score `mother ⇒ parent` high and `parent ⇒ mother` low. Embedding similarity cannot do this by construction, since the two sentences of a pair are the same distance apart in both directions. `usage` counts the model work spent, under units the judge names, such as pairs and calls.

`threshold` is the score at or above which the default admission policy admits a proposal. It is calibrated, never guessed (see Calibration).

Reference implementations: three NLI cross-encoders (DeBERTa-v3 base and large, and a DeBERTa-v3-large checkpoint trained on WANLI among other sets), a yes/no reranker (Qwen3-Reranker-0.6B), Jev (TypeSafe's hosted System One model, asked one yes/no question per proposal, with its answers cached on disk), two cosine judges kept as the similarity baseline, and an oracle judge that admits exactly a given set of implications, used by tests and to separate engine behaviour from judge error.

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
- Ask the judge both ways when it is cheap. Direction-aware admission lowered every judge's false-rewrite rate.

## Limits and non-goals

- Function symbols, arithmetic and negation are outside the protocol, because they break the bounded term-size property or monotonicity that the guarantees rest on.
- Discovery never invents predicates and never reads facts from text. Both would unfreeze the catalogue or the constants.
- The rewriter admits rewrites and then reasons exactly. It does not score proofs, as soft-unification systems do; a score per answer would change unification, scoring and the stopping condition, and is a different system.
- Admission decides each rewrite on its own. Choosing the rewrite set jointly under transitivity, as entailment-graph methods do, needs every pair's score before admitting anything, so it cannot run goal by goal.
- The rewrite set grows while a query runs and is kept only in the memo. Maintaining answers incrementally as facts change is left to the engine.
- A depth bound counting steps counts the knowledge base's own rule steps as well as rewrite steps, so besides limiting how far a false rewrite spreads, it cuts recursion through user rules deeper than the bound, such as an ancestor rule over more generations than the bound allows. A bound on rewrites alone (`rewrite-depth-tabled`) keeps that recursion complete. It guarantees termination only with tables: without them a recursive user rule can loop, as it would in Prolog.

## Prior art

- Query rewriting under ontologies compiles subclass and subproperty axioms into a union of queries over the original vocabulary (Calvanese et al., the DL-Lite family, Journal of Automated Reasoning 2007), and Ontop does it over relational databases (Calvanese et al., Semantic Web 2016). Semantic rewriting is the same compilation with the axioms found by a model at query time instead of written by hand.
- HEX programs let an answer-set program call external sources through external atoms during evaluation (Eiter, Ianni, Schindlauer and Tompits, IJCAI 2005). The `expand` callback is an external atom that returns rules instead of tuples.
- NLProlog lets Prolog unify symbols whose embeddings are similar and scores each proof (Weber et al., ACL 2019). The rewriter admits implications instead, so evaluation stays exact and terminates, and it asks a judge that reads direction instead of a symmetric similarity.
- Entailment graphs learn directional implications between predicates from text and choose them jointly under transitivity (Berant, Dagan and Goldberger, ACL 2011; Hosseini et al., TACL 2018). Their typed relation data, with the Levy/Holt directional pairs, is how the experiment measured the judges.
- SLG resolution and its termination on programs with bounded term size (Verbaeten, De Schreye and Sagonas, arXiv cs/0003045; Riguzzi and Swift, arXiv 1110.0631) give the termination theorem.
- Knuth's generalisation of Dijkstra's algorithm finds least-cost derivations in grammars whose costs combine by superior functions, processing facts in cost order (Information Processing Letters 6(1), 1977, doi:10.1016/0020-0190(77)90002-3). A bound on rewrites alone is such a cost, the most rewrites on any path, with rule steps free, so evaluating it level by level finds each fact at its least cost.
