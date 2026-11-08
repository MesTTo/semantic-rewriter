# Results on the first gold, before the audit

This is `RESULTS.md` as it stood on 8 November 2026 at 07:58. Every table in it was rendered from runs on the benchmarks' first gold, which missed implications the glosses force; the commit "Audit the benchmark gold and define a true rewrite as entailment by canonical database" replaced that gold (see "How answers and rewrites are scored" in `RESULTS.md`). `RESULTS.md` supersedes all of it; this copy is kept only as the evidence for the before-audit figures that `RESULTS.md` cites.

This file answers the four questions behind the experiment. The tables between `GENERATED` markers come from the files under `results/`, rendered by `python experiments/report.py`; the commands that regenerate those files are at the end.

## Short answers

**Does semantic reduction loop?** No, as long as goals are tabled and each predicate is expanded once. Every tabled strategy terminated with exactly the right answers on all 400 generated catalogues, each containing an alias cycle. Untabled rewriting loops exactly when the goal sits on a cycle of rewrites. Tables stop loops, not blow-ups, though: with a permissive judge, the finite closure cost up to 1,560 times the engine work.

**Which implementation?** Ten were built, as settings of four knobs. The one to use is goal-directed discovery with tables and a per-predicate memo. Discovering inside SLG evaluation (`lazy-tabled`) is the natural fit; compiling rewrites to native equations (`eager-worklist`) and merging aliases with a union-find (`alias-canonical`) are cheaper in engine work. All of them cost about 107 judge questions per query on their own, or 16 once a benchmark's queries share the memo.

**Can a cross-encoder or Jev tell which way?** Yes. On the Levy/Holt and Wikidata pairs, Jev gets 0.82 to 0.90 direction accuracy, the NLI cross-encoders 0.77 to 0.87, and embedding similarity sits at chance. Inside the rewriter, the deciding quantity is admission precision, because every false rewrite becomes a rule applied everywhere. Jev at a threshold calibrated on the other benchmarks reaches answer F1 0.839. The best cross-encoder reaches 0.63 on the same basis, and similarity alone at most 0.37. Asking each judge both ways and rejecting the weaker direction lowered every cross-encoder's false-rewrite rate.

**Is pymetta stable enough?** Yes. Tabling, ops called inside tables, table refresh and budgets all did what the rewriter needed. I found four defects, each with a script under `repros/`, and none blocked the work.

## What was built

A query such as `(parent $x tom)` finds nothing in a knowledge base that only states `(mother alice tom)` and `(father bob tom)`, because MeTTa rewrites a goal only with an equation whose head matches it symbol for symbol. The semantic rewriter keeps pymetta's evaluator, unification and tabling unchanged and adds one step where a goal is first reached:

1. Retrieve the k catalogue predicates whose glosses are nearest the goal's (k = 8, BAAI/bge-small-en-v1.5).
2. Ask a judge, for each candidate Q, whether `Q(A, B)` entails `goal(A, B)` and whether `Q(B, A)` does, and whether the goal is symmetric.
3. Admit each proposal that reaches the judge's threshold as an ordinary rewrite: `(= (parent $X $Y) (mother $X $Y))`, or `(= (parent $X $Y) (child_of $Y $X))` when the arguments swap.

Each rewrite only renames a predicate or swaps its two arguments. Rewriting a parent goal into a mother goal is sound because every mother answer is a parent answer; the reverse rewrite would be unsound, so the judge's direction matters.

The strategies are settings of four independent knobs:

| strategy | discovery timing | memo | evaluation | aliases |
|---|---|---|---|---|
| exact | none | | tabled | |
| naive | lazy, inside evaluation | off | untabled (SLD) | |
| memo-untabled | lazy | on | untabled | |
| nomemo-tabled | lazy | off | tabled (SLG) | |
| depth-bounded | lazy | on | SLD to depth 3 | |
| lazy-tabled | lazy | on | tabled | |
| rounds | evaluate, expand what was reached, repeat to a fixpoint | on | tabled | |
| eager-worklist | worklist from the query before evaluation | on | tabled | |
| alias-canonical | eager | on | tabled | union-find renames each alias class to one predicate |
| saturate | every catalogue predicate before evaluation | on | tabled | |

`lazy` runs discovery through a pymetta op (`impliers`) that the tabled `holds/3` clause calls when evaluation first reaches a predicate. That is SLG resolution with semantic goal expansion. `rounds` is the loop in your original question: reduce, discover for whatever was reached, add the rewrites, reduce again, stop when a round reaches nothing new.

## When does it stop, and can it loop?

It stops when no new (goal, rewrite) pair and no new answer can appear. With tables it always gets there. Without tables it loops exactly when the goal sits on a cycle of rewrites, and the property tests confirm both statements.

`tests/test_termination.py` generated 400 random catalogues per property (2 to 6 predicates, random gold implications always including an alias cycle, random facts, an optional join rule, a random query) and compared every answer with a Python fixpoint (`results/termination.txt`):

- Every tabled strategy (lazy-tabled, nomemo-tabled, rounds, eager-worklist, alias-canonical, saturate) terminated with exactly the fixpoint's answers on all 400.
- Untabled rewriting never terminated on the 261 of the 400 catalogues where the query predicate sat on a rewrite cycle. On the 90 where nothing reachable was cyclic, it terminated with the right answers. The other 49, with cycles reachable only through a join rule, are left undecided. Hypothesis reports these as 61.70%, 21.28% and 11.58% of 423 draws, 23 of which were invalid.
- Depth-bounded rewriting always terminated, and every answer it gave was true.

The same tests found a real bug on the way. Alias canonicalisation dropped a swapped rewrite inside one alias class, which made that class lose its symmetry. It is fixed and pinned as a regression test.

### Why the tabled strategies always stop

With a frozen finite catalogue of predicates, rewrites that only rename or swap, function-free user rules, and every rewritten predicate tabled, every query terminates:

- The memo expands each predicate at most once, so discovery runs at most once per catalogue predicate and the program is fixed after finitely many expansions.
- The program has no function symbols. Every argument is a knowledge-base constant or a variable, so there are finitely many distinct goals and answers.
- SLG resolution evaluates each distinct goal once and each answer once, and it terminates on programs with this bounded term-size property (Verbaeten, De Schreye and Sagonas, arXiv cs/0003045; Riguzzi and Swift, arXiv 1110.0631, Theorem 3).

### Why the untabled strategies loop

Without tables, a cycle of rewrites such as mother ⇐ mum ⇐ mother repeats the same goal forever, since a renaming consumes nothing. The property tests below confirm this exactly: untabled rewriting never stops when the query sits on a rewrite cycle, and stops with the right answers when nothing it reaches is cyclic.

### When it could still loop with tables

- A rewrite builds a bigger term each step, such as one that wraps an argument in a function. That breaks the bounded term-size property.
- Discovery invents predicates that weren't in the catalogue, so the catalogue never freezes.
- Discovery pulls in new constants without bound, for example by extracting facts from an open text corpus.

None of these happen in the design built here.

| strategy | budget (inferences) | status | inferences | expansions | proposals | answers | seconds |
|---|---|---|---|---|---|---|---|
| naive | 10,000 | inference_limit | 67,744 | 0 | 0 | 0 | 0.032 |
| memo-untabled | 10,000 | inference_limit | 67,748 | 0 | 0 | 0 | 0.030 |
| lazy-tabled | 10,000 | ok | 3,570 | 5 | 255 | 14 | 0.052 |
| naive | 100,000 | inference_limit | 100,738 | 126 | 6,426 | 0 | 0.046 |
| memo-untabled | 100,000 | inference_limit | 100,737 | 4 | 204 | 0 | 0.040 |
| lazy-tabled | 100,000 | ok | 3,570 | 5 | 255 | 14 | 0.052 |
| naive | 1,000,000 | inference_limit | 1,003,653 | 3,632 | 185,232 | 0 | 0.734 |
| memo-untabled | 1,000,000 | inference_limit | 1,003,166 | 4 | 204 | 0 | 0.641 |
| lazy-tabled | 1,000,000 | ok | 3,570 | 5 | 255 | 14 | 0.061 |
| naive | 3,000,000 | inference_limit | 3,000,739 | 11,424 | 582,624 | 0 | 4.548 |
| memo-untabled | 3,000,000 | inference_limit | 3,000,737 | 4 | 204 | 0 | 4.034 |
| lazy-tabled | 3,000,000 | ok | 3,570 | 5 | 255 | 14 | 0.060 |

The query is mother(X, Y) on the family benchmark, where mother, mum and mom are aliases of each other, with a perfect judge.

- **Tabled rewriting:** answers all 14 pairs with 5 expansions and 3,570 inferences, whatever the budget.
- **Untabled rewriting:** spends whatever budget it gets and answers nothing.
- **Naive rewriting:** without a memo, it also keeps asking the judge, so its judge proposals grow with the budget, to 582,624 at 3M inferences.

At the 10k budget, both untabled runs stop after about 67,700 counted inferences, before their first discovery call, so the inference limit overshoots budgets this small.

### Tables stop loops, not blow-ups

Termination is guaranteed, but the guarantee says nothing about cost. When a judge admits rewrites between most pairs of predicates, nearly every predicate reaches nearly every other, and the fixpoint becomes very large. With the budget lifted, every hard query below still terminated. Compared with the perfect judge on the same query, nli-deberta-base at its default threshold needed 4 to 1,560 times the engine work (inside(X, palir): 70,198 against 14,052,792 inferences) and up to 1,437 times as many tables (brings_about(X, numbness): 6 against 8,620). It found every true answer, but most of what it returned was false: 3 true answers out of 26 for superior_of(X, finn).

| query | judge | strategy | status | inferences | tables | rewrites | true/answers | gold |
|---|---|---|---|---|---|---|---|---|
| parent(X, Y) | nli-deberta-base | lazy-tabled | ok | 24,389 | 20 | 69 | 30/80 | 30 |
| parent(X, Y) | nli-deberta-base | eager-worklist | ok | 18,572 | 19 | 69 | 30/80 | 30 |
| parent(X, Y) | oracle | lazy-tabled | ok | 6,410 | 11 | 28 | 30/30 | 30 |
| parent(X, Y) | oracle | eager-worklist | ok | 4,203 | 10 | 28 | 30/30 | 30 |
| ancestor(X, leo) | nli-deberta-base | lazy-tabled | ok | 422,906 | 1,712 | 77 | 6/17 | 6 |
| ancestor(X, leo) | nli-deberta-base | eager-worklist | ok | 132,664 | 1,612 | 77 | 6/17 | 6 |
| ancestor(X, leo) | oracle | lazy-tabled | ok | 81,972 | 350 | 28 | 6/6 | 6 |
| ancestor(X, leo) | oracle | eager-worklist | ok | 20,331 | 320 | 28 | 6/6 | 6 |
| superior_of(X, finn) | nli-deberta-base | lazy-tabled | ok | 6,114,069 | 10,835 | 102 | 3/26 | 3 |
| superior_of(X, finn) | nli-deberta-base | eager-worklist | ok | 3,613,286 | 10,835 | 102 | 3/26 | 3 |
| superior_of(X, finn) | oracle | lazy-tabled | ok | 23,580 | 111 | 7 | 2/2 | 3 |
| superior_of(X, finn) | oracle | eager-worklist | ok | 7,357 | 111 | 7 | 2/2 | 3 |
| inside(X, palir) | nli-deberta-base | lazy-tabled | ok | 14,052,792 | 24,989 | 76 | 17/40 | 17 |
| inside(X, palir) | nli-deberta-base | eager-worklist | ok | 8,510,087 | 24,989 | 76 | 17/40 | 17 |
| inside(X, palir) | oracle | lazy-tabled | ok | 70,198 | 261 | 21 | 17/17 | 17 |
| inside(X, palir) | oracle | eager-worklist | ok | 24,346 | 261 | 21 | 17/17 | 17 |
| brings_about(X, numbness) | nli-deberta-base | lazy-tabled | ok | 4,646,219 | 8,620 | 91 | 3/23 | 3 |
| brings_about(X, numbness) | nli-deberta-base | eager-worklist | ok | 2,748,007 | 8,620 | 91 | 3/23 | 3 |
| brings_about(X, numbness) | oracle | lazy-tabled | ok | 2,980 | 6 | 13 | 3/3 | 3 |
| brings_about(X, numbness) | oracle | eager-worklist | ok | 1,889 | 6 | 13 | 3/3 | 3 |

## What implementations are possible, and how do they compare?

With a perfect judge, every tabled strategy gives the same answers. They differ only in cost, and the knobs explain the differences:

- **The memo** decides how often the model is asked. Each expansion asks the judge 2k + 1 = 17 questions: 8 candidates in two argument orders, plus whether the goal is symmetric. So model calls per query are 17 times the number of predicates expanded. Discovering lazily during evaluation asked 107 questions per query. Without a memo, tables repeat discovery for every goal variant and the count rose to 664. Saturating the whole catalogue up front asked 326. Sharing one memo across a benchmark's queries (the warm column) brought goal-directed discovery down to 16 per query.
- **Tables** decide termination. With tables, all 58 queries finished. Without tables, only the 2 queries whose reachable rewrites were acyclic finished. Naive rewriting also re-asks the judge on every call, so it posed 259,295 questions per query before its budget ran out.
- **The depth bound** stops the search but loses answers. At depth 3, recall fell to 0.829 at ten times the engine work of tabling, and the same answer was proved more than once.
- **The encoding** sets the engine cost. Compiling rewrites into native equations (eager-worklist, 4,010 inferences per query) was cheaper than calling discovery through `holds/3` (lazy-tabled, 10,791). Merging alias classes with a union-find (alias-canonical) halved the work again and cut tables from 38 to 21. Rounds, the loop of reducing, discovering and reducing again from your original question, reached the same fixpoint at 16,478 inferences, because every round re-evaluates the query.

Recall is 0.949 rather than 1.0 because of retrieval, not the judge. All the missing answers are on the work benchmark. There, k = 8 nearest glosses failed to bring "A reports to B" close enough to "A is the boss of B" for the swapped rewrite to be proposed. Retrieval recall per benchmark is in the table below; chains through other predicates recover most of what a single retrieval misses.

All runs below use the oracle judge, which admits exactly the gold implications. That isolates what each strategy does from what a judge gets wrong. The figures cover all four benchmarks (family, work, geography, medicine): 75 predicates, 223 facts and 58 queries.

| strategy | precision | recall | terminated | proposals/query cold | proposals/query warm | expansions/query | inferences/query | tables/query | seconds/query |
|---|---|---|---|---|---|---|---|---|---|
| exact | 1.000 | 0.018 | 1.000 | 0.000 | 0.000 | 0.000 | 1,115 | 0.330 | 0.665 |
| naive | 1.000 | 0.009 | 0.035 | 259,295 | n/a | 15,253 | 1,904,593 | 0.000 | 21 |
| memo-untabled | 1.000 | 0.009 | 0.035 | 54 | 11 | 3.170 | 1,960,173 | 0.000 | 7.065 |
| nomemo-tabled | 1.000 | 0.949 | 1.000 | 664 | n/a | 39 | 10,791 | 39 | 1.122 |
| depth-bounded | 1.000 | 0.829 | 1.000 | 100 | 16 | 5.860 | 112,061 | 0.000 | 0.503 |
| lazy-tabled | 1.000 | 0.949 | 1.000 | 107 | 16 | 6.280 | 10,791 | 39 | 0.698 |
| rounds | 1.000 | 0.949 | 1.000 | 107 | 16 | 6.280 | 16,478 | 39 | 4.358 |
| eager-worklist | 1.000 | 0.949 | 1.000 | 107 | 16 | 6.280 | 4,010 | 38 | 2.082 |
| alias-canonical | 1.000 | 0.949 | 1.000 | 107 | 16 | 6.280 | 2,154 | 21 | 1.438 |
| saturate | 1.000 | 0.949 | 1.000 | 326 | 22 | 19 | 4,010 | 38 | 1.365 |

| knowledge base | k | gold implications | premise retrieved | recall |
|---|---|---|---|---|
| family | 8 | 51 | 45 | 0.882 |
| work | 8 | 50 | 35 | 0.700 |
| geography | 8 | 42 | 31 | 0.738 |
| medicine | 8 | 26 | 19 | 0.731 |

Four other implementations were considered and not built:

- **Proof scores instead of admitted rewrites.** This is approach B, as in NLProlog, neural theorem provers and DeepSoftLog: aggregate similarity along each proof and rank the answers. Running the rewriter with the cosine judge already measures similarity-only admission, which is where approach B starts. Proof scores would also change unification, scoring and the stopping condition, so they amount to a different system.
- **Discovery from text for bound subgoals.** This is stage 4 of the design discussion: retrieve passages that mention Alice and extract facts from them. It needs a corpus and a fact extractor that your problem never named. An open corpus also brings in new constants without bound, which is the one case where tables stop guaranteeing termination.
- **Choosing the rewrite set jointly under transitivity.** Entailment graphs (Berant et al. 2011; Hosseini et al. 2018) do this with an integer program. I recorded it as untried; it is the natural next step after direction-aware admission.
- **Bottom-up maintenance with semi-naive deltas.** The `saturate` strategy measures what materialising everything costs in model calls. Incremental maintenance only pays off when facts keep arriving, which this experiment does not simulate.

## Can a cross-encoder or Jev tell which way an implication runs?

Yes. Every judge that reads the two sentences together can tell which way an implication runs, and Jev does it best. Embedding similarity can't, by construction. Direction accuracy is the share of one-way pairs, such as mother ⇒ parent but not the reverse, where the true direction scores higher:

- **Jev:** 0.822 on the Levy/Holt directional test split, 0.895 on Wikidata's property hierarchy and 1.000 on the hand-labelled hard pairs.
- **NLI cross-encoders:** 0.770 to 0.834 on Levy/Holt and 0.790 to 0.871 on Wikidata. Of the three, the WANLI-trained DeBERTa-large is best.
- **Qwen3 reranker, asked yes or no:** 0.694 on Levy/Holt and 0.762 on Wikidata.
- **Both cosine judges:** chance, 0.46 to 0.52.

For mother and parent specifically, Jev scores mother ⇒ parent at 0.99 and parent ⇒ mother at 0.03, and every NLI model is just as clear.

Telling the direction is not the same as admitting only true rules, though. Each judge has to cut its scores at a threshold. The judge thread calibrated those thresholds on Wikidata dev pairs by balanced accuracy, and at those cuts the share of admitted implications that are true is only 0.33 to 0.54 on Wikidata's test split. The named cases show why. Legal guardian ⇒ parent, stepmother ⇒ parent and employs ⇔ manages pass for most judges, and Jev's low cut (0.165) lets through employs ⇔ manages and stepmother ⇒ parent even though it scores them low. These pair-set measurements are the judge thread's (`results/judges.json`).

### The cases you named

Scores on the hand-labelled hard set, with arguments written as A and B. A star means the score reaches that judge's default threshold, so the rewriter would admit the implication.

| implication | holds | cosine-bge | cosine-qwen3 | jev | nli-deberta-base | nli-deberta-large | nli-deberta-large-wanli | reranker-qwen3 |
|---|---|---|---|---|---|---|---|---|
| A is the mother of B ⇒ A is a parent of B | yes | 0.97* | 0.98* | 0.99* | 0.99* | 1.00* | 1.00* | 0.99* |
| A is a parent of B ⇒ A is the mother of B | no | 0.97* | 0.98* | 0.03 | 0.00 | 0.03 | 0.00 | 0.38 |
| A is the legal guardian of B ⇒ A is a parent of B | no | 0.90 | 0.97 | 0.05 | 0.98* | 0.82* | 0.00 | 0.97* |
| A is the stepmother of B ⇒ A is a parent of B | no | 0.93 | 0.97 | 0.32* | 0.99* | 0.99* | 0.95* | 0.94* |
| A is the mother-in-law of B ⇒ A is the mother of B | no | 0.97* | 0.99* | 0.05 | 0.99* | 0.96* | 0.00 | 0.96* |
| A employs B ⇒ A manages B | no | 0.93 | 0.97 | 0.29* | 0.99* | 1.00* | 0.98* | 0.92* |
| A manages B ⇒ A employs B | no | 0.93 | 0.97 | 0.18* | 0.99* | 0.99* | 0.96* | 0.98* |
| A causes B ⇒ A is correlated with B | no | 0.90 | 0.95 | 0.65* | 0.98* | 0.97* | 0.39* | 0.82* |
| A causes B ⇒ A prevents B | no | 0.91 | 0.96 | 0.01 | 0.00 | 0.00 | 0.00 | 0.41 |

### On labelled predicate pairs

| judge | pair set | rows | one-way pairs | direction accuracy | ROC AUC | threshold | precision | recall | balanced accuracy |
|---|---|---|---|---|---|---|---|---|---|
| cosine-bge | levyholt | 1,784 | 892 | 0.483 | 0.491 | 0.957 | 0.469 | 0.043 | 0.497 |
| cosine-bge | wikidata | 3,329 | 552 | 0.462 | 0.531 | 0.957 | 0.314 | 0.118 | 0.526 |
| cosine-bge | hard | 140 | 29 | 0.517 | 0.648 | 0.957 | 0.518 | 0.518 | 0.598 |
| cosine-qwen3 | levyholt | 1,784 | 892 | 0.495 | 0.502 | 0.975 | 0.529 | 0.040 | 0.502 |
| cosine-qwen3 | wikidata | 3,329 | 552 | 0.491 | 0.532 | 0.975 | 0.259 | 0.180 | 0.524 |
| cosine-qwen3 | hard | 140 | 29 | 0.483 | 0.671 | 0.975 | 0.507 | 0.625 | 0.610 |
| jev | levyholt | 1,784 | 892 | 0.822 | 0.811 | 0.165 | 0.770 | 0.678 | 0.738 |
| jev | wikidata | 3,329 | 552 | 0.895 | 0.878 | 0.165 | 0.536 | 0.830 | 0.823 |
| jev | hard | 140 | 29 | 1.000 | 0.988 | 0.165 | 0.789 | 1.000 | 0.911 |
| nli-deberta-base | levyholt | 1,784 | 892 | 0.804 | 0.740 | 0.222 | 0.691 | 0.672 | 0.686 |
| nli-deberta-base | wikidata | 3,329 | 552 | 0.790 | 0.744 | 0.222 | 0.400 | 0.666 | 0.706 |
| nli-deberta-base | hard | 140 | 29 | 0.897 | 0.866 | 0.222 | 0.642 | 0.929 | 0.792 |
| nli-deberta-large | levyholt | 1,784 | 892 | 0.770 | 0.720 | 0.659 | 0.710 | 0.565 | 0.667 |
| nli-deberta-large | wikidata | 3,329 | 552 | 0.851 | 0.754 | 0.659 | 0.443 | 0.583 | 0.698 |
| nli-deberta-large | hard | 140 | 29 | 1.000 | 0.841 | 0.659 | 0.662 | 0.839 | 0.777 |
| nli-deberta-large-wanli | levyholt | 1,784 | 892 | 0.834 | 0.779 | 0.054 | 0.671 | 0.817 | 0.708 |
| nli-deberta-large-wanli | wikidata | 3,329 | 552 | 0.871 | 0.802 | 0.054 | 0.423 | 0.753 | 0.745 |
| nli-deberta-large-wanli | hard | 140 | 29 | 0.966 | 0.919 | 0.054 | 0.737 | 1.000 | 0.881 |
| reranker-qwen3 | levyholt | 1,784 | 892 | 0.694 | 0.656 | 0.808 | 0.631 | 0.614 | 0.628 |
| reranker-qwen3 | wikidata | 3,329 | 552 | 0.762 | 0.668 | 0.808 | 0.327 | 0.535 | 0.627 |
| reranker-qwen3 | hard | 140 | 29 | 0.828 | 0.778 | 0.808 | 0.577 | 0.804 | 0.705 |

### Inside the rewriter

The rewriter turns every admission error into a logical error. A false rewrite such as legal guardian ⇒ parent becomes a rule, and that rule adds false answers to every query that reaches parent. Enough false rewrites connect nearly every predicate to every other, and the closure blows up, as the previous section showed. At the default thresholds, results split into three groups:

- **Jev:** lazy-tabled reaches answer F1 0.765, with precision 0.820 and recall 0.718, and finishes 76% of queries inside the 500k-inference budget.
- **The NLI cross-encoders:** they admit rewrites at a precision of 0.35 to 0.43, so 60 to 71% of their queries run out of budget, and F1 is 0.34 to 0.53. On the queries that do finish, recall is 1.000.
- **Cosine similarity:** it admits both directions and both argument orders, so its rewrite precision is the lowest, 0.18 to 0.24.

A depth bound hides the blow-up rather than curing it. With Jev, depth-bounded finished every query at F1 0.815, but its precision of 0.727 is below lazy-tabled's 0.820.

| judge | rewrites admitted | rewrite precision | reversed | wrong argument order | trap | unrelated | finished (default threshold) | answer F1 (default threshold) | answer F1 (held-out threshold) |
|---|---|---|---|---|---|---|---|---|---|
| jev | 301 | 0.449 | 13 | 0 | 21 | 132 | 0.759 | 0.765 | 0.839 |
| nli-deberta-base | 365 | 0.351 | 23 | 16 | 44 | 154 | 0.328 | 0.342 | 0.500 |
| nli-deberta-large | 279 | 0.427 | 21 | 2 | 22 | 115 | 0.397 | 0.528 | 0.628 |
| nli-deberta-large-wanli | 344 | 0.398 | 22 | 18 | 29 | 138 | 0.293 | 0.346 | 0.520 |
| reranker-qwen3 | 401 | 0.264 | 26 | 38 | 37 | 194 | 0.103 | 0.097 | 0.448 |
| cosine-bge | 225 | 0.244 | 22 | 22 | 31 | 95 | 0.966 | 0.372 | 0.137 |
| cosine-qwen3 | 342 | 0.181 | 29 | 43 | 37 | 171 | 0.707 | 0.313 | 0.054 |

### Choosing the threshold for the rewriter

The pair-set thresholds maximise balanced accuracy, which favours recall. The rewriter needs precision, because it applies every admitted rewrite everywhere. So I also calibrated each judge's threshold for the rewriter's own objective, answer F1. For each benchmark, the threshold is the one that maximises F1 on the other three, and it is then scored on the held-out benchmark.

- **Jev** picks 0.62 on every fold and reaches F1 0.839: precision 0.791 and recall 0.893.
- **The NLI cross-encoders** score most related predicates near 1.0, so their useful thresholds sit between 0.958 and 0.998. They transfer poorly between benchmarks, with held-out F1 of 0.50 to 0.63, though the best single threshold chosen in hindsight reaches 0.59 to 0.78.
- **Similarity alone (cosine)** gives F1 0.05 to 0.37 whatever the threshold.

| judge | default threshold | F1 at default | held-out threshold per benchmark | F1 held-out | best threshold in hindsight | F1 | precision | recall |
|---|---|---|---|---|---|---|---|---|
| jev | 0.165 | 0.765 | family 0.620, work 0.620, geography 0.620, medicine 0.620 | 0.839 | 0.620 | 0.839 | 0.791 | 0.893 |
| nli-deberta-base | 0.222 | 0.342 | family 0.992, work 0.983, geography 0.983, medicine 0.958 | 0.500 | 0.983 | 0.592 | 0.905 | 0.440 |
| nli-deberta-large | 0.659 | 0.528 | family 0.998, work 0.995, geography 0.995, medicine 0.995 | 0.628 | 0.995 | 0.784 | 0.822 | 0.750 |
| nli-deberta-large-wanli | 0.054 | 0.346 | family 0.991, work 0.991, geography 0.979, medicine 0.993 | 0.520 | 0.991 | 0.743 | 0.970 | 0.602 |
| reranker-qwen3 | 0.808 | 0.097 | family 0.984, work 0.984, geography 0.994, medicine 0.984 | 0.448 | 0.984 | 0.579 | 0.453 | 0.801 |
| cosine-bge | 0.957 | 0.372 | family 0.943, work 0.957, geography 0.957, medicine 0.957 | 0.137 | 0.957 | 0.372 | 0.445 | 0.319 |
| cosine-qwen3 | 0.975 | 0.313 | family 0.973, work 0.986, geography 0.986, medicine 0.986 | 0.054 | 0.986 | 0.328 | 0.448 | 0.259 |

| judge | strategy | finished | precision | recall | F1 | precision (finished) | recall (finished) | proposals/query |
|---|---|---|---|---|---|---|---|---|
| cosine-bge | exact | 1.000 | 1.000 | 0.018 | 0.036 | 1.000 | 0.018 | 0.000 |
| cosine-bge | depth-bounded | 0.966 | 0.500 | 0.333 | 0.400 | 0.500 | 0.340 | 80 |
| cosine-bge | lazy-tabled | 0.966 | 0.445 | 0.319 | 0.372 | 0.445 | 0.338 | 100 |
| cosine-bge | eager-worklist | 0.966 | 0.445 | 0.319 | 0.372 | 0.445 | 0.338 | 100 |
| cosine-qwen3 | exact | 1.000 | 1.000 | 0.018 | 0.036 | 1.000 | 0.018 | 0.000 |
| cosine-qwen3 | depth-bounded | 0.879 | 0.503 | 0.329 | 0.398 | 0.503 | 0.425 | 115 |
| cosine-qwen3 | lazy-tabled | 0.707 | 0.366 | 0.273 | 0.313 | 0.366 | 0.391 | 144 |
| cosine-qwen3 | eager-worklist | 0.707 | 0.366 | 0.273 | 0.313 | 0.366 | 0.391 | 144 |
| jev | exact | 1.000 | 1.000 | 0.018 | 0.036 | 1.000 | 0.018 | 0.000 |
| jev | depth-bounded | 1.000 | 0.727 | 0.926 | 0.815 | 0.727 | 0.926 | 150 |
| jev | lazy-tabled | 0.759 | 0.820 | 0.718 | 0.765 | 0.820 | 0.975 | 166 |
| jev | eager-worklist | 0.759 | 0.820 | 0.718 | 0.765 | 0.820 | 0.975 | 166 |
| nli-deberta-base | exact | 1.000 | 1.000 | 0.018 | 0.036 | 1.000 | 0.018 | 0.000 |
| nli-deberta-base | depth-bounded | 1.000 | 0.500 | 0.857 | 0.631 | 0.500 | 0.857 | 171 |
| nli-deberta-base | lazy-tabled | 0.328 | 0.381 | 0.310 | 0.342 | 0.381 | 1.000 | 258 |
| nli-deberta-base | eager-worklist | 0.328 | 0.381 | 0.310 | 0.342 | 0.381 | 1.000 | 258 |
| nli-deberta-large | exact | 1.000 | 1.000 | 0.018 | 0.036 | 1.000 | 0.018 | 0.000 |
| nli-deberta-large | depth-bounded | 0.966 | 0.569 | 0.759 | 0.651 | 0.569 | 0.921 | 162 |
| nli-deberta-large | lazy-tabled | 0.397 | 0.919 | 0.370 | 0.528 | 0.919 | 1.000 | 188 |
| nli-deberta-large | eager-worklist | 0.397 | 0.919 | 0.370 | 0.528 | 0.919 | 1.000 | 188 |
| nli-deberta-large-wanli | exact | 1.000 | 1.000 | 0.018 | 0.036 | 1.000 | 0.018 | 0.000 |
| nli-deberta-large-wanli | depth-bounded | 0.966 | 0.392 | 0.768 | 0.519 | 0.392 | 0.933 | 164 |
| nli-deberta-large-wanli | lazy-tabled | 0.293 | 0.539 | 0.255 | 0.346 | 0.539 | 1.000 | 201 |
| nli-deberta-large-wanli | eager-worklist | 0.328 | 0.493 | 0.310 | 0.381 | 0.493 | 1.000 | 201 |
| reranker-qwen3 | exact | 1.000 | 1.000 | 0.018 | 0.036 | 1.000 | 0.018 | 0.000 |
| reranker-qwen3 | depth-bounded | 0.828 | 0.361 | 0.657 | 0.466 | 0.361 | 0.922 | 193 |
| reranker-qwen3 | lazy-tabled | 0.103 | 1.000 | 0.051 | 0.097 | 1.000 | 0.733 | 244 |
| reranker-qwen3 | eager-worklist | 0.103 | 1.000 | 0.051 | 0.097 | 1.000 | 0.733 | 244 |

### Asking the judge both ways

Your question suggested this: you can ask the cross-encoder which way the implication runs, and not only whether it holds. Direction-aware admission scores each proposal together with its converse. It admits "Q ⇒ P" only when the judge does not score "P ⇒ Q" higher by more than a slack, and the slack is calibrated per judge on Wikidata dev pairs. A true equivalence such as mum ⇔ mother scores nearly the same both ways and survives. A one-way implication read backwards, such as parent ⇒ mum at 0.955 against mum ⇒ parent at 0.998, is dropped.

The cost is one extra judge question per candidate. In return, the rewriter's false-rewrite rate fell for every cross-encoder, and answer F1 rose or held. The largest gains were for the reranker, from 0.097 to 0.350, because far fewer runs blew past the budget, and for the WANLI model, from 0.346 to 0.381. On the Levy/Holt test pairs, precision rose 3 to 11 points for 1.5 to 6 points of recall. Traps that the judge reads as entailments in only one direction, such as legal guardian ⇒ parent, are untouched by this check.

| judge | policy | slack | Wikidata precision | Wikidata recall | Wikidata balanced accuracy | answer precision | answer recall | answer F1 | false rewrites |
|---|---|---|---|---|---|---|---|---|---|
| jev | plain |  | 0.536 | 0.830 | 0.823 | 0.820 | 0.718 | 0.765 | 0.462 |
| jev | directional | 0.200 | 0.574 | 0.801 | 0.824 | 0.820 | 0.718 | 0.765 | 0.435 |
| nli-deberta-base | plain |  | 0.400 | 0.666 | 0.706 | 0.381 | 0.310 | 0.342 | 0.623 |
| nli-deberta-base | directional | 0.100 | 0.413 | 0.648 | 0.706 | 0.381 | 0.310 | 0.342 | 0.611 |
| nli-deberta-large | plain |  | 0.443 | 0.584 | 0.698 | 0.919 | 0.370 | 0.528 | 0.536 |
| nli-deberta-large | directional | 0.020 | 0.464 | 0.573 | 0.702 | 0.941 | 0.370 | 0.532 | 0.499 |
| nli-deberta-large-wanli | plain |  | 0.422 | 0.753 | 0.745 | 0.539 | 0.255 | 0.346 | 0.603 |
| nli-deberta-large-wanli | directional | 0.050 | 0.470 | 0.711 | 0.753 | 0.493 | 0.310 | 0.381 | 0.482 |
| reranker-qwen3 | plain |  | 0.327 | 0.538 | 0.628 | 1.000 | 0.051 | 0.097 | 0.718 |
| reranker-qwen3 | directional | 0.010 | 0.384 | 0.473 | 0.640 | 0.459 | 0.282 | 0.350 | 0.677 |

## Is pymetta stable enough?

Yes, for this experiment. Everything the rewriter needs from upstream pymetta 0.9.4 worked, and the defects it hit have workarounds. Each item below was checked in this session:

- **Tabling:** `(tabled ...)` from `lib_tabling` made cyclic rewrite programs terminate. SWI-Prolog's SLG resolution does the work underneath.
- **Ops inside tables:** a Python generator op called inside tabled `holds/3` ran once per table, which is what discovery inside SLG needs.
- **Table refresh:** adding an equation to a tabled function refreshed its table, and the rounds strategy depends on that.
- **Budgets:** the inference and time budgets of `eval()` stopped every runaway query.
- **Tests:** all of this held across the engine tests and the 1,200 property-test catalogues.

It also has four defects worth reporting upstream. Each has a script under `repros/`:

| defect | reproduction |
|---|---|
| An inference limit reached while a Python op runs can arrive as a plain `EngineError` whose cause is a janus `PrologError`, so `except InferenceLimitError` misses it. It appears only after an earlier context in the process ran the exact strategy. The engine works around it by reading its own inference count. | `repros/limit_wrapped_after_exact.py` |
| Symbol-use state crosses contexts. Evaluating a call to an undefined predicate in one context makes a fresh context warn that the predicate "is defined after already being used". Answers stay correct: the isolation check below ran all 170 family-benchmark oracle runs both in one shared process and in a fresh process each, and found no difference. | `repros/symbol_use_leaks_across_contexts.py` |
| `run()` takes 4 to 63 times longer than `eval()` to exhaust the same inference budget, and the gap grows faster than the budget. | `repros/run_budget_slower_than_eval.py` |
| A budget that interrupts tabled evaluation printed a raw SWI message, `Unknown message: tabling(unexpected_result(...))`, to stderr during the cosine sweep. My attempt to reproduce it did not trigger it. | none yet |

Three things caught me out, but they're MeTTa semantics rather than defects:

- A rewrite that returns its own goal is reduced again, forever. Return `True` or a value that can't be reduced.
- A tuple whose head names a defined function is evaluated. `(mother "gloss")` came back as a partial application, so predicate names carried as data need a constructor head that isn't a function.
- Without tables, a rewrite cycle runs in constant stack, so `max-stack-depth` and the stack limit never stop it. Only an inference or time budget does.

| knowledge base | runs compared | mismatches |
|---|---|---|
| family | 170 | 0 |

## Reproduce

```sh
uv sync
.venv/bin/python -m pytest tests/
.venv/bin/python experiments/judge_eval.py
.venv/bin/python experiments/impl_eval.py --judges oracle --tag oracle
.venv/bin/python experiments/impl_eval.py --judges cosine-bge,cosine-qwen3,reranker-qwen3,jev \
    --strategies exact,nomemo-tabled,depth-bounded,lazy-tabled,rounds,eager-worklist,alias-canonical,saturate \
    --skip-extras --sweep --tag judges-a
.venv/bin/python experiments/impl_eval.py --judges nli-deberta-base,nli-deberta-large,nli-deberta-large-wanli \
    --strategies exact,nomemo-tabled,depth-bounded,lazy-tabled,rounds,eager-worklist,alias-canonical,saturate \
    --skip-extras --sweep --tag judges-b
.venv/bin/python experiments/report.py
.venv/bin/python examples/parent_of_tom.py nli-deberta-large
```

Here is what each step produces:

- **`pytest`** runs the engine tests, the termination properties and the judge tests.
- **`judge_eval.py`** writes `results/judges.json` and the calibrated thresholds in `semrewrite/judges/thresholds.json`. Jev's answers are cached under `data/raw/jev`, so a rerun sends no requests unless it asks something new; a new question needs a TypeSafe key.
- **`impl_eval.py`** writes `results/implementations.<tag>.json` and `.jsonl`.
- **`report.py`** fills the tables in this file.
- **`parent_of_tom.py`** runs your own example with any judge.
