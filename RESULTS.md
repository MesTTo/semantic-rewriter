# Semantic rewriter on pymetta: results

This file answers the four questions behind the experiment. The tables between `GENERATED` markers come from the files under `results/`, rendered by `python experiments/report.py`; the commands that regenerate those files are at the end.

## Short answers

Does semantic reduction loop? No, as long as goals are tabled and each predicate is expanded once. Every tabled strategy terminated with exactly the right answers on all 400 generated catalogues, each containing an alias cycle. Untabled rewriting loops exactly when the goal sits on a cycle of rewrites. Tables stop loops, not blow-ups, though: with a permissive judge, the finite closure cost up to 1,559 times the engine work.

Which implementation? Eleven were built, as settings of five knobs. The one to use is goal-directed discovery with tables and a per-predicate memo. Discovering inside SLG evaluation (`lazy-tabled`) is the natural fit; compiling rewrites to native equations (`eager-worklist`) and merging aliases with a union-find (`alias-canonical`) are cheaper in engine work. Each of the three costs about 116 judge questions per query on its own, or 17 once a benchmark's queries share the memo. With an imperfect judge, also bound the depth: tabled evaluation bounded at three steps (`depth-tabled`) raised answer F1 at calibrated thresholds for six of the seven judges.

Can a cross-encoder or Jev tell which way? Yes. On the Levy/Holt and Wikidata pairs, Jev gets 0.82 to 0.89 direction accuracy, the NLI cross-encoders 0.77 to 0.87, and embedding similarity sits at chance. Inside the rewriter, the deciding quantity is admission precision, because every false rewrite becomes a rule applied everywhere. Jev at a threshold calibrated on the other benchmarks reaches answer F1 0.892. On the same basis the best cross-encoder reaches 0.70, and 0.91 once evaluation is tabled and bounded at depth 3; similarity alone reaches at most 0.47. Asking each judge both ways and rejecting the weaker direction lowered every cross-encoder's false-rewrite rate.

Is pymetta stable enough? Yes. Tabling, ops called inside tables, table refresh and budgets all did what the rewriter needed. I found eight defects, five of which appear only after earlier evaluations in the same process; five have a script under `repros/`, and none blocked the work.

## What was built

A query such as `(parent $x tom)` finds nothing in a knowledge base that only states `(mother alice tom)` and `(father bob tom)`, because MeTTa rewrites a goal only with an equation whose head matches it symbol for symbol. The semantic rewriter keeps pymetta's evaluator, unification and tabling unchanged and adds one step where a goal is first reached:

1. Retrieve the k catalogue predicates whose glosses are nearest the goal's (k = 8, BAAI/bge-small-en-v1.5).
2. Ask a judge, for each candidate Q, whether `Q(A, B)` entails `goal(A, B)` and whether `Q(B, A)` does, and whether the goal is symmetric.
3. Admit each proposal that reaches the judge's threshold as an ordinary rewrite: `(= (parent $X $Y) (mother $X $Y))`, or `(= (parent $X $Y) (child_of $Y $X))` when the arguments swap.

Each rewrite only renames a predicate or swaps its two arguments. Rewriting a parent goal into a mother goal is sound because every mother answer is a parent answer; the reverse rewrite would be unsound, so the judge's direction matters.

The strategies are settings of five independent knobs:

| strategy | discovery timing | memo | tables | depth bound | aliases |
|---|---|---|---|---|---|
| exact | none | | on | | |
| naive | lazy, inside evaluation | off | off (SLD) | | |
| memo-untabled | lazy | on | off | | |
| nomemo-tabled | lazy | off | on (SLG) | | |
| depth-bounded | lazy | on | off | 3 | |
| depth-tabled | lazy | on | on | 3 | |
| lazy-tabled | lazy | on | on | | |
| rounds | evaluate, expand what was reached, repeat to a fixpoint | on | on | | |
| eager-worklist | worklist from the query before evaluation | on | on | | |
| alias-canonical | eager | on | on | | union-find renames each alias class to one predicate |
| saturate | every catalogue predicate before evaluation | on | on | | |

`lazy` runs discovery through a pymetta op (`impliers`) that the tabled `holds/3` clause calls when evaluation first reaches a predicate. That is SLG resolution with semantic goal expansion. `rounds` is the loop the experiment started from: reduce, discover for whatever was reached, add the rewrites, reduce again, stop when a round reaches nothing new. A depth bound of 3 allows at most three rewrite or rule steps below the query; `depth-tabled` keys each table on the depth left as well, so it finds exactly what `depth-bounded` finds without proving anything twice.

## How answers and rewrites are scored

The four benchmarks (family, work, geography and medicine) are generated worlds with 75 predicates, 223 facts, 10 user rules and 58 queries in all. Each states its facts under many surface predicates, such as mother, mum, mom and has_mother. Each also comes with gold implications that I wrote, such as mother ⇒ parent or mother(y, x) ⇒ has_mother(x, y), and with traps that look related but aren't, such as legal guardian ⇒ parent.

A gold answer is one that the facts, the gold implications and the user rules entail. A rewrite such as mother ⇒ parent counts as true when the gold program entails it. The test takes one fact mother(a, b) over two fresh constants, computes everything the gold implications and user rules derive from it, and looks for parent(a, b), or for parent(b, a) when the rewrite swaps the arguments. For Horn rules this one fixpoint decides entailment; it is the standard test for Datalog query containment.

I audited the gold once, against every rewrite the judges admitted, because my first labels missed implications that the glosses force. Of 467 distinct rewrites that the five direction-reading judges admitted and the first gold called false, the ones that three or more judges admitted included mother(y, x) ⇒ has_mother(x, y), parent, father and grandparent ⇒ ancestor, boss_of and skip_level_boss ⇒ superior_of, contains(y, x) ⇒ inside(x, y), capital_of and largest_city_of ⇒ city_in, and the symmetry of near. All of these are true by the glosses' meaning. The first gold also treated geographic containment as direct only, which its English glosses deny, and its truth test read the gold implications but ignored the user rules. The audit added the missing implications, made containment transitive, retyped the work glosses whose second argument is a company, and replaced the old test with the entailment test above.

Every number in this file comes from runs on the audited benchmarks. The audit raised every judge's rewrite precision, by 0.06 to 0.24, and every judge's answer F1 at a threshold calibrated on the other benchmarks; Jev's went from 0.839 to 0.892. `results/before_gold_audit.md` keeps this file as it stood on the first gold.

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

Without tables, a cycle of rewrites such as mother ⇐ mum ⇐ mother repeats the same goal forever, since a renaming consumes nothing. The property tests above confirm this exactly: untabled rewriting never stops when the query sits on a rewrite cycle, and stops with the right answers when nothing it reaches is cyclic.

### When it could still loop with tables

- A rewrite builds a bigger term each step, such as one that wraps an argument in a function. That breaks the bounded term-size property.
- Discovery invents predicates that weren't in the catalogue, so the catalogue never freezes.
- Discovery pulls in new constants without bound, for example by extracting facts from an open text corpus.

None of these happen in the design built here.

<!-- BEGIN GENERATED: divergence -->
| strategy | budget (inferences) | status | inferences | expansions | proposals | answers | seconds |
|---|---|---|---|---|---|---|---|
| naive | 10,000 | work_limit | 100,112 | 0 | 0 | 0 | 1.888 |
| memo-untabled | 10,000 | work_limit | 100,116 | 0 | 0 | 0 | 1.866 |
| lazy-tabled | 10,000 | ok | 4,172 | 5 | 255 | 14 | 2.630 |
| naive | 100,000 | work_limit | 100,786 | 0 | 0 | 0 | 2.051 |
| memo-untabled | 100,000 | work_limit | 100,788 | 0 | 0 | 0 | 2.411 |
| lazy-tabled | 100,000 | ok | 4,172 | 5 | 255 | 14 | 3.005 |
| naive | 1,000,000 | work_limit | 1,005,366 | 4,399 | 224,349 | 0 | 2.942 |
| memo-untabled | 1,000,000 | work_limit | 1,005,366 | 4 | 204 | 0 | 2.903 |
| lazy-tabled | 1,000,000 | ok | 4,172 | 5 | 255 | 14 | 2.744 |
| naive | 3,000,000 | work_limit | 3,014,970 | 14,227 | 725,577 | 0 | 7.775 |
| memo-untabled | 3,000,000 | work_limit | 3,015,191 | 4 | 204 | 0 | 7.299 |
| lazy-tabled | 3,000,000 | ok | 4,172 | 5 | 255 | 14 | 2.722 |
<!-- END GENERATED: divergence -->

The query is mother(X, Y) on the family benchmark, where mother, mum and mom are aliases of each other, with a perfect judge.

- Tabled rewriting answers all 14 pairs with 5 expansions and 4,172 inferences, whatever the budget.
- Untabled rewriting spends whatever budget it gets and answers nothing.
- Naive rewriting has no memo, so it also keeps asking the judge, and its judge proposals grow with the budget, to 725,577 at 3M inferences.

At the 10k budget, both untabled runs stopped after about 100,100 counted inferences, before their first discovery call, about what they spent at the 100k budget, so the inference limit overshoots budgets this small.

### Tables stop loops, not blow-ups

Termination is guaranteed, but the guarantee says nothing about cost. When a judge admits rewrites between most pairs of predicates, nearly every predicate reaches nearly every other, and the fixpoint becomes very large. With the budget lifted, every hard query below still terminated. Compared with the perfect judge on the same query and strategy, nli-deberta-base at its default threshold needed 1.6 to 1,559 times the engine work and up to 1,437 times as many tables. Both extremes are on brings_about(X, numbness): 4,646,219 inferences against 2,980, and 8,620 tables against 6. The largest closure, for inside(X, palir), took 17.5 million inferences against the perfect judge's 119,970. Most of what the permissive judge returned was false: 3 of its 23 answers to brings_about(X, numbness) were true, and 30 of its 80 answers to parent(X, Y). It found every true answer except one of the three for superior_of(X, finn).

<!-- BEGIN GENERATED: blowup -->
| query | judge | strategy | status | inferences | tables | rewrites | true/answers | gold |
|---|---|---|---|---|---|---|---|---|
| parent(X, Y) | nli-deberta-base | lazy-tabled | ok | 24,361 | 20 | 69 | 30/80 | 30 |
| parent(X, Y) | nli-deberta-base | eager-worklist | ok | 18,572 | 19 | 69 | 30/80 | 30 |
| parent(X, Y) | oracle | lazy-tabled | ok | 7,012 | 11 | 32 | 30/30 | 30 |
| parent(X, Y) | oracle | eager-worklist | ok | 4,760 | 10 | 32 | 30/30 | 30 |
| ancestor(X, leo) | nli-deberta-base | lazy-tabled | ok | 423,142 | 1,712 | 77 | 6/17 | 6 |
| ancestor(X, leo) | nli-deberta-base | eager-worklist | ok | 132,900 | 1,612 | 77 | 6/17 | 6 |
| ancestor(X, leo) | oracle | lazy-tabled | ok | 88,837 | 380 | 37 | 6/6 | 6 |
| ancestor(X, leo) | oracle | eager-worklist | ok | 24,659 | 350 | 37 | 6/6 | 6 |
| superior_of(X, finn) | nli-deberta-base | lazy-tabled | ok | 68,410 | 258 | 42 | 2/3 | 3 |
| superior_of(X, finn) | nli-deberta-base | eager-worklist | ok | 22,734 | 258 | 42 | 2/3 | 3 |
| superior_of(X, finn) | oracle | lazy-tabled | ok | 41,554 | 172 | 20 | 3/3 | 3 |
| superior_of(X, finn) | oracle | eager-worklist | ok | 13,462 | 172 | 20 | 3/3 | 3 |
| inside(X, palir) | nli-deberta-base | lazy-tabled | ok | 17,502,284 | 24,999 | 76 | 17/40 | 17 |
| inside(X, palir) | nli-deberta-base | eager-worklist | ok | 11,958,653 | 24,999 | 76 | 17/40 | 17 |
| inside(X, palir) | oracle | lazy-tabled | ok | 119,970 | 336 | 34 | 17/17 | 17 |
| inside(X, palir) | oracle | eager-worklist | ok | 56,021 | 336 | 34 | 17/17 | 17 |
| brings_about(X, numbness) | nli-deberta-base | lazy-tabled | ok | 4,646,219 | 8,620 | 91 | 3/23 | 3 |
| brings_about(X, numbness) | nli-deberta-base | eager-worklist | ok | 2,748,007 | 8,620 | 91 | 3/23 | 3 |
| brings_about(X, numbness) | oracle | lazy-tabled | ok | 2,980 | 6 | 13 | 3/3 | 3 |
| brings_about(X, numbness) | oracle | eager-worklist | ok | 1,889 | 6 | 13 | 3/3 | 3 |
<!-- END GENERATED: blowup -->

## What implementations are possible, and how do they compare?

With a perfect judge, every tabled strategy gives the same answers. They differ only in cost, and the knobs explain the differences:

- The memo decides how often the model is asked. Each expansion asks the judge 2k + 1 = 17 questions: 8 candidates in two argument orders, plus whether the goal is symmetric. So model calls per query are 17 times the number of predicates expanded. Discovering lazily during evaluation asked 116 questions per query. Without a memo, tables repeat discovery for every goal variant and the count rose to 1,013. Saturating the whole catalogue up front asked 326. Sharing one memo across a benchmark's queries (the warm column) brought goal-directed discovery down to 17 per query.
- Tables decide termination. With tables, all 58 queries finished. Without tables, only 2 finished, capital_of(X, mimun) and capital_of(X, finer) on geography, the two queries whose reachable rewrites are acyclic. Naive rewriting also re-asks the judge on every call, so it posed 73,303 questions per query before its 500k-inference budget ran out.
- The depth bound stops the search but loses answers. Untabled at depth 3 (depth-bounded), recall fell to 0.784, and two geography containment queries, inside(X, palir) and inside(X, suner), still ran out of the 500k budget within three steps. It cost 7.3 times the engine work of lazy tabling, and it derived its 174 answers 1,868 times, since nothing stops it proving the same answer along different paths. Keying each table on the depth left as well (depth-tabled) finished all 58 queries at recall 0.955, derived each of its 212 answers once and cost 11,019 inferences per query, 0.6 times lazy tabling's. The answers it loses need more than three steps: two ancestor(X, mia) answers through the family benchmark's own ancestor rule, and one encloses(X, rarir) answer through geography's transitive containment.
- The encoding sets the engine cost. Compiling rewrites into native equations (eager-worklist, 7,722 inferences per query) was cheaper than calling discovery through `holds/3` (lazy-tabled, 18,438). Merging alias classes with a union-find (alias-canonical) cut the work to 2,999 and the tables from 58 to 28 per query. Rounds, the loop of reducing, discovering and reducing again that the experiment started from, reached the same fixpoint at 32,073 inferences, because every round re-evaluates the query.

Recall is 0.969 rather than 1.0 because of retrieval, not the judge. All the missing answers are on the work benchmark, where recall is 0.877. There, the 8 nearest glosses never include "A reports to B" for boss_of, manager_of, supervises, subordinate_of or superior_of, so the rewrite from reports_to is never proposed, and facts stated with reports_to can't be reached. Retrieval recall per benchmark is in the table below; chains through other predicates recover most of what a single retrieval misses.

All runs in these two tables use the oracle judge, which admits exactly the rewrites the gold program entails. That isolates what each strategy does from what a judge gets wrong.

<!-- BEGIN GENERATED: strategies -->
| strategy | precision | recall | finished | proposals/query cold | proposals/query warm | expansions/query | inferences/query | tables/query |
|---|---|---|---|---|---|---|---|---|
| exact | 1.000 | 0.018 | 1.000 | 0.000 | 0.000 | 0.000 | 1,115 | 0.330 |
| naive | 1.000 | 0.009 | 0.035 | 73,303 | n/a | 4,312 | 648,245 | 0.000 |
| memo-untabled | 1.000 | 0.009 | 0.035 | 60 | 12 | 3.550 | 655,808 | 0.000 |
| nomemo-tabled | 1.000 | 0.969 | 1.000 | 1,013 | n/a | 60 | 18,438 | 60 |
| depth-bounded | 1.000 | 0.784 | 0.966 | 110 | 17 | 6.480 | 134,601 | 0.000 |
| depth-tabled | 1.000 | 0.955 | 1.000 | 110 | 17 | 6.480 | 11,019 | 77 |
| lazy-tabled | 1.000 | 0.969 | 1.000 | 116 | 17 | 6.830 | 18,438 | 60 |
| rounds | 1.000 | 0.969 | 1.000 | 116 | 17 | 6.830 | 32,073 | 59 |
| eager-worklist | 1.000 | 0.969 | 1.000 | 116 | 17 | 6.830 | 7,722 | 58 |
| alias-canonical | 1.000 | 0.969 | 1.000 | 116 | 17 | 6.830 | 2,999 | 28 |
| saturate | 1.000 | 0.969 | 1.000 | 326 | 22 | 19 | 7,722 | 58 |
<!-- END GENERATED: strategies -->

<!-- BEGIN GENERATED: retrieval -->
| knowledge base | k | gold implications | premise retrieved | recall |
|---|---|---|---|---|
| family | 8 | 68 | 53 | 0.779 |
| work | 8 | 62 | 55 | 0.887 |
| geography | 8 | 68 | 48 | 0.706 |
| medicine | 8 | 49 | 34 | 0.694 |
<!-- END GENERATED: retrieval -->

Four other implementations were considered and not built:

- Scoring proofs instead of admitting rewrites. This is what NLProlog, neural theorem provers and DeepSoftLog do: aggregate similarity along each proof and rank the answers. Running the rewriter with the cosine judge already measures similarity-only admission, which is where approach B starts. Proof scores would also change unification, scoring and the stopping condition, so they amount to a different system.
- Discovering facts from text for bound subgoals. This would retrieve passages that mention Alice and extract facts from them. It needs a corpus and a fact extractor, which this setting does not have. An open corpus also brings in new constants without bound, which is the one case where tables stop guaranteeing termination.
- Choosing the rewrite set jointly under transitivity. Entailment graphs (Berant et al. 2011; Hosseini et al. 2018) do this with an integer program over the scores of every pair of predicates. It has to see all those scores before it admits any rewrite, so it can't run goal by goal; it belongs with the `saturate` end of the design, not with goal-directed discovery.
- Maintaining the closure bottom-up with semi-naive deltas. The `saturate` strategy measures what materialising everything costs in model calls. Incremental maintenance only pays off when facts keep arriving, which this experiment does not simulate.

## Can a cross-encoder or Jev tell which way an implication runs?

Yes. Every judge that reads the two sentences together can tell which way an implication runs, and Jev does it best. Embedding similarity can't, by construction. Direction accuracy is the share of one-way pairs, such as mother ⇒ parent but not the reverse, where the true direction scores higher:

- Jev scores 0.822 on the Levy/Holt directional test split, 0.895 on Wikidata's property hierarchy and 1.000 on the hand-labelled hard pairs.
- The NLI cross-encoders score 0.770 to 0.834 on Levy/Holt and 0.790 to 0.871 on Wikidata. Of the three, the WANLI-trained DeBERTa-large is best.
- The Qwen3 reranker, asked yes or no, scores 0.694 on Levy/Holt and 0.762 on Wikidata.
- Both cosine judges sit at chance, 0.46 to 0.52.

For mother and parent specifically, Jev scores mother ⇒ parent at 0.99 and parent ⇒ mother at 0.03, and every NLI model is just as clear.

Telling the direction is not the same as admitting only true rules, though. Each judge has to cut its scores at a threshold. The judge thread calibrated those thresholds on Wikidata dev pairs by balanced accuracy. At those cuts, the share of admitted implications that are true on Wikidata's test split is only 0.33 to 0.54 for the judges that read both sentences, and 0.26 to 0.31 for cosine. The named cases show why. Stepmother ⇒ parent and employs ⇔ manages pass for five of the seven judges, and legal guardian ⇒ parent for three. Jev's low cut (0.165) lets through employs ⇔ manages and stepmother ⇒ parent although it scores them only 0.18 to 0.32. These pair-set measurements are the judge thread's (`results/judges.json`).

### The hand-picked cases

Scores on the hand-labelled hard set, with arguments written as A and B. A star means the score reaches that judge's default threshold, so the rewriter would admit the implication.

<!-- BEGIN GENERATED: named-cases -->
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
<!-- END GENERATED: named-cases -->

### On labelled predicate pairs

<!-- BEGIN GENERATED: pairs -->
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
<!-- END GENERATED: pairs -->

### Inside the rewriter

The rewriter turns every admission error into a logical error. A false rewrite such as legal guardian ⇒ parent becomes a rule, and that rule adds false answers to every query that reaches parent. Enough false rewrites connect nearly every predicate to every other, and the closure blows up, as the termination section showed.

In the first table below, rewrite precision is the share of the distinct rewrites a judge admits, over each benchmark's whole catalogue, that the gold program entails; the next four columns sort the false ones by kind. The answer columns are lazy-tabled's over the 58 queries. At the default thresholds, the judges fall into four groups:

- Jev admits rewrites at precision 0.689. Lazy-tabled reaches answer F1 0.783, with precision 0.852 and recall 0.725, and finishes 76% of queries inside the 500k-inference budget, with recall 0.976 on those.
- The NLI cross-encoders admit at precision 0.57 to 0.67, so 36 to 47% of their queries run out of budget, and F1 is 0.47 to 0.65. On the queries that do finish, recall is 0.92 to 0.95.
- The Qwen3 reranker admits at precision 0.416, and its default cut admits so much that 90% of its queries run out of budget, for F1 0.094.
- Cosine similarity can't tell argument order, so its rewrite precision is the lowest, 0.24 to 0.31; 104 and 112 of its false rewrites have the arguments the wrong way round, more than any other judge's.

A depth bound keeps queries inside the budget, at some cost in precision. At Jev's default threshold, depth-tabled finished every query and reached F1 0.839, with precision 0.759 and recall 0.937, against lazy-tabled's 0.783 with precision 0.852 and 24% of its queries out of budget. Untabled, depth-bounded finished 97% of queries at F1 0.742, because it proves each answer along every path and so exhausts its budget sooner.

<!-- BEGIN GENERATED: judges-in-kb -->
| judge | rewrites admitted | rewrite precision | reversed | wrong argument order | trap | unrelated | finished (default threshold) | answer F1 (default threshold) | answer F1 (held-out threshold) |
|---|---|---|---|---|---|---|---|---|---|
| jev | 289 | 0.689 | 9 | 1 | 12 | 68 | 0.759 | 0.783 | 0.892 |
| nli-deberta-base | 326 | 0.567 | 27 | 16 | 22 | 76 | 0.569 | 0.491 | 0.698 |
| nli-deberta-large | 263 | 0.665 | 19 | 2 | 9 | 58 | 0.638 | 0.648 | 0.680 |
| nli-deberta-large-wanli | 329 | 0.611 | 21 | 28 | 13 | 66 | 0.534 | 0.472 | 0.539 |
| reranker-qwen3 | 392 | 0.416 | 34 | 48 | 27 | 120 | 0.103 | 0.094 | 0.585 |
| cosine-bge | 254 | 0.307 | 22 | 104 | 11 | 39 | 0.897 | 0.404 | 0.201 |
| cosine-qwen3 | 360 | 0.239 | 32 | 112 | 18 | 112 | 0.569 | 0.283 | 0.120 |
<!-- END GENERATED: judges-in-kb -->

### Choosing the threshold for the rewriter

The pair-set thresholds maximise balanced accuracy, which favours recall. The rewriter needs precision, because it applies every admitted rewrite everywhere. So I also calibrated each judge's threshold for the rewriter's own objective, answer F1. For each benchmark, the threshold is the one that maximises F1 on the other three, and it is then scored on the held-out benchmark.

With lazy-tabled evaluation:

- Jev picks 0.65 on every fold and reaches F1 0.892, with precision 0.871 and recall 0.914.
- The NLI cross-encoders score most related predicates near 1.0, so their useful thresholds sit between 0.93 and 0.995. They transfer poorly between benchmarks, with held-out F1 of 0.54 to 0.70, though the best single threshold chosen in hindsight reaches 0.725 to 0.854.
- The Qwen3 reranker reaches 0.585 held out and 0.734 in hindsight.
- Similarity alone (cosine) gives held-out F1 0.12 to 0.20, and at most 0.40 in hindsight.

Each strategy was calibrated on its own. Tables with a depth bound of 3 (depth-tabled) raised held-out F1 for six of the seven judges: nli-deberta-large from 0.680 to 0.910, the WANLI model from 0.539 to 0.858, nli-deberta-base from 0.698 to 0.824, the reranker from 0.585 to 0.715, and the two cosine judges from 0.201 and 0.120 to 0.447 and 0.472. Every depth-tabled query finished at its held-out threshold, against 57 to 83% of lazy-tabled's with the NLI and cosine judges, and calibration settled on lower thresholds, admitting more true rewrites, because a false one can spread at most three steps. Jev is the exception. Its depth-tabled calibration picked 0.26 for the work benchmark, the threshold that maximised F1 on the other three, and scored 0.699 there; at 0.65, depth-bounded scored 0.916 on work. Where depth-bounded and depth-tabled used the same threshold and both finished, they scored the same on every benchmark, so the difference between them is cost and finishing, not answers.

<!-- BEGIN GENERATED: sweep -->
| judge | strategy | default threshold | F1 at default | held-out threshold per benchmark | F1 held-out | precision held-out | recall held-out | finished held-out | best threshold in hindsight | F1 in hindsight |
|---|---|---|---|---|---|---|---|---|---|---|
| jev | lazy-tabled | 0.165 | 0.783 | family 0.650, work 0.650, geography 0.650, medicine 0.650 | 0.892 | 0.871 | 0.914 | 1.000 | 0.650 | 0.892 |
| jev | depth-bounded | 0.165 | 0.742 | family 0.650, work 0.650, geography 0.650, medicine 0.650 | 0.896 | 0.969 | 0.833 | 1.000 | 0.650 | 0.896 |
| jev | depth-tabled | 0.165 | 0.839 | family 0.650, work 0.260, geography 0.650, medicine 0.650 | 0.829 | 0.819 | 0.838 | 1.000 | 0.650 | 0.896 |
| nli-deberta-base | lazy-tabled | 0.222 | 0.491 | family 0.992, work 0.981, geography 0.992, medicine 0.981 | 0.698 | 0.827 | 0.604 | 0.793 | 0.981 | 0.725 |
| nli-deberta-base | depth-bounded | 0.222 | 0.650 | family 0.992, work 0.981, geography 0.981, medicine 0.981 | 0.824 | 0.899 | 0.761 | 1.000 | 0.981 | 0.858 |
| nli-deberta-base | depth-tabled | 0.222 | 0.714 | family 0.992, work 0.981, geography 0.981, medicine 0.981 | 0.824 | 0.899 | 0.761 | 1.000 | 0.981 | 0.858 |
| nli-deberta-large | lazy-tabled | 0.659 | 0.648 | family 0.995, work 0.995, geography 0.981, medicine 0.995 | 0.680 | 0.959 | 0.527 | 0.828 | 0.995 | 0.854 |
| nli-deberta-large | depth-bounded | 0.659 | 0.784 | family 0.995, work 0.995, geography 0.981, medicine 0.981 | 0.712 | 0.941 | 0.572 | 0.966 | 0.981 | 0.809 |
| nli-deberta-large | depth-tabled | 0.659 | 0.882 | family 0.981, work 0.981, geography 0.981, medicine 0.981 | 0.910 | 0.906 | 0.914 | 1.000 | 0.981 | 0.910 |
| nli-deberta-large-wanli | lazy-tabled | 0.054 | 0.472 | family 0.991, work 0.991, geography 0.932, medicine 0.991 | 0.539 | 1.000 | 0.369 | 0.828 | 0.991 | 0.753 |
| nli-deberta-large-wanli | depth-bounded | 0.054 | 0.548 | family 0.981, work 0.981, geography 0.932, medicine 0.981 | 0.727 | 0.970 | 0.581 | 0.966 | 0.981 | 0.806 |
| nli-deberta-large-wanli | depth-tabled | 0.054 | 0.658 | family 0.932, work 0.541, geography 0.932, medicine 0.932 | 0.858 | 0.857 | 0.860 | 1.000 | 0.932 | 0.877 |
| reranker-qwen3 | lazy-tabled | 0.808 | 0.094 | family 0.985, work 0.985, geography 0.994, medicine 0.985 | 0.585 | 0.630 | 0.545 | 1.000 | 0.985 | 0.734 |
| reranker-qwen3 | depth-bounded | 0.808 | 0.535 | family 0.985, work 0.985, geography 0.990, medicine 0.985 | 0.704 | 0.812 | 0.622 | 1.000 | 0.985 | 0.736 |
| reranker-qwen3 | depth-tabled | 0.808 | 0.642 | family 0.969, work 0.969, geography 0.990, medicine 0.969 | 0.715 | 0.737 | 0.694 | 1.000 | 0.969 | 0.777 |
| cosine-bge | lazy-tabled | 0.957 | 0.404 | family 0.947, work 0.957, geography 0.965, medicine 0.965 | 0.201 | 0.926 | 0.113 | 0.638 | 0.957 | 0.404 |
| cosine-bge | depth-bounded | 0.957 | 0.458 | family 0.947, work 0.965, geography 0.965, medicine 0.965 | 0.346 | 0.549 | 0.252 | 0.914 | 0.965 | 0.464 |
| cosine-bge | depth-tabled | 0.957 | 0.494 | family 0.937, work 0.930, geography 0.957, medicine 0.957 | 0.447 | 0.377 | 0.549 | 1.000 | 0.937 | 0.518 |
| cosine-qwen3 | lazy-tabled | 0.975 | 0.283 | family 0.974, work 0.987, geography 0.974, medicine 0.974 | 0.120 | 0.536 | 0.068 | 0.569 | 0.974 | 0.283 |
| cosine-qwen3 | depth-bounded | 0.975 | 0.051 | family 0.970, work 0.970, geography 0.970, medicine 0.965 | 0.150 | 0.613 | 0.086 | 0.207 | 0.970 | 0.157 |
| cosine-qwen3 | depth-tabled | 0.975 | 0.505 | family 0.970, work 0.970, geography 0.974, medicine 0.975 | 0.472 | 0.390 | 0.599 | 1.000 | 0.975 | 0.505 |
<!-- END GENERATED: sweep -->

<!-- BEGIN GENERATED: judges-strategies -->
| judge | strategy | finished | precision | recall | F1 | precision (finished) | recall (finished) | proposals/query |
|---|---|---|---|---|---|---|---|---|
| cosine-bge | exact | 1.000 | 1.000 | 0.018 | 0.035 | 1.000 | 0.018 | 0.000 |
| cosine-bge | depth-bounded | 0.931 | 0.543 | 0.396 | 0.458 | 0.543 | 0.427 | 81 |
| cosine-bge | depth-tabled | 1.000 | 0.546 | 0.451 | 0.494 | 0.546 | 0.451 | 81 |
| cosine-bge | lazy-tabled | 0.897 | 0.476 | 0.351 | 0.404 | 0.476 | 0.470 | 101 |
| cosine-bge | eager-worklist | 0.897 | 0.476 | 0.351 | 0.404 | 0.476 | 0.470 | 101 |
| cosine-qwen3 | exact | 1.000 | 1.000 | 0.018 | 0.035 | 1.000 | 0.018 | 0.000 |
| cosine-qwen3 | depth-bounded | 0.224 | 0.500 | 0.027 | 0.051 | 0.500 | 0.171 | 9.090 |
| cosine-qwen3 | depth-tabled | 1.000 | 0.439 | 0.595 | 0.505 | 0.439 | 0.595 | 117 |
| cosine-qwen3 | lazy-tabled | 0.569 | 0.359 | 0.234 | 0.283 | 0.359 | 0.481 | 145 |
| cosine-qwen3 | eager-worklist | 0.569 | 0.359 | 0.234 | 0.283 | 0.359 | 0.481 | 145 |
| jev | exact | 1.000 | 1.000 | 0.018 | 0.035 | 1.000 | 0.018 | 0.000 |
| jev | depth-bounded | 0.966 | 0.720 | 0.766 | 0.742 | 0.720 | 0.924 | 143 |
| jev | depth-tabled | 1.000 | 0.759 | 0.937 | 0.839 | 0.759 | 0.937 | 143 |
| jev | lazy-tabled | 0.759 | 0.852 | 0.725 | 0.783 | 0.852 | 0.976 | 174 |
| jev | eager-worklist | 0.759 | 0.852 | 0.725 | 0.783 | 0.852 | 0.976 | 174 |
| nli-deberta-base | exact | 1.000 | 1.000 | 0.018 | 0.035 | 1.000 | 0.018 | 0.000 |
| nli-deberta-base | depth-bounded | 1.000 | 0.597 | 0.887 | 0.714 | 0.597 | 0.887 | 152 |
| nli-deberta-base | depth-tabled | 1.000 | 0.597 | 0.887 | 0.714 | 0.597 | 0.887 | 152 |
| nli-deberta-base | lazy-tabled | 0.569 | 0.471 | 0.513 | 0.491 | 0.471 | 0.919 | 220 |
| nli-deberta-base | eager-worklist | 0.569 | 0.471 | 0.513 | 0.491 | 0.471 | 0.919 | 220 |
| nli-deberta-large | exact | 1.000 | 1.000 | 0.018 | 0.035 | 1.000 | 0.018 | 0.000 |
| nli-deberta-large | depth-bounded | 0.966 | 0.799 | 0.770 | 0.784 | 0.799 | 0.929 | 142 |
| nli-deberta-large | depth-tabled | 1.000 | 0.829 | 0.941 | 0.882 | 0.829 | 0.941 | 142 |
| nli-deberta-large | lazy-tabled | 0.638 | 0.726 | 0.586 | 0.648 | 0.726 | 0.949 | 171 |
| nli-deberta-large | eager-worklist | 0.638 | 0.726 | 0.586 | 0.648 | 0.726 | 0.949 | 171 |
| nli-deberta-large-wanli | exact | 1.000 | 1.000 | 0.018 | 0.035 | 1.000 | 0.018 | 0.000 |
| nli-deberta-large-wanli | depth-bounded | 0.966 | 0.456 | 0.784 | 0.576 | 0.456 | 0.946 | 151 |
| nli-deberta-large-wanli | depth-tabled | 1.000 | 0.502 | 0.955 | 0.658 | 0.502 | 0.955 | 151 |
| nli-deberta-large-wanli | lazy-tabled | 0.534 | 0.471 | 0.473 | 0.472 | 0.471 | 0.938 | 191 |
| nli-deberta-large-wanli | eager-worklist | 0.569 | 0.455 | 0.527 | 0.488 | 0.455 | 0.944 | 191 |
| reranker-qwen3 | exact | 1.000 | 1.000 | 0.018 | 0.035 | 1.000 | 0.018 | 0.000 |
| reranker-qwen3 | depth-bounded | 0.897 | 0.439 | 0.685 | 0.535 | 0.439 | 0.905 | 179 |
| reranker-qwen3 | depth-tabled | 1.000 | 0.494 | 0.919 | 0.642 | 0.494 | 0.919 | 179 |
| reranker-qwen3 | lazy-tabled | 0.103 | 1.000 | 0.050 | 0.094 | 1.000 | 0.733 | 230 |
| reranker-qwen3 | eager-worklist | 0.103 | 1.000 | 0.050 | 0.094 | 1.000 | 0.733 | 230 |
<!-- END GENERATED: judges-strategies -->

### Asking the judge both ways

A cross-encoder can be asked which way an implication runs, and not only whether it holds. Direction-aware admission scores each proposal together with its converse. It admits Q ⇒ P only when the judge does not score P ⇒ Q higher by more than a slack, and the slack is calibrated per judge on Wikidata dev pairs. A true equivalence such as mum ⇔ mother scores nearly the same both ways and survives. A one-way implication read backwards is dropped: in the parent-of-Tom example, nli-deberta-large scores parent ⇒ mum at 0.955 but mum ⇒ parent at 0.998, further apart than its slack of 0.02.

The cost is one extra judge question per candidate. In return, the false-rewrite rate fell for every judge, and answer F1 rose or held. This rate is the share of the rewrites the rewriter applied while answering the queries that are false, so a rewrite that many queries reach counts many times. The largest gain was the reranker's: its F1 rose from 0.094 to 0.500, because the share of its queries that finished rose from 10% to 67%. The WANLI model came next, from 0.472 to 0.491. Jev's answers did not change, although its false-rewrite rate fell from 0.206 to 0.168, so the rewrites it dropped had added no answers. On the Levy/Holt test pairs, precision rose 3 to 11 points for 1.5 to 6.3 points of recall with the cross-encoders and the reranker, and 6.3 points for 0.8 points of recall with Jev. Traps that the judge reads as entailments in only one direction, such as legal guardian ⇒ parent, are untouched by this check.

<!-- BEGIN GENERATED: direction-policy -->
| judge | policy | slack | Wikidata precision | Wikidata recall | Wikidata balanced accuracy | answer precision | answer recall | answer F1 | false rewrites |
|---|---|---|---|---|---|---|---|---|---|
| jev | plain |  | 0.536 | 0.830 | 0.823 | 0.852 | 0.725 | 0.783 | 0.206 |
| jev | directional | 0.200 | 0.574 | 0.801 | 0.824 | 0.852 | 0.725 | 0.783 | 0.168 |
| nli-deberta-base | plain |  | 0.400 | 0.666 | 0.706 | 0.471 | 0.513 | 0.491 | 0.447 |
| nli-deberta-base | directional | 0.100 | 0.413 | 0.648 | 0.706 | 0.473 | 0.513 | 0.492 | 0.436 |
| nli-deberta-large | plain |  | 0.443 | 0.584 | 0.698 | 0.726 | 0.586 | 0.648 | 0.315 |
| nli-deberta-large | directional | 0.020 | 0.464 | 0.573 | 0.702 | 0.743 | 0.586 | 0.655 | 0.295 |
| nli-deberta-large-wanli | plain |  | 0.422 | 0.753 | 0.745 | 0.471 | 0.473 | 0.472 | 0.389 |
| nli-deberta-large-wanli | directional | 0.050 | 0.470 | 0.711 | 0.753 | 0.459 | 0.527 | 0.491 | 0.295 |
| reranker-qwen3 | plain |  | 0.327 | 0.538 | 0.628 | 1.000 | 0.050 | 0.094 | 0.599 |
| reranker-qwen3 | directional | 0.010 | 0.384 | 0.473 | 0.640 | 0.514 | 0.486 | 0.500 | 0.539 |
<!-- END GENERATED: direction-policy -->

## Is pymetta stable enough?

Yes, for this experiment. Everything the rewriter needs from upstream pymetta 0.9.4 worked, and every defect it hit has a workaround in the engine. These held throughout:

- `(tabled ...)` from `lib_tabling` made cyclic rewrite programs terminate, with SWI-Prolog's SLG resolution doing the work underneath.
- A Python generator op called inside tabled `holds/3` ran once per table, which is what discovery inside SLG needs.
- Adding an equation to a tabled function refreshed its table, which the rounds strategy depends on.
- Every runaway query stopped, by its inference budget or, three times, by its time budget.
- All of this held across the engine tests and the 1,200 property-test catalogues.

It also has eight defects worth reporting upstream. Five of them appear only after earlier evaluations in the same process (the wrapped limit, the lost tabled budget, the missed limit, the symbol-use leak and the count drift), which suggests state that outlives a `MeTTa()` context:

| defect | reproduction |
|---|---|
| A constant spelled like a builtin, such as `max` or `min`, loses its answers in untabled evaluation when the query binds it, while tabled evaluation and a direct lookup find them. The depth-bounded strategy lost parent(hope, max) under every judge until the engine renamed such constants. | `repros/builtin_named_constant_untabled.py` |
| An inference limit reached while a Python op runs can arrive as a plain `EngineError` whose cause is a janus `PrologError`, so `except InferenceLimitError` misses it. The experiment's engine hit it every time on a five-fact family program, but only once its exact strategy had run earlier in the process: the naive run then raised `EngineError("Python InferenceLimitError in (impliers mother)")`. This release's engine reads such an error as a work limit by its own inference count. | found with the experiment's own engine, which this release replaced; replaying its two programs in plain pymetta, or running them through this release's engine, raises `InferenceLimitError`, so no script here reproduces it |
| A budget that interrupts tabled evaluation can be lost. SWI prints `Unknown message: tabling(unexpected_result(..., external_exception(inference_limit_exceeded)))` to stderr and evaluation goes on: with the reranker judge, lazy-tabled `within(parin, Y)` ran to 18,507,105 inferences against its 500,000 budget, in both runs of the experiment (18,507,021 the first time). The bundled SWI's `boot/tabling.pl` has no `external_exception` case in `finished_leader/4`. The engine still reports `work_limit`. | rerun the reranker's judge run (`--tag judges-5`); the row is in `results/implementations.judges-5.jsonl`. A fresh process stops the same query at 501,302. |
| In the 70-minute oracle run, three untabled runs of `superior_of(X, finn)` (naive, and memo-untabled cold and warm) ran to 9.4 to 10.0 million inferences against their 500,000 budget and stopped only at the 120-second time limit. The earlier run of the same experiment showed the same three. Replaying the run's history reproduced it at an earlier commit but not at the final one. | rerun the oracle experiment; the rows are in `results/implementations.oracle.jsonl` |
| `MeTTa.eval(timeout=float("inf"))` raises an `EngineError` from SWI's arithmetic (`float_overflow`), although `timeout=None` means no limit and `timeout=0` is refused with a `ValueError`. The engine passes `None` for an unlimited budget. | `repros/infinite_timeout_overflows.py` |
| Symbol-use state crosses contexts. Evaluating a call to an undefined predicate in one context makes a fresh context warn that the predicate "is defined after already being used". Answers stay correct: the isolation check below ran all 170 family-benchmark oracle runs both in one shared process and in a fresh process each, and found no difference. | `repros/symbol_use_leaks_across_contexts.py` |
| Every call of a function name costs more inferences, in every later context of the process, for each earlier context that tabled a function of that name: an untabled ten-call chain costs 7,264 inferences fresh and 13,979 after 300 such contexts, and abolishing all tables does not undo it. It explains the drift in counts between runs: when the same 51 family runs were repeated with three strategies interleaved, all 17 untabled counts grew, by 5,092 to 11,076 inferences (up to 1.9%); the other two changes, of 2 and 6 inferences, were tabled runs of parent(X, max), which it does not explain. It also starved untabled runs of their budget after the conformance tests until the engine gave each solve's function names a tag of their own. Budgets below about 100,000 inferences also overshoot: at 10,000, both untabled divergence runs stopped after about 100,100. | `repros/tabled_contexts_raise_inference_cost.py` |
| `run()` takes 3 to 63 times longer than `eval()` to exhaust the same inference budget (budgets of 10,000 to 100,000 inferences, over several runs and machine loads), and the gap in seconds grows faster than the budget. | `repros/run_budget_slower_than_eval.py` |

Apart from the tabled overshoot, every inference-limited run in the final results stopped within 10,500 inferences of its 500,000 budget (567 of 568 runs).

Three things caught me out, but they're MeTTa semantics rather than defects:

- A rewrite that returns its own goal is reduced again, forever. Return `True` or a value that can't be reduced.
- A tuple whose head names a defined function is evaluated. `(mother "gloss")` came back as a partial application, so predicate names carried as data need a constructor head that isn't a function.
- Without tables, a rewrite cycle runs in constant stack, so `max-stack-depth` and the stack limit never stop it. Only an inference or time budget does.

<!-- BEGIN GENERATED: isolation -->
| knowledge base | runs compared | mismatches |
|---|---|---|
| family | 187 | 0 |
<!-- END GENERATED: isolation -->

## Reproduce

The files under `results/` come from the experiment code this release grew from, on the PyMeTTa engine, with their field names renamed to this release's (`work` for `inferences`, `finished` for `terminated`, `work_limit` for `inference_limit`). These commands rerun them with this release's code. Its tests check that with a perfect judge every tabled strategy answers every benchmark query exactly on both engines; the judge runs have not been repeated with this code, and engine work can differ, as the pymetta section explains. The Levy/Holt pairs are fetched first, as the README says:

```sh
uv sync
.venv/bin/python -m pytest tests/
.venv/bin/python -m semrewrite.pairs fetch levyholt
.venv/bin/python -m semrewrite.pairs build levyholt
.venv/bin/python experiments/judge_eval.py
.venv/bin/python experiments/impl_eval.py --judges oracle --budget 500000 --tag oracle
for spec in "judges-1 jev" "judges-2 nli-deberta-base" "judges-3 nli-deberta-large" \
            "judges-4 nli-deberta-large-wanli" "judges-5 reranker-qwen3,cosine-bge,cosine-qwen3"; do
  set -- $spec
  .venv/bin/python experiments/impl_eval.py --judges "$2" --tag "$1" \
      --strategies exact,depth-bounded,depth-tabled,lazy-tabled,eager-worklist \
      --modes cold --budget 500000 --skip-extras --sweep lazy-tabled,depth-bounded,depth-tabled
done
.venv/bin/python experiments/direction_policy.py --tag xenc \
    --judges nli-deberta-base,nli-deberta-large,nli-deberta-large-wanli,reranker-qwen3
.venv/bin/python experiments/direction_policy.py --tag jev --judges jev
.venv/bin/python experiments/blowup.py
.venv/bin/python experiments/report.py --complete
.venv/bin/python examples/parent_of_tom.py --judge nli-deberta-large
```

What each step produces:

- `pytest` runs the engine tests, the termination properties and the judge tests.
- `judge_eval.py` writes `results/judges.json` and the calibrated thresholds in `semrewrite/judges/thresholds.json`. Jev's answers are cached on disk, so a rerun sends no requests unless it asks something new, which needs a TypeSafe key; this repository does not include the experiment's cache.
- `impl_eval.py` writes `results/implementations.<tag>.json` and `.jsonl`. The judge runs were split into five processes; one process with every judge in its `--judges` list gives the same answers, though inference counts can differ slightly, as the pymetta section explains.
- `direction_policy.py` writes `results/direction_policy.<tag>.json`, and `blowup.py` writes `results/blowup.json`.
- `report.py` fills the tables in this file; `--complete` makes a missing results file an error.
- `parent_of_tom.py` runs the parent-of-Tom example with any judge.
