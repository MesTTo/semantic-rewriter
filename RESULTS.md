# Semantic rewriter on pymetta: results

This file answers the five questions behind the experiment. The tables between `GENERATED` markers come from the files under `results/`, rendered by `python experiments/report.py`; the commands that regenerate those files are at the end.

## Short answers

Does semantic reduction loop? No, as long as goals are tabled and each predicate is expanded once. Every tabled strategy terminated with exactly the right answers on all 400 generated catalogues, each containing an alias cycle. Untabled rewriting loops exactly when the goal sits on a cycle of rewrites. Tables stop loops, not blow-ups, though: with a permissive judge, the finite closure cost up to 1,556 times the engine work.

Which implementation? Twelve were built, as settings of five knobs. The one to use is goal-directed discovery with tables and a per-predicate memo. Discovering inside SLG evaluation (`lazy-tabled`) is the natural fit; compiling rewrites to native equations (`eager-worklist`) and merging aliases with a union-find (`alias-canonical`) are cheaper in engine work. Each of the three costs about 116 judge questions per query on its own, or 17 once a benchmark's queries share the memo. With an imperfect judge, also bound the depth, and calibrate what the bound counts as you calibrate the threshold. Tabled evaluation bounded at three steps (`depth-tabled`) raised answer F1 at calibrated thresholds for six of the seven judges. Bounding three rewrites alone (`rewrite-depth-tabled`), so that the knowledge base's own rules recurse freely, gave Jev its best F1 and the perfect judge every answer, but scored below the step bound with every other judge.

Can a cross-encoder or Jev tell which way? Yes. On the Levy/Holt and Wikidata pairs, Jev gets 0.82 to 0.89 direction accuracy, the NLI cross-encoders 0.77 to 0.87, and embedding similarity sits at chance. Inside the rewriter, the deciding quantity is admission precision, because every false rewrite becomes a rule applied everywhere. Jev at a threshold calibrated on the other benchmarks reaches answer F1 0.892, and 0.916 with its rewrites bounded at three. On the same basis the best cross-encoder reaches 0.70, and 0.91 once evaluation is tabled and bounded at three steps; similarity alone reaches at most 0.47. Asking each judge both ways and rejecting the weaker direction lowered every cross-encoder's false-rewrite rate.

How sure is each answer? Each answer can carry an opinion: the probability that every rewrite of one of its derivations is right, each right as its judge's opinion says at the rate labelled pairs hold, discounted by how far the judge proved right on them. Jev's trust fits to 0.958. On the Wikidata questions, Jev admitting at 0.1 instead of 0.65 brings in 21 answers outside Wikidata's record for Victoria and one for Alphabet, and every one of them ranks below every answer Wikidata records. The knowledge base's own statements, as a second source on each rewrite, add nothing to Jev, whose trust in them fits to 0, and about 0.01 to the NLI cross-encoders' log loss on labelled pairs once their opinions take the pairs' base rate. They give no evidence on a question asked in its own words, which states none of its predicates.

Is pymetta stable enough? Yes. Tabling, ops called inside tables, table refresh and budgets all did what the rewriter needed. I found twelve defects, seven of which involve state that outlives a `MeTTa()` context; ten have a script under `repros/`, and the engine works around every one.

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
| depth-bounded | lazy | on | off | 3 steps | |
| depth-tabled | lazy | on | on | 3 steps | |
| rewrite-depth-tabled | lazy | on | on | 3 rewrites | |
| lazy-tabled | lazy | on | on | | |
| rounds | evaluate, expand what was reached, repeat to a fixpoint | on | on | | |
| eager-worklist | worklist from the query before evaluation | on | on | | |
| alias-canonical | eager | on | on | | union-find renames each alias class to one predicate |
| saturate | every catalogue predicate before evaluation | on | on | | |

`lazy` runs discovery through a pymetta op (`impliers`) that the tabled `holds/3` clause calls when evaluation first reaches a predicate. That is SLG resolution with semantic goal expansion. `rounds` is the loop the experiment started from: reduce, discover for whatever was reached, add the rewrites, reduce again, stop when a round reaches nothing new. A bound of 3 steps allows at most three rewrite or rule steps on each path below the query; `depth-tabled` keys each table on the depth left as well, so it finds exactly what `depth-bounded` finds without proving anything twice. A bound of 3 rewrites (`rewrite-depth-tabled`) counts rewrite steps alone, so the knowledge base's own rules recurse as deep as their facts allow, while a false rewrite still spreads through at most three rewrites.

## How answers and rewrites are scored

The four benchmarks (family, work, geography and medicine) are generated worlds with 75 predicates, 223 facts, 10 user rules and 58 queries in all. Each states its facts under many surface predicates, such as mother, mum, mom and has_mother. Each also comes with gold implications that I wrote, such as mother ⇒ parent or mother(y, x) ⇒ has_mother(x, y), and with traps that look related but aren't, such as legal guardian ⇒ parent.

A gold answer is one that the facts, the gold implications and the user rules entail. A rewrite such as mother ⇒ parent counts as true when the gold program entails it. The test takes one fact mother(a, b) over two fresh constants, computes everything the gold implications and user rules derive from it, and looks for parent(a, b), or for parent(b, a) when the rewrite swaps the arguments. For Horn rules this one fixpoint decides entailment; it is the standard test for Datalog query containment.

I audited the gold once, against every rewrite the judges admitted, because my first labels missed implications that the glosses force. Of 467 distinct rewrites that the five direction-reading judges admitted and the first gold called false, the ones that three or more judges admitted included mother(y, x) ⇒ has_mother(x, y), parent, father and grandparent ⇒ ancestor, boss_of and skip_level_boss ⇒ superior_of, contains(y, x) ⇒ inside(x, y), capital_of and largest_city_of ⇒ city_in, and the symmetry of near. All of these are true by the glosses' meaning. The first gold also treated geographic containment as direct only, which its English glosses deny, and its truth test read the gold implications but ignored the user rules. The audit added the missing implications, made containment transitive, retyped the work glosses whose second argument is a company, and replaced the old test with the entailment test above.

Every number in this file comes from runs on the audited benchmarks. The audit raised every judge's rewrite precision, by 0.06 to 0.24, and every judge's answer F1 at a threshold calibrated on the other benchmarks; Jev's went from 0.839 to 0.892. `results/before_gold_audit.md` keeps this file as it stood on the first gold.

## When does it stop, and can it loop?

It stops when no new (goal, rewrite) pair and no new answer can appear. With tables it always gets there. Without tables it loops exactly when the goal sits on a cycle of rewrites, and the property tests confirm both statements.

`tests/test_termination.py` generated 400 random catalogues per property and engine (2 to 6 predicates, random gold implications always including an alias cycle, random facts, an optional join rule, a random query) and compared every answer with a Python fixpoint (`results/termination.txt`):

- Every tabled strategy (lazy-tabled, nomemo-tabled, rounds, eager-worklist, alias-canonical, saturate) terminated with exactly the fixpoint's answers on all 400, on both engines.
- Untabled rewriting, which only the PyMeTTa engine runs, never terminated on the 250 of the 400 catalogues where the query predicate sat on a rewrite cycle. On the 98 where nothing reachable was cyclic, it terminated with the right answers. The other 52, with cycles reachable only through a join rule, are left undecided. Hypothesis reports these as 58.00%, 22.74% and 12.06% of 431 draws, 31 of which were invalid.
- Every depth-bounded strategy (depth-bounded, depth-tabled and rewrite-depth-tabled) always terminated on both engines, every answer it gave was true, and tables left the answers under each bound unchanged.

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
| naive | 10,000 | work_limit | 10,824 | 47 | 2,397 | 0 | 0.113 |
| memo-untabled | 10,000 | work_limit | 10,840 | 4 | 204 | 0 | 0.150 |
| lazy-tabled | 10,000 | ok | 4,187 | 5 | 255 | 14 | 0.197 |
| naive | 100,000 | work_limit | 101,298 | 512 | 26,112 | 0 | 0.264 |
| memo-untabled | 100,000 | work_limit | 101,295 | 4 | 204 | 0 | 0.201 |
| lazy-tabled | 100,000 | ok | 4,187 | 5 | 255 | 14 | 0.181 |
| naive | 1,000,000 | work_limit | 1,000,773 | 5,163 | 263,313 | 0 | 1.369 |
| memo-untabled | 1,000,000 | work_limit | 1,000,778 | 4 | 204 | 0 | 0.986 |
| lazy-tabled | 1,000,000 | ok | 4,187 | 5 | 255 | 14 | 0.217 |
| naive | 3,000,000 | work_limit | 3,016,283 | 15,499 | 790,449 | 0 | 7.255 |
| memo-untabled | 3,000,000 | work_limit | 3,016,283 | 4 | 204 | 0 | 4.536 |
| lazy-tabled | 3,000,000 | ok | 4,187 | 5 | 255 | 14 | 0.181 |
<!-- END GENERATED: divergence -->

The query is mother(X, Y) on the family benchmark, where mother, mum and mom are aliases of each other, with a perfect judge.

- Tabled rewriting answers all 14 pairs with 5 expansions and 4,187 inferences, whatever the budget.
- Untabled rewriting spends whatever budget it gets and answers nothing.
- Naive rewriting has no memo, so it also keeps asking the judge, and its judge proposals grow with the budget, to 790,449 at 3M inferences.

Even at the 10k budget, both untabled runs stopped within 840 counted inferences of it, the naive one after 47 expansions.

### Tables stop loops, not blow-ups

Termination is guaranteed, but the guarantee says nothing about cost. When a judge admits rewrites between most pairs of predicates, nearly every predicate reaches nearly every other, and the fixpoint becomes very large. With the budget lifted, every hard query below still terminated. Compared with the perfect judge on the same query and strategy, nli-deberta-base at its default threshold needed 1.6 to 1,556 times the engine work and up to 1,437 times as many tables. Both largest ratios are on brings_about(X, numbness): 4,622,356 inferences against 2,970, and 8,620 tables against 6. The largest closure, for inside(X, palir), took 17.3 million inferences against the perfect judge's 117,926. Most of what the permissive judge returned was false: 3 of its 23 answers to brings_about(X, numbness) were true, and 30 of its 80 answers to parent(X, Y). It found every true answer except one of the three for superior_of(X, finn).

<!-- BEGIN GENERATED: blowup -->
| query | judge | strategy | status | inferences | tables | rewrites | true/answers | gold |
|---|---|---|---|---|---|---|---|---|
| parent(X, Y) | nli-deberta-base | lazy-tabled | ok | 24,261 | 20 | 69 | 30/80 | 30 |
| parent(X, Y) | nli-deberta-base | eager-worklist | ok | 21,364 | 20 | 69 | 30/80 | 30 |
| parent(X, Y) | oracle | lazy-tabled | ok | 6,973 | 11 | 32 | 30/30 | 30 |
| parent(X, Y) | oracle | eager-worklist | ok | 6,294 | 11 | 32 | 30/30 | 30 |
| ancestor(X, leo) | nli-deberta-base | lazy-tabled | ok | 411,754 | 1,712 | 77 | 6/17 | 6 |
| ancestor(X, leo) | nli-deberta-base | eager-worklist | ok | 365,848 | 1,712 | 77 | 6/17 | 6 |
| ancestor(X, leo) | oracle | lazy-tabled | ok | 86,009 | 380 | 37 | 6/6 | 6 |
| ancestor(X, leo) | oracle | eager-worklist | ok | 76,111 | 380 | 37 | 6/6 | 6 |
| superior_of(X, finn) | nli-deberta-base | lazy-tabled | ok | 67,119 | 258 | 42 | 2/3 | 3 |
| superior_of(X, finn) | nli-deberta-base | eager-worklist | ok | 56,980 | 258 | 42 | 2/3 | 3 |
| superior_of(X, finn) | oracle | lazy-tabled | ok | 40,404 | 172 | 20 | 3/3 | 3 |
| superior_of(X, finn) | oracle | eager-worklist | ok | 35,939 | 172 | 20 | 3/3 | 3 |
| inside(X, palir) | nli-deberta-base | lazy-tabled | ok | 17,346,375 | 24,999 | 76 | 17/40 | 17 |
| inside(X, palir) | nli-deberta-base | eager-worklist | ok | 15,273,283 | 24,999 | 76 | 17/40 | 17 |
| inside(X, palir) | oracle | lazy-tabled | ok | 117,926 | 336 | 34 | 17/17 | 17 |
| inside(X, palir) | oracle | eager-worklist | ok | 99,204 | 336 | 34 | 17/17 | 17 |
| brings_about(X, numbness) | nli-deberta-base | lazy-tabled | ok | 4,622,356 | 8,620 | 91 | 3/23 | 3 |
| brings_about(X, numbness) | nli-deberta-base | eager-worklist | ok | 3,861,543 | 8,620 | 91 | 3/23 | 3 |
| brings_about(X, numbness) | oracle | lazy-tabled | ok | 2,970 | 6 | 13 | 3/3 | 3 |
| brings_about(X, numbness) | oracle | eager-worklist | ok | 2,728 | 6 | 13 | 3/3 | 3 |
<!-- END GENERATED: blowup -->

## What implementations are possible, and how do they compare?

With a perfect judge, every tabled strategy gives the same answers. They differ only in cost, and the knobs explain the differences:

- The memo decides how often the model is asked. Each expansion asks the judge 2k + 1 = 17 questions: 8 candidates in two argument orders, plus whether the goal is symmetric. So model calls per query are 17 times the number of predicates expanded. Discovering lazily during evaluation asked 116 questions per query. Without a memo, tables repeat discovery for every goal variant and the count rose to 1,013. Saturating the whole catalogue up front asked 326. Sharing one memo across a benchmark's queries (the warm column) brought goal-directed discovery down to 17 per query.
- Tables decide termination. With tables, all 58 queries finished. Without tables, only 2 finished, capital_of(X, mimun) and capital_of(X, finer) on geography, the two queries whose reachable rewrites are acyclic. Naive rewriting also re-asks the judge on every call, so it posed 72,207 questions per query before its 500k-inference budget ran out.
- A bound on every step stops the search but loses answers. Untabled at 3 steps (depth-bounded), recall fell to 0.784, and two geography containment queries, inside(X, palir) and inside(X, suner), still ran out of the 500k budget within three steps. It cost 2.5 times the engine work of lazy tabling, and it derived its 174 answers 1,868 times, since nothing stops it proving the same answer along different paths. Keying each table on the depth left as well (depth-tabled) finished all 58 queries at recall 0.955, derived each of its 212 answers once and cost 10,196 inferences per query, 0.6 times lazy tabling's. The answers it loses need more than three steps: two ancestor(X, mia) answers through the family benchmark's own ancestor rule, and one encloses(X, rarir) answer through geography's transitive containment.
- A bound on rewrites alone keeps those answers. At 3 rewrites (rewrite-depth-tabled), the knowledge base's own rules recurse freely, and every query got exactly lazy tabling's answers. Keying each table on the rewrites left made it the costlier bound, 28,178 inferences and 152 tables per query, 1.6 times lazy tabling's work.
- The encoding sets the engine cost. Compiling rewrites into native equations (eager-worklist, 15,705 inferences per query) was cheaper than calling discovery through `holds/3` (lazy-tabled, 18,073), although about half of the native count is the reach fallback, one call of a Python op per table, which keeps pymetta's tabling declarations close to linear (see the pymetta section). Merging alias classes with a union-find (alias-canonical) cut the work to 7,000 and the tables from 60 to 30 per query. Rounds, the loop of reducing, discovering and reducing again that the experiment started from, reached the same fixpoint at 35,491 inferences, because every round re-evaluates the query.

Recall is 0.969 rather than 1.0 because of retrieval, not the judge. All the missing answers are on the work benchmark, where recall is 0.877. There, the 8 nearest glosses never include "A reports to B" for boss_of, manager_of, supervises, subordinate_of or superior_of, so the rewrite from reports_to is never proposed, and facts stated with reports_to can't be reached. Retrieval recall per benchmark is in the table below; chains through other predicates recover most of what a single retrieval misses.

All runs in these two tables use the oracle judge, which admits exactly the rewrites the gold program entails. That isolates what each strategy does from what a judge gets wrong.

<!-- BEGIN GENERATED: strategies -->
| strategy | precision | recall | finished | proposals/query cold | proposals/query warm | expansions/query | inferences/query | tables/query |
|---|---|---|---|---|---|---|---|---|
| exact | 1.000 | 0.018 | 1.000 | 0.000 | 0.000 | 0.000 | 1,518 | 1.340 |
| naive | 1.000 | 0.009 | 0.035 | 67,262 | n/a | 3,957 | 489,091 | 0.000 |
| memo-untabled | 1.000 | 0.009 | 0.035 | 60 | 12 | 3.550 | 489,087 | 0.000 |
| nomemo-tabled | 1.000 | 0.969 | 1.000 | 1,013 | n/a | 60 | 18,069 | 60 |
| depth-bounded | 1.000 | 0.784 | 0.966 | 110 | 17 | 6.480 | 44,923 | 0.000 |
| depth-tabled | 1.000 | 0.955 | 1.000 | 110 | 17 | 6.480 | 11,838 | 77 |
| rewrite-depth-tabled | 1.000 | 0.969 | 1.000 | 114 | 17 | 6.720 | 28,178 | 152 |
| lazy-tabled | 1.000 | 0.969 | 1.000 | 116 | 17 | 6.830 | 18,072 | 60 |
| rounds | 1.000 | 0.969 | 1.000 | 116 | 17 | 6.830 | 35,490 | 60 |
| eager-worklist | 1.000 | 0.969 | 1.000 | 116 | 17 | 6.830 | 15,705 | 60 |
| alias-canonical | 1.000 | 0.969 | 1.000 | 116 | 17 | 6.830 | 7,000 | 30 |
| saturate | 1.000 | 0.969 | 1.000 | 326 | 22 | 19 | 15,702 | 60 |
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
Bounding the rewrites alone, rewrite-depth-tabled also finished every query at Jev's default threshold, at F1 0.799 with precision 0.690 and recall 0.951: the rules recursing past three steps found more true answers and more false ones.

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

Bounding the rewrites alone splits the judges. With Jev, rewrite-depth-tabled reached held-out F1 0.916, the best of any judge and strategy here, against 0.892 unbounded and 0.829 with the step bound: Jev admits few false rewrites, so letting the knowledge base's own rules recurse past three steps recovered answers the step bound cut, as it did for the perfect judge, whose answers it matched exactly. With every other judge it scored below the step bound: 0.588 against 0.824 with nli-deberta-base, 0.814 against 0.910 with nli-deberta-large, 0.462 against 0.858 with the WANLI model, 0.704 against 0.715 with the reranker, and 0.355 and 0.304 against 0.447 and 0.472 with the cosine judges. Most of the gap is on the two benchmarks whose rules recurse, geography's transitive containment and family's ancestor rule, where a false rewrite's answers then travel as far as the recursion reaches instead of three steps: on geography, nli-deberta-base scored 0.362 against 0.789, the WANLI model 0.221 against 0.855, and the cosine judges 0.085 and 0.060 against 0.364 and 0.359. What a bound counts is therefore one more setting to calibrate on held-out knowledge bases, beside the threshold.

<!-- BEGIN GENERATED: sweep -->
| judge | strategy | default threshold | F1 at default | held-out threshold per benchmark | F1 held-out | precision held-out | recall held-out | finished held-out | best threshold in hindsight | F1 in hindsight |
|---|---|---|---|---|---|---|---|---|---|---|
| jev | lazy-tabled | 0.165 | 0.783 | family 0.650, work 0.650, geography 0.650, medicine 0.650 | 0.892 | 0.871 | 0.914 | 1.000 | 0.650 | 0.892 |
| jev | depth-bounded | 0.165 | 0.742 | family 0.650, work 0.650, geography 0.650, medicine 0.650 | 0.896 | 0.969 | 0.833 | 1.000 | 0.650 | 0.896 |
| jev | depth-tabled | 0.165 | 0.839 | family 0.650, work 0.260, geography 0.650, medicine 0.650 | 0.829 | 0.819 | 0.838 | 1.000 | 0.650 | 0.896 |
| jev | rewrite-depth-tabled | 0.165 | 0.799 | family 0.650, work 0.650, geography 0.650, medicine 0.650 | 0.916 | 0.919 | 0.914 | 1.000 | 0.650 | 0.916 |
| nli-deberta-base | lazy-tabled | 0.222 | 0.491 | family 0.992, work 0.981, geography 0.992, medicine 0.981 | 0.698 | 0.827 | 0.604 | 0.793 | 0.981 | 0.725 |
| nli-deberta-base | depth-bounded | 0.222 | 0.714 | family 0.992, work 0.981, geography 0.981, medicine 0.981 | 0.824 | 0.899 | 0.761 | 1.000 | 0.981 | 0.858 |
| nli-deberta-base | depth-tabled | 0.222 | 0.714 | family 0.992, work 0.981, geography 0.981, medicine 0.981 | 0.824 | 0.899 | 0.761 | 1.000 | 0.981 | 0.858 |
| nli-deberta-base | rewrite-depth-tabled | 0.222 | 0.490 | family 0.994, work 0.992, geography 0.981, medicine 0.994 | 0.588 | 0.645 | 0.540 | 0.966 | 0.992 | 0.805 |
| nli-deberta-large | lazy-tabled | 0.659 | 0.648 | family 0.995, work 0.995, geography 0.981, medicine 0.995 | 0.680 | 0.959 | 0.527 | 0.828 | 0.995 | 0.854 |
| nli-deberta-large | depth-bounded | 0.659 | 0.784 | family 0.981, work 0.981, geography 0.981, medicine 0.981 | 0.910 | 0.906 | 0.914 | 1.000 | 0.981 | 0.910 |
| nli-deberta-large | depth-tabled | 0.659 | 0.882 | family 0.981, work 0.981, geography 0.981, medicine 0.981 | 0.910 | 0.906 | 0.914 | 1.000 | 0.981 | 0.910 |
| nli-deberta-large | rewrite-depth-tabled | 0.659 | 0.757 | family 0.995, work 0.995, geography 0.981, medicine 0.995 | 0.814 | 0.831 | 0.797 | 1.000 | 0.995 | 0.862 |
| nli-deberta-large-wanli | lazy-tabled | 0.054 | 0.472 | family 0.991, work 0.991, geography 0.932, medicine 0.991 | 0.539 | 1.000 | 0.369 | 0.828 | 0.991 | 0.753 |
| nli-deberta-large-wanli | depth-bounded | 0.054 | 0.548 | family 0.981, work 0.981, geography 0.932, medicine 0.981 | 0.727 | 0.970 | 0.581 | 0.966 | 0.981 | 0.806 |
| nli-deberta-large-wanli | depth-tabled | 0.054 | 0.658 | family 0.932, work 0.541, geography 0.932, medicine 0.932 | 0.858 | 0.857 | 0.860 | 1.000 | 0.932 | 0.877 |
| nli-deberta-large-wanli | rewrite-depth-tabled | 0.054 | 0.400 | family 0.991, work 0.991, geography 0.932, medicine 0.991 | 0.462 | 0.485 | 0.441 | 0.931 | 0.991 | 0.753 |
| reranker-qwen3 | lazy-tabled | 0.808 | 0.094 | family 0.985, work 0.985, geography 0.994, medicine 0.985 | 0.585 | 0.630 | 0.545 | 1.000 | 0.985 | 0.734 |
| reranker-qwen3 | depth-bounded | 0.808 | 0.535 | family 0.969, work 0.969, geography 0.990, medicine 0.969 | 0.715 | 0.737 | 0.694 | 1.000 | 0.969 | 0.777 |
| reranker-qwen3 | depth-tabled | 0.808 | 0.642 | family 0.969, work 0.969, geography 0.990, medicine 0.969 | 0.715 | 0.737 | 0.694 | 1.000 | 0.969 | 0.777 |
| reranker-qwen3 | rewrite-depth-tabled | 0.808 | 0.356 | family 0.985, work 0.985, geography 0.990, medicine 0.985 | 0.704 | 0.777 | 0.644 | 1.000 | 0.985 | 0.799 |
| cosine-bge | lazy-tabled | 0.957 | 0.404 | family 0.947, work 0.957, geography 0.965, medicine 0.965 | 0.201 | 0.926 | 0.113 | 0.638 | 0.957 | 0.404 |
| cosine-bge | depth-bounded | 0.957 | 0.485 | family 0.937, work 0.957, geography 0.957, medicine 0.957 | 0.349 | 0.527 | 0.261 | 0.914 | 0.957 | 0.485 |
| cosine-bge | depth-tabled | 0.957 | 0.494 | family 0.937, work 0.930, geography 0.957, medicine 0.957 | 0.447 | 0.377 | 0.549 | 1.000 | 0.937 | 0.518 |
| cosine-bge | rewrite-depth-tabled | 0.957 | 0.414 | family 0.947, work 0.957, geography 0.965, medicine 0.957 | 0.355 | 0.391 | 0.324 | 0.845 | 0.957 | 0.414 |
| cosine-qwen3 | lazy-tabled | 0.975 | 0.283 | family 0.974, work 0.987, geography 0.974, medicine 0.974 | 0.120 | 0.536 | 0.068 | 0.569 | 0.974 | 0.283 |
| cosine-qwen3 | depth-bounded | 0.975 | 0.482 | family 0.970, work 0.980, geography 0.983, medicine 0.975 | 0.307 | 0.481 | 0.225 | 0.914 | 0.975 | 0.482 |
| cosine-qwen3 | depth-tabled | 0.975 | 0.505 | family 0.970, work 0.970, geography 0.974, medicine 0.975 | 0.472 | 0.390 | 0.599 | 1.000 | 0.975 | 0.505 |
| cosine-qwen3 | rewrite-depth-tabled | 0.975 | 0.365 | family 0.970, work 0.983, geography 0.983, medicine 0.983 | 0.304 | 0.290 | 0.320 | 0.897 | 0.983 | 0.385 |
<!-- END GENERATED: sweep -->

<!-- BEGIN GENERATED: judges-strategies -->
| judge | strategy | finished | precision | recall | F1 | precision (finished) | recall (finished) | proposals/query |
|---|---|---|---|---|---|---|---|---|
| cosine-bge | exact | 1.000 | 1.000 | 0.018 | 0.035 | 1.000 | 0.018 | 0.000 |
| cosine-bge | depth-bounded | 0.966 | 0.552 | 0.432 | 0.485 | 0.552 | 0.440 | 81 |
| cosine-bge | depth-tabled | 1.000 | 0.546 | 0.451 | 0.494 | 0.546 | 0.451 | 81 |
| cosine-bge | rewrite-depth-tabled | 0.914 | 0.494 | 0.356 | 0.414 | 0.494 | 0.468 | 86 |
| cosine-bge | lazy-tabled | 0.897 | 0.476 | 0.351 | 0.404 | 0.476 | 0.470 | 101 |
| cosine-bge | eager-worklist | 0.897 | 0.476 | 0.351 | 0.404 | 0.476 | 0.470 | 101 |
| cosine-qwen3 | exact | 1.000 | 1.000 | 0.018 | 0.035 | 1.000 | 0.018 | 0.000 |
| cosine-qwen3 | depth-bounded | 0.931 | 0.440 | 0.531 | 0.482 | 0.440 | 0.573 | 117 |
| cosine-qwen3 | depth-tabled | 1.000 | 0.439 | 0.595 | 0.505 | 0.439 | 0.595 | 117 |
| cosine-qwen3 | rewrite-depth-tabled | 0.793 | 0.309 | 0.446 | 0.365 | 0.309 | 0.635 | 118 |
| cosine-qwen3 | lazy-tabled | 0.569 | 0.359 | 0.234 | 0.283 | 0.359 | 0.481 | 145 |
| cosine-qwen3 | eager-worklist | 0.569 | 0.359 | 0.234 | 0.283 | 0.359 | 0.481 | 145 |
| jev | exact | 1.000 | 1.000 | 0.018 | 0.035 | 1.000 | 0.018 | 0.000 |
| jev | depth-bounded | 0.966 | 0.720 | 0.766 | 0.742 | 0.720 | 0.924 | 143 |
| jev | depth-tabled | 1.000 | 0.759 | 0.937 | 0.839 | 0.759 | 0.937 | 143 |
| jev | rewrite-depth-tabled | 1.000 | 0.690 | 0.951 | 0.799 | 0.690 | 0.951 | 152 |
| jev | lazy-tabled | 0.759 | 0.852 | 0.725 | 0.783 | 0.852 | 0.976 | 174 |
| jev | eager-worklist | 0.759 | 0.852 | 0.725 | 0.783 | 0.852 | 0.976 | 174 |
| nli-deberta-base | exact | 1.000 | 1.000 | 0.018 | 0.035 | 1.000 | 0.018 | 0.000 |
| nli-deberta-base | depth-bounded | 1.000 | 0.597 | 0.887 | 0.714 | 0.597 | 0.887 | 152 |
| nli-deberta-base | depth-tabled | 1.000 | 0.597 | 0.887 | 0.714 | 0.597 | 0.887 | 152 |
| nli-deberta-base | rewrite-depth-tabled | 0.931 | 0.363 | 0.757 | 0.490 | 0.363 | 0.944 | 163 |
| nli-deberta-base | lazy-tabled | 0.569 | 0.471 | 0.513 | 0.491 | 0.471 | 0.919 | 220 |
| nli-deberta-base | eager-worklist | 0.569 | 0.471 | 0.513 | 0.491 | 0.471 | 0.919 | 220 |
| nli-deberta-large | exact | 1.000 | 1.000 | 0.018 | 0.035 | 1.000 | 0.018 | 0.000 |
| nli-deberta-large | depth-bounded | 0.966 | 0.799 | 0.770 | 0.784 | 0.799 | 0.929 | 142 |
| nli-deberta-large | depth-tabled | 1.000 | 0.829 | 0.941 | 0.882 | 0.829 | 0.941 | 142 |
| nli-deberta-large | rewrite-depth-tabled | 1.000 | 0.621 | 0.969 | 0.757 | 0.621 | 0.969 | 146 |
| nli-deberta-large | lazy-tabled | 0.638 | 0.726 | 0.586 | 0.648 | 0.726 | 0.949 | 171 |
| nli-deberta-large | eager-worklist | 0.638 | 0.726 | 0.586 | 0.648 | 0.726 | 0.949 | 171 |
| nli-deberta-large-wanli | exact | 1.000 | 1.000 | 0.018 | 0.035 | 1.000 | 0.018 | 0.000 |
| nli-deberta-large-wanli | depth-bounded | 0.897 | 0.442 | 0.721 | 0.548 | 0.442 | 0.952 | 151 |
| nli-deberta-large-wanli | depth-tabled | 1.000 | 0.502 | 0.955 | 0.658 | 0.502 | 0.955 | 151 |
| nli-deberta-large-wanli | rewrite-depth-tabled | 0.862 | 0.280 | 0.698 | 0.400 | 0.280 | 0.957 | 152 |
| nli-deberta-large-wanli | lazy-tabled | 0.534 | 0.471 | 0.473 | 0.472 | 0.471 | 0.938 | 191 |
| nli-deberta-large-wanli | eager-worklist | 0.534 | 0.471 | 0.473 | 0.472 | 0.471 | 0.938 | 191 |
| reranker-qwen3 | exact | 1.000 | 1.000 | 0.018 | 0.035 | 1.000 | 0.018 | 0.000 |
| reranker-qwen3 | depth-bounded | 0.897 | 0.439 | 0.685 | 0.535 | 0.439 | 0.905 | 179 |
| reranker-qwen3 | depth-tabled | 1.000 | 0.494 | 0.919 | 0.642 | 0.494 | 0.919 | 179 |
| reranker-qwen3 | rewrite-depth-tabled | 0.793 | 0.248 | 0.635 | 0.356 | 0.248 | 0.922 | 178 |
| reranker-qwen3 | lazy-tabled | 0.103 | 1.000 | 0.050 | 0.094 | 1.000 | 0.733 | 230 |
| reranker-qwen3 | eager-worklist | 0.103 | 1.000 | 0.050 | 0.094 | 1.000 | 0.733 | 230 |
<!-- END GENERATED: judges-strategies -->

### Asking the judge both ways

A cross-encoder can be asked which way an implication runs, and not only whether it holds. Direction-aware admission scores each proposal together with its converse. It admits Q ⇒ P only when the judge does not score P ⇒ Q higher by more than a slack, and the slack is calibrated per judge on Wikidata dev pairs. A true equivalence such as mum ⇔ mother scores nearly the same both ways and survives. A one-way implication read backwards is dropped: in the parent-of-Tom example, nli-deberta-large scores parent ⇒ mum at 0.955 but mum ⇒ parent at 0.998, further apart than its slack of 0.02.

The cost is one extra judge question per candidate. In return, the false-rewrite rate fell for every judge, and answer F1 rose or held. This rate is the share of the rewrites the rewriter applied while answering the queries that are false, so a rewrite that many queries reach counts many times. The largest gain was the reranker's: its F1 rose from 0.094 to 0.500, because the share of its queries that finished rose from 10% to 67%. The WANLI model came next, from 0.472 to 0.491. Jev's answers did not change, although its false-rewrite rate fell from 0.206 to 0.172, so the rewrites it dropped had added no answers. On the Levy/Holt test pairs, precision rose 3 to 11 points for 1.5 to 6.3 points of recall with the cross-encoders and the reranker, and 6.3 points for 0.8 points of recall with Jev. Traps that the judge reads as entailments in only one direction, such as legal guardian ⇒ parent, are untouched by this check.

<!-- BEGIN GENERATED: direction-policy -->
| judge | policy | slack | Wikidata precision | Wikidata recall | Wikidata balanced accuracy | answer precision | answer recall | answer F1 | false rewrites |
|---|---|---|---|---|---|---|---|---|---|
| jev | plain |  | 0.536 | 0.830 | 0.823 | 0.852 | 0.725 | 0.783 | 0.206 |
| jev | directional | 0.200 | 0.574 | 0.801 | 0.824 | 0.852 | 0.725 | 0.783 | 0.172 |
| nli-deberta-base | plain |  | 0.400 | 0.666 | 0.706 | 0.471 | 0.513 | 0.491 | 0.447 |
| nli-deberta-base | directional | 0.100 | 0.413 | 0.648 | 0.706 | 0.473 | 0.513 | 0.492 | 0.436 |
| nli-deberta-large | plain |  | 0.443 | 0.584 | 0.698 | 0.726 | 0.586 | 0.648 | 0.317 |
| nli-deberta-large | directional | 0.020 | 0.464 | 0.573 | 0.702 | 0.743 | 0.586 | 0.655 | 0.298 |
| nli-deberta-large-wanli | plain |  | 0.422 | 0.753 | 0.745 | 0.471 | 0.473 | 0.472 | 0.390 |
| nli-deberta-large-wanli | directional | 0.050 | 0.470 | 0.711 | 0.753 | 0.459 | 0.527 | 0.491 | 0.298 |
| reranker-qwen3 | plain |  | 0.327 | 0.538 | 0.628 | 1.000 | 0.050 | 0.094 | 0.601 |
| reranker-qwen3 | directional | 0.010 | 0.384 | 0.473 | 0.640 | 0.514 | 0.486 | 0.500 | 0.539 |
<!-- END GENERATED: direction-policy -->

## How sure is each answer?

With `--beliefs`, every answer comes with an opinion that it holds: a belief, a disbelief and an uncertainty, and the probability P they project. DESIGN.md says how they are computed. Two measurements decide what the opinions are worth: how well each judge's opinions are calibrated, and whether the answers' probabilities sort Wikidata's own answer from the rest.

### How far each judge is trusted

A judge's opinions are taken at the base rate of the pairs it is asked about, the share of them that hold, which is what an opinion projects to where it says nothing. Its trust is fitted on the Wikidata dev pairs, as its threshold is: it is the trust whose discounted opinions give the labels the highest likelihood. The table gives each judge's trust, the test pairs' ROC AUC, and the log loss and Brier score of its calibrated opinions on them. A judge that answered every pair with the base rate would score 0.505 and 0.162.

<!-- BEGIN GENERATED: trust -->
Every judge's opinions are taken at the base rate 0.214, the share of the Wikidata dev pairs that hold.

| judge | trust | test ROC AUC | test log loss | test Brier | data trust | test pairs with data | their log loss, judge alone | fused with the data |
|---|---|---|---|---|---|---|---|---|
| jev | 0.958 | 0.878 | 0.321 | 0.097 | 0.000 | 933 | 0.365 | 0.365 |
| nli-deberta-large-wanli | 0.524 | 0.802 | 0.432 | 0.134 | 0.421 | 933 | 0.465 | 0.453 |
| nli-deberta-large | 0.348 | 0.754 | 0.454 | 0.143 | 0.407 | 933 | 0.488 | 0.481 |
| nli-deberta-base | 0.311 | 0.744 | 0.464 | 0.147 | 0.364 | 933 | 0.492 | 0.482 |
| reranker-qwen3 | 0.078 | 0.668 | 0.499 | 0.160 | 0.263 | 933 | 0.539 | 0.535 |
| cosine-bge | 0.002 | 0.531 | 0.505 | 0.162 | 0.171 | 933 | 0.553 | 0.551 |
| cosine-qwen3 | 0.001 | 0.532 | 0.505 | 0.162 | 0.169 | 933 | 0.553 | 0.551 |
<!-- END GENERATED: trust -->

Jev is trusted almost completely, at 0.958, and its opinions reach a test log loss of 0.321. The NLI cross-encoders are trusted at 0.31 to 0.52. Their neutral class says nothing about a pair, so it projects to the base rate, and their test log loss is 0.43 to 0.46; projected to one half instead, as before the base rate was calibrated, it was 0.62 to 0.66, worse than answering every pair with the base rate. The cosine judges get no trust, since they rank the test pairs at chance, and score exactly what the base rate scores. The reranker ranks them better, at an AUC of 0.668, but its probabilities run high: two thirds of the Wikidata test pairs it scores at 0.81 or more are false, so it is trusted at 0.08. Trust only shrinks a probability toward the base rate; it cannot recalibrate it.

The last three columns test the knowledge base as a second source, on the 933 test pairs whose two properties both have statements in the Wikidata examples' knowledge bases. Its evidence on a pair counts the premise's statements the conclusion also states as for it, and the rest as against it. Its trust, fitted jointly with each judge, is 0 for Jev. With the NLI cross-encoders it is 0.36 to 0.42, and fusing the statements in lowers their log loss on those pairs by about 0.01, to between 0.45 and 0.48. Before the base rate was calibrated the same fusion lowered it from about 0.64 to 0.48: most of what the statements added then was the correction the base rate now makes.

### Answer beliefs on the Wikidata questions

Each question is answered with one rewrite on a path, by Jev at the rewrite threshold calibrated for answer F1, 0.65, and at 0.1, and by nli-deberta-large at its calibrated threshold, 0.659. Every answer is ranked by P and scored against Wikidata's own answer: the average precision of the ranking, the best F1 an answer threshold reaches and the P it sits at, and the lowest P inside Wikidata's answer beside the highest outside it.

<!-- BEGIN GENERATED: wikidata-beliefs -->
| question | judge | rewrite threshold | answers | not in Wikidata's | F1 of all answers | average precision by P | best F1 by P (at P) | lowest P inside | highest P outside |
|---|---|---|---|---|---|---|---|---|---|
| victoria | jev | 0.65 | 1,164 | 0 | 1.000 | 1.000 | 1.000 (0.938) | 0.938 | none |
| victoria | jev | 0.1 | 1,185 | 21 | 0.991 | 1.000 | 1.000 (0.938) | 0.938 | 0.324 |
| victoria | nli-deberta-large | 0.659 | 1,188 | 24 | 0.990 | 1.000 | 0.999 (0.358) | 0.358 | 0.358 |
| leonardo | jev | 0.65 | 851 | 2 | 0.996 | 0.994 | 0.997 (0.689) | 0.689 | 0.679 |
| leonardo | jev | 0.1 | 858 | 5 | 0.996 | 0.999 | 0.997 (0.689) | 0.239 | 0.679 |
| leonardo | nli-deberta-large | 0.659 | 858 | 5 | 0.996 | 0.999 | 0.996 (0.486) | 0.486 | 0.487 |
| alphabet | jev | 0.65 | 2,849 | 0 | 0.938 | 0.883 | 0.938 (0.916) | 0.916 | none |
| alphabet | jev | 0.1 | 2,903 | 1 | 0.947 | 0.900 | 0.947 (0.210) | 0.210 | 0.100 |
| alphabet | nli-deberta-large | 0.659 | 2,906 | 4 | 0.947 | 0.899 | 0.947 (0.237) | 0.237 | 0.357 |
<!-- END GENERATED: wikidata-beliefs -->

Lowering Jev's rewrite threshold to 0.1 admits six times as many rewrites into Victoria's descendants, 110 against 18, and brings in 21 answers outside Wikidata's record. All 21 rank below all 1,164 of Wikidata's, at P 0.324 or less against 0.938 or more, so an answer threshold between them keeps exactly Wikidata's answer. On Alphabet the lower threshold finds 53 more of Wikidata's holdings, and its one answer outside the record, at P 0.100, ranks below all of them, so F1 rises from 0.938 to 0.947. On Leonardo it finds four more of Wikidata's works and three more answers outside it. There P does not separate the two sets: the highest answer outside, at 0.679, ranks above the lowest of Wikidata's, at 0.239, though the best answer threshold, 0.689, still reaches F1 0.997.

The knowledge base gives evidence on a rewrite only where it states the rewrite's conclusion, the setting each data trust was fitted in. None of the three questions states any of its own predicates, so every opinion here is its judge's alone. A first version also counted what the other admitted rewrites derive, reading a row they missed as evidence against a rewrite. With nli-deberta-large that separated Victoria's answers completely, but it lowered Leonardo's average precision from 0.999 to 0.983: of the 858 works `creator` states, the other admitted rewrites derive about 40, so 818 counted against it, and its P fell nearly as low as a rewrite through the material a work is made from. The other rewrites cover only part of a relation, so a row they miss is no evidence against a rewrite.

### SNARS's ranking beside the rewriter's

`semrewrite rewrites --snars` puts the admitted rewrites into the world of mettabase's SNARS, where GoalChainer reasons. To see whether SNARS ranks the answers as the rewriter does, `experiments/snars_ranking.py` gives SNARS the statements those rewrites and the question's own rules read and has it prove and fuse every answer itself, Jev admitting at 0.1 as above.

<!-- BEGIN GENERATED: snars-ranking -->
| question | SNARS depth | answers both rank | with several proofs | average precision over them, rewriter | and SNARS | Kendall's tau-b | P, rewriter | P, SNARS | answers outside Wikidata's: rank by each |
|---|---|---|---|---|---|---|---|---|---|
| leonardo | 1 | 858 of 858 | 50 | 1.000 | 1.000 | 1.000 | 0.239 to 0.978 | 0.566 to 0.923 | annunciation_q154326: 850 and 850; copy_of_the_leonardo_da_vinci_s_last_supper: 851 and 851; leonardo_da_vinci_the_mind_of_the_renaissance_q101731196: 853 and 853; the_child_in_time_q1747269: 854 and 854; the_da_vinci_code_q124474978: 855 and 855 |
| alphabet | 3 | 2,894 of 2,903 | 2,602 | 1.000 | 1.000 | 0.028 | 0.100 to 0.998 | 0.631 to 0.961 | google_earth_vr: 2,894 and 2,827 |
<!-- END GENERATED: snars-ranking -->

The two agree exactly where no derivation uses a rewrite twice. Every answer to Leonardo's question comes through one rewrite, and the two rankings order every pair of answers alike: SNARS's P is a monotone function of the rewriter's, its five answers outside Wikidata's at the same five ranks.

They part where a derivation chains rewrites. Alphabet's question owns what Alphabet's holdings own, through its transitive rule, so most answers need the same rewrite at several steps. The rewriter counts a rewrite once however often a derivation uses it, so an answer's P depends on which rewrites it needs and not on how long its chain is. SNARS applies the rewrite, and its structural prior, at every step: the median P of answers whose shortest proof takes 1, 3, 5 and 7 rule applications falls from 0.928 to 0.867, 0.833 and 0.801, and the rankings barely agree. The rewriter ranks Alphabet's one answer outside Wikidata's last; SNARS puts 67 of Wikidata's answers below it.

## Is pymetta stable enough?

Yes, for this experiment. Everything the rewriter needs from upstream pymetta 0.9.4 worked, and every defect it hit has a workaround in the engine. These held throughout:

- `(tabled ...)` from `lib_tabling` made cyclic rewrite programs terminate, with SWI-Prolog's SLG resolution doing the work underneath.
- A Python generator op called inside tabled `holds/3` ran once per table, which is what discovery inside SLG needs.
- Adding an equation to a tabled function refreshed its table, which the rounds strategy depends on.
- Every runaway query stopped by its inference budget; none of the 3,712 runs reached its time limit.
- All of this held across the engine tests and the 1,200 property-test catalogues.

It also has eleven defects worth reporting upstream. Seven of them involve state that outlives a `MeTTa()` context: the wrapped limit, the lost tabled budget, the missed limit, the symbol-use leak, the count drift, the tabling-import leak and the table declared before its equations. Five of the rows link the issue filed for them on PyMeTTa's tracker:

| defect | reproduction |
|---|---|
| A constant spelled like a builtin, such as `max` or `min`, loses its answers in untabled evaluation when the query binds it, while tabled evaluation and a direct lookup find them. The depth-bounded strategy lost parent(hope, max) under every judge until the engine renamed such constants. | `repros/builtin_named_constant_untabled.py`, filed as [PyMeTTa#1](https://github.com/MesTTo/PyMeTTa/issues/1) |
| An op whose parameter is annotated `object` changes how its inference limit arrives: from the second context of a process on, a limit reached in a loop through the op comes back as a plain `EngineError` whose cause is a janus `PrologError`, `EngineError("Python InferenceLimitError in (next n)")`, so `except InferenceLimitError` misses it. The annotation also more than doubles what a call costs, 77 inferences against 33 unannotated, and with the rewriter's discovery op it let the first naive solve of a process run mother(X, bea) to 16 to 23 million inferences in 20 seconds against a 300,000 budget, stopping only at the time limit; that is why the isolation check's fresh processes once disagreed with the shared one. The experiment's engine annotated its ops that way and hit the wrapped error on its naive runs once another context had run. This release's engine annotates the discovery op's parameter `Symbol`, which stops at the budget and was the cheapest of the annotations measured, and still reads a wrapped limit as a work limit by its own inference count. | `repros/op_object_parameter_wraps_limit.py`, filed as [PyMeTTa#7](https://github.com/MesTTo/PyMeTTa/issues/7) |
| A budget can be lost inside tabled evaluation. SWI sometimes prints `Unknown message: tabling(unexpected_result(..., external_exception(inference_limit_exceeded)))` to stderr, a case the bundled SWI's `boot/tabling.pl` does not handle in `finished_leader/4`, and evaluation goes on. In the experiment, the reranker's lazy-tabled `within(parin, Y)` ran to 18,507,105 inferences against its 500,000 budget in both of its runs. In this release's runs SWI printed the message 10 times, and every run but two stopped within 15,000 inferences of its budget. Both are eager-worklist runs that printed no message: nli-deberta-base's `remedy_for(X, Y)` reached 3,828,885, while the same program in a fresh process stops at 501,187, and nli-deberta-large's `inside(X, suner)` reached 6,599,433, as it had two reruns before, while the rerun between them stopped it at 500,922. Which runs overshoot depends on what ran before them in the process. The engine still reports `work_limit`. | the overshooting runs' rows are in `results/implementations.judges-2.jsonl` and `results/implementations.judges-3.jsonl`; no script here reproduces either alone |
| In the experiment's 70-minute oracle run, three untabled runs of `superior_of(X, finn)` (naive, and memo-untabled cold and warm) ran to 9.4 to 10.0 million inferences against their 500,000 budget and stopped only at the 120-second time limit, in both of its runs; replaying the run's history reproduced it at an earlier commit but not at the final one. That engine annotated its discovery op's parameter `object`, under which a limit can be missed (two rows up). In this release's oracle run, which took 12 minutes, the same three stopped at the inference budget, at 505,830, within 2 seconds, and no run of the 1,276 reached its time limit. | not reproduced with this release's engine; the experiment's rows are in this repository's history |
| `MeTTa.eval(timeout=float("inf"))` raises an `EngineError` from SWI's arithmetic (`float_overflow`), although `timeout=None` means no limit and `timeout=0` is refused with a `ValueError`. The engine passes `None` for an unlimited budget. | `repros/infinite_timeout_overflows.py`, filed as [PyMeTTa#2](https://github.com/MesTTo/PyMeTTa/issues/2) |
| Symbol-use state crosses contexts. Evaluating a call to an undefined predicate in one context makes a fresh context warn that the predicate "is defined after already being used". Answers stay correct: the isolation check below ran all 204 family-benchmark oracle runs both in one shared process and in a fresh process each, and found no difference in answers or status. | `repros/symbol_use_leaks_across_contexts.py`, filed as [PyMeTTa#3](https://github.com/MesTTo/PyMeTTa/issues/3) |
| Every call of a function name costs more inferences, in every later context of the process, for each earlier context that tabled a function of that name: an untabled ten-call chain costs 7,264 inferences fresh and 13,979 after 300 such contexts, and abolishing all tables does not undo it. It explains the drift in counts between runs: when the same 51 family runs were repeated with three strategies interleaved, all 17 untabled counts grew, by 5,092 to 11,076 inferences (up to 1.9%); the other two changes, of 2 and 6 inferences, were tabled runs of parent(X, max), which it does not explain. It also starved untabled runs of their budget after the conformance tests until the engine gave each solve's function names a tag of their own. The experiment's engine tabled one function name in every context, and its runs fit this cost: with the three permissive judges in one process, untabled depth-bounded runs stopped at the budget 6, 4 and 45 times in the order the judges ran, and the third judge's 17 family runs all stopped there, while a fresh process of the same engine stopped 4 of them. | `repros/tabled_contexts_raise_inference_cost.py`, filed as [PyMeTTa#4](https://github.com/MesTTo/PyMeTTa/issues/4) |
| `run()` takes 3 to 63 times longer than `eval()` to exhaust the same inference budget (budgets of 10,000 to 100,000 inferences, over several runs and machine loads), and the gap in seconds grows faster than the budget. | `repros/run_budget_slower_than_eval.py`, filed as [PyMeTTa#5](https://github.com/MesTTo/PyMeTTa/issues/5) |
| Every `MeTTa()` context that imports `lib_tabling` keeps about 25 KB after it closes, whether or not it tables anything and whatever its names, and `abolish_all_tables` does not free it; contexts without the import stay flat. The three permissive judges in one process outgrew a 4 GiB memory cap through it, so each judge run now takes a process of its own. | `repros/tabling_import_leaks_memory.py`, filed as [PyMeTTa#6](https://github.com/MesTTo/PyMeTTa/issues/6) |
| Declaring n tabled functions that call each other takes time quadratic in n, 2.5 s for 240, and close to linear, 0.6 s, when each function also has a clause calling a Python op. The native encoding keeps such a clause, its reach fallback, on every predicate; without it, a solve on the geography benchmark took three times as long. | `repros/tabled_declarations_quadratic.py`, filed as [PyMeTTa#8](https://github.com/MesTTo/PyMeTTa/issues/8) |
| Declaring a function tabled before its equations are loaded makes its query raise an internal type error (``Type error: `metta_function_call' expected``), and every later context that tables a function of that name raises the same error, even with its equations loaded first. The engine declares each table once its function has equations. | `repros/tabled_before_equations_breaks_eval.py`, filed as [PyMeTTa#9](https://github.com/MesTTo/PyMeTTa/issues/9) |
| Closing a `MeTTa()` context takes time growing faster than quadratically in the ground equations one function has: 0.5 to 0.6 s for 1,250, 11 to 19 s for 5,000 and 106 s for 10,000, against under 0.1 s for the same facts as atoms, and `Space.clear()` first takes as long. The meta encoding held every fact as an equation of one function, so a solve over 10,000 facts spent 75 of its 77 s closing its context; it holds them as atoms now, and a solve over the 23,116 statements of the Victoria example in examples/README.md takes about 3 s. | `repros/equation_teardown_superquadratic.py`, filed as [PyMeTTa#10](https://github.com/MesTTo/PyMeTTa/issues/10) |

Apart from two tabled overshoots, every inference-limited run in the final results stopped within 15,000 inferences of its 500,000 budget (573 of 575 runs).

Three things caught me out, but they're MeTTa semantics rather than defects:

- A rewrite that returns its own goal is reduced again, forever. Return `True` or a value that can't be reduced.
- A tuple whose head names a defined function is evaluated. `(mother "gloss")` came back as a partial application, so predicate names carried as data need a constructor head that isn't a function.
- Without tables, a rewrite cycle runs in constant stack, so `max-stack-depth` and the stack limit never stop it. Only an inference or time budget does.

<!-- BEGIN GENERATED: isolation -->
| knowledge base | runs compared | mismatches |
|---|---|---|
| family | 204 | 0 |
<!-- END GENERATED: isolation -->

## Reproduce

Every file under `results/` comes from this release's code on the PyMeTTa engine, and these commands regenerate them. The Levy/Holt pairs are fetched first, as the README says:

```sh
uv sync
.venv/bin/python -m pytest tests/
.venv/bin/python -m pytest tests/test_termination.py --hypothesis-show-statistics -q > results/termination.txt
.venv/bin/python -m semrewrite.pairs fetch levyholt
.venv/bin/python -m semrewrite.pairs build levyholt
.venv/bin/python experiments/judge_eval.py
.venv/bin/python examples/wikidata/evaluate.py --beliefs
.venv/bin/python experiments/impl_eval.py --judges oracle --budget 500000 --tag oracle
for spec in "judges-1 jev" "judges-2 nli-deberta-base" "judges-3 nli-deberta-large" "judges-4 nli-deberta-large-wanli" \
            "judges-5 reranker-qwen3" "judges-6 cosine-bge" "judges-7 cosine-qwen3"; do
  set -- $spec
  .venv/bin/python experiments/impl_eval.py --judges "$2" --tag "$1" \
      --strategies exact,depth-bounded,depth-tabled,rewrite-depth-tabled,lazy-tabled,eager-worklist \
      --modes cold --budget 500000 --skip-extras --sweep lazy-tabled,depth-bounded,depth-tabled,rewrite-depth-tabled
done
.venv/bin/python experiments/direction_policy.py --tag xenc \
    --judges nli-deberta-base,nli-deberta-large,nli-deberta-large-wanli,reranker-qwen3
.venv/bin/python experiments/direction_policy.py --tag jev --judges jev
.venv/bin/python experiments/blowup.py
.venv/bin/python experiments/report.py --complete
.venv/bin/python examples/parent_of_tom.py --judge nli-deberta-large
```

What each step produces:

- `pytest` runs the engine tests, the termination properties and the judge tests. The second `pytest` line writes the termination statistics this file quotes; Hypothesis draws new catalogues on each run, so the shares it reports move a little from run to run.
- `judge_eval.py` writes `results/judges.json` and the calibrated thresholds and trusts in `semrewrite/judges/calibration.json`. Jev's answers are cached on disk, so a rerun sends no requests unless it asks something new, which needs a TypeSafe key; this repository does not include the experiment's cache.
- `evaluate.py --beliefs` answers the three Wikidata questions with Jev admitting at two rewrite thresholds and nli-deberta-large at its own, gives every answer its opinion, and writes how well they rank Wikidata's answer to `results/wikidata_beliefs.json`. It loads nli-deberta-large for the retrieval cascade and reads Jev's answer cache.
- `impl_eval.py` writes `results/implementations.<tag>.json` and `.jsonl`. Each judge run takes a process of its own: pymetta keeps about 25 KB for every context that imports `lib_tabling` (see the pymetta section), and three judges in one process outgrew 4 GiB.
- `direction_policy.py` writes `results/direction_policy.<tag>.json`, and `blowup.py` writes `results/blowup.json`.
- `report.py` fills the tables in this file; `--complete` makes a missing results file an error.
- `parent_of_tom.py` runs the parent-of-Tom example with any judge.
