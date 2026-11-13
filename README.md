# Semantic rewriter

A logic program answers `owns(alphabet_inc, X)` only from clauses whose head says `owns`. Wikidata never says "owns". It records that YouTube is owned by Google and that XXVI Holdings is owner of Google, in two of the 1,790 properties it uses to link one entity to another. The semantic rewriter closes that gap without changing how the program is evaluated. When evaluation first reaches a predicate, it asks an entailment model which other predicates imply it and in which argument order, admits the implications the model trusts as ordinary rules, such as `owns(X, Y) :- owned_by(Y, X)`, and lets the engine carry on. Each admitted rule keeps the model's opinion of it, so every answer can also say how probable it is: the chance that every rule of one of its derivations is right.

The rewriter runs on any logic engine that implements a small protocol. Two ship with it: a pure-Python engine with no dependencies, and [PyMeTTa](https://github.com/MesTTo/PyMeTTa) (MeTTa on SWI-Prolog). With tabling, every query terminates and gets exactly the answers its admitted rules imply; [DESIGN.md](DESIGN.md) states the conditions and the protocols, and [RESULTS.md](RESULTS.md) the experiment behind the defaults.

## Asking Wikidata in your own words

Each question below is a MeTTa file written in its own words, with its own rules. Beside each, `examples/wikidata` holds a knowledge base of Wikidata's statements about the question's subject, in Wikidata's 1,790 properties that link two entities, each glossed with its English label. None of the questions names one of those properties.

<!-- BEGIN GENERATED: wikidata-questions -->
```metta
; Who descends from Queen Victoria? Wikidata has no descendant property, and none of its 1,790 entity-valued
; properties is called "parent of"; the question uses its own words and its own rules.
(gloss parent_of "{x} is a parent of {y}")
(gloss descendant_of "{x} is a descendant of {y}")
(= (descendant_of $X $Y) (parent_of $Y $X))
(= (descendant_of $X $Z) (let True (parent_of $Y $X) (descendant_of $Y $Z)))
(query (descendant_of $X queen_victoria))

; What did Leonardo da Vinci create? Wikidata files a work's maker under many properties, recorded on the work.
(gloss created "{x} created {y}")
(query (created leonardo_da_vinci $W))

; What does Alphabet own, directly or through what it owns?
(gloss owns "{x} owns {y}")
(= (owns $X $Z) (let True (owns $X $Y) (owns $Y $Z)))
(query (owns alphabet_inc $C))
```
<!-- END GENERATED: wikidata-questions -->

### One query, step by step

`--trace` prints each step on standard error as it happens. Here PyMeTTa evaluates whether Alphabet owns YouTube, the cross-encoder nli-deberta-large narrows the candidates, and Jev, TypeSafe's judge model, decides which of them mean ownership:

<!-- BEGIN GENERATED: wikidata-trace -->
```text
$ semrewrite ask examples/wikidata/alphabet.metta 'owns(alphabet_inc, youtube_q866)' --judge jev --threshold 0.65 --strategy rewrite-depth-tabled --depth 1 --retriever judge --k 50 --engine pymetta --work 50000000 --seconds inf --trace --explain
pymetta starts on owns(alphabet_inc, youtube_q866), over 18,841 facts and 1 rule, with no rewrites yet
evaluation reached owns, which no fact states, and asks for the rewrites concluding it
  the nli-deberta-large retriever ranked the 1,790 predicates of the same arity as premises of "A owns B" and kept the 50 it ranks highest
  jev was asked 101 questions, whether each premise sentence entails "A owns B", and admits a score of 0.650 or more:
    admit  0.990  "B is owned by A"            owned_by_p127, ranked 2
           0.010  "A is owned by B"            owned_by_p127, ranked 2
    admit  0.990  "A is owner of B"            owner_of_p1830, ranked 3
           0.010  "B is owner of A"            owner_of_p1830, ranked 3
    admit  0.860  "B is afterward owned by A"  afterward_owned_by_p11812, ranked 11
           0.010  "A is afterward owned by B"  afterward_owned_by_p11812, ranked 11
    95 more rejected, the highest at 0.620: "B is beforehand owned by A" (beforehand_owned_by_p11811)
  admitted as rules:
    owns(X0, X1) ⇐ owned_by_p127(X1, X0)
    owns(X0, X1) ⇐ owner_of_p1830(X0, X1)
    owns(X0, X1) ⇐ afterward_owned_by_p11812(X1, X0)
pymetta finished: ok, 1 answer, 2,296,924 work
  the memo answered 5,699 more asks for owns, without asking the judge again
owns(alphabet_inc, youtube_q866)
    owns(alphabet_inc, youtube_q866)    [by owns(X, Z) :- owns(X, Y), owns(Y, Z)]
      owns(alphabet_inc, google_q95)    [by owns(X, Z) :- owns(X, Y), owns(Y, Z)]
        owns(alphabet_inc, xxvi_holdings)    [by owns(X0, X1) ⇐ owned_by_p127(X1, X0), score 0.990]
          owned_by_p127(xxvi_holdings, alphabet_inc)    [fact]
        owns(xxvi_holdings, google_q95)    [by owns(X0, X1) ⇐ owner_of_p1830(X0, X1), score 0.990]
          owner_of_p1830(xxvi_holdings, google_q95)    [fact]
      owns(google_q95, youtube_q866)    [by owns(X0, X1) ⇐ owned_by_p127(X1, X0), score 0.990]
        owned_by_p127(youtube_q866, google_q95)    [fact]

rewrites admitted (3):
  owns(X0, X1) ⇐ afterward_owned_by_p11812(X1, X0)
  owns(X0, X1) ⇐ owned_by_p127(X1, X0)
  owns(X0, X1) ⇐ owner_of_p1830(X0, X1)
ok: 1 answer, 2,296,924 work, 101 judge questions, 32.33 s
```
<!-- END GENERATED: wikidata-trace -->

No statement says `owns`, so the engine called back for rewrites in the middle of evaluating. The cross-encoder read all 1,790 properties as premises of "A owns B", in both argument orders, and Jev judged its 50 best, each both ways: "B is owned by A" entails "A owns B" at 0.990, and "A is owned by B" at 0.010. Three of Jev's 101 answers reached the threshold, and the engine carried on with those three as rules. The question's own transitive rule chained them: Alphabet owns XXVI Holdings, which owns Google, which owns YouTube. Wikidata states the first and last of those links on what is owned, as owned by, and the middle one on the owner, as owner of. As the rule recursed, evaluation asked for `owns` 5,699 more times, and the memo answered each, so Jev was asked once.

### Every answer against Wikidata's own

Wikidata files the properties that hold each relation in its own subproperty hierarchy, so its own answer to each question follows from the same statements ([questions.py](examples/wikidata/questions.py)). Each run allows one rewrite on any path, and both judges see the same 50 candidates; cosine-bge is an embedding-similarity judge:

<!-- BEGIN GENERATED: wikidata-summary -->
| question | Wikidata's answer | read literally | judged by cosine-bge | judged by jev |
|---|---|---|---|---|
| victoria: `descendant_of(X, queen_victoria)` | 1,164 | 0 | 1,165: 1,164 in Wikidata's answer, 1 not | 1,164, all in Wikidata's answer |
| leonardo: `created(leonardo_da_vinci, W)` | 854 | 0 | 0 | 851: 849 in Wikidata's answer, 2 not |
| alphabet: `owns(alphabet_inc, C)` | 3,225 | 0 | none, the work budget stopped it | 2,849, all in Wikidata's answer |
<!-- END GENERATED: wikidata-summary -->

Read literally, no question finds anything. Jev returned exactly Victoria's 1,164 descendants. Its two answers outside Wikidata's for Leonardo are works his own entry lists as notable. The 376 holdings it missed for Alphabet hang from companies that only three properties connect to the rest: copyright holder, which Jev scored 0.22, and parent and child organization, which the cross-encoder ranked 68th and 555th, so Jev never saw them. The embedding judge can't tell which argument is which. On Alphabet's question it admitted owned by in both directions, which the transitive rule turns into every company owning every other, and the work budget stopped it. [examples/README.md](examples/README.md) has every rewrite each run admitted and what each missed.

### A derivation five generations deep

<!-- BEGIN GENERATED: wikidata-descent -->
```text
$ semrewrite ask examples/wikidata/victoria.metta 'descendant_of(frederik_x_of_denmark, queen_victoria)' --judge jev --threshold 0.65 --strategy rewrite-depth-tabled --depth 1 --retriever judge --k 50 --engine pymetta --work 50000000 --seconds inf --explain
descendant_of(frederik_x_of_denmark, queen_victoria)
    descendant_of(frederik_x_of_denmark, queen_victoria)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
      parent_of(margrethe_ii_of_denmark, frederik_x_of_denmark)    [by parent_of(X0, X1) ⇐ mother_p25(X1, X0), score 0.990]
        mother_p25(frederik_x_of_denmark, margrethe_ii_of_denmark)    [fact]
      descendant_of(margrethe_ii_of_denmark, queen_victoria)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
        parent_of(ingrid_of_sweden, margrethe_ii_of_denmark)    [by parent_of(X0, X1) ⇐ mother_p25(X1, X0), score 0.990]
          mother_p25(margrethe_ii_of_denmark, ingrid_of_sweden)    [fact]
        descendant_of(ingrid_of_sweden, queen_victoria)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
          parent_of(princess_margaret_of_connaught, ingrid_of_sweden)    [by parent_of(X0, X1) ⇐ mother_p25(X1, X0), score 0.990]
            mother_p25(ingrid_of_sweden, princess_margaret_of_connaught)    [fact]
          descendant_of(princess_margaret_of_connaught, queen_victoria)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
            parent_of(prince_arthur_duke_of_connaught_and_strathearn, princess_margaret_of_connaught)    [by parent_of(X0, X1) ⇐ father_p22(X1, X0), score 0.980]
              father_p22(princess_margaret_of_connaught, prince_arthur_duke_of_connaught_and_strathearn)    [fact]
            descendant_of(prince_arthur_duke_of_connaught_and_strathearn, queen_victoria)    [by descendant_of(X, Y) :- parent_of(Y, X)]
              parent_of(queen_victoria, prince_arthur_duke_of_connaught_and_strathearn)    [by parent_of(X0, X1) ⇐ mother_p25(X1, X0), score 0.990]
                mother_p25(prince_arthur_duke_of_connaught_and_strathearn, queen_victoria)    [fact]

rewrites admitted (18):
  descendant_of(X0, X1) ⇐ derived_from_base_unit_p12571(X0, X1)
  descendant_of(X0, X1) ⇐ filial_church_p12476(X1, X0)
  descendant_of(X0, X1) ⇐ gens_p5025(X0, X1)
  descendant_of(X0, X1) ⇐ has_spin_off_p2512(X1, X0)
  descendant_of(X0, X1) ⇐ is_proceedings_from_p4745(X0, X1)
  descendant_of(X0, X1) ⇐ parent_club_p831(X0, X1)
  descendant_of(X0, X1) ⇐ parent_peak_p3137(X0, X1)
  parent_of(X0, X1) ⇐ child_p40(X0, X1)
  parent_of(X0, X1) ⇐ father_p22(X1, X0)
  parent_of(X0, X1) ⇐ mother_house_p612(X1, X0)
  parent_of(X0, X1) ⇐ mother_p25(X1, X0)
  parent_of(X0, X1) ⇐ parent_astronomical_body_p397(X1, X0)
  parent_of(X0, X1) ⇐ parent_cell_line_p3432(X1, X0)
  parent_of(X0, X1) ⇐ parent_club_p831(X1, X0)
  parent_of(X0, X1) ⇐ parent_organization_or_unit_p749(X1, X0)
  parent_of(X0, X1) ⇐ parent_p8810(X1, X0)
  parent_of(X0, X1) ⇐ parent_peak_p3137(X1, X0)
  parent_of(X0, X1) ⇐ subdivision_of_this_unit_p9059(X0, X1)
ok: 1 answer, 35,172 work, 202 judge questions, 42.17 s
```
<!-- END GENERATED: wikidata-descent -->

King Frederik X of Denmark descends from Victoria through his mother Margrethe II, her mother Ingrid of Sweden, Ingrid's mother Princess Margaret of Connaught, and Margaret's father Prince Arthur, Victoria's son. Each step is a Wikidata statement read through one of Jev's rewrites, shown with its score. Jev also admitted premises such as parent peak and parent club, which add nothing here because the knowledge base holds no statement in them.

### Without a sense of direction

<!-- BEGIN GENERATED: wikidata-similarity -->
```text
$ semrewrite ask examples/wikidata/victoria.metta 'descendant_of(queen_victoria, queen_victoria)' --judge cosine-bge --strategy rewrite-depth-tabled --depth 1 --retriever judge --k 50 --engine pymetta --work 50000000 --seconds inf --explain
descendant_of(queen_victoria, queen_victoria)
    descendant_of(queen_victoria, queen_victoria)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
      parent_of(princess_louise_duchess_of_argyll, queen_victoria)    [by parent_of(X0, X1) ⇐ child_p40(X1, X0), score 0.968]
        child_p40(queen_victoria, princess_louise_duchess_of_argyll)    [fact]
      descendant_of(princess_louise_duchess_of_argyll, queen_victoria)    [by descendant_of(X, Y) :- parent_of(Y, X)]
        parent_of(queen_victoria, princess_louise_duchess_of_argyll)    [by parent_of(X0, X1) ⇐ mother_p25(X1, X0), score 0.969]
          mother_p25(princess_louise_duchess_of_argyll, queen_victoria)    [fact]

rewrites admitted (8):
  descendant_of(X0, X1) ⇐ descendant_of(X1, X0)
  parent_of(X0, X1) ⇐ child_p40(X0, X1)
  parent_of(X0, X1) ⇐ child_p40(X1, X0)
  parent_of(X0, X1) ⇐ father_p22(X1, X0)
  parent_of(X0, X1) ⇐ mother_p25(X1, X0)
  parent_of(X0, X1) ⇐ parent_of(X1, X0)
  parent_of(X0, X1) ⇐ parent_p8810(X0, X1)
  parent_of(X0, X1) ⇐ parent_p8810(X1, X0)
ok: 1 answer, 2,581,151 work, 202 judge questions, 43.58 s
```
<!-- END GENERATED: wikidata-similarity -->

Embedding similarity scores how alike two sentences are, whichever way they run. As premises of "A is a parent of B", the embedding judge scored "B is the child of A" at 0.96 and "A is the child of B" at 0.97, and admitted both. So "Princess Louise is the child of Queen Victoria" also made Louise a parent of Victoria, and Victoria a descendant of herself.

### How sure is each answer?

`--beliefs` gives every answer an opinion that it holds, in subjective logic's terms: a probability P, and the belief b, disbelief d and uncertainty u it comes from. A rewrite is one claim, right or wrong for every use of it, so an answer is as probable as the chance that every rewrite of one of its derivations is right. Each rewrite is right as Jev's opinion of it says, drawn toward the rate at which labelled pairs hold by how far Jev proved right on them; [DESIGN.md](DESIGN.md#answer-beliefs) says how the probabilities combine.

That makes a low rewrite threshold cheap. At 0.1 instead of 0.65, Jev admits 110 rewrites into Victoria's descendants instead of 18, and 21 of the answers they add are not in Wikidata's record. Each of those 21 ranks below each of Wikidata's 1,164, so an answer threshold anywhere between them keeps exactly Wikidata's answer:

<!-- BEGIN GENERATED: wikidata-placed -->
| answer | rank | in Wikidata's answer | P | b | d | u |
|---|---|---|---|---|---|---|
| `sophia_hermann` | 1,160 of 1,185 | yes | 0.938 | 0.929 | 0.029 | 0.042 |
| `sophia_wise_glossinger` | 1,161 of 1,185 | yes | 0.938 | 0.929 | 0.029 | 0.042 |
| `stephan_albert_bernadotte_count_bernadotte` | 1,162 of 1,185 | yes | 0.938 | 0.929 | 0.029 | 0.042 |
| `stephan_straehl` | 1,163 of 1,185 | yes | 0.938 | 0.929 | 0.029 | 0.042 |
| `victoria_bayard_romanov_ilyinsky` | 1,164 of 1,185 | yes | 0.938 | 0.929 | 0.029 | 0.042 |
| `british_army` | 1,165 of 1,185 | no | 0.324 | 0.321 | 0.637 | 0.042 |
| `hellenic_armed_forces` | 1,166 of 1,185 | no | 0.324 | 0.321 | 0.637 | 0.042 |
| `imperial_german_army` | 1,167 of 1,185 | no | 0.324 | 0.321 | 0.637 | 0.042 |
| `luftwaffe` | 1,168 of 1,185 | no | 0.324 | 0.321 | 0.637 | 0.042 |
| `royal_air_force` | 1,169 of 1,185 | no | 0.324 | 0.321 | 0.637 | 0.042 |
<!-- END GENERATED: wikidata-placed -->

Asked about one of them, the rewriter shows where its doubt comes from:

<!-- BEGIN GENERATED: wikidata-graded -->
```text
$ semrewrite ask examples/wikidata/victoria.metta 'descendant_of(british_army, queen_victoria)' --judge jev --threshold 0.1 --strategy rewrite-depth-tabled --depth 1 --retriever judge --k 50 --engine pymetta --work 50000000 --seconds inf --beliefs --explain 2>/dev/null
descendant_of(british_army, queen_victoria)    P 0.324  b 0.321  d 0.637  u 0.042
    descendant_of(british_army, queen_victoria)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
      parent_of(edward_vii, british_army)    [by parent_of(X0, X1) ⇐ military_branch_p241(X0, X1), score 0.330]
        military_branch_p241(edward_vii, british_army)    [fact]
      descendant_of(edward_vii, queen_victoria)    [by descendant_of(X, Y) :- parent_of(Y, X)]
        parent_of(queen_victoria, edward_vii)    [by parent_of(X0, X1) ⇐ mother_p25(X1, X0), score 0.990]
          mother_p25(edward_vii, queen_victoria)    [fact]
```
<!-- END GENERATED: wikidata-graded -->

The British Army descends from Victoria because one of her sons served in it, read through a rewrite that takes a person's military branch for their child. Jev admitted that rewrite at 0.330, so the answer is more probably wrong than right. [RESULTS.md](RESULTS.md#how-sure-is-each-answer) has the same measurement on all three questions.

## A small example: choosing the threshold

`examples/parent_of_tom.kb` states Tom's family in words that never say "parent", in the rewriter's plain-text format. Each predicate has a gloss, a sentence with a placeholder per argument, which is what the entailment model reads, and the `gold` lines say which implications a careful reader accepts:

<!-- BEGIN GENERATED: tom-source -->
```text
% Tom's family, stated in words that never say "parent".
gloss mum "{x} is the mum of {y}"
gloss mother "{x} is the mother of {y}"
gloss dad "{x} is the dad of {y}"
gloss parent "{x} is a parent of {y}"
gloss child_of "{x} is a child of {y}"
gloss guardian "{x} is the legal guardian of {y}"
gloss grandparent "{x} is a grandparent of {y}"

mum(alice, tom).
dad(bob, tom).
mother(carol, alice).
child_of(bob, dave).
guardian(erin, tom).

grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

query parent(X, tom).
query grandparent(X, tom).

% The implications a careful reader accepts; erin, a legal guardian, is not a parent.
gold mum => mother.
gold mother => mum.
gold mother => parent.
gold dad => parent.
gold child_of => parent swap.
```
<!-- END GENERATED: tom-source -->

Asking for Tom's parents with the NLI cross-encoder nli-deberta-large as the judge, at the threshold it was calibrated to on labelled predicate pairs:

<!-- BEGIN GENERATED: tom-explain -->
```text
$ semrewrite ask examples/parent_of_tom.kb 'parent(X, tom)' --judge nli-deberta-large --retriever all --explain
parent(alice, tom)
    parent(alice, tom)    [by parent(X0, X1) ⇐ mum(X0, X1), score 0.998]
      mum(alice, tom)    [fact]
parent(bob, tom)
    parent(bob, tom)    [by parent(X0, X1) ⇐ dad(X0, X1), score 0.997]
      dad(bob, tom)    [fact]
parent(erin, tom)
    parent(erin, tom)    [by parent(X0, X1) ⇐ guardian(X0, X1), score 0.823]
      guardian(erin, tom)    [fact]

rewrites admitted (14):
  child_of(X0, X1) ⇐ dad(X1, X0)
  child_of(X0, X1) ⇐ mother(X1, X0)
  child_of(X0, X1) ⇐ mum(X1, X0)
  child_of(X0, X1) ⇐ parent(X1, X0)
  mother(X0, X1) ⇐ child_of(X1, X0)
  mother(X0, X1) ⇐ mum(X0, X1)
  mum(X0, X1) ⇐ child_of(X1, X0)
  mum(X0, X1) ⇐ mother(X0, X1)
  mum(X0, X1) ⇐ parent(X0, X1)
  parent(X0, X1) ⇐ child_of(X1, X0)
  parent(X0, X1) ⇐ dad(X0, X1)
  parent(X0, X1) ⇐ guardian(X0, X1)
  parent(X0, X1) ⇐ mother(X0, X1)
  parent(X0, X1) ⇐ mum(X0, X1)
ok: 3 answers, 63 work, 78 judge questions, 0.75 s
```
<!-- END GENERATED: tom-explain -->

Every answer comes with its derivation, and each step names the rule or the admitted rewrite it used, with the judge's score for a rewrite. The third answer is wrong: a legal guardian need not be a parent, and at this threshold the judge admits `guardian ⇒ parent` along with other false rewrites such as `child_of(y, x) ⇒ mother(x, y)`. Raising the threshold to 0.995, which calibration for answer F1 chose for lazy-tabled evaluation on three of the experiment's four held-out benchmarks, leaves the answers alice and bob:

<!-- BEGIN GENERATED: tom-strict -->
```text
$ semrewrite ask examples/parent_of_tom.kb 'parent(X, tom)' --judge nli-deberta-large --retriever all --threshold 0.995
parent(alice, tom)
parent(bob, tom)
ok: 2 answers, 14 work, 65 judge questions, 0.24 s
```
<!-- END GENERATED: tom-strict -->

At that threshold the judge admits six rewrites, all true, and the grandparent query composes them with the knowledge base's own rule:

<!-- BEGIN GENERATED: tom-grandparent -->
```text
$ semrewrite ask examples/parent_of_tom.kb 'grandparent(X, tom)' --judge nli-deberta-large --retriever all --threshold 0.995 --explain
grandparent(carol, tom)
    grandparent(carol, tom)    [by grandparent(X, Z) :- parent(X, Y), parent(Y, Z)]
      parent(carol, alice)    [by parent(X0, X1) ⇐ mother(X0, X1), score 0.996]
        mother(carol, alice)    [fact]
      parent(alice, tom)    [by parent(X0, X1) ⇐ mum(X0, X1), score 0.998]
        mum(alice, tom)    [fact]
grandparent(dave, tom)
    grandparent(dave, tom)    [by grandparent(X, Z) :- parent(X, Y), parent(Y, Z)]
      parent(dave, bob)    [by parent(X0, X1) ⇐ child_of(X1, X0), score 0.998]
        child_of(bob, dave)    [fact]
      parent(bob, tom)    [by parent(X0, X1) ⇐ dad(X0, X1), score 0.997]
        dad(bob, tom)    [fact]

rewrites admitted (6):
  mother(X0, X1) ⇐ mum(X0, X1)
  mum(X0, X1) ⇐ mother(X0, X1)
  parent(X0, X1) ⇐ child_of(X1, X0)
  parent(X0, X1) ⇐ dad(X0, X1)
  parent(X0, X1) ⇐ mother(X0, X1)
  parent(X0, X1) ⇐ mum(X0, X1)
ok: 2 answers, 24 work, 78 judge questions, 0.29 s
```
<!-- END GENERATED: tom-grandparent -->

That trade-off between admitting enough and admitting only what is true, and how to calibrate a judge for it, is most of what [RESULTS.md](RESULTS.md) measures.

## Install

Clone the repository and install the parts you need:

```sh
git clone https://github.com/MesTTo/semantic-rewriter
cd semantic-rewriter
pip install .                  # the core: pure-Python engine, no dependencies
pip install ".[models]"        # NLI, reranker and embedding judges, embedding retrieval (torch)
pip install ".[pymetta]"       # the PyMeTTa engine
pip install ".[all]"           # everything, plus the conformance kit and the Jev judge
```

The core runs on Python 3.11 or later; the PyMeTTa engine needs 3.12 or later. `uv sync --all-extras` builds the environment the experiment used, with CUDA 12.8 torch on Linux and Windows. A command that needs a library you have not installed names the extra that installs it.

## Command line

```sh
semrewrite ask KB QUERY [--judge NAME] [--engine python|pymetta] [--strategy NAME] [--depth N] [--explain] [--trace] [--json]
semrewrite ask KB QUERY --beliefs [--answer-threshold P]  # each answer with its opinion, ranked by P
semrewrite rewrites KB PREDICATE [--judge NAME] [--all]   # what discovery admits, and with --all what it rejects
semrewrite rewrites KB PREDICATE --snars                  # the admitted rewrites as SNARS beliefs and rules for GoalChainer
semrewrite judge [--judge NAME] [--trust T] [--base-rate A] < PAIRS  # each sentence pair's opinion, as JSON lines
semrewrite engines                                       # installed engines and their capabilities
semrewrite strategies                                    # the strategies and their knobs
semrewrite conformance --engine NAME                     # check an engine against the protocol
semrewrite convert family.metta family.kb                # between the text, JSON and MeTTa formats
```

`--judge oracle` uses the `gold` lines of the knowledge base instead of a model, which is how tests and the experiment separate engine behaviour from judge error. `--retriever all` judges every predicate of the right arity instead of the 8 nearest by gloss embedding, which suits small catalogues and loads no embedding model. `--retriever judge` ranks every predicate with the `--prefilter` judge, the cross-encoder nli-deberta-large by default, and the admitting judge sees only the `--k` it ranks highest, which suits a catalogue too large to judge whole, such as Wikidata's 1,790 properties above. `--depth N` replaces the strategy's depth bound, counted the way the strategy counts it, and `--depth 0` removes it. `--explain` prints each answer's derivation and the rewrites admitted, and `--trace` prints each step on standard error as it happens, as in the Alphabet example above. `--beliefs` prints each answer with its opinion, highest P first, as in the Victoria example above, and `--answer-threshold P` drops the answers below P. `rewrites --snars` prints each admitted rewrite as two lines of the live world of mettabase's SNARS, the subjective-logic reasoner GoalChainer uses: its validity, a statement `(valid rw-owns-owned_by_p127-10)` believed with the rewrite's opinion as evidence, and a row of SNARS's rule kit deriving `owns(X0, X1)` from `owned_by_p127(X1, X0)` and that statement. SNARS's `sn prove` then applies the rewrite by subjective-logic deduction, and every application of it shares the one validity statement, as every answer through it does here. `judge` reads sentence pairs from standard input, one JSON object a line with a `premise` and a `conclusion`, and writes each pair's opinion as a JSON line, calibrated at the judge's trust and base rate unless `--trust` and `--base-rate` give others, for pairs drawn from another population; GoalChainer uses it to ask whether a request states an element of a problem. The exit status is 0 when the query finished, 2 when a budget stopped it and 1 on an error.

## Python

```python
from semrewrite import Rewriter, formats
from semrewrite.engines import make as engine
from semrewrite.judges import make as judge

kb = formats.load("examples/parent_of_tom.kb")
rewriter = Rewriter(kb, judge("nli-deberta-large"), engine("python"))
outcome = rewriter.ask("grandparent(X, tom)", "lazy-tabled")
print(outcome.answers)        # {('carol', 'tom'), ('dave', 'tom')}
print(outcome.admitted)       # the rewrites admitted, as (premise, conclusion, argument mapping)
```

A knowledge base can also be built in code with `KB`, `atom` and `Rule`, and predicates may have any arity: `gave(x, y, z)` with the gloss `{x} gave {y} to {z}` implies `received(z, y, x)`, and discovery proposes every argument mapping.

## Engines

| engine | needs | tables | depth bound | bound on rewrites | lazy discovery | untabled | proofs |
|---|---|---|---|---|---|---|---|
| `python` | nothing | yes | yes | yes | yes | no | yes |
| `pymetta` | `pymetta` | yes | yes | yes | yes | yes | yes |

An engine is anything with a `solve(program, query, budget, expand)` method and a `capabilities` record; DESIGN.md gives the five rules it must keep. `semrewrite conformance --engine NAME` checks them with property tests against a reference evaluator: exact answers with tables, one meaning of depth, lazy expansion equal to eager, hard budgets, and engine keywords accepted as ordinary names. A Prolog through janus, a Datalog engine or SQL with recursive queries can implement the same protocol; add it to `semrewrite/engines/__init__.py` and run the kit.

## Judges

| judge | what it is | extra |
|---|---|---|
| `nli-deberta-base`, `nli-deberta-large`, `nli-deberta-large-wanli` | NLI cross-encoders reading premise and conclusion together | `models` |
| `reranker-qwen3` | Qwen3-Reranker asked yes or no | `models` |
| `typical-qwen3-4b` | Qwen3-4B-Instruct asked whether the conclusion typically follows, answering yes, no or unsure | `models` |
| `jev` | TypeSafe's hosted System One model, with answers cached on disk; reads a TypeSafe API key from `TYPESAFE_API_KEY`, or from the file `JEV_KEY_FILE` names | `jev` |
| `cosine-bge`, `cosine-qwen3` | embedding similarity, the baseline that cannot tell direction | `models` |
| `oracle` | admits exactly a given set of implications | nothing |

Each judge's default threshold, trust and base rate in `semrewrite/judges/calibration.json` were calibrated on the labelled pairs of the kind of question it is asked: Wikidata's property pairs for the judges of what must follow, δ-ATOMIC's for `typical-qwen3-4b`. For the rewriter, calibrate it on your own knowledge bases instead (see RESULTS.md): the pair-set thresholds favour recall, and the rewriter needs precision because it applies every admitted rule everywhere. A judge is anything with `name`, `threshold`, `trust`, `base_rate`, `usage`, `opinions(proposals)` and `close()`, which DESIGN.md describes.

## Data

The pair sets the judges were measured on live under `data/pairs`: a hand-labelled hard set, and pairs from Wikidata's property hierarchy (CC0), whose query results are under `data/raw/wikidata`. Two more sets hold implications that are true typically rather than by meaning. δ-ATOMIC's defeasible inferences ([Rudinger et al. 2020](https://github.com/rudinger/defeasible-nli), MIT, under `data/raw/delta-atomic`) pair an everyday event about PersonX and a human-written update with an inference the update strengthens or defeats. Balanced COPA's causes and effects ([Kavumba et al. 2019](https://huggingface.co/datasets/pkavumba/balanced-copa), CC BY 4.0 on its author's card) give each question's two alternatives, the more plausible one holding. The Levy/Holt directional pairs come from [entgraph_eval](https://github.com/mjhosseini/entgraph_eval), which states no license, and COPA's own questions state none either, so this repository includes neither. These commands download them at the revisions the experiment used, check each file's SHA-256, and build the pair files:

```sh
python -m semrewrite.pairs fetch levyholt
python -m semrewrite.pairs fetch copa
python -m semrewrite.pairs build levyholt copa
```

The Jev judge's cached answers are not included either, so rerunning it needs a TypeSafe API key.

## What the experiment found

The experiment ran twelve strategies with a perfect judge, and six of them with each of seven model judges, over four generated knowledge bases (75 predicates, 58 queries) on PyMeTTa:

- With tables and a per-predicate memo, every query terminated, on the benchmarks and on 400 generated catalogues per property; without tables, rewriting looped exactly when the query reached a cycle of rewrites.
- Jev and the NLI cross-encoders tell which way an implication runs (direction accuracy 0.77 to 0.89 on two external pair sets); embedding similarity sits at chance.
- Admission precision decides answer quality, so the judge's threshold has to be calibrated for answer F1, not for pair accuracy.
- With an imperfect judge, a depth bound of 3 limits how far a false rule spreads. At calibrated thresholds, tabled evaluation bounded at three steps (`depth-tabled`) raised answer F1 for six of the seven judges, nli-deberta-large's from 0.68 to 0.91, and cost the perfect judge three answers that need more than three steps. Bounding three rewrites alone (`rewrite-depth-tabled`) keeps those answers and gave Jev its best F1, 0.92, but scored below the step bound with every other judge.
- Asked what typically follows rather than what is entailed, Qwen3-4B-Instruct-2507 (`typical-qwen3-4b`) scores held-out log loss 0.588 on δ-ATOMIC's defeasible inferences and 0.485 on Balanced COPA, against nli-deberta-large's 0.661 and 0.637, and on a knowledge base of actions ranks the three that waste time first, where the entailment judge finds one.

RESULTS.md has every number, the method and the twelve PyMeTTa defects found on the way, ten of them with a script under `repros/` that reproduces them.

## Guarantees and limits

Every query terminates when the catalogue of predicates is finite, rewrites only rename and permute arguments, user rules have no function symbols, discovery is memoised, and evaluation is tabled or depth-bounded. Answers are exact with respect to the admitted rules; whether they are true depends on the judge. Function symbols, negation and discovery of facts from text are outside the design. DESIGN.md explains each condition and what breaks without it.

## Development

```sh
uv sync --all-extras
uv run pytest                                  # unit, property and conformance tests, 400 cases per property
uv run pytest --hypothesis-profile=ci -n auto  # the same with 100 cases per property on every CPU, as CI runs PyMeTTa
uv run pytest --hypothesis-profile=ci -n 4 --store-durations --clean-durations  # refresh .test_durations, which balances CI's PyMeTTa runners
uv run ruff check .                            # lint, as CI does
uv run ty check semrewrite                     # type check, with every extra installed
uv run semrewrite conformance --engine pymetta
uv run python examples/walkthrough.py          # rerun every command this README and examples/README.md show
```

Every transcript in this README and in examples/README.md sits between `BEGIN GENERATED` and `END GENERATED` markers and is what the walkthrough printed; `tests/test_examples.py` checks that it covers every block. The experiment's scripts are under `experiments/` and its results under `results/`; RESULTS.md says how to rerun them.

## License

Apache License 2.0; see [LICENSE](LICENSE).
