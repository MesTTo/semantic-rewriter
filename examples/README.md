# Semantic chaining in MeTTa

Each example here is a knowledge base written in MeTTa and a question asked in words that none of its facts use. The rewriter asks a judge which statements entail which, and admits the rewrites the judge accepts. The PyMeTTa engine then evaluates the knowledge base's own rules together with those rewrites, with tables, and returns every answer with its shortest derivation. A derivation chains the judge's rewrites and the knowledge base's rules, and shows the judge's score beside each rewrite step.

The judge is Jev, TypeSafe's hosted model, at the settings [RESULTS.md](../RESULTS.md) found best for it: a threshold of 0.65, the value calibrated for answer F1 on every held-out benchmark, and at most three rewrites below any answer (`rewrite-depth-tabled`), which reached held-out F1 0.916. Both examples start from programs in the MeTTa example corpus ([MesTTo/MeTTa-Examples](https://github.com/MesTTo/MeTTa-Examples)) and in PyMeTTa's own examples ([MesTTo/PyMeTTa](https://github.com/MesTTo/PyMeTTa/tree/main/examples)).

Every transcript below was printed by the command above it. `python examples/walkthrough.py` runs them all again and rewrites this file.

## Fish oil and Raynaud's syndrome

In 1986 Don Swanson found that fish oil might help Raynaud's syndrome, a hypothesis no paper had stated. One literature reported that fish oil lowers blood viscosity and platelet aggregation; another, which never cited the first, reported that both make Raynaud's syndrome worse. PyMeTTa's [literature_discovery.py](https://github.com/MesTTo/PyMeTTa/blob/main/examples/reasoning/literature_discovery.py) reproduces the discovery with Swanson's rule and closes a gap between terms, `fish-oil` and `omega-3`, with an embedding.

Here the gap is in the relations. Each paper states its finding in its own words, and Swanson's rule is written once, in the reader's words, `lowers` and `aggravates`. Two claims are red herrings: dehydration raises blood viscosity, which is the opposite of lowering it, and inflammation is only associated with the disease. The gold and trap lines at the end are for the oracle judge, which answers from them when no model is available.

<!-- BEGIN GENERATED: swanson-source -->
```metta
; Literature-based discovery, after Swanson (1986): fish oil might help Raynaud's syndrome.
; No paper below says so. The chain runs through two literatures that never cite each other,
; each written in its own words, and Swanson's rule is stated once, in the reader's words.
; PyMeTTa's examples/reasoning/literature_discovery.py closes the same gap with an embedding;
; here a judge decides which phrasings entail the rule's, and in which direction.

(gloss lowers "{x} lowers the level of {y}")
(gloss aggravates "{x} aggravates the disease {y}")
(gloss worth_testing_against "{x} is worth testing as a treatment for the disease {y}")
(gloss is_rich_in "{x} is rich in {y}")
(gloss reduces "{x} reduces the level of {y}")
(gloss is_reduced_by "the level of {x} is reduced by {y}")
(gloss exacerbates "{x} exacerbates the disease {y}")
(gloss is_made_worse_by "the disease {x} is made worse by {y}")
(gloss increases "{x} raises the level of {y}")
(gloss is_associated_with "{x} is associated with the disease {y}")

; The nutrition literature.
(= (reduces omega_3 blood_viscosity) True)
(= (is_reduced_by platelet_aggregation omega_3) True)
(= (reduces aspirin inflammation) True)
; The vascular literature.
(= (is_made_worse_by raynauds blood_viscosity) True)
(= (exacerbates platelet_aggregation raynauds) True)
(= (is_associated_with inflammation raynauds) True)
(= (increases dehydration blood_viscosity) True)
; Food chemistry.
(= (is_rich_in fish_oil omega_3) True)

; Swanson's rule: what lowers something that aggravates a condition is worth testing against it.
(= (worth_testing_against $A $C) (let True (lowers $A $B) (aggravates $B $C)))
; A food lowers whatever a compound it is rich in lowers.
(= (lowers $F $B) (let True (is_rich_in $F $A) (lowers $A $B)))

(query (worth_testing_against $X raynauds))

; What a careful reader accepts, for the oracle judge: an association is not aggravation,
; and an increase is the opposite of lowering.
(gold reduces lowers (0 1))
(gold is_reduced_by lowers (1 0))
(gold exacerbates aggravates (0 1))
(gold is_made_worse_by aggravates (1 0))
(trap increases)
(trap is_associated_with)
```
<!-- END GENERATED: swanson-source -->

Read literally, the knowledge base answers nothing, because no fact says `lowers` or `aggravates`:

<!-- BEGIN GENERATED: swanson-exact -->
```text
$ semrewrite ask examples/swanson.metta 'worth_testing_against(X, raynauds)' --strategy exact --engine pymetta
ok: 0 answers, 2,261 work, 0 judge questions, 0.07 s
```
<!-- END GENERATED: swanson-exact -->

With Jev deciding which phrasings entail the rule's, MeTTa finds both hypotheses, omega-3 and fish oil, and neither red herring. Fish oil's derivation goes through the knowledge base's food rule, through Jev's reading of "omega-3 reduces the level of blood viscosity" as "omega-3 lowers the level of blood viscosity", and through its reading of "the disease raynauds is made worse by blood viscosity" as "blood viscosity aggravates the disease raynauds", with the arguments swapped.

<!-- BEGIN GENERATED: swanson-jev -->
```text
$ semrewrite ask examples/swanson.metta 'worth_testing_against(X, raynauds)' --judge jev --threshold 0.65 --strategy rewrite-depth-tabled --retriever all --engine pymetta --explain
worth_testing_against(fish_oil, raynauds)
    worth_testing_against(fish_oil, raynauds)    [by worth_testing_against(A, C) :- lowers(A, B), aggravates(B, C)]
      lowers(fish_oil, blood_viscosity)    [by lowers(F, B) :- is_rich_in(F, A), lowers(A, B)]
        is_rich_in(fish_oil, omega_3)    [fact]
        lowers(omega_3, blood_viscosity)    [by lowers(X0, X1) ⇐ reduces(X0, X1), score 0.970]
          reduces(omega_3, blood_viscosity)    [fact]
      aggravates(blood_viscosity, raynauds)    [by aggravates(X0, X1) ⇐ is_made_worse_by(X1, X0), score 0.970]
        is_made_worse_by(raynauds, blood_viscosity)    [fact]
worth_testing_against(omega_3, raynauds)
    worth_testing_against(omega_3, raynauds)    [by worth_testing_against(A, C) :- lowers(A, B), aggravates(B, C)]
      lowers(omega_3, blood_viscosity)    [by lowers(X0, X1) ⇐ reduces(X0, X1), score 0.970]
        reduces(omega_3, blood_viscosity)    [fact]
      aggravates(blood_viscosity, raynauds)    [by aggravates(X0, X1) ⇐ is_made_worse_by(X1, X0), score 0.970]
        is_made_worse_by(raynauds, blood_viscosity)    [fact]

rewrites admitted (12):
  aggravates(X0, X1) ⇐ exacerbates(X0, X1)
  aggravates(X0, X1) ⇐ is_made_worse_by(X1, X0)
  exacerbates(X0, X1) ⇐ aggravates(X0, X1)
  exacerbates(X0, X1) ⇐ is_made_worse_by(X1, X0)
  is_made_worse_by(X0, X1) ⇐ aggravates(X1, X0)
  is_made_worse_by(X0, X1) ⇐ exacerbates(X1, X0)
  is_reduced_by(X0, X1) ⇐ lowers(X1, X0)
  is_reduced_by(X0, X1) ⇐ reduces(X1, X0)
  lowers(X0, X1) ⇐ is_reduced_by(X1, X0)
  lowers(X0, X1) ⇐ reduces(X0, X1)
  reduces(X0, X1) ⇐ is_reduced_by(X1, X0)
  reduces(X0, X1) ⇐ lowers(X0, X1)
ok: 2 answers, 8,476 work, 152 judge questions, 0.04 s
```
<!-- END GENERATED: swanson-jev -->

Each answer shows one shortest derivation, and omega-3 has a second one of the same length, through the other literature. Asked for what omega-3 lowers and what aggravates the disease, the same rewritten program finds both paths, blood viscosity and platelet aggregation:

<!-- BEGIN GENERATED: swanson-paths -->
```text
$ semrewrite ask examples/swanson.metta 'lowers(omega_3, B)' --judge jev --threshold 0.65 --strategy rewrite-depth-tabled --retriever all --engine pymetta
lowers(omega_3, blood_viscosity)
lowers(omega_3, platelet_aggregation)
ok: 2 answers, 2,901 work, 76 judge questions, 0.04 s

$ semrewrite ask examples/swanson.metta 'aggravates(B, raynauds)' --judge jev --threshold 0.65 --strategy rewrite-depth-tabled --retriever all --engine pymetta
aggravates(blood_viscosity, raynauds)
aggravates(platelet_aggregation, raynauds)
ok: 2 answers, 2,495 work, 57 judge questions, 0.03 s
```
<!-- END GENERATED: swanson-paths -->

Jev's scores show why the red herrings stay out. Asked what entails "X lowers the level of Y", it accepts the two paraphrases, the reversed passive among them, and gives "X raises the level of Y" 0.01. It accepts the two ways the papers say "aggravates" and gives "is associated with" 0.06:

<!-- BEGIN GENERATED: swanson-judged -->
```text
$ semrewrite rewrites examples/swanson.metta lowers --judge jev --threshold 0.65 --retriever all --all
admit   0.980  lowers(X0, X1) ⇐ is_reduced_by(X1, X0)
admit   0.970  lowers(X0, X1) ⇐ reduces(X0, X1)
reject  0.180  lowers(X0, X1) ⇐ worth_testing_against(X0, X1)
reject  0.060  lowers(X0, X1) ⇐ is_made_worse_by(X1, X0)
reject  0.050  lowers(X0, X1) ⇐ aggravates(X1, X0)
reject  0.040  lowers(X0, X1) ⇐ exacerbates(X0, X1)
reject  0.040  lowers(X0, X1) ⇐ exacerbates(X1, X0)
reject  0.040  lowers(X0, X1) ⇐ increases(X1, X0)
reject  0.040  lowers(X0, X1) ⇐ is_associated_with(X1, X0)
reject  0.040  lowers(X0, X1) ⇐ reduces(X1, X0)
reject  0.030  lowers(X0, X1) ⇐ aggravates(X0, X1)
reject  0.030  lowers(X0, X1) ⇐ is_associated_with(X0, X1)
reject  0.030  lowers(X0, X1) ⇐ is_made_worse_by(X0, X1)
reject  0.030  lowers(X0, X1) ⇐ is_reduced_by(X0, X1)
reject  0.030  lowers(X0, X1) ⇐ is_rich_in(X1, X0)
reject  0.030  lowers(X0, X1) ⇐ lowers(X1, X0)
reject  0.020  lowers(X0, X1) ⇐ is_rich_in(X0, X1)
reject  0.020  lowers(X0, X1) ⇐ worth_testing_against(X1, X0)
reject  0.010  lowers(X0, X1) ⇐ increases(X0, X1)
19 judge questions at threshold 0.650

$ semrewrite rewrites examples/swanson.metta aggravates --judge jev --threshold 0.65 --retriever all --all
admit   0.970  aggravates(X0, X1) ⇐ is_made_worse_by(X1, X0)
admit   0.950  aggravates(X0, X1) ⇐ exacerbates(X0, X1)
reject  0.110  aggravates(X0, X1) ⇐ increases(X0, X1)
reject  0.070  aggravates(X0, X1) ⇐ lowers(X0, X1)
reject  0.060  aggravates(X0, X1) ⇐ is_associated_with(X0, X1)
reject  0.060  aggravates(X0, X1) ⇐ reduces(X1, X0)
reject  0.050  aggravates(X0, X1) ⇐ exacerbates(X1, X0)
reject  0.040  aggravates(X0, X1) ⇐ increases(X1, X0)
reject  0.040  aggravates(X0, X1) ⇐ is_associated_with(X1, X0)
reject  0.040  aggravates(X0, X1) ⇐ is_made_worse_by(X0, X1)
reject  0.040  aggravates(X0, X1) ⇐ is_reduced_by(X0, X1)
reject  0.040  aggravates(X0, X1) ⇐ is_reduced_by(X1, X0)
reject  0.040  aggravates(X0, X1) ⇐ lowers(X1, X0)
reject  0.040  aggravates(X0, X1) ⇐ worth_testing_against(X1, X0)
reject  0.040  aggravates(X0, X1) ⇐ aggravates(X1, X0)
reject  0.030  aggravates(X0, X1) ⇐ is_rich_in(X0, X1)
reject  0.030  aggravates(X0, X1) ⇐ is_rich_in(X1, X0)
reject  0.030  aggravates(X0, X1) ⇐ reduces(X0, X1)
reject  0.020  aggravates(X0, X1) ⇐ worth_testing_against(X0, X1)
19 judge questions at threshold 0.650
```
<!-- END GENERATED: swanson-judged -->

An embedding judge cannot do this. Cosine similarity is symmetric, so at the threshold calibrated for it the rewriter admits that if Y lowers X then X lowers Y, and it never connects "exacerbates" or "is made worse by" to "aggravates". It finds neither hypothesis:

<!-- BEGIN GENERATED: swanson-similarity -->
```text
$ semrewrite ask examples/swanson.metta 'worth_testing_against(X, raynauds)' --judge cosine-bge --strategy rewrite-depth-tabled --retriever all --engine pymetta --explain

rewrites admitted (18):
  aggravates(X0, X1) ⇐ aggravates(X1, X0)
  is_reduced_by(X0, X1) ⇐ is_reduced_by(X1, X0)
  is_reduced_by(X0, X1) ⇐ lowers(X0, X1)
  is_reduced_by(X0, X1) ⇐ lowers(X1, X0)
  is_reduced_by(X0, X1) ⇐ reduces(X0, X1)
  is_reduced_by(X0, X1) ⇐ reduces(X1, X0)
  is_rich_in(X0, X1) ⇐ is_rich_in(X1, X0)
  lowers(X0, X1) ⇐ is_reduced_by(X0, X1)
  lowers(X0, X1) ⇐ is_reduced_by(X1, X0)
  lowers(X0, X1) ⇐ lowers(X1, X0)
  lowers(X0, X1) ⇐ reduces(X0, X1)
  lowers(X0, X1) ⇐ reduces(X1, X0)
  reduces(X0, X1) ⇐ is_reduced_by(X0, X1)
  reduces(X0, X1) ⇐ is_reduced_by(X1, X0)
  reduces(X0, X1) ⇐ lowers(X0, X1)
  reduces(X0, X1) ⇐ lowers(X1, X0)
  reduces(X0, X1) ⇐ reduces(X1, X0)
  worth_testing_against(X0, X1) ⇐ worth_testing_against(X1, X0)
ok: 0 answers, 25,390 work, 114 judge questions, 0.53 s
```
<!-- END GENERATED: swanson-similarity -->

Tables make the rewrites safe to chain. Jev admits "reduces" and "lowers" in both directions, so the rewrites form a cycle, and plain top-down evaluation without tables or a memo goes round it, asking the judge again at every turn, until the work budget stops it with no answer:

<!-- BEGIN GENERATED: swanson-naive -->
```text
$ semrewrite ask examples/swanson.metta 'worth_testing_against(X, raynauds)' --judge jev --threshold 0.65 --strategy naive --retriever all --engine pymetta --seconds 600
work_limit: 0 answers, 2,058,835 work, 548,777 judge questions, 2.80 s
```
<!-- END GENERATED: swanson-naive -->

## Bratko's family, recorded by five hands

[Chapter 2](https://github.com/MesTTo/MeTTa-Examples/blob/main/ch02-programming-a-family-tree/04-the-whole-program.metta) of the MeTTa examples programs the family from Ivan Bratko's Prolog textbook, Tom, Bob, Liz, Ann, Pat and Jim, as `parent` facts and asks for Tom's descendants: bob, liz, ann, pat and jim. Here the same family is recorded by five hands, a birth certificate, a hospital register, a census, the family bible and a school record, none of which writes "parent", and Pam appears as Bob's mother, as in PyMeTTa's [family_algebras.py](https://github.com/MesTTo/PyMeTTa/blob/main/examples/gallery/family_algebras.py). Three more facts name people close to a parent who are not one: a godfather, a legal guardian and a namesake.

<!-- BEGIN GENERATED: family-source -->
```metta
; The family tree of chapter 2 of the MeTTa examples (Bratko's family: Tom, Pam, Bob, Liz, Ann,
; Pat and Jim), recorded by five hands, none of which wrote "parent". The chapter's question,
; Tom's descendants, is asked in words no fact uses, and its answer is the chapter's own:
; bob, liz, ann, pat and jim.

(gloss parent_of "{x} is a parent of {y}")
(gloss descendant_of "{x} is a descendant of {y}")
(gloss mother_of "{x} is the mother of {y}")
(gloss father_of "{x} is the father of {y}")
(gloss gave_birth_to "{x} gave birth to {y}")
(gloss daughter_of "{x} is the daughter of {y}")
(gloss begat "{x} begat {y}")
(gloss son_of "{x} is the son of {y}")
(gloss godfather_of "{x} is the godfather of {y}")
(gloss legal_guardian_of "{x} is the legal guardian of {y}")
(gloss named_after "{x} is named after {y}")

; A birth certificate, a hospital register, a census, the family bible and a school record.
(= (father_of tom bob) True)
(= (gave_birth_to pam bob) True)
(= (daughter_of liz tom) True)
(= (begat bob ann) True)
(= (begat bob pat) True)
(= (son_of jim pat) True)
; Close to a parent, and not one.
(= (godfather_of tom kim) True)
(= (legal_guardian_of pat lee) True)
(= (named_after max tom) True)

; The chapter's descendants-of, as two rules.
(= (descendant_of $X $Y) (parent_of $Y $X))
(= (descendant_of $X $Z) (let True (parent_of $Y $X) (descendant_of $Y $Z)))

(query (descendant_of $X tom))
(query (mother_of $X bob))
(query (mother_of $X jim))

; What a careful reader accepts, for the oracle judge. Nothing says whether pat is jim's mother.
(gold father_of parent_of (0 1))
(gold mother_of parent_of (0 1))
(gold gave_birth_to mother_of (0 1))
(gold gave_birth_to parent_of (0 1))
(gold daughter_of parent_of (1 0))
(gold son_of parent_of (1 0))
(gold begat parent_of (0 1))
(gold begat father_of (0 1))
(gold father_of begat (0 1))
(trap godfather_of)
(trap legal_guardian_of)
(trap named_after)
```
<!-- END GENERATED: family-source -->

The chapter's question, asked in words no fact uses, gets the chapter's answer. Jev reads "jim is the son of pat" as "pat is a parent of jim", with the arguments swapped, and "tom is the father of bob" as "bob is a descendant of tom". Kim, Lee and Max stay out:

<!-- BEGIN GENERATED: family-jev -->
```text
$ semrewrite ask examples/family_tree.metta 'descendant_of(X, tom)' --judge jev --threshold 0.65 --strategy rewrite-depth-tabled --retriever all --engine pymetta --explain
descendant_of(ann, tom)
    descendant_of(ann, tom)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
      parent_of(bob, ann)    [by parent_of(X0, X1) ⇐ begat(X0, X1), score 0.900]
        begat(bob, ann)    [fact]
      descendant_of(bob, tom)    [by descendant_of(X0, X1) ⇐ father_of(X1, X0), score 0.960]
        father_of(tom, bob)    [fact]
descendant_of(bob, tom)
    descendant_of(bob, tom)    [by descendant_of(X0, X1) ⇐ father_of(X1, X0), score 0.960]
      father_of(tom, bob)    [fact]
descendant_of(jim, tom)
    descendant_of(jim, tom)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
      parent_of(pat, jim)    [by parent_of(X0, X1) ⇐ son_of(X1, X0), score 0.980]
        son_of(jim, pat)    [fact]
      descendant_of(pat, tom)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
        parent_of(bob, pat)    [by parent_of(X0, X1) ⇐ begat(X0, X1), score 0.900]
          begat(bob, pat)    [fact]
        descendant_of(bob, tom)    [by descendant_of(X0, X1) ⇐ father_of(X1, X0), score 0.960]
          father_of(tom, bob)    [fact]
descendant_of(liz, tom)
    descendant_of(liz, tom)    [by descendant_of(X0, X1) ⇐ daughter_of(X0, X1), score 0.980]
      daughter_of(liz, tom)    [fact]
descendant_of(pat, tom)
    descendant_of(pat, tom)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
      parent_of(bob, pat)    [by parent_of(X0, X1) ⇐ begat(X0, X1), score 0.900]
        begat(bob, pat)    [fact]
      descendant_of(bob, tom)    [by descendant_of(X0, X1) ⇐ father_of(X1, X0), score 0.960]
        father_of(tom, bob)    [fact]

rewrites admitted (19):
  begat(X0, X1) ⇐ daughter_of(X1, X0)
  begat(X0, X1) ⇐ father_of(X0, X1)
  begat(X0, X1) ⇐ gave_birth_to(X0, X1)
  begat(X0, X1) ⇐ mother_of(X0, X1)
  begat(X0, X1) ⇐ son_of(X1, X0)
  descendant_of(X0, X1) ⇐ begat(X1, X0)
  descendant_of(X0, X1) ⇐ daughter_of(X0, X1)
  descendant_of(X0, X1) ⇐ father_of(X1, X0)
  descendant_of(X0, X1) ⇐ gave_birth_to(X1, X0)
  descendant_of(X0, X1) ⇐ mother_of(X1, X0)
  descendant_of(X0, X1) ⇐ parent_of(X1, X0)
  descendant_of(X0, X1) ⇐ son_of(X0, X1)
  mother_of(X0, X1) ⇐ gave_birth_to(X0, X1)
  parent_of(X0, X1) ⇐ begat(X0, X1)
  parent_of(X0, X1) ⇐ daughter_of(X1, X0)
  parent_of(X0, X1) ⇐ father_of(X0, X1)
  parent_of(X0, X1) ⇐ gave_birth_to(X0, X1)
  parent_of(X0, X1) ⇐ mother_of(X0, X1)
  parent_of(X0, X1) ⇐ son_of(X1, X0)
ok: 5 answers, 29,085 work, 168 judge questions, 0.05 s
```
<!-- END GENERATED: family-jev -->

The same rewritten program answers the question the other way round, Jim's ancestors:

<!-- BEGIN GENERATED: family-ancestors -->
```text
$ semrewrite ask examples/family_tree.metta 'descendant_of(jim, Y)' --judge jev --threshold 0.65 --strategy rewrite-depth-tabled --retriever all --engine pymetta
descendant_of(jim, bob)
descendant_of(jim, pam)
descendant_of(jim, pat)
descendant_of(jim, tom)
ok: 4 answers, 17,542 work, 168 judge questions, 0.04 s
```
<!-- END GENERATED: family-ancestors -->

The rewriter also has to keep direction. Pam gave birth to Bob, so she is his mother. Jim is Pat's son, so Pat is his parent, but nothing says whether Pat is his mother or his father, and the second query rightly finds no answer:

<!-- BEGIN GENERATED: family-mothers -->
```text
$ semrewrite ask examples/family_tree.metta 'mother_of(X, bob)' --judge jev --threshold 0.65 --strategy rewrite-depth-tabled --retriever all --engine pymetta --explain
mother_of(pam, bob)
    mother_of(pam, bob)    [by mother_of(X0, X1) ⇐ gave_birth_to(X0, X1), score 0.820]
      gave_birth_to(pam, bob)    [fact]

rewrites admitted (1):
  mother_of(X0, X1) ⇐ gave_birth_to(X0, X1)
ok: 1 answers, 1,916 work, 42 judge questions, 0.04 s

$ semrewrite ask examples/family_tree.metta 'mother_of(X, jim)' --judge jev --threshold 0.65 --strategy rewrite-depth-tabled --retriever all --engine pymetta --explain

rewrites admitted (1):
  mother_of(X0, X1) ⇐ gave_birth_to(X0, X1)
ok: 0 answers, 1,826 work, 42 judge questions, 0.04 s
```
<!-- END GENERATED: family-mothers -->

Of every way the knowledge base could say "X is the mother of Y", Jev accepts only "X gave birth to Y". It gives "X is a parent of Y" 0.03, and "Y is the son of X" 0.09:

<!-- BEGIN GENERATED: family-judged -->
```text
$ semrewrite rewrites examples/family_tree.metta mother_of --judge jev --threshold 0.65 --retriever all --all
admit   0.820  mother_of(X0, X1) ⇐ gave_birth_to(X0, X1)
reject  0.100  mother_of(X0, X1) ⇐ daughter_of(X1, X0)
reject  0.090  mother_of(X0, X1) ⇐ son_of(X1, X0)
reject  0.070  mother_of(X0, X1) ⇐ begat(X0, X1)
reject  0.040  mother_of(X0, X1) ⇐ legal_guardian_of(X0, X1)
reject  0.030  mother_of(X0, X1) ⇐ parent_of(X0, X1)
reject  0.020  mother_of(X0, X1) ⇐ begat(X1, X0)
reject  0.020  mother_of(X0, X1) ⇐ descendant_of(X1, X0)
reject  0.020  mother_of(X0, X1) ⇐ named_after(X1, X0)
reject  0.010  mother_of(X0, X1) ⇐ daughter_of(X0, X1)
reject  0.010  mother_of(X0, X1) ⇐ descendant_of(X0, X1)
reject  0.010  mother_of(X0, X1) ⇐ father_of(X0, X1)
reject  0.010  mother_of(X0, X1) ⇐ father_of(X1, X0)
reject  0.010  mother_of(X0, X1) ⇐ gave_birth_to(X1, X0)
reject  0.010  mother_of(X0, X1) ⇐ godfather_of(X0, X1)
reject  0.010  mother_of(X0, X1) ⇐ godfather_of(X1, X0)
reject  0.010  mother_of(X0, X1) ⇐ legal_guardian_of(X1, X0)
reject  0.010  mother_of(X0, X1) ⇐ named_after(X0, X1)
reject  0.010  mother_of(X0, X1) ⇐ parent_of(X1, X0)
reject  0.010  mother_of(X0, X1) ⇐ son_of(X0, X1)
reject  0.010  mother_of(X0, X1) ⇐ mother_of(X1, X0)
21 judge questions at threshold 0.650
```
<!-- END GENERATED: family-judged -->

The similarity judge gets both wrong, because it cannot tell which argument is which. It reads "liz is the daughter of tom" as "liz is a parent of tom", and "tom is a descendant of liz" as "liz is a descendant of tom", which together make Tom his own descendant; it finds no way to Ann, Pat and Jim. It reads "jim is the son of pat" as "jim is a parent of pat", and that as "pat is the mother of jim":

<!-- BEGIN GENERATED: family-similarity -->
```text
$ semrewrite ask examples/family_tree.metta 'descendant_of(X, tom)' --judge cosine-bge --strategy rewrite-depth-tabled --retriever all --engine pymetta --explain
descendant_of(bob, tom)
    descendant_of(bob, tom)    [by descendant_of(X, Y) :- parent_of(Y, X)]
      parent_of(tom, bob)    [by parent_of(X0, X1) ⇐ father_of(X0, X1), score 0.965]
        father_of(tom, bob)    [fact]
descendant_of(liz, tom)
    descendant_of(liz, tom)    [by descendant_of(X0, X1) ⇐ descendant_of(X1, X0), score 0.987]
      descendant_of(tom, liz)    [by descendant_of(X, Y) :- parent_of(Y, X)]
        parent_of(liz, tom)    [by parent_of(X0, X1) ⇐ daughter_of(X0, X1), score 0.963]
          daughter_of(liz, tom)    [fact]
descendant_of(tom, tom)
    descendant_of(tom, tom)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
      parent_of(bob, tom)    [by parent_of(X0, X1) ⇐ parent_of(X1, X0), score 0.990]
        parent_of(tom, bob)    [by parent_of(X0, X1) ⇐ father_of(X0, X1), score 0.965]
          father_of(tom, bob)    [fact]
      descendant_of(bob, tom)    [by descendant_of(X, Y) :- parent_of(Y, X)]
        parent_of(tom, bob)    [by parent_of(X0, X1) ⇐ father_of(X0, X1), score 0.965]
          father_of(tom, bob)    [fact]

rewrites admitted (21):
  daughter_of(X0, X1) ⇐ daughter_of(X1, X0)
  daughter_of(X0, X1) ⇐ mother_of(X0, X1)
  daughter_of(X0, X1) ⇐ parent_of(X0, X1)
  descendant_of(X0, X1) ⇐ descendant_of(X1, X0)
  father_of(X0, X1) ⇐ father_of(X1, X0)
  father_of(X0, X1) ⇐ parent_of(X0, X1)
  father_of(X0, X1) ⇐ parent_of(X1, X0)
  father_of(X0, X1) ⇐ son_of(X0, X1)
  father_of(X0, X1) ⇐ son_of(X1, X0)
  mother_of(X0, X1) ⇐ daughter_of(X0, X1)
  mother_of(X0, X1) ⇐ mother_of(X1, X0)
  mother_of(X0, X1) ⇐ parent_of(X0, X1)
  mother_of(X0, X1) ⇐ parent_of(X1, X0)
  parent_of(X0, X1) ⇐ daughter_of(X0, X1)
  parent_of(X0, X1) ⇐ father_of(X0, X1)
  parent_of(X0, X1) ⇐ mother_of(X0, X1)
  parent_of(X0, X1) ⇐ parent_of(X1, X0)
  parent_of(X0, X1) ⇐ son_of(X0, X1)
  son_of(X0, X1) ⇐ father_of(X0, X1)
  son_of(X0, X1) ⇐ parent_of(X0, X1)
  son_of(X0, X1) ⇐ son_of(X1, X0)
ok: 3 answers, 60,984 work, 126 judge questions, 0.18 s

$ semrewrite ask examples/family_tree.metta 'mother_of(X, jim)' --judge cosine-bge --strategy rewrite-depth-tabled --retriever all --engine pymetta --explain
mother_of(pat, jim)
    mother_of(pat, jim)    [by mother_of(X0, X1) ⇐ parent_of(X1, X0), score 0.960]
      parent_of(jim, pat)    [by parent_of(X0, X1) ⇐ son_of(X0, X1), score 0.960]
        son_of(jim, pat)    [fact]

rewrites admitted (20):
  daughter_of(X0, X1) ⇐ daughter_of(X1, X0)
  daughter_of(X0, X1) ⇐ mother_of(X0, X1)
  daughter_of(X0, X1) ⇐ parent_of(X0, X1)
  father_of(X0, X1) ⇐ father_of(X1, X0)
  father_of(X0, X1) ⇐ parent_of(X0, X1)
  father_of(X0, X1) ⇐ parent_of(X1, X0)
  father_of(X0, X1) ⇐ son_of(X0, X1)
  father_of(X0, X1) ⇐ son_of(X1, X0)
  mother_of(X0, X1) ⇐ daughter_of(X0, X1)
  mother_of(X0, X1) ⇐ mother_of(X1, X0)
  mother_of(X0, X1) ⇐ parent_of(X0, X1)
  mother_of(X0, X1) ⇐ parent_of(X1, X0)
  parent_of(X0, X1) ⇐ daughter_of(X0, X1)
  parent_of(X0, X1) ⇐ father_of(X0, X1)
  parent_of(X0, X1) ⇐ mother_of(X0, X1)
  parent_of(X0, X1) ⇐ parent_of(X1, X0)
  parent_of(X0, X1) ⇐ son_of(X0, X1)
  son_of(X0, X1) ⇐ father_of(X0, X1)
  son_of(X0, X1) ⇐ parent_of(X0, X1)
  son_of(X0, X1) ⇐ son_of(X1, X0)
ok: 1 answers, 5,005 work, 105 judge questions, 0.12 s
```
<!-- END GENERATED: family-similarity -->

## Running the examples

The commands need the PyMeTTa engine and, for the similarity judge, the models extra:

```sh
pip install ".[pymetta,models,jev]"
```

Jev needs a TypeSafe key in `TYPESAFE_API_KEY`, or the path of a file holding one in `JEV_KEY_FILE`. Without a key, `--judge oracle` answers from each file's gold lines, and `--judge nli-deberta-large` runs a local cross-encoder; RESULTS.md compares them. To run a knowledge base on the pure-Python engine instead, drop `--engine pymetta`; the answers are the same, and the derivations may differ where two are equally short.
