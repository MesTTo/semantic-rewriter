# Semantic chaining in MeTTa

Each example here is a knowledge base written in MeTTa and a question asked in words that none of its facts use. The rewriter asks a judge which statements entail which, and admits the rewrites the judge accepts. The PyMeTTa engine then evaluates the knowledge base's own rules together with those rewrites, with tables, and returns every answer with its shortest derivation. A derivation chains the judge's rewrites and the knowledge base's rules, and shows the judge's score beside each rewrite step.

The judge is Jev, TypeSafe's hosted model, at the threshold [RESULTS.md](../RESULTS.md) calibrated for answer F1 on every held-out benchmark, 0.65. The first example asks three questions of Wikidata, whose statements are spread over 1,790 properties. The other two start from programs in the MeTTa example corpus ([MesTTo/MeTTa-Examples](https://github.com/MesTTo/MeTTa-Examples)) and in PyMeTTa's own examples ([MesTTo/PyMeTTa](https://github.com/MesTTo/PyMeTTa/tree/main/examples)), and allow at most three rewrites below any answer (`rewrite-depth-tabled`), the bound RESULTS.md found best for Jev, at which it reached held-out F1 0.916.

Every transcript below was printed by the command above it, and every table by the script named above it. `python examples/walkthrough.py` runs them all again and rewrites this file.

## Asking Wikidata in your own words

Wikidata records how its items relate in 1,790 properties whose values are other items, and their names are not the words a person asks in. It has no property for descent, and it spreads parenthood over four: child (P40), stated on the parent, and father (P22), mother (P25) and parent (P8810), stated on the child. It files godparent, stepparent, spouse and sibling beside them, under relative. A work's maker is stated on the work, under creator (P170) or one of the 27 properties Wikidata files beneath it, such as author, architect and founder. Ownership is stated from both ends, as owner of (P1830) on the owner and owned by (P127) on what is owned, with child organization and parent organization beneath them.

Each question below is asked in its own words, with its own rules, and names no Wikidata property:

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

[build.py](wikidata/build.py) fetched each knowledge base from Wikidata's public endpoint. Each holds all 1,790 properties, glossed from their English labels (`child_p40` reads "{y} is the child of {x}"), and every statement Wikidata holds about the question's subject and about each entity in Wikidata's own answer. Leonardo's also holds the statements of everything else that names him, such as what is named after him or depicts him. Wikidata's own answer is what the question's relation reaches through the properties that Wikidata's subproperty hierarchy files the relation under, each read in its own direction, as [questions.py](wikidata/questions.py) defines it. The builder fails unless the statements it fetched reach exactly the answer Wikidata's endpoint computes from those properties. Beside each knowledge base, a `.ids.tsv` file gives every name's Wikidata id, so any answer or derivation can be checked on wikidata.org.

With 1,790 candidates for every predicate, Jev can't be asked about each one, and embedding retrieval, which the other examples use, misses the ones that matter: among the premises of "{x} created {y}" it ranked author 327th and notable work 794th ([DESIGN.md](../DESIGN.md#retriever)). So retrieval here is a cascade (`--retriever judge`). The cross-encoder nli-deberta-large, which runs locally, scores every property as a premise in both argument orders, and Jev judges the 50 it ranks highest.

[evaluate.py](wikidata/evaluate.py) asks each question without rewriting, then with the embedding judge cosine-bge and with Jev admitting rewrites, and scores every answer set against Wikidata's own answer. Each judge runs at the threshold RESULTS.md calibrated for answer F1: 0.65 for Jev, and 0.957 for cosine-bge, which is also its default. Each path from the question to a statement may take one rewrite. On Victoria's question both judges run again with chains of up to three rewrites, the bound the other examples use:

<!-- BEGIN GENERATED: wikidata-evaluation -->
### victoria: `descendant_of(X, queen_victoria)`

The knowledge base holds 23,116 statements, in 120 of the 1,790 properties it glosses.
Wikidata's own answer, through `child_p40`, `father_p22`, `mother_p25`: 1,164 entities.

| judge | most rewrites on a path | answers | in Wikidata's answer | not in it | missed | judge questions | rewrites admitted | seconds |
|---|---|---|---|---|---|---|---|---|
| none, read literally | none | 0 | 0 | 0 | 1,164 | 0 | 0 | 49.7 |
| cosine-bge | 1 | 1,165 | 1,164 | 1 | 0 | 202 | 8 | 44.9 |
| jev | 1 | 1,164 | 1,164 | 0 | 0 | 202 | 18 | 7.6 |
| cosine-bge | 3 | none, the work budget stopped it | 0 | 0 | 1,164 | 606 | 24 | 67.4 |
| jev | 3 | 1,191 | 1,164 | 27 | 0 | 5,454 | 184 | 1059.3 |

cosine-bge, at most 1 rewrite on a path: ok, 2,723,404 inferences.
Rewrites it admitted into the question's own predicates:

- `descendant_of` ⇐ `descendant_of` (arguments swapped) 0.99
- `parent_of` ⇐ `parent_p8810` (arguments swapped) 1.00
- `parent_of` ⇐ `parent_of` (arguments swapped) 0.99
- `parent_of` ⇐ `parent_p8810` 0.98
- `parent_of` ⇐ `mother_p25` (arguments swapped) 0.97
- `parent_of` ⇐ `child_p40` (arguments swapped) 0.97
- `parent_of` ⇐ `father_p22` (arguments swapped) 0.96
- `parent_of` ⇐ `child_p40` 0.96

Some answers not in Wikidata's: `queen_victoria`

jev, at most 1 rewrite on a path: ok, 2,459,364 inferences.
Rewrites it admitted into the question's own predicates:

- `descendant_of` ⇐ `parent_peak_p3137` 0.91
- `descendant_of` ⇐ `parent_club_p831` 0.87
- `descendant_of` ⇐ `gens_p5025` 0.83
- `descendant_of` ⇐ `derived_from_base_unit_p12571` 0.82
- `descendant_of` ⇐ `has_spin_off_p2512` (arguments swapped) 0.79
- `descendant_of` ⇐ `is_proceedings_from_p4745` 0.67
- `descendant_of` ⇐ `filial_church_p12476` (arguments swapped) 0.65
- `parent_of` ⇐ `parent_p8810` (arguments swapped) 1.00
- `parent_of` ⇐ `mother_p25` (arguments swapped) 0.99
- `parent_of` ⇐ `father_p22` (arguments swapped) 0.98
- `parent_of` ⇐ `child_p40` 0.97
- `parent_of` ⇐ `subdivision_of_this_unit_p9059` 0.87
- `parent_of` ⇐ `parent_peak_p3137` (arguments swapped) 0.83
- `parent_of` ⇐ `parent_cell_line_p3432` (arguments swapped) 0.80
- `parent_of` ⇐ `mother_house_p612` (arguments swapped) 0.77
- `parent_of` ⇐ `parent_organization_or_unit_p749` (arguments swapped) 0.73
- `parent_of` ⇐ `parent_astronomical_body_p397` (arguments swapped) 0.69
- `parent_of` ⇐ `parent_club_p831` (arguments swapped) 0.68

cosine-bge, at most 3 rewrites on a path: work_limit, 50,002,221 inferences.
Into the question's own predicates it admitted the same rewrites as above.

It chained 16 more rewrites into the properties those premises name.

jev, at most 3 rewrites on a path: ok, 34,403,137 inferences.
Into the question's own predicates it admitted the same rewrites as above.

It chained 166 more rewrites into the properties those premises name.

Some answers not in Wikidata's: `army_air_corps`, `auxiliary_territorial_service`, `blues_and_royals`, `bm_granollers`, `british_army`, `danish_defence`, `drammen_og_omegn_rk`, `fc_barcelona_handbol`

### leonardo: `created(leonardo_da_vinci, W)`

The knowledge base holds 13,291 statements, in 340 of the 1,790 properties it glosses.
Wikidata's own answer, through `architect_p84`, `author_p50`, `composer_p86`, `creator_p170`, `designed_by_p287`, `developer_p178`, `director_p57`, `exhibited_creator_p10661`, `founder_p112`, `illustrator_p110`, `landscape_architect_p7514`, `screenwriter_p58`: 854 entities.

| judge | most rewrites on a path | answers | in Wikidata's answer | not in it | missed | judge questions | rewrites admitted | seconds |
|---|---|---|---|---|---|---|---|---|
| none, read literally | none | 0 | 0 | 0 | 854 | 0 | 0 | 16.7 |
| cosine-bge | 1 | 0 | 0 | 0 | 854 | 101 | 1 | 16.0 |
| jev | 1 | 851 | 849 | 2 | 5 | 101 | 16 | 1.2 |

cosine-bge, at most 1 rewrite on a path: ok, 1,681 inferences.
Rewrites it admitted into the question's own predicates:

- `created` ⇐ `created` (arguments swapped) 0.96

Some of Wikidata's answers it missed: `a_copse_of_trees`, `a_hillside_with_an_outcrop_of_stratified_rock`, `a_marshy_plain_seen_from_above_and_notes`, `a_mechanical_military_drum`, `a_miracle_of_saint_donatus_of_arezzo`, `a_portrait_of_a_woman_in_profile`, `a_ravine`, `a_rider_at_a_gallop_a_horse_s_head_in_profile_and_a_horse_viewed_from_behind`

jev, at most 1 rewrite on a path: ok, 34,633 inferences.
Rewrites it admitted into the question's own predicates:

- `created` ⇐ `creator_p170` (arguments swapped) 0.92
- `created` ⇐ `produced_by_p2849` (arguments swapped) 0.89
- `created` ⇐ `founder_p112` (arguments swapped) 0.88
- `created` ⇐ `collection_creator_p6241` (arguments swapped) 0.87
- `created` ⇐ `developer_p178` (arguments swapped) 0.87
- `created` ⇐ `exhibited_creator_p10661` (arguments swapped) 0.85
- `created` ⇐ `calligrapher_p6819` (arguments swapped) 0.84
- `created` ⇐ `author_p50` (arguments swapped) 0.83
- `created` ⇐ `composer_p86` (arguments swapped) 0.81
- `created` ⇐ `form_of_creative_work_p7937` 0.73
- `created` ⇐ `architect_p84` (arguments swapped) 0.71
- `created` ⇐ `manufacturer_p176` (arguments swapped) 0.71
- `created` ⇐ `producer_p162` (arguments swapped) 0.71
- `created` ⇐ `notable_work_p800` 0.70
- `created` ⇐ `in_work_p2553` 0.69
- `created` ⇐ `fabrication_method_p2079` (arguments swapped) 0.66

Some answers not in Wikidata's: `annunciation_q154326`, `copy_of_the_leonardo_da_vinci_s_last_supper`

Some of Wikidata's answers it missed: `de_divina_proportione`, `leonardo_da_vinci_s_catapult`, `porto_canale_leonardesco`, `q121978147`, `q122017993`

### alphabet: `owns(alphabet_inc, C)`

The knowledge base holds 18,841 statements, in 162 of the 1,790 properties it glosses.
Wikidata's own answer, through `child_organization_or_unit_p355`, `copyright_holder_p3931`, `owned_by_p127`, `owner_of_p1830`, `parent_organization_or_unit_p749`: 3,225 entities.

| judge | most rewrites on a path | answers | in Wikidata's answer | not in it | missed | judge questions | rewrites admitted | seconds |
|---|---|---|---|---|---|---|---|---|
| none, read literally | none | 0 | 0 | 0 | 3,225 | 0 | 0 | 100.5 |
| cosine-bge | 1 | none, the work budget stopped it | 0 | 0 | 3,225 | 101 | 8 | 25.3 |
| jev | 1 | 2,849 | 2,849 | 0 | 376 | 101 | 3 | 2.7 |

cosine-bge, at most 1 rewrite on a path: work_limit, 50,000,892 inferences.
Rewrites it admitted into the question's own predicates:

- `owns` ⇐ `owns` (arguments swapped) 0.99
- `owns` ⇐ `owned_by_p127` 0.99
- `owns` ⇐ `owner_of_p1830` 0.98
- `owns` ⇐ `owner_of_p1830` (arguments swapped) 0.97
- `owns` ⇐ `owned_by_p127` (arguments swapped) 0.97
- `owns` ⇐ `afterward_owned_by_p11812` 0.97
- `owns` ⇐ `beforehand_owned_by_p11811` 0.97
- `owns` ⇐ `beforehand_owned_by_p11811` (arguments swapped) 0.96

jev, at most 1 rewrite on a path: ok, 1,268,948 inferences.
Rewrites it admitted into the question's own predicates:

- `owns` ⇐ `owned_by_p127` (arguments swapped) 0.99
- `owns` ⇐ `owner_of_p1830` 0.99
- `owns` ⇐ `afterward_owned_by_p11812` (arguments swapped) 0.86

Some of Wikidata's answers it missed: `accessing_a_local_data_network_via_a_mobile_data_connection`, `actions_on_google`, `adaptive_interface_switching`, `adaptive_user_interface_display_size_for_curved_display_edges`, `adaptive_video_interface_switching`, `adhoc_geo_fiducial_mats_for_landing_uavs`, `admob`, `andsf_and_lwa_coexistence`
<!-- END GENERATED: wikidata-evaluation -->

Read literally, every question finds nothing, since no statement uses its words. Those runs take 17 to 100 s because PyMeTTa's native encoding holds the facts as equations, which pymetta 0.9.4 takes superquadratic time to drop when a context closes. In a literal read of Alphabet's knowledge base timed on its own, closing took 61.5 of its 65 s ([DESIGN.md](../DESIGN.md#engine)). The other times include discovery. The cross-encoder ranks a predicate's candidates the first time any run expands it, so the first judged run of each question, the embedding judge's, also ranks them for the runs after it.

With one rewrite on a path, Jev returned exactly the 1,164 people Wikidata records as Victoria's descendants. It admitted parent, mother and father with the arguments swapped, and child as stated, at 0.97 to 1.00. It also admitted seven properties that name other kinds of parent, such as subdivision of this unit at 0.87, parent peak at 0.83 and parent club at 0.68, and seven rewrites into the question's own `descendant_of`, such as one from parent peak at 0.91. None of them adds an answer, because this knowledge base holds no statement in any of those properties.

For Leonardo it returned 849 of the 854 works Wikidata attributes to him, and 2 more. The five it missed are connected to him only by illustrator, landscape architect or designed by, which Wikidata files under creator. Jev scored illustrator at 0.24 and designed by at 0.57, below its threshold, and landscape architect wasn't among the 50 candidates it saw. The two extra are works that his own entry lists as notable and that no creator statement attributes to him: a copy of The Last Supper and an Annunciation.

For Alphabet it returned 2,849 of Wikidata's 3,225 holdings, and nothing else. It admitted owned by, with the arguments swapped, and owner of, at 0.99. It scored copyright holder at 0.22, and the cross-encoder ranked parent organization 68th and child organization 555th, so Jev never saw those two. Wikidata files all three under ownership, and the 376 holdings Jev missed hang from 104 that only those three properties connect to the rest, such as AdMob and Apigee, whose parent organization is Google, and Calculator and Clock, whose copyright Google holds.

The embedding judge can't tell which argument is which. On Victoria's question it admitted child in both directions, and read backwards, "Princess Helena is the child of Queen Victoria" makes Helena Victoria's parent, so it returned Victoria as her own descendant. On Leonardo's it admitted only `created` itself with the arguments swapped, scoring creator and the other 49 candidates below its threshold, and found none of the 854 works. On Alphabet's it admitted owned by, owner of and beforehand owned by in both directions. With the question's transitive rule, that joins every company to every other, and the work budget stopped it before it returned anything.

With chains of up to three rewrites on Victoria's question, Jev returned the same 1,164 people and 27 more, each an armed force, a military unit or a sports club, and asked 5,454 questions instead of 202. It expanded each premise it admitted in turn, adding 166 rewrites into Wikidata's own properties, and two misreadings in a row are enough to make the Army Air Corps a descendant of Victoria, as the derivation at the end of this section shows. The embedding judge, given the same chains, spent its budget without an answer.

Each answer comes with its derivation, the statements and rewrites it rests on, each rewrite with Jev's score:

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

<!-- BEGIN GENERATED: wikidata-work -->
```text
$ semrewrite ask examples/wikidata/leonardo.metta 'created(leonardo_da_vinci, mona_lisa)' --judge jev --threshold 0.65 --strategy rewrite-depth-tabled --depth 1 --retriever judge --k 50 --engine pymetta --work 50000000 --seconds inf --explain
created(leonardo_da_vinci, mona_lisa)
    created(leonardo_da_vinci, mona_lisa)    [by created(X0, X1) ⇐ creator_p170(X1, X0), score 0.920]
      creator_p170(mona_lisa, leonardo_da_vinci)    [fact]

rewrites admitted (16):
  created(X0, X1) ⇐ architect_p84(X1, X0)
  created(X0, X1) ⇐ author_p50(X1, X0)
  created(X0, X1) ⇐ calligrapher_p6819(X1, X0)
  created(X0, X1) ⇐ collection_creator_p6241(X1, X0)
  created(X0, X1) ⇐ composer_p86(X1, X0)
  created(X0, X1) ⇐ creator_p170(X1, X0)
  created(X0, X1) ⇐ developer_p178(X1, X0)
  created(X0, X1) ⇐ exhibited_creator_p10661(X1, X0)
  created(X0, X1) ⇐ fabrication_method_p2079(X1, X0)
  created(X0, X1) ⇐ form_of_creative_work_p7937(X0, X1)
  created(X0, X1) ⇐ founder_p112(X1, X0)
  created(X0, X1) ⇐ in_work_p2553(X0, X1)
  created(X0, X1) ⇐ manufacturer_p176(X1, X0)
  created(X0, X1) ⇐ notable_work_p800(X0, X1)
  created(X0, X1) ⇐ produced_by_p2849(X1, X0)
  created(X0, X1) ⇐ producer_p162(X1, X0)
ok: 1 answer, 3,169 work, 101 judge questions, 26.76 s
```
<!-- END GENERATED: wikidata-work -->

<!-- BEGIN GENERATED: wikidata-holding -->
```text
$ semrewrite ask examples/wikidata/alphabet.metta 'owns(alphabet_inc, youtube_q866)' --judge jev --threshold 0.65 --strategy rewrite-depth-tabled --depth 1 --retriever judge --k 50 --engine pymetta --work 50000000 --seconds inf --explain
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
ok: 1 answer, 2,294,032 work, 101 judge questions, 37.30 s
```
<!-- END GENERATED: wikidata-holding -->

King Frederik X of Denmark descends from Victoria through his mother Margrethe II, her mother Ingrid of Sweden, Ingrid's mother Princess Margaret of Connaught, and Margaret's father Prince Arthur, Victoria's son. The Mona Lisa names Leonardo as its creator. Alphabet owns YouTube through XXVI Holdings and Google. Wikidata states the first and last links on what is owned, as owned by, and the middle one on the owner, as owner of.

<details>
<summary>How chains of three rewrites made the Army Air Corps a descendant of Queen Victoria</summary>

Jev read "the Army Air Corps is the military branch of Prince Harry" as "Prince Harry is the parent organization of the Army Air Corps", at 0.88, and that as "Prince Harry is a parent of the Army Air Corps", at 0.73. With one rewrite on a path, the second can't build on the first, and the knowledge base holds no parent organization statement for it to read directly.

<!-- BEGIN GENERATED: wikidata-chained -->
```text
$ semrewrite ask examples/wikidata/victoria.metta 'descendant_of(army_air_corps, queen_victoria)' --judge jev --threshold 0.65 --strategy rewrite-depth-tabled --depth 3 --retriever judge --k 50 --engine pymetta --work 50000000 --seconds inf --explain
descendant_of(army_air_corps, queen_victoria)
    descendant_of(army_air_corps, queen_victoria)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
      parent_of(prince_harry_duke_of_sussex, army_air_corps)    [by parent_of(X0, X1) ⇐ parent_organization_or_unit_p749(X1, X0), score 0.730]
        parent_organization_or_unit_p749(army_air_corps, prince_harry_duke_of_sussex)    [by parent_organization_or_unit_p749(X0, X1) ⇐ military_branch_p241(X1, X0), score 0.880]
          military_branch_p241(prince_harry_duke_of_sussex, army_air_corps)    [fact]
      descendant_of(prince_harry_duke_of_sussex, queen_victoria)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
        parent_of(charles_iii_q43274, prince_harry_duke_of_sussex)    [by parent_of(X0, X1) ⇐ father_p22(X1, X0), score 0.980]
          father_p22(prince_harry_duke_of_sussex, charles_iii_q43274)    [fact]
        descendant_of(charles_iii_q43274, queen_victoria)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
          parent_of(prince_philip_duke_of_edinburgh, charles_iii_q43274)    [by parent_of(X0, X1) ⇐ father_p22(X1, X0), score 0.980]
            father_p22(charles_iii_q43274, prince_philip_duke_of_edinburgh)    [fact]
          descendant_of(prince_philip_duke_of_edinburgh, queen_victoria)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
            parent_of(princess_alice_of_battenberg, prince_philip_duke_of_edinburgh)    [by parent_of(X0, X1) ⇐ mother_p25(X1, X0), score 0.990]
              mother_p25(prince_philip_duke_of_edinburgh, princess_alice_of_battenberg)    [fact]
            descendant_of(princess_alice_of_battenberg, queen_victoria)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
              parent_of(princess_victoria_of_hesse_and_by_rhine, princess_alice_of_battenberg)    [by parent_of(X0, X1) ⇐ mother_p25(X1, X0), score 0.990]
                mother_p25(princess_alice_of_battenberg, princess_victoria_of_hesse_and_by_rhine)    [fact]
              descendant_of(princess_victoria_of_hesse_and_by_rhine, queen_victoria)    [by descendant_of(X, Z) :- parent_of(Y, X), descendant_of(Y, Z)]
                parent_of(princess_alice_of_the_united_kingdom, princess_victoria_of_hesse_and_by_rhine)    [by parent_of(X0, X1) ⇐ mother_p25(X1, X0), score 0.990]
                  mother_p25(princess_victoria_of_hesse_and_by_rhine, princess_alice_of_the_united_kingdom)    [fact]
                descendant_of(princess_alice_of_the_united_kingdom, queen_victoria)    [by descendant_of(X, Y) :- parent_of(Y, X)]
                  parent_of(queen_victoria, princess_alice_of_the_united_kingdom)    [by parent_of(X0, X1) ⇐ mother_p25(X1, X0), score 0.990]
                    mother_p25(princess_alice_of_the_united_kingdom, queen_victoria)    [fact]

rewrites admitted (184):
  by_product_of_p2822(X0, X1) ⇐ by_product_p2821(X1, X0)
  by_product_of_p2822(X0, X1) ⇐ has_spin_off_p2512(X1, X0)
  by_product_p2821(X0, X1) ⇐ has_spin_off_p2512(X0, X1)
  by_product_p2821(X0, X1) ⇐ spore_print_color_p787(X0, X1)
  child_astronomical_body_p398(X0, X1) ⇐ parent_astronomical_body_p397(X1, X0)
  child_organization_or_unit_p355(X0, X1) ⇐ category_contains_p4224(X1, X0)
  child_organization_or_unit_p355(X0, X1) ⇐ cleavage_p693(X0, X1)
  child_organization_or_unit_p355(X0, X1) ⇐ filial_church_p12476(X0, X1)
  child_organization_or_unit_p355(X0, X1) ⇐ gens_p5025(X1, X0)
  child_organization_or_unit_p355(X0, X1) ⇐ has_spin_off_p2512(X0, X1)
  child_organization_or_unit_p355(X0, X1) ⇐ military_branch_p241(X0, X1)
  child_organization_or_unit_p355(X0, X1) ⇐ parent_club_p831(X1, X0)
  child_organization_or_unit_p355(X0, X1) ⇐ youth_wing_p4379(X0, X1)
  child_p40(X0, X1) ⇐ child_astronomical_body_p398(X0, X1)
  child_p40(X0, X1) ⇐ child_organization_or_unit_p355(X0, X1)
  child_p40(X0, X1) ⇐ father_p22(X1, X0)
  child_p40(X0, X1) ⇐ mother_p25(X1, X0)
  child_p40(X0, X1) ⇐ parent_astronomical_body_p397(X1, X0)
  child_p40(X0, X1) ⇐ parent_cell_line_p3432(X1, X0)
  child_p40(X0, X1) ⇐ parent_club_p831(X1, X0)
  child_p40(X0, X1) ⇐ parent_of(X0, X1)
  child_p40(X0, X1) ⇐ parent_p8810(X1, X0)
  child_p40(X0, X1) ⇐ parent_peak_p3137(X1, X0)
  child_p40(X0, X1) ⇐ parent_taxon_p171(X1, X0)
  child_p40(X0, X1) ⇐ patronym_or_matronym_p5056(X1, X0)
  creator_p170(X0, X1) ⇐ architect_p84(X0, X1)
  creator_p170(X0, X1) ⇐ author_p50(X0, X1)
  creator_p170(X0, X1) ⇐ central_bank_issuer_p562(X0, X1)
  creator_p170(X0, X1) ⇐ composer_p86(X0, X1)
  creator_p170(X0, X1) ⇐ created_during_p10408(X1, X0)
  creator_p170(X0, X1) ⇐ designed_by_p287(X0, X1)
  creator_p170(X0, X1) ⇐ developer_p178(X0, X1)
  creator_p170(X0, X1) ⇐ exhibited_creator_p10661(X0, X1)
  creator_p170(X0, X1) ⇐ form_of_creative_work_p7937(X1, X0)
  creator_p170(X0, X1) ⇐ founder_p112(X0, X1)
  creator_p170(X0, X1) ⇐ in_work_p2553(X1, X0)
  creator_p170(X0, X1) ⇐ notable_work_p800(X1, X0)
  creator_p170(X0, X1) ⇐ produced_by_p2849(X0, X1)
  creator_p170(X0, X1) ⇐ producer_p162(X0, X1)
  creator_p170(X0, X1) ⇐ quotes_work_p6166(X1, X0)
  decays_to_p816(X0, X1) ⇐ decay_mode_p817(X0, X1)
  decays_to_p816(X0, X1) ⇐ deterioration_p12680(X0, X1)
  derived_from_base_unit_p12571(X0, X1) ⇐ gens_p5025(X0, X1)
  derived_from_base_unit_p12571(X0, X1) ⇐ has_spin_off_p2512(X1, X0)
  derived_from_base_unit_p12571(X0, X1) ⇐ radix_p3264(X0, X1)
  derived_from_base_unit_p12571(X0, X1) ⇐ raw_material_processed_p10672(X0, X1)
  derived_from_base_unit_p12571(X0, X1) ⇐ this_taxon_is_source_of_p1672(X1, X0)
  derived_from_base_unit_p12571(X0, X1) ⇐ twinning_p537(X1, X0)
  derived_from_organism_type_p9072(X0, X1) ⇐ autologous_cell_line_p3578(X1, X0)
  derived_from_organism_type_p9072(X0, X1) ⇐ eusocial_caste_p14415(X1, X0)
  derived_from_organism_type_p9072(X0, X1) ⇐ harvested_organism_s_p10960(X1, X0)
  derived_from_organism_type_p9072(X0, X1) ⇐ hybrid_of_p1531(X0, X1)
  derived_from_organism_type_p9072(X0, X1) ⇐ inflorescence_p3739(X1, X0)
  derived_from_organism_type_p9072(X0, X1) ⇐ is_parent_form_of_the_active_substance_p12099(X1, X0)
  derived_from_organism_type_p9072(X0, X1) ⇐ juvenile_form_of_p14990(X0, X1)
  derived_from_organism_type_p9072(X0, X1) ⇐ mother_p25(X0, X1)
  descendant_of(X0, X1) ⇐ derived_from_base_unit_p12571(X0, X1)
  descendant_of(X0, X1) ⇐ filial_church_p12476(X1, X0)
  descendant_of(X0, X1) ⇐ gens_p5025(X0, X1)
  descendant_of(X0, X1) ⇐ has_spin_off_p2512(X1, X0)
  descendant_of(X0, X1) ⇐ is_proceedings_from_p4745(X0, X1)
  descendant_of(X0, X1) ⇐ parent_club_p831(X0, X1)
  descendant_of(X0, X1) ⇐ parent_peak_p3137(X0, X1)
  develops_from_p3094(X0, X1) ⇐ derivative_work_p4969(X1, X0)
  develops_from_p3094(X0, X1) ⇐ has_spin_off_p2512(X1, X0)
  develops_from_p3094(X0, X1) ⇐ inflorescence_p3739(X1, X0)
  develops_from_p3094(X0, X1) ⇐ location_of_formation_p740(X0, X1)
  develops_from_p3094(X0, X1) ⇐ seed_p5828(X0, X1)
  distributary_p5998(X0, X1) ⇐ distributed_from_p10754(X1, X0)
  filial_church_p12476(X0, X1) ⇐ gens_p5025(X1, X0)
  generalization_of_p7719(X0, X1) ⇐ broader_concept_p4900(X1, X0)
  generalization_of_p7719(X0, X1) ⇐ category_contains_p4224(X1, X0)
  generalization_of_p7719(X0, X1) ⇐ gens_p5025(X1, X0)
  generalization_of_p7719(X0, X1) ⇐ meta_category_for_p14387(X0, X1)
  generalization_of_p7719(X0, X1) ⇐ populates_category_p9926(X0, X1)
  gens_p5025(X0, X1) ⇐ general_secretary_p3975(X0, X1)
  gens_p5025(X0, X1) ⇐ generalization_of_p7719(X1, X0)
  gens_p5025(X0, X1) ⇐ grammatical_gender_p5185(X0, X1)
  gens_p5025(X0, X1) ⇐ in_work_p2553(X1, X0)
  gens_p5025(X0, X1) ⇐ is_proceedings_from_p4745(X0, X1)
  gens_p5025(X0, X1) ⇐ member_of_roman_tribe_p11491(X1, X0)
  grammatical_gender_p5185(X0, X1) ⇐ has_grammatical_gender_p5109(X0, X1)
  has_spin_off_p2512(X0, X1) ⇐ by_product_of_p2822(X1, X0)
  has_spin_off_p2512(X0, X1) ⇐ by_product_p2821(X0, X1)
  has_spin_off_p2512(X0, X1) ⇐ is_proceedings_from_p4745(X1, X0)
  inflorescence_p3739(X0, X1) ⇐ tala_p8535(X0, X1)
  is_proceedings_from_p4745(X0, X1) ⇐ by_product_of_p2822(X0, X1)
  is_proceedings_from_p4745(X0, X1) ⇐ by_product_p2821(X1, X0)
  is_proceedings_from_p4745(X0, X1) ⇐ creator_p170(X0, X1)
  is_proceedings_from_p4745(X0, X1) ⇐ decays_to_p816(X1, X0)
  is_proceedings_from_p4745(X0, X1) ⇐ derived_from_base_unit_p12571(X0, X1)
  is_proceedings_from_p4745(X0, X1) ⇐ derived_from_organism_type_p9072(X0, X1)
  is_proceedings_from_p4745(X0, X1) ⇐ develops_from_p3094(X0, X1)
  is_proceedings_from_p4745(X0, X1) ⇐ gens_p5025(X0, X1)
  is_proceedings_from_p4745(X0, X1) ⇐ manufacturer_p176(X0, X1)
  is_proceedings_from_p4745(X0, X1) ⇐ raw_material_processed_p10672(X0, X1)
  is_proceedings_from_p4745(X0, X1) ⇐ spoken_by_p10894(X0, X1)
  manufacturer_p176(X0, X1) ⇐ created_during_p10408(X0, X1)
  manufacturer_p176(X0, X1) ⇐ type_foundry_p4586(X0, X1)
  member_of_sports_team_p54(X0, X1) ⇐ team_captain_p634(X0, X1)
  mod_of_p7075(X0, X1) ⇐ inhibitor_of_p3776(X0, X1)
  mod_of_p7075(X0, X1) ⇐ media_modality_p12548(X1, X0)
  mother_house_p612(X0, X1) ⇐ filial_church_p12476(X1, X0)
  mother_house_p612(X0, X1) ⇐ has_spin_off_p2512(X1, X0)
  organizational_division_p199(X0, X1) ⇐ military_branch_p241(X0, X1)
  organizational_division_p199(X0, X1) ⇐ parent_organization_or_unit_p749(X1, X0)
  organizational_division_p199(X0, X1) ⇐ vice_county_p1887(X0, X1)
  organizational_division_p199(X0, X1) ⇐ youth_wing_p4379(X0, X1)
  parent_astronomical_body_p397(X0, X1) ⇐ asteroid_family_p744(X1, X0)
  parent_astronomical_body_p397(X0, X1) ⇐ child_astronomical_body_p398(X1, X0)
  parent_astronomical_body_p397(X0, X1) ⇐ tala_p8535(X1, X0)
  parent_cell_line_p3432(X0, X1) ⇐ gens_p5025(X0, X1)
  parent_cell_line_p3432(X0, X1) ⇐ kigo_of_p13504(X0, X1)
  parent_club_p831(X0, X1) ⇐ youth_wing_p4379(X1, X0)
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
  parent_organization_or_unit_p749(X0, X1) ⇐ distributary_p5998(X1, X0)
  parent_organization_or_unit_p749(X0, X1) ⇐ house_publication_p2813(X1, X0)
  parent_organization_or_unit_p749(X0, X1) ⇐ inflorescence_p3739(X1, X0)
  parent_organization_or_unit_p749(X0, X1) ⇐ kigo_of_p13504(X0, X1)
  parent_organization_or_unit_p749(X0, X1) ⇐ member_of_sports_team_p54(X1, X0)
  parent_organization_or_unit_p749(X0, X1) ⇐ military_branch_p241(X1, X0)
  parent_organization_or_unit_p749(X0, X1) ⇐ mod_of_p7075(X0, X1)
  parent_organization_or_unit_p749(X0, X1) ⇐ organizational_division_p199(X1, X0)
  parent_organization_or_unit_p749(X0, X1) ⇐ parent_club_p831(X0, X1)
  parent_organization_or_unit_p749(X0, X1) ⇐ youth_wing_p4379(X1, X0)
  parent_p8810(X0, X1) ⇐ child_p40(X1, X0)
  parent_p8810(X0, X1) ⇐ derived_from_base_unit_p12571(X0, X1)
  parent_p8810(X0, X1) ⇐ distributary_p5998(X1, X0)
  parent_p8810(X0, X1) ⇐ father_p22(X0, X1)
  parent_p8810(X0, X1) ⇐ gens_p5025(X0, X1)
  parent_p8810(X0, X1) ⇐ inflorescence_p3739(X1, X0)
  parent_p8810(X0, X1) ⇐ mother_house_p612(X0, X1)
  parent_p8810(X0, X1) ⇐ mother_p25(X0, X1)
  parent_p8810(X0, X1) ⇐ parent_astronomical_body_p397(X0, X1)
  parent_p8810(X0, X1) ⇐ parent_cell_line_p3432(X0, X1)
  parent_p8810(X0, X1) ⇐ parent_club_p831(X0, X1)
  parent_p8810(X0, X1) ⇐ parent_of(X1, X0)
  parent_p8810(X0, X1) ⇐ parent_organization_or_unit_p749(X0, X1)
  parent_p8810(X0, X1) ⇐ parent_peak_p3137(X0, X1)
  parent_p8810(X0, X1) ⇐ patronym_or_matronym_p5056(X0, X1)
  parent_p8810(X0, X1) ⇐ reissue_of_p9237(X0, X1)
  parent_p8810(X0, X1) ⇐ subclass_of_p279(X0, X1)
  parent_p8810(X0, X1) ⇐ subdivision_of_this_unit_p9059(X1, X0)
  parent_peak_p3137(X0, X1) ⇐ neoschizomer_p4875(X1, X0)
  parent_peak_p3137(X0, X1) ⇐ pendant_of_p1639(X0, X1)
  parent_taxon_p171(X0, X1) ⇐ asteroid_family_p744(X1, X0)
  parent_taxon_p171(X0, X1) ⇐ gens_p5025(X0, X1)
  parent_taxon_p171(X0, X1) ⇐ inflorescence_p3739(X1, X0)
  parent_taxon_p171(X0, X1) ⇐ kigo_of_p13504(X0, X1)
  parent_taxon_p171(X0, X1) ⇐ natural_product_of_taxon_p1582(X1, X0)
  parent_taxon_p171(X0, X1) ⇐ populates_category_p9926(X1, X0)
  parent_taxon_p171(X0, X1) ⇐ subclass_of_p279(X0, X1)
  patronym_or_matronym_p5056(X0, X1) ⇐ patronym_or_matronym_for_this_name_p2976(X0, X1)
  radix_p3264(X0, X1) ⇐ based_on_p144(X0, X1)
  radix_p3264(X0, X1) ⇐ radical_p5280(X0, X1)
  spoken_by_p10894(X0, X1) ⇐ native_language_p103(X1, X0)
  spoken_by_p10894(X0, X1) ⇐ speaker_p823(X0, X1)
  spoken_by_p10894(X0, X1) ⇐ terminal_speaker_p13464(X0, X1)
  spoken_by_p10894(X0, X1) ⇐ voice_actor_p725(X0, X1)
  subdivision_of_this_unit_p9059(X0, X1) ⇐ mother_house_p612(X1, X0)
  this_taxon_is_source_of_p1672(X0, X1) ⇐ basionym_of_p12766(X0, X1)
  this_taxon_is_source_of_p1672(X0, X1) ⇐ by_product_of_p2822(X1, X0)
  this_taxon_is_source_of_p1672(X0, X1) ⇐ by_product_p2821(X0, X1)
  this_taxon_is_source_of_p1672(X0, X1) ⇐ dialect_of_p4913(X1, X0)
  this_taxon_is_source_of_p1672(X0, X1) ⇐ distributary_p5998(X0, X1)
  this_taxon_is_source_of_p1672(X0, X1) ⇐ distributed_from_p10754(X1, X0)
  this_taxon_is_source_of_p1672(X0, X1) ⇐ harvested_organism_s_p10960(X0, X1)
  this_taxon_is_source_of_p1672(X0, X1) ⇐ natural_product_of_taxon_p1582(X0, X1)
  this_taxon_is_source_of_p1672(X0, X1) ⇐ outflows_p201(X0, X1)
  this_taxon_is_source_of_p1672(X0, X1) ⇐ produced_by_p2849(X1, X0)
  this_taxon_is_source_of_p1672(X0, X1) ⇐ product_or_material_produced_p1056(X0, X1)
  this_taxon_is_source_of_p1672(X0, X1) ⇐ protonym_of_p12765(X0, X1)
  twinning_p537(X0, X1) ⇐ permanent_duplicated_item_p2959(X0, X1)
  twinning_p537(X0, X1) ⇐ twinning_p537(X1, X0)
ok: 1 answer, 794,410 work, 5,454 judge questions, 1333.21 s
```
<!-- END GENERATED: wikidata-chained -->

</details>

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
ok: 1 answer, 1,993 work, 42 judge questions, 1.28 s

$ semrewrite ask examples/family_tree.metta 'mother_of(X, jim)' --judge jev --threshold 0.65 --strategy rewrite-depth-tabled --retriever all --engine pymetta --explain

rewrites admitted (1):
  mother_of(X0, X1) ⇐ gave_birth_to(X0, X1)
ok: 0 answers, 1,868 work, 42 judge questions, 0.17 s
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
ok: 3 answers, 70,332 work, 126 judge questions, 0.49 s

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
ok: 1 answer, 5,526 work, 105 judge questions, 0.85 s
```
<!-- END GENERATED: family-similarity -->

## Running the examples

The commands need the PyMeTTa engine and, for the similarity judge, the models extra:

```sh
pip install ".[pymetta,models,jev]"
```

Jev needs a TypeSafe key in `TYPESAFE_API_KEY`, or the path of a file holding one in `JEV_KEY_FILE`. Without a key, `--judge oracle` answers from each file's gold lines, and `--judge nli-deberta-large` runs a local cross-encoder; RESULTS.md compares them. The Wikidata knowledge bases have no gold lines, and `tests/test_examples.py` checks them with an oracle that admits exactly the rewrites Wikidata's hierarchy names, under which both engines return Wikidata's own answer to every question. To run a knowledge base on the pure-Python engine instead, drop `--engine pymetta`; the answers are the same, and the derivations may differ where two are equally short.

`python examples/wikidata/build.py` fetches the Wikidata knowledge bases again from Wikidata's public endpoint, and `python examples/wikidata/evaluate.py` prints their evaluation. The cascade's cross-encoder runs on a GPU if one is present.
