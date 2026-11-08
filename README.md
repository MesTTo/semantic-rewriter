# Semantic rewriter

A logic program answers `parent(X, tom)` only from clauses whose head says `parent`. When the facts say `mum(alice, tom)` and `dad(bob, tom)` instead, it finds nothing. The semantic rewriter fixes that without changing how the program is evaluated. When evaluation first reaches a predicate, it asks an entailment model which other predicates imply it and in which argument order, admits the implications the model trusts as ordinary rules, such as `parent(X, Y) :- dad(X, Y)` or `parent(X, Y) :- child_of(Y, X)`, and lets the engine carry on.

The rewriter runs on any logic engine that implements a small protocol. Two ship with it: a pure-Python engine with no dependencies, and [PyMeTTa](https://github.com/MesTTo/PyMeTTa) (MeTTa on SWI-Prolog). With tabling, every query terminates and gets exactly the answers its admitted rules imply; [DESIGN.md](DESIGN.md) states the conditions and the protocols, and [RESULTS.md](RESULTS.md) the experiment behind the defaults.

## Example

`examples/parent_of_tom.kb` states Tom's family in words that never say "parent":

```
gloss mum "{x} is the mum of {y}"
gloss dad "{x} is the dad of {y}"
gloss parent "{x} is a parent of {y}"
gloss child_of "{x} is a child of {y}"
gloss guardian "{x} is the legal guardian of {y}"
gloss grandparent "{x} is a grandparent of {y}"
...
mum(alice, tom).
dad(bob, tom).
mother(carol, alice).
child_of(bob, dave).
guardian(erin, tom).
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

Each predicate has a gloss, a sentence with a placeholder per argument, which is what the entailment model reads. Asking for Tom's parents with an NLI cross-encoder as the judge, at the threshold it was calibrated to on labelled predicate pairs:

```
$ semrewrite ask examples/parent_of_tom.kb "parent(X, tom)" --judge nli-deberta-large --retriever all --explain
parent(alice, tom)
    parent(alice, tom)    [by parent(X0, X1) ⇐ mum(X0, X1)]
      mum(alice, tom)    [fact]
parent(bob, tom)
    parent(bob, tom)    [by parent(X0, X1) ⇐ dad(X0, X1)]
      dad(bob, tom)    [fact]
parent(erin, tom)
    parent(erin, tom)    [by parent(X0, X1) ⇐ guardian(X0, X1)]
      guardian(erin, tom)    [fact]

rewrites admitted (14):
  child_of(X0, X1) ⇐ dad(X1, X0)
  mother(X0, X1) ⇐ child_of(X1, X0)
  parent(X0, X1) ⇐ guardian(X0, X1)
  ...
ok: 3 answers, 63 work, 78 judge questions, 0.65 s
```

Every answer comes with its derivation, and each step names the rule or the admitted rewrite it used. The third answer is wrong: a legal guardian need not be a parent, and at this threshold the judge admits `guardian ⇒ parent` along with other false rewrites such as `child_of(y, x) ⇒ mother(x, y)`. Raising the threshold to 0.995, which calibration for answer F1 chose for lazy-tabled evaluation on three of the experiment's four held-out benchmarks, leaves six rewrites, all true, and the answers alice and bob:

```
$ semrewrite ask examples/parent_of_tom.kb "parent(X, tom)" --judge nli-deberta-large --retriever all --threshold 0.995
parent(alice, tom)
parent(bob, tom)
ok: 2 answers, 14 work, 65 judge questions, 0.88 s
```

The grandparent query composes rewrites with the knowledge base's own rule, `grandparent(carol, tom)` through `mother(carol, alice)` and `mum(alice, tom)`, and `grandparent(dave, tom)` through `child_of(bob, dave)` read backwards and `dad(bob, tom)`. That trade-off between admitting enough and admitting only what is true, and how to calibrate a judge for it, is most of what [RESULTS.md](RESULTS.md) measures.

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
semrewrite ask KB QUERY [--judge NAME] [--engine python|pymetta] [--strategy NAME] [--explain] [--json]
semrewrite rewrites KB PREDICATE [--judge NAME] [--all]   # what discovery admits, and with --all what it rejects
semrewrite engines                                       # installed engines and their capabilities
semrewrite strategies                                    # the strategies and their knobs
semrewrite conformance --engine NAME                     # check an engine against the protocol
semrewrite convert family.metta family.kb                # between the text, JSON and MeTTa formats
```

`--judge oracle` uses the `gold` lines of the knowledge base instead of a model, which is how tests and the experiment separate engine behaviour from judge error. `--retriever all` judges every predicate of the right arity instead of the 8 nearest by gloss embedding, which suits small catalogues and loads no embedding model. The exit status is 0 when the query finished, 2 when a budget stopped it and 1 on an error.

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

| engine | needs | tables | depth bound | lazy discovery | untabled | proofs |
|---|---|---|---|---|---|---|
| `python` | nothing | yes | yes | yes | no | yes |
| `pymetta` | `pymetta` | yes | yes | yes | yes | no |

An engine is anything with a `solve(program, query, budget, expand)` method and a `capabilities` record; DESIGN.md gives the five rules it must keep. `semrewrite conformance --engine NAME` checks them with property tests against a reference evaluator: exact answers with tables, one meaning of depth, lazy expansion equal to eager, hard budgets, and engine keywords accepted as ordinary names. A Prolog through janus, a Datalog engine or SQL with recursive queries can implement the same protocol; add it to `semrewrite/engines/__init__.py` and run the kit.

## Judges

| judge | what it is | extra |
|---|---|---|
| `nli-deberta-base`, `nli-deberta-large`, `nli-deberta-large-wanli` | NLI cross-encoders reading premise and conclusion together | `models` |
| `reranker-qwen3` | Qwen3-Reranker asked yes or no | `models` |
| `jev` | TypeSafe's hosted System One model, with answers cached on disk; reads a TypeSafe API key from `TYPESAFE_API_KEY`, or from the file `JEV_KEY_FILE` names | `jev` |
| `cosine-bge`, `cosine-qwen3` | embedding similarity, the baseline that cannot tell direction | `models` |
| `oracle` | admits exactly a given set of implications | nothing |

Each judge's default threshold in `semrewrite/judges/thresholds.json` was calibrated on labelled predicate pairs. For the rewriter, calibrate it on your own knowledge bases instead (see RESULTS.md): the pair-set thresholds favour recall, and the rewriter needs precision because it applies every admitted rule everywhere. A judge is anything with `name`, `threshold`, `usage` and `score(proposals)`.

## Data

The pair sets the judges were measured on live under `data/pairs`: a hand-labelled hard set, and pairs from Wikidata's property hierarchy (CC0), whose query results are under `data/raw/wikidata`. The Levy/Holt directional pairs come from [entgraph_eval](https://github.com/mjhosseini/entgraph_eval), which states no license, so this repository does not include them. These commands download them at the commit the experiment used, check each file's SHA-256, and build the pair file:

```sh
python -m semrewrite.pairs fetch levyholt
python -m semrewrite.pairs build levyholt
```

The Jev judge's cached answers are not included either, so rerunning it needs a TypeSafe API key.

## What the experiment found

The experiment ran eleven strategies with a perfect judge, and five of them with each of seven model judges, over four generated knowledge bases (75 predicates, 58 queries) on PyMeTTa:

- With tables and a per-predicate memo, every query terminated, on the benchmarks and on 400 generated catalogues per property; without tables, rewriting looped exactly when the query reached a cycle of rewrites.
- Jev and the NLI cross-encoders tell which way an implication runs (direction accuracy 0.77 to 0.89 on two external pair sets); embedding similarity sits at chance.
- Admission precision decides answer quality, so the judge's threshold has to be calibrated for answer F1, not for pair accuracy.
- With an imperfect judge, a depth bound of 3 limits how far a false rule spreads. At calibrated thresholds, tabled evaluation with the bound (`depth-tabled`) raised answer F1 for six of the seven judges, nli-deberta-large's from 0.68 to 0.91, and cost the perfect judge three answers that need more than three steps.

RESULTS.md has every number, the method and the eight PyMeTTa defects found on the way, five of them with a script under `repros/` that reproduces them.

## Guarantees and limits

Every query terminates when the catalogue of predicates is finite, rewrites only rename and permute arguments, user rules have no function symbols, discovery is memoised, and evaluation is tabled or depth-bounded. Answers are exact with respect to the admitted rules; whether they are true depends on the judge. Function symbols, negation and discovery of facts from text are outside the design. DESIGN.md explains each condition and what breaks without it.

## Development

```sh
uv sync --all-extras
uv run pytest                                  # unit, property and conformance tests, 400 cases per property
uv run pytest --hypothesis-profile=ci          # the same with 100 cases per property, as CI runs PyMeTTa
uv run ruff check .                            # lint, as CI does
uv run semrewrite conformance --engine pymetta
```

The experiment's scripts are under `experiments/` and its results under `results/`; RESULTS.md says how to rerun them.

## License

Apache License 2.0; see [LICENSE](LICENSE).
