"""Purpose: the judge factory, which builds judges by name for the command
line and the experiments. make(name) builds a direction judge, its threshold,
trust, base rate and check trust read from semrewrite/judges/calibration.json,
where experiments/judge_eval.py writes each judge's threshold and trust
calibrated on the dev rows of the labelled set its entry names, and each such
set's base rate once, and experiments/check_trust.py writes the trust the claim
check puts in the judge's reading of a statement against a claim;
make("oracle", gold=...) builds the OracleJudge of semrewrite/judges/oracle.py.

    from semrewrite.judges import make
    judge = make("nli-deberta-large")
    judge.score([Proposal(mother, parent, (0, 1)), Proposal(parent, mother, (0, 1))])  # high, then low
    judge.opinions([Proposal(mother, parent, (0, 1))])  # belief, disbelief and uncertainty of each

Every judge keeps its model loaded until close(); load one at a time on a
shared GPU.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Iterable
from typing import Any

from semrewrite.judges.base import DEFAULT_SET, SentenceJudge
from semrewrite.judges.embedding import CosineJudge
from semrewrite.judges.jev import JevJudge
from semrewrite.judges.nli import NLIJudge
from semrewrite.judges.oracle import OracleJudge
from semrewrite.judges.reranker import RerankerJudge
from semrewrite.judges.typical import TypicalJudge
from semrewrite.protocols import Judge

CALIBRATION = pathlib.Path(__file__).with_name("calibration.json")
# Jev's answer cache and request log: SEMREWRITE_JEV_DIR when set, else the
# repository's data/raw/jev when running from a checkout, else the user cache.
_CHECKOUT_JEV = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw" / "jev"
JEV_DIR = pathlib.Path(
    os.environ.get("SEMREWRITE_JEV_DIR")
    or (_CHECKOUT_JEV if _CHECKOUT_JEV.is_dir() else pathlib.Path.home() / ".cache" / "semrewrite" / "jev")
)
JEV_ANSWERS = JEV_DIR / "answers.jsonl"
JEV_REQUESTS = JEV_DIR / "requests.jsonl"

# Qwen3-Embedding's instruction for symmetric similarity tasks, given to both
# sentences: https://github.com/QwenLM/Qwen3-Embedding/blob/main/evaluation/qwen3_embedding_model.py
QWEN3_SIMILARITY = "Instruct: Retrieve semantically similar text\nQuery:"

# Each judge's class and the model it wraps, by the name make() takes.
REGISTRY: dict[str, tuple[type[SentenceJudge], dict[str, Any]]] = {
    "cosine-bge": (CosineJudge, {"model": "BAAI/bge-small-en-v1.5"}),
    "cosine-qwen3": (CosineJudge, {"model": "Qwen/Qwen3-Embedding-0.6B", "prompt": QWEN3_SIMILARITY}),
    "nli-deberta-base": (NLIJudge, {"model": "cross-encoder/nli-deberta-v3-base"}),
    "nli-deberta-large": (NLIJudge, {"model": "cross-encoder/nli-deberta-v3-large"}),
    "nli-deberta-large-wanli": (NLIJudge, {"model": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"}),
    "reranker-qwen3": (RerankerJudge, {"model": "Qwen/Qwen3-Reranker-0.6B"}),
    "typical-qwen3-4b": (TypicalJudge, {"model": "Qwen/Qwen3-4B-Instruct-2507"}),
    "jev": (JevJudge, {"cache": JEV_ANSWERS, "log": JEV_REQUESTS}),
}
NAMES = tuple(REGISTRY)


def _table(path: pathlib.Path | None) -> tuple[pathlib.Path, dict[str, Any]]:
    """The calibration file at `path`, the committed one by default, and what it holds, empty when it is missing."""
    path = CALIBRATION if path is None else path
    return path, json.loads(path.read_text()) if path.is_file() else {}


def _calibrated(name: str, field: str, option: str, path: pathlib.Path | None) -> float:
    """The value of `field` calibrated for judge `name`, which `option` of make() overrides; raises when none was
    calibrated."""
    path, table = _table(path)
    entry = table.get("judges", {}).get(name, {})
    if field not in entry:
        msg = f"no calibrated {option} for {name!r} in {path}; run experiments/judge_eval.py or pass {option}="
        raise LookupError(msg)
    return float(entry[field])


def default_threshold(name: str, path: pathlib.Path | None = None) -> float:
    """The threshold calibrated for `name` on dev rows; raises when none was calibrated."""
    return _calibrated(name, "default", "threshold", path)


def default_trust(name: str, path: pathlib.Path | None = None) -> float:
    """The trust calibrated for `name` on dev rows, the probability that its opinions are right; raises when none was
    calibrated."""
    return _calibrated(name, "trust", "trust", path)


def calibration_set(name: str, path: pathlib.Path | None = None) -> str:
    """The labelled pair set judge `name`'s defaults were calibrated on: the set its entry names, Wikidata's pairs
    where it names none."""
    _, table = _table(path)
    return str(table.get("judges", {}).get(name, {}).get("set", DEFAULT_SET))


def default_base_rate(name: str, path: pathlib.Path | None = None) -> float:
    """The base rate judge `name`'s opinions take: the rate at which the dev proposals of the set its defaults were
    calibrated on hold, what an opinion projects to where it says nothing. It belongs to that set's proposals, not to
    a judge, so the calibration file holds it once per set; raises when none was calibrated."""
    path, table = _table(path)
    home = calibration_set(name, path)
    entry = table.get("sets", {}).get(home, {})
    if "base_rate" not in entry:
        msg = f"no calibrated base_rate for set {home!r} in {path}; run experiments/judge_eval.py or pass base_rate="
        raise LookupError(msg)
    return float(entry["base_rate"])


def default_data_trust(name: str, path: pathlib.Path | None = None) -> float:
    """The trust in the knowledge base's evidence on a rewrite, calibrated on dev rows jointly with judge `name`'s
    opinions, which it is fused with; the oracle needs none, being right by construction, so it gets 0."""
    return 0.0 if name == "oracle" else _calibrated(name, "data_trust", "data_trust", path)


# The labelled set the claim check's trust is fitted on: delta-ATOMIC's updates, each a statement, read against the
# inference it strengthens or weakens, the claim. Its base rate sits in the calibration file's sets table with the
# others'.
CHECK_SET = "delta-atomic"


def default_check_trust(name: str, path: pathlib.Path | None = None) -> float:
    """The trust the claim check puts in judge `name`'s reading of a statement against a rewrite's claim, fitted by
    experiments/check_trust.py on CHECK_SET's dev updates read against their inferences; raises when none was
    fitted."""
    return _calibrated(name, "check_trust", "check_trust", path)


def make(
    name: str, *, threshold: float | None = None, trust: float | None = None, base_rate: float | None = None,
    check_trust: float | None = None, gold: Iterable[tuple[str, str, Any]] | None = None, **options: Any,
) -> Judge:
    """The judge called `name`, at `threshold`, `trust`, `base_rate` and `check_trust` or their calibrated defaults.

    `options` reach the judge's constructor: `fillers=("X", "Y")` for any judge,
    `budget=` or `cache=` for Jev. "oracle" needs `gold`, the (premise,
    conclusion, mapping) triples that hold; a bool stands for a binary swap. The
    oracle is right by construction, so its trust is complete unless given, and
    its opinions are certain, so its base rate never shows; it reads no
    sentences, so it has no check trust. A judge with no calibrated check trust
    gets None, which the claim check refuses only when it has something to read.
    """
    if name == "oracle":
        if gold is None:
            msg = "the oracle judge needs gold=(premise, conclusion, mapping) triples"
            raise ValueError(msg)
        oracle = OracleJudge.of(gold)
        if threshold is not None:
            oracle.threshold = threshold
        if trust is not None:
            oracle.trust = trust
        if base_rate is not None:
            oracle.base_rate = base_rate
        return oracle
    if name not in REGISTRY:
        msg = f"unknown judge {name!r}; known: {', '.join(NAMES)}, oracle"
        raise ValueError(msg)
    cls, defaults = REGISTRY[name]
    chosen = default_threshold(name) if threshold is None else threshold
    judge = cls(name, chosen, **{**defaults, **options})
    judge.trust = default_trust(name) if trust is None else trust
    judge.base_rate = default_base_rate(name) if base_rate is None else base_rate
    if check_trust is None:
        try:
            check_trust = default_check_trust(name)
        except LookupError:
            check_trust = None
    judge.check_trust = check_trust
    # The bar the claim check defeats a rewrite at, which a run admitting at another threshold does not move.
    judge.calibrated_threshold = default_threshold(name) if _has(name, "default") else chosen
    return judge


def _has(name: str, field: str, path: pathlib.Path | None = None) -> bool:
    """Whether the calibration file holds `field` for judge `name`."""
    _, table = _table(path)
    return field in table.get("judges", {}).get(name, {})
