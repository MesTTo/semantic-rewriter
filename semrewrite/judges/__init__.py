"""Purpose: the judge factory, which builds judges by name for the command
line and the experiments. make(name) builds a direction judge, its threshold
read from semrewrite/judges/thresholds.json, where experiments/judge_eval.py
writes each judge's threshold calibrated on dev rows; make("oracle", gold=...)
builds the OracleJudge of semrewrite/judges/oracle.py.

    from semrewrite.judges import make
    judge = make("nli-deberta-large")
    judge.score([Proposal(mother, parent, (0, 1)), Proposal(parent, mother, (0, 1))])  # high, then low

Every judge keeps its model loaded until close(); load one at a time on a
shared GPU.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Iterable
from typing import Any

from semrewrite.judges.base import SentenceJudge
from semrewrite.judges.embedding import CosineJudge
from semrewrite.judges.jev import JevJudge
from semrewrite.judges.nli import NLIJudge
from semrewrite.judges.oracle import OracleJudge
from semrewrite.judges.reranker import RerankerJudge
from semrewrite.protocols import Judge

THRESHOLDS = pathlib.Path(__file__).with_name("thresholds.json")
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
    "jev": (JevJudge, {"cache": JEV_ANSWERS, "log": JEV_REQUESTS}),
}
NAMES = tuple(REGISTRY)


def default_threshold(name: str, path: pathlib.Path | None = None) -> float:
    """The threshold calibrated for `name` on dev rows; raises when none was calibrated."""
    path = THRESHOLDS if path is None else path
    table = json.loads(path.read_text()) if path.is_file() else {}
    entry = table.get("judges", {}).get(name)
    if entry is None:
        msg = f"no calibrated threshold for {name!r} in {path}; run experiments/judge_eval.py or pass threshold="
        raise LookupError(msg)
    return float(entry["default"])


def make(
    name: str, *, threshold: float | None = None, gold: Iterable[tuple[str, str, Any]] | None = None, **options: Any
) -> Judge:
    """The judge called `name`, at `threshold` or its calibrated default.

    `options` reach the judge's constructor: `fillers=("X", "Y")` for any judge,
    `budget=` or `cache=` for Jev. "oracle" needs `gold`, the (premise,
    conclusion, mapping) triples that hold; a bool stands for a binary swap.
    """
    if name == "oracle":
        if gold is None:
            msg = "the oracle judge needs gold=(premise, conclusion, mapping) triples"
            raise ValueError(msg)
        oracle = OracleJudge.of(gold)
        if threshold is not None:
            oracle.threshold = threshold
        return oracle
    if name not in REGISTRY:
        msg = f"unknown judge {name!r}; known: {', '.join(NAMES)}, oracle"
        raise ValueError(msg)
    cls, defaults = REGISTRY[name]
    chosen = default_threshold(name) if threshold is None else threshold
    return cls(name, chosen, **{**defaults, **options})
