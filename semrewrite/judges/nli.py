"""Purpose: the NLI judge: a three-way natural language inference cross-encoder
reads the premise sentence and the conclusion sentence together, premise first,
and the softmax probability of its entailment class is the score. Scoring
a proposal and its reverse reads the model in both directions.
"""

from __future__ import annotations

from collections.abc import Sequence

from semrewrite.judges.base import LETTERS, Fillers, SentenceJudge


class NLIJudge(SentenceJudge):
    """P(entailment | premise, conclusion) from an NLI cross-encoder.

    The entailment class index is read from the model's id2label, since NLI
    checkpoints disagree on label order: cross-encoder/nli-deberta-v3-base puts
    entailment at 1, MoritzLaurer's DeBERTa-v3 checkpoints at 0.
    """

    def __init__(self, name: str, threshold: float, fillers: Fillers = LETTERS, *, model: str) -> None:
        from sentence_transformers import CrossEncoder

        super().__init__(name, threshold, fillers)
        self.model_name = model
        self._model = CrossEncoder(model)
        labels = {str(label).lower(): int(index) for index, label in self._model.model.config.id2label.items()}
        if "entailment" not in labels:
            msg = f"{model} has no entailment class among {sorted(labels)}"
            raise ValueError(msg)
        self._entailment = labels["entailment"]

    # Time: one cross-encoder forward pass per pair, batched by sentence-transformers.
    def _score_sentences(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        import torch

        probabilities = self._model.predict(list(pairs), activation_fn=torch.nn.Softmax(dim=-1), convert_to_numpy=True)
        return [float(p[self._entailment]) for p in probabilities]

    def _release(self) -> None:
        del self._model
