"""Purpose: the NLI judge: a three-way natural language inference cross-encoder
reads the premise sentence and the conclusion sentence together, premise first,
and its softmax over entailment, contradiction and neutral is the opinion's
belief, disbelief and uncertainty, so the score is the entailment probability.
Judging a proposal and its reverse reads the model in both directions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from semrewrite.judges.base import LETTERS, Fillers, SentenceJudge
from semrewrite.opinion import Opinion


class NLIJudge(SentenceJudge):
    """An NLI cross-encoder's entailment, contradiction and neutral probabilities as belief, disbelief and
    uncertainty. Neutral is the class for a conclusion that might be true given the premise, between definitely
    true and definitely false (SNLI's annotation; Bowman, Angeli, Potts and Manning 2015), so its mass is the
    absence of evidence either way that uncertainty stands for.

    The class indices are read from the model's id2label, since NLI checkpoints
    disagree on label order: cross-encoder/nli-deberta-v3-base puts entailment
    at 1, MoritzLaurer's DeBERTa-v3 checkpoints at 0.
    """

    # Disbelief is the contradiction class, so the claim check counts it against a claim directly.
    reads_contradiction: ClassVar[bool] = True

    def __init__(self, name: str, threshold: float, fillers: Fillers = LETTERS, *, model: str) -> None:
        from sentence_transformers import CrossEncoder

        super().__init__(name, threshold, fillers)
        self.model_name = model
        self._model = CrossEncoder(model)
        config = getattr(self._model.model, "config", None)
        if config is None or not config.id2label:
            msg = f"{model} has no label map to find its classes in"
            raise ValueError(msg)
        labels = {str(label).lower(): int(index) for index, label in config.id2label.items()}
        missing = sorted({"entailment", "contradiction", "neutral"} - set(labels))
        if missing:
            msg = f"{model} has no {', '.join(missing)} class among {sorted(labels)}"
            raise ValueError(msg)
        self._classes = (labels["entailment"], labels["contradiction"], labels["neutral"])

    # Time: one cross-encoder forward pass per pair, batched by sentence-transformers.
    def _opinions(self, pairs: Sequence[tuple[str, str]]) -> list[Opinion]:
        import torch

        probabilities = self._model.predict(list(pairs), activation_fn=torch.nn.Softmax(dim=-1), convert_to_numpy=True)
        out = []
        # A float32 softmax sums to one only within about 1e-7, so the three masses are summed and divided in float64.
        for row in probabilities.astype("float64"):
            entail, contradict, neutral = (float(row[i]) for i in self._classes)
            total = entail + contradict + neutral
            out.append(Opinion(entail / total, contradict / total, neutral / total))
        return out

    def _release(self) -> None:
        del self._model
