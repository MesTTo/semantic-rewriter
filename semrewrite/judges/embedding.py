"""Purpose: the cosine judge, the similarity baseline: a proposal scores the
cosine between its premise and conclusion sentence embeddings, mapped from
[-1, 1] to [0, 1]. It scores a proposal and its reverse alike, so it measures
how far relatedness alone gets a direction judge.
"""

from __future__ import annotations

from collections.abc import Sequence

from semrewrite.judges.base import LETTERS, Fillers, SentenceJudge


class CosineJudge(SentenceJudge):
    """(1 + cos) / 2 between the two sentences' normalised embeddings.

    `prompt` is prepended to both sentences alike, keeping the score symmetric;
    `usage["encoded"]` counts sentences embedded, each distinct sentence once a call.
    """

    def __init__(
        self, name: str, threshold: float, fillers: Fillers = LETTERS, *, model: str, prompt: str | None = None
    ) -> None:
        from sentence_transformers import SentenceTransformer

        super().__init__(name, threshold, fillers)
        self.model_name = model
        self.prompt = prompt
        self.usage["encoded"] = 0
        self._model = SentenceTransformer(model)

    # Time: one batched encoding of the distinct sentences, then |pairs| dot products of the embedding width.
    def _score_sentences(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        sentences = sorted({s for pair in pairs for s in pair})
        vectors = self._model.encode(sentences, prompt=self.prompt, normalize_embeddings=True, convert_to_numpy=True)
        self.usage["encoded"] += len(sentences)
        row = {s: v for s, v in zip(sentences, vectors, strict=True)}
        # Normalised vectors can overshoot |cos| = 1 by float rounding, hence the clip.
        return [min(1.0, max(0.0, (1.0 + float(row[p] @ row[c])) / 2.0)) for p, c in pairs]

    def _release(self) -> None:
        del self._model
