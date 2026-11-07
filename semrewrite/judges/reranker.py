"""Purpose: the reranker judge: Qwen3-Reranker is a cross-encoder that answers
yes or no to whether a document meets a query under an instruction, so it is
asked whether the conclusion sentence (document) must hold whenever the premise
sentence (query) does, and its yes probability is the score.
"""

from __future__ import annotations

from collections.abc import Sequence

from semrewrite.judges.base import LETTERS, Fillers, SentenceJudge

# The Instruct line of the reranker's documented chat template; the model's own
# system prompt already restricts the answer to "yes" or "no".
INSTRUCTION = (
    "Judge whether the Document must be true whenever the Query is true: "
    "answer yes only if the Query entails the Document."
)


class RerankerJudge(SentenceJudge):
    """P(yes) of an instruction-following yes/no reranker, premise as query, conclusion as document.

    sentence-transformers scores Qwen3-Reranker as the logit of "yes" minus the
    logit of "no", so a sigmoid of it is the two-way softmax probability of yes,
    the score the model card's transformers recipe computes.
    """

    def __init__(
        self, name: str, threshold: float, fillers: Fillers = LETTERS, *, model: str, instruction: str = INSTRUCTION
    ) -> None:
        from sentence_transformers import CrossEncoder

        super().__init__(name, threshold, fillers)
        self.model_name = model
        self._model = CrossEncoder(model, prompts={"entailment": instruction}, default_prompt_name="entailment")

    # Time: one causal-LM forward pass per pair, batched by sentence-transformers.
    def _score_sentences(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        import torch

        scores = self._model.predict(list(pairs), activation_fn=torch.nn.Sigmoid(), convert_to_numpy=True)
        return [float(s) for s in scores]

    def _release(self) -> None:
        del self._model
