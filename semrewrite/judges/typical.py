"""Purpose: the typicality judge: an instruction-tuned language model is asked whether the conclusion
sentence typically follows from the premise sentence and answers Yes, No or Unsure; the probabilities it
gives the three answers, taken relative to each other, are the opinion's belief, disbelief and uncertainty.
It asks the defeasible, commonsense question that a knowledge base of actions needs, whether "A watches TV"
typically means "A wastes time", where an entailment judge abstains because the conclusion is not necessary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

from semrewrite.judges.base import LETTERS, Fillers, SentenceJudge
from semrewrite.opinion import Opinion

# The question the model is asked about each pair, in its chat template's user turn. "Typically" is the defeasible
# reading: what most people would expect knowing only the premise, though it need not hold, so a bird typically flies
# and a penguin, a bird, does not (Reiter 1980). Of three phrasings tried on the typical sets' dev rows this one gave
# the lowest log loss; asking instead how often the conclusion holds left the model unsure on most rows. It errs
# toward Yes: it calls a parent typically a mother.
QUESTION = (
    "Premise: {premise}\n"
    "Conclusion: {conclusion}\n\n"
    "Does the conclusion typically follow from the premise? Answer Yes if, knowing only the premise, most people "
    "would expect the conclusion to be true, even though it need not be; No if they would expect it to be false or "
    "would have no reason to expect it; Unsure if you cannot tell. Answer with one word: Yes, No or Unsure."
)

# The answers whose probabilities become belief, disbelief and uncertainty, in that order.
ANSWERS = ("Yes", "No", "Unsure")

# Sequences per forward pass. Qwen3-4B-Instruct-2507 in bfloat16 peaked at 8.27 GiB of GPU memory on 32 sequences
# of about 100 tokens, 8.05 GiB of it weights, since logits are kept for the answer positions only.
BATCH = 32


class TypicalJudge(SentenceJudge):
    """An instruction-tuned causal language model's probabilities of answering Yes, No and Unsure to whether the
    conclusion typically follows from the premise, normalised over the three, as belief, disbelief and uncertainty.

    Each answer's probability is that of its whole token sequence after the chat template's assistant prompt, so
    an answer the tokenizer splits, such as "Un" "sure", is scored exactly; the mass the model puts on any other
    reply is dropped by the normalisation. A model with a thinking mode is asked with it off.
    """

    # Its defaults come from delta-ATOMIC's dev rows: what typically follows from an everyday event about a person,
    # the kind of question a knowledge base of actions asks.
    calibration_set: ClassVar[str] = "delta-atomic"

    def __init__(
        self, name: str, threshold: float, fillers: Fillers = LETTERS, *, model: str, question: str = QUESTION,
        answers: Sequence[str] = ANSWERS,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if len(answers) != 3:
            msg = f"{name} needs three answers, for belief, disbelief and uncertainty, not {list(answers)}"
            raise ValueError(msg)
        super().__init__(name, threshold, fillers)
        self.model_name = model
        self.question = question
        tokenizer = AutoTokenizer.from_pretrained(model)
        if tokenizer is None:
            msg = f"{model} has no tokenizer"
            raise ValueError(msg)
        self._tokenizer = tokenizer
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self._device).eval()
        self._answers = [self._tokenizer.encode(a, add_special_tokens=False) for a in answers]
        if len({tuple(a) for a in self._answers}) != 3 or not all(self._answers):
            msg = f"{model}'s tokenizer does not tell the answers {list(answers)} apart"
            raise ValueError(msg)
        pad = self._tokenizer.pad_token_id
        self._pad = pad if pad is not None else self._tokenizer.eos_token_id

    def _prompt(self, premise: str, conclusion: str) -> list[int]:
        """The token ids of the question about one pair, through the assistant prompt, thinking off."""
        messages = [{"role": "user", "content": self.question.format(premise=premise, conclusion=conclusion)}]
        text = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                   enable_thinking=False)
        return self._tokenizer.encode(str(text), add_special_tokens=False)

    # Time: max(1, m) causal-LM forward passes per pair, m the answers longer than one token (1 for Yes, No and
    # "Un" "sure"), batched BATCH at a time in length order, with logits kept for the answer positions only.
    def _opinions(self, pairs: Sequence[tuple[str, str]]) -> list[Opinion]:
        import torch

        # One sequence per answer longer than a token, the prompt and that answer but its last token, which scores
        # the answer token by token; a one-token answer is scored at the prompt's end, which every sequence of the
        # pair shares, so from its first. With no answer longer than a token, the prompt alone is the sequence.
        multi = [k for k, a in enumerate(self._answers) if len(a) > 1]
        extras = [self._answers[k][:-1] for k in multi] or [[]]
        serves = multi or [None]
        prompts = [self._prompt(premise, conclusion) for premise, conclusion in pairs]
        sequences = [(i, j, prompt + extra) for i, prompt in enumerate(prompts) for j, extra in enumerate(extras)]
        logprob = [[0.0] * len(self._answers) for _ in prompts]
        order = sorted(range(len(sequences)), key=lambda j: len(sequences[j][2]))
        for start in range(0, len(order), BATCH):
            chunk = [sequences[j] for j in order[start : start + BATCH]]
            width = max(len(s) for _, _, s in chunk)
            keep = max(len(extras[j]) for _, j, _ in chunk) + 1
            # Left padding ends every row at its last token, so the logits of the last `keep` positions cover the
            # prompt's last position and each extra answer token's in every row; positions count from each row's
            # first real token.
            ids = torch.full((len(chunk), width), self._pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, (_, _, s) in enumerate(chunk):
                ids[row, width - len(s) :] = torch.tensor(s, dtype=torch.long)
                mask[row, width - len(s) :] = 1
            positions = (mask.cumsum(-1) - 1).clamp(min=0)
            with torch.inference_mode():
                logits = self._model(input_ids=ids.to(self._device), attention_mask=mask.to(self._device),
                                     position_ids=positions.to(self._device), logits_to_keep=keep).logits
            logp = torch.log_softmax(logits.float(), dim=-1).cpu()
            for row, (i, j, _) in enumerate(chunk):
                # The prompt's last position, which predicts an answer's first token, is kept index keep - 1 - e for
                # a row with e extra tokens; answer token t is predicted t positions later.
                first = keep - 1 - len(extras[j])
                served = serves[j]
                if served is not None:
                    answer = self._answers[served]
                    logprob[i][served] = float(sum(logp[row, first + t, answer[t]] for t in range(len(answer))))
                if j == 0:
                    for k, answer in enumerate(self._answers):
                        if len(answer) == 1:
                            logprob[i][k] = float(logp[row, first, answer[0]])
        out = []
        for row in logprob:
            top = max(row)
            weights = [math.exp(v - top) for v in row]
            total = sum(weights)
            out.append(Opinion(*(w / total for w in weights)))
        return out

    def _release(self) -> None:
        del self._model
