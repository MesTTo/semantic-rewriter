"""Purpose: a direction judge backed by Jev, TypeSafe's hosted System One
model. Each proposal becomes one yes/no (Noul) question quoting its premise and
conclusion sentences; many questions ride in one request over a shared state
that states the task and what yes and no mean; the yes probability is the score.

Answers are cached on disk keyed by the model alias, the state and the question,
so a rerun spends no requests; every request sent is appended to a log, and a
judge refuses to send once the log holds `budget` requests. The API key comes
from TYPESAFE_API_KEY, or else from the file JEV_KEY_FILE names, and is never
logged, printed or stored.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from collections.abc import Sequence
from typing import Any

from semrewrite.judges.base import LETTERS, Fillers, SentenceJudge

MODEL = "jev-latest"

# The shared state every question refers to. Yes is reserved for entailment by
# meaning; "related", "similar" and "usually true together" are named as no,
# since those are the ways a direction judge admits an unsound rewrite.
STATE: dict[str, str] = {
    "task": (
        "Each question quotes two statements about the same two things. Decide whether the "
        "second statement is necessarily true whenever the first statement is true."
    ),
    "yes": (
        "The second statement follows from the first by the meaning of the words: no possible "
        "situation makes the first statement true and the second false."
    ),
    "no": (
        "Some possible situation makes the first statement true and the second false, even an "
        "unusual one; statements that are merely related, similar, or often true together get no."
    ),
}

# A request of 874 questions answered in 0.55 s; 800 stays inside what was measured.
QUESTIONS_PER_REQUEST = 800

# The default cap on Jev requests, counted over the request log, so a run cannot spend without bound.
DEFAULT_BUDGET = 300


def resolve_key() -> str:
    """The API key, from the environment or the key file; never returned anywhere else."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key.strip()
    named = os.environ.get("JEV_KEY_FILE")
    path = pathlib.Path(named).expanduser() if named else None
    if path is None or not path.is_file():
        msg = f"set TYPESAFE_API_KEY, or JEV_KEY_FILE to a file holding the key (JEV_KEY_FILE is {named!r})"
        raise RuntimeError(msg)
    return path.read_text().strip()


def question_text(premise: str, conclusion: str) -> str:
    """The Noul instruction for one proposal."""
    return (
        f'First statement: "{premise}". Second statement: "{conclusion}". '
        "Is the second statement necessarily true whenever the first is true?"
    )


def _key(model: str, question: str) -> str:
    payload = json.dumps({"model": model, "state": STATE, "question": question}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class JevJudge(SentenceJudge):
    """Jev's yes probability that the conclusion sentence follows from the premise sentence.

    `usage` adds to the shared counts: `requests` sent, `questions` sent,
    `cached` answered from the cache, and the billed `input_tokens` and free
    `output_tokens` the API reported.
    """

    def __init__(
        self,
        name: str,
        threshold: float,
        fillers: Fillers = LETTERS,
        *,
        model: str = MODEL,
        cache: pathlib.Path | None = None,
        log: pathlib.Path | None = None,
        budget: int | None = DEFAULT_BUDGET,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(name, threshold, fillers)
        self.model = model
        self.cache_path = cache
        self.log_path = log
        self.budget = budget
        self.timeout = timeout
        self.usage.update({"requests": 0, "questions": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0})
        self._answers: dict[str, float] = {row["key"]: row["noul"] for row in _read_jsonl(cache)} if cache else {}
        self._client: Any = None

    def _connect(self) -> Any:
        if self._client is None:
            from typesafe_sdk import RetryPolicy, TypeSafeClient

            # No SDK retries: a retried request is another billed request the log would miss.
            self._client = TypeSafeClient(api_key=resolve_key(), retry=RetryPolicy(max_retries=0), timeout=self.timeout)
        return self._client

    def _logged_requests(self) -> int:
        return len(_read_jsonl(self.log_path)) if self.log_path else self.usage["requests"]

    def _append(self, path: pathlib.Path | None, rows: list[dict[str, Any]]) -> None:
        if path is None or not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    # Time: one request per QUESTIONS_PER_REQUEST uncached distinct questions; cache hits cost a dict lookup.
    def _score_sentences(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        questions = [question_text(p, c) for p, c in pairs]
        keys = [_key(self.model, q) for q in questions]
        missing = list(dict.fromkeys(k for k in keys if k not in self._answers))
        self.usage["cached"] += sum(1 for k in keys if k in self._answers)
        text_of = dict(zip(keys, questions, strict=True))
        for start in range(0, len(missing), QUESTIONS_PER_REQUEST):
            self._ask([(k, text_of[k]) for k in missing[start : start + QUESTIONS_PER_REQUEST]])
        return [self._answers[k] for k in keys]

    def _ask(self, batch: list[tuple[str, str]]) -> None:
        from typesafe_sdk import Noul

        if self.budget is not None and self._logged_requests() >= self.budget:
            msg = f"Jev request budget of {self.budget} reached in {self.log_path}; pass a larger budget to raise it"
            raise RuntimeError(msg)
        names = {f"q{i}": key for i, (key, _) in enumerate(batch)}
        questions = {f"q{i}": Noul(instructions=text) for i, (_, text) in enumerate(batch)}
        started = time.monotonic()
        try:
            response = self._connect().system_one(state=STATE, model=self.model, questions=questions)
        except Exception as error:
            # A refused or failed request may still be billed, so it enters the ledger too.
            self.usage["requests"] += 1
            self._append(
                self.log_path,
                [{"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "questions": len(batch), "error": type(error).__name__}],
            )
            raise
        elapsed = time.monotonic() - started
        answered = response.nouls
        lost = sorted(set(names) - set(answered))
        self.usage["requests"] += 1
        self.usage["questions"] += len(batch)
        self.usage["input_tokens"] += response.usage.input_tokens or 0
        self.usage["output_tokens"] += response.usage.output_tokens or 0
        self._append(
            self.log_path,
            [
                {
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "model": response.model,
                    "questions": len(batch),
                    "answered": len(answered),
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "seconds": round(elapsed, 3),
                }
            ],
        )
        if lost:
            msg = f"Jev answered {len(answered)} of {len(batch)} questions; missing {lost[:5]}"
            raise RuntimeError(msg)
        rows = []
        text = dict(batch)
        for name, key in names.items():
            self._answers[key] = float(answered[name].noul)
            rows.append({"key": key, "model": response.model, "question": text[key], "noul": self._answers[key]})
        self._append(self.cache_path, rows)

    def _release(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
