"""Route with a Decision 2.0 model instead of the ModernBERT-TR encoder.

The policy is the same one :mod:`feast_ai.agent.router` applies - widen on a
near-tie, inherit follow-ups, narrow otherwise - so the comparison isolates
the *classifier*.  What changes is how the scores are made: one ``choice``
question whose options are the 19 feast intents, each described by its
summary and two of its router exemplars (never suite sentences), answered by
``/v1/systemone`` as a probability per intent.

Probabilities live on a different scale from cosine similarities, so this
router carries its own thresholds (:data:`THRESHOLDS`) instead of the
Settings values calibrated for ModernBERT.

With the 13 :mod:`feast_ai.agent.classes` the question changes in three ways
(``classes=True``): each option is described by what the SLM must do rather
than by what the user says, the instructions ask about the *latest* message,
and the state carries the previous user message, so a bare "150 lirayi
gecmesin ama" can be read as the food search it continues.  Fewer, shorter
options are also the only latency lever: the decision2 server builds
``Context: <message> ... Options ...`` with ``cache_prompt = false``, and the
message comes first, so no part of the question is ever reused across calls.
"""

from __future__ import annotations

import os
import time

import httpx

from .intents import INTENTS
from .router import Router, RouterError

INSTRUCTIONS = (
    "Which kind of request is this message to the assistant of the feast "
    "restaurant app? Messages are often casual Turkish."
)

#: (fall back below, near-tie within, answer without the SLM at or above).
#: Probabilities are well calibrated here: on the standard suite a top
#: probability >= 0.50 was right on every turn, but covered only 38% of them;
#: 0.30 keeps most turns routed and sends the rest to the full prompt.
THRESHOLDS = (0.30, 0.15, 0.80)
#: The class question spreads probability over 13 options. Router-only sweep
#: on the 13 classes: top >= 0.30 kept 38/60 standard turns at 82% (15/31
#: kanka at 93%); >= 0.25 kept 46/60 at 78% (18/31 at 94%); >= 0.20 kept
#: 55/60 at 69%. 0.25 routes three quarters of turns without admitting the
#: low-probability guesses, which were wrong more often than right.
CLASS_THRESHOLDS = (0.25, 0.10, 0.80)


#: Router exemplars quoted per intent description. Each one adds ~25 prompt
#: tokens per intent to *every* routing call: the message comes first in the
#: prompt, so llama.cpp cannot cache the question across requests.
#: Measured on the standard suite: 2 examples -> 70% top-1 at 2.6 s per call;
#: none -> 50% at 1.1 s. FEAST_DECISION_EXAMPLES overrides it for experiments.
EXAMPLES_PER_INTENT = int(os.environ.get("FEAST_DECISION_EXAMPLES", "2"))
#: The same knob for the class taxonomy, whose summaries say more on their own.
EXAMPLES_PER_CLASS = int(os.environ.get("FEAST_DECISION_CLASS_EXAMPLES", "1"))

CLASS_INSTRUCTIONS = (
    "What must the assistant of the feast food app do for the user's latest message? "
    "Users write casual Turkish."
)


def criteria(specs=INTENTS, examples: int = EXAMPLES_PER_INTENT) -> dict[str, str]:
    """``id -> description`` - the options of the routing question."""
    out = {}
    for spec in specs:
        quoted = "; ".join(f'"{e}"' for e in spec.exemplars[:examples])
        out[spec.id] = f"{spec.summary} (e.g. {quoted})" if quoted else spec.summary
    return out


class DecisionRouter(Router):
    """Same policy, scores from a Decision 2.0 ``choice`` question."""

    def __init__(self, base_url: str, settings, taxonomy: dict | None = None) -> None:
        super().__init__(base_url, settings, taxonomy)
        self.classes = taxonomy is not None
        if self.classes:
            self._criteria = criteria(tuple(taxonomy.values()), EXAMPLES_PER_CLASS)
            self._instructions = CLASS_INSTRUCTIONS
        else:
            self._criteria = criteria()
            self._instructions = INSTRUCTIONS

    def thresholds(self) -> tuple[float, float, float]:
        return CLASS_THRESHOLDS if self.classes else THRESHOLDS

    def question(self) -> dict:
        """The routing question minus the message - what the report shows."""
        state = ("Previous user message: ...\nLatest user message: ..." if self.classes
                 else "<the message>")
        return {"instructions": self._instructions, "criteria": self._criteria, "state": state}

    def state(self, text: str, previous=None) -> str:
        if self.classes and previous is not None and previous.text:
            return f"Previous user message: {previous.text}\nLatest user message: {text}"
        return text

    async def classify(self, text: str, previous=None) -> dict:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))
        body = {
            "state": self.state(text, previous),
            "questions": {"intent": {
                "type": "choice", "instructions": self._instructions, "criteria": self._criteria,
            }},
        }
        started = time.perf_counter()
        try:
            response = await self._client.post(f"{self.base_url}/v1/systemone", json=body)
        except httpx.HTTPError as exc:
            raise RouterError(f"Decision router unreachable: {exc}") from exc
        if response.status_code != 200:
            raise RouterError(f"Decision router HTTP {response.status_code}: {response.text[:200]}")
        data = response.json()
        probs = data["answers"]["intent"]["probabilities"]
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        return {
            "scores": [[k, float(v)] for k, v in ranked],
            "nearest": "",
            "tokens": int(data.get("usage", {}).get("input_tokens", 0)),
            # The server reports no split timing; the whole request is the
            # inference cost (one localhost hop included).
            "tokenize_ms": 0.0,
            "infer_ms": (time.perf_counter() - started) * 1000.0,
        }
