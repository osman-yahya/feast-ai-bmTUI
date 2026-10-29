"""One structured-output call to Claude, for dataset generation only.

The data is synthetic, so using a cloud model here is fine (§5.1); nothing
from a real user ever reaches this module.  ``anthropic`` is an optional
dependency (``pip install .[memeval]``) and is imported lazily, so the rest
of feast AI keeps its four core dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

DEFAULT_MODEL = "claude-opus-5-5"


class GenError(RuntimeError):
    """A generation call failed in a way the user can act on."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


def complete_json(system: str, prompt: str, schema: dict, *, model: str = DEFAULT_MODEL,
                  effort: str = "medium") -> tuple[dict, Usage]:
    try:
        import anthropic
    except ImportError:
        raise GenError("the generator needs the Anthropic SDK: pip install anthropic") from None

    client = anthropic.Anthropic()
    try:
        # Streaming: a persona's full history is a long answer, and a
        # non-streaming call that long risks the HTTP timeout. The default
        # fallback chain answers a safety decline instead of failing the run.
        with client.beta.messages.stream(
            model=model,
            max_tokens=64000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort,
                           "format": {"type": "json_schema", "schema": schema}},
        ) as stream:
            message = stream.get_final_message()
    except anthropic.AuthenticationError:
        raise GenError("no valid credentials: set ANTHROPIC_API_KEY") from None
    except anthropic.RateLimitError:
        raise GenError("rate limited - wait a minute and re-run; finished personas are cached") from None
    except anthropic.APIStatusError as exc:
        raise GenError(f"API error {exc.status_code}: {exc.message}") from None
    except anthropic.APIConnectionError:
        raise GenError("could not reach the Anthropic API") from None

    if message.stop_reason == "refusal":
        raise GenError("the model declined this request")
    if message.stop_reason == "max_tokens":
        raise GenError("output hit max_tokens; generate fewer sessions per call")
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenError(f"model returned invalid JSON: {exc}") from None
    usage = Usage(message.usage.input_tokens, message.usage.output_tokens)
    return data, usage
