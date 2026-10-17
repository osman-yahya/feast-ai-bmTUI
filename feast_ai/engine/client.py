"""Streaming client for the llama.cpp OpenAI-compatible endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from ..config import Settings


@dataclass
class StreamEvent:
    """One token (or the terminal event) from a completion stream."""

    kind: str  # "token" | "reasoning" | "done" | "error"
    text: str = ""
    usage: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class Message:
    role: str
    content: str


class ChatClient:
    """Talks to ``llama-server`` over ``/v1/chat/completions``."""

    def __init__(self, base_url: str, settings: Settings, model: str = "feast") -> None:
        self.base_url = base_url.rstrip("/")
        self.settings = settings
        self.model = model
        #: llama.cpp reuses the KV cache of the longest unchanged prompt
        #: prefix. On by default, as a phone app would run it; the benchmark
        #: turns it off to measure cold prefill, which is where a long tool
        #: prompt costs the most.
        self.cache_prompt = True

    # ------------------------------------------------------------------
    def build_messages(
        self,
        history: list[Message],
        user_text: str,
        system_prompt: str | None = None,
    ) -> list[dict]:
        """Assemble the request payload, applying the history window."""
        s = self.settings
        prompt = s.system_prompt if system_prompt is None else system_prompt
        messages: list[dict] = []
        if prompt.strip():
            messages.append({"role": "system", "content": prompt.strip()})

        if s.keep_history and s.history_turns > 0:
            window = history[-(s.history_turns * 2):]
            messages += [{"role": m.role, "content": m.content} for m in window]

        messages.append({"role": "user", "content": user_text})
        return messages

    # ------------------------------------------------------------------
    def _payload(self, messages: list[dict], extra: dict | None = None) -> dict:
        s = self.settings
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "timings_per_token": True,
            "temperature": s.temperature,
            "top_p": s.top_p,
            "top_k": s.top_k,
            "repeat_penalty": s.repeat_penalty,
            "max_tokens": s.max_tokens,
            "cache_prompt": self.cache_prompt,
        }
        if s.seed is not None and s.seed >= 0:
            payload["seed"] = s.seed

        # Reasoning control. Verified per-request on llama.cpp b10144, so this
        # takes effect on the next message with no model reload. Templates
        # without an `enable_thinking` variable ignore it rather than failing,
        # which is why it is safe to send to every model.
        if s.thinking_mode in ("on", "off"):
            payload["chat_template_kwargs"] = {
                "enable_thinking": s.thinking_mode == "on"
            }
        if extra:
            payload.update(extra)
        return payload

    # ------------------------------------------------------------------
    async def stream(
        self,
        messages: list[dict],
        timeout: float = 600.0,
        extra: dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield :class:`StreamEvent` objects until the completion finishes.

        ``extra`` adds request fields for this call only (e.g. ``stop``).
        """
        usage: dict = {}
        timings: dict = {}
        url = f"{self.base_url}/v1/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=15.0)) as client:
                async with client.stream("POST", url, json=self._payload(messages, extra)) as response:
                    if response.status_code != 200:
                        body = (await response.aread()).decode("utf-8", "replace")
                        yield StreamEvent("error", error=f"HTTP {response.status_code}: {body[:400]}")
                        return

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        if chunk.get("timings"):
                            timings = chunk["timings"]

                        for choice in chunk.get("choices", []):
                            delta = choice.get("delta") or {}
                            reasoning = delta.get("reasoning_content")
                            if reasoning:
                                yield StreamEvent("reasoning", text=reasoning)
                            content = delta.get("content")
                            if content:
                                yield StreamEvent("token", text=content)

        except httpx.HTTPError as exc:
            yield StreamEvent("error", error=f"Connection error: {exc}")
            return
        except (OSError, ValueError) as exc:  # pragma: no cover - defensive
            yield StreamEvent("error", error=str(exc))
            return

        yield StreamEvent("done", usage=usage, timings=timings)

    # ------------------------------------------------------------------
    async def slot_action(self, action: str, filename: str, timeout: float = 120.0) -> dict:
        """Save or restore slot 0's KV state (needs ``--slot-save-path``).

        Returns llama.cpp's reply - ``n_written``/``n_read`` bytes and
        ``timings.save_ms``/``restore_ms`` - or ``{"error": ...}``.
        """
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/slots/0?action={action}", json={"filename": filename}
                )
                data = response.json()
                if response.status_code != 200:
                    return {"error": f"HTTP {response.status_code}: {str(data)[:200]}"}
                return data
        except (httpx.HTTPError, ValueError, OSError) as exc:
            return {"error": str(exc)}

    async def complete_once(self, messages: list[dict], max_tokens: int = 1,
                            timeout: float = 600.0) -> dict:
        """One non-streamed completion - used to prefill a prompt to snapshot."""
        payload = self._payload(messages)
        payload.update(stream=False, max_tokens=max_tokens)
        payload.pop("stream_options", None)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                return response.json() if response.status_code == 200 else {
                    "error": f"HTTP {response.status_code}: {response.text[:200]}"}
        except (httpx.HTTPError, ValueError, OSError) as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    async def tokenize_count(self, text: str, timeout: float = 30.0) -> int | None:
        """Exact token count via llama.cpp's ``/tokenize`` endpoint."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/tokenize", json={"content": text}
                )
                if response.status_code == 200:
                    return len(response.json().get("tokens", []))
        except (httpx.HTTPError, ValueError, OSError):
            return None
        return None

    # ------------------------------------------------------------------
    async def props(self, timeout: float = 15.0) -> dict:
        """Server properties: chat template, context size, model metadata."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{self.base_url}/props")
                if response.status_code == 200:
                    return response.json()
        except (httpx.HTTPError, ValueError, OSError):
            return {}
        return {}


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class ThinkFilter:
    """Split inline ``<think>...</think>`` spans out of a token stream.

    Qwen3 and SmolLM3 emit reasoning inline rather than in a separate field.
    Feeding tokens through this keeps the visible transcript clean while
    still counting reasoning tokens toward throughput.
    """

    def __init__(self) -> None:
        self.in_think = False
        self._buffer = ""

    def feed(self, text: str) -> tuple[str, str]:
        """Return ``(visible, reasoning)`` for one delta."""
        self._buffer += text
        visible, reasoning = "", ""

        while self._buffer:
            if self.in_think:
                index = self._buffer.find(THINK_CLOSE)
                if index == -1:
                    if _maybe_partial(self._buffer, THINK_CLOSE):
                        break
                    reasoning += self._buffer
                    self._buffer = ""
                else:
                    reasoning += self._buffer[:index]
                    self._buffer = self._buffer[index + len(THINK_CLOSE):]
                    self.in_think = False
            else:
                index = self._buffer.find(THINK_OPEN)
                if index == -1:
                    if _maybe_partial(self._buffer, THINK_OPEN):
                        keep = len(THINK_OPEN) - 1
                        visible += self._buffer[:-keep]
                        self._buffer = self._buffer[-keep:]
                        break
                    visible += self._buffer
                    self._buffer = ""
                else:
                    visible += self._buffer[:index]
                    self._buffer = self._buffer[index + len(THINK_OPEN):]
                    self.in_think = True
        return visible, reasoning

    def flush(self) -> str:
        rest, self._buffer = self._buffer, ""
        return "" if self.in_think else rest


def _maybe_partial(buffer: str, tag: str) -> bool:
    """True if ``buffer`` might end mid-way through ``tag``."""
    tail = buffer[-(len(tag) - 1):]
    return any(tag.startswith(tail[i:]) and tail[i:] for i in range(len(tail)))
