"""Render and read Gemma 4's native turns exactly as the phone will.

Training on one prompt format and serving another is the cheapest way to
lose a fine-tune, so there is one renderer and it uses the template that is
*inside the LiteRT-LM bundle* (``assets/chat_template.jinja``, byte-identical
to ``litert-community/gemma-4-E2B-it-litert-lm``).  Google's newer canonical
template on the Hugging Face repo is not used: it calls ``dict.get``, which
LiteRT-LM's on-device Jinja does not implement.

What a model turn looks like in this format::

    <|turn>model
    <|tool_call>call:search_menu_items{max_price:150,query:<|"|>kokoreç<|"|>}<tool_call|>

Argument keys are sorted, strings sit between ``<|"|>`` tokens, and
generation ends on ``<|tool_response>`` (token 50) after a call or on
``<turn|>`` (token 106) after text - the bundle's metadata lists both as
stop tokens.  :func:`target_text` writes that ending for training, and
:func:`parse_reply` reads a generated turn back into ``(name, args)`` so the
feast AI validator and suites can judge it.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from .paths import CHAT_TEMPLATE

BOS = "<bos>"
QUOTE = '<|"|>'
CALL_OPEN, CALL_CLOSE = "<|tool_call>", "<tool_call|>"
END_TURN = "<turn|>"
#: What the model emits right after a call: the app's cue to run the tool.
END_CALL = "<|tool_response>"
RESPONSE_CLOSE = "<tool_response|>"
MODEL_TURN = "<|turn>model\n"


def tool_result(name: str, result: dict) -> dict:
    """A tool result as a history message, in the shape the template reads
    (and the shape LiteRT-LM's ``send_message`` takes)."""
    return {"role": "tool",
            "content": [{"type": "tool_response", "name": name, "response": result}]}

#: The malformed call LiteRT-LM's own parser rejects: a quote token doubled.
DOUBLED_QUOTE = re.compile(r'(?:<\|"\|>){2,}\s*[,}\]]')
_CALL = re.compile(re.escape(CALL_OPEN) + r"call:([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*?\})\s*"
                   + re.escape(CALL_CLOSE), re.S)
_THOUGHT = re.compile(r"<\|channel>.*?<channel\|>", re.S)


@lru_cache(maxsize=1)
def _template():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    # The same environment transformers builds for apply_chat_template.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(CHAT_TEMPLATE.read_text(encoding="utf-8"))


def render(messages: list[dict], tools: list[dict] | None = None,
           add_generation_prompt: bool = True) -> str:
    """The full prompt for a conversation, starting with ``<bos>``."""
    return _template().render(messages=messages, tools=tools or None, bos_token=BOS,
                              add_generation_prompt=add_generation_prompt)


# --------------------------------------------------------------------------
# Writing a call
# --------------------------------------------------------------------------
def _value(value: object) -> str:
    """One argument value, as the template's ``format_argument`` writes it."""
    if isinstance(value, str):
        return QUOTE + value + QUOTE
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{_value(v)}" for k, v in _sorted(value)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_value(v) for v in value) + "]"
    return str(value)


def _sorted(args: dict) -> list[tuple[str, object]]:
    return sorted(args.items(), key=lambda kv: kv[0].lower())  # Jinja's dictsort


def call_text(name: str, args: dict) -> str:
    body = ",".join(f"{key}:{_value(value)}" for key, value in _sorted(args))
    return f"{CALL_OPEN}call:{name}{{{body}}}{CALL_CLOSE}"


def assistant_call(name: str, args: dict) -> dict:
    """A history message holding one call, in the shape the template reads."""
    return {"role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}


def live_history(prompt_tail: str) -> str:
    """A replayed history rewritten as a live LiteRT-LM conversation holds it.

    Replaying a past call through the template closes its turn
    (``...<tool_call|><turn|>``).  A conversation that is still open never
    saw that close: generation stopped at ``<|tool_response>`` and the next
    user turn was appended straight after the call - checked with
    ``Conversation.render_message_to_string``, which renders the turn after
    a call as ``<|turn>user...`` with nothing before it.  An app may do
    either, so training shows the model both.
    """
    # The same holds after a tool result: replayed, the answer to it opens a
    # new model turn; live, the model went on writing in the turn of its call.
    return (prompt_tail.replace(CALL_CLOSE + END_TURN + "\n", CALL_CLOSE)
            .replace(RESPONSE_CLOSE + MODEL_TURN, RESPONSE_CLOSE))


def target_text(message: dict) -> str:
    """What the model must generate for this assistant message, stop token included."""
    calls = message.get("tool_calls")
    if calls:
        function = calls[0]["function"]
        return call_text(function["name"], function["arguments"]) + END_CALL
    return str(message.get("content", "")).strip() + END_TURN


# --------------------------------------------------------------------------
# Reading a reply
# --------------------------------------------------------------------------
class _Reader:
    """Recursive-descent reader for ``{key:<|"|>text<|"|>,n:5,flag:true,list:[...]}``."""

    def __init__(self, text: str) -> None:
        self.text, self.pos = text, 0

    def _skip(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \n\t":
            self.pos += 1

    def _take(self, token: str) -> bool:
        self._skip()
        if self.text.startswith(token, self.pos):
            self.pos += len(token)
            return True
        return False

    def _string(self) -> str:
        end = self.text.find(QUOTE, self.pos)
        if end < 0:
            raise ValueError("unterminated string")
        out, self.pos = self.text[self.pos:end], end + len(QUOTE)
        # A 4-bit export sometimes writes the closing quote token twice
        # ("...kafe<|"|><|"|>}"). The value is not in doubt, so it is read as
        # one quote; DOUBLED_QUOTE finds these for the report.
        while self.text.startswith(QUOTE, self.pos) and \
                self.text[self.pos + len(QUOTE):].lstrip()[:1] in (",", "}", "]"):
            self.pos += len(QUOTE)
        return out

    def value(self) -> object:
        self._skip()
        if self._take(QUOTE):
            return self._string()
        if self._take('"'):  # plain JSON quotes: a format slip the app can read
            end = self.text.find('"', self.pos)
            if end < 0:
                raise ValueError("unterminated string")
            out, self.pos = self.text[self.pos:end], end + 1
            return out
        if self._take("{"):
            return self.object_body()
        if self._take("["):
            items: list = []
            if self._take("]"):
                return items
            while True:
                items.append(self.value())
                if self._take("]"):
                    return items
                if not self._take(","):
                    raise ValueError("expected , or ] in a list")
        match = re.compile(r"[^,}\]\s]+").match(self.text, self.pos)
        if not match:
            raise ValueError("expected a value")
        self.pos = match.end()
        word = match.group(0)
        if word in ("true", "false"):
            return word == "true"
        if word in ("null", "None"):
            return None
        try:
            return json.loads(word)
        except json.JSONDecodeError:
            return word  # a bare word where a string was due: kept, judged by the validator

    def object_body(self) -> dict:
        out: dict = {}
        if self._take("}"):
            return out
        while True:
            self._skip()
            if self._take(QUOTE):
                key = self._string()
            else:
                match = re.compile(r'"?([A-Za-z_][A-Za-z0-9_]*)"?').match(self.text, self.pos)
                if not match:
                    raise ValueError("expected an argument name")
                key, self.pos = match.group(1), match.end()
            if not self._take(":"):
                raise ValueError(f"expected : after {key}")
            out[key] = self.value()
            if self._take("}"):
                return out
            if not self._take(","):
                raise ValueError("expected , or } in arguments")


def parse_args(text: str) -> dict:
    reader = _Reader(text.strip())
    if not reader._take("{"):
        raise ValueError("arguments do not start with {")
    return reader.object_body()


def clean(reply: str) -> str:
    """Drop thinking spans and the stop tokens a raw decode leaves behind."""
    reply = _THOUGHT.sub("", reply)
    for token in (END_CALL, END_TURN, "<eos>", "<end_of_turn>"):
        reply = reply.replace(token, "")
    return reply.strip()


def parse_reply(reply: str) -> tuple[str, str, dict]:
    """``("call", name, args)``, ``("text", text, {})`` or ``("error", why, {})``."""
    reply = _THOUGHT.sub("", reply)
    match = _CALL.search(reply)
    if match:
        try:
            return "call", match.group(1), parse_args(match.group(2))
        except ValueError as exc:
            return "error", f"{match.group(1)}: {exc}", {}
    if CALL_OPEN in reply or "call:" in reply.split("\n")[0][:12]:
        return "error", "a call was started but is not well formed", {}
    return "text", clean(reply), {}
