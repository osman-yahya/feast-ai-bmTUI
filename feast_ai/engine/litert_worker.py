"""A subprocess that serves a ``.litertlm`` bundle over HTTP with LiteRT-LM.

LiteRT-LM is the runtime the feast mobile app will ship, so this is the
closest the desktop gets to the phone: the same bundle, the same chat
template (it is inside the bundle), the same tool-call parser.  It runs as a
subprocess for the reasons :mod:`feast_ai.engine.onnx_worker` gives - the
core budget, the RAM ceiling and the telemetry must apply to the model, not
to the TUI.

The wire format is llama.cpp's OpenAI-compatible one, with two additions the
native agent pipeline needs:

* ``tools`` in the request are handed to the runtime as declarations, so the
  bundle's own template writes them (``<|tool>declaration:...``);
* a tool call comes back as an OpenAI ``tool_calls`` delta.  LiteRT-LM
  delivers a call in one piece when it is complete, so there is no
  token-by-token stream for it and time-to-first-token is the whole call.

**The conversation is kept, as a phone app would keep it.**  The client
sends the whole history each time, as it does to llama.cpp.  Opening a new
LiteRT-LM conversation from that history re-reads all of it - measured on
Gemma 4 E2B with a 2,137-token prompt: 8.5 s to the first token on every
turn, at 4 threads.  So the worker remembers the conversation it is in:

``live``    the request continues the last one (same prompt, same history):
            only the new message is sent, and only its tokens are read;
``clone``   a new chat with a prompt seen before: a clone of that prompt's
            base conversation, which shares the prefilled prompt (measured:
            5.6 s for the first clone, 0.3 s for later ones);
``replay``  the history differs from what the runtime holds - the app
            rewrote a turn, or cleared part of the chat: a new conversation,
            read in full.

Each reply says which of the three it was (``timings.path``).

**A call the runtime cannot parse is an exception there, and a reply here.**
LiteRT-LM raises ``Failed to parse FC tool calls`` when the model writes a
malformed call (seen with a 4-bit export: a doubled quote token).  The raw
text is in the error, so it is passed on as a broken ``<tool_call>`` for the
judge to count, instead of failing the request.

**What the timings mean.**  ``prompt_n`` is what the runtime read for this
reply; on the ``clone`` path that includes the shared prompt, which it
reports at thousands of tokens per second because it was not read again.
``prompt_ms`` is the measured wall time to the first token - what the user
waits for either way.

Usage (not meant to be run by hand)::

    python -m feast_ai.engine.litert_worker --model X.litertlm --port 8080 --threads 4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE: dict = {"ready": False, "error": "", "engine": None, "lm": None,
               "bases": {}, "chat": None, "live": None}
#: One generation at a time: the engine holds a single KV cache.
LOCK = threading.Lock()
#: Prompt variants that keep a base conversation to clone from.
MAX_BASES = 4
_PARSE_FAILED = "Failed to parse FC tool calls"
_FULL_RESPONSE = re.compile(r"full response:\s*(.*?)\s*(?:\nerror:|$)", re.S)


def load_model(model_path: str, threads: int, context: int) -> None:
    import litert_lm

    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
    backend = (litert_lm.Backend.CPU(thread_count=threads) if threads > 0
               else litert_lm.Backend.CPU())
    # No vision or audio backend is requested: a text-only bundle has no
    # encoder sections, and asking for one fails the load.
    STATE["engine"] = litert_lm.Engine(model_path, backend=backend, max_num_tokens=context,
                                       enable_benchmark=True)
    STATE.update(lm=litert_lm, threads=threads, context=context, model_path=model_path,
                 ready=True)


def recover_raw(error: str) -> str | None:
    """The model's text out of LiteRT-LM's "could not parse the call" error."""
    if _PARSE_FAILED not in error:
        return None
    found = _FULL_RESPONSE.search(error)
    return found.group(1) if found else ""


def broken_call(raw: str) -> str:
    """A malformed native call in feast AI's wire form, so the judge sees an
    attempted call (``parse_error``), not a sentence."""
    inner = raw.replace("<|tool_call>", "").replace("<tool_call|>", "").strip()
    return f"<tool_call>{inner}</tool_call>"


class _Declared:
    """A tool as the runtime wants one: a schema. The app executes, not us."""

    def __init__(self, spec: dict) -> None:
        self.spec = spec

    def get_tool_description(self) -> dict:
        return self.spec

    def execute(self, param):  # never reached: automatic tool calling is off
        return {}


def _tool_objects(specs: list[dict]) -> list:
    base = STATE.get("tool_class")
    if base is None:
        base = STATE["tool_class"] = type("Declared", (_Declared, STATE["lm"].Tool), {})
    return [base(spec) for spec in specs]


def to_runtime_messages(messages: list[dict]) -> list[dict]:
    """The client's history, with past calls back in structured form.

    feast AI keeps a call in history as its canonical text
    (``<tool_call>{json}</tool_call>``); the bundle's template renders a call
    from ``tool_calls``, so they are converted here.  A tool result arrives
    as OpenAI's ``tool`` message (``name`` + JSON ``content``) and leaves as
    the ``tool_response`` part LiteRT-LM renders
    (``<|tool_response>response:name{...}<tool_response|>``).
    """
    from ..agent.calls import extract

    out = []
    for message in messages:
        role, content = message.get("role"), message.get("content") or ""
        if role == "tool":
            if isinstance(content, str):
                try:
                    response = json.loads(content)
                except ValueError:
                    response = {"value": content}
                content = [{"type": "tool_response", "name": message.get("name") or "tool",
                            "response": response}]
            out.append({"role": "tool", "content": content})
            continue
        if role == "assistant" and message.get("tool_calls"):
            out.append({"role": "assistant", "tool_calls": message["tool_calls"]})
            continue
        if role == "assistant" and "<tool_call>" in content:
            call = extract(content)
            if call.name and call.status != "parse_error":
                out.append({"role": "assistant", "tool_calls": [{
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.args}}]})
                continue
        out.append({"role": role, "content": content})
    return out


def _fingerprint(messages: list[dict]) -> list:
    """Messages reduced to what must match for a conversation to be the same.

    Two things the app may have edited do not count, because the runtime
    cannot take an edit without re-reading the whole chat (measured: 8 s for
    a 3,000-token conversation, to change one sentence):

    * the arguments of a call - the app's guard may have repaired one
      ("trioçe" -> the user's "trileçe").  The tool ran with the repaired
      call and its result is what follows;
    * the sentence after a tool result - the app's reply check may have
      replaced it.  The result it summarised is in the conversation.

    So the runtime keeps the model's own wording of those two, and the next
    turn is ``live``.  A call the guard turned into text, or any other
    difference, is still a different conversation and is replayed.
    """
    out = []
    for index, message in enumerate(messages):
        calls = message.get("tool_calls")
        if calls:
            function = calls[0].get("function") or {}
            out.append((message.get("role"), "call", function.get("name")))
        elif message.get("role") == "assistant" and index and                 messages[index - 1].get("role") == "tool":
            out.append(("assistant", "answer"))
        elif message.get("role") == "tool":
            out.append(("tool", "result", json.dumps(message.get("content"), sort_keys=True,
                                                     ensure_ascii=False)))
        else:
            out.append((message.get("role"), "text", str(message.get("content") or "").strip()))
    return out


def _sampler(body: dict):
    lm = STATE["lm"]
    temperature = float(body.get("temperature", 0.7))
    seed = body.get("seed")
    seed = int(seed) if isinstance(seed, (int, float)) and seed >= 0 else None
    if temperature <= 0.0:
        # LiteRT-LM has no "greedy" switch; the single best token is top_k=1.
        return lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0, seed=seed or 0)
    return lm.SamplerConfig(top_k=int(body.get("top_k", 40)), top_p=float(body.get("top_p", 0.95)),
                            temperature=temperature, seed=seed)


def _open(body: dict, messages: list[dict]):
    """``(conversation, path)`` for this request - see the module docstring."""
    lm, engine = STATE["lm"], STATE["engine"]
    history = messages[:-1]
    preface = [m for m in history[:1] if m.get("role") == "system"]
    turns = history[len(preface):]
    tools = body.get("tools") or []
    thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
    key = json.dumps([preface, tools, body.get("temperature"), body.get("top_p"),
                      body.get("top_k"), body.get("seed"), body.get("max_tokens"), thinking],
                     sort_keys=True, ensure_ascii=False)

    chat = STATE.get("chat")
    if chat and chat["key"] == key and chat["seen"] == _fingerprint(history):
        return chat["conversation"], "live", key

    kwargs: dict = {
        "automatic_tool_calling": False,
        "sampler_config": _sampler(body),
        "max_output_tokens": int(body.get("max_tokens") or 512),
    }
    if tools:
        kwargs["tools"] = _tool_objects(tools)
    if thinking is not None:
        kwargs["thinking_config"] = lm.ThinkingConfig(enable_thinking=bool(thinking))
    if turns:
        return engine.create_conversation(messages=history, **kwargs), "replay", key
    bases = STATE["bases"]
    if key not in bases:
        if len(bases) >= MAX_BASES:
            bases.pop(next(iter(bases)))
        bases[key] = engine.create_conversation(messages=preface, **kwargs)
    return bases[key].clone(), "clone", key


def generate(body: dict):
    """Yield ``("text", str)``, ``("call", dict)``, ``("reasoning", str)`` then ``("done", stats)``."""
    messages = to_runtime_messages(body.get("messages") or [])
    if not messages:
        raise ValueError("no messages")
    started = time.monotonic()
    conversation, path, key = _open(body, messages)
    STATE["chat"] = None  # not reusable until this reply is complete
    STATE["live"] = conversation
    first_at = 0.0
    produced = 0
    text_parts: list[str] = []
    reply: dict | None = None
    try:
        for chunk in conversation.send_message_async(messages[-1]):
            if not first_at:
                first_at = time.monotonic()
            for call in chunk.get("tool_calls") or []:
                produced += 1
                reply = {"role": "assistant", "tool_calls": [call]}
                yield "call", call
            for _name, thought in (chunk.get("channels") or {}).items():
                if thought:
                    yield "reasoning", str(thought)
            content = chunk.get("content") or []
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            for part in content:
                if part.get("type") == "text" and part.get("text"):
                    produced += 1
                    text_parts.append(part["text"])
                    yield "text", part["text"]
    except RuntimeError as exc:
        raw = recover_raw(str(exc))
        if raw is None:
            raise
        first_at = first_at or time.monotonic()
        conversation = None  # its state after a failed parse is not worth keeping
        yield "text", broken_call(raw)
    finally:
        STATE["live"] = None

    finished = time.monotonic()
    prompt_n = predicted_n = 0
    prefill_tps = decode_tps = 0.0
    if conversation is not None:
        try:
            info = conversation.get_benchmark_info()
            prompt_n = int(info.last_prefill_token_count)
            predicted_n = int(info.last_decode_token_count)
            prefill_tps = float(info.last_prefill_tokens_per_second)
            decode_tps = float(info.last_decode_tokens_per_second)
        except (RuntimeError, AttributeError):
            predicted_n = produced
        if reply is None:
            reply = {"role": "assistant", "content": "".join(text_parts)}
        STATE["chat"] = {"key": key, "conversation": conversation,
                         "seen": _fingerprint(messages + [reply])}
    prompt_ms = ((first_at or finished) - started) * 1000.0
    decode_ms = (predicted_n / decode_tps * 1000.0) if decode_tps > 0 else \
        (finished - (first_at or started)) * 1000.0
    yield "done", {
        "usage": {"prompt_tokens": prompt_n, "completion_tokens": predicted_n,
                  "total_tokens": prompt_n + predicted_n},
        "timings": {
            "prompt_n": prompt_n,
            "prompt_ms": prompt_ms,
            "prompt_per_second": prefill_tps,
            "predicted_n": predicted_n,
            "predicted_ms": decode_ms,
            "predicted_per_second": decode_tps or (
                predicted_n / decode_ms * 1000.0 if decode_ms > 0 else 0.0),
            "path": path,
        },
    }


def _call_delta(call: dict) -> dict:
    """LiteRT-LM's parsed call as an OpenAI ``tool_calls`` delta entry."""
    function = call.get("function") or {}
    return {"index": 0, "id": "call_0", "type": "function",
            "function": {"name": function.get("name", ""),
                         "arguments": json.dumps(function.get("arguments") or {},
                                                 ensure_ascii=False)}}


# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003 - silence stderr spam
        pass

    def _json(self, code: int, payload: dict) -> None:
        blob = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.startswith("/health"):
            if STATE["error"]:
                self._json(500, {"status": "error", "error": STATE["error"]})
            elif STATE["ready"]:
                self._json(200, {"status": "ok"})
            else:
                self._json(503, {"status": "loading"})
            return
        if self.path.startswith("/props"):
            self._json(200, {"engine": "litert-lm", "threads": STATE.get("threads", 0),
                             "n_ctx": STATE.get("context", 0),
                             "model_path": STATE.get("model_path", "")})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if not STATE["ready"]:
            self._json(503, {"error": STATE["error"] or "model still loading"})
            return
        try:
            body = self._body()
        except (ValueError, OSError) as exc:
            self._json(400, {"error": f"bad request: {exc}"})
            return
        if self.path.startswith("/tokenize"):
            try:
                tokens = STATE["engine"].tokenize(str(body.get("content", "")))
            except Exception as exc:  # noqa: BLE001 - must reach the client
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._json(200, {"tokens": list(tokens)})
            return
        if not self.path.startswith("/v1/chat/completions"):
            self._json(404, {"error": "not found"})
            return
        if body.get("stream"):
            self._stream(body)
        else:
            self._once(body)

    def _once(self, body: dict) -> None:
        chunks: list[str] = []
        calls: list[dict] = []
        stats: dict = {}
        try:
            with LOCK:
                for kind, payload in generate(body):
                    if kind == "text":
                        chunks.append(payload)
                    elif kind == "call":
                        calls.append(_call_delta(payload))
                    elif kind == "done":
                        stats = payload
        except Exception as exc:  # noqa: BLE001 - must reach the client
            STATE["chat"] = None
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        message: dict = {"role": "assistant", "content": "".join(chunks)}
        if calls:
            message["tool_calls"] = calls
        self._json(200, {
            "id": "feast-litert", "object": "chat.completion",
            "model": body.get("model", "feast"),
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if calls else "stop"}],
            **stats,
        })

    def _stream(self, body: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        model_name = body.get("model", "feast")

        def send(delta: dict, finish: str | None = None, extra: dict | None = None) -> None:
            chunk = {"id": "feast-litert", "object": "chat.completion.chunk",
                     "model": model_name,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            chunk.update(extra or {})
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            with LOCK:
                finish = "stop"
                for kind, payload in generate(body):
                    if kind == "text":
                        send({"content": payload})
                    elif kind == "reasoning":
                        send({"reasoning_content": payload})
                    elif kind == "call":
                        finish = "tool_calls"
                        send({"tool_calls": [_call_delta(payload)]})
                    elif kind == "done":
                        send({}, finish, payload)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The user pressed Esc: stop decoding and forget the conversation,
            # whose state now ends mid-reply.
            STATE["chat"] = None
            live = STATE.get("live")
            if live is not None:
                try:
                    live.cancel_process()
                except Exception:  # noqa: BLE001 - nothing left to report to
                    pass
            return
        except Exception as exc:  # noqa: BLE001 - surface it in the transcript
            STATE["chat"] = None
            try:
                self.wfile.write(("data: " + json.dumps(
                    {"error": {"message": f"{type(exc).__name__}: {exc}"}}) + "\n\n")
                    .encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except OSError:
                pass


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="feast-ai-litert-worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--ctx", type=int, default=4096)
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), Handler)

    def load() -> None:
        try:
            load_model(args.model, args.threads, args.ctx)
        except Exception as exc:  # noqa: BLE001 - reported over /health
            STATE["error"] = f"{type(exc).__name__}: {exc}"
            print(f"load failed: {STATE['error']}", file=sys.stderr, flush=True)

    # Load in the background so /health answers "loading" instead of the
    # connection being refused - the manager needs to tell those apart.
    threading.Thread(target=load, daemon=True).start()
    print(f"listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
