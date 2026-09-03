"""A subprocess that serves an ONNX Runtime GenAI model over HTTP.

Why a subprocess rather than an in-process call
-----------------------------------------------
``onnxruntime_genai`` is a Python library, so the obvious thing would be to
call it directly from the TUI.  That would break every measurement feast AI
exists to make:

* CPU affinity would pin the *TUI* to the device's core budget, not the model;
* the RAM ceiling would cap the TUI's working set, evicting Textual's widgets
  along with the weights;
* :mod:`feast_ai.metrics.sampler` measures a process tree, and there would be
  no separate tree to measure.

Running the model in its own process keeps all three honest and lets the ONNX
backend reuse the constraint machinery built for ``llama-server`` unchanged.

The wire format deliberately mimics llama.cpp's OpenAI-compatible endpoint -
including its non-standard ``timings`` object - so
:class:`feast_ai.engine.client.ChatClient` drives either backend with no
branching.

Usage (not meant to be run by hand)::

    python -m feast_ai.engine.onnx_worker --model DIR --port 8080 --threads 4
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATE: dict = {"ready": False, "error": "", "model": None, "tokenizer": None}
#: ONNX Runtime holds one mutable generator per request; serialise them so two
#: concurrent requests cannot interleave into the same session.
LOCK = threading.Lock()


# --------------------------------------------------------------------------
def find_chat_template(model_dir: Path) -> str:
    """Locate the model's chat template.

    Two conventions are in the wild and feast AI's catalog contains both:
    most repos inline the template under ``chat_template`` in
    ``tokenizer_config.json`` (Llama 3.2, Phi-3), while others ship a
    side-car ``chat_template.jinja`` (Qwen3).  The runtime only reads the
    first, so the second has to be passed in explicitly - without it it
    raises "Empty chat template".
    """
    side_car = model_dir / "chat_template.jinja"
    if side_car.exists():
        try:
            text = side_car.read_text(encoding="utf-8").strip()
            if text:
                return text
        except OSError:
            pass
    config = model_dir / "tokenizer_config.json"
    if config.exists():
        try:
            inline = json.loads(config.read_text(encoding="utf-8")).get("chat_template")
            if isinstance(inline, str) and inline.strip():
                return inline
        except (OSError, json.JSONDecodeError):
            pass
    return ""


def load_model(model_dir: Path, threads: int) -> None:
    """Load the model onto CPU, pinned to ``threads`` intra-op threads."""
    import onnxruntime_genai as og

    config = og.Config(str(model_dir))
    # Never let ORT pick a GPU/NPU provider: feast AI is a CPU inference
    # simulator, and a silent DirectML fallback would invalidate every number.
    config.clear_providers()
    STATE["model"] = og.Model(config)
    STATE["tokenizer"] = og.Tokenizer(STATE["model"])
    STATE["og"] = og
    STATE["threads"] = threads
    STATE["template"] = find_chat_template(model_dir)
    STATE["ready"] = True


def build_prompt(messages: list[dict], enable_thinking: bool | None) -> str:
    """Render chat messages with the model's own template.

    A wrong prompt does not fail loudly - the model simply emits fluent
    nonsense - so a missing template raises instead of falling back to a
    hand-rolled ``role: content`` transcript.  That fallback was measurably
    worse than useless: it produced degenerate, repeating output that looked
    like a model quality problem rather than a prompting bug.
    """
    tokenizer = STATE["tokenizer"]
    template = STATE.get("template") or ""
    payload = json.dumps(messages)

    if template:
        prompt = tokenizer.apply_chat_template(
            template_str=template, messages=payload, add_generation_prompt=True
        )
    else:
        prompt = tokenizer.apply_chat_template(
            messages=payload, add_generation_prompt=True
        )

    # Reasoning control. ORT does not forward arbitrary template variables,
    # so `enable_thinking=False` is applied the way Qwen's own template does
    # it - by pre-filling an empty think block the model then continues past.
    if enable_thinking is False and "enable_thinking" in template:
        prompt += "<think>\n\n</think>\n\n"
    return prompt


def generate(body: dict):
    """Yield ``(text, done, stats)`` for one completion."""
    og = STATE["og"]
    model, tokenizer = STATE["model"], STATE["tokenizer"]

    messages = body.get("messages") or []
    max_tokens = int(body.get("max_tokens") or 512)
    kwargs = body.get("chat_template_kwargs") or {}
    enable_thinking = kwargs.get("enable_thinking")

    prompt = build_prompt(messages, enable_thinking)
    started = time.monotonic()
    input_tokens = tokenizer.encode(prompt)
    prompt_n = len(input_tokens)

    temperature = float(body.get("temperature", 0.7))
    params = og.GeneratorParams(model)
    options = {
        "max_length": prompt_n + max_tokens,
        # Without a repetition penalty greedy decoding falls into loops on
        # small models; llama.cpp applies one by default, so matching it keeps
        # the two backends comparable.
        "repetition_penalty": float(body.get("repeat_penalty", 1.1)),
        # ORT treats temperature 0 as still-sampling, so switch to greedy
        # explicitly rather than relying on it to mean deterministic.
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        options["temperature"] = temperature
        options["top_p"] = float(body.get("top_p", 0.95))
        options["top_k"] = int(body.get("top_k", 40))
    params.set_search_options(**options)

    generator = og.Generator(model, params)
    stream = tokenizer.create_stream()
    generator.append_tokens(input_tokens)

    prompt_end = time.monotonic()
    first_token_at = 0.0
    predicted_n = 0

    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        if not first_token_at:
            first_token_at = time.monotonic()
        predicted_n += 1
        text = stream.decode(token)
        if text:
            yield text, False, None
        if predicted_n >= max_tokens:
            break

    finished = time.monotonic()
    prompt_ms = (prompt_end - started) * 1000.0
    decode_ms = (finished - (first_token_at or prompt_end)) * 1000.0
    yield "", True, {
        "usage": {
            "prompt_tokens": prompt_n,
            "completion_tokens": predicted_n,
            "total_tokens": prompt_n + predicted_n,
        },
        # llama.cpp's vocabulary, so the UI needs no special case.
        "timings": {
            "prompt_n": prompt_n,
            "prompt_ms": prompt_ms,
            "prompt_per_second": (prompt_n / prompt_ms * 1000.0) if prompt_ms > 0 else 0.0,
            "predicted_n": predicted_n,
            "predicted_ms": decode_ms,
            "predicted_per_second": (
                (predicted_n / decode_ms * 1000.0) if decode_ms > 0 else 0.0
            ),
        },
    }


# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003 - silence stderr spam
        pass

    # ------------------------------------------------------------------
    def _json(self, code: int, payload: dict) -> None:
        blob = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

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
            self._json(200, {
                "engine": "onnxruntime-genai",
                "threads": STATE.get("threads", 0),
                "model_type": str(getattr(STATE.get("model"), "type", "")),
            })
            return
        self._json(404, {"error": "not found"})

    # ------------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if not self.path.startswith("/v1/chat/completions"):
            self._json(404, {"error": "not found"})
            return
        if not STATE["ready"]:
            self._json(503, {"error": STATE["error"] or "model still loading"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError) as exc:
            self._json(400, {"error": f"bad request: {exc}"})
            return

        if body.get("stream"):
            self._stream(body)
        else:
            self._once(body)

    # ------------------------------------------------------------------
    def _once(self, body: dict) -> None:
        chunks: list[str] = []
        stats: dict = {}
        try:
            with LOCK:
                for text, done, payload in generate(body):
                    if done:
                        stats = payload or {}
                    elif text:
                        chunks.append(text)
        except Exception as exc:  # noqa: BLE001 - must reach the client
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._json(200, {
            "id": "feast-onnx",
            "object": "chat.completion",
            "model": body.get("model", "feast"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "".join(chunks)},
                "finish_reason": "stop",
            }],
            **stats,
        })

    # ------------------------------------------------------------------
    def _stream(self, body: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        model_name = body.get("model", "feast")

        def send(chunk: dict) -> None:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            with LOCK:
                for text, done, payload in generate(body):
                    if done:
                        send({
                            "id": "feast-onnx",
                            "object": "chat.completion.chunk",
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {},
                                         "finish_reason": "stop"}],
                            **(payload or {}),
                        })
                    elif text:
                        send({
                            "id": "feast-onnx",
                            "object": "chat.completion.chunk",
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": text},
                                         "finish_reason": None}],
                        })
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The user pressed Esc; the generator is abandoned with the request.
            return
        except Exception as exc:  # noqa: BLE001 - surface it in the transcript
            try:
                send({"error": {"message": f"{type(exc).__name__}: {exc}"}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except OSError:
                pass


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="feast-ai-onnx-worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)

    if args.threads > 0:
        # ORT reads these at session creation, so they must be set before the
        # model loads. Belt and braces alongside the affinity mask: the mask
        # caps what the OS will schedule, these cap what ORT even creates.
        import os

        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", str(args.threads))

    server = ThreadingHTTPServer((args.host, args.port), Handler)

    def load() -> None:
        try:
            load_model(Path(args.model), args.threads)
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
