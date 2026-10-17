"""A subprocess that serves the ModernBERT-TR intent router over HTTP.

Same reasoning as :mod:`feast_ai.engine.onnx_worker`: the encoder runs in its
own process so the CPU affinity, the RAM ceiling and the telemetry apply to
*it*, not to the TUI.  On a phone the router and the SLM share one SoC, so
:class:`~feast_ai.engine.encoder_server.EncoderServer` pins this process to the
same cores as the model and the sampler adds its usage to the session.

What it does
------------
At load it embeds an *exemplar bank* - a few sample utterances per feast
intent - once.  ``/route`` then embeds the incoming message and returns, per
intent, the cosine similarity of its nearest exemplar.  The routing *policy*
(thresholds, direct calls, follow-ups) lives in :mod:`feast_ai.agent.router`;
this process only measures similarity, so the policy can change without
touching the measured hot path.

The ONNX graph returns ``last_hidden_state``; mean pooling over the attention
mask and L2 normalisation are done here, matching the model's
sentence-transformers config.  The instruction prefix is applied to queries
*and* exemplars - measured 15/16 vs 13/16 for the asymmetric variant.

Usage (not meant to be run by hand)::

    python -m feast_ai.engine.encoder_worker --model DIR --bank BANK.json \\
        --port 8081 --threads 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATE: dict = {"ready": False, "error": ""}
#: One ONNX session, one request at a time - the router is called once per
#: message, so serialising costs nothing and keeps timings clean.
LOCK = threading.Lock()


# --------------------------------------------------------------------------
def load(model_dir: Path, onnx_file: str, bank_path: Path, threads: int,
         max_length: int) -> None:
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    started = time.perf_counter()
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, threads)
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # ORT's pool spin-waits after each run by default. The SLM starts on the
    # same pinned cores right after routing, so spinning threads would steal
    # its CPU and show up as router cost in the telemetry.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(
        str(model_dir / onnx_file), options, providers=["CPUExecutionProvider"]
    )
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length)
    tokenizer.no_padding()

    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    STATE.update(
        np=np, session=session, tokenizer=tokenizer, threads=threads,
        prefix=bank.get("prefix", ""), max_length=max_length,
        model=model_dir.name, onnx_file=onnx_file,
    )
    load_ms = (time.perf_counter() - started) * 1000.0

    labels = [item["label"] for item in bank["items"]]
    texts = [item["text"] for item in bank["items"]]
    vectors, cached, bank_ms = _bank_vectors(model_dir / onnx_file, bank_path, texts)
    STATE.update(
        labels=labels, texts=texts, bank=vectors, intents=sorted(set(labels)),
        load_ms=load_ms, bank_ms=bank_ms, bank_cached=cached,
    )
    STATE["ready"] = True


def _bank_vectors(onnx_path: Path, bank_path: Path, texts: list[str]):
    """Embed the exemplars, reusing a cache keyed on everything that matters.

    On a phone these vectors would ship precomputed with the app, so paying
    for them on every launch would overstate the router's startup cost. The
    first launch still embeds them and reports how long that took.
    """
    np = STATE["np"]
    key = hashlib.sha1()
    key.update(f"{onnx_path.name}:{onnx_path.stat().st_size}\n".encode())
    key.update(STATE["prefix"].encode("utf-8"))
    for text in texts:
        key.update(b"\x00" + text.encode("utf-8"))
    cache = bank_path.with_name(f"bank-{key.hexdigest()[:12]}.npy")
    if cache.exists():
        try:
            vectors = np.load(cache)
            if vectors.shape[0] == len(texts):
                return vectors, True, 0.0
        except (OSError, ValueError):
            pass

    started = time.perf_counter()
    # Similar lengths per batch keep padding - and wasted compute - down.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    chunks = []
    for start in range(0, len(order), 16):
        idx = order[start:start + 16]
        chunks.append((idx, embed([texts[i] for i in idx])[0]))
    dim = chunks[0][1].shape[1] if chunks else 0
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for idx, block in chunks:
        vectors[idx] = block
    bank_ms = (time.perf_counter() - started) * 1000.0
    try:
        np.save(cache, vectors)
    except OSError:
        pass
    return vectors, False, bank_ms


def embed(texts: list[str]):
    """``(vectors, token_counts, tokenize_ms, infer_ms)`` for ``texts``."""
    np = STATE["np"]
    t0 = time.perf_counter()
    encodings = STATE["tokenizer"].encode_batch([STATE["prefix"] + t for t in texts])
    width = max(len(e.ids) for e in encodings)
    ids = np.zeros((len(encodings), width), dtype=np.int64)
    mask = np.zeros_like(ids)
    for row, enc in enumerate(encodings):
        ids[row, :len(enc.ids)] = enc.ids
        mask[row, :len(enc.ids)] = 1
    t1 = time.perf_counter()
    hidden = STATE["session"].run(None, {"input_ids": ids, "attention_mask": mask})[0]
    weights = mask[..., None].astype(np.float32)
    pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    t2 = time.perf_counter()
    counts = [len(e.ids) for e in encodings]
    return pooled.astype(np.float32), counts, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0


def route(text: str) -> dict:
    """Nearest-exemplar similarity per intent, best first."""
    vectors, counts, tok_ms, infer_ms = embed([text])
    sims = STATE["bank"] @ vectors[0]
    best: dict[str, tuple[float, int]] = {}
    for index, (label, score) in enumerate(zip(STATE["labels"], sims.tolist())):
        if label not in best or score > best[label][0]:
            best[label] = (score, index)
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
    top_label, (_, top_index) = ranked[0]
    return {
        "scores": [[label, round(score, 5)] for label, (score, _) in ranked],
        "nearest": STATE["texts"][top_index],
        "tokens": counts[0],
        "tokenize_ms": tok_ms,
        "infer_ms": infer_ms,
    }


# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003 - silence stderr spam
        pass

    def _json(self, code: int, payload: dict) -> None:
        blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
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
            keys = ("model", "onnx_file", "threads", "max_length", "load_ms",
                    "bank_ms", "bank_cached", "intents")
            props = {k: STATE.get(k) for k in keys}
            props["bank_size"] = len(STATE.get("labels") or [])
            self._json(200, props)
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if not STATE["ready"]:
            self._json(503, {"error": STATE["error"] or "router still loading"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError) as exc:
            self._json(400, {"error": f"bad request: {exc}"})
            return
        try:
            if self.path.startswith("/route"):
                with LOCK:
                    started = time.perf_counter()
                    result = route(str(body.get("text", "")))
                    result["total_ms"] = (time.perf_counter() - started) * 1000.0
                self._json(200, result)
            elif self.path.startswith("/embed"):
                with LOCK:
                    vectors, counts, tok_ms, infer_ms = embed(
                        [str(t) for t in body.get("texts", [])] or [""]
                    )
                self._json(200, {"vectors": vectors.tolist(), "tokens": counts,
                                 "tokenize_ms": tok_ms, "infer_ms": infer_ms})
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001 - must reach the client
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="feast-ai-encoder-worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--onnx", default="model_quantized.onnx")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), Handler)

    def boot() -> None:
        try:
            load(Path(args.model), args.onnx, Path(args.bank), args.threads,
                 args.max_length)
        except Exception as exc:  # noqa: BLE001 - reported over /health
            STATE["error"] = f"{type(exc).__name__}: {exc}"
            print(f"load failed: {STATE['error']}", file=sys.stderr, flush=True)

    threading.Thread(target=boot, daemon=True).start()
    print(f"listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
