"""Dense retrieval beside BM25: an ONNX sentence encoder, an optional
cross-encoder reranker, and the fusions the retrieval eval compares.

Why not dense alone: dates are code's job (:mod:`timeparse`), names such as
"Dün Gece Burger" match better as words, and "alerjim var" / "alerjim artık
yok" sit next to each other in vector space.  What a vector adds is the
paraphrase - "et yemiyorum" answering "vejetaryen miyim" - the
``indirect_evidence`` failure tag.

Every mode keeps the rest of :func:`retrieve.retrieve`'s score unchanged
(time-window match, recency, user turn); only the relevance term differs:

==========  =====================================================
``bm25``    the v0.1 retriever, max-normalized BM25
``dense``   cosine, min-max normalized over the persona's episodes
``hybrid``  ``w·bm25 + (1-w)·dense``
``rrf``     reciprocal rank fusion of the two rankings
==========  =====================================================

``rerank`` reorders the top ``pool`` of any mode with the cross-encoder.
Vectors are computed once per text and cached next to the stores, keyed by
encoder and text hash - on a phone they would be written at session end,
with consolidation, and only the question embedded at ask time.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .retrieve import ALPHA, BETA, DELTA, GAMMA, BM25, _days, tokens
from .store import Episode
from .timeparse import resolve

MODES = ("bm25", "dense", "hybrid", "rrf")
RRF_K = 60
DATE_NOTE = re.compile(r"\s*\(=[^)]*\)")


@dataclass(frozen=True)
class EncoderSpec:
    """An ONNX sentence encoder on disk and how to prompt it."""

    id: str
    dir: Path
    query_prefix: str = ""
    doc_prefix: str = ""
    onnx_file: str = "model_quantized.onnx"
    max_length: int = 256


def doc_text(e: Episode) -> str:
    """What gets embedded: the turn with its cuisine tags, without the
    resolved-date notes - a vector cannot compare dates, the time term does."""
    return DATE_NOTE.sub("", e.text)


class Encoder:
    """Mean-pooled, L2-normalized embeddings; a graph that already returns
    ``sentence_embedding`` is used as is."""

    def __init__(self, spec: EncoderSpec, threads: int = 6, cache_dir: Path | None = None) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.spec = spec
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(spec.dir / spec.onnx_file), opts,
                                            providers=["CPUExecutionProvider"])
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.tokenizer = Tokenizer.from_file(str(spec.dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(spec.max_length)
        self.tokenizer.enable_padding()
        self.cache_path = cache_dir / f"{spec.id}.npz" if cache_dir else None
        self.cache: dict[str, np.ndarray] = {}
        if self.cache_path and self.cache_path.exists():
            with np.load(self.cache_path) as z:
                self.cache = {k: z[k] for k in z.files}
        self.query_ms: list[float] = []

    def _run(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.session.run(None, feed)
        if "sentence_embedding" in self.outputs:
            vec = out[self.outputs.index("sentence_embedding")]
        else:
            w = mask[..., None].astype(np.float32)
            vec = (out[0] * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)
        return (vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)).astype(np.float32)

    def docs(self, texts: list[str], batch: int = 32) -> np.ndarray:
        texts = [self.spec.doc_prefix + t for t in texts]
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        todo = sorted({(k, t) for k, t in zip(keys, texts) if k not in self.cache})
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            for (k, _), v in zip(chunk, self._run([t for _, t in chunk])):
                self.cache[k] = v
        if todo and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.cache_path, **self.cache)
        return np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, 1), np.float32)

    def query(self, text: str) -> np.ndarray:
        t0 = time.perf_counter()
        v = self._run([self.spec.query_prefix + text])[0]
        self.query_ms.append((time.perf_counter() - t0) * 1000)
        return v


class Reranker:
    """Cross-encoder: one forward pass per (question, turn) pair."""

    def __init__(self, model_dir: Path, threads: int = 6, max_length: int = 256) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_dir / "model_quantized.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length)
        self.tokenizer.enable_padding()
        self.call_ms: list[float] = []

    def score(self, query: str, docs: list[str]) -> list[float]:
        t0 = time.perf_counter()
        enc = self.tokenizer.encode_batch([(query, d) for d in docs])
        feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
        logits = self.session.run(None, feed)[0][:, 0]
        self.call_ms.append((time.perf_counter() - t0) * 1000)
        return [1 / (1 + math.exp(-float(x))) for x in logits]


def _minmax(xs: np.ndarray) -> np.ndarray:
    lo, hi = float(xs.min()), float(xs.max())
    return (xs - lo) / (hi - lo) if hi > lo else np.zeros_like(xs)


def _ranks(xs) -> np.ndarray:
    order = np.argsort(-np.asarray(xs), kind="stable")
    r = np.empty(len(order))
    r[order] = np.arange(1, len(order) + 1)
    return r


def retrieve_dense(episodes: list[Episode], question: str, asked_at: str, k: int = 8,
                   names: tuple[str, ...] = (), *, mode: str = "hybrid", encoder: Encoder | None = None,
                   lexical_query: str | None = None, weight: float = 0.5,
                   reranker: Reranker | None = None, pool: int = 20) -> list[Episode]:
    """Top-``k`` episodes, best first.  ``lexical_query`` lets BM25 keep the
    gate's query expansion while the encoder sees the question as asked."""
    if not episodes:
        return []
    asked = datetime.fromisoformat(asked_at)
    windows = [(s.first, s.last) for s in resolve(question, asked, names)]
    n = len(episodes)
    lex = np.zeros(n)
    if mode != "dense":
        bm = BM25([tokens(e.text) for e in episodes])
        q = tokens(lexical_query or question)
        lex = np.array([bm.score(q, i) for i in range(n)])
        lex = lex / (lex.max() or 1.0)
    den = np.zeros(n)
    if mode != "bm25":
        vecs = encoder.docs([doc_text(e) for e in episodes])
        den = _minmax(vecs @ encoder.query(question))
    if mode == "bm25":
        rel = lex
    elif mode == "dense":
        rel = den
    elif mode == "hybrid":
        rel = weight * lex + (1 - weight) * den
    else:                                   # rrf, rescaled so the best fused rank is 1.0
        fused = 1 / (RRF_K + _ranks(lex)) + 1 / (RRF_K + _ranks(den))
        rel = fused / fused.max()

    def context(i: int, e: Episode) -> float:
        age = (asked.date() - date.fromisoformat(e.ts[:10])).days
        hit = any(a <= wl and wf <= b for a, b in _days(e) for wf, wl in windows)
        return BETA * hit + GAMMA * math.exp(-max(age, 0) / 30) + DELTA * (e.role == "user")

    extra = [context(i, e) for i, e in enumerate(episodes)]
    scored = sorted(range(n), key=lambda i: (-(ALPHA * rel[i] + extra[i]), i))
    if reranker is not None:
        head = scored[:pool]
        ce = reranker.score(question, [doc_text(episodes[i]) for i in head])
        head = sorted(range(len(head)), key=lambda j: (-(ALPHA * ce[j] + extra[head[j]]), j))
        scored = [scored[j] for j in head] + scored[pool:]
    return [episodes[i] for i in scored[:k]]
