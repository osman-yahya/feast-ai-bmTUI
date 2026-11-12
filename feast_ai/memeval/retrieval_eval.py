"""Retrieval-only comparison: which past turns reach MEMORY, no SLM involved.

Every question with ``evidence_turn_ids`` is a labelled retrieval query, so
retrievers can be compared in minutes instead of a 25-minute Gemma run.
Each retriever ranks the consolidated store's episodes (the same ones the
engine runs use) and is scored on:

- ``recall@k``  share of the evidence turns in the top ``k`` (what the
  ``raw`` arm can show the model);
- ``hit@k``     at least one evidence turn in the top ``k``;
- ``mrr``       1 / rank of the first evidence turn (top 50, else 0).

Evidence turns a KVKK erasure removed from the store are not counted; a
question whose evidence is all gone is skipped.  The gate's query expansion
for recommendations is kept for the lexical side, as in ``runner``.

A dataset version with no saved stores (v0.2 = v0.1 + filler) is replayed
from the v0.1 stores, the filler as episodes only - the episodes are what a
full consolidation would store, only the filler's facts are skipped (see
:func:`consolidate.build_store`).
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..paths import ENCODERS_DIR, HOME
from ..encoders import EMBED_QUERY_PREFIX
from .engine.consolidate import build_stores, memory_names
from .engine.embed import Encoder, EncoderSpec, Reranker, doc_text, retrieve_dense
from .engine.gate import gate, retriever_for
from .engine.retrieve import searchable
from .schema import load

DATA = Path(__file__).parent / "data" / "v0.1"
RECOMMEND_EXPANSION = " alerji diyet vegan vejetaryen sevmem"


def encoder_specs() -> dict[str, EncoderSpec]:
    d = ENCODERS_DIR / "modernbert-tr-embed"
    e5 = ENCODERS_DIR / "multilingual-e5-base"
    eg = ENCODERS_DIR / "embeddinggemma-300m"
    return {
        # The model card's retrieval setup: instruction on the query only.
        "mbert-asym": EncoderSpec("mbert-asym", d, query_prefix=EMBED_QUERY_PREFIX),
        # The router's finding (encoders.py): the prefix on both sides.
        "mbert-sym": EncoderSpec("mbert-sym", d, query_prefix=EMBED_QUERY_PREFIX,
                                 doc_prefix=EMBED_QUERY_PREFIX),
        # multilingual-e5-base (Xenova int8 ONNX): the card's query/passage
        # pair, and "query:" on both sides, its advice for symmetric tasks.
        "e5-asym": EncoderSpec("e5-asym", e5, query_prefix="query: ", doc_prefix="passage: ",
                               onnx_file="onnx/model_quantized.onnx"),
        "e5-sym": EncoderSpec("e5-sym", e5, query_prefix="query: ", doc_prefix="query: ",
                              onnx_file="onnx/model_quantized.onnx"),
        # EmbeddingGemma-300m (onnx-community export, returns sentence_embedding):
        # the card's retrieval prompts, and its sentence-similarity prompt on both sides.
        "egemma-asym": EncoderSpec("egemma-asym", eg, query_prefix="task: search result | query: ",
                                   doc_prefix="title: none | text: ", onnx_file="onnx/model_quantized.onnx"),
        "egemma-sym": EncoderSpec("egemma-sym", eg, query_prefix="task: sentence similarity | query: ",
                                  doc_prefix="task: sentence similarity | query: ",
                                  onnx_file="onnx/model_quantized.onnx"),
    }


@dataclass(frozen=True)
class Arm:
    name: str
    mode: str
    encoder: str = ""
    weight: float = 0.5
    rerank: bool = False
    prune: bool = False


ARMS = (
    Arm("bm25 (v0.1)", "bm25"),
    Arm("dense · asym", "dense", "mbert-asym"),
    Arm("dense · sym", "dense", "mbert-sym"),
    Arm("hybrid 0.7 bm25", "hybrid", "mbert-asym", 0.7),
    Arm("hybrid 0.5", "hybrid", "mbert-asym", 0.5),
    Arm("hybrid 0.3 bm25", "hybrid", "mbert-asym", 0.3),
    Arm("rrf", "rrf", "mbert-asym"),
    Arm("bm25 + rerank", "bm25", rerank=True),
    Arm("hybrid 0.5 + rerank", "hybrid", "mbert-asym", 0.5, rerank=True),
)
#: The second round: the asymmetric prefix lost to the symmetric one, the
#: reranker lost outright (and costs ~1 s per question on 6 cores).
ARMS_SYM = (
    Arm("bm25 (v0.1)", "bm25"),
    Arm("dense · sym", "dense", "mbert-sym"),
    Arm("hybrid 0.5 · sym", "hybrid", "mbert-sym", 0.5),
    Arm("hybrid 0.3 bm25 · sym", "hybrid", "mbert-sym", 0.3),
    Arm("rrf · asym", "rrf", "mbert-asym"),
    Arm("rrf · sym", "rrf", "mbert-sym"),
)
#: The long-history test (v0.2): the v0.1 baseline against the sym arms.
ARMS_LONG = (
    Arm("bm25 (v0.1)", "bm25"),
    Arm("dense · sym", "dense", "mbert-sym"),
    Arm("hybrid 0.5 · sym", "hybrid", "mbert-sym", 0.5),
    Arm("rrf · sym", "rrf", "mbert-sym"),
)
#: Other encoders under the same measurement; an arm whose model is not on
#: disk is skipped (EmbeddingGemma needs a Hugging Face licence and token).
ARMS_ENCODERS = (
    Arm("bm25 (v0.1)", "bm25"),
    Arm("rrf · mbert-sym", "rrf", "mbert-sym"),
    Arm("dense · e5-asym", "dense", "e5-asym"),
    Arm("rrf · e5-asym", "rrf", "e5-asym"),
    Arm("rrf · e5-sym", "rrf", "e5-sym"),
    Arm("dense · egemma-asym", "dense", "egemma-asym"),
    Arm("rrf · egemma-asym", "rrf", "egemma-asym"),
    Arm("rrf · egemma-sym", "rrf", "egemma-sym"),
)
#: v0.3: BM25 or RRF chosen per question (``auto``), and empty turns left out.
ARMS_ROUTE = (
    Arm("bm25 (v0.1)", "bm25"),
    Arm("rrf · sym", "rrf", "mbert-sym"),
    Arm("auto", "auto", "mbert-sym"),
    Arm("bm25 + prune", "bm25", prune=True),
    Arm("rrf + prune", "rrf", "mbert-sym", prune=True),
    Arm("auto + prune", "auto", "mbert-sym", prune=True),
)
ARM_SETS = {"all": ARMS, "sym": ARMS_SYM, "long": ARMS_LONG, "encoders": ARMS_ENCODERS,
            "route": ARMS_ROUTE}


def run(data: Path = DATA, model_id: str = "gemma-4-e2b", k: int = 8, threads: int = 6,
        arms: tuple[Arm, ...] = ARMS, out: Path | None = None, haystack: str = "persona",
        log=print) -> dict:
    """``haystack="all"`` ranks every persona's episodes for every question:
    v0.1 has ~27 turns per persona, so top-8 is a third of the haystack and
    every retriever looks good; the other users' turns are realistic
    distractors (same restaurants, same phrasing)."""
    import psutil

    ds = load(data)
    names = memory_names(ds)
    stores = asyncio.run(build_stores(None, ds, model_id, HOME, log=lambda m: None, save=False))
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    specs = encoder_specs()
    cache = HOME / "memstore" / model_id / "vectors"
    encoders: dict[str, Encoder] = {}
    encoder_rss: dict[str, float] = {}
    missing = {a.encoder for a in arms if a.encoder and not (specs[a.encoder].dir / specs[a.encoder].onnx_file).exists()}
    for name in sorted(missing):
        log(f"  skipping {name}: {specs[name].dir / specs[name].onnx_file} not found")
    arms = tuple(a for a in arms if a.encoder not in missing)
    for a in arms:
        if a.encoder and a.encoder not in encoders:
            r0 = proc.memory_info().rss
            encoders[a.encoder] = Encoder(specs[a.encoder], threads, cache)
            encoder_rss[a.encoder] = (proc.memory_info().rss - r0) / 2**20
    rss_enc = proc.memory_info().rss
    reranker = Reranker(ENCODERS_DIR / "modernbert-tr-reranker", threads) if any(a.rerank for a in arms) else None
    rss_all = proc.memory_info().rss

    queries = []
    for q in ds.questions:
        present = {e.turn_id for e in stores[q.persona].episodes}
        evidence = [t for t in q.evidence_turn_ids if t in present]
        if evidence:
            queries.append((q, evidence))
    log(f"{len(queries)} retrieval queries (of {len(ds.questions)} questions)")

    t0 = time.perf_counter()
    for enc in encoders.values():                # index every store once, as session end would
        for s in stores.values():
            enc.docs([doc_text(e) for e in s.episodes])
    index_s = time.perf_counter() - t0

    results = {}
    pooled = [e for s in stores.values() for e in s.episodes]
    for a in arms:
        rows, ms = [], []
        enc = encoders.get(a.encoder)
        for q, evidence in queries:
            decision = gate(q.question)
            lexical = q.question + (RECOMMEND_EXPANSION if decision == "recommend" else "")
            pool = pooled if haystack == "all" else stores[q.persona].episodes
            if a.prune:
                pool = searchable(pool)
            t1 = time.perf_counter()
            mode = retriever_for(q.question, q.asked_at, names, decision) if a.mode == "auto" else a.mode
            ranked = retrieve_dense(pool, q.question, q.asked_at, 50, names,
                                    mode=mode, encoder=enc, lexical_query=lexical, weight=a.weight,
                                    reranker=reranker if a.rerank else None)
            ms.append((time.perf_counter() - t1) * 1000)
            ids = [e.turn_id for e in ranked]
            top = set(ids[:k])
            first = next((i + 1 for i, t in enumerate(ids) if t in evidence), None)
            rows.append({"qid": q.qid, "category": q.category, "tags": q.failure_tags,
                         "recall": sum(t in top for t in evidence) / len(evidence),
                         "hit": float(bool(top & set(evidence))), "mrr": 1 / first if first else 0.0})
        results[a.name] = summarize(rows)
        results[a.name]["rows"] = rows
        # Ask-time cost end to end: BM25 + question embedding + fusion (doc vectors are cached).
        results[a.name]["ms_p50"] = statistics.median(ms)
        log(f"  {a.name:<22} recall@{k} {results[a.name]['recall']:.0%}  hit {results[a.name]['hit']:.0%}"
            f"  mrr {results[a.name]['mrr']:.2f}  {results[a.name]['ms_p50']:.0f} ms")

    cost = {"index_s": index_s, "episodes": sum(len(s.episodes) for s in stores.values()),
            "encoder_rss_mb": (rss_enc - rss0) / 2**20, "reranker_rss_mb": (rss_all - rss_enc) / 2**20,
            "threads": threads}
    for name, enc in encoders.items():
        qs = sorted(enc.query_ms)
        cost[f"{name}_query_ms_p50"] = qs[len(qs) // 2] if qs else None
        cost[f"{name}_load_ms"] = enc.load_ms
        cost[f"{name}_rss_mb"] = encoder_rss[name]
    if reranker and reranker.call_ms:
        cs = sorted(reranker.call_ms)
        cost["rerank_ms_p50"] = cs[len(cs) // 2]
    report = {"k": k, "queries": len(queries), "haystack": haystack, "dataset": Path(data).name,
              "cost": cost, "arms": results}
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "retrieval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "retrieval.md").write_text(render(report), encoding="utf-8")
    return report


def summarize(rows: list[dict]) -> dict:
    def mean(key, sel=lambda r: True):
        xs = [r[key] for r in rows if sel(r)]
        return statistics.mean(xs) if xs else None

    by_cat, by_tag = defaultdict(list), defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
        for t in r["tags"]:
            by_tag[t].append(r)
    return {"recall": mean("recall"), "hit": mean("hit"), "mrr": mean("mrr"),
            "by_category": {c: {"n": len(rs), "recall": statistics.mean(r["recall"] for r in rs)}
                            for c, rs in sorted(by_cat.items())},
            "by_tag": {t: {"n": len(rs), "recall": statistics.mean(r["recall"] for r in rs)}
                       for t, rs in sorted(by_tag.items())}}


def render(report: dict) -> str:
    arms = report["arms"]
    names = list(arms)
    pct = lambda x: "–" if x is None else f"{x:.0%}"
    head = "| | " + " | ".join(names) + " |\n|---|" + "---:|" * len(names) + "\n"
    out = [f"# Getirme karşılaştırması ({report.get('dataset', 'v0.1')}, k={report['k']}, "
           f"{report['queries']} soru)\n",
           head + f"| **recall@{report['k']}** | " + " | ".join(f"**{pct(arms[n]['recall'])}**" for n in names) + " |",
           "| hit@k | " + " | ".join(pct(arms[n]["hit"]) for n in names) + " |",
           "| MRR | " + " | ".join(f"{arms[n]['mrr']:.2f}" for n in names) + " |",
           "| ms/soru (p50) | " + " | ".join(f"{arms[n]['ms_p50']:.0f}" if "ms_p50" in arms[n] else "–"
                                         for n in names) + " |"]
    cats = sorted({c for n in names for c in arms[n]["by_category"]})
    out.append("\n## Kategori bazında recall@k\n\n" + head.rstrip("\n"))
    for c in cats:
        n0 = arms[names[0]]["by_category"].get(c, {}).get("n", 0)
        out.append(f"| {c} (n={n0}) | " + " | ".join(pct(arms[n]["by_category"].get(c, {}).get("recall")) for n in names) + " |")
    tags = sorted({t for n in names for t in arms[n]["by_tag"]})
    out.append("\n## Hata modu etiketleri, recall@k\n\n" + head.rstrip("\n"))
    for t in tags:
        n0 = arms[names[0]]["by_tag"].get(t, {}).get("n", 0)
        out.append(f"| {t} (n={n0}) | " + " | ".join(pct(arms[n]["by_tag"].get(t, {}).get("recall")) for n in names) + " |")
    c = report["cost"]
    out.append("\n## Maliyet\n")
    out.append("\n".join(f"- {k}: {v:.1f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in c.items()))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("haystack", nargs="?", default="persona", choices=("persona", "all"))
    ap.add_argument("arms", nargs="?", default="all", choices=tuple(ARM_SETS))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--k", default="8", help="comma-separated, one report per k")
    a = ap.parse_args()
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    for k in map(int, a.k.split(",")):
        out = HOME / "evals" / f"retrieval-{stamp}-{Path(a.data).name}-{a.haystack}-k{k}"
        run(Path(a.data), k=k, arms=ARM_SETS[a.arms], haystack=a.haystack, out=out)
        print(f"Report written to {out / 'retrieval.md'}")
