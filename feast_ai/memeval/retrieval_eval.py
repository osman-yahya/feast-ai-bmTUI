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
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..paths import ENCODERS_DIR, HOME
from ..encoders import EMBED_QUERY_PREFIX
from .engine.consolidate import memory_names, store_path
from .engine.embed import Encoder, EncoderSpec, Reranker, doc_text, retrieve_dense
from .engine.gate import gate
from .engine.store import MemoryStore
from .schema import load

DATA = Path(__file__).parent / "data" / "v0.1"
RECOMMEND_EXPANSION = " alerji diyet vegan vejetaryen sevmem"


def encoder_specs() -> dict[str, EncoderSpec]:
    d = ENCODERS_DIR / "modernbert-tr-embed"
    return {
        # The model card's retrieval setup: instruction on the query only.
        "mbert-asym": EncoderSpec("mbert-asym", d, query_prefix=EMBED_QUERY_PREFIX),
        # The router's finding (encoders.py): the prefix on both sides.
        "mbert-sym": EncoderSpec("mbert-sym", d, query_prefix=EMBED_QUERY_PREFIX,
                                 doc_prefix=EMBED_QUERY_PREFIX),
    }


@dataclass(frozen=True)
class Arm:
    name: str
    mode: str
    encoder: str = ""
    weight: float = 0.5
    rerank: bool = False


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
    stores = {pid: MemoryStore.load(store_path(HOME, model_id, pid)) for pid in ds.personas}
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    specs = encoder_specs()
    cache = HOME / "memstore" / model_id / "vectors"
    encoders: dict[str, Encoder] = {}
    for a in arms:
        if a.encoder and a.encoder not in encoders:
            encoders[a.encoder] = Encoder(specs[a.encoder], threads, cache)
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
        rows = []
        enc = encoders.get(a.encoder)
        for q, evidence in queries:
            lexical = q.question + (RECOMMEND_EXPANSION if gate(q.question) == "recommend" else "")
            pool = pooled if haystack == "all" else stores[q.persona].episodes
            ranked = retrieve_dense(pool, q.question, q.asked_at, 50, names,
                                    mode=a.mode, encoder=enc, lexical_query=lexical, weight=a.weight,
                                    reranker=reranker if a.rerank else None)
            ids = [e.turn_id for e in ranked]
            top = set(ids[:k])
            first = next((i + 1 for i, t in enumerate(ids) if t in evidence), None)
            rows.append({"qid": q.qid, "category": q.category, "tags": q.failure_tags,
                         "recall": sum(t in top for t in evidence) / len(evidence),
                         "hit": float(bool(top & set(evidence))), "mrr": 1 / first if first else 0.0})
        results[a.name] = summarize(rows)
        results[a.name]["rows"] = rows
        log(f"  {a.name:<22} recall@{k} {results[a.name]['recall']:.0%}  hit {results[a.name]['hit']:.0%}"
            f"  mrr {results[a.name]['mrr']:.2f}")

    cost = {"index_s": index_s, "episodes": sum(len(s.episodes) for s in stores.values()),
            "encoder_rss_mb": (rss_enc - rss0) / 2**20, "reranker_rss_mb": (rss_all - rss_enc) / 2**20,
            "threads": threads}
    for name, enc in encoders.items():
        qs = sorted(enc.query_ms)
        cost[f"{name}_query_ms_p50"] = qs[len(qs) // 2] if qs else None
        cost[f"{name}_load_ms"] = enc.load_ms
    if reranker and reranker.call_ms:
        cs = sorted(reranker.call_ms)
        cost["rerank_ms_p50"] = cs[len(cs) // 2]
    report = {"k": k, "queries": len(queries), "haystack": haystack, "cost": cost, "arms": results}
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
    out = [f"# Getirme karşılaştırması (k={report['k']}, {report['queries']} soru)\n",
           head + f"| **recall@{report['k']}** | " + " | ".join(f"**{pct(arms[n]['recall'])}**" for n in names) + " |",
           "| hit@k | " + " | ".join(pct(arms[n]["hit"]) for n in names) + " |",
           "| MRR | " + " | ".join(f"{arms[n]['mrr']:.2f}" for n in names) + " |"]
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
    import sys
    from datetime import datetime

    haystack = sys.argv[1] if len(sys.argv) > 1 else "persona"
    which = ARMS_SYM if "sym" in sys.argv[2:] else ARMS
    out = HOME / "evals" / f"retrieval-{datetime.now():%Y%m%d-%H%M%S}-{haystack}"
    run(arms=which, haystack=haystack, out=out)
    print(f"Report written to {out / 'retrieval.md'}")
