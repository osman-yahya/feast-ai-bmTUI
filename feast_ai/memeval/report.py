"""Score one or more runs and write a Markdown + JSON report comparing them.

``python -m feast_ai memeval report RUN_DIR [RUN_DIR ...]`` - each run is
scored with :mod:`feast_ai.memeval.score` (``scores.jsonl`` lands next to its
``predictions.jsonl``) and the runs become columns, keyed by condition.
Runs over several dataset versions get the version in their column name,
and ``--retrieval`` adds ``retrieval_eval`` reports as a table beside them.
The report is in Turkish: it is what goes in front of reviewers.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from .render import CATEGORY_TR
from .runner import EVALS_DIR
from .schema import CATEGORIES, load
from .score import RUBRIC, cuisine_vocab, score_row

ORDER = ("none", "oracle", "full", "always", "gated")
COND_TR = {"none": "bellek yok", "oracle": "oracle", "full": "hepsini koy",
           "always": "her zaman", "gated": "kapılı"}


def _pct(xs: list) -> str:
    xs = [x for x in xs if x is not None]
    return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "–"


def _p(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def label(meta: dict, version: bool = False) -> str:
    c = meta["config"]
    name = COND_TR.get(c["condition"], c["condition"])
    if c["condition"] in ("always", "gated"):
        name += f" · {c['arm']}" + (f" · {c['retriever']}" if c.get("retriever", "bm25") != "bm25" else "")
        name += " · tarih v2" if meta.get("store_rev", 0) >= 2 else ""
        name += " · ayıklama" if c.get("prune") else ""
        name += " · ziyaret" if c.get("visits") else ""
    if c.get("tools") == "app":
        name += " · araç"
    return f"{Path(meta['dataset']).name} · {name}" if version else name


def _order(run: dict) -> tuple:
    c = run["meta"]["config"]
    arms = ("raw", "facts", "facts+raw")
    return (Path(run["meta"]["dataset"]).name,
            ORDER.index(c["condition"]) if c["condition"] in ORDER else 99,
            arms.index(c.get("arm", "raw")) if c["condition"] in ("always", "gated") else 0,
            c.get("retriever", "bm25") != "bm25")


def score_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    ds = load(meta["dataset"])
    qs = {q.qid: q for q in ds.questions}
    vocab = cuisine_vocab(ds.questions)
    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").open(encoding="utf-8")]
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            row["result"] = score_row(row, qs[row["qid"]], vocab)
            row["failure_tags"] = qs[row["qid"]].failure_tags
            row["evidence_turn_ids"] = qs[row["qid"]].evidence_turn_ids
            fh.write(json.dumps({"qid": row["qid"], **row["result"]}, ensure_ascii=False) + "\n")
    return {"meta": meta, "rows": rows, "dataset": ds}


def summarize(run: dict, version: bool = False) -> dict:
    rows, meta = run["rows"], run["meta"]
    scored = [r for r in rows if r["result"]["score"] is not None]
    answerable = [r for r in scored if r["category"] not in ("abstention", "deleted_info")]
    done = [r for r in rows if not r.get("error")]
    by_cat = {c: [r["result"]["score"] for r in rows if r["category"] == c] for c in CATEGORIES}
    tags: dict[str, list] = {}
    for r in scored:
        for t in r["failure_tags"]:
            tags.setdefault(t, []).append(r["result"]["score"])
    rubric = [r for r in rows if r["category"] in RUBRIC]
    safety = [r for r in rows if r["category"] == "safety"]
    opinion = [r for r in rows if r["category"] == "attributed_opinion"]
    with_ev = [r for r in done if r["evidence_turn_ids"] and r["memory_turns"]]
    return {
        "condition": meta["config"]["condition"], "label": label(meta, version), "run_id": meta["run_id"],
        "dataset": Path(meta["dataset"]).name,
        "n": len(rows), "facts_eval": meta.get("facts_eval"), "consolidate_s": meta.get("consolidate_s"),
        "accuracy": sum(r["result"]["score"] for r in scored) / len(scored) if scored else 0.0,
        "by_category": {c: (sum(v) / len(v) if v and v[0] is not None else None, len(v))
                        for c, v in by_cat.items() if v},
        "by_tag": {t: sum(v) / len(v) for t, v in tags.items()},
        "hallucinated": sum(r["result"]["score"] == 0 and not r["result"]["abstained"]
                            for r in answerable) / max(1, len(answerable)),
        "abstained_answerable": sum(r["result"]["abstained"] for r in answerable) / max(1, len(answerable)),
        "cites_evidence": (sum(r["result"]["checks"].get("cites_evidence", False) for r in with_ev)
                           / len(with_ev)) if with_ev else None,
        "evidence_recall": (statistics.mean([r["evidence_in_memory"] for r in rows if r["evidence_turn_ids"]])
                            if any(r["evidence_turn_ids"] for r in rows) else None),
        "gate_closed": (sum(r.get("gate") == "closed" for r in rows) / len(rows))
                       if rows and rows[0].get("gate") else None,
        "ungrounded": sum(bool(r["result"]["checks"].get("ungrounded")) for r in rubric) / max(1, len(rubric)),
        "unsafe": sum(bool(r["result"]["checks"].get("unsafe_named")) for r in safety) / max(1, len(safety)),
        "authors_cited": statistics.mean([r["result"]["checks"].get("authors_cited") or 0 for r in opinion])
                         if opinion else None,
        "prompt_tokens": statistics.mean([r["prompt_tokens"] for r in done]) if done else 0,
        "memory_tokens": statistics.mean([r["tokens"].get("memory", 0) for r in rows]) if rows else 0,
        "max_prompt_tokens": max((sum(r["tokens"].values()) for r in rows), default=0),
        "ttft_p50": _p([r["ttft_s"] for r in done], 0.5),
        "ttft_p95": _p([r["ttft_s"] for r in done], 0.95),
        "decode_tps": statistics.mean([r["decode_tps"] for r in done]) if done else 0,
        "overflow": meta.get("overflow", 0),
        "peak_rss_gb": (meta.get("peak_rss") or 0) / 1e9,
    }


def _examples(runs: list[dict], per_cat: int = 1) -> list[str]:
    """A failure per scored category, preferring oracle (the evidence was there)."""
    pick = sorted(runs, key=lambda r: r["meta"]["config"]["condition"] != "oracle")
    out, seen = [], set()
    for run in pick:
        cond = run["meta"]["config"]["condition"]
        for r in run["rows"]:
            s = r["result"]["score"]
            if s is None or s >= 1 or r["category"] in seen or r.get("error"):
                continue
            seen.add(r["category"])
            reply = " ".join(r["reply"].split())[:260]
            out.append(f"**{CATEGORY_TR.get(r['category'], r['category'])}** · {cond} · `{r['qid']}`  \n"
                       f"Soru: {r['question']}  \nBeklenen: {r['gold_answer']}  \n"
                       f"Model: {reply}  \nPuan: {s:.2f} {json.dumps(r['result']['checks'], ensure_ascii=False)}\n")
    return out


def _judged(runs: list[dict], head: str, line) -> list[str]:
    """The LLM judge's scores beside the code's, for the runs that were judged."""
    from .judge import best_judgments, summarize as judge_summary

    cells = []
    for run in runs:
        name, verdicts = best_judgments(EVALS_DIR / run["meta"]["run_id"])
        rows = [r for r in run["rows"] if r["qid"] in verdicts]
        s = judge_summary(run["dataset"], rows, verdicts) if rows else None
        waiting = sum(v.get("escalate_pending", False) and v["category"] in RUBRIC for v in verdicts.values())
        cells.append((name, s, len(run["rows"]), waiting))
    if not any(s for _, s, _, _ in cells):
        return []
    judges = sorted({name for name, s, _, _ in cells if s})

    def cell(fn) -> list[str]:
        return [fn(s) if s else "–" for _, s, _, _ in cells]

    out = ["\n## Hakem puanı (LLM)\n",
           f"Hakem: `{'`, `'.join(judges)}` (a+b: a her cevabı puanlar, b rubrikli kategorileri yeniden "
           "puanlar ve onun kararı geçerlidir). Hakem cevabı altın cevap, tolerans notu ve gereken "
           "uygulama verisiyle (menü, kullanıcının güncel olguları) görür; 0 / 0,5 / 1 verir.\n\n", head,
           line("**Genel (kodun da puanladığı 8 kategori)**", cell(lambda s: f"**{_pct([s['accuracy']])}**")),
           line("Genel (11 kategori, rubrikliler dahil)", cell(lambda s: _pct([s["accuracy_all"]])))]
    for c in CATEGORIES:
        if any(s and c in s["by_category"] for _, s, _, _ in cells):
            out.append(line(CATEGORY_TR.get(c, c) + (" (rubrik)" if c in RUBRIC else ""),
                            cell(lambda s: _pct([s["by_category"][c][0]]) if c in s["by_category"] else "–")))
    out += [line("Kod–hakem uyumu (aynı puan)", cell(lambda s: _pct([s["agreement"]]))),
            line("Puanlanan cevap", [f"{s['n']}/{n}" if s else "–" for _, s, n, _ in cells]),
            line("Üst hakemi bekleyen rubrikli cevap", [str(w) if s else "–" for _, s, _, w in cells])]
    return out


def _write_path(sums: list[dict]) -> list[str]:
    versions = sorted({s["dataset"] for s in sums})
    out = []
    for version in versions:
        fe = max((s for s in sums if s["facts_eval"] and s["dataset"] == version),
                 key=lambda s: s["consolidate_s"] or 0, default=None)
        if fe:
            out.append(_write_path_line(fe, version if len(versions) > 1 else ""))
    return ["\n## Yazma yolu (konsolidasyon, facts_gold'a göre)\n\n", *out] if out else []


def _write_path_line(fe: dict, version: str) -> str:
    f = fe["facts_eval"]
    return ((f"**{version}:** " if version else "")
            + f"Oturum sonu çıkarımı aynı model yapar, kod kanonikleştirir ve uygular. "
            f"Olgu kesinliği **{100 * f['precision']:.0f}%**, kapsama **{100 * f['recall']:.0f}%** "
            f"({f['tp']} doğru, {f['fp']} fazla, {f['fn']} eksik); eşleşen olguların son durumu "
            f"(açık/kapalı/silindi) {100 * f['status_accuracy']:.0f}% doğru. "
            f"{f['facts']} olgu, {f['episodes']} bölüm; konsolidasyon "
            + (f"{fe['consolidate_s']:.0f} s ({fe['consolidate_s'] / max(1, f.get('sessions', 0)):.1f} s/oturum).\n\n"
               if (fe["consolidate_s"] or 0) > 1 else "bu koşularda önbellekten okundu.\n\n"))


def _retrieval(paths: list[Path]) -> list[str]:
    """``retrieval_eval`` reports, one row per (dataset, k, retriever)."""
    if not paths:
        return []
    out = ["\n## Getirme (Gemma'sız, etiketli kanıt turları)\n\n",
           "Her soru kendi personasının bölümleri içinde sıralanır; recall@k = kanıt turlarının "
           "ilk k'daki payı. Süre: soru başına BM25 + soru vektörü + birleştirme (tur vektörleri "
           "oturum sonunda bir kez hesaplanır).\n\n",
           "| Veri | k | Getirici | recall@k | hit@k | MRR | dolaylı kanıt | güvenlik | ms/soru |\n"
           "|---|---:|---|---:|---:|---:|---:|---:|---:|\n"]
    for p in paths:
        r = json.loads((p / "retrieval.json" if p.is_dir() else p).read_text(encoding="utf-8"))
        for name, a in r["arms"].items():
            ind = a["by_tag"].get("indirect_evidence", {}).get("recall")
            saf = a["by_category"].get("safety", {}).get("recall")
            out.append(f"| {r.get('dataset', 'v0.1')} | {r['k']} | {name} | {_pct([a['recall']])} | "
                       f"{_pct([a['hit']])} | {a['mrr']:.2f} | {_pct([ind])} | {_pct([saf])} | "
                       + (f"{a['ms_p50']:.0f}" if "ms_p50" in a else "–") + " |\n")
    return out


def write_report(run_dirs: list[Path], out: Path | None = None, retrieval: list[Path] = ()) -> Path:
    runs = [score_run(Path(d)) for d in run_dirs]
    runs.sort(key=_order)
    versions = sorted({Path(r["meta"]["dataset"]).name for r in runs})
    sums = [summarize(r, len(versions) > 1) for r in runs]
    out = out or EVALS_DIR / f"report-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    m0, ds = runs[0]["meta"], runs[0]["dataset"]
    cols = [s["label"] for s in sums]
    head = "| | " + " | ".join(cols) + " |\n|---|" + "---:|" * len(cols) + "\n"

    def line(label: str, values) -> str:
        return f"| {label} | " + " | ".join(values) + " |\n"

    sizes = {Path(r["meta"]["dataset"]).name: r["dataset"] for r in runs}
    md = [f"# Mustachio bellek değerlendirmesi — {m0['model_name']}\n",
          "Veri: " + " · ".join(f"`{v}` {sum(len(x) for x in d.sessions.values())} oturum / "
                                f"{sum(len(s.turns) for x in d.sessions.values() for s in x)} tur"
                                for v, d in sizes.items())
          + f" · {len(ds.personas)} persona · {len(ds.questions)} soru  \n"
          f"Model: {m0['model_name']} ({m0['quant']}) · profil {m0['profile']} · "
          f"{m0['threads']} çekirdek · bağlam {m0['context_size']} · greedy (temperature 0)  \n"
          f"Araçlar: {', '.join(sorted({r['meta']['config']['tools'] for r in runs}))} · koşular: " + ", ".join(f"`{s['run_id']}`" for s in sums) + "\n",
          "Koşullar: **bellek yok** = MEMORY boş; **oracle** = yalnız kanıtı içeren oturumlar "
          "(okuma tavanı); **hepsini koy** = tüm geçmiş ham; **her zaman** = bellek motoru, kapısız; "
          "**kapılı** = bellek motoru, müdahale kapısıyla. Kollar: raw = getirilen ham turlar "
          "(tarihleri kodla çözülmüş), facts = konsolide olgular, facts+raw = ikisi. Getirici: "
          "varsayılan BM25 (v0.1), rrf = BM25 ile ModernBERT-TR vektör sıralamasının RRF birleşimi. "
          "Prompt'un geri kalanı koşullar arasında bayt bayt aynıdır. **tarih v2** = turun satır "
          "tarihi anlatılan gün, soru içindeki zaman ifadesi de çözülmüş; **ziyaret** = soruyla ilgili "
          "ziyaret kaydı satırları (v) MEMORY'de; **araç** = sayı soran sorulara my_visits / "
          "search_saved sunulur, çağrı gerçekten çalıştırılır ve model sonuçtan cevaplar.\n",
          "## Doğruluk (deterministik puan)\n", head,
          line("**Genel (puanlanan 8 kategori)**", [f"**{100 * s['accuracy']:.0f}%**" for s in sums])]
    for c in CATEGORIES:
        if c in RUBRIC:
            continue
        n = next((s["by_category"][c][1] for s in sums if c in s["by_category"]), 0)
        md.append(line(f"{CATEGORY_TR.get(c, c)} (n={n})",
                       [f"{100 * s['by_category'][c][0]:.0f}%" if c in s["by_category"] else "–"
                        for s in sums]))
    md += [*_judged(runs, head, line), "\n## Davranış\n", head,
           line("Uydurma (cevaplanabilir soruda yanlış ve emin)", [_pct([s["hallucinated"]]) for s in sums]),
           line("Cevaplanabilir soruda 'bilmiyorum'", [_pct([s["abstained_answerable"]]) for s in sums]),
           line("Doğru kanıt turuna atıf (tN / fN)", [_pct([s["cites_evidence"]]) for s in sums]),
           line("Kanıt turu MEMORY'de (retrieval recall)", [_pct([s["evidence_recall"]]) for s in sums]),
           line("Kapının belleği kapattığı soru", [_pct([s["gate_closed"]]) for s in sums]),
           "\n## Hata modları (etiketli sorularda doğruluk)\n", head]
    for t in sorted({t for s in sums for t in s["by_tag"]}):
        md.append(line(t, [_pct([s["by_tag"].get(t)]) for s in sums]))
    md += [*_retrieval(list(retrieval)), *_write_path(sums), "\n## Rubrikli kategoriler — sert alt kontroller\n",
           "Tercih, güvenlik ve atıflı görüş tam puanı hakem aşamasında; burada yalnız koddan "
           "ölçülebilen ihlaller.\n\n", head,
           line("Atıfsız övgü ('meşhur', 'efsane'…)", [_pct([s["ungrounded"]]) for s in sums]),
           line("Güvenlikte yasak kalemi önerme", [_pct([s["unsafe"]]) for s in sums]),
           line("Atıflı görüşte yazarı (@) anma", [_pct([s["authors_cited"]]) for s in sums]),
           "\n## Verimlilik\n", head,
           line("Ort. prompt token", [f"{s['prompt_tokens']:.0f}" for s in sums]),
           line("Ort. MEMORY token", [f"{s['memory_tokens']:.0f}" for s in sums]),
           line("En büyük prompt", [f"{s['max_prompt_tokens']}" for s in sums]),
           line("Bağlama sığmayan soru", [str(s["overflow"]) for s in sums]),
           line("TTFT p50 / p95 (s)", [f"{s['ttft_p50']:.1f} / {s['ttft_p95']:.1f}" for s in sums]),
           line("Decode (token/s)", [f"{s['decode_tps']:.1f}" for s in sums]),
           line("Tepe RAM (GB)", [f"{s['peak_rss_gb']:.2f}" for s in sums]),
           "\nTTFT masaüstü CPU'da çekirdek sınırıyla ölçüldü; telefonla mutlak değil göreli "
           "karşılaştırma içindir. Sorular persona sırasıyla koşulur ve llama.cpp önek cache'i "
           "açıktır: MEMORY bloğu bir personanın bütün sorularında aynı kalan koşulda (hepsini koy) "
           "önek önbellekten gelir, TTFT düşer; soruya göre değişen blokta (oracle, bellek motoru) "
           "her soru yeniden prefill öder.\n",
           "\n## Örnek hatalar\n", *_examples(runs),
           "\n## Notlar\n",
           "- Bu koşularda araçlar kapalı: tercih/güvenlik/atıflı görüş ve 'kaç restorana gittim' "
           "soruları uygulama verisi (menü, post, ziyaret kaydı) gerektirir; bu kategorilerdeki düşük puan "
           "beklenen tabandır.\n",
           "- Puanlama deterministik (kod); tercih gibi rubrikli kategoriler için hakem, ~40 elle "
           "etiketlenmiş cevapla doğrulandıktan sonra eklenecek.\n",
           "- 'Hepsini koy' KVKK silmelerini bilerek uygulamaz: silinen bilgi sızıntısı bu koşulun "
           "beklenen hatasıdır, bellek motoru bunu düzeltmelidir.\n",
           "- Bilinen karışıklık (v0.1): CONTEXT'teki konum etiketi personanın güncel semti. "
           "Bu yüzden 'hangi semtte oturuyorum' soruları bellek olmadan da cevaplanabiliyor ve evini "
           "silen kullanıcıda konum adı geçebiliyor. v0.2 (v0.1 + şablon dolgu) bunu devralır; "
           "GPS konumu ile ev semtinin ayrılması sonraki sürüme kaldı.\n"]
    if "v0.2" in versions:
        md.append("- v0.2 = v0.1 + persona başına 50 şablon dolgu oturumu (saat, fiyat, konum, kampanya "
                  "soruları; `gen/filler.py`). Sorular ve altın olgular v0.1 ile aynıdır; dolgu hiçbir "
                  "cevabı değiştirmeyecek biçimde kurallıdır. v0.1 oturumlarının konsolidasyonu kayıtlı "
                  "op'lardan yeniden oynatılır, dolgu oturumları Gemma ile konsolide edilir.\n")
    # Blank line before every heading, table and paragraph that follows a block.
    text = re.sub(r"\n(?=## |\| \| |Koşullar:|Tercih, güvenlik|TTFT masaüstü|\*\*)", "\n\n", "".join(md))
    (out / "report.md").write_text(re.sub(r"\n{3,}", "\n\n", text), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(sums, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
