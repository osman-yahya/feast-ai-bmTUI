"""A dataset as Markdown, one page per persona, for hand review.

LongMemEval hand-edited most of its sessions; reviewing JSON lines is how
that step gets skipped.  Each page puts the profile, the gold facts, every
session with its turns, and the questions side by side, and marks each
evidence turn with the questions that rely on it, so a reviewer can check a
gold answer against the exact turn it comes from.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .schema import Dataset

DAY_TR = ("Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz")
CATEGORY_TR = {
    "user_info": "Kullanıcı bilgisi", "assistant_info": "Asistan bilgisi",
    "preference": "Tercih", "multi_session": "Oturumlar arası",
    "knowledge_update": "Bilgi güncelleme", "temporal": "Zamansal",
    "abstention": "Cevap vermeme", "safety": "Güvenlik",
    "app_data": "Uygulama verisi", "deleted_info": "Silinen bilgi",
    "attributed_opinion": "Atıflı görüş",
}


def _when(stamp: str) -> str:
    d = datetime.fromisoformat(stamp)
    return f"{d:%d.%m.%Y} {DAY_TR[d.weekday()]} {d:%H:%M}"


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def persona_page(ds: Dataset, persona_id: str) -> str:
    p = ds.personas[persona_id]
    names = {r.id: r.name for r in ds.world.restaurants}
    questions = [q for q in ds.questions if q.persona == persona_id]
    cited: dict[str, list[str]] = {}
    for q in questions:
        for tid in q.evidence_turn_ids:
            cited.setdefault(tid, []).append(q.qid.rsplit("_", 1)[-1])

    out = [f"# {p.name} (`{p.id}`)", ""]
    out += ["## Uygulama durumu", "",
            f"- **Ev:** {p.home_area}",
            f"- **Arkadaşlar:** {', '.join(p.friends)}",
            "- **Kaydedilenler:** " + (", ".join(
                f"{names[s['restaurant_id']]} ({s['saved_at'][:10]})" for s in p.saved_restaurants) or "—"),
            "- **Gidilenler (uygulama):** " + (", ".join(
                f"{names[v['restaurant_id']]} ({v['visited_at'][:10]})" for v in p.visited_restaurants) or "—"),
            "- **Paylaşılanlar:** " + (", ".join(
                f"{names.get(m.get('restaurant_id'), m.get('restaurant_id'))} → {m['friend']}"
                for m in p.shared_messages) or "—")]
    if p.memory_deletions:
        out.append("- **Hafızadan sildikleri:** " + ", ".join(
            f"{d['key']}={d['value']} ({d['at'][:10]})" for d in p.memory_deletions))
    out.append("")

    out += ["## Beklenen olgular (gold)", "",
            "| Anahtar | Değer | Tür | Geçerlilik | Durum | Kaynak tur |",
            "|---|---|---|---|---|---|"]
    for f in ds.facts.get(persona_id, []):
        until = f" → {f.valid_until}" if f.valid_until else " →"
        out.append(f"| {f.key} | {f.value} | {f.kind} | {f.valid_from}{until} | {f.status} "
                   f"| `{f.source_turn_id}` |")
    out.append("")

    out += ["## Sohbet geçmişi", "",
            "> 📌 = bir sorunun kanıtı olan tur (yanında soru numaraları)", ""]
    for s in ds.sessions.get(persona_id, []):
        out += [f"### Oturum {s.session_id.rsplit('_', 1)[-1]} — {_when(s.started_at)}", ""]
        for t in s.turns:
            who = "🧑 **Kullanıcı**" if t.role == "user" else "🥸 **Mustachio**"
            mark = f" 📌 `{', '.join(cited[t.turn_id])}`" if t.turn_id in cited else ""
            out.append(f"- {who}{mark}: {t.text}  ")
            out.append(f"  <sub>`{t.turn_id}`</sub>")
        if s.recommended:
            out.append("")
            out.append("  *Öneri kaydı:* " + ", ".join(
                f"{names.get(r['id'], r['id'])} → {r['outcome']}" for r in s.recommended))
        out.append("")

    out += ["## Sorular", ""]
    for q in questions:
        num = q.qid.rsplit("_", 1)[-1]
        out += [f"### {num} · {CATEGORY_TR.get(q.category, q.category)}"
                + (f" · `{', '.join(q.failure_tags)}`" if q.failure_tags else ""), "",
                f"**Soru:** {q.question}  ",
                f"**Beklenen cevap:** {q.gold_answer}  "]
        if q.tolerance:
            out.append(f"**Tolerans / rubrik:** {q.tolerance}  ")
        if q.hard_constraints:
            out.append(f"**Sert kısıt:** {q.hard_constraints}  ")
        if q.expected_tool_calls:
            out.append("**Beklenen araç:** " + ", ".join(f"`{c['tool']}`" for c in q.expected_tool_calls) + "  ")
        if q.evidence_turn_ids:
            out.append("**Kanıt:** " + ", ".join(f"`{t}`" for t in q.evidence_turn_ids) + "  ")
        elif q.should_abstain:
            out.append("**Kanıt:** yok — model cevap vermemeli  ")
        out.append("")
    return "\n".join(out)


def index_page(ds: Dataset) -> str:
    out = ["# memeval veri seti — inceleme", "",
           f"{len(ds.personas)} persona · {sum(map(len, ds.sessions.values()))} oturum · "
           f"{len(ds.questions)} soru", "",
           "| Persona | Ev | Oturum | Soru |", "|---|---|---|---|"]
    for pid, p in ds.personas.items():
        n = sum(q.persona == pid for q in ds.questions)
        out.append(f"| [{p.name}]({pid}.md) | {p.home_area} | {len(ds.sessions.get(pid, []))} | {n} |")
    out += ["", "| Kategori | Soru |", "|---|---|"]
    for key, label in CATEGORY_TR.items():
        out.append(f"| {label} | {sum(q.category == key for q in ds.questions)} |")
    return "\n".join(out) + "\n"


def render(ds: Dataset, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "README.md"]
    paths[0].write_text(index_page(ds), encoding="utf-8")
    for pid in ds.personas:
        path = out_dir / f"{pid}.md"
        path.write_text(persona_page(ds, pid) + "\n", encoding="utf-8")
        paths.append(path)
    return paths
