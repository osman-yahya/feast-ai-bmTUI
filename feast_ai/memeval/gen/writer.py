"""Turning a plan into Turkish chat, and checking the result against the plan.

One call per persona, so the voice stays consistent across sessions.  The
model must tag every turn with the beats it carries; :func:`check` rejects a
history where a beat is missing, carried by the wrong speaker, or where a
restaurant name shows up that no beat asked for (an invented visit would
silently break the abstention and counting questions).  A rejected history is
retried with the problems fed back.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta

from ..schema import ROLES
from .llm import DEFAULT_MODEL, GenError, Usage, complete_json
from .plan import DAY_TR, MONTH_TR, UNGROUNDED, Plan

SYSTEM = """Sen bir eval veri seti icin gercekci Turkce sohbet yazarisin.
Uygulama: feast, Istanbul'da restoran kesfi. Asistan: Mustachio - samimi, kisa (1-2 cumle) konusur.

Kurallar:
- Her oturum icin verilen "beat"leri SIRAYLA isle. Her beat tam olarak bir turda gecer ve o turun
  "beats" listesine id'si yazilir. Beat'in rolu ile turun rolu ayni olmali.
- Bir turda birden fazla beat olabilir; beat tasimayan kisa dolgu turlari ekleyebilirsin.
- Oturum her zaman kullanici turuyla baslar. Oturum basina 2-8 tur; oturumda "tur_hedefi" varsa
  ona yakin yaz (beat tasimayan dolgu turlari da plan disi bilgi icermez).
- Plan disinda kisisel bilgi EKLEME: alerji, diyet, evcil hayvan, adres, saglik, gidilen yer,
  restoran adi. Beat'te gecmeyen hicbir restoran adini yazma.
- "DOLAYLI" denen beat'lerde bilgiyi acikca soyleme, gunluk dille ima et.
- Kullanicilar gercek insanlar gibi yazsin: bazen kucuk harf, diakritiksiz, kisa; ama Turkce dogal olsun.
- Tarih ifadelerini (dun, gecen cuma...) beat'te verildigi gibi kullan; kendin tarih uydurma.
- Asistan yalniz uygulamanin verisini bilir: yemek adi, fiyat, etiketler (vegan, glutensiz,
  'populer', 'sefin onerisi'), semt, fiyat seviyesi. Puan, yorum, populerlik bilgisi YOK:
  "meshur", "efsane", "cok sevilir", "herkes bayiliyor", "lezzetli" gibi degerlendirme YAZMA.
  Baskasinin postunu aktarirsan yazariyla ve cift tirnak icinde aktar: @kullanici: "...".
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "sessions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "session_index": {"type": "integer"},
                    "turns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": list(ROLES)},
                                "text": {"type": "string"},
                                "beats": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["role", "text", "beats"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["session_index", "turns"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["sessions"],
    "additionalProperties": False,
}


#: Double-quoted spans. Single quotes are not quotations in Turkish text - the
#: apostrophe separates suffixes ("Moda'da").
QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|«[^»]*»")


def _fold(text: str) -> str:
    from ...agent.text import fold
    return fold(text)


def _prompt(plan: Plan) -> str:
    sessions = []
    for s in plan.sessions:
        d = s.started_at
        sessions.append({
            "session_index": s.index,
            "tarih": f"{d.day} {MONTH_TR[d.month]} {d.year} {DAY_TR[d.weekday()]} {d:%H:%M}",
            **({"tur_hedefi": s.target_turns} if s.target_turns else {}),
            "beats": [{"id": b.id, "role": b.role, "talimat": b.instruction} for b in s.beats],
        })
    voice = f" Yazim tarzi: {plan.spec.voice}." if plan.spec.voice else ""
    return (f"Kullanici: {plan.spec.name}.{voice} Asagidaki {len(sessions)} oturumu yaz.\n\n"
            + json.dumps(sessions, ensure_ascii=False, indent=1))


def check(plan: Plan, data: dict, restaurant_names: list[str]) -> list[str]:
    problems: list[str] = []
    beats = {b.id: b for s in plan.sessions for b in s.beats}
    seen: dict[str, int] = {}
    by_index = {s.get("session_index"): s for s in data.get("sessions", [])}
    for s in plan.sessions:
        written = by_index.get(s.index)
        if not written or not written["turns"]:
            problems.append(f"session {s.index} missing")
            continue
        if written["turns"][0]["role"] != "user":
            problems.append(f"session {s.index} does not start with the user")
        for t_index, turn in enumerate(written["turns"]):
            allowed = " ".join(_fold(beats[b].instruction) for b in turn["beats"] if b in beats)
            for bid in turn["beats"]:
                if bid not in beats:
                    problems.append(f"session {s.index} turn {t_index}: unknown beat {bid}")
                    continue
                seen[bid] = seen.get(bid, 0) + 1
                if beats[bid].role != turn["role"]:
                    problems.append(f"beat {bid} carried by {turn['role']}, planned for {beats[bid].role}")
            text = _fold(turn["text"])
            if turn["role"] == "assistant":
                # A verdict inside a quotation is someone's post being cited,
                # not Mustachio's own claim (prompt rule: always with author).
                own = _fold(QUOTED.sub(" ", turn["text"]))
                if QUOTED.search(turn["text"]) and "@" not in turn["text"]:
                    problems.append(f"session {s.index} turn {t_index}: quotation without an @author")
                for word in UNGROUNDED:
                    if _fold(word) in own:
                        problems.append(f"session {s.index} turn {t_index}: ungrounded verdict "
                                        f"'{word}' - the app has no ratings or reviews")
            for name in restaurant_names:
                key = _fold(name)
                if key in text and key not in allowed:
                    problems.append(f"session {s.index} turn {t_index}: '{name}' not asked for by a beat")
    for bid, beat in beats.items():
        if seen.get(bid, 0) != 1:
            problems.append(f"beat {bid} carried {seen.get(bid, 0)} times: {beat.instruction[:60]}")
    # Every name a beat asks for must actually be written.
    for s in by_index.values():
        for turn in s["turns"]:
            for bid in turn["beats"]:
                if bid in beats:
                    for name in restaurant_names:
                        if f"'{name}'" in beats[bid].instruction and _fold(name) not in _fold(turn["text"]):
                            problems.append(f"beat {bid}: '{name}' missing from its turn")
    return problems


def dry_run(plan: Plan) -> dict:
    """Instructions as text, one beat per turn: exercises the whole pipeline
    without a model or a key. The output is not an eval set."""
    return {"sessions": [{"session_index": s.index,
                          "turns": [{"role": b.role, "text": b.instruction, "beats": [b.id]}
                                    for b in s.beats]}
                         for s in plan.sessions]}


def write(plan: Plan, restaurant_names: list[str], *, model: str = DEFAULT_MODEL,
          attempts: int = 3) -> tuple[dict, Usage]:
    usage = Usage()
    prompt = _prompt(plan)
    for attempt in range(1, attempts + 1):
        data, used = complete_json(SYSTEM, prompt, SCHEMA, model=model)
        usage.add(used)
        problems = check(plan, data, restaurant_names)
        if not problems:
            return data, usage
        prompt = (_prompt(plan) + "\n\nOnceki denemende su sorunlar vardi, hepsini duzelt:\n- "
                  + "\n- ".join(problems[:30]))
    raise GenError(f"{plan.spec.id}: history still invalid after {attempts} attempts: "
                   + "; ".join(problems[:5]))


def to_sessions(plan: Plan, data: dict) -> tuple[list[dict], dict[str, str]]:
    """Schema sessions with timestamps and turn ids, plus beat -> turn id."""
    beat_turn: dict[str, str] = {}
    out = []
    by_index = {s["session_index"]: s for s in data["sessions"]}
    for s in plan.sessions:
        turns = []
        for t_index, turn in enumerate(by_index[s.index]["turns"]):
            tid = f"t_{plan.spec.id}_{s.index:02d}_{t_index:02d}"
            ts = s.started_at + timedelta(seconds=25 * t_index)
            turns.append({"turn_id": tid, "ts": ts.isoformat(), "role": turn["role"], "text": turn["text"]})
            for bid in turn["beats"]:
                beat_turn[bid] = tid
        recommended = [{**r, "turn_id": beat_turn[r["turn_id"]]} for r in s.recommended]
        out.append({"session_id": f"s_{plan.spec.id}_{s.index:02d}", "persona": plan.spec.id,
                    "started_at": s.started_at.isoformat(), "turns": turns,
                    "recommended": recommended})
    return out, beat_turn
