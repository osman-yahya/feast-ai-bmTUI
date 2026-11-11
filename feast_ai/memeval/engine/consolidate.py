"""Session-end write path: the SLM proposes, code decides.

1. **extract**   - the on-device model reads the session's user turns and
   emits ``[{"op", "key", "value"}]``, forced into shape by a GBNF grammar.
2. **filter + canonicalize** - code maps the free Turkish value onto the
   app's vocabulary (EU-14 allergen keys, diet, cuisine labels, areas) and
   drops anything it cannot map: a hallucinated fact has nowhere to land.
3. **apply** - ADD / UPDATE (single-valued keys close the old value, history
   kept) / DELETE (user says it no longer holds) / NOOP (already known).
   Dates come from the turn's timestamp, never from the model.

Raw turns are kept as episodes with their relative dates resolved
(:func:`timeparse.annotate`).  A KVKK erasure from the "Mustachio ne biliyor?"
screen marks the fact ``deleted`` and drops every earlier episode that
mentions the value - §3.4: the user's erasure outranks the chat log.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from ...agent.text import fold
from ..gen.plan import TRAP_NAMES
from ..gen.world import AREAS, CUISINES, cuisine_of
from ..schema import Dataset
from .store import HARD, SINGLE, Episode, Fact, MemoryStore
from .timeparse import annotate, resolve

KEYS = ("allergen", "diet", "likes", "dislikes", "home_area")
EXTRACT_PROMPT = (
    "You read a user's messages to a food app assistant and write down durable facts the "
    "user states about THEMSELVES. Output a JSON array only.\n"
    "Each item: {\"op\": \"add\" or \"remove\", \"key\": ..., \"value\": short Turkish words}.\n"
    "Keys:\n"
    "- allergen: a food the user is allergic to\n"
    "- diet: vegetarian or vegan eating\n"
    "- likes: a kind of food or cuisine the user says they like or often eat\n"
    "- dislikes: a taste the user avoids, e.g. spicy\n"
    "- home_area: the neighbourhood the user lives in (a move means add the new one)\n"
    "Use remove when the user says a fact no longer holds (e.g. an allergy is gone).\n"
    "Ignore: places visited once, what the assistant said, one-off cravings, other people.\n"
    "If there is nothing durable, output []."
)
GRAMMAR = r'''root ::= "[" ws ( fact ( "," ws fact )* )? ws "]"
fact ::= "{" ws "\"op\":" ws op "," ws "\"key\":" ws key "," ws "\"value\":" ws str ws "}"
op ::= "\"add\"" | "\"remove\""
key ::= "\"allergen\"" | "\"diet\"" | "\"likes\"" | "\"dislikes\"" | "\"home_area\""
str ::= "\"" [^"\\\n]{1,48} "\""
ws ::= [ \n]{0,3}
'''
#: Folded word stems -> EU-14 key; first hit wins, so the specific ones lead.
ALLERGEN_CUES = (
    (("yer fist", "fistik ezme", "peanut"), "peanuts"),
    (("antep", "ceviz", "findik", "kuruyemis", "badem", "tree"), "tree_nuts"),
    (("kabuklu", "karides", "istakoz", "yengec", "crustac"), "crustaceans"),
    (("kalamar", "midye", "ahtapot", "yumusakca", "mollus"), "molluscs"),
    (("sut", "laktoz", "milk"), "milk"),
    (("gluten", "bugday", "colyak"), "gluten"),
    (("balik", "fish"), "fish"),
    (("yumurta", "egg"), "eggs"),
    (("susam", "sesame"), "sesame"),
    (("soya", "soy"), "soybeans"),
    (("fistik",), "peanuts"),
)


def memory_names(ds: Dataset) -> tuple[str, ...]:
    """Names to mask before time parsing: every restaurant, trap names first."""
    return TRAP_NAMES + tuple(r.name for r in ds.world.restaurants)


def cuisine_index(ds: Dataset) -> dict[str, str]:
    """Restaurant name -> cuisine label, from the app's own restaurant table."""
    return {r.name: CUISINES[cuisine_of(r.id)]["label"] for r in ds.world.restaurants}


def link(text: str, cuisines: dict[str, str]) -> str:
    """Tag each restaurant a turn names with its cuisine: "Bitki Kafe [vegan kafe]".

    The app knows what a place serves; the 2B model should not have to guess
    it from the name, and retrieval can then match "uzakdoğu" to "Tokyo Sokağı"."""
    if len(fold(text)) != len(text):
        return text
    folded = fold(text)
    hits = []
    for name, label in cuisines.items():
        m = re.search(r"\b" + re.escape(fold(name)) + r"\w*", folded)
        if m:
            hits.append((m.end(), label))
    for end, label in sorted(hits, reverse=True):
        text = text[:end] + f" [{label}]" + text[end:]
    return text


def canonical(key: str, value: str) -> str | None:
    v = fold(value)
    if key == "allergen":
        return next((k for cues, k in ALLERGEN_CUES if any(c in v for c in cues)), None)
    if key == "diet":
        if "vegan" in v:
            return "vegan"
        if any(c in v for c in ("vejet", "vegetar", "et yem")):
            return "vegetarian"
        return None
    if key == "likes":
        for code, c in CUISINES.items():
            cues = {fold(c["label"]), code, *map(fold, c["tags"]), *fold(c["label"]).split("/")}
            if any(cue and cue in v for cue in cues):
                return c["label"]
        return None
    if key == "dislikes":
        return "acılı" if "aci" in v else None
    if key == "home_area":
        return next((a for a in AREAS if fold(a).split()[0] in v), None)
    return None


def apply(store: MemoryStore, op: str, key: str, value: str, turn_id: str, day: str) -> str:
    """One canonical operation; returns what it did (ADD/UPDATE/DELETE/NOOP)."""
    same = [f for f in store.open_facts() if f.key == key and f.value == value]
    if op == "remove":
        for f in same:
            f.status, f.valid_until = "closed", day
        return "DELETE" if same else "NOOP"
    if same:
        return "NOOP"
    did = "ADD"
    if key in SINGLE:
        for f in store.open_facts():
            if f.key == key:
                f.status, f.valid_until, did = "closed", day, "UPDATE"
    store.facts.append(Fact(fid=f"f{len(store.facts) + 1}", key=key, value=value,
                            kind="hard" if key in HARD else "soft", valid_from=day,
                            source_turn_id=turn_id))
    return did


def erase(store: MemoryStore, key: str, value: str, at: str) -> None:
    """KVKK erasure: the fact, every earlier user turn that states it, and the
    assistant's reply to each.  Not every turn naming the value: "Beşiktaş"
    is also where recommended restaurants are, and those stay."""
    for f in store.facts:
        if f.key == key and fold(f.value) == fold(value):
            f.status = "deleted"
    needle = fold(value)
    drop: set[int] = set()
    for i, e in enumerate(store.episodes):
        if e.role == "user" and e.ts <= at and needle in fold(e.text):
            drop.add(i)
            if i + 1 < len(store.episodes) and store.episodes[i + 1].role == "assistant":
                drop.add(i + 1)
    store.episodes = [e for i, e in enumerate(store.episodes) if i not in drop]
    store.log.append({"erase": [key, value], "at": at})


def parse_ops(raw: str) -> list[dict]:
    try:
        items = json.loads(raw)
    except ValueError:
        return []
    return [i for i in items if isinstance(i, dict) and i.get("key") in KEYS] if isinstance(items, list) else []


async def extract(client, session, names) -> tuple[str, list[dict]]:
    from ...agent.bench import _generate
    import time

    lines = [f"[{t.ts[:10]}] {annotate(t.text, datetime.fromisoformat(t.ts), names)}"
             for t in session.turns if t.role == "user"]
    if not lines:
        return "", []
    messages = client.build_messages([], "\n".join(lines), system_prompt=EXTRACT_PROMPT)
    raw, _ = await _generate(client, messages, time.perf_counter(), grammar=GRAMMAR)
    return raw, parse_ops(raw)


async def build_store(client, ds: Dataset, persona: str, until: str, log=print) -> MemoryStore:
    """Replay a persona's history up to ``until``: sessions and erasures in time order."""
    names = memory_names(ds)
    cuisines = cuisine_index(ds)
    store = MemoryStore(persona=persona)
    events = [(s.started_at, "session", s) for s in ds.sessions.get(persona, []) if s.started_at < until]
    events += [(d["at"], "erase", d) for d in ds.personas[persona].memory_deletions if d["at"] < until]
    for _, kind, item in sorted(events, key=lambda e: e[0]):
        if kind == "erase":
            erase(store, item["key"], item["value"], item["at"])
            continue
        for t in item.turns:
            ref = datetime.fromisoformat(t.ts)
            spans = resolve(t.text, ref, names)
            store.episodes.append(Episode(t.turn_id, t.ts, t.role, annotate(link(t.text, cuisines), ref, names),
                                          [f"{s.first}..{s.last}" for s in spans]))
        raw, ops = await extract(client, item, names)
        done = []
        first_user = next((t for t in item.turns if t.role == "user"), item.turns[0])
        for o in ops:
            value = canonical(o["key"], o["value"])
            if value is None:
                done.append([o["op"], o["key"], o["value"], "DROPPED"])
                continue
            done.append([o["op"], o["key"], value,
                         apply(store, o["op"], o["key"], value, first_user.turn_id, first_user.ts[:10])])
        store.log.append({"session": item.session_id, "raw": raw, "ops": done})
        log(f"  {item.session_id}: {' '.join(f'{d[3]}:{d[1]}={d[2]}' for d in done) or '-'}")
    return store


def store_path(root: Path, model_id: str, persona: str) -> Path:
    return root / "memstore" / model_id / f"{persona}.json"


async def build_stores(client, ds: Dataset, model_id: str, root: Path, log=print) -> dict[str, MemoryStore]:
    """One store per persona, built once per model and cached under ``root``."""
    stores = {}
    for pid in sorted(ds.personas):
        path = store_path(root, model_id, pid)
        if path.exists():
            stores[pid] = MemoryStore.load(path)
            continue
        until = max((q.asked_at for q in ds.questions if q.persona == pid), default="9999")
        log(f"Consolidating {pid}…")
        stores[pid] = await build_store(client, ds, pid, until, log)
        stores[pid].save(path)
    return stores


def facts_eval(ds: Dataset, stores: dict[str, MemoryStore]) -> dict:
    """Write-path quality against ``facts_gold``: (key, value) precision/recall
    over everything extracted, and whether each matched fact ends in the gold status."""
    tp = fp = fn = status_ok = 0
    for pid, store in stores.items():
        gold = {(g.key, g.value): g.status for g in ds.facts.get(pid, [])}
        got = {(f.key, f.value): f.status for f in store.facts}
        tp += len(gold.keys() & got.keys())
        fp += len(got.keys() - gold.keys())
        fn += len(gold.keys() - got.keys())
        status_ok += sum(got[k] == gold[k] for k in gold.keys() & got.keys())
    return {"precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
            "status_accuracy": status_ok / tp if tp else 0.0, "tp": tp, "fp": fp, "fn": fn,
            "facts": sum(len(s.facts) for s in stores.values()),
            "sessions": sum(sum("session" in e for e in s.log) for s in stores.values()),
            "episodes": sum(len(s.episodes) for s in stores.values())}
