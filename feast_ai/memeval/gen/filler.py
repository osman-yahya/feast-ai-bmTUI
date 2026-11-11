"""Filler sessions for a long-history dataset, from templates, no model.

v0.1 has ~27 turns per persona, so retrieving 8 turns shows the model a
third of the haystack and every retriever looks good.  A real user's log is
mostly small talk with the app - opening hours, prices, where a place is -
so v0.2 is v0.1 plus ~50 such sessions per persona, dated between the
existing ones.

The filler must change no answer.  The user only *asks* about a named
restaurant and the assistant answers from ``world.json``; neither side ever:

- says the user went somewhere, ate something or saved a place (temporal,
  multi_session, app_data);
- says where the user lives, or names an area the user erased (knowledge
  update, deleted_info);
- recommends a place or gives an opinion (assistant_info, attributed_opinion);
- mentions allergies, diet or likes, or asks about an item the user's gold
  facts rule out (user_info, safety, preference);
- touches an abstention question's topic - :func:`audit` rejects the banned
  words and restaurant names sharing a stem with those questions.

Only "bugün / bu akşam / bu öğlen" carry a date, and they resolve to the
session's own day; weekdays are habitual ("pazar günleri"), which
:mod:`timeparse` leaves alone - "pazar günü" would resolve to the past one.

    python -m feast_ai.memeval.gen.filler [SRC] [OUT] [--sessions 50] [--seed 7]
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from ...agent.text import fold
from ..engine.retrieve import tokens
from ..schema import ChatTurn, Dataset, Persona, Restaurant, Session, load
from .plan import ASKED_AT, DAY_TR, DIET_BADGES, MONTH_TR, _cap
from .world import SIDE_OF

DATA = Path(__file__).parent.parent / "data"
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
PRICE_TR = {1: "uygun fiyatlı", 2: "orta fiyatlı", 3: "ortanın biraz üstünde", 4: "pahalı tarafta"}
#: Folded substrings no filler turn - and no restaurant or item it names - may contain.
BANNED = (
    "alerj", "gluten", "susam", "fistik", "vegan", "vejet", "diyet", "et yemi",
    "sevdi", "seviyor", "sevmi", "sevmem", "oturuyor", "tasindi", "evim",
    "gittim", "gittik", "gitmis", "yedim", "yedik", "yemistim", "kaydet",
    "hekim", "disci", "pastane", "anne", "dogum gun", "kedi", "mama", "spor", "salon",
    "oner", "tavsiye", "begen", "lezzet", "harika", "guzel", "iyi mi", "afiyet",
)
TURN_GAP = timedelta(seconds=25)


# ----------------------------------------------------------------------
# What a persona may be asked about
# ----------------------------------------------------------------------
def constraints(ds: Dataset, pid: str) -> tuple[set[str], tuple[str, ...] | None, bool]:
    """Allergens (any status: a closed allergy is still not something to
    bring up), the badges the strictest diet ever stated needs, and whether
    spicy food is avoided."""
    facts = ds.facts.get(pid, [])
    allergens = {f.value for f in facts if f.key == "allergen"}
    diets = {f.value for f in facts if f.key == "diet"}
    diet = DIET_BADGES["vegan"] if "vegan" in diets else DIET_BADGES.get("vegetarian") if diets else None
    no_spicy = any(f.key == "dislikes" and "acı" in f.value for f in facts)
    return allergens, diet, no_spicy


def abstention_stems(ds: Dataset, pid: str) -> set[str]:
    return {t for q in ds.questions if q.persona == pid and q.category == "abstention"
            for t in tokens(q.question)}


def restaurants_for(ds: Dataset, p: Persona) -> tuple[list[Restaurant], list[Restaurant]]:
    """(home area, same side) pools.  An erased home area is left out
    entirely, and so is a name that is banned or shares a stem with an
    abstention question ("Anne Eli Lokantası" vs "Annemin doğum günü")."""
    erased = {fold(d["value"]) for d in p.memory_deletions if d["key"] == "home_area"}
    stems = abstention_stems(ds, p.id)
    side = SIDE_OF.get(p.home_area)
    ok = [r for r in ds.world.restaurants
          if fold(r.area) not in erased and not any(b in fold(r.name) for b in BANNED)
          and not _overlap(tokens(r.name), stems)]
    home = [r for r in ok if r.area == p.home_area and fold(p.home_area) not in erased]
    near = [r for r in ok if SIDE_OF.get(r.area) == side and r not in home]
    return home, near


def items_for(ds: Dataset, pid: str) -> dict[str, list]:
    allergens, diet, no_spicy = constraints(ds, pid)
    out: dict[str, list] = {}
    for m in ds.world.menu_items:
        if set(m.allergens) & allergens:
            continue
        if diet and not set(m.badges) & set(diet):
            continue
        if no_spicy and {"spicy", "extra_spicy"} & set(m.badges):
            continue
        if any(b in fold(m.name) for b in BANNED):
            continue
        out.setdefault(m.restaurant_id, []).append(m)
    return out


def _overlap(words: list[str], stems: set[str]) -> list[str]:
    """Stems are cut to five letters, so "anne" and "annem" only meet as prefixes."""
    return sorted(w for w in set(words) for s in stems if min(len(w), len(s)) >= 4
                  and (w.startswith(s) or s.startswith(w)))


def audit(text: str, stems: set[str]) -> None:
    folded = fold(text)
    bad = [b for b in BANNED if b in folded] + _overlap(tokens(text), stems)
    if bad:
        raise ValueError(f"filler turn breaks the rules {bad}: {text!r}")


# ----------------------------------------------------------------------
# Templates: (user, assistant) from the restaurant and the session time
# ----------------------------------------------------------------------
def _hours(r: Restaurant, day: int) -> dict:
    return r.working_hours[WEEKDAYS[day % 7]]


def _next_open(r: Restaurant, day: int) -> str:
    for k in range(1, 8):
        h = _hours(r, day + k)
        if h["is_open"]:
            return f"{DAY_TR[(day + k) % 7]} {h['open']}"
    return "yakında"


def t_today(r, when, items, rng):
    h = _hours(r, when.weekday())
    q = rng.choice([f"{r.name} bugün kaça kadar açık?", f"{r.name} bugün açık mı?",
                    f"bugün {r.name} kaçta kapanıyor?"])
    if h["is_open"]:
        return q, f"{r.name} bugün {h['open']}–{h['close']} arası açık."
    return q, f"{r.name} bugün kapalı. Bir sonraki açılış: {_next_open(r, when.weekday())}."


def t_tonight(r, when, items, rng):
    if when.hour < 12:
        return None
    h = _hours(r, when.weekday())
    q = rng.choice([f"bu akşam {r.name} açık mı?", f"{r.name} bu akşam kaça kadar açık?"])
    if h["is_open"]:
        return q, f"Evet, bu akşam açık; kapanış saati {h['close']}."
    return q, f"Hayır, {r.name} {DAY_TR[when.weekday()]} günleri kapalı."


def t_weekday(r, when, items, rng):
    d = rng.randrange(7)
    h = _hours(r, d)
    q = rng.choice([f"{r.name} {DAY_TR[d]} günleri açık mı?", f"{r.name} {DAY_TR[d]} günleri kaçta açılıyor?"])
    if h["is_open"]:
        return q, f"{_cap(DAY_TR[d])} günleri {h['open']}–{h['close']} arası açık."
    return q, f"{r.name} {DAY_TR[d]} günleri kapalı."


def t_weekend(r, when, items, rng):
    parts = []
    for d in (5, 6):
        h = _hours(r, d)
        parts.append(f"{DAY_TR[d]} {h['open']}–{h['close']}" if h["is_open"] else f"{DAY_TR[d]} kapalı")
    q = rng.choice([f"{r.name} hafta sonları kaçta açılıyor?", f"hafta sonları {r.name} açık mı?"])
    return q, _cap(", ".join(parts)) + "."


def t_price(r, when, items, rng):
    if not r.price_level:
        return None
    q = rng.choice([f"{r.name} pahalı bir yer mi?", f"{r.name} fiyat olarak nasıl?",
                    f"{r.name} fiyatları ne durumda?"])
    return q, f"{r.name} fiyat seviyesi {'₺' * r.price_level}, yani {PRICE_TR[r.price_level]}."


def t_item(r, when, items, rng):
    menu = items.get(r.id)
    if not menu:
        return None
    m = rng.choice(menu)
    q = rng.choice([f"{r.name} menüsünde {m.name} kaç TL?", f"{r.name} için {m.name} fiyatı ne kadar?"])
    a = f"{m.name} {m.price} TL."
    return q, a + (" Şu an tükenmiş görünüyor." if m.sold_out else "")


def t_where(r, when, items, rng):
    q = rng.choice([f"{r.name} hangi semtte?", f"{r.name} nerede, adresi ne?", f"{r.name} tam olarak nerede?"])
    return q, f"{r.name} {r.area} semtinde ({r.address or r.area + ', İstanbul'})."


def t_campaign(r, when, items, rng, campaigns=()):
    q = rng.choice([f"{r.name} için şu an bir kampanya var mı?", f"{r.name} indirim yapıyor mu bu aralar?"])
    live = [c for c in campaigns if c.restaurant_id == r.id
            and datetime.fromisoformat(c.start_date) <= when <= datetime.fromisoformat(c.end_date)]
    if live:
        c = live[0]
        end = datetime.fromisoformat(c.end_date)
        return q, f"Var: \"{c.title}\" ({c.description}). Son gün {end.day} {MONTH_TR[end.month]}."
    return q, f"Şu an {r.name} için kayıtlı bir kampanya yok."


def t_unknown(r, when, items, rng):
    q = rng.choice([f"{r.name} paket servis yapıyor mu?", f"{r.name} için rezervasyon gerekiyor mu?",
                    f"{r.name} kartla ödeme alıyor mu?"])
    return q, "Bu bilgi uygulamada yok; restoranın sayfasından ya da telefonla sorabilirsin."


TEMPLATES = (t_today, t_tonight, t_weekday, t_weekend, t_price, t_item, t_item, t_where,
             t_campaign, t_unknown)
CLOSERS = (("tamam, teşekkürler", "Rica ederim."), ("süper, sağ ol", "Ne demek."),
           ("anladım", "Başka bir şey sorarsan buradayım."))


# ----------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------
def _times(rng: random.Random, sessions: list[Session], n: int) -> list[datetime]:
    """``n`` session starts between the persona's first session and the
    question time, 09:00-22:45, at least an hour from any other session."""
    taken = [datetime.fromisoformat(s.started_at) for s in sessions]
    lo = min(taken) + timedelta(days=1)
    hi = ASKED_AT - timedelta(days=1)
    span = (hi.date() - lo.date()).days
    out: list[datetime] = []
    while len(out) < n:
        day = lo.date() + timedelta(days=rng.randrange(span + 1))
        t = datetime(day.year, day.month, day.day, rng.randrange(9, 23), rng.choice((0, 15, 30, 45)))
        if all(abs((t - x).total_seconds()) >= 3600 for x in taken + out):
            out.append(t)
    return sorted(out)


def persona_filler(ds: Dataset, pid: str, n: int, seed: int) -> list[Session]:
    p = ds.personas[pid]
    rng = random.Random(f"{seed}:{pid}")
    home, near = restaurants_for(ds, p)
    items = items_for(ds, pid)
    stems = abstention_stems(ds, pid)
    out = []
    for k, start in enumerate(_times(rng, ds.sessions[pid], n)):
        r = rng.choice(home if home and rng.random() < 0.5 else near)
        pairs, used = [], set()
        while len(pairs) < (1 if rng.random() < 0.6 else 2):
            tpl = rng.choice(TEMPLATES)
            if tpl in used:
                continue
            got = tpl(r, start, items, rng, ds.world.campaigns) if tpl is t_campaign else tpl(r, start, items, rng)
            if got:
                used.add(tpl)
                pairs.append(got)
        if rng.random() < 0.3:
            pairs.append(rng.choice(CLOSERS))
        sid = f"s_{pid}_f{k:02d}"
        turns, ts = [], start
        for user, assistant in pairs:
            for role, text in (("user", user), ("assistant", assistant)):
                audit(text, stems)
                turns.append(ChatTurn(f"t_{pid}_f{k:02d}_{len(turns):02d}", ts.isoformat(), role, text))
                ts += TURN_GAP
        out.append(Session(sid, pid, start.isoformat(), turns))
    return out


def build(src: Path, out: Path, n: int = 50, seed: int = 7) -> dict:
    """Copy ``src`` to ``out`` with ``n`` filler sessions per persona merged
    into each history in time order; everything else is unchanged."""
    ds = load(src)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("world.json", "questions.jsonl"):
        shutil.copy(src / name, out / name)
    for name in ("personas", "facts_gold"):
        shutil.copytree(src / name, out / name)
    (out / "histories").mkdir()
    added = {}
    for pid in sorted(ds.personas):
        filler = persona_filler(ds, pid, n, seed)
        added[pid] = {"sessions": len(filler), "turns": sum(len(s.turns) for s in filler)}
        merged = sorted(ds.sessions[pid] + filler, key=lambda s: s.started_at)
        with (out / "histories" / f"{pid}.jsonl").open("w", encoding="utf-8") as fh:
            for s in merged:
                fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    manifest = json.loads((src / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest.update(derived_from=src.name, filler={"generator": "feast_ai.memeval.gen.filler",
                                                    "seed": seed, "sessions_per_persona": n,
                                                    "added": added})
    (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return added


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", nargs="?", default=str(DATA / "v0.1"))
    ap.add_argument("out", nargs="?", default=str(DATA / "v0.2"))
    ap.add_argument("--sessions", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    added = build(Path(a.src), Path(a.out), a.sessions, a.seed)
    for pid, row in added.items():
        print(f"{pid:<10} +{row['sessions']} sessions, +{row['turns']} turns")
