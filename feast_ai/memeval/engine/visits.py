"""The visit log: where the user ate and on which day, kept by code.

A 2B model cannot count visits over forty sessions, and it copies the date a
turn was written instead of the day the turn talks about (v0.3: every wrong
temporal answer had its evidence in MEMORY).  So a visit the user tells about
("dün akşam Rakı Sofrası'nda muhammara yedim") is written down when the
session is consolidated - restaurant from the app's table, day from
:mod:`.timeparse` - and joined with the visits the app itself recorded.
Counting and dating are then lookups: ``v`` lines in MEMORY and the
``my_visits`` tool (:mod:`..tools`) both read this log.

Rule, not model: a user turn counts as a visit when it names a restaurant,
has a first-person past verb of eating or going, and dates to one day that
is not in the future.  A plan ("cumartesi gideceğiz") or a cancelled one has
no such verb and is not a visit.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from ...agent.text import fold
from .timeparse import Span

#: First-person past: "yedim", "içtik", "gittim", "kahvaltı yaptık", ...
WENT = re.compile(r"\b(?:yedi[mk]|icti[mk]|gitti[mk]|oturdu[mk]|denedi[mk]|ugradi[mk]|"
                  r"yapti[mk]|aldi[mk]|tatti[mk]|bulustu[mk]|kacti[mk])\b")
#: "gittim" after one of these is not a visit to the place named: "gitmeden", "gidemedim".
NOT_WENT = re.compile(r"\b(?:gideme\w*|gitmedi\w*|yiyemedi\w*|iptal\w*|vazgec\w*)\b")


#: An assistant turn that recommends: "... deneyebilirsin", "öneririm", "uygun olur".
#: v0.3: the (turn, restaurant) pairs equal the app's own record of recommendation
#: cards (``Session.recommended``), precision and recall 1.00.
RECOMMENDS = re.compile(r"\b(?:oner\w*|tavsiye\w*|deneyebilir\w*|gidebilir\w*|bakabilir\w*|"
                        r"tercih edebilir\w*|uygun\w*|harika bir secim\w*|guzel bir secim\w*)\b")


def _named(folded: str, restaurants: dict[str, tuple[str, str]]) -> list[tuple[str, str]]:
    found = []
    for key in sorted(restaurants, key=len, reverse=True):      # longest name first
        if re.search(r"\b" + re.escape(key), folded):
            folded = folded.replace(key, "#" * len(key))
            found.append(restaurants[key])
    return found


def visits_in(text: str, ref: datetime, spans: list[Span], restaurants: dict[str, tuple[str, str]],
              turn_id: str = "") -> list[dict]:
    """Visits a user turn reports. ``restaurants``: folded name -> (id, name)."""
    f = fold(text)
    if not WENT.search(f) or NOT_WENT.search(f):
        return []
    days = [s.first for s in spans if s.first == s.last and s.first <= ref.date()]
    if len(set(days)) != 1:
        return []
    return [{"restaurant_id": rid, "restaurant": name, "date": days[0].isoformat(),
             "turn_id": turn_id, "source": "chat"} for rid, name in _named(f, restaurants)]


def recommendations(episodes, restaurants: dict[str, tuple[str, str]], until: str) -> list[dict]:
    """The places the assistant recommended before ``until``, one line per
    (restaurant, day), oldest first - read from the stored turns at ask time."""
    out: dict[tuple[str, str], dict] = {}
    for e in episodes:
        if e.role != "assistant" or e.ts >= until:
            continue
        f = fold(e.text)
        if RECOMMENDS.search(f):
            for rid, name in _named(f, restaurants):
                out.setdefault((rid, e.ts[:10]), {"restaurant_id": rid, "restaurant": name,
                                                  "date": e.ts[:10], "turn_id": e.turn_id})
    return sorted(out.values(), key=lambda v: (v["date"], v["restaurant"]))


def merged(chat: list[dict], app: list[dict], names: dict[str, str], until: str) -> list[dict]:
    """Chat visits and the app's own records as one log, oldest first, one line
    per (restaurant, day).  ``app``: the persona's ``visited_restaurants``."""
    out: dict[tuple[str, str], dict] = {}
    for v in app:
        if v["visited_at"] < until:
            out[(v["restaurant_id"], v["visited_at"][:10])] = {
                "restaurant_id": v["restaurant_id"], "restaurant": names[v["restaurant_id"]],
                "date": v["visited_at"][:10], "turn_id": "", "source": "app"}
    for v in chat:                              # told in chat wins: "source" says the user said it
        if v["date"] <= until[:10]:
            out[(v["restaurant_id"], v["date"])] = v
    return sorted(out.values(), key=lambda v: (v["date"], v["restaurant"]))


def select(log: list[dict], question: str, windows: list[tuple[date, date]], limit: int = 6) -> list[dict]:
    """The visits a question is about: those to a restaurant it names (latest
    first), else those inside a time window it gives.  Nothing otherwise."""
    f = fold(question)
    named = [v for v in log if fold(v["restaurant"]) in f]
    if named:
        return sorted(named, key=lambda v: v["date"], reverse=True)[:limit]
    hit = [v for v in log if any(a <= date.fromisoformat(v["date"]) <= b for a, b in windows)]
    return hit[-limit:]
