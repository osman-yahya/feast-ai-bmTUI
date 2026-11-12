"""Tool names in questions, and which real tools satisfy them.

Two tool sets exist and neither is final: the 15-tool Python agent in
:mod:`feast_ai.agent.tools` (bench work) and the 4-tool Dart agent on
feast-cur's ``mustachio-ai`` branch (``lib/core/ai/tools/mustachio_tools.dart``).
Questions name a *capability*; :data:`ALIASES` says which concrete tools of
either set provide it, so one dataset scores both without being rewritten
when a tool is renamed.  Capabilities neither set has yet (``my_visits``,
``search_chat_memory`` on the Dart side) are kept: a question that needs them
is exactly how the eval shows the gap.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime

from ..agent.text import fold

ALIASES: dict[str, frozenset[str]] = {
    "search_restaurants": frozenset({"search_restaurants"}),
    "search_menu_items": frozenset({"search_menu_items", "get_menu"}),
    "get_menu": frozenset({"get_menu", "search_menu_items", "get_restaurant_details"}),
    "search_saved": frozenset({"search_saved"}),
    "search_chat_memory": frozenset({"search_chat_memory"}),
    "get_user_preferences": frozenset({"get_user_preferences"}),
    "my_visits": frozenset({"my_visits", "search_posts"}),
    # Other users' captions about a place - the only opinions the app holds.
    "search_posts": frozenset({"search_posts", "ask_about_posts"}),
}


def satisfied_by(expected: str, called: str) -> bool:
    """``expected`` may list alternatives (``"a|b"``), as in agent/suite.py."""
    return any(called in ALIASES.get(name, frozenset({name})) for name in expected.split("|"))


def unknown(expected: str) -> list[str]:
    return [name for name in expected.split("|") if name not in ALIASES]


# ----------------------------------------------------------------------
# App tools the eval runs for real
# ----------------------------------------------------------------------
#: As on the phone (feast-cur ``AgentLimits``): tools run for one message, and
#: bad calls answered with an error before the message ends without an answer.
MAX_TOOL_CALLS = 3
MAX_INVALID_CALLS = 1
RESULT_NOTE = "Answer the user's question from this result, briefly, in the user's language."


# ----------------------------------------------------------------------
# Every app tool the eval runs
# ----------------------------------------------------------------------
#: One line per tool; a question is offered only the tools it needs
#: (:func:`.engine.gate.tools_for`).  The menu and restaurant tools filter by
#: the user's allergies and diet in code - the model never sees an unsafe dish.
TOOL_TEXT = {
    "my_visits": "my_visits(since, until, restaurant, cuisine) - the restaurants this user went to: the day "
                 "and cuisine of each visit, a count, and the cuisines. Every argument is optional; dates are "
                 "YYYY-MM-DD.",
    "search_saved": "search_saved() - the restaurants this user saved in the app, with a count.",
    "my_recommendations": "my_recommendations(since, until, cuisine) - the places you, the assistant, "
                          "recommended to this user: the day and cuisine of each, and the cuisines. All optional.",
    "search_restaurants": "search_restaurants(cuisine, area) - restaurants open now near the user, each with "
                          "up to two dishes safe for the user's allergies and diet. Both optional.",
    "get_menu": "get_menu(restaurant, category) - one restaurant's dishes that are safe for the user's "
                "allergies and diet, and for a friend the user named. category is optional (Tatlı, Meze, ...).",
    "search_posts": "search_posts(restaurant) - what people wrote about a restaurant in the app, with the "
                    "author of each post.",
}
ARGS = {"my_visits": ("since", "until", "restaurant", "cuisine"), "search_saved": (),
        "my_recommendations": ("since", "until", "cuisine"), "search_restaurants": ("cuisine", "area"),
        "get_menu": ("restaurant", "category"), "search_posts": ("restaurant",)}
DAYS_EN = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def tools_prompt(names: tuple[str, ...]) -> str:
    return "TOOLS\n" + "\n".join(TOOL_TEXT[n] for n in names)


@dataclass
class AppData:
    """What the app holds for one question: the world, the user's logs, and
    the constraints memory says every recommendation must keep."""
    world: object                   # schema.World
    visits: list[dict]              # engine.visits.merged
    saved: list[dict]               # Persona.saved_restaurants
    recs: list[dict]                # engine.visits.recommendations
    constraints: object             # engine.constraints.Constraints
    home: tuple[float, float]
    now: datetime


def _cuisine(rid: str) -> str:
    from .gen.world import CUISINES, cuisine_of

    return CUISINES[cuisine_of(rid)]["label"]


def _name(world, rid: str) -> str:
    return next((r.name for r in world.restaurants if r.id == rid), rid)


def _restaurant(arg, world):
    """The restaurant ``arg`` names: the exact name first, so "Sushi Nami" is
    never "Sushi Nomi"; else the longest name inside it, or containing it."""
    want = fold(str(arg or "")).strip()
    if not want:
        return None
    exact = [r for r in world.restaurants if fold(r.name) == want]
    if exact:
        return exact[0]
    inside = [r for r in world.restaurants if fold(r.name) in want or want in fold(r.name)]
    return max(inside, key=lambda r: len(r.name)) if inside else None


def _km(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def _open_until(r, now: datetime) -> str:
    """Closing time if ``r`` is open at ``now``, else ""."""
    h = r.working_hours.get(WEEKDAYS[now.weekday()], {})
    if not h.get("is_open"):
        return ""
    t = f"{now:%H:%M}"
    if h["close"] <= h["open"]:                          # closes after midnight
        return h["close"] if t >= h["open"] or t < h["close"] else ""
    return h["close"] if h["open"] <= t < h["close"] else ""


def _dates(args: dict) -> tuple[dict, str]:
    out = {}
    for key in ("since", "until"):
        value = args.get(key)
        if value in (None, ""):
            continue
        if not (isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)):
            return {}, f"error: {key} must be a date written YYYY-MM-DD."
        out[key] = value
    return out, ""


def _cuisines(rows: list[dict]) -> str:
    return ", ".join(sorted({_cuisine(v["restaurant_id"]) for v in rows})) or "none"


def _dish(m) -> str:
    tags = [b for b in m.badges if b in ("vegan", "vegetarian", "gluten_free")]
    return f"{m.name} ({m.id}) {m.price} TL" + (f" [{', '.join(tags)}]" if tags else "")


def _logs(name: str, args: dict, data: AppData) -> tuple[bool, str]:
    """my_visits and my_recommendations: one line per (restaurant, day), and the cuisines."""
    span, err = _dates(args)
    if err:
        return False, err
    rows = [v for v in (data.visits if name == "my_visits" else data.recs)
            if span.get("since", "") <= v["date"] <= span.get("until", "9999")]
    shown = [f"{k}={v}" for k, v in span.items()]
    if name == "my_visits" and args.get("restaurant"):
        r = _restaurant(args["restaurant"], data.world)
        rows = [v for v in rows if r is not None and v["restaurant_id"] == r.id]
        shown.append(f"restaurant={r.name if r else args['restaurant']}")
    if args.get("cuisine"):                     # v0.3: 22 of 48 "önerdiğin kahvaltı yeri" calls passed one
        want = fold(str(args["cuisine"]))
        rows = [v for v in rows if want in fold(_cuisine(v["restaurant_id"]))]
        shown.append(f"cuisine={args['cuisine']}")
    lines = [f"- {v['date']} {DAYS_EN[date.fromisoformat(v['date']).weekday()]} | {v['restaurant']} "
             f"[{_cuisine(v['restaurant_id'])}]" + (" | app record" if v.get("source") == "app" else "")
             for v in rows[-12:]]
    if len(rows) > 12:
        lines.insert(0, f"(the latest 12 of {len(rows)})")
    head = f"{name} {' '.join(shown)}".rstrip()
    if name == "my_recommendations":
        return True, "\n".join([f"{head}: {len(rows)} recommendation(s)", *lines,
                                f"cuisines recommended: {_cuisines(rows)}"])
    # The user's "hangi mutfakları denemiştim" is what they told; the app's
    # own visit records can hold more, so they are kept apart.
    told = [v for v in rows if v.get("source") != "app"]
    app_only = sorted({_cuisine(v["restaurant_id"]) for v in rows} - {_cuisine(v["restaurant_id"]) for v in told})
    return True, "\n".join(
        [f"{head}: {len(rows)} visit(s) to {len({v['restaurant_id'] for v in rows})} different restaurant(s)",
         *lines, f"cuisines the user told you about: {_cuisines(told)}"]
        + ([f"cuisines only in the app's records: {', '.join(app_only)}"] if app_only else []))


def _menu(args: dict, data: AppData) -> tuple[bool, str]:
    c = data.constraints
    r = _restaurant(args.get("restaurant"), data.world)
    if r is None:
        return False, f"error: no restaurant named '{args.get('restaurant', '')}'."
    items = [m for m in data.world.menu_items if m.restaurant_id == r.id]
    want = fold(str(args.get("category") or ""))
    picked = [m for m in items if want and want in fold(m.category_name)] or items
    note = f" (no '{args['category']}' section; whole menu)" if want and picked is items else ""
    safe = [m for m in picked if c.allows(m)]
    out = [f"get_menu {r.name}{note}: {len(safe)} of {len(picked)} dish(es) safe. Safe for: {c.label()}"]
    out += [f"- {_dish(m)} · {m.category_name}" for m in safe]
    left = [m for m in picked if not c.allows(m)]
    if left:
        out.append("left out: " + "; ".join(f"{m.name} ({c.why_not(m)})" for m in left))
    return True, "\n".join(out)


#: Kitchens where an allergen is the trade, so a safe dish still shares the pans with it.
CROSS_CONTACT = {"deniz ürünleri": {"fish", "molluscs", "crustaceans"}}


def _nearby(args: dict, data: AppData) -> tuple[bool, str]:
    c = data.constraints
    want_c, want_a = fold(str(args.get("cuisine") or "")), fold(str(args.get("area") or ""))
    places = data.world.restaurants
    # A filter no place matches ("any", "herhangi", a district with no restaurant) is dropped, not obeyed.
    ignored = []
    if want_c and not any(want_c in fold(_cuisine(r.id)) or any(want_c in fold(t) for t in r.tags) for r in places):
        ignored, want_c = ignored + [f"cuisine '{args['cuisine']}'"], ""
    if want_a and not any(want_a in fold(r.area) for r in places):
        ignored, want_a = ignored + [f"area '{args['area']}'"], ""
    rows = []
    for r in places:
        label = _cuisine(r.id)
        if want_c and want_c not in fold(label) and not any(want_c in fold(t) for t in r.tags):
            continue
        if want_a and want_a not in fold(r.area):
            continue
        until = _open_until(r, data.now)
        dishes = [m for m in data.world.menu_items if m.restaurant_id == r.id and fold(m.category_name) != "icecek"]
        safe = [m for m in dishes if c.allows(m)]
        if until and safe:
            mostly = 2 * len(safe) > len(dishes) and not (CROSS_CONTACT.get(label, set()) & set(c.allergens))
            rows.append((label not in c.likes, _km(data.home, (r.lat, r.lng)), r, label, until, safe, mostly))
    if any(row[6] for row in rows):                    # a place where half the menu or more is off limits only as a last resort
        rows = [row for row in rows if row[6]]
    rows.sort(key=lambda x: (x[0], x[1]))              # cuisines the user likes first, then nearest
    out = [f"search_restaurants: {len(rows)} open place(s) with a dish safe for: {c.label()}"]
    if ignored:
        out.append(f"(no place matches {' or '.join(ignored)}: that filter was left out)")
    out += [f"- {r.name} [{label}{'' if other else ', the user likes it'}] {r.area}, {km:.1f} km, "
            f"open until {until} | safe: {'; '.join(_dish(m) for m in safe[:2])}"
            for other, km, r, label, until, safe, _ in rows[:3]]
    return True, "\n".join(out)


def _posts(args: dict, data: AppData) -> tuple[bool, str]:
    r = _restaurant(args.get("restaurant"), data.world)
    if r is None:
        return False, f"error: no restaurant named '{args.get('restaurant', '')}'."
    posts = [p for p in data.world.posts if p.restaurant_id == r.id]
    if not posts:
        return True, f"search_posts {r.name}: no posts or reviews about this place in the app, and no rating."
    return True, "\n".join([f"search_posts {r.name}: {len(posts)} post(s)"]
                           + [f'- @{p.author} ({p.created_at[:10]}): "{p.caption}"' for p in posts])


def run_tool(name: str, args: dict, data: AppData) -> tuple[bool, str]:
    """``(ok, text for the model)``; a bad call gets an error the model can fix."""
    name = name.removesuffix("()")             # "search_saved()": v0.3 run C, 6 of 8 saved-count calls
    if name not in ARGS:
        return False, f"error: no tool named '{name}'."
    unknown = sorted(set(args) - set(ARGS[name]))
    if unknown:
        return False, f"error: {name} has no argument '{unknown[0]}'. Arguments: {', '.join(ARGS[name]) or 'none'}."
    if name == "search_saved":
        lines = [f"- {_name(data.world, s['restaurant_id'])} (saved {s['saved_at'][:10]})" for s in data.saved]
        return True, "\n".join([f"search_saved: {len(lines)} saved restaurant(s)"] + lines)
    if name in ("my_visits", "my_recommendations"):
        return _logs(name, args, data)
    return {"get_menu": _menu, "search_restaurants": _nearby, "search_posts": _posts}[name](args, data)
