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
#: The two counting tools, in the Dart agent's style: short English
#: descriptions (fewer prefill tokens), values read back in Turkish.
APP_TOOLS = (
    "TOOLS\n"
    "my_visits(since, until, restaurant) - the restaurants this user went to, with the day of "
    "each visit and a count. Every argument is optional; dates are YYYY-MM-DD. Call it whenever "
    "the user asks how many places they went to, or to list their visits.\n"
    "search_saved() - the restaurants this user saved in the app, with a count."
)
APP_TOOL_NAMES = ("my_visits", "search_saved")
#: As on the phone (feast-cur ``AgentLimits``): tools run for one message, and
#: bad calls answered with an error before the message ends without an answer.
MAX_TOOL_CALLS = 3
MAX_INVALID_CALLS = 1
RESULT_NOTE = "Answer the user's question from this result, briefly, in the user's language."


def run_app_tool(name: str, args: dict, visits: list[dict], saved: list[dict],
                 names: dict[str, str]) -> tuple[bool, str]:
    """``(ok, text for the model)``.  ``visits``: the merged visit log
    (:func:`.engine.visits.merged`); ``saved``: the persona's ``saved_restaurants``."""
    import re
    from datetime import date

    from ..agent.text import fold

    if name == "search_saved":
        lines = [f"- {names[s['restaurant_id']]} (saved {s['saved_at'][:10]})" for s in saved]
        return True, "\n".join([f"search_saved: {len(lines)} saved restaurant(s)"] + lines)
    if name != "my_visits":
        return False, f"error: no tool named '{name}'. Tools: {', '.join(APP_TOOL_NAMES)}."
    unknown = sorted(set(args) - {"since", "until", "restaurant"})
    if unknown:
        return False, f"error: my_visits has no argument '{unknown[0]}'. Arguments: since, until, restaurant."
    rows = visits
    shown = []
    for key in ("since", "until"):
        value = args.get(key)
        if value in (None, ""):
            continue
        if not (isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)):
            return False, f"error: {key} must be a date written YYYY-MM-DD."
        rows = [v for v in rows if (v["date"] >= value if key == "since" else v["date"] <= value)]
        shown.append(f"{key}={value}")
    if args.get("restaurant"):
        want = fold(str(args["restaurant"]))
        rows = [v for v in rows if want in fold(v["restaurant"])]
        shown.append(f"restaurant={args['restaurant']}")
    head = (f"my_visits {' '.join(shown)}".rstrip() + f": {len(rows)} visit(s) to "
            f"{len({v['restaurant_id'] for v in rows})} different restaurant(s)")
    days = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    lines = [f"- {v['date']} {days[date.fromisoformat(v['date']).weekday()]} | {v['restaurant']}"
             for v in rows[-12:]]
    if len(rows) > 12:
        lines.insert(0, f"(the latest 12 of {len(rows)})")
    return True, "\n".join([head] + lines)
