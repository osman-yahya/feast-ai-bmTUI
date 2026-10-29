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
