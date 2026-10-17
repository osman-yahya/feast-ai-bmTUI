"""Tool manuals: the long form of each tool, shown only when it is needed.

The compact signature in :mod:`feast_ai.agent.tools` keeps the all-tools
prompt short, and the first Gemma 4 E2B benchmarks show what that costs: the
model invents tool names (``screen_open``, ``use_current_location``, a tool
called ``query``), drops arguments it was never told about (a friend's name,
the id of the venue a message is about) and calls in its own trained syntax.
A manual fixes that with what a signature cannot say: a hint per argument,
one worked example written as a complete call, and the tool's protocol.

Two pipelines use the same manuals, so a difference between them is caused by
routing, not by wording:

``active``   one prompt lists every tool by name and purpose only; the model
             picks one, the app appends that tool's manual, the model writes
             the call, and the manual is dropped from the history again;
``classes``  the router picks a class, and the class prompt carries the full
             manual of each of its tools.

The examples follow the rule of the router exemplars: none paraphrases a
sentence of the benchmark suite (:mod:`feast_ai.agent.suite`).  They teach an
argument with a different dish, place or name than the suite tests.
"""

from __future__ import annotations

import json

from .tools import BY_NAME, TOOLS, ToolSpec

#: One line per tool for the ``active`` pipeline's first pass.
INDEX: dict[str, str] = {
    "search_restaurants": "find venues (restaurants, cafés, bars) near the active location",
    "search_menu_items": "find dishes and drinks, and their prices, in nearby menus",
    "get_restaurant_details": "hours, address and menu of one venue listed earlier",
    "get_menu_item_details": "ingredients and price of one dish listed earlier",
    "get_directions": "the route to one venue listed earlier",
    "set_active_location": "move the search area elsewhere, or back to the phone's GPS",
    "search_posts": "what people posted nearby about a food, venue or keyword",
    "ask_about_posts": "places with a quality only posts reveal (quiet, cosy, a view, pet-friendly)",
    "search_saved": "the user's own saved items",
    "search_shared_items": "venues and dishes sent in chats, by or to the user",
    "search_chat_memory": "the user's earlier conversations with you",
    "list_friends": "the user's friends",
    "get_suggested_users": "people the user may know, to add as friends",
    "open_screen": "open a screen of the app",
    "send_message_to_friends": "send a message to friends (only after the user confirms it)",
}

#: Per-argument hints. Missing ones render the bare signature.
HINTS: dict[str, dict[str, str]] = {
    "search_restaurants": {
        "query": "the kind of venue or cuisine, e.g. \"balıkçı\"",
        "tags": "only tags from this list",
        "open_now": "true if they want it open now",
        "open_at": "a time they gave",
        "max_walk_minutes": "a walking limit they gave",
        "max_distance_meters": "a distance limit they gave",
        "sort_by": "\"popularity\" for popular or best-known",
    },
    "search_menu_items": {
        "query": "the dish or drink only, without filler words",
        "max_price": "a price limit they gave, in lira",
        "open_now": "true if they want it open now",
        "max_walk_minutes": "a walking limit they gave",
        "sort_by": "\"price_asc\" for the cheapest, \"price_desc\" for the most expensive",
    },
    "get_restaurant_details": {"restaurant_id": "the id written next to that venue earlier"},
    "get_menu_item_details": {"menu_item_id": "the id written next to that dish earlier"},
    "get_directions": {
        "restaurant_id": "the id written next to that venue earlier",
        "mode": "walk, drive or transit as they said",
    },
    "set_active_location": {
        "location_query": "the city or neighbourhood they named",
        "use_current_location": "true to go back to the phone's own location",
    },
    "search_posts": {"query": "the food, venue or keyword"},
    "ask_about_posts": {"question": "the quality they want, as one plain question"},
    "search_saved": {
        "item_type": "restaurant for venues, menu_item for dishes, post for posts",
        "query": "a word to filter by",
        "sort_by": "\"price_asc\" for cheapest first, \"distance\" for nearest",
    },
    "search_shared_items": {
        "direction": "\"received\" if sent to the user, \"sent\" if the user sent it",
        "friend_name": "the friend's name, whenever they name one",
        "query": "a word to filter by",
    },
    "search_chat_memory": {"query": "the topic to look for"},
    "list_friends": {"query": "a name to look for"},
    "open_screen": {"screen": "one of the listed screens"},
    "send_message_to_friends": {
        "friend_names": "list of names",
        "message": "the exact text the user agreed to",
        "attach_restaurant_id": "the id of the venue the message is about, if any",
    },
}

#: ``(user message, arguments)`` - one per tool, a complete call each.
#: ``r_9xx`` / ``m_9xx`` ids stand for "the id listed next to it earlier".
EXAMPLES: dict[str, tuple[tuple[str, dict], ...]] = {
    "search_restaurants": (("1 km içinde vegan seçeneği olan bir yer",
                            {"query": "vegan", "max_distance_meters": 1000}),),
    "search_menu_items": (("Buralarda en pahalı biftek hangisi",
                           {"query": "biftek", "sort_by": "price_desc"}),),
    "get_restaurant_details": (("(after a list) Üçüncüsünün menüsünü göster",
                                {"restaurant_id": "r_903"}),),
    "get_menu_item_details": (("(after a list) İkinci yemeğin fiyatı neydi",
                               {"menu_item_id": "m_902"}),),
    "get_directions": (("(after a list) Üçüncüsüne toplu taşımayla gideyim",
                        {"restaurant_id": "r_903", "mode": "transit"}),),
    "set_active_location": (("Bostancı'ya göre ara", {"location_query": "Bostancı"}),
                            ("Telefonumun konumunu kullan", {"use_current_location": True})),
    "search_posts": (("Ramen fotoğrafı paylaşan var mı", {"query": "ramen"}),),
    "ask_about_posts": (("Bahçesi olan sakin bir yer var mı",
                         {"question": "Bahçesi olan sakin bir yer var mı?"}),),
    "search_saved": (("Kaydettiğim balıkçılardan en yakını",
                      {"item_type": "restaurant", "query": "balıkçı", "sort_by": "distance"}),),
    "search_shared_items": (("Bora geçen ay bir balıkçı yollamıştı",
                             {"friend_name": "Bora", "direction": "received", "query": "balıkçı"}),),
    "search_chat_memory": (("Sana söylediğim alerji neydi", {"query": "alerji"}),),
    "list_friends": (("Ece arkadaşım mı", {"query": "Ece"}),),
    "get_suggested_users": (("Tanıyor olabileceğim insanlar", {}),),
    "open_screen": (("Haritayı aç", {"screen": "map"}),),
    "send_message_to_friends": (),  # the protocol below is its example
}

#: Protocol lines that replace the compact rule in a manual.
PROTOCOL: dict[str, tuple[str, ...]] = {
    "send_message_to_friends": (
        "Step 1 - the user asks for a message: do NOT call. Reply in text with who gets it "
        "and the exact message, and ask \"Göndereyim mi?\".",
        "Step 2 - the user's last message confirms your proposal: call it with the same "
        "names and wording, plus attach_restaurant_id if the message is about a listed venue.",
        "If they cancel or change it, do not call. Never say a message was sent unless "
        "you called this tool.",
        'e.g. "Bora\'ya 5 dakikaya oradayım de" -> Bora\'ya "5 dakikaya oradayım" '
        'göndereyim mi? -> "evet" -> <tool_call>{"name": "send_message_to_friends", '
        '"arguments": {"friend_names": ["Bora"], "message": "5 dakikaya oradayım"}}</tool_call>',
    ),
}


def _call(name: str, args: dict) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": args},
                                      ensure_ascii=False) + "</tool_call>"


def render_manual(name: str, tags: tuple[str, ...]) -> str:
    """The long form of one tool: purpose, a hint per argument, an example."""
    spec: ToolSpec = BY_NAME[name]
    hints = HINTS.get(name, {})
    lines = [f"{name} — {INDEX[name]}."]
    for param in spec.params:
        if param.name == "limit":  # a default-only knob: ~8 tokens of prefill per tool
            continue
        hint = hints.get(param.name, "")
        lines.append(f"  {param.render(tags)}" + (f": {hint}" if hint else ""))
    for group in spec.exactly_one:
        lines.append(f"  Give exactly one of {' / '.join(group)}.")
    lines += [f"  {rule}" for rule in PROTOCOL.get(name, spec.rules)]
    for text, args in EXAMPLES.get(name, ()):
        lines.append(f'  e.g. "{text}" -> {_call(name, args)}')
    return "\n".join(lines)


def render_manuals(names: tuple[str, ...] | list[str], tags: tuple[str, ...]) -> str:
    """Manuals for a tool set, in registry order (stable prompts cache)."""
    wanted = set(names)
    return "\n\n".join(render_manual(t.name, tags) for t in TOOLS if t.name in wanted)


def render_index() -> str:
    return "\n".join(f"{t.name} — {INDEX[t.name]}" for t in TOOLS)
