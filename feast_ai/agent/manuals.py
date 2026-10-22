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
``active2``  ``active`` with a sharper index (:data:`INDEX2`) and a compact
             manual (:func:`render_manual_compact`): the same hints and
             examples, minus what its grammar already enforces.

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


# --------------------------------------------------------------------------
# active2: a sharper index, and manuals cut to what a grammar cannot say
# --------------------------------------------------------------------------
#: One line per tool for ``active2``'s first pass. ``active`` picked from
#: names plus a bare purpose, and its misses were neighbours confused - shared
#: items read as chat memory, a moved search area as a venue search, a dish
#: with a price as a venue, a feeling as a venue kind, the QR scanner as a
#: menu search, suggestions as friends.  Each line here states the tool's
#: scope against its neighbour, as the OpenAI and Anthropic tool guides ask
#: ("describe when, and when not, to use each function").  The quality
#: examples for ``ask_about_posts`` deliberately avoid the ones the suite
#: tests (working on a laptop, quiet, a view, pets): the model has to
#: generalise from the category, not match a word.  The whole block lives in
#: the static prompt, so after the first message it is cached and costs no
#: prefill per turn.
INDEX2: dict[str, str] = {
    "search_restaurants": "venues by kind or cuisine (a fish restaurant, a kebab house, a bar), "
                          "filtered by open now, a time, distance or popularity",
    "search_menu_items": "a dish or drink by name: where it is served and what it costs "
                         "(cheapest first, a price limit)",
    "get_restaurant_details": "hours, address or menu of a venue already listed in this chat",
    "get_menu_item_details": "ingredients or price of a dish already listed in this chat",
    "get_directions": "the way to a venue already listed in this chat",
    "set_active_location": "move where every search looks: a district or city the user "
                           "names or is heading to, or back to their own GPS position "
                           "(searching comes in the next turn)",
    "search_posts": "what people posted about a dish, a venue or a keyword",
    "ask_about_posts": "places with a quality no tag or keyword captures, judged from what "
                       "people wrote: the atmosphere, romantic, live music, a garden, good "
                       "for groups",
    "search_saved": "the user's own saved venues, dishes or posts: show, filter or sort them",
    "search_shared_items": "venues and dishes sent in chats: by a friend to the user, or by "
                           "the user to a friend",
    "search_chat_memory": "the user's earlier conversations with you, feast AI (not messages "
                          "between friends)",
    "list_friends": "people who are already the user's friends",
    "get_suggested_users": "new people the user may know and could add as friends",
    "open_screen": "open a screen of the app: saved, chats, friends, add_friends, feed, map, "
                   "profile, notifications, settings, qr_scanner",
    "send_message_to_friends": "write to friends: pick it when they ask (you then propose the "
                               "text) and when they answer your proposal",
}

#: A cue-first decision list after the index: the cue, then the tool, most
#: specific first. Small models are pulled by tool *names* (the Hammer paper's
#: finding); a line that states the user's cue before the name lets the cue
#: decide. Measured on a 48-sentence dev set written for this (none from the
#: suite; word-overlap checked), pass-A picks with Gemma 4 E2B:
#: index alone 38/48, index + this list 44/48, this list alone 43/48 - the
#: misses it fixed were vibe questions read as venue or post searches, the
#: map screen read as a venue search, and a moved search area answered with
#: a clarifying question. Rewording the message and no-tool lines (one reply
#: began "plain text: ...") gave 45/48; moving the "listed earlier" line above
#: the venue line (a listed venue's hours had matched "opening hours" there)
#: gave 47/48. Exact full-name scoring from llama.cpp logprobs
#: (instead of greedy decoding over the grammar) changed none of 21 picks:
#: the pick distributions were 0.9-1.0 peaked, so it is not used.
CUES2: tuple[tuple[str, str], ...] = (
    ("how a place feels or who it suits (mood, atmosphere, music, romantic, for kids or "
     "groups)", "ask_about_posts"),
    ("another district or city to search in, or back to the phone's own location",
     "set_active_location"),
    ("open or show a part of the app (map, chats, feed, profile, settings, notifications, "
     "QR scanner, the saved screen)", "open_screen"),
    ("something a friend sent the user, or the user sent a friend", "search_shared_items"),
    ("something the user told you or asked you before", "search_chat_memory"),
    ("the user's own saved items, to filter or sort", "search_saved"),
    ("a venue or dish you listed earlier: its hours, address, menu or ingredients",
     "get_restaurant_details or get_menu_item_details; the way there -> get_directions"),
    ("a dish or drink: where, how much, cheapest, under a price", "search_menu_items"),
    ("a kind of venue or cuisine, open now or at a time, distance, popular",
     "search_restaurants"),
    ("what people posted about something", "search_posts"),
    ("the user's friends", "list_friends; new people to add or follow (suggestions) -> "
     "get_suggested_users"),
    ("a message to a friend (write, tell, send), or their answer to your proposal",
     "send_message_to_friends"),
)
CUES2_TAIL = ("Anything else - greetings, thanks, questions about you, the time, date or "
              "location, things no tool does - needs no tool: answer it directly.")

#: Hints the compact manual drops: the parameter's name already says it, and
#: the grammar enforces its type, enum or tag list (~10 prompt tokens each,
#: ~110 ms of prefill on the 3-core budget).
TERSE: dict[str, tuple[str, ...]] = {
    "search_restaurants": ("tags", "open_now", "open_at", "max_walk_minutes",
                           "max_distance_meters"),
    "search_menu_items": ("open_now", "max_walk_minutes"),
    "get_directions": ("mode",),
    "open_screen": ("screen",),
}

#: ``send_message_to_friends`` is the one tool that changes anything, and the
#: app knows which step it is in (did its last reply propose a message?), so
#: each step gets its own manual - and its own grammar
#: (:mod:`feast_ai.agent.grammar`): no call is possible before a proposal.
#: On synthetic chats the bare propose rule once asked "what should I write?"
#: although the user had said it, and the confirm step dropped the venue id
#: of a listed place. A worked proposal example ("Bora'ya ...") was then
#: copied - a request for Burak was proposed to Bora - so the proposal gets
#: a template in obvious capitals (the rule of CALL_FORMAT), and only the
#: confirm step, which copies names from the proposal above it, keeps a
#: worked example (a name, venue and id the suite does not use).
SEND_STEPS: dict[str, tuple[str, ...]] = {
    "propose": (
        "Do NOT send yet. Say who gets it and the exact message, then ask "
        "\"Göndereyim mi?\". Never say a message was sent.",
        'Form, with their real names and words: NAME\'a "MESSAGE" göndereyim mi?',
    ),
    "confirm": (
        "The user answered your proposal. If they confirmed, reply with the call - the same "
        "names and wording, plus attach_restaurant_id if the message is about a listed venue:",
        '<tool_call>{"name": "send_message_to_friends", "arguments": {...}}</tool_call>',
        "If they cancelled or changed it, reply in plain text and do not call. Never say a "
        "message was sent unless you called the tool.",
        'e.g. after a list with Liman (id: r_903): Bora\'ya "Liman\'da buluşalım" göndereyim '
        'mi? -> "evet" -> <tool_call>{"name": "send_message_to_friends", "arguments": '
        '{"friend_names": ["Bora"], "message": "Liman\'da buluşalım", "attach_restaurant_id": '
        '"r_903"}}</tool_call>',
    ),
}

#: Shared last line of a compact manual. The call's opening is prefilled, so
#: the model writes the arguments object only; "earlier" is what lets a
#: follow-up ("and under 100 lira") keep the values of the call it refines.
ARGS_ONLY = "Use every value the user gave, now or earlier in this chat; leave the rest out."


def render_index2() -> str:
    return "\n".join(f"{t.name}: {INDEX2[t.name]}" for t in TOOLS)


def render_cues2() -> str:
    lines = [f"- {cue} -> {tool}" for cue, tool in CUES2]
    return "WHICH TOOL - take the first line that fits:\n" + "\n".join(lines) + "\n" + CUES2_TAIL


def render_manual_compact(name: str, tags: tuple[str, ...], step: str = "") -> str:
    """``active2``'s manual: signature, non-obvious hints, one example as a
    bare arguments object. Hint-less parameters share one line.

    ``active``'s manual spent ~75 of its 150-356 tokens on a header and a
    trailer about the call format; with the opening prefilled and a grammar
    on the arguments, format words buy nothing.
    """
    spec: ToolSpec = BY_NAME[name]
    if name in PROTOCOL and step == "propose":
        # No call can follow (text-only grammar): its arguments are noise.
        return "\n".join([name] + [f"  {rule}" for rule in SEND_STEPS["propose"]])
    hints = HINTS.get(name, {})
    terse = set(TERSE.get(name, ()))
    lines = [f"{name} (* = required)"]
    bare: list[str] = []
    for param in spec.params:
        if param.name == "limit":
            continue
        hint = "" if param.name in terse else hints.get(param.name, "")
        if hint:
            if bare:
                lines.append("  " + ", ".join(bare))
                bare = []
            lines.append(f"  {param.render(tags)}: {hint}")
        else:
            bare.append(param.render(tags))
    if bare:
        lines.append("  " + ", ".join(bare))
    for group in spec.exactly_one:
        lines.append(f"  Give exactly one of {' / '.join(group)}.")
    if name in PROTOCOL:
        lines += [f"  {rule}" for rule in SEND_STEPS[step or "confirm"]]
        return "\n".join(lines)
    lines += [f"  {rule}" for rule in spec.rules]
    for text, args in EXAMPLES.get(name, ()):
        lines.append(f'  e.g. "{text}" -> {json.dumps(args, ensure_ascii=False)}')
    lines.append(ARGS_ONLY)
    return "\n".join(lines)
