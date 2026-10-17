"""Classes: the second routing taxonomy, cut along what the SLM must *do*.

The 19 intents of :mod:`feast_ai.agent.intents` were cut along what a user
*says*, which splits one action across classes the routers confuse: venue vs
dish search (both "find food"), saved-list vs saved-filter, friends vs
suggestions, three flavours of "answer in text".  The ModernBERT router's
misses on the kanka suite were mostly inside those pairs, and Decision 2.0
paid ~25 prompt tokens per option per message for distinctions the SLM did
not need.  Here each class is one prompt: 13 classes, each with

* ``summary`` - what the routers read (Decision 2.0's option description);
* ``task`` and ``rules`` - the class prompt's own instructions, so a correct
  route buys more than a shorter tool list: which of its tools fits which
  phrasing, and the protocol (confirm-before-send lives in ``message``);
* ``tools`` - whose manuals (:mod:`feast_ai.agent.manuals`) the prompt carries.

The router exemplars are the old intents' exemplars, relabelled - nothing new
was written, so nothing new can leak from the suite.  The suite still labels
turns with the old intent ids; :func:`gold` maps them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import intents as v1


@dataclass(frozen=True)
class ClassSpec:
    id: str
    route: str  # "slm" | "direct" (no-tool: needs confidence) | "inherit"
    tools: tuple[str, ...]
    summary: str
    task: str = ""
    rules: tuple[str, ...] = ()
    merges: tuple[str, ...] = ()
    exemplars: tuple[str, ...] = field(default=())


_SPECS = (
    ClassSpec(
        "food", "slm", ("search_menu_items", "search_restaurants", "set_active_location"),
        "looking for food nearby: a place to eat or drink, or a dish, a drink or its price",
        "The user is looking for food: a venue or a dish.",
        ("A dish, a drink or a price -> search_menu_items. A kind of venue or cuisine -> "
         "search_restaurants.",
         "Put every limit the user gives (price, distance, open now) into the call.",
         "If the user moves to another place, call set_active_location first."),
        ("restaurant_search", "menu_search"),
    ),
    ClassSpec(
        "vibe", "slm", ("ask_about_posts", "search_restaurants", "search_posts"),
        "a place with a feeling no tag captures: quiet, cosy, good for working, a view, "
        "pet- or child-friendly, romantic",
        "The user wants a place with a quality only people's posts can tell.",
        ("Use ask_about_posts with one plain question about that quality.",
         "Use search_restaurants only if the user adds a plain filter to a place you "
         "already found (distance, open now)."),
        ("vibe_search",),
    ),
    ClassSpec(
        "posts", "slm", ("search_posts", "ask_about_posts"),
        "what other people posted or shared about a food, a place or a topic",
        "The user asks what people posted.",
        ("A food, venue or keyword -> search_posts with that keyword.",
         "An opinion question over many posts -> ask_about_posts."),
        ("posts",),
    ),
    ClassSpec(
        "details", "slm", ("get_restaurant_details", "get_menu_item_details", "get_directions"),
        "a follow-up on one venue or dish already listed in this chat: hours, address, "
        "menu, ingredients, or the way there",
        "The user asks about a venue or dish you listed earlier.",
        ("Take its id from your earlier list: first = 1., second = 2. Never invent an id.",
         "Hours, address, menu -> get_restaurant_details. A dish -> get_menu_item_details. "
         "The way there -> get_directions."),
        ("details",),
    ),
    ClassSpec(
        "location", "slm", ("set_active_location",),
        "change the search area to another city or neighbourhood, or back to the "
        "phone's own location",
        "The user wants to change where you search.",
        ("A named place -> location_query. Their own, current or GPS position -> "
         "use_current_location: true.",),
        ("location_change", "location_reset"),
    ),
    ClassSpec(
        "saved", "slm", ("search_saved", "open_screen"),
        "the user's own saved items: open the saved list, or filter or ask about it",
        "The user means their own saved items.",
        ("Just show the list -> open_screen with screen \"saved\".",
         "A question or filter about them (which is open, nearest, cheapest, a kind) -> "
         "search_saved."),
        ("saved_open", "saved_search"),
    ),
    ClassSpec(
        "shared", "slm", ("search_shared_items", "list_friends"),
        "venues or dishes the user and friends sent each other in chats",
        "The user asks about venues or dishes sent in chats.",
        ("Use search_shared_items. A named friend always goes in friend_name.",),
        ("shared",),
    ),
    ClassSpec(
        "memory", "slm", ("search_chat_memory",),
        "something from the user's earlier conversations with you, the assistant",
        "The user asks about an earlier conversation with you.",
        ("Always search with the topic as query; never say you don't remember first.",),
        ("memory",),
    ),
    ClassSpec(
        "people", "slm", ("list_friends", "get_suggested_users", "open_screen"),
        "the user's friends, or new people they may know and could add",
        "The user asks about friends.",
        ("Their friends -> list_friends. New people to add -> get_suggested_users.",),
        ("friends", "suggested_users"),
    ),
    ClassSpec(
        "message", "slm", ("send_message_to_friends", "list_friends"),
        "write or send a message to friends",
        "The user wants to message friends.",
        ("Follow send_message_to_friends' two steps exactly.",),
        ("message",),
    ),
    ClassSpec(
        "navigate", "direct", ("open_screen",),
        "open a screen of the app: settings, profile, notifications, chats, map, feed, "
        "QR scanner",
        "The user wants a screen of the app.",
        ("Call open_screen with the screen from its list.",),
        ("navigate",),
    ),
    ClassSpec(
        "talk", "direct", (),
        "small talk, thanks, questions about you, the time, date or current location, or "
        "anything the app does not do (weather, alarms, money, music)",
        "Answer in plain text.",
        ("Time, date and location: answer from CONTEXT.",
         "Greetings and thanks: reply briefly and warmly.",
         "Things this app cannot do (weather, alarms, money, music): say so and say what "
         "you can help with."),
        ("context_info", "chitchat", "out_of_scope"),
    ),
    ClassSpec(
        "followup", "inherit", (),
        "a short reply that only makes sense after the previous turn: yes, no, confirm, "
        "cancel, more, cheaper, or one more filter",
        merges=("followup",),
    ),
)


def _with_exemplars(spec: ClassSpec) -> ClassSpec:
    texts = tuple(t for old in spec.merges for t in v1.BY_ID[old].exemplars)
    return ClassSpec(spec.id, spec.route, spec.tools, spec.summary, spec.task, spec.rules,
                     spec.merges, texts)


CLASSES: tuple[ClassSpec, ...] = tuple(_with_exemplars(s) for s in _SPECS)
BY_ID: dict[str, ClassSpec] = {c.id: c for c in CLASSES}
FROM_V1: dict[str, str] = {old: c.id for c in CLASSES for old in c.merges}


def gold(label: str) -> str:
    """A suite label (``"followup|menu_search"``) in class ids (``"followup|food"``)."""
    out: list[str] = []
    for old in label.split("|"):
        new = FROM_V1.get(old, old)
        if new not in out:
            out.append(new)
    return "|".join(out)


def exemplar_bank() -> list[tuple[str, str]]:
    return [(c.id, text) for c in CLASSES for text in c.exemplars]
