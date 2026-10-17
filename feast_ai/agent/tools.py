"""The feast mobile agent's tools: one definition, rendered and enforced.

Each tool is data, not prose, so the same definition does two jobs:

* :meth:`ToolSpec.render` writes the compact line format the system prompt
  uses (``query*(str), limit(int=10)``) - dense on purpose, because every
  prompt token is prefill a phone CPU has to pay for;
* :func:`validate` checks a call the model emitted against it, which is how
  hallucination is measured: an invented tool, an invented argument or a value
  outside an enum is a defect the report can count, not a vibe.

The first eleven tools are the project's original design.  Four were added,
all read-only, each grounded in something feast mobile actually has:

``get_menu_item_details``  dishes are first-class menu items in the backend,
                           so a dish needs a follow-up just like a venue;
``get_directions``         the obvious next step after picking a venue -
                           opens the phone's maps app, changes nothing;
``open_screen``            navigation ("kaydedilenlerimi goster") - the
                           cheapest possible action and the canonical case
                           for skipping the SLM entirely;
``get_suggested_users``    the friends-of-friends list the app already shows
                           on the add-friends screen and in the feed.

``send_message_to_friends`` remains the only tool that changes anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HHMM = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d$")


@dataclass(frozen=True)
class Param:
    name: str
    type: str  # "str" | "int" | "num" | "bool" | "str[]" | "hhmm"
    required: bool = False
    enum: tuple[str, ...] = ()
    default: object = None
    #: Free-text hint rendered after the type, e.g. ``TRY``.
    note: str = ""
    #: ``"tags"`` -> allowed values come from the context's tag list.
    choices_from: str = ""

    def render(self, tags: tuple[str, ...] = ()) -> str:
        star = "*" if self.required else ""
        if self.enum:
            inner = "|".join(f'"{v}"' for v in self.enum)
            if self.default is not None:
                inner += f'="{self.default}"'
        elif self.type == "hhmm":
            inner = '"HH:MM"'
        elif self.choices_from == "tags":
            inner = "str[]: " + ", ".join(tags)
        else:
            inner = self.type
            if self.default is not None:
                inner += f"={self.default}"
        if self.note:
            inner += f", {self.note}"
        return f"{self.name}{star}({inner})"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: tuple[Param, ...] = ()
    #: Groups of parameters of which exactly one must be given.
    exactly_one: tuple[tuple[str, ...], ...] = ()
    #: Extra lines printed under the signature (the confirmation protocol).
    rules: tuple[str, ...] = ()
    mutates: bool = False
    added: bool = False  # not in the original design - see module docstring

    def param(self, name: str) -> Param | None:
        return next((p for p in self.params if p.name == name), None)

    @property
    def required(self) -> list[str]:
        return [p.name for p in self.params if p.required]

    def signature(self, tags: tuple[str, ...] = ()) -> str:
        if self.exactly_one:
            grouped = {n for group in self.exactly_one for n in group}
            parts = [
                " OR ".join(self.param(n).render(tags) for n in group) + " — exactly one"
                for group in self.exactly_one
            ]
            parts += [p.render(tags) for p in self.params if p.name not in grouped]
            return ", ".join(parts)
        return ", ".join(p.render(tags) for p in self.params)

    def render(self, tags: tuple[str, ...] = ()) -> str:
        lines = [f"{self.name} — {self.description}"]
        signature = self.signature(tags)
        if signature:
            lines.append(signature)
        lines += list(self.rules)
        return "\n".join(lines)


_WALK = Param("max_walk_minutes", "int")
_DIST = Param("max_distance_meters", "int")
_ITEM_TYPES = ("restaurant", "menu_item", "post", "any")

SCREENS = (
    "saved", "chats", "friends", "add_friends", "feed", "map", "profile",
    "notifications", "settings", "qr_scanner",
)

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "search_restaurants",
        "venues (restaurants, cafés, breakfast places) near the active location.",
        (
            Param("query", "str"),
            Param("tags", "str[]", choices_from="tags"),
            Param("open_now", "bool"),
            Param("open_at", "hhmm"),
            _WALK,
            _DIST,
            Param("sort_by", "str", enum=("distance", "popularity", "name")),
            Param("limit", "int", default=10),
        ),
    ),
    ToolSpec(
        "search_menu_items",
        "dishes, drinks and prices inside nearby menus.",
        (
            Param("query", "str", required=True),
            Param("max_price", "num", note="TRY"),
            Param("min_price", "num"),
            Param("open_now", "bool"),
            _WALK,
            _DIST,
            Param("sort_by", "str", enum=("price_asc", "price_desc", "distance")),
            Param("limit", "int", default=10),
        ),
    ),
    ToolSpec(
        "get_restaurant_details",
        "one venue's address, hours, tags and full menu. Follow-up only.",
        (Param("restaurant_id", "str", required=True, note="from an earlier result"),),
    ),
    ToolSpec(
        "get_menu_item_details",
        "one dish's description, price and the venue serving it. Follow-up only.",
        (Param("menu_item_id", "str", required=True, note="from an earlier result"),),
        added=True,
    ),
    ToolSpec(
        "get_directions",
        "open the route to one venue in the phone's maps app. Changes nothing.",
        (
            Param("restaurant_id", "str", required=True, note="from an earlier result"),
            Param("mode", "str", enum=("walk", "drive", "transit"), default="walk"),
        ),
        added=True,
    ),
    ToolSpec(
        "set_active_location",
        "move the search area to another city or neighbourhood, or back to GPS. "
        "Then search in the NEXT turn.",
        (Param("location_query", "str"), Param("use_current_location", "bool")),
        exactly_one=(("location_query", "use_current_location"),),
    ),
    ToolSpec(
        "search_posts",
        "keyword search over nearby user posts (caption + tagged venue).",
        (
            Param("query", "str", required=True),
            _WALK,
            _DIST,
            Param("limit", "int", default=20),
        ),
    ),
    ToolSpec(
        "ask_about_posts",
        "a separate reading pass over nearby captions, for a subjective quality no "
        "keyword or tag captures (working with a laptop, quiet, good view, "
        "pet-friendly, child-friendly). Slower; use only when needed.",
        (
            Param("question", "str", required=True,
                  note="one plain question in the user's language"),
            Param("scope", "str", enum=("nearby", "saved", "all"), default="nearby"),
            Param("limit", "int", default=20),
        ),
    ),
    ToolSpec(
        "search_saved",
        "only the items this user saved on this phone.",
        (
            Param("item_type", "str", enum=_ITEM_TYPES, default="any"),
            Param("query", "str"),
            Param("location_query", "str"),
            _WALK,
            _DIST,
            Param("sort_by", "str",
                  enum=("recency", "price_asc", "price_desc", "distance")),
            Param("limit", "int", default=20),
        ),
    ),
    ToolSpec(
        "search_shared_items",
        "venues and dishes sent in chats, by or to the user.",
        (
            Param("direction", "str", enum=("sent", "received", "any"), default="any"),
            Param("item_type", "str", enum=_ITEM_TYPES),
            Param("friend_name", "str"),
            Param("query", "str"),
            Param("limit", "int", default=20),
        ),
    ),
    ToolSpec(
        "search_chat_memory",
        "your own earlier conversations with this user.",
        (Param("query", "str", required=True), Param("limit", "int", default=10)),
    ),
    ToolSpec(
        "list_friends",
        "the user's friends on feast, optionally filtered by name.",
        (Param("query", "str"),),
    ),
    ToolSpec(
        "get_suggested_users",
        "people the user may know (friends of friends), to add as friends.",
        (Param("limit", "int", default=20),),
        added=True,
    ),
    ToolSpec(
        "open_screen",
        "open a screen of the feast app. Changes nothing.",
        (Param("screen", "str", required=True, enum=SCREENS),),
        added=True,
    ),
    ToolSpec(
        "send_message_to_friends",
        "THE ONLY TOOL THAT CHANGES ANYTHING.",
        (
            Param("friend_names", "str[]", required=True),
            Param("message", "str", required=True),
            Param("attach_restaurant_id", "str"),
        ),
        rules=(
            "NEVER call it in the same turn the user asks. First state in plain text "
            "who receives it and the exact wording, and ask to confirm. Call it only "
            "after the user's next message confirms. If the wording changes, ask again.",
        ),
        mutates=True,
    ),
)

BY_NAME: dict[str, ToolSpec] = {t.name: t for t in TOOLS}
ALL_TOOL_NAMES: tuple[str, ...] = tuple(BY_NAME)


def render_tools(names: tuple[str, ...] | list[str], tags: tuple[str, ...]) -> str:
    """The TOOLS block for a subset, in registry order (stable prompts cache)."""
    wanted = set(names)
    return "\n\n".join(t.render(tags) for t in TOOLS if t.name in wanted)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
@dataclass
class Validation:
    """Defects in one call. ``issues`` are hallucinations; ``warnings`` are a
    right value in a sloppy type ("10" for 10) that an app could coerce."""

    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _looks_number(value: object) -> bool:
    try:
        float(str(value))
        return True
    except ValueError:
        return False


def _check_value(param: Param, value: object, tags: tuple[str, ...], out: Validation) -> None:
    name = param.name
    if param.enum:
        if value not in param.enum:
            out.issues.append(f"{name}={value!r} is not one of {list(param.enum)}")
        return
    kind = param.type
    if kind == "str":
        if not isinstance(value, str):
            out.warnings.append(f"{name} should be a string, got {type(value).__name__}")
    elif kind == "hhmm":
        if not (isinstance(value, str) and _HHMM.match(value)):
            out.issues.append(f"{name}={value!r} is not HH:MM")
    elif kind == "bool":
        if not isinstance(value, bool):
            if str(value).lower() in ("true", "false"):
                out.warnings.append(f"{name} is a string, not a bool")
            else:
                out.issues.append(f"{name}={value!r} is not a bool")
    elif kind in ("int", "num"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            if _looks_number(value):
                out.warnings.append(f"{name} is a string, not a number")
            else:
                out.issues.append(f"{name}={value!r} is not a number")
        elif kind == "int" and isinstance(value, float) and not value.is_integer():
            out.warnings.append(f"{name}={value} should be a whole number")
    elif kind == "str[]":
        items = value if isinstance(value, list) else [value]
        if not isinstance(value, list):
            out.warnings.append(f"{name} should be a list")
        if param.choices_from == "tags" and tags:
            unknown = [v for v in items if v not in tags]
            if unknown:
                out.issues.append(f"{name} has tags not in the list: {unknown}")


def validate(name: str, args: dict, tags: tuple[str, ...] = ()) -> Validation:
    """Check ``args`` against the tool's schema. Assumes ``name`` exists."""
    spec = BY_NAME[name]
    out = Validation()
    for key, value in args.items():
        param = spec.param(key)
        if param is None:
            out.issues.append(f"unknown argument '{key}'")
            continue
        if value is None:
            continue
        _check_value(param, value, tags, out)
    for required in spec.required:
        if args.get(required) in (None, "", []):
            out.issues.append(f"missing required '{required}'")
    for group in spec.exactly_one:
        given = [n for n in group if args.get(n) not in (None, "", False)]
        if len(given) != 1:
            out.issues.append(f"exactly one of {list(group)} must be given, got {given or 'none'}")
    return out
