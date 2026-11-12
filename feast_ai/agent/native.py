"""The native pipeline: the fine-tuned model's own tool format, one pass, guarded.

feast AI's other pipelines work *around* a stock model: a router narrows its
prompt, a grammar fences its output, a second pass shows it a manual.  This
one assumes a model that was trained on the 15 tools (the ``mustachio``
workspace) and asks it once, in the syntax Gemma 4 was built with - tools as
declarations, the answer as ``<|tool_call>call:name{...}<tool_call|>`` - which
is also the only syntax LiteRT-LM turns into tool calls on a phone.

Two things live here, and both are shared with training so they cannot drift:

**The prompt** (:data:`SYSTEM`, :func:`declarations`, :func:`user_content`).
The system turn is static.  The native chat template writes the declarations
*after* the system text, so a clock in it would re-prefill every declaration
each minute; the volatile state rides in a one-line ``[ctx]`` prefix on the
user's message instead.  Declarations come from the same ``ToolSpec`` objects
the validator uses.  ``terse`` is what a trained model needs (names, types,
enums, a few words per tool); ``full`` carries ``active2``'s descriptions and
per-argument hints and is what an untrained model should be judged with.

**The guard** (:func:`guard`): what the app can check in code after the
model has answered, so it does not have to be trusted for it.  It is the
grammar of ``active2`` moved behind the model - a grammar could only make an
argument well-formed, the guard can also see the conversation:

``schema``    unknown arguments dropped, types and enum values coerced, a
              clock time normalised ("24:00" is midnight, "9" is "09:00");
``ids``       a venue or dish id must be one the app has listed in this chat;
``grounding`` a searched dish, a place and a person must come from something
              the user wrote.  A near miss is repaired from the user's own
              words ("trioçe" where they typed "trileçe"); a value with no
              source is reported as ungrounded, which is the definition of a
              hallucinated argument;
``confirm``   ``send_message_to_friends`` runs only as the answer to the
              assistant's own proposal, with the names and wording of that
              proposal.  Anything else becomes a proposal again - the unsafe
              send turns into the question it skipped;
``fallback``  an answer that is neither a call nor a sentence ("]}") is
              replaced by a request to rephrase, never shown.

A call that passes is run, and its result goes back to the model for one
sentence.  That sentence is checked too, in :mod:`feast_ai.agent.replies`: a
number or a name the result does not contain replaces it with a sentence
written from the result.

Every intervention is returned in :attr:`Guarded.notes`, and the benchmark
reports the model with and without the guard, so the guard cannot hide what
the model did.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .calls import CallCheck
from .context import DEFAULT_TAGS, AgentContext
from .manuals import HINTS, INDEX2
from .text import detect_lang, fold
from .tools import BY_NAME, TOOLS, Param, ToolSpec, validate

SEND = "send_message_to_friends"
_JSON_TYPE = {"str": "string", "int": "integer", "num": "number", "bool": "boolean",
              "hhmm": "string"}

# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------
#: The prompt of the first terse run (``runs/v1``, ``runs/v2``): no persona,
#: and nothing about what to say after a result, because nothing ran the tool.
SYSTEM_V1 = (
    "You are feast AI, the assistant in the feast mobile app. You help with nearby "
    "restaurants, cafés and dishes, and with the user's saved items, posts, friends and "
    "chats.\n"
    "Call a tool when one fits; otherwise answer briefly. Never invent venues, dishes, "
    "prices, people or messages. Reply in the user's language; Turkish with English words "
    "mixed in is still Turkish.\n"
    "Each user message starts with a [ctx] line: the time, the active search location, and "
    "gps or manual. Never call a tool to learn them.\n"
    "send_message_to_friends: never call it in the turn the user asks. First say who gets "
    "it and the exact text and ask \"Göndereyim mi?\"; call it only after the user confirms."
)
#: The served prompt.  It adds the app's persona (Mustachio) and the step the
#: earlier prompts left out: the tool is run, its result comes back, and the
#: assistant says a sentence about it - the place a small model invents most.
SYSTEM = (
    "You are Mustachio, the assistant inside the feast app: the mustachioed friend who knows "
    "every street in the neighbourhood and is always a little hungry. You are warm, quick and "
    "a bit playful, and you talk like a good esnaf: short and direct, no filler like "
    "\"Elbette!\" or \"Tabii ki!\". You help with nearby restaurants, cafés and dishes, and "
    "with the user's saved items, posts, friends and chats. You run on this phone; nothing "
    "the user writes leaves it.\n"
    "Call a tool when one fits; otherwise answer briefly. Never invent venues, dishes, "
    "prices, people or messages. Reply in the user's language; Turkish with English words "
    "mixed in is still Turkish.\n"
    "Each user message starts with a [ctx] line: the time, the active search location, and "
    "gps or manual. Never call a tool to learn them.\n"
    "After a tool result: the app shows the results as cards, so do not list them. Say one "
    "or two sentences, naming at most three places, and state nothing that is not in the "
    "result. If count is 0, say so and suggest one way to widen the search. If ok is false, "
    "say once that you could not reach the data.\n"
    "send_message_to_friends: never call it in the turn the user asks. First say who gets "
    "it and the exact text and ask \"Göndereyim mi?\"; call it only after the user confirms."
)

#: A few words per tool - enough for a model that has been trained on them.
TERSE = {
    "search_restaurants": "find venues by kind or cuisine",
    "search_menu_items": "find a dish or drink and its price",
    "get_restaurant_details": "hours, address or menu of a listed venue",
    "get_menu_item_details": "ingredients or price of a listed dish",
    "get_directions": "route to a listed venue",
    "set_active_location": "move the search area, or back to GPS; give exactly one argument",
    "search_posts": "posts about a dish, venue or keyword",
    "ask_about_posts": "places by a quality people describe: atmosphere, who it suits",
    "search_saved": "the user's saved venues, dishes or posts",
    "search_shared_items": "items sent in chats, to or by the user",
    "search_chat_memory": "the user's earlier talks with you",
    "list_friends": "the user's friends",
    "get_suggested_users": "people the user could add",
    "open_screen": "open an app screen",
    "send_message_to_friends": "message friends, only after the user confirms",
}
_FULL_EXTRA = {
    "set_active_location": " Give exactly one of location_query / use_current_location.",
}
#: Left out of every declaration: a default-only knob no sentence sets.
HIDDEN = ("limit",)
#: Also left out of the terse declarations: the training labels never use
#: them (a venue kind is a ``query``), and ``tags`` alone lists 16 values.
HIDDEN_TERSE = HIDDEN + ("tags", "min_price")
STYLES = ("terse", "full")

#: English words for generic food and venue kinds -> the Turkish query. The
#: menus being searched are Turkish, so this is the one translation the model
#: is trained to make and the one ungrounded value the guard accepts.
EN_TR = {
    "coffee": "kahve", "coffee shop": "kahve", "tea": "çay", "breakfast": "kahvaltı",
    "dessert": "tatlı", "fish": "balık", "fish restaurant": "balık", "chicken": "tavuk",
    "grilled chicken": "tavuk", "meat": "et", "soup": "çorba", "meatballs": "köfte",
    "meatball place": "köfte", "kebab place": "kebap", "kebab": "kebap",
    "bakery": "fırın", "ice cream": "dondurma", "ice cream shop": "dondurma",
    "sea bass": "levrek", "steak": "biftek", "tacos": "taco", "pasta": "makarna",
    "omelette": "omlet", "pancakes": "pankek", "lemonade": "limonata",
    "italian restaurant": "İtalyan", "turkish coffee": "türk kahvesi",
}


def ctx_line(context: AgentContext) -> str:
    """``[ctx] 2026-11-10 Tue 19:30 | Moda, Kadıköy, İstanbul | gps``."""
    return (f"[ctx] {context.clock()} | {context.label} | "
            f"{'gps' if context.is_gps else 'manual'}")


def user_content(text: str, context: AgentContext) -> str:
    """A user turn as the model was trained to read it."""
    return f"{ctx_line(context)}\n{text}"


def strip_ctx(content: str) -> str:
    """The user's own words back out of a ``[ctx]``-prefixed turn."""
    if content.startswith("[ctx] "):
        return content.split("\n", 1)[1] if "\n" in content else ""
    return content


def _property(tool: ToolSpec, param: Param, tags: tuple[str, ...], style: str) -> dict:
    hint = HINTS.get(tool.name, {}).get(param.name, "") if style == "full" else ""
    if style == "full" and param.note and param.note not in hint:
        hint = f"{hint} ({param.note})" if hint else param.note
    if param.type == "hhmm":
        hint = (hint + ", " if hint else "") + ("as HH:MM" if style == "full" else "HH:MM")
    if param.type == "str[]":
        items: dict = {"type": "string"}
        if param.choices_from == "tags":
            items["enum"] = list(tags)
        # The on-device template writes ",items:{" unconditionally, so an
        # array with no description renders as "{,items:{": always give one.
        return {"type": "array", "items": items, "description": hint or "names"}
    out: dict = {"type": _JSON_TYPE[param.type]}
    if param.enum:
        out["enum"] = list(param.enum)
    if hint:
        out["description"] = hint
    return out


def declaration(tool: ToolSpec, tags: tuple[str, ...] = DEFAULT_TAGS,
                style: str = "terse") -> dict:
    hidden = HIDDEN_TERSE if style == "terse" else HIDDEN
    params = [p for p in tool.params if p.name not in hidden]
    description = (TERSE[tool.name] if style == "terse"
                   else INDEX2[tool.name] + _FULL_EXTRA.get(tool.name, ""))
    # A tool without arguments still declares an empty object, so every Jinja
    # implementation takes the same branch of the template.
    return {"type": "function", "function": {
        "name": tool.name, "description": description,
        "parameters": {
            "type": "object",
            "properties": {p.name: _property(tool, p, tags, style) for p in params},
            "required": [p.name for p in params if p.required],
        }}}


def declarations(tags: tuple[str, ...] = DEFAULT_TAGS, style: str = "terse") -> list[dict]:
    """Every tool, in registry order - a stable order keeps the prompt cacheable."""
    return [declaration(tool, tags, style) for tool in TOOLS]


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------
ID_ARGS = {"restaurant_id": "r", "attach_restaurant_id": "r", "menu_item_id": "m"}
#: String arguments that must come from the user's own words.
GROUNDED = {"query", "location_query", "friend_name", "friend_names"}
#: Below this similarity a value is not a misspelling of what the user wrote.
REPAIR_AT = 0.72
_ID = re.compile(r"\b([rm]_\d+)\b")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_MARKUP = re.compile(r"<\|?[A-Za-z_\"/|]{1,24}\|?>")
_CLOCK = re.compile(r"^\s*(\d{1,2})(?:[:.](\d{1,2}))?\s*$")
_PROPOSAL = ("göndereyim mi", "gondereyim mi", "shall i send", "should i send")
REPHRASE = {"tr": "Tam anlayamadım, biraz daha açık yazar mısın?",
            "en": "I didn't quite get that - could you rephrase?"}


@dataclass
class Guarded:
    """A model answer after the app's checks."""

    check: CallCheck
    #: What the user is shown when the call was turned into text (a proposal
    #: or a request to rephrase); empty when the call stands or the model
    #: answered in text itself.
    text: str = ""
    notes: list[str] = field(default_factory=list)
    #: String arguments nothing in the conversation supports.
    ungrounded: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.notes)


#: How much conversation a native request carries, in characters.  The
#: static prompt is ~1,450 tokens of the 4,096-token context; measured on a
#: 14-turn chat, a turn with its tool result is ~130 tokens (~350 characters),
#: and a five-item result alone can be 300 tokens.
WINDOW_CHARS = 4800


def window(history: list, chars: int = WINDOW_CHARS) -> list:
    """The part of ``history`` (objects with ``.role``/``.content``) a request carries.

    An engine that keeps the conversation (LiteRT-LM here, llama.cpp's
    prompt cache) re-reads all of it when its start changes - measured on
    the mustachio bundle at 4 threads: 17 s for 2,650 tokens.  A window that
    drops one turn per message pays that on every message.  So the start
    moves only when the conversation no longer fits, and then far: turns
    are dropped until half the budget is free.  The start is computed by
    replaying that rule over the whole history, so it is the same on every
    call until the next move.
    """
    starts = [i for i, m in enumerate(history) if m.role == "user"]
    if not starts:
        return []
    ends = starts[1:] + [len(history)]
    sizes = [sum(len(m.content) for m in history[a:b]) for a, b in zip(starts, ends)]
    first, held = 0, 0
    for turn, size in enumerate(sizes):
        held += size
        if held > chars:
            while first < turn and held > chars // 2:
                held -= sizes[first]
                first += 1
    return history[starts[first]:]


def tool_message(name: str, result: dict) -> dict:
    """A tool result as it travels in a request: OpenAI's ``tool`` role."""
    import json

    return {"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False)}


def cards(result: dict, limit: int = 5) -> list[str]:
    """A result as the lines a transcript can show - the app's cards, in text."""
    lines = []
    for item in (result.get("items") or [])[:limit]:
        parts = [str(item.get("name") or item.get("caption") or item.get("text") or "")]
        if "price" in item:
            parts.append(f"{item['price']} TL")
        if "walk_min" in item:
            parts.append(f"{item['walk_min']} dk")
        if item.get("open") is False:
            parts.append("kapalı")
        for key in ("restaurant", "hours", "address", "why", "user", "friend", "when"):
            if item.get(key):
                parts.append(str(item[key]))
        if "mutual" in item:
            parts.append(f"{item['mutual']} ortak")
        if item.get("id"):
            parts.append(f"[{item['id']}]")
        lines.append("▸ " + " · ".join(p for p in parts if p))
    more = int(result.get("count") or 0) - len(lines)
    if more > 0:
        lines.append(f"  (+{more})")
    if "items" not in result:
        facts = ", ".join(f"{k}={v}" for k, v in result.items() if k != "ok")
        lines.append(("✓ " if result.get("ok") else "✗ ") + facts)
    elif not result.get("ok", True):
        lines.append("✗ data unavailable")
    elif not lines:
        lines.append("∅ no results")
    return lines


def shown_ids(assistant_texts: list[str]) -> dict[str, tuple[str, ...]]:
    """Ids the app has listed, by kind. Never read from a call: an id the
    model once invented must not become known.  A tool result is the app's
    own data, so its text may be passed here like a result list."""
    out: dict[str, list[str]] = {"r": [], "m": []}
    for text in assistant_texts:
        if "<tool_call>" in text or "<|tool_call>" in text:
            continue
        for found in _ID.findall(text):
            if found not in out[found[0]]:
                out[found[0]].append(found)
    return {kind: tuple(ids) for kind, ids in out.items()}


def is_proposal(text: str) -> bool:
    """Did the assistant just ask whether to send a message?"""
    return any(marker in fold(text) for marker in (fold(m) for m in _PROPOSAL))


def proposal_text(args: dict, english: bool = False) -> str:
    """The question a send must be preceded by, written from its own arguments."""
    names = [str(n) for n in (args.get("friend_names") or [])] or ["?"]
    message = str(args.get("message") or "")
    if english:
        return f'Send "{message}" to {" and ".join(names)}?'
    return f'{_dative_list(names)} "{message}" göndereyim mi?'


def _dative_list(names: list[str]) -> str:
    """"Onur ve Ece'ye" - Turkish puts the case on the last name."""
    last = names[-1]
    lowered = last.replace("İ", "i").replace("I", "ı").lower()
    vowels = [c for c in lowered if c in "aeıioöuü"]
    vowel = "a" if vowels and vowels[-1] in "aıou" else "e"
    buffer = "y" if lowered[-1:] in tuple("aeıioöuü") else ""
    return " ve ".join(names[:-1] + [f"{last}'{buffer}{vowel}"])


def _says(haystack: str, needle: str) -> bool:
    """Do the words of ``needle`` appear in ``haystack``, in order and together?

    Word-level and folded, so a full stop or a capital the model added to a
    message does not count as a changed wording.
    """
    words, wanted = _WORD.findall(fold(haystack)), _WORD.findall(fold(needle))
    if not wanted:
        return False
    return any(words[i:i + len(wanted)] == wanted
               for i in range(len(words) - len(wanted) + 1))


def normalise_clock(value: object) -> str | None:
    """``"24:00"`` -> ``"00:00"``, ``"9"`` -> ``"09:00"``; ``None`` if not a time."""
    match = _CLOCK.match(str(value))
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2) or 0)
    if hour == 24 and minute == 0:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def _ngrams(text: str, size: int) -> list[tuple[str, str]]:
    """``(folded, as typed)`` word n-grams of ``text``."""
    words = _WORD.findall(text)
    return [(fold(" ".join(words[i:i + size])), " ".join(words[i:i + size]))
            for i in range(0, max(0, len(words) - size + 1))]


def ground(value: str, sources: list[str]) -> tuple[str, str]:
    """Is ``value`` something the user wrote?

    Returns ``("ok", value)``, ``("repaired", the user's wording)`` or
    ``("ungrounded", value)``.  A value matches a span it equals or begins
    (so "Kadıköy" is grounded by "kadıköye" and "kokoreç" by "kokorec"), and
    the one trained translation in :data:`EN_TR`.
    """
    wanted = fold(value).strip()
    if not wanted:
        return "ok", value
    size = max(1, len(wanted.split()))
    best, best_span = 0.0, ""
    for source in sources:
        folded_source = fold(source)
        if wanted in folded_source:
            return "ok", value
        if any(fold(tr) == wanted and en in folded_source for en, tr in EN_TR.items()):
            return "ok", value
        for folded, typed in _ngrams(source, size):
            # A suffix on the user's word is still that word: "pideciyi".
            if folded.startswith(wanted) or wanted.startswith(folded) and len(folded) >= 4:
                return "ok", value
            score = difflib.SequenceMatcher(None, wanted, folded).ratio()
            if score > best:
                best, best_span = score, typed
    if best >= REPAIR_AT and best_span:
        return "repaired", best_span
    return "ungrounded", value


def _schema(name: str, args: dict, notes: list[str]) -> dict:
    """Coerce what an app could coerce; drop what the tool does not have."""
    spec = BY_NAME[name]
    out: dict = {}
    for key, value in args.items():
        param = spec.param(key)
        if param is None:
            notes.append(f"dropped unknown argument {key}")
            continue
        if value is None:
            continue
        if param.enum:
            if value not in param.enum:
                match = [e for e in param.enum if fold(str(value)) == fold(e)]
                if not match:
                    notes.append(f"dropped {key}={value!r}: not one of {list(param.enum)}")
                    continue
                value = match[0]
        elif param.type == "hhmm":
            clock = normalise_clock(value)
            if clock is None:
                notes.append(f"dropped {key}={value!r}: not a time")
                continue
            if clock != value:
                notes.append(f"{key} {value!r} -> {clock!r}")
            value = clock
        elif param.type == "bool" and not isinstance(value, bool):
            if str(value).lower() not in ("true", "false"):
                notes.append(f"dropped {key}={value!r}: not a bool")
                continue
            value = str(value).lower() == "true"
        elif param.type in ("int", "num") and not isinstance(value, (int, float)):
            try:
                number = float(str(value).replace(",", "."))
            except ValueError:
                notes.append(f"dropped {key}={value!r}: not a number")
                continue
            value = int(number) if param.type == "int" or number.is_integer() else number
        elif param.type == "str[]" and not isinstance(value, list):
            value = [value]
        out[key] = value
    for group in spec.exactly_one:
        given = [n for n in group if out.get(n) not in (None, "", False)]
        if len(given) > 1:  # "Karaköy" and use_current_location: the place wins
            for extra in given[1:]:
                out.pop(extra, None)
                notes.append(f"dropped {extra}: only one of {list(group)} may be given")
    return out


def guard(check: CallCheck, user_texts: list[str], assistant_texts: list[str],
          reply: str = "", results: list[str] | None = None,
          context_line: str = "") -> Guarded:
    """Apply the app-side checks to one model answer.

    ``user_texts`` are the user's messages so far, the current one last and
    without their ``[ctx]`` lines; ``assistant_texts`` are the assistant's
    earlier turns (result lists, proposals, calls).  ``reply`` is the raw
    text of a non-call answer, checked only for being a sentence at all.
    ``results`` are the tool results of this chat (JSON text): the other
    place an id can have been listed.  ``context_line`` is the ``[ctx]``
    line, which a text answer may quote (the time, the place).

    A text answer is not rewritten, but it is read: a number or a name that
    nothing in the chat contains is listed in ``ungrounded`` - an answer
    from memory where a tool was due.
    """
    english = bool(user_texts) and detect_lang(user_texts[-1]) == "en"
    if not check.is_call:
        # Format tokens are not words: "}<tool_call|>" is the tail of a call
        # that never started, and must not reach the user as an answer.
        if reply and len(_WORD.findall(_MARKUP.sub(" ", reply))) < 1:
            return Guarded(check, REPHRASE["en" if english else "tr"],
                           ["fallback: the answer was not a sentence"])
        from .replies import unsupported

        invented = unsupported(reply, user_texts + assistant_texts + list(results or [])
                               + [context_line]) if reply else []
        return Guarded(check, ungrounded=[f"text states {m!r}" for m in invented[:4]])
    if check.status in ("parse_error", "unknown_tool") or check.name not in BY_NAME:
        return Guarded(CallCheck(), REPHRASE["en" if english else "tr"],
                       [f"fallback: {check.status} {check.name}".strip()])

    notes: list[str] = []
    ungrounded: list[str] = []
    args = _schema(check.name, dict(check.args), notes)

    # --- ids must be ones the app listed --------------------------------
    shown = shown_ids(assistant_texts + list(results or []))
    for key, kind in ID_ARGS.items():
        if key in args and str(args[key]) not in shown[kind]:
            notes.append(f"dropped {key}={args[key]!r}: not an id listed in this chat")
            ungrounded.append(key)
            args.pop(key)

    # --- searched words, places and people must be the user's -----------
    for key in [k for k in args if k in GROUNDED]:
        values = args[key] if isinstance(args[key], list) else [args[key]]
        fixed = []
        for value in values:
            # A friend's name may also come from the proposal being confirmed.
            sources = user_texts + (assistant_texts[-1:] if key.startswith("friend") else [])
            status, replacement = ground(str(value), sources)
            if status == "repaired":
                notes.append(f"{key} {value!r} -> {replacement!r} (the user's wording)")
            elif status == "ungrounded":
                ungrounded.append(f"{key}={value!r}")
            fixed.append(replacement)
        args[key] = fixed if isinstance(args[key], list) else fixed[0]

    # --- a message is sent only as the answer to its own proposal --------
    if check.name == SEND:
        previous = assistant_texts[-1] if assistant_texts else ""
        names = [str(n) for n in args.get("friend_names") or []]
        message = str(args.get("message") or "")
        why = ""
        if not is_proposal(previous):
            why = "no proposal was confirmed"
        elif not _says(previous, message):
            why = "the wording differs from the proposal"
        elif not names or any(not _says(previous, n) for n in names):
            why = "the recipients differ from the proposal"
        if why:
            return Guarded(CallCheck(), proposal_text(args, english),
                           notes + [f"send blocked ({why}); asked to confirm instead"],
                           ungrounded)

    result = validate(check.name, args)
    out = CallCheck(status="invalid_args" if result.issues else "ok", name=check.name,
                    args=args, format=check.format, issues=list(result.issues),
                    warnings=list(result.warnings))
    return Guarded(out, "", notes, ungrounded)
