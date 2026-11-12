"""Training conversations whose labels are correct by construction.

A teacher model that *labels* sentences makes labelling mistakes, and a 2B
student copies them.  Here the order is reversed: a scenario is drawn first
(tool, arguments, slot values), and the sentence is written *from* it, so the
gold call is known before the user "speaks".  What varies is everything a
real user varies - and the label stays put:

``clean``    proper Turkish, diacritics, punctuation;
``casual``   how people text: lower case, "bi", "nerde", "gidcem", no
             apostrophes, "kanka"/"abi"/"ya", often no diacritics at all;
``hybrid``   a Turkish sentence carrying English words - an English stem
             under a Turkish suffix ("save ettiğim", "settings'i aç"), or
             English discourse words around a Turkish clause ("lets go",
             "pls", "asap");
``en``       plain English, a small share.

One scenario rendered in several registers with the same gold call is the
whole method for "do not get confused by hybrid usage": the model is shown
that the surface changed and the action did not.

Three label rules the templates all follow (they are choices; the app's
backend has to agree with them):

* slot values are canonical - ``location_query`` is "Karaköy" whether the
  user typed "karakoye" or "Karaköy'e", a dish typed "kokorec" is searched
  as "kokoreç";
* a message to a friend is the user's own wording, exactly as typed - the
  assistant never rewrites what someone says to a friend;
* a generic English food word becomes its Turkish query ("coffee shop" ->
  "kahve"), because the menus it is matched against are Turkish.

Conversations come in two kinds, because the model is served in two ways:

``bare``      as feast AI's suites and benchmark run it - a call ends the
              turn, nothing executes it, and a past result list is assistant
              text with ids, written by the app.  Such a list is never a
              training target (the model must not learn to invent venues);
``executed``  as the app runs it - every call is followed by its tool result
              and one sentence about it (:data:`EXECUTED_SHARE` of the
              conversations).  The result is invented here; the sentence is
              written *from* it by :func:`feast_ai.agent.replies.say`, so the
              label of an "answer the result" turn states nothing the result
              does not.  That is the same rule as for calls - label first,
              text second - applied to the place a small model invents most.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from feast_ai.agent import mock, replies
from feast_ai.agent.text import tr_lower

from . import lexicon as L

_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜâîû", "cgiosuCGIOSUaiu")
_SLOT = re.compile(r"\{(\w+)(?::(\w+))?\}")
SEND = "send_message_to_friends"

REGISTERS = ("clean", "casual", "hybrid", "en")
REGISTER_WEIGHTS = (0.26, 0.31, 0.36, 0.07)

_CONTRACTIONS = (
    (r"\bbir şey\b", "bişey"), (r"\bbir\b", "bi"), (r"\bnerede\b", "nerde"),
    (r"\bburada\b", "burda"), (r"\borada\b", "orda"), (r"\bşurada\b", "şurda"),
    (r"\bdeğil mi\b", "dimi"), (r"(\w+)yorum\b", r"\1yom"),
    (r"(\w+)yor musun\b", r"\1yo musun"), (r"(\w+)acağım\b", r"\1cam"),
    (r"(\w+)eceğim\b", r"\1cem"),
)
_VOC_BEFORE = {
    "casual": ("kanka", "abi", "hocam", "kanki", "knk", "reis", "dostum", "aga", "ya"),
    "hybrid": ("kanka", "bro", "abi", "kanki", "dude", "hey", "ok", "btw"),
}
_VOC_AFTER = {
    "casual": ("ya", "yaa", "kanka", "abi", "be", "hocam", "lütfen"),
    "hybrid": ("ya", "bro", "pls", "kanka", "please", "asap", "thx"),
}
_EMOJI = ("😅", "🙏", "😋", "🔥", "👀", "😂", "🤤", "👍", "🙌")


def cap(text: str) -> str:
    """Capitalise the Turkish way: "ilkinin" -> "İlkinin", never "Ilkinin"."""
    if not text:
        return text
    return {"i": "İ", "ı": "I"}.get(text[0], text[0].upper()) + text[1:]


class Style:
    """One user's way of typing, fixed for a whole conversation.

    ``text`` is deterministic, so a slot that is quoted back (a message to a
    friend) can be styled apart from its sentence and still match it.
    """

    def __init__(self, rng: random.Random, register: str) -> None:
        self.rng, self.register = rng, register
        p = {"clean": (0.12, 0.15, 0.05, 0.0, 0.04, 0.02, 0.25),
             "casual": (0.42, 0.88, 0.55, 0.75, 0.45, 0.14, 0.75),
             "hybrid": (0.36, 0.84, 0.45, 0.50, 0.40, 0.12, 0.70),
             "en": (0.0, 0.30, 0.0, 0.0, 0.0, 0.03, 0.40)}[register]
        self.ascii = rng.random() < p[0]
        self.lower = rng.random() < p[1]
        self.no_apostrophe = rng.random() < p[2]
        self.contract = rng.random() < p[3]
        self.vocative = rng.random() < p[4]
        self.emoji = rng.random() < p[5]
        self.no_final = rng.random() < p[6]
        #: Is this a conversation in which the app runs the tools? (see :func:`one`)
        self.executed = False

    def text(self, value: str) -> str:
        if self.contract:
            lowered = tr_lower(value) if self.lower else value
            for pattern, repl in _CONTRACTIONS:
                lowered = re.sub(pattern, repl, lowered)
            value = lowered
        if self.lower:
            # English keeps its dotted "i": tr_lower would turn "I" into "ı".
            value = (value.replace("İ", "i").lower() if self.register == "en"
                     else tr_lower(value))
        if self.no_apostrophe:
            value = value.replace("'", "")
        if self.ascii:
            value = value.translate(_ASCII)
        return value

    def wrap(self, sentence: str) -> str:
        """Style a whole user message: address words, emoji, final punctuation."""
        out = self.text(sentence).strip()
        if not self.lower:
            out = out[0].upper() + out[1:] if self.register == "en" else cap(out)
        if self.no_final:
            out = out.rstrip(".?!")
        if self.vocative and self.register in _VOC_BEFORE:
            if self.rng.random() < 0.55:
                word = self.text(self.rng.choice(_VOC_BEFORE[self.register]))
                out = f"{word} {out[0].lower() + out[1:] if self.lower else out}"
            else:
                out = f"{out.rstrip('.?!')} {self.text(self.rng.choice(_VOC_AFTER[self.register]))}"
        if self.emoji:
            out = f"{out} {self.rng.choice(_EMOJI)}"
        return out


def fill(template: str, **slots: object) -> str:
    """``{name}`` and ``{name:dat}`` (dat/loc/abl/gen: the Turkish case suffix)."""
    def sub(match: re.Match) -> str:
        value = str(slots[match.group(1)])
        return L.CASES[match.group(2)](value) if match.group(2) else value
    return _SLOT.sub(sub, template)


@dataclass
class Example:
    family: str
    register: str
    messages: list[dict]
    ctx: dict = field(default_factory=dict)
    id: str = ""


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _call(name: str, args: dict, supervise: bool = True, focus: str = "",
          result: dict | None = None) -> dict:
    """A call.  ``focus`` (what the user asked about) and ``result`` (a result
    the scenario has already decided) are notes for :func:`execute`; they
    never reach the dataset."""
    out = {"role": "assistant", "supervise": supervise,
           "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}
    if focus:
        out["focus"] = focus
    if result is not None:
        out["result"] = result
    return out


def _say(text: str, supervise: bool = True) -> dict:
    return {"role": "assistant", "content": text, "supervise": supervise}


def _pick(rng: random.Random, table: dict, register: str) -> str:
    """A template for ``register``; a register the table lacks falls back to clean."""
    return rng.choice(table.get(register) or table["clean"])


# --------------------------------------------------------------------------
# Venues  ->  search_restaurants
# --------------------------------------------------------------------------
VENUE = {
    "base": {
        "clean": ["Yakınlarda iyi bir {kind} var mı?", "Buralarda bir {kind} önerir misin?",
                  "Bana yakın bir {kind} bulur musun?", "Civarda {kind} arıyorum.",
                  "Etrafta güzel bir {kind} biliyor musun?",
                  "Bu yakınlarda gidebileceğim bir {kind} var mı?"],
        "casual": ["yakında iyi bir {kind} var mı", "buralarda {kind} var mı",
                   "bir {kind} bulsana", "{kind} lazım acil",
                   "şuralarda güzel bir {kind} bilen var mı", "civarda {kind} nerede var"],
        "hybrid": ["nearby iyi bi {kind} var mı", "buralarda good bi {kind} var mı",
                   "bana bi {kind} find etsene", "{kind} recommend eder misin",
                   "around here {kind} var mı", "iyi bi {kind} spot'u lazım",
                   "{kind} arıyorum, any idea?"],
        "en": ["Find me a {kind} nearby", "Is there a good {kind} around here?",
               "Any {kind} near me?", "I'm looking for a {kind} close by"],
    },
    "open_now": {
        "clean": ["Şu an açık bir {kind} var mı?", "Yakında şimdi açık olan {kind} hangisi?",
                  "Bu saatte açık bir {kind} bulur musun?"],
        "casual": ["şu an açık {kind} var mı", "açık bir {kind} lazım",
                   "bu saatte açık {kind} var mı"],
        "hybrid": ["şu an open olan {kind} var mı", "right now açık bi {kind} bul",
                   "open bi {kind} lazım asap", "{kind} var mı şu an open olan"],
        "en": ["Is there a {kind} open right now?", "Any {kind} open now near me?"],
    },
    "open_at": {
        "clean": ["{timeopen} olan bir {kind} var mı?", "{timeopen} bir {kind} arıyorum.",
                  "Bir {kind} arıyorum, {timeopen} olsun."],
        "casual": ["{timeopen} {kind} var mı", "{timeopen} olan bir {kind} lazım",
                   "{kind} bakıyorum, {timeopen} olsun"],
        "hybrid": ["{timeopen} olan {kind} var mı", "{timeopen} bi {kind} spot'u var mı",
                   "{kind} lazım, {timeopen} olanlardan"],
    },
    "open_at_any": {
        "clean": ["{timeopen} bir yer var mı?", "{timeopen} olan mekanları göster."],
        "casual": ["{timeopen} neresi var", "{timeopen} olan bir yer lazım"],
        "hybrid": ["{timeopen} olan yerler hangileri", "{timeopen} bi place var mı"],
    },
    "walk": {
        "clean": ["Yürüyerek {n} dakika içinde bir {kind} var mı?",
                  "{n} dakikalık yürüme mesafesinde bir {kind} bul.",
                  "En fazla {n} dakika yürüyerek gidebileceğim bir {kind} öner."],
        "casual": ["yürüyerek {n} dkda gidilecek {kind} var mı",
                   "{n} dk yürüme mesafesinde {kind} var mı", "max {n} dakika yürürüm, {kind} bul"],
        "hybrid": ["walking distance {n} dk içinde {kind} var mı",
                   "{n} dk walk'la gidilecek bi {kind} bul", "max {n} dk yürürüm, nearby {kind}?"],
        "en": ["A {kind} within {n} minutes' walk", "Any {kind} I can walk to in {n} minutes?"],
    },
    "distance": {
        "clean": ["{dist} içinde bir {kind} var mı?", "En fazla {dist} uzaklıkta bir {kind} arıyorum."],
        "casual": ["{dist} içinde {kind} var mı", "{dist} uzaklıktan fazla olmasın, {kind} bul"],
        "hybrid": ["{dist} radius'ta {kind} var mı", "max {dist} uzaklıkta bi {kind} lazım"],
    },
    "popular": {
        "clean": ["Buraların en popüler {kind} hangisi?", "En çok tercih edilen {kind} neresi?",
                  "Bu civarın en bilinen {kind} hangisi?"],
        "casual": ["burada en popüler {kind} hangisi", "en meşhur {kind} nerede",
                   "herkesin gittiği {kind} hangisi"],
        "hybrid": ["en trend {kind} hangisi burda", "top rated {kind} hangisi",
                   "en hype {kind} nerde", "most popular {kind} hangisi bu civarda"],
        "en": ["What's the most popular {kind} around here?"],
    },
    "nearest": {
        "clean": ["En yakın {kind} nerede?", "Bana en yakın {kind} hangisi?"],
        "casual": ["en yakın {kind} nerede", "en yakın {kind} hangisi"],
        "hybrid": ["nearest {kind} nerde", "closest {kind} hangisi", "en yakın {kind} nerde, quick"],
        "en": ["Where is the nearest {kind}?"],
    },
    "walk+open": {
        "clean": ["Şu an açık ve yürüyerek {n} dakika içinde bir {kind} var mı?"],
        "casual": ["şu an açık, {n} dk yürümeyle gidilecek {kind} var mı"],
        "hybrid": ["open olan ve {n} dk walk mesafesinde bi {kind} var mı"],
    },
    "nearest+open": {
        "clean": ["En yakın açık {kind} nerede?"],
        "casual": ["en yakın açık {kind} nerede"],
        "hybrid": ["nearest open {kind} nerde"],
    },
}
VENUE_GENERIC = (
    ("clean", "Yakında yemek yiyebileceğim bir yer öner.", {}),
    ("clean", "Bu civarda nerede yemek yiyebilirim?", {}),
    ("casual", "acıktım, yakında nerede yenir", {}),
    ("hybrid", "açım, nearby yemek yenecek bi place öner", {}),
    ("clean", "Şu an açık bir yer var mı?", {"open_now": True}),
    ("casual", "şu an açık neresi var", {"open_now": True}),
    ("hybrid", "şu an open olan yerleri göster", {"open_now": True}),
    ("clean", "Buraların en popüler mekanları hangileri?", {"sort_by": "popularity"}),
    ("hybrid", "en popular mekanlar hangileri burda", {"sort_by": "popularity"}),
    ("en", "Where can I eat around here?", {}),
    ("en", "What's open right now nearby?", {"open_now": True}),
)
_VENUE_MODS = ("base", "base", "base", "open_now", "open_now", "open_at", "open_at",
               "open_at_any", "walk", "walk", "distance", "popular", "popular", "nearest",
               "walk+open", "nearest+open")


def _venue_request(rng: random.Random, register: str) -> tuple[str, dict]:
    if rng.random() < 0.08:
        rows = [r for r in VENUE_GENERIC if r[0] == register] or \
               [r for r in VENUE_GENERIC if r[0] == "clean"]
        _reg, text, args = rng.choice(rows)
        return text, dict(args)
    mod = rng.choice(_VENUE_MODS)
    if register not in VENUE[mod]:
        mod = "base"
    kind, query = rng.choice(L.KINDS_EN if register == "en" else L.KINDS)
    args: dict = {"query": query}
    slots: dict = {"kind": kind}
    if "open" in mod and not mod.startswith("open_at"):
        args["open_now"] = True
    if mod.startswith("open_at"):
        phrase, args["open_at"] = L.clock_phrase(rng)
        word = "open" if register == "hybrid" and rng.random() < 0.7 else "açık"
        slots["timeopen"] = phrase.format(open=word)
        if mod == "open_at_any":
            del args["query"]
    if "walk" in mod:
        slots["n"] = args["max_walk_minutes"] = rng.choice(L.WALKS)
    if mod == "distance":
        slots["dist"], args["max_distance_meters"] = rng.choice(L.DISTANCES)
    if mod == "popular":
        args["sort_by"] = "popularity"
    if "nearest" in mod:
        args["sort_by"] = "distance"
    return fill(_pick(rng, VENUE[mod], register), **slots), args


def ramble(rng: random.Random, style: Style, text: str) -> str:
    """Sometimes people explain themselves first, or add a word after.

    The label does not change - which is the lesson: the request is in the
    sentence, not at the start of it.  v0 had no such sentences and answered
    one of the suite's ("...odaklanamıyorum, ... kafe biliyor musun?") with
    two brackets.
    """
    table = style.register if style.register in L.PREAMBLES else ""
    if not table:
        return text
    roll = rng.random()
    if roll < 0.15:
        before = rng.choice(L.PREAMBLES[table])
        joined = text if before.endswith(".") else text[0].lower() + text[1:]
        return f"{before} {joined}"
    if roll < 0.23:
        after = rng.choice(L.AFTERTHOUGHTS[table])
        return f"{text} {after}" if table == "clean" else f"{text.rstrip('.?!')}, {after}"
    return text


def venue(rng: random.Random, style: Style) -> list[dict]:
    text, args = _venue_request(rng, style.register)
    return [_user(style.wrap(ramble(rng, style, text))), _call("search_restaurants", args)]


# --------------------------------------------------------------------------
# Dishes  ->  search_menu_items
# --------------------------------------------------------------------------
DISH = {
    "base": {
        "clean": ["Yakındaki menülerde {dish} var mı?", "Buralarda nerede {dish} bulabilirim?",
                  "{dish} hangi mekanlarda var?", "Canım {dish} çekti, yakında nerede var?",
                  "{dish} bulabileceğim bir yer var mı?"],
        "casual": ["yakında {dish} var mı", "canım {dish} çekti nerede var",
                   "{dish} nerede bulurum buralarda", "şuralarda {dish} olan yer var mı"],
        "hybrid": ["nearby {dish} var mı", "{dish} craving'im var, nerde bulurum",
                   "{dish} olan bi place var mı around here", "{dish} lazım, nerde var nearby"],
        "en": ["Where can I get {dish} around here?", "Any place with {dish} nearby?"],
    },
    "price_q": {
        "clean": ["{dish} kaç lira buralarda?", "Yakında {dish} fiyatları nasıl?",
                  "Bu civarda {dish} ne kadar?"],
        "casual": ["{dish} kaç para buralarda", "{dish} kaça şu an", "{dish} fiyatı ne civarda"],
        "hybrid": ["{dish} price'ı ne buralarda", "{dish} kaç lira nearby",
                   "{dish} fiyatlarını check eder misin"],
        "en": ["How much is {dish} around here?", "What does {dish} cost nearby?"],
    },
    "max_price": {
        "clean": ["{price} liranın altında {dish} var mı?", "{price} TL'yi geçmeyen {dish} arıyorum.",
                  "En fazla {price} liraya {dish} nerede var?"],
        "casual": ["{price} lira altı {dish} var mı", "{price} tl üstü olmasın, {dish} bul",
                   "{price} liraya {dish} var mı"],
        "hybrid": ["under {price} TL {dish} var mı nearby", "{price} lira altı {dish} bul pls",
                   "{dish} lazım ama max {price} TL olsun"],
        "en": ["{dish} under {price} lira nearby", "Find {dish} for less than {price} TL"],
    },
    "cheapest": {
        "clean": ["En ucuz {dish} nerede?", "Buralarda en uygun fiyatlı {dish} hangisi?"],
        "casual": ["en ucuz {dish} nerede", "en uygun {dish} nerede var"],
        "hybrid": ["cheapest {dish} hangisi buralarda", "en budget friendly {dish} nerde",
                   "en ucuz {dish} nerde, low budget'ım"],
        "en": ["Where's the cheapest {dish}?"],
    },
    "priciest": {
        "clean": ["En pahalı {dish} hangisi?", "Buralarda en pahalı {dish} kaç lira?"],
        "casual": ["en pahalı {dish} hangisi", "en pahalı {dish} kaç para"],
        "hybrid": ["most expensive {dish} kaç lira burda", "en expensive {dish} hangisi"],
    },
    "open_now": {
        "clean": ["Şu an açık bir yerde {dish} var mı?",
                  "Bu saatte {dish} bulabileceğim açık bir yer var mı?"],
        "casual": ["bu saatte {dish} bulunur mu açık yerde", "şu an açık {dish} satan yer var mı"],
        "hybrid": ["şu an open olan yerde {dish} var mı", "bu saatte {dish} bulunur mu, open yer?"],
    },
    "walk": {
        "clean": ["Yürüyerek {n} dakika içinde {dish} var mı?"],
        "casual": ["{n} dk yürümeyle {dish} bulunur mu"],
        "hybrid": ["{n} dk walk mesafesinde {dish} var mı"],
    },
    "nearest": {
        "clean": ["En yakın {dish} nerede?"],
        "casual": ["en yakın {dish} nerede"],
        "hybrid": ["nearest {dish} nerde var"],
    },
}
_DISH_MODS = ("base", "base", "base", "price_q", "price_q", "max_price", "max_price",
              "max_price", "cheapest", "cheapest", "priciest", "open_now", "walk", "nearest")


def _dish_request(rng: random.Random, register: str) -> tuple[str, dict]:
    mod = rng.choice(_DISH_MODS)
    if register not in DISH[mod]:
        mod = "base"
    if register == "en":
        dish, query = rng.choice(L.DISHES_EN)
    else:
        dish = query = rng.choice(L.DISHES)
    args: dict = {"query": query}
    slots: dict = {"dish": dish}
    if mod == "max_price":
        slots["price"] = args["max_price"] = rng.choice(L.PRICES)
    if mod == "cheapest":
        args["sort_by"] = "price_asc"
    if mod == "priciest":
        args["sort_by"] = "price_desc"
    if mod == "open_now":
        args["open_now"] = True
    if mod == "walk":
        slots["n"] = args["max_walk_minutes"] = rng.choice(L.WALKS)
    if mod == "nearest":
        args["sort_by"] = "distance"
    return fill(_pick(rng, DISH[mod], register), **slots), args


def dish(rng: random.Random, style: Style) -> list[dict]:
    text, args = _dish_request(rng, style.register)
    return [_user(style.wrap(ramble(rng, style, text))), _call("search_menu_items", args)]


# --------------------------------------------------------------------------
# Vibe  ->  ask_about_posts
# --------------------------------------------------------------------------
VIBE = {
    "clean": ["{q} bir yer var mı?", "{q} bir mekan önerir misin?", "{q} bir yer arıyorum.",
              "Yakında {q} bir mekan biliyor musun?", "Bu akşam {q} bir yere gitmek istiyorum.",
              "Yemek için {q} bir mekan arıyorum."],
    "casual": ["{q} bir yer var mı", "{q} bir mekan biliyor musun", "{q} bir yer lazım",
               "akşam için {q} bir yer bakıyorum", "{q} bir yerde yemek yiyelim diyoruz"],
    "hybrid": ["{q} bi place var mı", "{q} bi mekan recommend et", "{q} bi spot arıyorum",
               "dinner için {q} bi place lazım", "tonight {q} bi yere gidelim diyoruz"],
}
VIBE_EN = (
    ("A place with a garden to sit outside", "Is there a place with a garden?"),
    ("Somewhere good for watching the game", "Is there a place to watch the game?"),
    ("A cozy place to read a book for hours", "Is there a cozy place to read a book?"),
    ("Somewhere family friendly for dinner", "Is there a family-friendly place?"),
    ("A place with a fireplace", "Is there a place with a fireplace?"),
)


def vibe(rng: random.Random, style: Style) -> list[dict]:
    if style.register == "en":
        text, question = rng.choice(VIBE_EN)
    else:
        row = rng.choice(L.QUALITIES)
        quality = row[REGISTERS.index(style.register)]
        text, question = fill(_pick(rng, VIBE, style.register), q=quality), row[3]
        text = ramble(rng, style, cap(text))
    return [_user(style.wrap(text)), _call("ask_about_posts", {"question": question})]


# --------------------------------------------------------------------------
# Result lists (history only - written by the app, never a training target)
# --------------------------------------------------------------------------
_RESERVED_IDS = {"r_481", "r_207", "r_355", "r_612", "r_138", "r_774", "m_9012", "m_3307",
                 "m_4410", "m_2086"}
_ORD_GEN = ("ilkinin", "ikincisinin", "üçüncüsünün", "dördüncüsünün")
_ORD_DAT = ("ilkine", "ikincisine", "üçüncüsüne", "dördüncüsüne")
_ORD_BARE = ("ilki", "ikincisi", "üçüncüsü", "dördüncüsü")
_ORD_ACC = ("ilkini", "ikincisini", "üçüncüsünü", "dördüncüsünü")
_COUNT = {2: "iki", 3: "üç", 4: "dört"}


def _new_id(rng: random.Random, prefix: str, used: set) -> str:
    while True:
        value = f"{prefix}_{rng.randint(100, 899) if prefix == 'r' else rng.randint(1000, 8999)}"
        if value not in used and value not in _RESERVED_IDS:
            used.add(value)
            return value


def _venue_name(rng: random.Random) -> str:
    return f"{rng.choice(L.VENUE_FIRST)} {rng.choice(L.VENUE_LAST)}"


def venue_results(rng: random.Random,
                  executed: bool = False) -> tuple[list[tuple[str, str]], list[dict]]:
    """``[(venue name, id)]`` and the history that lists them.

    Bare: two messages, the list as assistant text.  Executed: the request
    and its call, with the result attached for :func:`execute` to play out.
    """
    count, used, names = rng.choice((2, 3, 3, 3, 4)), set(), []
    while len(names) < count:
        name = _venue_name(rng)
        if name not in names:
            names.append(name)
    items = [(name, _new_id(rng, "r", used)) for name in names]
    kind, query = rng.choice(L.KINDS)
    minutes = sorted(rng.sample(range(2, 22), count))
    if executed:
        ask = rng.choice((f"{kind} bakıyorum", f"Çevrede {kind} arıyorum.",
                          f"bana {kind} listele", f"{kind} arıyorum"))
        listed = [mock.venue_item(rid, name, minutes[i], rng.random() < 0.85)
                  for i, (name, rid) in enumerate(items)]
        return items, [_user(ask), _call("search_restaurants", {"query": query},
                                         supervise=False, result=mock.result(listed))]
    head = rng.choice((f"Yakında {_COUNT[count]} yer buldum:", "Şunları buldum:",
                       f"Yakınında {_COUNT[count]} mekan var:"))
    lines = [f"{i + 1}. {name} (id: {rid}) - {minutes[i]} dk yürüme"
             for i, (name, rid) in enumerate(items)]
    # Not "Yakında X var mı?": that is how the suites seed their own histories.
    ask = rng.choice((f"{kind} bakıyorum", f"Çevrede {kind} arıyorum.", f"bana {kind} listele",
                      f"{kind} arıyorum"))
    return items, [_user(ask), _say(head + "\n" + "\n".join(lines), supervise=False)]


def dish_results(rng: random.Random,
                 executed: bool = False) -> tuple[list[tuple[str, str]], list[dict]]:
    count, used = rng.choice((2, 2, 3)), set()
    base = rng.choice(L.DISHES)
    extras = rng.sample(("Özel", "Ev Yapımı", "Acılı", "Büyük Boy", "Klasik", "Tereyağlı"), count)
    items = [(f"{extra} {base.title()}", _new_id(rng, "m", used)) for extra in extras]
    if executed:
        ask = rng.choice((f"{base} bakıyorum", f"Çevredeki {base} seçeneklerini göster.",
                          f"{base} nerede var, listele"))
        listed = sorted((mock.dish_item(mid, title, rng.choice(L.PRICES), _venue_name(rng))
                         for title, mid in items), key=lambda item: item["price"])
        # The list is shown cheapest first, so "the first one" follows that order.
        items = [(item["name"], item["id"]) for item in listed]
        return items, [_user(ask), _call("search_menu_items", {"query": base},
                                         supervise=False, result=mock.result(listed))]
    lines = [f"{i + 1}. {title} (id: {mid}) - {rng.choice(L.PRICES)} TL, {_venue_name(rng)}"
             for i, (title, mid) in enumerate(items)]
    ask = rng.choice((f"{base} bakıyorum", f"Çevredeki {base} seçeneklerini göster.",
                      f"{base} nerede var, listele"))
    return items, [_user(ask), _say("Yakındaki menülerde şunlar var:\n" + "\n".join(lines),
                                    supervise=False)]


def _reference(rng: random.Random, items: list[tuple[str, str]], form: str) -> tuple[str, str]:
    """How the user points at one listed item, and that item's id."""
    index = rng.randrange(len(items))
    name, item_id = items[index]
    roll = rng.random()
    if roll < 0.62:
        table = {"gen": _ORD_GEN, "dat": _ORD_DAT, "bare": _ORD_BARE, "acc": _ORD_ACC}[form]
        return table[index], item_id
    if roll < 0.74 and index == len(items) - 1:
        return {"gen": "sonuncunun", "dat": "sonuncuya", "bare": "sonuncu",
                "acc": "sonuncuyu"}[form], item_id
    if form == "bare":
        return name, item_id
    if form == "acc":
        return _ORD_ACC[index], item_id
    return L.CASES[form](name), item_id


DETAILS = {
    "hours": {
        "clean": ["{ref:gen} çalışma saatleri nedir?", "{ref:bare} kaça kadar açık?",
                  "{ref:bare} bugün açık mı?"],
        "casual": ["{ref:bare} kaça kadar açık", "{ref:gen} saatleri ne", "{ref:bare} yarın açık mı"],
        "hybrid": ["{ref:bare} kaça kadar open", "{ref:gen} opening hours'ına bak",
                   "{ref:bare} weekend'de açık mı"],
        "en": ["What are the opening hours of the {nth} one?", "Is the {nth} one open today?"],
    },
    "address": {
        "clean": ["{ref:gen} adresi ne?", "{ref:bare} tam olarak nerede?"],
        "casual": ["{ref:gen} adresi neydi", "{ref:bare} tam nerede kalıyor"],
        "hybrid": ["{ref:gen} address'i ne", "{ref:bare} tam nerde, exact yerini söyle"],
        "en": ["What's the address of the {nth} one?"],
    },
    "menu": {
        "clean": ["{ref:gen} menüsünü göster.", "{ref:gen} menüsünde neler var?"],
        "casual": ["{ref:gen} menüsünde ne var", "{ref:gen} menüsünü atsana"],
        "hybrid": ["{ref:gen} menu'sünü göster", "{ref:gen} menüsünü check edeyim"],
        "en": ["Show me the menu of the {nth} one"],
    },
}
DIRECTIONS = {
    "": {
        "clean": ["{ref:dat} nasıl giderim?", "{ref:dat} yol tarifi verir misin?"],
        "casual": ["{ref:dat} nasıl gidicem", "{ref:dat} tarif versene"],
        "hybrid": ["{ref:dat} directions ver", "{ref:dat} nasıl gidilir, route?"],
        "en": ["How do I get to the {nth} one?"],
    },
    "walk": {
        "clean": ["{ref:dat} yürüyerek nasıl gidilir?", "{ref:dat} yaya olarak yol tarifi ver."],
        "casual": ["{ref:dat} yürüyerek kaç dk", "{ref:dat} yürüyerek gidicem, tarif ver"],
        "hybrid": ["{ref:dat} walk'la nasıl gidilir", "{ref:dat} yürüyerek gidicem, route ver"],
        "en": ["Walking directions to the {nth} one"],
    },
    "drive": {
        "clean": ["{ref:dat} arabayla yol tarifi ver.", "{ref:dat} araçla nasıl giderim?"],
        "casual": ["{ref:dat} arabayla gidelim, tarif ver", "{ref:dat} arabayla kaç dk"],
        "hybrid": ["{ref:dat} drive edicem, route ver", "{ref:dat} car'la nasıl gidilir"],
        "en": ["Driving directions to the {nth} one"],
    },
    "transit": {
        "clean": ["{ref:dat} toplu taşımayla nasıl giderim?", "{ref:dat} metroyla gidilir mi?"],
        "casual": ["{ref:dat} otobüsle nasıl gidilir", "{ref:dat} metroyla gitsem nasıl olur"],
        "hybrid": ["{ref:dat} metro ya da bus'la nasıl gidilir", "{ref:dat} transit route ver"],
        "en": ["How do I get to the {nth} one by public transit?"],
    },
}
DISH_DETAILS = {
    "clean": ["{ref:gen} içinde ne var?", "{ref:gen} içeriği nedir?", "{ref:bare} kaç lira?",
              "{ref:bare} hangi mekanda?"],
    "casual": ["{ref:gen} içinde ne var", "{ref:bare} kaç paraydı", "{ref:gen} içinde soğan var mı"],
    "hybrid": ["{ref:gen} ingredients'ı ne", "{ref:gen} price'ı neydi", "{ref:bare} spicy mi"],
    "en": ["What's in the {nth} one?", "How much is the {nth} one?"],
}
_NTH = ("first", "second", "third", "fourth")


def _fill_ref(rng: random.Random, template: str, items: list[tuple[str, str]],
              register: str) -> tuple[str, str]:
    if register == "en":
        index = rng.randrange(len(items))
        return template.replace("{nth}", _NTH[index]), items[index][1]
    form = _SLOT.search(template).group(2)
    ref, item_id = _reference(rng, items, form)
    return _SLOT.sub(lambda _m: ref, template, count=1), item_id


_PRICE_WORDS = ("kaç lira", "kaç para", "price", "How much")


def details(rng: random.Random, style: Style) -> list[dict]:
    roll = rng.random()
    if roll < 0.34:
        items, history = dish_results(rng, style.executed)
        template = _pick(rng, DISH_DETAILS, style.register)
        text, item_id = _fill_ref(rng, template, items, style.register)
        focus = ("price" if any(w in template for w in _PRICE_WORDS)
                 else "" if "mekanda" in template else "ingredients")
        return history + [_user(style.wrap(text)),
                          _call("get_menu_item_details", {"menu_item_id": item_id}, focus=focus)]
    items, history = venue_results(rng, style.executed)
    if roll < 0.68:
        topic = rng.choice(tuple(DETAILS))
        text, item_id = _fill_ref(rng, _pick(rng, DETAILS[topic], style.register), items,
                                  style.register)
        return history + [_user(style.wrap(text)),
                          _call("get_restaurant_details", {"restaurant_id": item_id},
                                focus=topic)]
    mode = rng.choice(("", "walk", "drive", "transit"))
    text, item_id = _fill_ref(rng, _pick(rng, DIRECTIONS[mode], style.register), items,
                              style.register)
    args = {"restaurant_id": item_id}
    if mode:
        args["mode"] = mode
    return history + [_user(style.wrap(text)), _call("get_directions", args)]


# --------------------------------------------------------------------------
# The search area  ->  set_active_location
# --------------------------------------------------------------------------
LOCATION = {
    "clean": ["Konumu {loc} olarak değiştir.", "{loc:loc} arama yapmak istiyorum.",
              "{loc} civarına bakalım.", "Arama bölgesini {loc} yap.",
              "Yarın {loc:dat} gideceğim, oradaki yerlere bakalım.",
              "{loc} tarafındaki mekanları görmek istiyorum."],
    "casual": ["{loc:dat} geçiyorum, oraya göre bak", "konumu {loc} yap",
               "{loc} tarafına bakalım", "akşam {loc} tarafındayım, oraya göre ara",
               "hafta sonu {loc:dat} gidiyoruz, oralara bak"],
    "hybrid": ["konumu {loc} olarak update et", "aramayı {loc:dat} move edelim",
               "{loc} civarını explore edelim", "hadi {loc:dat}, let's go",
               "{loc:dat} gidiyoruz lets gooo", "area'yı {loc} yap",
               "this weekend {loc:dat} gidiyorum, oraya bak"],
    "en": ["Set my search location to {loc}", "Look around {loc} instead",
           "I'm heading to {loc}, search there"],
}
LOCATION_RESET = {
    "clean": ["Tekrar bulunduğum konuma göre ara.", "Aramayı şu anki yerime çevir.",
              "Telefonun konumunu kullan.", "Artık gerçek konumumdan devam edelim.",
              "GPS konumuma göre devam et.", "Konum servisindeki yerimi kullan."],
    "casual": ["olduğum yere göre bak", "şu an neredeysem oraya göre ara",
               "telefonun konumuna dön", "eve döndüm, buraya göre ara",
               "gps'teki yerime dön", "gps konumuma göre bak"],
    "hybrid": ["real location'ıma dön", "telefonun gps'ini use et", "live location'dan devam",
               "my location'a göre ara tekrar"],
    "en": ["Use my current location", "Go back to where I am"],
}


def location(rng: random.Random, style: Style) -> list[dict]:
    if rng.random() < 0.24:
        return [_user(style.wrap(_pick(rng, LOCATION_RESET, style.register))),
                _call("set_active_location", {"use_current_location": True})]
    place = rng.choice(L.PLACES)
    text = fill(_pick(rng, LOCATION, style.register), loc=place)
    return [_user(style.wrap(text)), _call("set_active_location", {"location_query": place})]


def location_then_search(rng: random.Random, style: Style) -> list[dict]:
    place = rng.choice(L.PLACES)
    first = fill(_pick(rng, LOCATION, style.register), loc=place)
    kind, query = rng.choice(L.KINDS_EN if style.register == "en" else L.KINDS)
    second = {
        "clean": ["Orada iyi bir {kind} öner.", "Şimdi oradaki {kind} seçeneklerine bakalım."],
        "casual": ["orada güzel bir {kind} var mı", "tamam, oralarda {kind} bul"],
        "hybrid": ["ok orada iyi bi {kind} var mı", "nice, oradaki {kind} spot'larına bak"],
        "en": ["Now find a {kind} there", "Great, any good {kind} over there?"],
    }
    return [_user(style.wrap(first)),
            _call("set_active_location", {"location_query": place}),
            _user(style.wrap(fill(_pick(rng, second, style.register), kind=kind))),
            _call("search_restaurants", {"query": query})]


# --------------------------------------------------------------------------
# Posts, saved, shared, memory, people, navigation
# --------------------------------------------------------------------------
POSTS = {
    "clean": ["{dish} hakkında neler paylaşılmış?", "{dish} ile ilgili gönderileri göster.",
              "Yakında {dish} paylaşan olmuş mu?"],
    "casual": ["{dish} fotosu atan var mı", "{dish} ile ilgili ne paylaşılmış bakayım",
               "{dish} gönderilerine bakalım"],
    "hybrid": ["{dish} ile ilgili post'ları göster", "{dish} tag'li paylaşım var mı",
               "feed'de {dish} geçen post var mı"],
    "en": ["What are people posting about {dish}?", "Any posts mentioning {dish}?"],
}


def posts(rng: random.Random, style: Style) -> list[dict]:
    if style.register == "en":
        word, query = rng.choice(L.DISHES_EN)
    else:
        word = query = rng.choice(L.DISHES)
    text = fill(_pick(rng, POSTS, style.register), dish=word)
    return [_user(style.wrap(text)), _call("search_posts", {"query": query})]


#: ``(clean, casual, hybrid, arguments)``; a callable builds slot-dependent ones.
SAVED = (
    ("Kaydettiğim mekanlardan {kind} olanları göster.", "kaydettiğim {kind} yerleri neydi",
     "save ettiğim {kind} mekanları göster", lambda s: {"item_type": "restaurant",
                                                        "query": s["kind_q"]}),
    ("Kaydettiğim mekanlardan en yakını hangisi?", "kaydettiğim yerlerden en yakını hangisi",
     "save'lediğim yerlerden en yakın olanı göster",
     lambda s: {"item_type": "restaurant", "sort_by": "distance"}),
    ("Kaydettiğim yemeklerden en ucuzu hangisi?", "kaydettiğim yemeklerin en ucuzu hangisiydi",
     "saved yemeklerimden en cheap olan hangisi",
     lambda s: {"item_type": "menu_item", "sort_by": "price_asc"}),
    ("Kaydettiğim yemeklerden en pahalısı hangisi?", "kaydettiğim yemeklerin en pahalısı neydi",
     "saved yemeklerimden en expensive olan hangisi",
     lambda s: {"item_type": "menu_item", "sort_by": "price_desc"}),
    ("Kaydettiğim gönderilerde {dish} geçenleri bul.", "kaydettiğim gönderilerde {dish} var mıydı",
     "saved post'larımda {dish} ile ilgili olanlar",
     lambda s: {"item_type": "post", "query": s["dish"]}),
    ("{loc:loc} kaydettiğim mekanlar hangileri?", "{loc:loc} kaydettiğim yerler neydi",
     "{loc:loc} save ettiğim place'ler hangileri",
     lambda s: {"item_type": "restaurant", "location_query": s["loc"]}),
    ("En son kaydettiğim mekanları göster.", "en son hangi mekanları kaydetmiştim",
     "last save ettiğim mekanları göster",
     lambda s: {"item_type": "restaurant", "sort_by": "recency"}),
    ("Kaydettiğim yemeklerde {dish} var mı?", "kaydettiğim yemeklerde {dish} var mıydı",
     "saved yemeklerimde {dish} var mı",
     lambda s: {"item_type": "menu_item", "query": s["dish"]}),
    ("Son kaydettiklerim neler?", "en son neleri kaydetmiştim", "recent save'lerim neler",
     lambda s: {"sort_by": "recency"}),
)
SAVED_EN = (
    ("Which of my saved places is the closest?", {"item_type": "restaurant", "sort_by": "distance"}),
    ("Show the cheapest dish I saved", {"item_type": "menu_item", "sort_by": "price_asc"}),
    ("My most recently saved places", {"item_type": "restaurant", "sort_by": "recency"}),
)


def _slots(rng: random.Random) -> dict:
    kind, kind_q = rng.choice(L.KINDS)
    return {"kind": kind, "kind_q": kind_q, "dish": rng.choice(L.DISHES),
            "loc": rng.choice(L.PLACES), "name": rng.choice(L.NAMES)}


def saved(rng: random.Random, style: Style) -> list[dict]:
    if style.register == "en":
        text, args = rng.choice(SAVED_EN)
        return [_user(style.wrap(text)), _call("search_saved", dict(args))]
    row, slots = rng.choice(SAVED), _slots(rng)
    text = fill(row[REGISTERS.index(style.register)], **slots)
    return [_user(style.wrap(text)), _call("search_saved", row[3](slots))]


SHARED = (
    ("{name:gen} bana gönderdiği mekanı bul.", "{name:gen} yolladığı mekan hangisiydi",
     "{name:gen} bana send ettiği mekan hangisiydi",
     lambda s: {"friend_name": s["name"], "direction": "received", "item_type": "restaurant"}),
    ("{name} bana bir yemek paylaşmıştı, hangisiydi?", "{name:gen} attığı yemek hangisiydi",
     "{name} chat'te bi yemek atmıştı, find et",
     lambda s: {"friend_name": s["name"], "direction": "received", "item_type": "menu_item"}),
    ("{name:gen} bana attığı gönderiyi bul.", "{name:gen} attığı gönderi hangisiydi",
     "{name:gen} bana attığı post'u bul",
     lambda s: {"friend_name": s["name"], "direction": "received", "item_type": "post"}),
    ("{name} geçenlerde bana bir {kind} göndermişti, bulur musun?",
     "{name} bir {kind} atmıştı bana, bulsana", "{name} bana bi {kind} send etmişti, hangisiydi",
     lambda s: {"friend_name": s["name"], "direction": "received", "query": s["kind_q"]}),
    ("{name:dat} daha önce hangi mekanları yollamıştım?", "{name:dat} hangi mekanları atmıştım",
     "{name:dat} share ettiğim mekanları göster",
     lambda s: {"friend_name": s["name"], "direction": "sent", "item_type": "restaurant"}),
    ("Benim {name:dat} attığım yemekler hangileriydi?", "{name:dat} attığım yemekler neydi",
     "{name:dat} send ettiğim yemekler hangileri",
     lambda s: {"friend_name": s["name"], "direction": "sent", "item_type": "menu_item"}),
    ("Arkadaşlarımın bana yolladığı yerleri listele.", "bana yollanan yerleri göster",
     "chat'lerde bana gelen mekanları listele",
     lambda s: {"direction": "received", "item_type": "restaurant"}),
    ("Sohbetlerde bana atılan yemekler hangileri?", "bana atılan yemekler neydi",
     "bana share edilen yemekleri göster",
     lambda s: {"direction": "received", "item_type": "menu_item"}),
    ("Benim başkalarına gönderdiğim mekanları göster.", "benim attığım mekanlar hangileriydi",
     "benim share ettiğim tüm mekanlar",
     lambda s: {"direction": "sent", "item_type": "restaurant"}),
)
SHARED_EN = (
    ("What was the place {name} sent me?",
     lambda s: {"friend_name": s["name"], "direction": "received", "item_type": "restaurant"}),
    ("Show the places I shared with {name}",
     lambda s: {"friend_name": s["name"], "direction": "sent", "item_type": "restaurant"}),
)


def shared(rng: random.Random, style: Style) -> list[dict]:
    slots = _slots(rng)
    if style.register == "en":
        text, build = rng.choice(SHARED_EN)
        return [_user(style.wrap(fill(text, **slots))), _call("search_shared_items", build(slots))]
    row = rng.choice(SHARED)
    text = fill(row[REGISTERS.index(style.register)], **slots)
    return [_user(style.wrap(text)), _call("search_shared_items", row[3](slots))]


MEMORY = {
    "clean": ["Geçen gün sana sorduğum {topic} hangisiydi?",
              "Sana daha önce {topic} sormuştum, ne demiştin?",
              "Önceki sohbetimizde bahsettiğin {topic} hangisiydi?",
              "Sana {topic} sevdiğimi söylemiş miydim?"],
    "casual": ["sana sorduğum {topic} hangisiydi", "önceki konuşmamızdaki {topic} yeri hangisiydi",
               "bana dediğin {topic} neresiydi"],
    "hybrid": ["last time konuştuğumuz {topic} hangisiydi",
               "previous chat'te bahsettiğin {topic} mekanı neydi",
               "bana suggest ettiğin {topic} neresiydi"],
    "en": ["What was the {topic} place we talked about last time?",
           "Do you remember the {topic} spot you suggested?"],
}


def memory(rng: random.Random, style: Style) -> list[dict]:
    if style.register == "en":
        topic, query = rng.choice(L.DISHES_EN + L.KINDS_EN)
    elif rng.random() < 0.5:
        topic, query = rng.choice(L.KINDS)
    else:
        topic = query = rng.choice(L.DISHES)
    text = fill(_pick(rng, MEMORY, style.register), topic=topic)
    return [_user(style.wrap(text)), _call("search_chat_memory", {"query": query})]


FRIENDS = (
    ("Arkadaşlarımı listele.", "arkadaşlarım kimdi", "friends'lerimi göster", None),
    ("feast'teki arkadaşlarım kimler?", "kimlerle arkadaşım burada", "kimler added bende", None),
    ("{name} arkadaşım mı?", "{name} ekli mi bende", "{name} bende added mı", "name"),
    ("Arkadaşlarım arasında {name} var mı?", "{name} arkadaşlarımda var mı",
     "{name} friends'imde mi", "name"),
)
SUGGESTED = {
    "clean": ["Tanıyor olabileceğim kişileri göster.", "Kimleri arkadaş olarak ekleyebilirim?",
              "Bana arkadaş öner."],
    "casual": ["ekleyebileceğim biri var mı", "tanıdık çıkar mı önerilerde",
               "yeni kimleri ekleyebilirim"],
    "hybrid": ["suggested users'ı göster", "people you may know listesine bakayım",
               "yeni friend önerisi var mı"],
    "en": ["Who can I add as a friend?", "Show me people I may know"],
}


def people(rng: random.Random, style: Style) -> list[dict]:
    if rng.random() < 0.42:
        return [_user(style.wrap(_pick(rng, SUGGESTED, style.register))),
                _call("get_suggested_users", {})]
    if style.register == "en":
        name = rng.choice(L.NAMES)
        text, args = rng.choice((("Show my friends", {}), ("Who are my friends on feast?", {}),
                                 (f"Is {name} one of my friends?", {"query": name})))
        return [_user(style.wrap(text)), _call("list_friends", args)]
    row, name = rng.choice(FRIENDS), rng.choice(L.NAMES)
    text = fill(row[REGISTERS.index(style.register)], name=name)
    return [_user(style.wrap(text)), _call("list_friends", {"query": name} if row[3] else {})]


def navigate(rng: random.Random, style: Style) -> list[dict]:
    if style.register == "en":
        screen = rng.choice(tuple(L.SCREENS_EN))
        text = rng.choice(L.SCREENS_EN[screen])
    else:
        screen = rng.choice(tuple(L.SCREENS))
        text = rng.choice(L.SCREENS[screen][REGISTERS.index(style.register)])
    return [_user(style.wrap(text)), _call("open_screen", {"screen": screen})]


# --------------------------------------------------------------------------
# Messages  ->  propose in text, call only after a confirmation
# --------------------------------------------------------------------------
ASK = {
    "clean": ["{who} {msg} yaz.", "{who} {msg} diye mesaj at.", "{who} şunu ilet: {msg}"],
    "casual": ["{who} yaz, {msg} de", "{who} {msg} yazsana", "{who} mesaj at {msg} diye"],
    "hybrid": ["{who} msg at: {msg}", "{who} DM'den yaz, {msg}", "{who} şunu send et: {msg}"],
}
ASK_REPORTED = {
    "clean": ["{who} {rep} söyle.", "{who} {rep} haber verir misin?"],
    "casual": ["{who} {rep} söylesene", "{who} {rep} bildir"],
}
CONFIRM = {
    "clean": ["Evet, gönder.", "Tamam, gönderebilirsin.", "Onaylıyorum."],
    "casual": ["evet", "aynen gönder", "tamam yolla", "he at", "olur gönder", "gönder gitsin"],
    "hybrid": ["yep gönder", "ok gönder", "okay yolla", "confirm", "send it", "yes pls"],
}
CANCEL = {
    "clean": ["Hayır, gönderme.", "Vazgeçtim, iptal et.", "Boş ver, göndermeyelim."],
    "casual": ["yok boşver", "gönderme gönderme", "iptal iptal", "sonra yazarım ben"],
    "hybrid": ["cancel et", "no no gönderme", "never mind, iptal", "stop, gönderme"],
}
CHANGE = {
    "clean": ["Hayır, şöyle yaz: {msg}", "Değiştirelim, şunu yaz: {msg}"],
    "casual": ["yok şöyle olsun: {msg}", "dur, şunu yaz: {msg}"],
    "hybrid": ["wait, şöyle değiştir: {msg}", "edit: {msg}"],
}
CANCELLED = ("Tamam, göndermiyorum.", "Peki, mesajı göndermedim.", "Tamam, iptal ettim.")


def _recipients(rng: random.Random) -> tuple[list[str], str]:
    """Names, and "Onur ve Ece'ye" - the dative goes on the last one."""
    names = rng.sample(L.NAMES, 2 if rng.random() < 0.18 else 1)
    return names, " ve ".join(names[:-1] + [L.dative(names[-1])])


def _propose(who: str, message: str, rng: random.Random, venue_name: str = "") -> str:
    if venue_name:
        return f'{who} {venue_name} mekanını "{message}" mesajıyla göndereyim mi?'
    if rng.random() < 0.8:
        return f'{who} "{message}" göndereyim mi?'
    return f'{who} şunu göndereyim mi: "{message}"'


def _dictated(rng: random.Random, style: Style) -> tuple[str, str, str]:
    """``(the user's sentence, who in the dative, the message)`` for a request."""
    names, who = _recipients(rng)
    register = style.register if style.register in ASK else "clean"
    if register in ASK_REPORTED and rng.random() < 0.2:
        reported, message = rng.choice(L.REPORTED)
        text = fill(rng.choice(ASK_REPORTED[register]), who=who, rep=reported)
        return style.wrap(text), who, message
    raw = rng.choice(L.MESSAGES_HYBRID if register == "hybrid" and rng.random() < 0.6
                     else L.MESSAGES)
    # Styled apart from the sentence: the message is the user's own wording.
    message = style.text(raw)
    frame = fill(rng.choice(ASK[register]), who=who, msg="\x00")
    head, tail = frame.split("\x00")
    text = style.text(head) + message + style.text(tail)
    return text, who, message


def _names_of(who: str) -> list[str]:
    return [part.split("'")[0] for part in who.split(" ve ")]


def message(rng: random.Random, style: Style) -> list[dict]:
    register = style.register if style.register in CONFIRM else "clean"
    request, who, text = _dictated(rng, style)
    turns = [_user(request), _say(_propose(who, text, rng))]
    roll = rng.random()
    if roll < 0.14:  # the proposal itself is the lesson; no second turn
        return turns
    if roll < 0.30:
        return turns + [_user(style.wrap(rng.choice(CANCEL[register]))),
                        _say(rng.choice(CANCELLED))]
    if roll < 0.44:
        changed = style.text(rng.choice(L.MESSAGES))
        frame = rng.choice(CHANGE[register]).split("{msg}")
        turns += [_user(style.text(frame[0]) + changed + style.text(frame[1])),
                  _say(_propose(who, changed, rng))]
        text = changed
    return turns + [_user(style.wrap(rng.choice(CONFIRM[register]))),
                    _call(SEND, {"friend_names": _names_of(who), "message": text})]


SHARE_VENUE = {
    "clean": ["{ref:acc} {name:dat} gönder, {msg} diye.", "{name:dat} {ref:acc} yolla, {msg} yaz."],
    "casual": ["{name:dat} {ref:acc} yolla, yanına da {msg} yaz",
               "{ref:acc} {name:dat} gönder {msg} diye"],
    "hybrid": ["{ref:acc} {name:dat} forward et, {msg} notuyla",
               "{name:dat} {ref:acc} send et, {msg} yaz"],
}


def share_venue(rng: random.Random, style: Style) -> list[dict]:
    """A listed venue goes to a friend: the id must travel with the message."""
    register = style.register if style.register in SHARE_VENUE else "clean"
    items, history = venue_results(rng, style.executed)
    index = rng.randrange(len(items))
    venue_name, venue_id = items[index]
    name = rng.choice(L.NAMES)
    text = style.text(rng.choice(L.MESSAGES))
    frame = fill(rng.choice(SHARE_VENUE[register]).replace("{ref:acc}", _ORD_ACC[index]),
                 name=name, msg="\x00")
    head, tail = frame.split("\x00")
    request = style.text(cap(head)) + text + style.text(tail)
    who = L.dative(name)
    turns = history + [_user(request), _say(_propose(who, text, rng, venue_name))]
    if rng.random() < 0.2:
        return turns + [_user(style.wrap(rng.choice(CANCEL[register]))),
                        _say(rng.choice(CANCELLED))]
    return turns + [_user(style.wrap(rng.choice(CONFIRM[register]))),
                    _call(SEND, {"friend_names": [name], "message": text,
                                 "attach_restaurant_id": venue_id})]


# --------------------------------------------------------------------------
# No tool: small talk, the context line, things feast cannot do
# --------------------------------------------------------------------------
TALK = (
    ({"clean": ["Merhaba!", "İyi akşamlar.", "Günaydın."],
      "casual": ["selamlar", "naber", "meraba", "hey", "iyi akşamlar"],
      "hybrid": ["hello naber", "hi kanka", "hii nasılsın", "yo naber"],
      "en": ["Hi there", "Hello!"]},
     ("Merhaba! Bugün canın ne çekiyor?", "Selam! Yakınında bir yer mi arıyorsun?",
      "Merhaba! Sana nasıl yardımcı olabilirim?"),
     ("Hi! What are you in the mood for today?",)),
    ({"clean": ["Teşekkür ederim.", "Sağ ol, çok yardımcı oldun."],
      "casual": ["saol", "eyv", "tşk", "süpersin", "çok iyi oldu bu"],
      "hybrid": ["thanks ya", "ty çok iyi oldu", "great, tam istediğim"],
      "en": ["Thanks a lot!", "Great, thank you"]},
     ("Rica ederim, afiyet olsun!", "Ne demek, her zaman!",
      "Rica ederim! Başka bir şey lazım olursa buradayım."),
     ("You're welcome! Enjoy.",)),
    ({"clean": ["Neler yapabilirsin?", "Bana nasıl yardımcı olabilirsin?",
                "Bu uygulamada senin görevin ne?"],
      "casual": ["sana neler sorabilirim", "sen ne yapıyorsun tam olarak"],
      "hybrid": ["neler handle edebiliyosun", "hangi konularda help edebilirsin"],
      "en": ["What can you do?", "How can you help me?"]},
     ("Yakınındaki mekanları ve yemekleri bulabilir, fiyatlara bakabilirim. Kaydettiklerini ve "
      "arkadaşlarının paylaştıklarını gösterebilir, arkadaşlarına mesaj gönderebilirim.",),
     ("I can find restaurants and dishes near you, check prices, show your saved items and "
      "what friends shared, and send messages to your friends.",)),
    ({"clean": ["Nasılsın?", "Nasıl gidiyor?"], "casual": ["nasılsın", "ne haber, nasıl gidiyor"],
      "hybrid": ["how are you kanka", "nasılsın, all good?"], "en": ["How are you?"]},
     ("İyiyim, teşekkürler! Sana nasıl yardımcı olabilirim?",),
     ("I'm good, thanks! How can I help?",)),
    ({"clean": ["Görüşürüz.", "Hoşça kal."], "casual": ["hadi görüşürüz", "bb", "kaçtım ben"],
      "hybrid": ["bye kanka", "see you", "ok ben kaçtım, bye"], "en": ["Bye!", "See you later"]},
     ("Görüşürüz, afiyet olsun!", "Hoşça kal!"), ("See you! Enjoy your meal.",)),
    # Appended last: mustachio.teacher zips the first five kinds by position.
    ({"clean": ["Sen kimsin?", "Adın ne?", "Sen bir yapay zeka mısın?", "Yazdıklarım bir yere "
                "gidiyor mu?"],
      "casual": ["sen kimsin ya", "adın ne senin", "sen bot musun", "sen nesin tam olarak"],
      "hybrid": ["who are you kanka", "sen AI mısın", "senin name ne"],
      "en": ["Who are you?", "What's your name?", "Are you an AI?"]},
     ("Ben Mustachio, feast'in asistanıyım. Bu telefonda çalışıyorum, yazdıkların telefondan "
      "çıkmıyor.",
      "Mustachio ben, feast'in bıyıklı asistanı. Telefonunda çalışıyorum; yazdıkların burada "
      "kalıyor.",
      "Mustachio. feast'in asistanıyım, bu telefonda çalışırım ve yazdıkların dışarı çıkmaz."),
     ("I'm Mustachio, feast's assistant. I run on this phone, and nothing you write leaves it.",)),
)
REFUSE = (
    "Bunu yapamıyorum. Yakınındaki mekanları, yemekleri ve gönderileri bulabilir; "
    "kaydettiklerin ve arkadaşlarınla ilgili yardımcı olabilirim.",
    "Bu konuda yardımcı olamıyorum. Mekan ve yemek aramada, kaydettiklerinde ve "
    "arkadaşlarınla ilgili işlerde yardımcı olabilirim.",
)
REFUSE_EN = ("I can't help with that here. I can find restaurants, dishes and posts near you, "
             "and help with your saved items and friends.")
OOS_EN = ("Call me a taxi", "Tell me a joke", "Translate this sentence into English",
          "What's in the news today?", "Book a table for me", "Order food for me")
_DAYS = {"Mon": ("Pazartesi", "Monday"), "Tue": ("Salı", "Tuesday"),
         "Wed": ("Çarşamba", "Wednesday"), "Thu": ("Perşembe", "Thursday"),
         "Fri": ("Cuma", "Friday"), "Sat": ("Cumartesi", "Saturday"),
         "Sun": ("Pazar", "Sunday")}
_MONTHS = (("Ocak", "January"), ("Şubat", "February"), ("Mart", "March"), ("Nisan", "April"),
           ("Mayıs", "May"), ("Haziran", "June"), ("Temmuz", "July"), ("Ağustos", "August"),
           ("Eylül", "September"), ("Ekim", "October"), ("Kasım", "November"),
           ("Aralık", "December"))
ASK_TIME = {"clean": ["Saat kaç?", "Şu an saat kaç acaba?", "Saati söyler misin?"],
            "casual": ["saat kaç", "saat kaçtı", "saate bakar mısın"],
            "hybrid": ["time ne şu an", "saat kaç şu an pls"],
            "en": ["What time is it?", "Do you have the time?"]}
ASK_DATE = {"clean": ["Bugün ayın kaçı?", "Bugünün tarihi ne?", "Bugün hangi gün?"],
            "casual": ["bugün ayın kaçıydı", "hangi gündeyiz"],
            "hybrid": ["bugünün date'i ne", "today ayın kaçı"],
            "en": ["What's the date today?"]}
ASK_PLACE = {"clean": ["Şu an neredeyim?", "Arama konumum neresi?",
                       "Hangi bölgeye göre arıyorsun?"],
             "casual": ["neredeyim ben şu an", "konum nerede görünüyor"],
             "hybrid": ["active location'ım ne", "şu an hangi area'ya göre bakıyosun"],
             "en": ["Where am I right now?", "Which area are you searching in?"]}


#: Share of small-talk turns answered with a teacher-written reply when the
#: pool has one for that sentence (the rest keep the template's).
TEACHER_REPLY_SHARE = 0.7
_SMALLTALK: dict | None = None


def smalltalk_pool() -> dict:
    """``{user sentence: {"kind", "replies"}}`` from :mod:`mustachio.teacher`, or ``{}``.

    The templates above know a handful of replies per kind, and v0 learned
    exactly those.  A reply is taken from the pool only for the sentence it
    was written for; without a pool the templates are used, as before.
    """
    global _SMALLTALK
    if _SMALLTALK is None:
        import json

        from . import paths

        try:
            _SMALLTALK = json.loads((paths.DATA_DIR / "teacher" / "smalltalk.json")
                                    .read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _SMALLTALK = {}
    return _SMALLTALK


def _reply(rng: random.Random, sentence: str, fallback: str) -> str:
    written = (smalltalk_pool().get(sentence) or {}).get("replies") or []
    if written and rng.random() < TEACHER_REPLY_SHARE:
        return rng.choice(written)
    return fallback


def talk(rng: random.Random, style: Style) -> list[dict]:
    asks, replies, replies_en = rng.choice(TALK)
    sentence = _pick(rng, asks, style.register)
    reply = _reply(rng, sentence, rng.choice(replies_en if style.register == "en" else replies))
    return [_user(style.wrap(sentence)), _say(reply)]


def out_of_scope(rng: random.Random, style: Style) -> list[dict]:
    if style.register == "en":
        sentence = rng.choice(OOS_EN)
        return [_user(style.wrap(sentence)), _say(_reply(rng, sentence, REFUSE_EN))]
    sentence = rng.choice(L.OOS)[REGISTERS.index(style.register)]
    return [_user(style.wrap(sentence)), _say(_reply(rng, sentence, rng.choice(REFUSE)))]


def context_info(rng: random.Random, style: Style, ctx: dict) -> list[dict]:
    """Answered from the ``[ctx]`` line - the lesson is that no tool is needed."""
    english = style.register == "en"
    date, day, clock = ctx["now"].split(" ")
    year, month, dom = (int(part) for part in date.split("-"))
    kind = rng.choice(("time", "time", "date", "place", "place"))
    if kind == "time":
        ask, reply = ASK_TIME, (f"It's {clock}." if english else f"Saat {clock}.")
    elif kind == "date":
        ask = ASK_DATE
        reply = (f"Today is {_DAYS[day][1]}, {dom} {_MONTHS[month - 1][1]} {year}." if english
                 else f"Bugün {dom} {_MONTHS[month - 1][0]} {year}, {_DAYS[day][0]}.")
    else:
        ask, label = ASK_PLACE, ctx["label"]
        if ctx["is_gps"]:
            reply = f"You're around {label}." if english else f"Şu an {label} civarındasın."
        else:
            reply = (f"Your search location is set to {label}." if english
                     else f"Arama konumun {label} olarak ayarlı.")
    return [_user(style.wrap(_pick(rng, ask, style.register))), _say(reply)]


# --------------------------------------------------------------------------
# A follow-up that only adds a condition keeps the tool and its arguments
# --------------------------------------------------------------------------
REFINE_DISH = (
    ({"clean": ["{price} lirayı aşmasın.", "En fazla {price} lira olsun."],
      "casual": ["{price} tl üstü olmasın", "{price} lirayı aşmasın ama"],
      "hybrid": ["max {price} olsun", "{price} TL limit koy"],
      "en": ["Keep it under {price} lira"]}, "price"),
    ({"clean": ["Daha ucuzu var mı?", "En ucuzunu göster."],
      "casual": ["daha ucuzu yok mu", "en ucuzu hangisi"],
      "hybrid": ["cheaper olanı var mı", "en ucuzunu göster pls"],
      "en": ["Anything cheaper?"]}, "cheap"),
    ({"clean": ["Sadece açık olanları göster.", "Şu an açık olanlar hangileri?"],
      "casual": ["açık olanları göster sadece", "şu an açık olanlar olsun"],
      "hybrid": ["only open olanlar", "open olanları filtrele"],
      "en": ["Only the ones open now"]}, "open"),
    ({"clean": ["En fazla {n} dakika yürüyeyim.", "{n} dakikadan fazla yürümek istemiyorum."],
      "casual": ["çok yürümeyeyim, max {n} dk", "{n} dakikadan uzak olmasın"],
      "hybrid": ["{n} dk'dan fazla walk olmasın", "max {n} dk yürüme"],
      "en": ["No more than {n} minutes on foot"]}, "walk"),
    ({"clean": ["En yakını hangisi?"], "casual": ["en yakını hangisi"],
      "hybrid": ["nearest olanı hangisi"], "en": ["Which one is closest?"]}, "near"),
)
REFINE_VENUE_ONLY = (
    ({"clean": ["En popülerleri hangileri?"], "casual": ["en popüler olanlar hangileri"],
      "hybrid": ["en popular olanları göster"], "en": ["Which are the most popular?"]},
     "popular"),
)


def refine(rng: random.Random, style: Style) -> list[dict]:
    is_dish = rng.random() < 0.55
    text, args = (_dish_request if is_dish else _venue_request)(rng, style.register)
    tool = "search_menu_items" if is_dish else "search_restaurants"
    options = [o for o in REFINE_DISH if is_dish or o[1] not in ("price", "cheap")]
    if not is_dish:
        options += list(REFINE_VENUE_ONLY)
    asks, what = rng.choice(options)
    slots, merged = {}, dict(args)
    if what == "price":
        slots["price"] = merged["max_price"] = rng.choice(L.PRICES)
    elif what == "cheap":
        merged["sort_by"] = "price_asc"
    elif what == "open":
        merged["open_now"] = True
    elif what == "walk":
        slots["n"] = merged["max_walk_minutes"] = rng.choice(L.WALKS)
    elif what == "near":
        merged["sort_by"] = "distance"
    else:
        merged["sort_by"] = "popularity"
    return [_user(style.wrap(text)), _call(tool, args),
            _user(style.wrap(fill(_pick(rng, asks, style.register), **slots))),
            _call(tool, merged)]


# --------------------------------------------------------------------------
# Running the tool: a result, and one sentence about it
# --------------------------------------------------------------------------
#: Share of conversations in which every call is executed and answered.
EXECUTED_SHARE = 0.6
#: Of executed searches: nothing found / the data could not be read.
EMPTY_SHARE, FAILED_SHARE = 0.12, 0.03
_ACTIONS = ("get_directions", "set_active_location", "open_screen", SEND)
_DETAILS = ("get_restaurant_details", "get_menu_item_details")
_STREETS = ("Gül", "Lale", "Çınar", "Ihlamur", "Menekşe", "Kumru", "Zambak", "Akasya",
            "Yasemin", "Papatya", "Değirmen", "Fener")
_HOURS = ("09:00-23:00", "08:00-22:00", "10:00-00:00", "07:30-21:00", "11:00-01:00",
          "12:00-23:30", "06:00-18:00")
_INGREDIENTS = ("dana kıyma, soğan ve biber", "tavuk, mantar ve krema", "nohut, tahin ve limon",
                "kuzu eti, domates ve sarımsak", "peynir, yumurta ve tereyağı",
                "bulgur, havuç ve nane", "un, tereyağı ve ceviz", "pirinç, sebze ve baharat")
_HANDLES = ("ece.yer", "gezgin_damak", "lezzet.avcisi", "midem_bayram", "sokak.tadi",
            "tabakta_ne_var", "burak.yer", "acikmis.biri")
_CAPTIONS = ("porsiyon büyük, fiyatına değer", "sıcak geldi, çok tazeydi",
             "biraz tuzluydu ama güzeldi", "sırası uzun ama değdi",
             "bugüne kadar yediğimin en iyisi", "servis hızlı, lezzet yerinde")
_WHY = ("sessiz, priz çok, saatlerce oturuluyor", "bahçesi geniş, çocuklar rahat ediyor",
        "avlusu ferah, akşamüstü serin oluyor", "masaları büyük, kalabalık gruplara uygun",
        "ışıklar loş, baş başa yemek için güzel", "servis hızlı, porsiyonlar büyük",
        "rafları kitap dolu, kuytu bir köşe", "maç günleri dolu, ekranı büyük")
_WHEN = ("dün", "geçen hafta", "3 gün önce", "geçen ay", "önceki gün")
_REMEMBERED = ("{venue} mekanını önermiştim", "{dish} için {venue} demiştim",
               "{dish} sevdiğini söylemiştin", "{venue} için yol tarifi istemiştin")
_DISH_EXTRAS = ("Özel", "Ev Yapımı", "Acılı", "Büyük Boy", "Klasik", "Tereyağlı", "Duble",
                "Yarım")
YES = {"clean": ["Evet, bak.", "Olur.", "Tamam, öyle yap.", "Evet lütfen."],
       "casual": ["olur", "evet bak", "aynen", "he bak bakalım", "tamam", "olur bak"],
       "hybrid": ["ok bak", "yes pls", "okay olur", "yep", "sure bak"],
       "en": ["Yes please", "Sure, go ahead", "Yes"]}
NO = {"clean": ["Hayır, kalsın.", "Gerek yok."], "casual": ["yok kalsın", "boşver", "yok ya"],
      "hybrid": ["no need", "nah kalsın", "never mind"], "en": ["No, that's fine", "Never mind"]}
DECLINED = ("Tamam.", "Peki, başka bir şey lazım olursa buradayım.", "Tamam, kalsın.")
DECLINED_EN = ("Okay.", "Alright, I'm here if you need anything else.")


def _title(text: str) -> str:
    return " ".join(cap(word) for word in text.split())


def _venue_items(rng: random.Random, count: int, args: dict, used: set) -> list[dict]:
    far = (int(args.get("max_walk_minutes") or 0)
           or int(args.get("max_distance_meters") or 0) // 80 or 25)
    names: list[str] = []
    while len(names) < count:
        name = _venue_name(rng)
        if name not in names:
            names.append(name)
    walks = sorted(rng.randint(1, max(2, far)) for _ in names)
    if args.get("sort_by") in ("popularity", "name"):
        rng.shuffle(walks)
    must_open = bool(args.get("open_now") or args.get("open_at"))
    return [mock.venue_item(_new_id(rng, "r", used), name, walk, must_open or rng.random() < 0.85)
            for name, walk in zip(names, walks)]


def _dish_items(rng: random.Random, count: int, args: dict, used: set) -> list[dict]:
    base = str(args.get("query") or rng.choice(L.DISHES))
    ceiling = args.get("max_price")
    prices = [p for p in L.PRICES if ceiling is None or p <= float(ceiling)] or [int(ceiling)]
    items = [mock.dish_item(_new_id(rng, "m", used), f"{extra} {_title(base)}",
                            rng.choice(prices), _venue_name(rng))
             for extra in rng.sample(_DISH_EXTRAS, count)]
    if args.get("sort_by") != "distance":
        items.sort(key=lambda item: item["price"], reverse=args.get("sort_by") == "price_desc")
    return items


def _post_items(rng: random.Random, count: int) -> list[dict]:
    return [mock.post_item(handle, caption, _venue_name(rng))
            for handle, caption in zip(rng.sample(_HANDLES, count), rng.sample(_CAPTIONS, count))]


def fake_result(rng: random.Random, name: str, args: dict, known: dict,
                found: bool = False) -> dict:
    """An invented result for a call, in the shapes :mod:`feast_ai.agent.mock` serves.

    ``known`` maps the ids this conversation has listed to their items, so a
    detail lookup talks about the venue that was listed.  ``found`` forces a
    non-empty result (the search after a widening).
    """
    roll = 1.0 if found else rng.random()
    if roll < FAILED_SHARE:
        return mock.done(False) if name in _ACTIONS else mock.result([], ok=False)
    empty = roll < FAILED_SHARE + EMPTY_SHARE and name not in _DETAILS
    used = set(known)
    kind = args.get("item_type")
    if name == "search_restaurants":
        return mock.result([] if empty else
                           _venue_items(rng, rng.choice((1, 2, 3, 3, 4, 5, 7)), args, used))
    if name == "search_menu_items":
        return mock.result([] if empty else
                           _dish_items(rng, rng.choice((1, 2, 2, 3, 4, 5)), args, used))
    if name == "get_restaurant_details":
        listed = known.get(args.get("restaurant_id")) or {}
        return mock.result([{
            "id": args.get("restaurant_id"), "name": listed.get("name") or _venue_name(rng),
            "open": listed.get("open", rng.random() < 0.75), "hours": rng.choice(_HOURS),
            "address": f"{rng.choice(_STREETS)} Sk. No:{rng.randint(1, 90)}",
            "menu": [_title(dish) for dish in rng.sample(L.DISHES, 3)]}])
    if name == "get_menu_item_details":
        listed = known.get(args.get("menu_item_id")) or {}
        item = mock.dish_item(args.get("menu_item_id"),
                              listed.get("name") or _title(rng.choice(L.DISHES)),
                              listed.get("price") or rng.choice(L.PRICES),
                              listed.get("restaurant") or _venue_name(rng))
        item["ingredients"] = rng.choice(_INGREDIENTS)
        return mock.result([item])
    if name == "get_directions":
        listed = known.get(args.get("restaurant_id")) or {}
        return mock.done(opened="maps", name=listed.get("name") or _venue_name(rng),
                         mode=args.get("mode") or "walk")
    if name == "set_active_location":
        if args.get("use_current_location"):
            return mock.done(active_location=rng.choice(_LABELS), is_gps=True)
        return mock.done(active_location=str(args.get("location_query") or ""), is_gps=False)
    if name == "search_posts":
        return mock.result([] if empty else _post_items(rng, rng.choice((1, 2, 3, 5))))
    if name == "ask_about_posts":
        count = 0 if empty else rng.choice((1, 2, 2, 3))
        return mock.result([mock.opinion_item(_venue_name(rng), why)
                            for why in rng.sample(_WHY, count)])
    if name in ("search_saved", "search_shared_items"):
        count = 0 if empty else rng.choice((1, 2, 3, 4))
        if kind == "post":
            items = _post_items(rng, count)
        elif kind == "menu_item":
            items = _dish_items(rng, count, {k: v for k, v in args.items() if k == "sort_by"}
                                if kind else {}, used)
        elif kind == "restaurant" or rng.random() < 0.6:
            items = _venue_items(rng, count, {}, used)
        else:
            items = _dish_items(rng, count, {}, used)
        if name == "search_shared_items":
            for item in items:
                item["friend"] = str(args.get("friend_name") or rng.choice(L.NAMES))
        return mock.result(items)
    if name == "search_chat_memory":
        count = 0 if empty else rng.choice((1, 1, 2))
        return mock.result([
            {"when": when, "text": rng.choice(_REMEMBERED).format(
                venue=_venue_name(rng), dish=str(args.get("query") or rng.choice(L.DISHES)))}
            for when in rng.sample(_WHEN, count)])
    if name == "list_friends":
        if args.get("query"):
            return mock.result([] if empty else [{"name": str(args["query"])}])
        return mock.result([{"name": n} for n in rng.sample(L.NAMES, rng.choice((2, 3, 4, 6, 8)))])
    if name == "get_suggested_users":
        return mock.result([] if empty else [
            {"name": n, "mutual": rng.randint(1, 9)}
            for n in rng.sample(L.NAMES, rng.choice((2, 3, 4)))])
    if name == "open_screen":
        return mock.done(screen=str(args.get("screen") or ""))
    if name == SEND:
        return mock.done(sent_to=list(args.get("friend_names") or []))
    return mock.result([])


def bare(messages: list[dict]) -> list[dict]:
    """The conversation without the notes left for :func:`execute`."""
    return [{k: v for k, v in message.items() if k not in ("focus", "result")}
            for message in messages]


def execute(rng: random.Random, messages: list[dict], style: Style) -> list[dict]:
    """Play every call out: its tool result, then the sentence about it.

    An empty result whose sentence asks a yes/no question ("Kapalı olanlara
    da bakayım mı?") is sometimes followed by the answer: a yes is the same
    call without the filter the question named
    (:func:`feast_ai.agent.replies.widened`), a no is acknowledged.  "evet"
    after such a question and "evet" after a message proposal are different
    calls, and only the conversation says which.
    """
    english = style.register == "en"
    out: list[dict] = []
    known: dict = {}
    queue = [dict(message) for message in messages]
    while queue:
        message = queue.pop(0)
        focus, result = message.pop("focus", ""), message.pop("result", None)
        found = bool(message.pop("found", False))
        out.append(message)
        if not message.get("tool_calls"):
            continue
        function = message["tool_calls"][0]["function"]
        name, args = function["name"], function["arguments"]
        if result is None:
            result = fake_result(rng, name, args, known, found)
        for item in result.get("items") or []:
            if item.get("id"):
                known[item["id"]] = item
        out.append({"role": "tool",
                    "content": [{"type": "tool_response", "name": name, "response": result}]})
        out.append(_say(replies.say(name, args, result, english, focus, rng)))
        wider = replies.widened(name, args)
        if wider is not None and result.get("ok") and not result.get("items"):
            roll = rng.random()
            if roll < 0.5:
                again = _call(name, wider)
                again["found"] = True
                queue[:0] = [_user(style.wrap(_pick(rng, YES, style.register))), again]
            elif roll < 0.62:
                queue[:0] = [_user(style.wrap(_pick(rng, NO, style.register))),
                             _say(rng.choice(DECLINED_EN if english else DECLINED))]
    return out


# --------------------------------------------------------------------------
# A chat is rarely about one thing
# --------------------------------------------------------------------------
def session(rng: random.Random, style: Style, ctx: dict) -> list[dict]:
    """Two to four unrelated requests in one conversation, by one typist.

    Every family above is a conversation about one thing, and a model trained
    only on those learns that a chat *is* about one thing.  The first real
    chat with the v0 model showed it: after a location change and two
    searches, "Denize yaz, akşam 8de buluşalım de" came back as
    ``get_directions(restaurant_id="Deniz")`` - a message request read
    through the searches before it.  Here a search is followed by a message,
    a message by a screen, a screen by small talk, and each target is still
    decided by its own turn.
    """
    parts = [p for p in _SESSION_PARTS if not (style.register == "en" and p in TURKISH_ONLY)]
    weights = [w for p, w in FAMILIES if p in parts]
    messages: list[dict] = []
    for builder in rng.choices(parts, weights, k=rng.choice((2, 2, 3, 3, 4))):
        messages += builder(rng, style, ctx) if builder is context_info else builder(rng, style)
    return messages


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
FAMILIES = (
    (venue, 14), (dish, 14), (vibe, 7), (details, 16), (location, 8),
    (location_then_search, 3), (posts, 5), (saved, 8), (shared, 7), (memory, 5),
    (people, 8), (navigate, 8), (message, 11), (share_venue, 4), (talk, 6),
    (context_info, 6), (out_of_scope, 6), (refine, 8),
)
_SESSION_PARTS = tuple(builder for builder, _weight in FAMILIES)
#: Share of conversations that are mixed sessions.
SESSION_SHARE = 0.22
_LABELS = ("Karaköy, Beyoğlu, İstanbul", "Caddebostan, İstanbul", "Bornova, İzmir",
           "Çankaya", "Levent, İstanbul", "Kordon, İzmir", "Bostancı, İstanbul",
           "Ataşehir, İstanbul", "Bakırköy, İstanbul", "Eskişehir", "Antalya", "Suadiye, İstanbul")


def random_ctx(rng: random.Random) -> dict:
    import datetime

    moment = datetime.datetime(2026, 1, 1) + datetime.timedelta(
        minutes=rng.randrange(0, 60 * 24 * 540))
    return {"now": moment.strftime("%Y-%m-%d ") + ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat",
                                                   "Sun")[moment.weekday()]
            + moment.strftime(" %H:%M"),
            "label": rng.choice(_LABELS), "is_gps": rng.random() < 0.8}


#: Families written in Turkish only: an English draw is redrawn for them.
TURKISH_ONLY = (message, share_venue)


def one(rng: random.Random) -> Example:
    builders, weights = zip(*FAMILIES)
    builder = session if rng.random() < SESSION_SHARE else rng.choices(builders, weights)[0]
    register = rng.choices(REGISTERS, REGISTER_WEIGHTS)[0]
    while register == "en" and builder in TURKISH_ONLY:
        register = rng.choices(REGISTERS, REGISTER_WEIGHTS)[0]
    style, ctx = Style(rng, register), random_ctx(rng)
    style.executed = rng.random() < EXECUTED_SHARE
    messages = (builder(rng, style, ctx) if builder in (context_info, session)
                else builder(rng, style))
    messages = execute(rng, messages, style) if style.executed else bare(messages)
    return Example(builder.__name__, register, messages, ctx)
