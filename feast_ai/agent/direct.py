"""Serve a message with no SLM at all.

When the router is confident and the intent needs no reasoning - "show my
saved items", "back to my location", "what time is it" - generating text
would only add latency and a chance to hallucinate.  These handlers answer
deterministically: a tool call the app executes, or a sentence built from the
CONTEXT block the SLM would have read anyway.

Each handler returns ``None`` when it cannot be sure (no screen keyword, a
time question that is really about something else).  ``None`` is not a
failure: the router then hands the message to the SLM with this intent's
tool group, so a direct path can only ever *skip* work, never block it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .context import AgentContext
from .text import contains_any, detect_lang

#: Screen keywords, most specific first ("arkadas ekle" before "arkadas").
_SCREENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("add_friends", ("arkadaş ekle", "arkadas ekle", "kişi ekle", "add friend")),
    ("qr_scanner", ("qr", "karekod", "kare kod", "menüyü tara", "scan")),
    ("saved", ("kaydet", "kayıtlı", "kaydedilen", "favori", "saved", "favorite")),
    ("notifications", ("bildirim", "notification")),
    ("settings", ("ayar", "setting")),
    ("profile", ("profil", "hesab", "hesap", "profile", "account")),
    ("chats", ("mesaj", "sohbet", "chat", "message")),
    ("friends", ("arkadaş", "arkadas", "friend")),
    ("map", ("harita", "map")),
    ("feed", ("akış", "akis", "ana sayfa", "anasayfa", "feed", "home")),
)

_TIME = ("saat", "tarih", "bugün", "gün", "time", "date", "today", "day")
_PLACE = ("konum", "nerede", "bölge", "semt", "lokasyon", "location", "where")


@dataclass(frozen=True)
class DirectResult:
    """A tool call (``tool`` set) or a finished sentence (``text`` set)."""

    tool: str = ""
    args: dict | None = None
    text: str = ""

    @property
    def is_call(self) -> bool:
        return bool(self.tool)


def _screen(text: str) -> DirectResult | None:
    for screen, keys in _SCREENS:
        if contains_any(text, keys):
            return DirectResult(tool="open_screen", args={"screen": screen})
    return None


def _context_answer(text: str, context: AgentContext) -> DirectResult | None:
    wants_time = contains_any(text, _TIME)
    wants_place = contains_any(text, _PLACE)
    if not (wants_time or wants_place):
        return None
    english = detect_lang(text) == "en"
    parts = []
    if wants_time:
        parts.append(f"It is {context.clock()}." if english else f"Şu an {context.clock()}.")
    if wants_place:
        if english:
            source = "GPS" if context.is_gps else "set manually"
            parts.append(f"Active location: {context.label} ({source}).")
        else:
            source = "GPS" if context.is_gps else "elle seçildi"
            parts.append(f"Aktif konum: {context.label} ({source}).")
    return DirectResult(text=" ".join(parts))


def _out_of_scope(text: str) -> DirectResult:
    if detect_lang(text) == "en":
        return DirectResult(text=(
            "I can't help with that here. I can find restaurants, dishes and posts "
            "near you, and help with your saved items and friends."
        ))
    return DirectResult(text=(
        "Bunu burada yapamıyorum. Yakınındaki restoranları, yemekleri ve "
        "gönderileri bulabilir; kaydettiklerin ve arkadaşlarınla ilgili "
        "yardımcı olabilirim."
    ))


def handle(intent_id: str, text: str, context: AgentContext) -> DirectResult | None:
    """The no-SLM answer for ``intent_id``, or ``None`` to fall through."""
    if intent_id == "saved_open":
        return DirectResult(tool="open_screen", args={"screen": "saved"})
    if intent_id == "location_reset":
        return DirectResult(tool="set_active_location", args={"use_current_location": True})
    if intent_id == "friends":
        return DirectResult(tool="list_friends", args={})
    if intent_id == "suggested_users":
        return DirectResult(tool="get_suggested_users", args={})
    if intent_id == "navigate":
        return _screen(text)
    if intent_id == "context_info":
        return _context_answer(text, context)
    if intent_id == "out_of_scope":
        return _out_of_scope(text)
    return None

