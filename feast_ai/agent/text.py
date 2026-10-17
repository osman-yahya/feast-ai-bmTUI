"""Turkish-aware text helpers for routing, direct calls and scoring.

``str.lower()`` is wrong for Turkish: it maps ``I`` to ``i`` (Turkish wants
``ı``) and ``İ`` to ``i`` plus a combining dot, so "İSTANBUL".lower() never
equals "istanbul".  Keyword rules and benchmark argument checks both go
through :func:`fold`, which lowercases the Turkish way and then drops the
diacritics, so "Kadıköy", "KADIKÖY" and "kadikoy" compare equal - users type
all three.
"""

from __future__ import annotations

import re

_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")
_WORD = re.compile(r"[\wçğıöşüÇĞİÖŞÜ']+")

#: Function words that only show up in one of the two languages. Short and
#: deliberately boring: this decides the reply language of templated answers,
#: nothing more.
_EN = {
    "the", "my", "me", "show", "what", "what's", "where", "is", "are", "find",
    "please", "open", "can", "you", "i", "a", "an", "near", "nearby", "how",
    "time", "it", "to", "and", "of", "for", "do", "any", "hi", "hello",
    "thanks", "thank", "who", "which", "weather",
}
_TR = {
    "mi", "mı", "mu", "mü", "misin", "mısın", "musun", "ne", "nerede", "göster",
    "bul", "var", "yakın", "bana", "bir", "ve", "bu", "şu", "ile", "için",
    "kaç", "nasıl", "aç", "lütfen", "merhaba", "selam", "teşekkürler", "saat",
}
#: Users often type Turkish without diacritics ("goster", "yakin").
_TR_ANY = _TR | {w.translate(_FOLD) for w in _TR}


def tr_lower(text: str) -> str:
    return text.replace("İ", "i").replace("I", "ı").lower()


def fold(text: str) -> str:
    """Lowercase the Turkish way, then strip diacritics."""
    return tr_lower(text).translate(_FOLD)


def words(text: str) -> list[str]:
    return _WORD.findall(tr_lower(text))


def detect_lang(text: str) -> str:
    """``"tr"`` or ``"en"``. Turkish wins ties - the app's users are Turkish."""
    if re.search(r"[çğıöşüÇĞİÖŞÜ]", text):
        return "tr"
    tokens = set(words(text))
    return "en" if len(tokens & _EN) > len(tokens & _TR_ANY) else "tr"


def contains_any(text: str, needles: tuple[str, ...]) -> bool:
    """Diacritic-insensitive substring test against folded ``needles``."""
    haystack = fold(text)
    return any(fold(n) in haystack for n in needles)
