"""Turkish relative time expressions -> absolute date windows, by rule.

The 2B model copies "dün" out of a turn instead of dating it (smoke run,
p_elif_q07); the architecture's answer is that dates are code's job.  Every
expression resolves against the timestamp of the turn (or question) it is in.

Restaurant names that contain a time word ("Cuma Pide", "Dün Gece Burger")
are masked first - the ``time_false_positive`` failure tag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from ...agent.text import fold

DAYS = ("pazartesi", "sali", "carsamba", "persembe", "cuma", "cumartesi", "pazar")
MONTHS = ("ocak", "subat", "mart", "nisan", "mayis", "haziran", "temmuz",
          "agustos", "eylul", "ekim", "kasim", "aralik")
NUM = {"bir": 1, "iki": 2, "uc": 3, "dort": 4, "bes": 5, "alti": 6, "yedi": 7,
       "sekiz": 8, "dokuz": 9, "on": 10}
_N = r"(\d+|" + "|".join(NUM) + r")"
_DAY = "(" + "|".join(DAYS) + ")"
_MONTH = "(" + "|".join(MONTHS) + r")(?:\s+ay\w*|da|de|ta|te|inda|inde|unda|unde)?"
PATTERN = re.compile(
    r"\b(?:"
    r"(?P<rel>dun|bugun|evvelsi gun|onceki gun)"
    r"|gecen " + _DAY.replace("(", "(?P<lastday>", 1) +
    r"|bu " + _DAY.replace("(", "(?P<thisday>", 1) +
    r"|" + _DAY.replace("(", "(?P<dayof>", 1) + r" (?:gunu|aksami|ogle\w*|sabahi)"
    r"|(?P<ago>" + _N[1:-1] + r") (?P<agounit>gun|hafta|ay) (?:once|evvel)"
    r"|gecen (?P<lastunit>hafta|ay)\w*"
    r"|son (?P<span>" + _N[1:-1] + r") (?P<spanunit>gun|hafta|ay)\w*"
    r"|bu (?P<this>aksam|ogle\w*|sabah|hafta|ay)"
    r"|" + _MONTH.replace("(", "(?P<month>", 1) +
    r")\b")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    first: date
    last: date
    #: The words can only mean a day gone by ("dün", "geçen cuma"), unlike "cuma günü".
    past: bool = False

    def label(self) -> str:
        if self.first == self.last:
            return f"{self.first:%Y-%m-%d} {DAY_EN[self.first.weekday()]}"
        return f"{self.first:%Y-%m-%d}..{self.last:%Y-%m-%d}"


DAY_EN = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _n(word: str) -> int:
    return int(word) if word.isdigit() else NUM.get(word, 0)


def _month_window(month: int, ref: date) -> tuple[date, date]:
    year = ref.year if month <= ref.month else ref.year - 1
    first = date(year, month, 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return first, nxt - timedelta(days=1)


def _months_back(day: date, n: int) -> date:
    m = day.month - 1 - n
    year, month = day.year + m // 12, m % 12 + 1
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return date(year, month, min(day.day, (nxt - timedelta(days=1)).day))


def _mask(folded: str, names: tuple[str, ...]) -> str:
    for name in names:
        n = fold(name)
        folded = folded.replace(n, "#" * len(n))
    return folded


def resolve(text: str, ref: datetime | date, names: tuple[str, ...] = ()) -> list[Span]:
    """Every time expression in ``text`` as a date window, relative to ``ref``."""
    today = ref.date() if isinstance(ref, datetime) else ref
    folded = _mask(fold(text), names)
    out: list[Span] = []
    for m in PATTERN.finditer(folded):
        g = m.groupdict()
        first = last = None
        if g["rel"]:
            back = {"dun": 1, "bugun": 0}.get(g["rel"], 2)
            first = last = today - timedelta(days=back)
        elif g["lastday"] or g["dayof"]:
            wd = DAYS.index(g["lastday"] or g["dayof"])
            back = (today.weekday() - wd) % 7
            if g["lastday"] and not back:            # "geçen pazar" on a Sunday: a week ago
                back = 7
            first = last = today - timedelta(days=back)
        elif g["thisday"]:                           # "bu perşembe": this week's, past or coming
            first = last = today - timedelta(days=today.weekday() - DAYS.index(g["thisday"]))
        elif g["ago"]:
            n, unit = _n(g["ago"]), g["agounit"]
            if unit == "gun":
                first = last = today - timedelta(days=n)
            elif unit == "hafta":
                mid = today - timedelta(days=7 * n)
                first, last = mid - timedelta(days=3), mid + timedelta(days=3)
            else:
                first, last = _month_window((today.month - n - 1) % 12 + 1, today)
        elif g["lastunit"] == "hafta":
            monday = today - timedelta(days=today.weekday() + 7)
            first, last = monday, monday + timedelta(days=6)
        elif g["lastunit"] == "ay":
            first, last = _month_window((today.month - 2) % 12 + 1, today)
        elif g["span"]:
            n, unit = _n(g["span"]), g["spanunit"]
            if unit == "ay":                         # calendar months: "son iki ay" on 28 Oct is from 28 Aug
                first, last = _months_back(today, n), today
            else:
                first, last = today - timedelta(days=n * {"gun": 1, "hafta": 7}[unit]), today
        elif g["this"]:
            if g["this"] == "hafta":
                first, last = today - timedelta(days=today.weekday()), today
            elif g["this"] == "ay":
                first, last = today.replace(day=1), today
            else:
                first = last = today
        elif g["month"]:
            first, last = _month_window(MONTHS.index(g["month"]) + 1, today)
        if first:
            told = bool(g["rel"] and g["rel"] != "bugun" or g["lastday"] or g["ago"] or g["thisday"])
            out.append(Span(m.start(), m.end(), first, last, told and last < today))
    return out


def annotate(text: str, ref: datetime | date, names: tuple[str, ...] = ()) -> str:
    """``text`` with each expression followed by its date: "dün (=2026-10-15 Thu)"."""
    spans = resolve(text, ref, names)
    if not spans:
        return text
    if len(fold(text)) != len(text):          # folding changed length: append instead
        return text + " (" + "; ".join(s.label() for s in spans) + ")"
    for s in reversed(spans):
        text = text[:s.end] + f" (={s.label()})" + text[s.end:]
    return text
