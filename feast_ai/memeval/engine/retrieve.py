"""Which past turns answer a question: Turkish BM25 plus time and recency.

score = α·bm25 (normalized) + β·time match + γ·recency + δ·user turn

Time match is the point of resolving dates on write: "2 hafta önce öğlen"
in the question and "dün (=2026-10-16)" in a turn meet as date windows, not
as words.  Tokens are folded and cut to five letters - Turkish suffixes
("meyhanesine", "meyhanede") collapse onto one stem.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date, datetime

from ...agent.text import fold
from .store import Episode
from .timeparse import resolve

ALPHA, BETA, GAMMA, DELTA = 1.0, 0.8, 0.15, 0.05
STOP = {"ve", "bir", "bu", "da", "de", "mi", "mu", "ben", "sen", "bana", "beni", "icin", "cok",
        "ama", "gibi", "ne", "ile", "o", "su", "ya", "ki", "en", "hic", "daha", "var", "yok",
        "hatir", "hatirl", "musun", "misin", "nerede", "hangi", "zaman", "once", "gecen",
        "soyle", "anlat", "sana", "olan", "biz"}
WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    out = []
    for w in WORD.findall(fold(re.sub(r"\(=[^)]*\)", " ", text))):
        s = w[:5]
        if len(w) > 1 and w not in STOP and s not in STOP:
            out.append(s)
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.docs, self.k1, self.b = docs, k1, b
        self.avg = sum(map(len, docs)) / len(docs) if docs else 1.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query: list[str], i: int) -> float:
        d = self.docs[i]
        tf = Counter(d)
        s = 0.0
        for t in set(query):
            if t in tf:
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
        return s


def _days(e: Episode) -> list[tuple[date, date]]:
    out = [tuple(date.fromisoformat(x) for x in w.split("..")) for w in e.dates]
    return out or [(date.fromisoformat(e.ts[:10]),) * 2]


def retrieve(episodes: list[Episode], question: str, asked_at: str, k: int = 8,
             names: tuple[str, ...] = ()) -> list[Episode]:
    """Top-``k`` episodes for ``question``, best first."""
    if not episodes:
        return []
    asked = datetime.fromisoformat(asked_at)
    windows = [(s.first, s.last) for s in resolve(question, asked, names)]
    bm = BM25([tokens(e.text) for e in episodes])
    q = tokens(question)
    raw = [bm.score(q, i) for i in range(len(episodes))]
    top = max(raw) or 1.0
    scored = []
    for i, e in enumerate(episodes):
        age = (asked.date() - date.fromisoformat(e.ts[:10])).days
        hit = any(a <= wl and wf <= b for a, b in _days(e) for wf, wl in windows)
        s = (ALPHA * raw[i] / top + BETA * hit + GAMMA * math.exp(-max(age, 0) / 30)
             + DELTA * (e.role == "user"))
        scored.append((s, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [episodes[i] for _, i in scored[:k]]
