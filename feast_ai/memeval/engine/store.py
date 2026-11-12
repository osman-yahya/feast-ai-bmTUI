"""What the engine remembers about one user: facts with history, raw episodes.

Mirrors the Drift tables the Dart port will add (``assistant_facts``,
``assistant_episodes``); JSON on disk here so a store can be read by eye.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Keys whose value is safety-relevant: never decayed, always offered before a recommendation.
HARD = ("allergen", "diet")
#: Keys with one value at a time: a new value closes the old one (UPDATE).
SINGLE = ("diet", "home_area")


@dataclass
class Fact:
    fid: str
    key: str
    value: str
    kind: str                      # hard | soft
    valid_from: str                # YYYY-MM-DD
    source_turn_id: str
    valid_until: str = ""
    status: str = "open"           # open | closed | deleted


@dataclass
class Episode:
    turn_id: str
    ts: str
    role: str
    text: str                      # with resolved dates: "dün (=2026-10-15 Thu)"
    dates: list[str] = field(default_factory=list)   # event days the text refers to
    event_day: str = ""            # the one earlier day the turn tells about, if it does


@dataclass
class MemoryStore:
    persona: str
    facts: list[Fact] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    #: Every write-path decision, for debugging consolidation: session, raw SLM output, ops.
    log: list[dict] = field(default_factory=list)
    #: Visits the user told about, written by rule (:mod:`.visits`): restaurant, day, turn.
    visits: list[dict] = field(default_factory=list)
    #: What code derived the episodes and visits (consolidate.STORE_REV); 0 = before visits.
    rev: int = 0

    def open_facts(self) -> list[Fact]:
        return [f for f in self.facts if f.status == "open"]

    def visible_facts(self) -> list[Fact]:
        """Open and superseded facts; erased ones never come back (KVKK)."""
        return [f for f in self.facts if f.status != "deleted"]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MemoryStore":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(persona=data["persona"], facts=[Fact(**f) for f in data["facts"]],
                   episodes=[Episode(**e) for e in data["episodes"]], log=data.get("log", []),
                   visits=data.get("visits", []), rev=data.get("rev", 0))
