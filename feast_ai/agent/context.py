"""The phone-side state the agent prompt's CONTEXT block is built from.

In the app these values come from the device - the clock, GPS, the active
search location, local caches.  Here they are a small JSON file,
``~/.feast-ai/agent_context.json``, created with plausible Istanbul defaults
the first time it is needed and editable by hand.  The values change the
prompt's length, so the report records the context a run used.

Placement matters as much as content: the block is rendered *last* in the
system prompt.  llama.cpp reuses the KV cache for the longest unchanged
prefix, and ``now`` changes every minute - anything after it is re-prefilled
on every message.  Static tools first, volatile context last.

The tag list is a placeholder until it is read from the restaurant backend's
real tags; edit it in the JSON file.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields

from ..paths import AGENT_CONTEXT_FILE

DEFAULT_TAGS = (
    "kahvalti", "kafe", "kebap", "pide", "lahmacun", "doner", "burger", "pizza",
    "deniz urunleri", "meyhane", "tatli", "kahve", "vegan", "glutensiz",
    "ev yemekleri", "sokak lezzetleri",
)


@dataclass
class AgentContext:
    lat: float = 40.9876
    lng: float = 29.0257
    label: str = "Moda, Kadıköy, İstanbul"
    is_gps: bool = True
    cached_restaurants: int = 48
    posts: int = 120
    saved: int = 17
    friends: int = 23
    tags: tuple[str, ...] = field(default=DEFAULT_TAGS)
    #: Frozen clock for reproducible benchmarks; empty = the live clock.
    now: str = ""

    # ------------------------------------------------------------------
    def clock(self) -> str:
        """Minute resolution: finer would bust the prompt cache every second."""
        return self.now or time.strftime("%Y-%m-%d %a %H:%M")

    def frozen(self) -> "AgentContext":
        """A copy whose clock no longer moves - one prompt per benchmark run."""
        copy = AgentContext(**{f.name: getattr(self, f.name) for f in fields(self)})
        copy.now = self.clock()
        return copy

    def render(self) -> str:
        return (
            "CONTEXT\n"
            f"now: {self.clock()} | user_gps: {self.lat:.4f},{self.lng:.4f} | "
            f"active_location: {self.label} | is_gps: {str(self.is_gps).lower()}\n"
            f"cached_restaurants: {self.cached_restaurants} | posts: {self.posts} | "
            f"saved: {self.saved} | friends: {self.friends}\n"
            "Never call a tool to learn the time, the location or the active "
            "location — they are above."
        )

    # ------------------------------------------------------------------
    @classmethod
    def load(cls) -> "AgentContext":
        try:
            data = json.loads(AGENT_CONTEXT_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            context = cls()
            context.save()
            return context
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        clean["tags"] = tuple(clean.get("tags") or DEFAULT_TAGS)
        try:
            return cls(**clean)
        except TypeError:
            return cls()

    def save(self) -> None:
        data = asdict(self)
        data["tags"] = list(self.tags)
        data.pop("now", None)  # the live clock is never persisted
        try:
            AGENT_CONTEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
            AGENT_CONTEXT_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass
