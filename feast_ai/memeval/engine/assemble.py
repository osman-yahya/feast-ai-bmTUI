"""The MEMORY block: handle'd lines, facts before turns, within a budget.

Facts are few and dense, so they go first and whole; retrieved turns fill
what is left in rank order, then are shown oldest first so the model reads
a timeline.  Every line has a handle (``f1``, ``t3``) the reply must cite,
and ``handles`` maps it back to a dataset turn id for scoring.
"""

from __future__ import annotations

from datetime import date

from ..gen.plan import ALLERGEN_TR, DIET_TR
from .store import Episode, Fact
from .timeparse import DAY_EN

ENGINE_RULES = (
    "MEMORY\n"
    "What you remember about this user.\n"
    "f lines are facts: handle | key | value | valid dates (an end date means it no longer "
    "holds; allergen and diet are hard constraints).\n"
    "t lines are past chats, oldest first: handle | date | who | text; a date in (=...) is "
    "the day that phrase meant.\n"
    "Use them only for questions about the user's own past or to respect their constraints, "
    "and cite the handle you rely on, e.g. (f1) or (t3). If the answer is not in MEMORY, say "
    "the user never told you - never guess. Not finding an allergy here does not mean the "
    "user has none: say they have not mentioned one."
)
#: Rough Turkish chars per Gemma token, for the greedy fill; the runner measures exactly.
CHARS_PER_TOKEN = 3.0


def _value(f: Fact) -> str:
    if f.key == "allergen":
        return ALLERGEN_TR.get(f.value, f.value)
    if f.key == "diet":
        return DIET_TR.get(f.value, f.value)
    return f.value


def _stamp(ts: str) -> str:
    d = date.fromisoformat(ts[:10])
    return f"{d:%Y-%m-%d} {DAY_EN[d.weekday()]}"


def fact_line(h: str, f: Fact) -> str:
    until = f"until {f.valid_until} (no longer)" if f.valid_until else "now"
    return f"{h} | {f.key} | {_value(f)} | {f.valid_from} -> {until}"


def assemble(facts: list[Fact], ranked: list[Episode], budget_tokens: int) -> tuple[str, dict]:
    budget = budget_tokens * CHARS_PER_TOKEN
    handles: dict[str, str] = {}
    lines = []
    for n, f in enumerate(facts, 1):
        lines.append(fact_line(f"f{n}", f))
        handles[f"f{n}"] = f.source_turn_id
    used = sum(len(x) + 1 for x in lines)
    chosen = []
    for e in ranked:
        cost = len(e.text) + 24
        if used + cost > budget:
            continue
        chosen.append(e)
        used += cost
    for n, e in enumerate(sorted(chosen, key=lambda e: e.ts), 1):
        lines.append(f"t{n} | {_stamp(e.ts)} | {e.role} | {e.text}")
        handles[f"t{n}"] = e.turn_id
    return ENGINE_RULES + "\n" + ("\n".join(lines) if lines else "(empty)"), handles
