"""The intervention gate: does this question get memory at all, and which.

``recall``     the user asks about their own past -> retrieved memory
``recommend``  the user wants something to eat -> the profile facts, always
               (§3.5: allergy and diet go in before any recommendation)
``closed``     anything else -> an empty MEMORY block, a shorter prompt

First version is cue words; the router (agent/router.py) replaces it once
the memory intents are in its exemplars.
"""

from __future__ import annotations

from ...agent.text import fold

RECALL = ("hatirl", "soyle", "anlatm", "bahset", "onerdig", "gecen", "once", "en son",
          "ne zaman", "gitmis", "gittim", "yemis", "yedim", "denedi", "alerj", "beslen",
          "oturuyor", "sevdi", "biliyor mu", "kac ")
RECOMMEND = ("ne yesem", "ner", "oner", "tavsiye", "ne icsem", "acikt", "canim", "ac misin")


def gate(question: str) -> str:
    f = fold(question)
    if any(c in f for c in RECALL):
        return "recall"
    if any(c in f for c in RECOMMEND):
        return "recommend"
    return "closed"
