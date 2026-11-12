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


#: A count is the app's to make (visits, saved places), not the model's: these
#: questions get the app tools (:mod:`..tools`) when the run offers them.
COUNT = ("kac ", "kac tane", "kac kere", "kac kez", "kac defa")
#: ...but not opening hours or prices: "kaça kadar açık", "kaç lira".
NOT_COUNT = ("kaca kadar", "kac lira", "kac tl", "kac para", "saat kac")


def wants_count(question: str) -> bool:
    f = fold(question) + " "
    return any(c in f for c in COUNT) and not any(c in f for c in NOT_COUNT)


def gate(question: str) -> str:
    f = fold(question)
    if any(c in f for c in RECALL):
        return "recall"
    if any(c in f for c in RECOMMEND):
        return "recommend"
    return "closed"
