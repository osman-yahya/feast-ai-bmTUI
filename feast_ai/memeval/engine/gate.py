"""The intervention gate: does this question get memory at all, and which.

``recall``     the user asks about their own past -> retrieved memory
``recommend``  the user wants something to eat -> the profile facts, always
               (§3.5: allergy and diet go in before any recommendation)
``closed``     anything else -> an empty MEMORY block, a shorter prompt

First version is cue words; the router (agent/router.py) replaces it once
the memory intents are in its exemplars.
"""

from __future__ import annotations

import re

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


#: "Hangi mutfakları denemiştim": the visit log answers, not eight turns.
TRIED = ("hangi mutfak", "kac farkli mutfak")
#: The assistant's own recommendations: "bana önerdiğin", "önerdin".
RECOMMENDED = ("onerdin", "onerdig", "onermistin")
#: Other people's words about a place - the app's posts - not the user's ("ne demiştim").
POSTS = re.compile(r"\b(?:nasil bir yer|ne yazilmis|ne yazmis|iyi mi|yorum)")
#: "...nereden almıştım": the gate's "ner" cue fires, but the user asks about the past.
PAST = re.compile(r"\w(?:mis|mus)t[iu]m\b")


def tools_for(question: str, decision: str, names: tuple[str, ...]) -> tuple[str, ...]:
    """The app tools a question is offered (:mod:`..tools`); none for most."""
    f = fold(question) + " "
    if wants_count(question):
        return ("my_visits", "search_saved")
    if any(c in f for c in RECOMMENDED):
        return ("my_recommendations",)
    if any(c in f for c in TRIED):
        return ("my_visits",)
    if POSTS.search(f) and "demistim" not in f:
        return ("search_posts",)
    if decision == "recommend" and not PAST.search(f):
        named = any(fold(n) in f for n in names)
        return ("get_menu",) if named else ("search_restaurants", "get_menu")
    return ()


def retriever_for(question: str, asked_at: str, names: tuple[str, ...], decision: str) -> str:
    """``bm25`` when the words carry the match, else ``rrf``.  v0.3, both runs'
    MEMORY: a restaurant name or a time expression in the question, or a
    recommendation (allergy and diet turns), and BM25 finds as much or more
    (recommend 32% vs 23%); questions with neither, rrf 87% vs 67%."""
    from datetime import datetime

    from .timeparse import resolve

    f = fold(question)
    if decision == "recommend" or any(fold(n) in f for n in names):
        return "bm25"
    return "bm25" if resolve(question, datetime.fromisoformat(asked_at), names) else "rrf"


def gate(question: str) -> str:
    f = fold(question)
    if any(c in f for c in RECALL):
        return "recall"
    if any(c in f for c in RECOMMEND):
        return "recommend"
    return "closed"
