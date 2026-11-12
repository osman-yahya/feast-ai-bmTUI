"""What a recommendation must respect, from memory, applied by code.

Allergy and diet are written by rule, not by the SLM (:func:`hard_ops`).
v0.3, today's stores: the SLM's open allergy+diet set was right for 6 of 16
personas - a friend's allergy filed as the user's ("Zeynep'in yer fıstığı
alerjisi var, benim değil"), an allergy told without the word missed
("simitli bir şey yedikten sonra boğazım kaşındı"), "et, süt, yumurta
tüketmiyorum" taken for a milk allergy.  The rule gets 16 of 16 (written on
half the personas, 7/8 on the other half before its last fixes) - on data
this project generated, so expect less on real users.

A friend's allergy is read at ask time from the user's turns that name the
friend and say "alerji".  The menu and restaurant tools (:mod:`..tools`)
filter with all of it: a 2B model asked to avoid peanuts still recommends
the brownie.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...agent.text import fold
from ..gen.world import CUISINES
from .consolidate import ALLERGEN_CUES, canonical
from .store import MemoryStore

#: The body reacting: an allergy told without the word.
REACTION = re.compile(r"(?:alerj\w*|nefes\w* daral\w*|nefes alam\w*|ayni etki\w*|bogaz\w* (?:sis|kasin)\w*|"
                      r"dudak\w* sis\w*|dil\w* sis\w*|acile? \w*|hastanede|kasin\w*|kramp\w*|"
                      r"iyi gelmiyor|dokunuyor)")
#: Foods the cue table lacks, with the allergen they carry.
FOOD_CUES = ((("simit", "susam"), "sesame"), (("sutlu", "sutlac"), "milk"),
             (("ekmek", "makarna", "bugday"), "gluten"))
GONE = re.compile(r"\b(?:gecti|gecmis|kalmadi)\b")
#: Someone else's: "benim değil, onun".
NOT_MINE = re.compile(r"\b(?:benim degil|bende yok|benim boyle bir sorunum yok|onun)\b")
VEGAN = re.compile(r"\b(?:vegan\w*|hayvansal hicbir)")
VEGAN_PLACE = re.compile(r"\bvegan (?:kafe\w*|cafe\w*)|filiz vegan")
VEGETARIAN = re.compile(r"\b(?:vejetaryen\w*|vejeteryen\w*|et yemiyorum|et yemedigim|etsiz|et yok bende)")


def _cue(part: str, cues) -> bool:
    return any(re.search(r"\b" + re.escape(c), part) for c in cues)


def _allergens(folded: str) -> list[str]:
    """One allergen per sentence, the cue table first: "ceviz fıstık" is tree nuts."""
    keys = []
    for part in re.split(r"[.;!?]", folded):
        key = (next((k for cues, k in ALLERGEN_CUES if _cue(part, cues)), None)
               or next((k for cues, k in FOOD_CUES if _cue(part, cues)), None))
        if key and key not in keys:
            keys.append(key)
    return keys


def hard_ops(text: str) -> list[tuple[str, str, str]]:
    """``(op, key, value)`` for the allergy and diet a user turn states about the user."""
    f = fold(text)
    if NOT_MINE.search(f):
        return []
    ops = []
    if REACTION.search(f):
        op = "remove" if "alerj" in f and GONE.search(f) else "add"
        ops += [(op, "allergen", k) for k in _allergens(f)]
    if VEGAN.search(f) and not VEGAN_PLACE.search(f):
        ops.append(("add", "diet", "vegan"))
    elif VEGETARIAN.search(f):
        ops.append(("add", "diet", "vegetarian"))
    return ops

ALLERGEN_TR = {"peanuts": "yer fıstığı", "tree_nuts": "kuruyemiş", "crustaceans": "kabuklu deniz ürünü",
               "molluscs": "yumuşakça", "milk": "süt", "gluten": "gluten", "fish": "balık", "eggs": "yumurta",
               "sesame": "susam", "soybeans": "soya", "celery": "kereviz", "mustard": "hardal"}


@dataclass(frozen=True)
class Constraints:
    #: allergen key -> whose it is: "you" or the friend's name
    allergens: dict[str, str] = field(default_factory=dict)
    diet: str = ""                          # vegan | vegetarian | ""
    likes: tuple[str, ...] = ()             # cuisine labels

    def allows(self, item) -> bool:
        if set(item.allergens) & set(self.allergens) or item.sold_out:
            return False
        if self.diet == "vegan":
            return "vegan" in item.badges
        if self.diet == "vegetarian":
            return bool({"vegan", "vegetarian"} & set(item.badges))
        return True

    def why_not(self, item) -> str:
        hit = sorted(set(item.allergens) & set(self.allergens))
        if hit:
            return "contains " + ", ".join(hit)
        return f"not {self.diet}" if self.diet else "sold out"

    def label(self) -> str:
        parts = [f"no {k} ({who})" for k, who in sorted(self.allergens.items())]
        if self.diet:
            parts.append(f"{self.diet} (you)")
        return ", ".join(parts) or "none known"


def friend_allergens(store: MemoryStore, friends: list[str]) -> dict[str, str]:
    """``{allergen key: friend}`` from the user's turns: "Aslı'yla ..., onun gluten alerjisi var"."""
    out: dict[str, str] = {}
    for e in store.episodes:
        if e.role != "user":
            continue
        f = fold(e.text)
        if "alerji" not in f:
            continue
        for name in friends:
            if re.search(r"\b" + re.escape(fold(name)), f):
                key = canonical("allergen", f)
                if key:
                    out[key] = name
    return out


def constraints(store: MemoryStore, question: str, friends: list[str]) -> Constraints:
    """The user's open facts, plus the allergies of the friends ``question`` names."""
    allergens, diet, likes = {}, "", []
    for fact in store.open_facts():
        if fact.key == "allergen":
            allergens[fact.value] = "you"
        elif fact.key == "diet":
            diet = fact.value
        elif fact.key == "likes":
            likes.append(fact.value)
    f = fold(question)
    named = [n for n in friends if re.search(r"\b" + re.escape(fold(n)), f)]
    for key, who in friend_allergens(store, named).items():
        allergens.setdefault(key, who)
    labels = {c["label"] for c in CUISINES.values()}
    return Constraints(allergens, diet, tuple(x for x in likes if x in labels))
