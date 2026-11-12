"""The memeval dataset format, and the checks a dataset must pass.

A dataset is a directory::

    world.json                 restaurants, menu_items, campaigns (shared)
    personas/<id>.json         one user + their app state (memory layer M4)
    histories/<id>.jsonl       that user's past assistant sessions, one per line
    facts_gold/<id>.json       facts consolidation should extract (write path)
    questions.jsonl            the questions, one per line

Field names follow the Drift tables in feast-cur
(``lib/core/data/sources/local/tables/restaurant_tables.dart`` and
``activity_tables.dart``) in snake_case, so the Dart side can load the same
files as fixtures without a mapping layer.  ``working_hours`` uses the
backend's ``{"monday": {"open", "close", "is_open"}, ...}`` object, which is
what ``WorkingHoursDto`` reads.

Validation is hand-written rather than JSON Schema: the project keeps to its
four core dependencies, and most of what matters here is cross-file (does the
evidence turn exist, does an abstention question accidentally have evidence),
which a schema cannot express anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path

from .tools import unknown as unknown_tools

#: LongMemEval's seven question types plus the three the architecture adds.
CATEGORIES = (
    "user_info",         # "Hangi alerjim oldugunu soylemistim?"
    "assistant_info",    # "Gecen hafta onerdigin lahmacuncu hangisiydi?"
    "preference",        # "Bu aksam ne yesem?" - rubric scored
    "multi_session",     # "Son ayda hangi mutfaklari denedim?"
    "knowledge_update",  # vejetaryen -> vegan
    "temporal",          # "Gecen cuma gittigim yer neydi?"
    "abstention",        # "Kedimin mamasini nereden almistim?"
    "safety",            # allergy/diet applied when recommending
    "app_data",          # counting/aggregation routed to SQL tools
    "deleted_info",      # retracted or user-deleted fact must not come back
    # "Burasi iyi mi?" - an opinion only with its source (a post's author, the
    # user's own past words); no ratings or reviews exist to back a verdict.
    "attributed_opinion",
)

#: Target share of the question set per category (§5.1). The validator warns
#: when a dataset of at least ``QUOTA_MIN_SIZE`` questions drifts more than
#: ``QUOTA_TOLERANCE`` from these.
QUOTAS = {
    "user_info": 0.10, "assistant_info": 0.10, "multi_session": 0.10,
    "temporal": 0.10, "preference": 0.12, "knowledge_update": 0.10,
    "abstention": 0.12, "safety": 0.06, "app_data": 0.07, "deleted_info": 0.05,
    "attributed_opinion": 0.08,
}
QUOTA_TOLERANCE = 0.04
QUOTA_MIN_SIZE = 100

#: Known failure modes get their own tagged tests (§5.1 "Bilinen hata modlari").
FAILURE_TAGS = (
    "indirect_evidence",      # stated indirectly, never saved (Coze)
    "consolidation_corrupt",  # compressed into something wrong (ChatGPT)
    "time_false_positive",    # "Cuma Pide" parsed as a date
    # v0.3
    "distractor",             # a friend's allergy, a plan that was called off
    "lookalike",              # "Sushi Nami" visited, "Sushi Nomi" only asked about
    "buried_evidence",        # the evidence sits mid-way through a long session
)

ANSWER_TYPES = ("exact", "rubric", "abstain", "count")

#: EU-14, exactly the keys feast-cur's ``resolveAllergen`` knows.
ALLERGENS = (
    "gluten", "crustaceans", "eggs", "fish", "peanuts", "soybeans", "milk",
    "tree_nuts", "celery", "mustard", "sesame", "sulphur_dioxide", "lupin",
    "molluscs",
)
#: The keys feast-cur's ``resolveBadge`` knows.
BADGES = (
    "halal", "vegan", "vegetarian", "gluten_free", "dairy_free", "organic",
    "sugar_free", "spicy", "extra_spicy", "new", "popular", "chef_special",
)
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
FACT_KINDS = ("hard", "soft")
ROLES = ("user", "assistant")


# ----------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------
@dataclass
class Restaurant:
    id: str
    name: str
    lat: float
    lng: float
    tags: list[str]
    working_hours: dict
    address: str = ""
    #: Not in the Drift table yet (§1.3 gap 2); kept so budget questions can
    #: be written now. ``1``-``4`` = ₺ .. ₺₺₺₺.
    price_level: int = 0
    #: Neighbourhood, also missing from the backend today.
    area: str = ""


@dataclass
class MenuItem:
    id: str
    restaurant_id: str
    name: str
    category_name: str
    price: str
    description: str = ""
    badges: list[str] = field(default_factory=list)
    allergens: list[str] = field(default_factory=list)
    calories: int | None = None
    sold_out: bool = False


@dataclass
class Campaign:
    id: int
    restaurant_id: str
    title: str
    description: str
    start_date: str
    end_date: str


@dataclass
class WorldPost:
    """Another user's public post (Drift ``posts``: ``caption`` is the text)."""

    id: str
    author: str
    restaurant_id: str
    created_at: str
    caption: str


@dataclass
class World:
    restaurants: list[Restaurant]
    menu_items: list[MenuItem]
    campaigns: list[Campaign]
    posts: list[WorldPost] = field(default_factory=list)


@dataclass
class Persona:
    id: str
    name: str
    #: Where "yakinda" is measured from unless a question overrides it.
    home_lat: float
    home_lng: float
    home_area: str = ""
    saved_restaurants: list[dict] = field(default_factory=list)   # restaurant_id, saved_at
    saved_menu_items: list[dict] = field(default_factory=list)    # item_id, restaurant_id, saved_at
    visited_restaurants: list[dict] = field(default_factory=list) # restaurant_id, visited_at
    posts: list[dict] = field(default_factory=list)               # id, restaurant_id, created_at, text
    shared_messages: list[dict] = field(default_factory=list)     # friend, restaurant_id|item_id, sent_at
    friends: list[str] = field(default_factory=list)
    #: Facts the user erased on the "Mustachio ne biliyor?" screen - outside
    #: the chat, so no turn records it. ``key, value, at``. KVKK erasure: the
    #: runner must drop the fact *and* its source turns (§3.4 authority order).
    memory_deletions: list[dict] = field(default_factory=list)


@dataclass
class ChatTurn:
    turn_id: str
    ts: str
    role: str
    text: str


@dataclass
class Session:
    session_id: str
    persona: str
    started_at: str
    turns: list[ChatTurn]
    #: Restaurant / menu ids the assistant showed as cards in this session -
    #: the rule-based recommendation log (M2) that "assistant_info" tests.
    recommended: list[dict] = field(default_factory=list)   # id, turn_id, outcome


@dataclass
class GoldFact:
    key: str
    value: str
    kind: str
    source_turn_id: str
    valid_from: str
    valid_until: str = ""
    #: ``explicit`` when the user said it outright, ``inferred`` otherwise.
    source: str = "explicit"
    #: ``closed`` (superseded, history kept) or ``deleted`` (user erased it,
    #: source turns must vanish too - §3.4 authority order).
    status: str = "open"


@dataclass
class Question:
    qid: str
    persona: str
    category: str
    asked_at: str
    question: str
    gold_answer: str
    answer_type: str
    expected_intent: str = ""
    #: ``[{"tool": "a|b", "args": {...}}]`` in agent/suite.py's loose syntax.
    expected_tool_calls: list[dict] = field(default_factory=list)
    evidence_turn_ids: list[str] = field(default_factory=list)
    should_abstain: bool = False
    #: Allergen/badge keys every recommended item must respect (safety).
    hard_constraints: dict = field(default_factory=dict)
    #: Free text for the judge: "+-1 gun kabul", rubric bullets, ...
    tolerance: str = ""
    failure_tags: list[str] = field(default_factory=list)
    #: Earlier turns of the *same* conversation, for multi-turn questions.
    history: list[dict] = field(default_factory=list)


@dataclass
class Dataset:
    root: Path
    world: World
    personas: dict[str, Persona]
    sessions: dict[str, list[Session]]
    facts: dict[str, list[GoldFact]]
    questions: list[Question]

    def turn_index(self) -> dict[str, tuple[Session, ChatTurn]]:
        return {
            turn.turn_id: (session, turn)
            for sessions in self.sessions.values()
            for session in sessions
            for turn in session.turns
        }


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
class DatasetError(ValueError):
    """A file is missing or not shaped like its record type."""


def _build(cls, data: dict, where: str):
    if not isinstance(data, dict):
        raise DatasetError(f"{where}: expected an object, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise DatasetError(f"{where}: unknown field(s) {sorted(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise DatasetError(f"{where}: {exc}") from None


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetError(f"{path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path}: invalid JSON ({exc})") from None


def _read_jsonl(path: Path) -> list[tuple[int, dict]]:
    rows = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"{path}: {exc}") from None
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append((number, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{number}: invalid JSON ({exc})") from None
    return rows


def load(root: str | Path) -> Dataset:
    root = Path(root)
    raw = _read_json(root / "world.json")
    world = World(
        restaurants=[_build(Restaurant, r, f"world.restaurants[{i}]")
                     for i, r in enumerate(raw.get("restaurants", []))],
        menu_items=[_build(MenuItem, m, f"world.menu_items[{i}]")
                    for i, m in enumerate(raw.get("menu_items", []))],
        campaigns=[_build(Campaign, c, f"world.campaigns[{i}]")
                   for i, c in enumerate(raw.get("campaigns", []))],
        posts=[_build(WorldPost, p, f"world.posts[{i}]")
               for i, p in enumerate(raw.get("posts", []))],
    )

    personas: dict[str, Persona] = {}
    for path in sorted((root / "personas").glob("*.json")):
        persona = _build(Persona, _read_json(path), path.name)
        personas[persona.id] = persona

    sessions: dict[str, list[Session]] = {}
    for path in sorted((root / "histories").glob("*.jsonl")):
        for number, row in _read_jsonl(path):
            where = f"{path.name}:{number}"
            turns = [_build(ChatTurn, t, f"{where}.turns[{i}]")
                     for i, t in enumerate(row.pop("turns", []))]
            session = _build(Session, {**row, "turns": turns}, where)
            sessions.setdefault(session.persona, []).append(session)

    facts: dict[str, list[GoldFact]] = {}
    facts_dir = root / "facts_gold"
    if facts_dir.is_dir():
        for path in sorted(facts_dir.glob("*.json")):
            facts[path.stem] = [_build(GoldFact, f, f"{path.name}[{i}]")
                                for i, f in enumerate(_read_json(path))]

    questions = [_build(Question, row, f"questions.jsonl:{number}")
                 for number, row in _read_jsonl(root / "questions.jsonl")]
    return Dataset(root, world, personas, sessions, facts, questions)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def validate(ds: Dataset) -> Report:
    report = Report()
    err, warn = report.errors.append, report.warnings.append

    # -- world ---------------------------------------------------------
    restaurant_ids = {r.id for r in ds.world.restaurants}
    if len(restaurant_ids) != len(ds.world.restaurants):
        err("world: duplicate restaurant id")
    for r in ds.world.restaurants:
        missing = [d for d in WEEKDAYS if d not in r.working_hours]
        if missing:
            err(f"restaurant {r.id}: working_hours lacks {missing}")
        if not 0 <= r.price_level <= 4:
            err(f"restaurant {r.id}: price_level {r.price_level} outside 0-4")
    item_ids = {m.id for m in ds.world.menu_items}
    if len(item_ids) != len(ds.world.menu_items):
        err("world: duplicate menu item id")
    for m in ds.world.menu_items:
        if m.restaurant_id not in restaurant_ids:
            err(f"menu item {m.id}: unknown restaurant {m.restaurant_id}")
        for key in m.allergens:
            if key not in ALLERGENS:
                err(f"menu item {m.id}: unknown allergen '{key}'")
        for key in m.badges:
            if key not in BADGES:
                err(f"menu item {m.id}: unknown badge '{key}'")
    for c in ds.world.campaigns:
        if c.restaurant_id not in restaurant_ids:
            err(f"campaign {c.id}: unknown restaurant {c.restaurant_id}")
    for p in ds.world.posts:
        if p.restaurant_id not in restaurant_ids:
            err(f"post {p.id}: unknown restaurant {p.restaurant_id}")

    # -- personas: every app-state row points at the world -------------
    for p in ds.personas.values():
        for kind, rows, key in (
            ("saved_restaurants", p.saved_restaurants, "restaurant_id"),
            ("visited_restaurants", p.visited_restaurants, "restaurant_id"),
            ("posts", p.posts, "restaurant_id"),
        ):
            for row in rows:
                if row.get(key) not in restaurant_ids:
                    err(f"persona {p.id}.{kind}: unknown restaurant {row.get(key)}")
        for row in p.saved_menu_items:
            if row.get("item_id") not in item_ids:
                err(f"persona {p.id}.saved_menu_items: unknown item {row.get('item_id')}")
        for row in p.shared_messages:
            if row.get("friend") not in p.friends:
                err(f"persona {p.id}.shared_messages: '{row.get('friend')}' is not a friend")

    # -- sessions ------------------------------------------------------
    turns = ds.turn_index()
    seen: set[str] = set()
    for persona, sessions in ds.sessions.items():
        if persona not in ds.personas:
            err(f"history for unknown persona {persona}")
        for s in sessions:
            start = _ts(s.started_at)
            if start is None:
                err(f"session {s.session_id}: bad started_at '{s.started_at}'")
            previous = start
            for t in s.turns:
                if t.turn_id in seen:
                    err(f"turn id {t.turn_id} used twice")
                seen.add(t.turn_id)
                if t.role not in ROLES:
                    err(f"turn {t.turn_id}: role '{t.role}'")
                stamp = _ts(t.ts)
                if stamp is None:
                    err(f"turn {t.turn_id}: bad ts '{t.ts}'")
                elif previous and stamp < previous:
                    err(f"turn {t.turn_id}: earlier than the turn before it")
                previous = stamp or previous
            for rec in s.recommended:
                if rec.get("id") not in restaurant_ids | item_ids:
                    err(f"session {s.session_id}: recommended unknown id {rec.get('id')}")

    # -- gold facts ----------------------------------------------------
    for persona, facts in ds.facts.items():
        for f in facts:
            if f.kind not in FACT_KINDS:
                err(f"fact {persona}/{f.key}: kind '{f.kind}'")
            if f.source_turn_id not in turns:
                err(f"fact {persona}/{f.key}: unknown source turn {f.source_turn_id}")
            if f.status not in ("open", "closed", "deleted"):
                err(f"fact {persona}/{f.key}: status '{f.status}'")
            if f.status == "closed" and not f.valid_until:
                err(f"fact {persona}/{f.key}: closed but no valid_until")

    # -- questions -----------------------------------------------------
    qids: set[str] = set()
    for q in ds.questions:
        where = f"question {q.qid}"
        if q.qid in qids:
            err(f"{where}: duplicate qid")
        qids.add(q.qid)
        if q.persona not in ds.personas:
            err(f"{where}: unknown persona {q.persona}")
        if q.category not in CATEGORIES:
            err(f"{where}: unknown category '{q.category}'")
        if q.answer_type not in ANSWER_TYPES:
            err(f"{where}: unknown answer_type '{q.answer_type}'")
        for tag in q.failure_tags:
            if tag not in FAILURE_TAGS:
                err(f"{where}: unknown failure tag '{tag}'")
        asked = _ts(q.asked_at)
        if asked is None:
            err(f"{where}: bad asked_at '{q.asked_at}'")

        # Evidence must exist, belong to this persona and predate the question.
        for tid in q.evidence_turn_ids:
            hit = turns.get(tid)
            if hit is None:
                err(f"{where}: unknown evidence turn {tid}")
                continue
            session, turn = hit
            if session.persona != q.persona:
                err(f"{where}: evidence {tid} belongs to {session.persona}")
            stamp = _ts(turn.ts)
            if asked and stamp and stamp >= asked:
                err(f"{where}: evidence {tid} is not before asked_at")

        # Abstention: nothing to find, and it must be scored as such. A false
        # premise about an allergy is still abstention - the right answer is
        # "bana bir alerjinden bahsetmedin", never "alerjin yok".
        abstains = q.should_abstain or q.answer_type == "abstain"
        if abstains and q.evidence_turn_ids:
            err(f"{where}: abstention question has evidence turns")
        if q.category == "abstention" and not abstains:
            err(f"{where}: abstention category but should_abstain is false")
        if not abstains and not q.evidence_turn_ids and q.category not in (
                "app_data", "safety", "attributed_opinion"):
            warn(f"{where}: no evidence turns - retrieval recall cannot be scored")

        for call in q.expected_tool_calls:
            for name in unknown_tools(call.get("tool", "")):
                err(f"{where}: tool '{name}' is not in memeval.tools.ALIASES")
        for key in q.hard_constraints.get("allergens", []):
            if key not in ALLERGENS:
                err(f"{where}: unknown allergen constraint '{key}'")
        if q.category == "safety" and not q.hard_constraints:
            err(f"{where}: safety question without hard_constraints")
        if q.answer_type == "rubric" and not q.tolerance:
            warn(f"{where}: rubric question without rubric text in tolerance")

    # -- coverage ------------------------------------------------------
    total = len(ds.questions)
    counts = {c: sum(q.category == c for q in ds.questions) for c in CATEGORIES}
    for c, n in counts.items():
        if n == 0:
            warn(f"coverage: no '{c}' questions")
    if total >= QUOTA_MIN_SIZE:
        for c, target in QUOTAS.items():
            share = counts[c] / total
            if abs(share - target) > QUOTA_TOLERANCE:
                warn(f"quota: '{c}' is {share:.0%}, target {target:.0%}")
    for tag in FAILURE_TAGS:
        tagged = sum(tag in q.failure_tags for q in ds.questions)
        if total >= QUOTA_MIN_SIZE and tagged < 5:
            warn(f"failure mode '{tag}': {tagged} tests, want at least 5")
    return report


def summary(ds: Dataset) -> str:
    turns = sum(len(s.turns) for ss in ds.sessions.values() for s in ss)
    lines = [
        f"dataset   {ds.root}",
        f"world     {len(ds.world.restaurants)} restaurants, "
        f"{len(ds.world.menu_items)} menu items, {len(ds.world.campaigns)} campaigns",
        f"personas  {len(ds.personas)}  sessions {sum(map(len, ds.sessions.values()))}  "
        f"turns {turns}  gold facts {sum(map(len, ds.facts.values()))}",
        f"questions {len(ds.questions)}",
    ]
    for c in CATEGORIES:
        n = sum(q.category == c for q in ds.questions)
        lines.append(f"  {c:<17} {n:>4}")
    return "\n".join(lines)
