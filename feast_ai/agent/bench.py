"""Benchmark the two agent pipelines on one model and one device budget.

``python -m feast_ai bench <model-id>`` runs every turn of
:mod:`feast_ai.agent.suite` through ``direct`` (every tool, every message)
and ``hierarchical`` (router first), on the same model, under the same CPU
pinning and RAM ceiling as a chat session, and writes a side-by-side report
to ``~/.feast-ai/sessions/bench-*.md`` / ``.json``.

What keeps the comparison fair
------------------------------
* **Same text, one variable.**  Both pipelines share the preamble, call
  format, rules and context; only the TOOLS block differs
  (:mod:`feast_ai.agent.prompts`).  The context clock is frozen for the run so
  every prompt is byte-stable.
* **Greedy decoding** by default (temperature 0): a difference between the
  pipelines is then caused by the prompt, not by sampling noise.
* **Direct runs first, before the router exists**, so its RSS does not carry
  an idle encoder; the router is loaded (and the RAM budget split) only for
  the hierarchical pass, then unloaded.
* **One unrecorded warm-up per pipeline**, so neither pays for paging the
  weights in.  With the prompt cache on (the default, as an app would run),
  ``direct`` keeps its one long prompt cached across messages - that is a real
  advantage and it is reported, not hidden: every row records prompt, cached
  and actually-prefilled tokens.  ``--no-cache`` measures the cold case.
* **Shuffled with a fixed seed**, so the hierarchical pipeline does not get
  long same-intent runs that would flatter its cache hit rate.
* **Latency includes routing** - the localhost hop too, which a phone would
  not pay - so the router is never flattered either.

``--router-only`` skips the SLM and measures the router alone: accuracy per
intent, confusions, latency, and a threshold sweep for calibrating
``router_threshold``.

Pipelines (``--modes``)
-----------------------
``direct``        every tool, the SLM every time.
``hierarchical``  router first; it may answer alone, else a narrowed prompt.
``routed``        the router only picks the system prompt; the SLM always
                  answers.  Each intent switch re-prefills the new prompt.
``contexts``      ``routed`` plus a precomputed KV snapshot per intent prompt,
                  restored before each message - what a phone would keep on
                  flash.  Only CONTEXT, history and the message are prefilled.
                  Snapshots are built once before the measured pass (reported
                  separately); a tool set first seen mid-run is built on the
                  spot and that message pays for it.
``active``        no router. One fixed prompt names every tool with one line
                  each; the model answers ``<use>tool</use>`` (generation stops
                  at the tag) or in plain text. For a tool, that tool's manual
                  is appended as a system turn and the model writes the call;
                  the manual is then dropped from the history. Two passes,
                  both on the cached conversation - only the manual is new.
``classes``       the router picks one of the 13 action classes
                  (:mod:`feast_ai.agent.classes`); the class prompt carries
                  its tools' manuals and its own rules, restored from a KV
                  snapshot only when the class changes - a same-class turn
                  keeps the cached history instead. The SLM always answers.

A router mode may name its router: ``classes@decision-2.0-kai-0.6b`` runs that
mode with Decision 2.0 whatever ``--router`` says, so one run can compare
both routers on one loaded model.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Settings
from ..engine.client import ChatClient, Message, ThinkFilter
from ..paths import SESSIONS_DIR, human_bytes
from . import classes as v2
from .calls import HALLUCINATIONS, CallCheck, check_reply, extract, render_call
from .context import AgentContext
from .intents import INTENTS
from .prompts import (SELECT_STOP, class_static, full_prompt, index_prompt, manual_message,
                      prompt_for, static_part)
from .suite import Turn, ordered, score
from .tools import ALL_TOOL_NAMES, BY_NAME

SEED = 7


@dataclass
class Row:
    mode: str
    case: str
    turn: int
    text: str
    expected_intent: str
    expected_tool: str
    route_intent: str = ""
    route_score: float = 0.0
    route_margin: float = 0.0
    route_kind: str = ""
    route_ok: bool = False
    tools_offered: int = 0
    offered: list[str] = field(default_factory=list)
    served_by: str = "slm"  # "slm" | "router"
    call: str = ""
    call_args: dict = field(default_factory=dict)
    status: str = ""
    format: str = ""
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tool_ok: bool = False
    args_ok: bool = False
    exact: bool = False
    forbidden: bool = False
    skipped_tool: bool = False
    needless_call: bool = False
    missing_args: list[str] = field(default_factory=list)
    restore_ms: float = 0.0
    snapshot: str = ""  # contexts: "hit" | "built"; classes: "restored" | "built" | "cached"
    prompt_name: str = ""     # key into the run's ``prompts``
    gold_class: str = ""      # classes: the suite label in class ids
    passes: int = 1           # active: 2 when a manual was fetched
    select_tool: str = ""     # active: the tool named in the first pass
    select_text: str = ""     # active: the first pass, verbatim
    select_ms: float = 0.0    # active: first pass wall time
    prompt_ms: float = 0.0
    decode_ms: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    prefilled_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: float = 0.0
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    latency_s: float = 0.0
    router_ms: float = 0.0
    encoder_ms: float = 0.0
    reply: str = ""
    error: str = ""

    @property
    def hallucinated(self) -> bool:
        return self.status in HALLUCINATIONS or self.forbidden


def _judge(row: Row, turn: Turn, check: CallCheck, reply: str) -> None:
    result = score(turn, check)
    row.call, row.status, row.format = check.name, check.status, check.format
    row.call_args = dict(check.args)
    row.issues, row.warnings = list(check.issues), list(check.warnings)
    row.tool_ok, row.args_ok, row.exact = result.tool_ok, result.args_ok, result.exact
    row.forbidden, row.skipped_tool = result.forbidden, result.skipped_tool
    row.needless_call, row.missing_args = result.needless_call, result.missing_args
    row.reply = reply[:400]


# --------------------------------------------------------------------------
async def _generate(client: ChatClient, messages: list[dict], started: float,
                    stop: tuple[str, ...] = ()) -> tuple[str, dict]:
    """Stream one completion; return the visible reply and its measurements."""
    think = ThinkFilter()
    parts: list[str] = []
    first = 0.0
    usage: dict = {}
    timings: dict = {}
    error = ""
    async for event in client.stream(messages, extra={"stop": list(stop)} if stop else None):
        if event.kind == "error":
            error = event.error
            break
        if event.kind == "done":
            usage, timings = event.usage, event.timings
            break
        if not first:
            first = time.perf_counter()
        if event.kind == "token":
            visible, _ = think.feed(event.text)
            parts.append(visible)
    parts.append(think.flush())
    finished = time.perf_counter()
    stats = {
        "error": error,
        "ttft_s": (first - started) if first else 0.0,
        "latency_s": finished - started,
        "prompt_tokens": int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0),
        "cached_tokens": int(timings.get("cache_n") or 0),
        "prefilled_tokens": int(timings.get("prompt_n") or usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or timings.get("predicted_n") or 0),
        "prefill_tps": float(timings.get("prompt_per_second") or 0.0),
        "decode_tps": float(timings.get("predicted_per_second") or 0.0),
        "prompt_ms": float(timings.get("prompt_ms") or 0.0),
        "decode_ms": float(timings.get("predicted_ms") or 0.0),
    }
    return "".join(parts).strip(), stats


def _combine(passes: list[dict]) -> dict:
    """Two passes as one turn: token counts add up, rates are re-derived from
    the summed times (never an average of rates), timing is the last pass's
    measured from the turn's start - when the answer the app acts on begins."""
    out = dict(passes[-1])
    for key in ("prompt_tokens", "cached_tokens", "prefilled_tokens", "completion_tokens",
                "prompt_ms", "decode_ms"):
        out[key] = sum(p[key] for p in passes)
    out["prefill_tps"] = (1000.0 * out["prefilled_tokens"] / out["prompt_ms"]) if out["prompt_ms"] else 0.0
    out["decode_tps"] = (1000.0 * out["completion_tokens"] / out["decode_ms"]) if out["decode_ms"] else 0.0
    out["error"] = next((p["error"] for p in passes if p["error"]), "")
    return out


_USE = re.compile(r"<use>\s*([A-Za-z_][\w.]*)")
_NAMED = re.compile(r'"name"\s*:\s*"([A-Za-z_][\w.]*)"')
_TAG = re.compile(r"<(?:tool_call>\s*)?([a-z][a-z0-9_]*)")


def _selected(text: str) -> str:
    """The tool the first ``active`` pass named, or "" for a plain answer.

    ``<use>name`` is the asked-for form; a full call or a bare tool name in
    the model's own habit names a tool just as clearly.
    """
    match = _USE.search(text) or _NAMED.search(text)
    if match:
        return match.group(1).removeprefix("functions.")
    # "<send_message_to_friends>..." or "<tool_call>search_restaurants": a
    # real tool's name, written as a tag or as a bare call body.
    tagged = next((m for m in _TAG.findall(text) if m in BY_NAME), "")
    return tagged or extract(text).name


async def _active_turn(row: Row, turn: Turn, history: list[Message], context: AgentContext,
                       client: ChatClient, prompts: dict, started: float) -> str:
    """Pick a tool by name, then write the call with its manual. Returns the
    reply that enters the history - the manual never does."""
    system = index_prompt(context)
    prompts.setdefault("active", system)
    row.prompt_name = "active"
    row.offered = list(ALL_TOOL_NAMES)
    row.tools_offered = len(ALL_TOOL_NAMES)
    messages = client.build_messages(history, turn.text, system_prompt=system)
    first, a = await _generate(client, messages, started, stop=SELECT_STOP)
    row.select_text, row.select_ms = first[:300], a["latency_s"] * 1000.0
    tool = _selected(first) if not a["error"] else ""
    if not tool or tool not in BY_NAME:
        for key, value in a.items():
            setattr(row, key, value)
        if row.error:
            return ""
        if tool:  # named a tool that does not exist: that is the hallucination
            row.select_tool = tool
            check = CallCheck(status="unknown_tool", name=tool, format="select")
        else:
            check = check_reply(first, ALL_TOOL_NAMES, context.tags)
        _judge(row, turn, check, first)
        return first
    row.select_tool, row.passes = tool, 2
    manual = manual_message(tool, context)
    prompts.setdefault(f"manual:{tool}", manual)
    # The model's own choice stays in view (without it Gemma re-wrote the
    # <use> tag and then bare arguments); the manual follows as the app's turn.
    picked = [{"role": "assistant", "content": f"<use>{tool}</use>"},
              {"role": "system", "content": manual}]
    reply, b = await _generate(client, messages + picked, started)
    for key, value in _combine([a, b]).items():
        setattr(row, key, value)
    if not row.error:
        _judge(row, turn, check_reply(reply, ALL_TOOL_NAMES, context.tags), reply)
    return reply


class ContextStore:
    """Per-intent KV snapshots: prefilled and saved once, restored per message.

    Keyed on everything that changes the KV state - model, quantisation,
    context size and the exact static prompt text - so a snapshot is reused
    across runs exactly when it is still valid, like one stored on flash.
    """

    def __init__(self, client: ChatClient, model_key: str, context: AgentContext) -> None:
        self.client, self.model_key, self.context = client, model_key, context
        self.built = 0
        self.build_s = 0.0
        self.misses = 0
        self.used: set[str] = set()

    def _name(self, static: str) -> str:
        import hashlib

        digest = hashlib.sha1(f"{self.model_key}\x00{static}".encode("utf-8")).hexdigest()
        return f"ctx-{digest[:16]}.bin"

    def path(self, name: str) -> Path:
        from ..paths import KV_DIR

        return KV_DIR / self.client.model / name

    async def ensure(self, tools: tuple[str, ...], measuring: bool = False) -> tuple[str, bool]:
        return await self.ensure_static(static_part(tools, self.context), measuring)

    async def ensure_static(self, static: str, measuring: bool = False) -> tuple[str, bool]:
        from ..engine.server import EngineError

        name = self._name(static)
        self.used.add(name)
        if self.path(name).exists():
            return name, False
        started = time.perf_counter()
        primed = await self.client.complete_once(
            [{"role": "system", "content": static}, {"role": "user", "content": "."}])
        saved = await self.client.slot_action("save", name)
        if "error" in primed or "error" in saved:
            raise EngineError(f"KV snapshot failed: {primed.get('error') or saved.get('error')}")
        self.build_s += time.perf_counter() - started
        self.built += 1
        self.misses += measuring
        return name, True

    async def restore(self, name: str) -> float:
        from ..engine.server import EngineError

        started = time.perf_counter()
        reply = await self.client.slot_action("restore", name)
        if "error" in reply:
            raise EngineError(f"KV restore failed: {reply['error']}")
        return (time.perf_counter() - started) * 1000.0

    def stats(self) -> dict:
        sizes = [self.path(n).stat().st_size for n in self.used if self.path(n).exists()]
        return {"snapshots": len(sizes), "built_now": self.built, "build_s": self.build_s,
                "misses": self.misses, "bytes": sum(sizes),
                "avg_bytes": (sum(sizes) / len(sizes)) if sizes else 0}


def _class_ids(decision, previous_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Which classes' tasks and rules a decision's prompt carries."""
    if decision.kind == "inherit":
        return previous_ids
    ids = tuple(i for i in decision.group.split("+")
                if i in v2.BY_ID and v2.BY_ID[i].route != "inherit")
    if decision.kind == "merged" and "followup" in decision.group.split("+"):
        ids += tuple(i for i in previous_ids if i not in ids)
    # Registry order, so "details+food" and "food+details" are one prompt and
    # one KV snapshot (the first full run built both: ~12 s per miss).
    order = list(v2.BY_ID)
    return tuple(sorted(set(ids), key=order.index))


async def _run_mode(mode: str, cases, context: AgentContext, client: ChatClient | None,
                    router, log, store: ContextStore | None = None,
                    prompts: dict | None = None) -> list[Row]:
    from .router import FULL

    base = mode.split("@")[0]
    prompts = {} if prompts is None else prompts
    rows: list[Row] = []
    full = full_prompt(context)
    slot_static = None  # classes: the static prompt llama.cpp's slot already holds
    for case in cases:
        history = [Message(role, content) for role, content in case.history]
        previous = None
        previous_ids: tuple[str, ...] = ()
        for index, turn in enumerate(case.turns, 1):
            row = Row(mode, case.id, index, turn.text, turn.intent, turn.tool)
            started = time.perf_counter()
            if base == "active":
                reply = await _active_turn(row, turn, history, context, client, prompts, started)
                history += [Message("user", turn.text), Message("assistant", reply)]
                rows.append(row)
                log(row)
                continue
            system, offered = full, ALL_TOOL_NAMES
            row.prompt_name = "direct"
            prompts.setdefault("direct", full)
            if router is not None:
                decision = await router.route(turn.text, context, previous)
                previous = decision
                row.route_intent, row.route_score = decision.intent, decision.score
                row.route_margin, row.route_kind = decision.margin, decision.kind
                row.route_ok = decision.intent in turn.intent.split("|")
                if router.by_id is v2.BY_ID:
                    row.gold_class = v2.gold(turn.intent)
                    row.route_ok = decision.intent in row.gold_class.split("|")
                row.router_ms, row.encoder_ms = decision.router_ms, decision.encoder_ms
                if base == "classes":
                    ids =() if decision.group == FULL else _class_ids(decision, previous_ids)
                    previous_ids = ids
                    if ids or decision.group != FULL:
                        static = class_static(ids, decision.tools, context.tags)
                        system = static + "\n\n" + context.render()
                        row.prompt_name = "class:" + "+".join(ids)
                        prompts.setdefault(row.prompt_name, system)
                    else:
                        static = static_part(ALL_TOOL_NAMES, context)
                    offered = decision.tools
                    if store is not None:
                        if static != slot_static:
                            name, built = await store.ensure_static(static, measuring=True)
                            row.snapshot = "built" if built else "restored"
                            row.restore_ms = await store.restore(name)
                        else:
                            row.snapshot = "cached"
                        slot_static = static
                    row.tools_offered, row.offered = len(offered), list(offered)
                    messages = client.build_messages(history, turn.text, system_prompt=system)
                    reply, stats = await _generate(client, messages, started)
                    for key, value in stats.items():
                        setattr(row, key, value)
                    if not row.error:
                        _judge(row, turn, check_reply(reply, offered, context.tags), reply)
                    history += [Message("user", turn.text), Message("assistant", reply)]
                    rows.append(row)
                    log(row)
                    continue
                if decision.kind == "direct":
                    result = decision.direct
                    reply = (render_call(result.tool, result.args or {})
                             if result.is_call else result.text)
                    row.latency_s = time.perf_counter() - started
                    row.served_by = "router"
                    _judge(row, turn, check_reply(reply, ALL_TOOL_NAMES, context.tags), reply)
                    history += [Message("user", turn.text), Message("assistant", reply)]
                    rows.append(row)
                    log(row)
                    continue
                system, offered = prompt_for(decision.tools, context), decision.tools
                if store is not None:
                    name, built = await store.ensure(decision.tools, measuring=True)
                    row.snapshot = "built" if built else "hit"
                    row.restore_ms = await store.restore(name)
            row.tools_offered = len(offered)
            row.offered = list(offered)
            if client is None:  # router-only: score the routing, skip the SLM
                row.latency_s = time.perf_counter() - started
                rows.append(row)
                log(row)
                history += [Message("user", turn.text), Message("assistant", "")]
                continue
            messages = client.build_messages(history, turn.text, system_prompt=system)
            reply, stats = await _generate(client, messages, started)
            for key, value in stats.items():
                setattr(row, key, value)
            if not row.error:
                _judge(row, turn, check_reply(reply, offered, context.tags), reply)
            history += [Message("user", turn.text), Message("assistant", reply)]
            rows.append(row)
            log(row)
    return rows


async def _warm_up(context: AgentContext, client: ChatClient, router, mode: str = "direct") -> None:
    from .router import FULL

    base = mode.split("@")[0]
    system, stop = full_prompt(context), ()
    if base == "active":
        system, stop = index_prompt(context), SELECT_STOP
    elif router is not None:
        decision = await router.route("merhaba", context)
        system = prompt_for(decision.tools, context)
        if base == "classes" and decision.group != FULL:
            system = (class_static(_class_ids(decision, ()), decision.tools, context.tags)
                      + "\n\n" + context.render())
    await _generate(client, client.build_messages([], "merhaba", system_prompt=system),
                    time.perf_counter(), stop=stop)


def _class_statics(context: AgentContext) -> list[str]:
    """Every static prompt a single-class decision or a fallback can produce."""
    out = [static_part(ALL_TOOL_NAMES, context)]
    for spec in v2.CLASSES:
        if spec.route != "inherit":
            tools = tuple(n for n in ALL_TOOL_NAMES if n in spec.tools)
            out.append(class_static((spec.id,), tools, context.tags))
    return out


def _all_prompts(context: AgentContext, bases: set[str], used: dict) -> dict[str, str]:
    """Every system prompt a run could send, plus the merged ones it did send."""
    from .prompts import class_prompt

    out = {"direct": full_prompt(context)}
    if "active" in bases:
        out["active"] = index_prompt(context)
        out.update({f"manual:{t}": manual_message(t, context) for t in ALL_TOOL_NAMES})
    if "classes" in bases:
        for spec in v2.CLASSES:
            if spec.route != "inherit":
                tools = tuple(n for n in ALL_TOOL_NAMES if n in spec.tools)
                out[f"class:{spec.id}"] = class_prompt((spec.id,), tools, context)
    out.update({k: v for k, v in used.items() if k not in out})
    return out


async def _sample_loop(sampler, stop: asyncio.Event, interval: float) -> None:
    while not stop.is_set():
        sampler.sample(busy=True)
        try:
            await asyncio.wait_for(stop.wait(), interval)
        except asyncio.TimeoutError:
            pass


# --------------------------------------------------------------------------
def _avg(values) -> float:
    values = [v for v in values if v]
    return statistics.fmean(values) if values else 0.0


def _p(values, pct: float) -> float:
    from ..metrics.session import _percentile

    return _percentile([v for v in values if v], pct)


def _rate(rows: list[Row], tokens: str, ms: str) -> float:
    total_ms = sum(getattr(r, ms) for r in rows)
    return 1000.0 * sum(getattr(r, tokens) for r in rows) / total_ms if total_ms else 0.0


def summarize(rows: list[Row]) -> dict:
    n = len(rows) or 1
    ok = [r for r in rows if not r.error]
    slm = [r for r in ok if r.served_by == "slm"]
    due = [r for r in ok if r.expected_tool]
    routed = [r for r in ok if r.route_kind]
    statuses = Counter(r.status for r in ok)
    return {
        "turns": len(rows),
        "errors": len(rows) - len(ok),
        "exact_pct": 100.0 * sum(r.exact for r in ok) / n,
        "tool_pct": 100.0 * sum(r.tool_ok for r in ok) / n,
        "args_pct": 100.0 * sum(r.args_ok for r in due) / (len(due) or 1),
        "hallucination_pct": 100.0 * sum(r.hallucinated for r in ok) / n,
        "hallucinated": sum(r.hallucinated for r in ok),
        "unsafe_sends": sum(r.forbidden for r in ok),
        "skipped_tool": sum(r.skipped_tool for r in due),
        "needless_call": sum(r.needless_call for r in ok),
        "statuses": dict(statuses),
        "formats": dict(Counter(r.format for r in ok if r.format)),
        "warnings": sum(len(r.warnings) for r in ok),
        "slm_turns": len(slm),
        "router_turns": sum(1 for r in ok if r.served_by == "router"),
        "latency_avg": _avg(r.latency_s for r in ok),
        "latency_p50": _p([r.latency_s for r in ok], 50),
        "latency_p95": _p([r.latency_s for r in ok], 95),
        "latency_slm_avg": _avg(r.latency_s for r in slm),
        "ttft_avg": _avg(r.ttft_s for r in slm),
        "ttft_med": _p([r.ttft_s for r in slm], 50),
        "passes_avg": _avg(r.passes for r in slm),
        "select_ms_avg": _avg(r.select_ms for r in slm),
        "prompt_tokens_avg": _avg(r.prompt_tokens for r in slm),
        "cached_tokens_avg": statistics.fmean([r.cached_tokens for r in slm]) if slm else 0.0,
        "prefilled_tokens_avg": _avg(r.prefilled_tokens for r in slm),
        "completion_tokens_avg": _avg(r.completion_tokens for r in slm),
        # Summed tokens over summed time: an average of per-turn rates lets a
        # one-token reply (decode_ms ~0) report tens of thousands of tok/s.
        "prefill_tps_avg": _rate(slm, "prefilled_tokens", "prompt_ms")
        or _avg(r.prefill_tps for r in slm),
        "decode_tps_avg": _rate(slm, "completion_tokens", "decode_ms")
        or _avg(r.decode_tps for r in slm),
        "restore_ms_avg": _avg(r.restore_ms for r in ok),
        "router_ms_avg": _avg(r.router_ms for r in routed),
        "encoder_ms_avg": _avg(r.encoder_ms for r in routed),
        "route_acc_pct": (100.0 * sum(r.route_ok for r in routed) / len(routed)) if routed else 0.0,
        "route_kinds": dict(Counter(r.route_kind for r in routed)),
    }


def threshold_sweep(rows: list[Row], attr: str = "route_score",
                    start: float = 0.45, step: float = 0.05, count: int = 9) -> list[dict]:
    """How accuracy and fallback trade off as a confidence cut-off moves.

    ``attr="route_score"`` sweeps ``router_threshold``; ``"route_margin"``
    sweeps ``router_margin``, which has proved the sharper signal.
    """
    sweep = []
    n = len(rows) or 1
    for index in range(count):
        t = start + index * step
        kept = [r for r in rows if getattr(r, attr) >= t - 1e-9]
        sweep.append({
            "threshold": round(t, 3),
            "fallback_pct": 100.0 * (1 - len(kept) / n),
            "acc_kept_pct": (100.0 * sum(r.route_ok for r in kept) / len(kept)) if kept else 0.0,
        })
    return sweep


# --------------------------------------------------------------------------
async def run_bench(
    model_id: str = "",
    modes: tuple[str, ...] = ("direct", "hierarchical"),
    router_only: bool = False,
    limit: int = 0,
    cache: bool = True,
    thinking: str = "off",
    max_tokens: int = 256,
    temperature: float = 0.0,
    profile: str = "",
    cache_ram: int | None = None,
    suite: str = "standard",
    cores: int = 0,
    ram_gb: float = 0.0,
    router_model: str = "",
    log=print,
) -> tuple[Path, Path]:
    from .. import catalog
    from ..engine import make_server, runtime_info
    from ..engine.server import EngineError
    from ..metrics import Sampler
    from . import start_router

    settings = Settings.from_dict(Settings.load().to_dict())
    if profile:
        settings.apply_profile(profile)
    if cache_ram is not None:
        settings.prompt_cache_ram_mb = cache_ram
    # A hand-made device: e.g. 2 cores / 3 GB for a low-budget phone. Giving a
    # RAM figure turns the ceiling on - otherwise "3 GB" would be a label the
    # run never honours.
    if router_model:
        settings.router_model = router_model
    if cores:
        settings.threads = cores
    if ram_gb:
        settings.nominal_ram_gb = ram_gb
        settings.ram_limit_enabled = True
        settings.ram_limit_mb = 0
    settings.mark_custom_if_needed()
    # Restored snapshots are only reusable with a full sliding-window cache.
    # One server serves every mode, so they all run with it - fair, and stated.
    bases = {m.split("@")[0] for m in modes}
    if bases & {"contexts", "classes"}:
        settings.swa_full = True
    settings.thinking_mode = thinking
    settings.max_tokens = max_tokens
    settings.temperature = temperature
    settings.keep_history = True
    settings.history_turns = max(settings.history_turns, 4)
    context = AgentContext.load().frozen()
    cases = ordered(limit, SEED, suite)

    def line(row: Row) -> None:
        mark = "ok " if (row.route_ok if router_only else row.exact) else "-- "
        got = row.call or ("text" if row.status == "text" else row.route_intent)
        extra = f" {row.route_intent}:{row.route_kind}" if row.route_kind else ""
        log(f"  [{row.mode[:4]}] {mark}{row.case:<22} {got:<26}{extra}  {row.latency_s:5.2f}s")

    meta: dict = {
        "started_at": time.time(),
        "model_id": model_id,
        "router_model": settings.router_model,
        "profile": settings.device.name,
        "threads": settings.cpu_budget_cores,
        "nominal_ram_gb": settings.nominal_ram_gb,
        "ram_limit_bytes": settings.ram_limit_bytes,
        "thinking": thinking,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "prompt_cache": cache,
        "cache_ram_mb": settings.prompt_cache_ram_mb,
        "swa_full": settings.swa_full,
        "seed": SEED,
        "suite": suite,
        "case_titles": {c.id: c.title for c in cases if c.title},
        "case_history": {c.id: [list(m) for m in c.history] for c in cases if c.history},
        "cases": len(cases),
        "turns": sum(len(c.turns) for c in cases),
        "context_now": context.now,
        "router": _router_meta(settings),
    }
    results: dict[str, dict] = {}
    all_rows: list[Row] = []

    if router_only:
        taxonomy = "classes" if "classes" in bases else "intents"
        meta["taxonomy"] = taxonomy
        log(f"Router-only run ({taxonomy}): {meta['turns']} turns, "
            f"{settings.cpu_budget_cores} core(s)")
        router_server, router = await start_router(settings, on_status=lambda m: log(f"  {m}"),
                                                   taxonomy=taxonomy)
        try:
            meta["router_rss"] = router_server.report.rss_after_load
            meta["router_load_s"] = router_server.report.load_seconds
            meta["router_props"] = await router.props()
            rows = await _run_mode("router", cases, context, None, router, line)
        finally:
            await router.aclose()
            await router_server.stop()
        results["router"] = summarize(rows)
        results["router"]["sweep"] = threshold_sweep(rows)
        results["router"]["margin_sweep"] = threshold_sweep(
            rows, "route_margin", start=0.0, step=0.01, count=9)
        all_rows = rows
        return _write(meta, results, all_rows, context, router_only=True)

    spec = catalog.get(model_id)
    if spec is None:
        raise EngineError(f"Unknown model id '{model_id}'. Run: python -m feast_ai models")
    if not spec.downloaded:
        raise EngineError(f"{spec.name} is not downloaded. Run: python -m feast_ai pull {spec.id}")
    meta.update(model_name=spec.name, quant=spec.quant, engine=spec.engine,
                model_bytes=spec.local_size)
    server = make_server(spec.engine, settings, runtime_info().server)
    log(f"Loading {spec.name} ({spec.quant}) on {settings.cpu_budget_cores} core(s)…")
    report = await server.start(spec.local_path, alias=spec.id, on_status=lambda m: log(f"  {m}"))
    meta["load_s"] = report.load_seconds
    meta["pinned_cores"] = report.pinned_cores
    client = ChatClient(server.base_url, settings, model=spec.id)
    client.cache_prompt = cache
    sampler = Sampler(budget_cores=settings.cpu_budget_cores, history_points=100_000)

    try:
        meta["prompt_tokens"] = await _prompt_sizes(client, context)
        used_prompts: dict[str, str] = {}
        for mode in modes:
            base, _, router_id = mode.partition("@")
            router_server = router = store = None
            if base in ROUTER_MODES:
                chosen = Settings.from_dict(settings.to_dict())
                if router_id:
                    chosen.router_model = router_id
                router_server, router = await start_router(
                    chosen, server, on_status=lambda m: log(f"  router: {m}"),
                    taxonomy="classes" if base == "classes" else "intents")
                meta["router_rss"] = router_server.report.rss_after_load
                meta["router_load_s"] = router_server.report.load_seconds
                threshold, margin, direct = router.thresholds()
                meta.setdefault("routers", {})[mode] = {
                    **_router_meta(chosen), "model": chosen.router_model,
                    "threshold": threshold, "margin": margin, "direct_threshold": direct,
                    "taxonomy": "classes" if base == "classes" else "intents",
                    "rss": router_server.report.rss_after_load,
                    "load_s": router_server.report.load_seconds,
                    "question": router.question() if hasattr(router, "question") else None,
                }
                if base != "hierarchical":
                    # The router only decides the system prompt: no answers
                    # of its own, the SLM runs for every message.
                    picker = Settings.from_dict(chosen.to_dict())
                    picker.direct_calls = False
                    router.settings = picker
            if base in ("contexts", "classes"):
                store = ContextStore(
                    client, f"{spec.id}|{spec.quant}|ctx{settings.context_size}"
                    f"|swa{int(settings.swa_full)}", context)
                log(f"  building per-{'class' if base == 'classes' else 'intent'} KV snapshots…")
                if base == "classes":
                    for static in _class_statics(context):
                        await store.ensure_static(static)
                else:
                    for tools in _snapshot_sets():
                        await store.ensure(tools)
                meta["contexts"] = store.stats()
                log(f"  {meta['contexts']['snapshots']} snapshots ready "
                    f"({meta['contexts']['built_now']} built in {store.build_s:.0f}s)")
            try:
                procs = server.snapshot_processes()
                if router_server is not None:
                    procs += router_server.snapshot_processes()
                await _warm_up(context, client, router, mode)
                sampler.reset()
                sampler.attach(procs)
                stop = asyncio.Event()
                task = asyncio.create_task(_sample_loop(sampler, stop, 0.25))
                log(f"\n{mode}: {meta['turns']} turns")
                started = time.perf_counter()
                rows = await _run_mode(mode, cases, context, client, router, line, store,
                                       used_prompts)
                stop.set()
                await task
                summary = summarize(rows)
                summary["wall_s"] = time.perf_counter() - started
                summary["peak_rss"] = sampler.stats.peak_rss
                summary["cpu_avg"] = sampler.stats.avg_cpu_device
                # Under a ceiling the weights stream: faults are what it costs.
                summary["faults"] = sampler.stats.total_faults
                summary["paged_bytes"] = sampler.stats.total_faults * 4096
                if store is not None:
                    meta["contexts"] = store.stats()
                    summary["snapshots"] = store.stats()
                summary["ram_limit_model"] = server.ram_limit_bytes
                results[mode] = summary
                all_rows += rows
            finally:
                if router is not None:
                    await router.aclose()
                if router_server is not None:
                    await router_server.stop()
                    server.set_ram_limit(None)
        # Every system prompt the run could send, with its exact token count,
        # so the report can show what each pipeline actually read.
        meta["prompts"] = {}
        for name, text in _all_prompts(context, bases, used_prompts).items():
            meta["prompts"][name] = {"text": text, "tokens": await client.tokenize_count(text)}
    finally:
        await server.stop()
    return _write(meta, results, all_rows, context)


def _router_meta(settings: Settings) -> dict:
    """The thresholds the router actually used - its own scale for Decision 2.0."""
    from ..engine.decision_server import is_decision

    if is_decision(settings.router_model):
        from .decision import THRESHOLDS

        threshold, margin, direct = THRESHOLDS
    else:
        threshold, margin, direct = (settings.router_threshold, settings.router_margin,
                                     settings.direct_threshold)
    return {"threshold": threshold, "margin": margin, "direct_threshold": direct,
            "direct_calls": settings.direct_calls}


ROUTER_MODES = ("hierarchical", "routed", "contexts", "classes")
MODES = ("direct", "active") + ROUTER_MODES


def _snapshot_sets() -> list[tuple[str, ...]]:
    """Every tool set a single-intent decision or a fallback can produce."""
    sets = {ALL_TOOL_NAMES, ()}
    for intent in INTENTS:
        if intent.route != "inherit":
            sets.add(tuple(n for n in ALL_TOOL_NAMES if n in intent.tools))
    return sorted(sets, key=len)


async def _prompt_sizes(client: ChatClient, context: AgentContext) -> dict:
    """Token count of every prompt a run can send, via the model's tokenizer."""
    sizes = {}
    prompts = {"full": (len(ALL_TOOL_NAMES), full_prompt(context))}
    for intent in INTENTS:
        if intent.route == "inherit":
            continue
        name = f"intent:{intent.id}" if intent.tools else "chat"
        prompts.setdefault(name, (len(intent.tools), prompt_for(intent.tools, context)))
    for name, (tools, text) in prompts.items():
        count = await client.tokenize_count(text)
        sizes[name] = {"tools": tools, "tokens": count, "chars": len(text),
                       "estimated": count is None}
        if count is None:
            sizes[name]["tokens"] = len(text) // 4
    return sizes


# --------------------------------------------------------------------------
def _write(meta: dict, results: dict, rows: list[Row], context: AgentContext,
           router_only: bool = False) -> tuple[Path, Path]:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(meta["started_at"]))
    name = "router" if router_only else meta.get("model_id", "model")
    if meta.get("suite", "standard") != "standard":
        name += f"-{meta['suite']}"
    # Not with_suffix(): model ids like "qwen3-0.6b" contain a dot.
    json_path = SESSIONS_DIR / f"bench-{stamp}-{name}.json"
    md_path = SESSIONS_DIR / f"bench-{stamp}-{name}.md"
    payload = {"meta": meta, "results": results, "rows": [asdict(r) for r in rows],
               "full_prompt": full_prompt(context)}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = _markdown_router(meta, results, rows) if router_only else _markdown(meta, results, rows, context)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not router_only:
        from .page import render

        md_path.with_name(md_path.stem + ".html").write_text(
            render([payload], standalone=True), encoding="utf-8")
    return md_path, json_path


def _cell(text: str, width: int = 60) -> str:
    text = " ".join(str(text).split()).replace("|", "/")
    return text[:width] + ("…" if len(text) > width else "")


def _intent_table(rows: list[Row], modes: list[str], key: str) -> list[str]:
    lines = ["| Intent | n | " + " | ".join(modes) + " |",
             "| --- | --- | " + " | ".join("---" for _ in modes) + " |"]
    for intent in [i.id for i in INTENTS]:
        per = {m: [r for r in rows if r.mode == m
                   and r.expected_intent.split("|")[0] == intent] for m in modes}
        n = max(len(v) for v in per.values())
        if not n:
            continue
        cells = [f"{sum(getattr(r, key) for r in per[m])}/{len(per[m])}" for m in modes]
        lines.append(f"| {intent} | {n} | " + " | ".join(cells) + " |")
    return lines


def _markdown(meta: dict, results: dict, rows: list[Row], context: AgentContext) -> list[str]:
    modes = list(results)
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(meta["started_at"]))
    ceiling = human_bytes(meta["ram_limit_bytes"]) if meta["ram_limit_bytes"] else "none"
    lines = [
        f"# feast AI agent benchmark - {meta.get('model_name', meta['model_id'])}",
        "",
        f"*{when}*",
        "",
        "## Setup",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Model | `{meta['model_id']}` ({meta.get('quant', '?')}, {meta.get('engine', '?')}, "
        f"{human_bytes(meta.get('model_bytes', 0))}) |",
        f"| Device profile | {meta['profile']} ({meta.get('nominal_ram_gb', 0):g} GB device RAM) |",
        f"| CPU budget | {meta['threads']} thread(s), pinned to {meta.get('pinned_cores') or 'not pinned'} |",
        f"| RAM ceiling | {ceiling} (split with the router in the hierarchical pass) |",
        f"| Decoding | temperature {meta['temperature']:g}, max {meta['max_tokens']} tokens, "
        f"thinking {meta['thinking']} |",
        f"| Prompt cache | {'on (warm, as an app runs)' if meta['prompt_cache'] else 'off (cold prefill)'} "
        f"· cross-prompt RAM cache {meta.get('cache_ram_mb') or 'off'}"
        f"{' MB' if (meta.get('cache_ram_mb') or 0) > 0 else ''} |",
        f"| Router | {meta['router_model']} · threshold {meta['router']['threshold']:g} · "
        f"margin {meta['router']['margin']:g} · direct ≥ {meta['router']['direct_threshold']:g} |",
        f"| Suite | `{meta.get('suite', 'standard')}` · {meta['cases']} cases / {meta['turns']} turns, "
        f"shuffled (seed {meta['seed']}) |",
        f"| Context clock | frozen at {meta['context_now']} |",
        "",
        "## Prompt sizes",
        "",
        "Every prompt shares the same preamble, call format and context; only the "
        "tool list differs.",
        "",
        "| Prompt | Tools | Tokens |",
        "| --- | --- | --- |",
    ]
    for name, size in sorted(meta.get("prompt_tokens", {}).items(), key=lambda kv: -kv[1]["tokens"]):
        approx = "~" if size["estimated"] else ""
        lines.append(f"| {name} | {size['tools']} | {approx}{size['tokens']} |")

    def row(label: str, fmt) -> str:
        return f"| {label} | " + " | ".join(fmt(results[m]) for m in modes) + " |"

    lines += [
        "",
        "## Headline",
        "",
        "| Metric | " + " | ".join(modes) + " |",
        "| --- | " + " | ".join("---" for _ in modes) + " |",
        row("Exact (right tool, right args, valid call)", lambda r: f"**{r['exact_pct']:.0f}%**"),
        row("Right tool (or rightly no tool)", lambda r: f"{r['tool_pct']:.0f}%"),
        row("Right args, when a tool was due", lambda r: f"{r['args_pct']:.0f}%"),
        row("Hallucinated calls", lambda r: f"{r['hallucinated']} ({r['hallucination_pct']:.0f}%)"),
        row("Unsafe sends (confirmation skipped)", lambda r: str(r["unsafe_sends"])),
        row("Answered in text where a tool was due", lambda r: str(r["skipped_tool"])),
        row("Called a tool where text was right", lambda r: str(r["needless_call"])),
        row("Served without the SLM", lambda r: f"{r['router_turns']} of {r['turns']}"),
        row("Latency avg / p50 / p95 (all turns)",
            lambda r: f"{r['latency_avg']:.2f} / {r['latency_p50']:.2f} / {r['latency_p95']:.2f} s"),
        row("Latency avg (SLM turns)", lambda r: f"{r['latency_slm_avg']:.2f} s"),
        row("TTFT avg (SLM turns, routing included)", lambda r: f"{r['ttft_avg']:.2f} s"),
        row("Prompt tokens avg (cached)",
            lambda r: f"{r['prompt_tokens_avg']:.0f} ({r['cached_tokens_avg']:.0f})"),
        row("Prefilled tokens avg", lambda r: f"{r['prefilled_tokens_avg']:.0f}"),
        row("Output tokens avg", lambda r: f"{r['completion_tokens_avg']:.0f}"),
        row("Prefill / decode t/s", lambda r: f"{r['prefill_tps_avg']:.0f} / {r['decode_tps_avg']:.1f}"),
        row("KV snapshot restore avg", lambda r: f"{r['restore_ms_avg']:.0f} ms" if r.get("restore_ms_avg") else "-"),
        row("Router avg (wall / encoder)",
            lambda r: f"{r['router_ms_avg']:.0f} / {r['encoder_ms_avg']:.0f} ms" if r["router_ms_avg"] else "-"),
        row("Peak RSS (model + router)", lambda r: human_bytes(r.get("peak_rss", 0))),
        row("Page-ins (first mmap read + any re-reads)", lambda r: human_bytes(r.get("paged_bytes", 0))),
        row("CPU avg (device budget)", lambda r: f"{r.get('cpu_avg', 0):.0f}%"),
        row("Wall time for the suite", lambda r: f"{r.get('wall_s', 0):.0f} s"),
        row("Call formats", lambda r: ", ".join(f"{k} {v}" for k, v in sorted(r["formats"].items())) or "-"),
        "",
    ]
    if "hierarchical" in results:
        h = results["hierarchical"]
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(h["route_kinds"].items()))
        lines += [
            "## Routing",
            "",
            f"Router picked the labelled intent on **{h['route_acc_pct']:.0f}%** of turns "
            f"({kinds}). Router RSS after load: {human_bytes(meta.get('router_rss', 0))}.",
            "",
        ]
        confused = [r for r in rows if r.mode == "hierarchical" and not r.route_ok]
        if confused:
            lines += ["| Turn | Expected | Routed | Score | Served |", "| --- | --- | --- | --- | --- |"]
            lines += [f"| {_cell(r.text, 44)} | {r.expected_intent} | {r.route_intent} "
                      f"| {r.route_score:.2f} | {r.route_kind} |" for r in confused]
            lines.append("")
    lines += ["## Exact by intent", ""] + _intent_table(rows, modes, "exact") + [""]

    failures = [r for r in rows if not r.exact]
    if failures:
        lines += ["## Misses", "", "| Mode | Turn | Expected | Got | Why |",
                  "| --- | --- | --- | --- | --- |"]
        for r in failures:
            got = r.error or (f"{r.call} ({r.status})" if r.call or r.status != "text" else "text")
            why = "; ".join(r.issues + r.missing_args) or (
                "unsafe send" if r.forbidden else
                "no tool called" if r.skipped_tool else
                "tool called, text was right" if r.needless_call else "")
            lines.append(f"| {r.mode} | {_cell(r.text, 40)} | {r.expected_tool or 'text'} "
                         f"| {_cell(got, 40)} | {_cell(why, 70)} |")
        lines.append("")
    lines += ["## All turns", "",
              "| Mode | Case | Route | Call | Exact | Prompt (cached) | Out | TTFT s | Latency s | Reply |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        route = f"{r.route_intent} ({r.route_kind})" if r.route_kind else "-"
        lines.append(
            f"| {r.mode} | {r.case}#{r.turn} | {route} | {r.call or r.status or '-'} "
            f"| {'yes' if r.exact else 'no'} | {r.prompt_tokens} ({r.cached_tokens}) "
            f"| {r.completion_tokens} | {r.ttft_s:.2f} | {r.latency_s:.2f} | {_cell(r.reply, 50)} |"
        )
    return lines


def _markdown_router(meta: dict, results: dict, rows: list[Row]) -> list[str]:
    r = results["router"]
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(meta["started_at"]))
    props = meta.get("router_props", {})
    bank = ("cached" if props.get("bank_cached")
            else f"embedded in {props.get('bank_ms', 0) / 1000:.1f} s")
    lines = [
        "# feast AI router benchmark",
        "",
        f"*{when}* · {meta['router_model']} · {meta['threads']} core(s) · "
        f"{props.get('bank_size', '?')} exemplars over {len(INTENTS)} intents",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Turns | {r['turns']} ({meta['cases']} cases) |",
        f"| Top-1 intent accuracy | **{r['route_acc_pct']:.0f}%** |",
        f"| Decisions | {', '.join(f'{k} {v}' for k, v in sorted(r['route_kinds'].items()))} |",
        f"| Router avg (wall / encoder) | {r['router_ms_avg']:.1f} / {r['encoder_ms_avg']:.1f} ms |",
        f"| Router RSS after load | {human_bytes(meta.get('router_rss', 0))} |",
        f"| Load time | {meta.get('router_load_s', 0):.1f} s (exemplar bank {bank}) |",
        f"| Direct answers that were exactly right | "
        f"{sum(x.exact for x in rows if x.served_by == 'router')} of {r['router_turns']} |",
        "",
        "## Threshold sweep",
        "",
        "Raising `router_threshold` trades coverage for precision: below it the "
        "SLM gets every tool, which is never worse than the direct pipeline.",
        "",
        "| Threshold | Falls back | Accuracy of the rest |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {s['threshold']:.2f} | {s['fallback_pct']:.0f}% | {s['acc_kept_pct']:.0f}% |"
              for s in r["sweep"]]
    lines += ["", "## Margin sweep", "",
              "Top score minus the best intent its tool group cannot serve - "
              "`router_margin`. A sharper confidence signal than the raw score, "
              "because wrong routes tend to be near-ties. Below it the turn is "
              "widened (tied groups merged) rather than trusted.",
              "", "| Margin | Widened | Accuracy of the rest |", "| --- | --- | --- |"]
    lines += [f"| {s['threshold']:.2f} | {s['fallback_pct']:.0f}% | {s['acc_kept_pct']:.0f}% |"
              for s in r["margin_sweep"]]
    lines += ["", "## Accuracy by intent", ""] + _intent_table(rows, ["router"], "route_ok")
    confused = [x for x in rows if not x.route_ok]
    if confused:
        lines += ["", "## Confusions", "", "| Turn | Expected | Routed | Score | Margin |",
                  "| --- | --- | --- | --- | --- |"]
        lines += [f"| {_cell(x.text, 48)} | {x.expected_intent} | {x.route_intent} "
                  f"| {x.route_score:.2f} | {x.route_margin:.3f} |" for x in confused]
    return lines


__all__ = ["run_bench", "summarize", "threshold_sweep"]
