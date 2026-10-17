"""The hierarchical pipeline's first stage: decide how a message is served.

The encoder worker only measures similarity (see
:mod:`feast_ai.engine.encoder_worker`).  The policy - which is what the
thesis is actually comparing - lives here, in four ordered rules:

1. **Unsure -> widen.**  Below ``router_threshold``, every tool: never worse
   than the ``direct`` pipeline, just not better.  When an intent this group
   *cannot* serve scores within ``router_margin`` (a near-tie), the SLM gets
   the union of the tied intents' tool groups instead - still a fraction of
   the full prompt - and the full prompt only if that union passes
   :data:`MAX_MERGED`.  Rivals whose tools the top group already contains do
   not count as ties: routing to either serves both.
2. **Follow-up -> inherit.**  "evet", "daha ucuzu var mi" reuse the previous
   turn's tool group (rule 1 if there is none).
3. **Confident and trivial -> no SLM.**  ``direct`` intents above
   ``direct_threshold`` (stricter than rule 1) go to
   :mod:`feast_ai.agent.direct`; if the handler declines, rule 4 - except
   for intents with *no* tools (time, out of scope), which fall back to the
   full prompt instead: an unsure guess must never strip every tool.  The
   first router benchmark routed "Bir latte kac lira" to out_of_scope at 0.62
   (the "Dolar kac lira" exemplar); with no tools the SLM could not recover.
4. **Otherwise -> SLM with the intent's tool group.**

Calibration (router benchmark, 60 turns, 4 cores): the top score separates
right from wrong poorly - wrong routes scored 0.62-0.82 - but the *margin*
separates well.  On the first run the plain top-1/top-2 margin >= 0.04 kept
94% accuracy at 20% fallback, where the raw score needed >= 0.80 and 60%
fallback for 96%.  With the margin measured against rivals the group cannot
serve and near-ties merged, the final run made 14 direct, 32 single-group,
9 merged, 3 inherited and only 2 full-prompt decisions.

Two latencies are kept apart.  ``encoder_ms`` is inference inside the worker -
the number a phone would see.  ``router_ms`` is the wall clock here,
including the localhost HTTP hop that would not exist on a device; the report
uses it for end-to-end latency anyway, so the comparison never flatters the
router.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

from ..config import Settings
from .context import AgentContext
from .direct import DirectResult, handle
from .intents import BY_ID
from .tools import ALL_TOOL_NAMES

FULL = "full"
#: A merged near-tie group larger than this is no cheaper than guessing - the
#: full prompt (15 tools) is used instead.
MAX_MERGED = 8


@dataclass
class RouteDecision:
    intent: str            # what the router saw
    score: float
    margin: float          # against the best intent this group cannot serve
    runner_up: str         # ...and that intent
    kind: str              # "direct" | "slm" | "merged" | "fallback" | "inherit"
    group: str             # whose tool group is in play: an intent id, or "full"
    tools: tuple[str, ...]
    direct: DirectResult | None = None
    nearest: str = ""      # the exemplar that matched best
    router_ms: float = 0.0
    encoder_ms: float = 0.0
    tokens: int = 0
    reason: str = ""
    top: list[tuple[str, float]] = field(default_factory=list)
    raw_margin: float = 0.0  # top-1 minus top-2, whatever top-2 is
    text: str = ""           # the message routed (Decision 2.0 reads it as context next turn)

    @property
    def uses_slm(self) -> bool:
        return self.kind != "direct"

    @property
    def prompt_name(self) -> str:
        if self.group == FULL:
            return "full"
        return f"intent:{self.group}" if self.tools else "chat"

    def summary(self) -> str:
        served = {
            "direct": "no SLM",
            "slm": f"SLM, {len(self.tools)} tool(s)" if self.tools else "SLM, no tools",
            "merged": f"near-tie -> SLM, {len(self.tools)} tool(s) of {self.group}",
            "fallback": f"unsure -> SLM, all {len(self.tools)} tools",
            "inherit": f"follow-up -> {self.group}, {len(self.tools)} tool(s)",
        }[self.kind]
        return (
            f"{self.intent} {self.score:.2f} "
            + (f"(rival {self.runner_up} {self.score - self.margin:.2f})"
               if self.runner_up else "(no rival outside its tools)")
            + f" · {served} · {self.router_ms:.0f} ms"
        )


class RouterError(RuntimeError):
    pass


class Router:
    """Policy plus the HTTP client for one running encoder worker."""

    def __init__(self, base_url: str, settings: Settings, taxonomy: dict | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.settings = settings
        #: id -> Intent (``intents.BY_ID``) or ClassSpec (``classes.BY_ID``):
        #: the policy only reads ``route`` and ``tools``.
        self.by_id = taxonomy if taxonomy is not None else BY_ID
        self._client: httpx.AsyncClient | None = None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def classify(self, text: str, previous: RouteDecision | None = None) -> dict:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        try:
            response = await self._client.post(f"{self.base_url}/route", json={"text": text})
        except httpx.HTTPError as exc:
            raise RouterError(f"router unreachable: {exc}") from exc
        if response.status_code != 200:
            raise RouterError(f"router HTTP {response.status_code}: {response.text[:200]}")
        return response.json()

    def thresholds(self) -> tuple[float, float, float]:
        """``(threshold, margin, direct_threshold)`` on this router's score scale."""
        s = self.settings
        return s.router_threshold, s.router_margin, s.direct_threshold

    async def warmup(self) -> None:
        """Pay the one-off costs at load time, not on the first message.

        Building the HTTP client and the first single-sentence ORT run cost
        ~160 ms here; neither exists per message on a device, so charging
        them to whichever query happened to come first would skew it.
        """
        try:
            await self.classify("merhaba")
        except RouterError:
            pass

    async def props(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/props")
                return response.json() if response.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            return {}

    # ------------------------------------------------------------------
    async def route(
        self,
        text: str,
        context: AgentContext,
        previous: RouteDecision | None = None,
    ) -> RouteDecision:
        started = time.perf_counter()
        raw = await self.classify(text, previous)
        decision = self.decide(raw, text, context, previous)
        decision.router_ms = (time.perf_counter() - started) * 1000.0
        decision.text = text
        return decision

    def decide(
        self,
        raw: dict,
        text: str,
        context: AgentContext,
        previous: RouteDecision | None,
    ) -> RouteDecision:
        """Apply the policy to the worker's scores. Pure, so it is testable."""
        s = self.settings
        threshold, margin, direct_threshold = self.thresholds()
        scores = [(str(label), float(score)) for label, score in raw.get("scores", [])]
        top, score = scores[0] if scores else ("", 0.0)
        raw_rival, raw_second = scores[1] if len(scores) > 1 else ("", 0.0)
        by_id = self.by_id
        intent = by_id.get(top)

        # A near-tie only matters if the runner-up would be served differently.
        # "Kaydedilenlerimi goster" scores saved_open 0.78 vs navigate 0.75, but
        # navigate's one tool is inside saved_open's group - the prompt covers
        # both, so that is not ambiguity. The margin is measured against the
        # best intent this intent's tool group can NOT serve.
        covered = set(intent.tools) if intent else set()

        def competes(label: str) -> bool:
            other = by_id.get(label)
            if other is None:
                return True
            if other.route == "inherit":
                return previous is None or not set(previous.tools) <= covered
            return not set(other.tools) <= covered

        rival, second = next(((k, v) for k, v in scores[1:] if competes(k)), ("", 0.0))
        decision = RouteDecision(
            intent=top, score=score, margin=score - second, runner_up=rival,
            kind="slm", group=top, tools=(), nearest=raw.get("nearest", ""),
            encoder_ms=float(raw.get("tokenize_ms", 0.0)) + float(raw.get("infer_ms", 0.0)),
            tokens=int(raw.get("tokens", 0)), top=scores[:5],
            raw_margin=score - raw_second,
        )

        def fallback(reason: str) -> RouteDecision:
            decision.kind, decision.group, decision.tools = "fallback", FULL, ALL_TOOL_NAMES
            decision.reason = reason
            return decision

        if intent is None:
            return fallback("unknown intent")
        if score < threshold:
            return fallback(f"score {score:.2f} < threshold {threshold:.2f}")
        if decision.margin < margin:
            return self._merge(decision, scores, previous, fallback)

        if intent.route == "inherit":
            if previous is None:
                return fallback("follow-up with nothing to follow")
            decision.kind, decision.group, decision.tools = "inherit", previous.group, previous.tools
            decision.reason = f"follow-up inherits {previous.group}"
            return decision

        decision.tools = tuple(n for n in ALL_TOOL_NAMES if n in intent.tools)
        if intent.route == "direct":
            served = None
            if not s.direct_calls:
                why = "direct calls disabled"
            elif score < direct_threshold:
                why = f"direct intent, but {score:.2f} < {direct_threshold:.2f}"
            else:
                served = handle(intent.id, text, context)
                why = "direct handler declined"
                # Acting without the SLM needs more than a covering prompt: a
                # close runner-up must want the very same action.
                if served is not None and decision.raw_margin < margin \
                        and handle(raw_rival, text, context) != served:
                    served = None
                    why = (f"near-tie with {raw_rival} ({decision.raw_margin:.3f}), "
                           "which would act differently")
            if served is not None:
                decision.kind, decision.direct = "direct", served
                decision.reason = "served without the SLM"
                return decision
            if not intent.tools:
                if not s.direct_calls and score >= direct_threshold:
                    # The router only picks the prompt here: a confident
                    # "saat kac" gets the no-tool prompt, the SLM answers.
                    decision.reason = "confident no-tool intent - SLM without tools"
                    return decision
                return fallback(f"{why}; a no-tool guess would hide every tool")
            decision.reason = why
            return decision
        decision.reason = "intent tool group"
        return decision

    def _merge(self, decision: RouteDecision, scores, previous, fallback) -> RouteDecision:
        """A near-tie: offer the tools of every intent within the margin.

        The router cannot tell them apart, but it has still ruled out the rest.
        "Bir latte kac lira" ties out_of_scope (no tools) with menu_search, so
        the menu tools are all the SLM needs; "mantı paylaşımı" ties posts with
        menu_search and gets both groups - five tools, not fifteen. Only when
        the union grows past :data:`MAX_MERGED` is the full prompt cheaper to
        trust.
        """
        margin = self.thresholds()[1]
        close = [k for k, v in scores if decision.score - v < margin]
        wanted: set[str] = set()
        for label in close:
            other = self.by_id.get(label)
            if other is None:
                return fallback(f"near-tie with unknown intent {label}")
            if other.route == "inherit":
                if previous is None:
                    return fallback("near-tie with a follow-up that has nothing to follow")
                wanted |= set(previous.tools)
            else:
                wanted |= set(other.tools)
        if not wanted or len(wanted) > MAX_MERGED:
            return fallback(
                f"margin {decision.margin:.3f} < {margin:.3f} "
                f"across {len(close)} intents ({len(wanted)} tools)"
            )
        decision.kind, decision.group = "merged", "+".join(close)
        decision.tools = tuple(n for n in ALL_TOOL_NAMES if n in wanted)
        decision.reason = (
            f"near-tie ({decision.margin:.3f}) - tools of {', '.join(close)}"
        )
        return decision
