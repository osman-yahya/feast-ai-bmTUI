"""Run a memeval dataset through a model under a phone-class budget, headless.

One server, one model, one condition per run; every question becomes a row
in ``predictions.jsonl`` with the exact prompt, its size per section, the
reply, any tool call, and llama.cpp's own timings.  Scoring is a separate
step (:mod:`feast_ai.memeval.score`), so a run is paid for once and re-scored
as the judges improve.

Conditions (§5.1).  Only the MEMORY block differs between them - preamble,
rules, tools and context are byte-identical - so a difference in accuracy or
prefill is the memory's doing:

``none``    nothing remembered: the floor.
``oracle``  only the sessions that hold the evidence: the 2B model's reading
            ceiling (LongMemEval's oracle setting).  Abstention questions get
            an empty block - there is no evidence to give.
``full``    every past turn, raw: slide 7's "Her seyi koy".  It shows what the
            memory engine saves, and whether it even fits; a prompt that does
            not fit the context is recorded as ``overflow``, not truncated.
            Deliberately naive: turns behind an erased fact stay in, which is
            the KVKK failure the engine must not have.

``always``  the memory engine (:mod:`feast_ai.memeval.engine`) with no gate:
            consolidated facts and/or the top-k retrieved turns, every question.
``gated``   the same behind the intervention gate: recall questions get
            retrieved memory, recommendations the profile facts, the rest none.

The engine conditions take an ``arm`` - ``raw`` (retrieved turns), ``facts``
(consolidated facts) or ``facts+raw`` - and their stores are built once per
model by replaying each persona's sessions through the same server.

The model sees the persona's state as the agent prompt's CONTEXT block, with
the clock frozen at the question's ``asked_at`` - "geçen cuma" must resolve
against the dataset's date, not today's.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from ..paths import HOME
from .schema import Dataset, Question, load

EVALS_DIR = HOME / "evals"
CONDITIONS = ("none", "oracle", "full", "always", "gated")
ENGINE_CONDITIONS = ("always", "gated")
ARMS = ("raw", "facts", "facts+raw")
TOOL_MODES = ("off", "all")
DAY_EN = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

#: Identical in every condition. ASCII like the agent prompts; the rule on
#: absence is the one §3.5 insists on for hard facts.
MEMORY_RULES = (
    "MEMORY\n"
    "Past chats with this user, oldest first, one per line: handle | date | who | text.\n"
    "Use them only for questions about the user's own past, and cite the handle you "
    "rely on, e.g. (t3). If the answer is not in MEMORY, say the user never told you - "
    "never guess. Not finding an allergy here does not mean the user has none: say "
    "they have not mentioned one."
)


@dataclass
class Row:
    qid: str
    persona: str
    category: str
    condition: str
    tools: str
    question: str
    gold_answer: str
    reply: str = ""
    call: str = ""
    call_status: str = ""
    call_args: dict = field(default_factory=dict)
    gate: str = ""
    memory_turns: int = 0
    memory_handles: dict = field(default_factory=dict)   # t1 -> turn_id
    evidence_in_memory: float = 0.0                       # share of evidence turns shown
    tokens: dict = field(default_factory=dict)            # per prompt section
    prompt_tokens: int = 0
    cached_tokens: int = 0
    prefilled_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: float = 0.0
    latency_s: float = 0.0
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    error: str = ""


# ----------------------------------------------------------------------
# Prompt assembly
# ----------------------------------------------------------------------
def _stamp(ts: str) -> str:
    d = datetime.fromisoformat(ts)
    return f"{d:%Y-%m-%d} {DAY_EN[d.weekday()]}"


def memory_turns(ds: Dataset, q: Question, condition: str) -> list:
    sessions = ds.sessions.get(q.persona, [])
    if condition == "none":
        return []
    if condition == "oracle":
        wanted = set(q.evidence_turn_ids)
        sessions = [s for s in sessions if any(t.turn_id in wanted for t in s.turns)]
    if condition not in ("oracle", "full"):
        raise ValueError(f"condition '{condition}' needs the memory engine (step 4)")
    asked = datetime.fromisoformat(q.asked_at)
    return [t for s in sessions for t in s.turns if datetime.fromisoformat(t.ts) < asked]


def memory_block(turns: list) -> tuple[str, dict]:
    handles = {f"t{n}": t.turn_id for n, t in enumerate(turns, 1)}
    lines = [f"{h} | {_stamp(t.ts)} | {t.role} | {t.text}" for h, t in zip(handles, turns)]
    return MEMORY_RULES + "\n" + ("\n".join(lines) if lines else "(empty)"), handles


def agent_context(ds: Dataset, q: Question):
    from ..agent.context import AgentContext

    p = ds.personas[q.persona]
    asked = datetime.fromisoformat(q.asked_at)
    return AgentContext(
        lat=p.home_lat, lng=p.home_lng, label=f"{p.home_area}, İstanbul", is_gps=True,
        cached_restaurants=len(ds.world.restaurants), posts=len(p.posts),
        saved=len(p.saved_restaurants), friends=len(p.friends),
        now=f"{asked:%Y-%m-%d} {DAY_EN[asked.weekday()]} {asked:%H:%M}",
    )


def engine_memory(store, q: Question, cfg: "RunConfig", names: tuple[str, ...]) -> tuple[str, dict, str]:
    """MEMORY block, handles and gate decision from a consolidated store."""
    from .engine.assemble import assemble
    from .engine.gate import gate
    from .engine.retrieve import retrieve

    decision = gate(q.question) if cfg.condition == "gated" else "recall"
    if decision == "closed":
        return (*assemble([], [], 0), decision)
    facts = []
    if "facts" in cfg.arm:
        facts = store.open_facts() if decision == "recommend" else store.visible_facts()
    ranked = []
    if "raw" in cfg.arm:
        query = q.question + (" alerji diyet vegan vejetaryen sevmem" if decision == "recommend" else "")
        ranked = retrieve(store.episodes, query, q.asked_at, cfg.k, names)
    return (*assemble(facts, ranked, cfg.memory_budget), decision)


def sections(ds: Dataset, q: Question, condition: str, tools: str,
             memory: tuple[str, dict] | None = None) -> tuple[dict, dict]:
    """The system prompt in order, static first: preamble + tools never change,
    MEMORY changes per persona (full) or per question (oracle), CONTEXT last."""
    from ..agent.prompts import CALL_FORMAT, NO_TOOLS, PREAMBLE
    from ..agent.tools import ALL_TOOL_NAMES, render_tools

    context = agent_context(ds, q)
    block, handles = memory or memory_block(memory_turns(ds, q, condition))
    out = {"preamble": PREAMBLE}
    if tools == "all":
        out["tools"] = "TOOLS  (* = required)\n\n" + render_tools(ALL_TOOL_NAMES, context.tags)
        out["call_format"] = CALL_FORMAT
    else:
        out["rules"] = NO_TOOLS
    out["memory"] = block
    out["context"] = context.render()
    return out, handles


# ----------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------
@dataclass
class RunConfig:
    data: str
    model_id: str
    condition: str = "none"
    tools: str = "off"
    profile: str = ""
    cores: int = 0
    ram_gb: float = 0.0
    limit: int = 0
    personas: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    max_tokens: int = 256
    thinking: str = "off"
    arm: str = "facts+raw"          # engine conditions only
    k: int = 8                      # retrieved turns
    memory_budget: int = 900        # MEMORY tokens, approx


def select(ds: Dataset, cfg: RunConfig) -> list[Question]:
    qs = [q for q in ds.questions
          if (not cfg.personas or q.persona in cfg.personas)
          and (not cfg.categories or q.category in cfg.categories)]
    if cfg.limit:
        # Spread a small sample over categories rather than taking one persona's.
        by_cat: dict[str, list[Question]] = {}
        for q in qs:
            by_cat.setdefault(q.category, []).append(q)
        picked: list[Question] = []
        while len(picked) < cfg.limit and any(by_cat.values()):
            for cat in list(by_cat):
                if by_cat[cat] and len(picked) < cfg.limit:
                    picked.append(by_cat[cat].pop(0))
        qs = picked
    # Persona order keeps a persona's MEMORY block hot in the prompt cache.
    return sorted(qs, key=lambda q: (q.persona, q.qid))


async def run(cfg: RunConfig, log=print) -> Path:
    from .. import catalog
    from ..agent.bench import _generate, _sample_loop
    from ..agent.calls import check_reply
    from ..agent.tools import ALL_TOOL_NAMES
    from ..config import Settings
    from ..engine import ChatClient, make_server, runtime_info
    from ..engine.server import EngineError
    from ..metrics import Sampler

    if cfg.condition not in CONDITIONS:
        raise EngineError(f"condition '{cfg.condition}' is not available yet: {', '.join(CONDITIONS)}")
    ds = load(cfg.data)
    questions = select(ds, cfg)
    spec = catalog.get(cfg.model_id)
    if spec is None or not spec.downloaded:
        raise EngineError(f"model '{cfg.model_id}' is not downloaded: python -m feast_ai pull {cfg.model_id}")

    settings = Settings.from_dict(Settings.load().to_dict())
    if cfg.profile:
        settings.apply_profile(cfg.profile)
    if cfg.cores:
        settings.threads = cfg.cores
    if cfg.ram_gb:
        settings.nominal_ram_gb = cfg.ram_gb
        settings.ram_limit_enabled = True
        settings.ram_limit_mb = 0
    settings.mark_custom_if_needed()
    settings.thinking_mode = cfg.thinking
    settings.max_tokens = cfg.max_tokens
    settings.temperature = 0.0          # greedy: conditions differ by prompt only
    settings.keep_history = False

    engine = cfg.condition in ENGINE_CONDITIONS
    if engine and cfg.arm not in ARMS:
        raise EngineError(f"arm '{cfg.arm}' is not one of {', '.join(ARMS)}")
    cond = f"{cfg.condition}-{cfg.arm.replace('+', '')}" if engine else cfg.condition
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{cfg.model_id}-{cond}-{cfg.tools}"
    out = EVALS_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    meta = {"config": asdict(cfg), "run_id": run_id, "dataset": str(Path(cfg.data).resolve()),
            "questions": len(questions), "model_name": spec.name, "quant": spec.quant,
            "profile": settings.device.name, "threads": settings.cpu_budget_cores,
            "ram_limit_bytes": settings.ram_limit_bytes, "context_size": settings.context_size,
            "started_at": time.time()}

    server = make_server(spec.engine, settings, runtime_info().server)
    log(f"Loading {spec.name} on {settings.cpu_budget_cores} core(s), ctx {settings.context_size}…")
    report = await server.start(spec.local_path, alias=spec.id, on_status=lambda m: log(f"  {m}"))
    meta["load_s"] = report.load_seconds
    client = ChatClient(server.base_url, settings, model=spec.id)
    sampler = Sampler(budget_cores=settings.cpu_budget_cores, history_points=100_000)
    rows: list[Row] = []
    stores: dict = {}
    names: tuple[str, ...] = ()

    def prompt(q: Question) -> tuple[dict, dict, str]:
        if not engine:
            return (*sections(ds, q, cfg.condition, cfg.tools), "")
        block, handles, decision = engine_memory(stores[q.persona], q, cfg, names)
        return (*sections(ds, q, cfg.condition, cfg.tools, (block, handles)), decision)

    try:
        if engine:
            from .engine.consolidate import build_stores, facts_eval, memory_names

            names = memory_names(ds)
            t0 = time.perf_counter()
            stores = await build_stores(client, ds, cfg.model_id, HOME, log)
            meta["consolidate_s"] = time.perf_counter() - t0
            meta["facts_eval"] = facts_eval(ds, stores)
            log(f"Write path: {meta['facts_eval']}")
        # One unrecorded warm-up so the first row does not pay for paging weights in.
        first, _, _ = prompt(questions[0])
        await _generate(client, client.build_messages([], "merhaba",
                                                      system_prompt="\n\n".join(first.values())),
                        time.perf_counter())
        sampler.attach(server.snapshot_processes())
        stop = asyncio.Event()
        task = asyncio.create_task(_sample_loop(sampler, stop, 0.25))
        budget = settings.context_size - cfg.max_tokens
        with (out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
            for n, q in enumerate(questions, 1):
                parts, handles, decision = prompt(q)
                system = "\n\n".join(parts.values())
                row = Row(q.qid, q.persona, q.category, cfg.condition, cfg.tools, q.question,
                          q.gold_answer, gate=decision, memory_turns=len(handles),
                          memory_handles=handles)
                shown = set(handles.values())
                row.evidence_in_memory = (sum(t in shown for t in q.evidence_turn_ids)
                                          / len(q.evidence_turn_ids)) if q.evidence_turn_ids else 0.0
                for name, text in parts.items():
                    row.tokens[name] = await client.tokenize_count(text) or 0
                row.tokens["question"] = await client.tokenize_count(q.question) or 0
                if sum(row.tokens.values()) > budget:
                    row.error = f"overflow: {sum(row.tokens.values())} tokens > {budget}"
                else:
                    messages = client.build_messages([], q.question, system_prompt=system)
                    reply, stats = await _generate(client, messages, time.perf_counter())
                    row.reply = reply
                    for key in ("prompt_tokens", "cached_tokens", "prefilled_tokens",
                                "completion_tokens", "ttft_s", "latency_s", "prefill_tps",
                                "decode_tps"):
                        setattr(row, key, stats[key])
                    row.error = stats["error"]
                    if cfg.tools == "all":
                        check = check_reply(reply, ALL_TOOL_NAMES)
                        row.call, row.call_status, row.call_args = check.name, check.status, dict(check.args)
                rows.append(row)
                fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                fh.flush()
                mark = "ERR " if row.error else "    "
                log(f"{mark}[{n:>3}/{len(questions)}] {q.qid:<14} {q.category:<18} "
                    f"{row.prompt_tokens or sum(row.tokens.values()):>5} tok  {row.ttft_s:5.2f}s  "
                    f"{(row.call or row.reply)[:60]!r}")
        stop.set()
        await task
    finally:
        await server.stop()

    done = [r for r in rows if not r.error]
    meta.update(finished_at=time.time(), answered=len(done), overflow=sum("overflow" in r.error for r in rows),
                peak_rss=sampler.stats.peak_rss, cpu_avg=sampler.stats.avg_cpu_device,
                avg_prompt_tokens=sum(r.prompt_tokens for r in done) / len(done) if done else 0,
                avg_ttft_s=sum(r.ttft_s for r in done) / len(done) if done else 0)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
