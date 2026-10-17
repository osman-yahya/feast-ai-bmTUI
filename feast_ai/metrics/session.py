"""Per-turn and per-session throughput accounting.

Definitions used throughout feast AI (matching llama.cpp's own vocabulary):

TTFT
    Time To First Token - wall clock from sending the request to the first
    visible token. On CPU this is dominated by prompt processing.

Prefill t/s
    Prompt tokens / prompt-eval time. How fast the device *reads*.

Decode t/s (TPS)
    Generated tokens / generation time, measured from first token onward.
    This is the number users perceive as "speed".

E2E t/s
    Generated tokens / total wall clock including TTFT. The honest
    end-to-end figure for short answers on slow devices.

Latency
    Wall clock from sending the message to the finished answer, router
    included.  The one figure that compares a turn the SLM served with one
    the router answered alone (which has no tokens/s at all).
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..paths import SESSIONS_DIR


@dataclass
class TurnStats:
    """Everything measured for one user -> assistant exchange."""

    index: int
    prompt: str = ""
    response_chars: int = 0
    thinking_mode: str = "auto"

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_chars: int = 0

    ttft_s: float = 0.0
    total_s: float = 0.0
    decode_s: float = 0.0

    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    e2e_tps: float = 0.0

    avg_cpu_device: float = 0.0
    peak_cpu_device: float = 0.0
    peak_rss: int = 0

    # Paging cost, non-zero only when a RAM ceiling forced the model to stream.
    faults: int = 0
    disk_read_bytes: int = 0
    peak_fault_rate: float = 0.0

    # Agent pipeline (see feast_ai.agent). Empty/zero when agent_mode is off.
    agent_mode: str = "off"
    route_intent: str = ""
    route_score: float = 0.0
    route_margin: float = 0.0
    route_kind: str = ""        # "direct" | "slm" | "fallback" | "inherit"
    prompt_name: str = ""       # "full" | "intent:<id>" | "chat"
    tools_offered: int = 0
    router_ms: float = 0.0      # wall clock, includes the localhost hop
    encoder_ms: float = 0.0     # inference only - what a phone would pay
    cached_tokens: int = 0      # prompt tokens reused from llama.cpp's cache
    tool_call: str = ""
    call_status: str = ""       # "text" | "ok" | a hallucination class
    call_format: str = ""

    stopped: bool = False
    error: str = ""

    @property
    def served_direct(self) -> bool:
        """Answered by the router alone - no SLM tokens exist for this turn."""
        return self.route_kind == "direct"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def faults_per_token(self) -> float:
        """Pages faulted back in per generated token.

        Multiply by 4 KB to compare against the model size: reaching it means
        the entire weight set is being re-read for every single token.
        """
        if not self.completion_tokens:
            return 0.0
        return self.faults / self.completion_tokens

    @property
    def reasoning_share(self) -> float:
        """Percentage of generated tokens spent thinking rather than answering."""
        if not self.completion_tokens:
            return 0.0
        return 100.0 * self.reasoning_tokens / self.completion_tokens

    @property
    def preview(self) -> str:
        text = " ".join(self.prompt.split())
        return text[:40] + ("..." if len(text) > 40 else "")


@dataclass
class SessionSummary:
    """Aggregates over a whole chat session."""

    model_id: str = ""
    model_name: str = ""
    quant: str = ""
    profile: str = ""
    engine: str = ""
    threads: int = 0
    pinned_cores: list[int] = field(default_factory=list)
    context_size: int = 0
    system_prompt: str = ""
    thinking_mode: str = "auto"
    agent_mode: str = "off"
    router_model: str = ""
    router_rss: int = 0
    prompt_cache_ram_mb: int = 0

    started_at: float = 0.0
    ended_at: float = 0.0
    load_seconds: float = 0.0
    model_file_bytes: int = 0
    rss_after_load: int = 0

    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    ttft_avg: float = 0.0
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_min: float = 0.0
    ttft_max: float = 0.0

    tps_avg: float = 0.0
    tps_best: float = 0.0
    tps_worst: float = 0.0
    prefill_tps_avg: float = 0.0
    e2e_tps_avg: float = 0.0

    cpu_avg: float = 0.0
    cpu_peak: float = 0.0
    cpu_host_peak: float = 0.0
    ram_peak: int = 0
    ram_avg: float = 0.0
    ram_peak_uss: int = 0
    nominal_ram_gb: float = 0.0

    # --- RAM ceiling / streaming ----------------------------------------
    ram_limit_bytes: int = 0
    ram_cap_method: str = ""
    total_faults: int = 0
    total_disk_read: int = 0
    peak_fault_rate: float = 0.0
    storage_mbps: float = 0.0
    storage_note: str = ""

    total_generation_s: float = 0.0
    wall_seconds: float = 0.0

    # --- agent pipeline -------------------------------------------------
    direct_turns: int = 0
    latency_avg: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    router_ms_avg: float = 0.0
    encoder_ms_avg: float = 0.0
    calls_ok: int = 0
    calls_hallucinated: int = 0
    text_replies: int = 0

    turn_rows: list[dict] = field(default_factory=list)
    cpu_series: list[float] = field(default_factory=list)
    ram_series: list[float] = field(default_factory=list)
    tps_series: list[float] = field(default_factory=list)
    fault_series: list[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def reasoning_share(self) -> float:
        if not self.completion_tokens:
            return 0.0
        return 100.0 * self.reasoning_tokens / self.completion_tokens

    @property
    def ram_headroom_pct(self) -> float | None:
        """Peak RSS as a share of the simulated device's nominal RAM."""
        if self.nominal_ram_gb <= 0:
            return None
        return 100.0 * self.ram_peak / (self.nominal_ram_gb * 1024**3)

    @property
    def capped(self) -> bool:
        return self.ram_limit_bytes > 0 and bool(self.ram_cap_method)

    @property
    def streamed(self) -> bool:
        """Did the model actually have to be paged through the RAM window?

        A ceiling that the model fits inside comfortably produces almost no
        faults, and that is a legitimate (and fast) result - the device simply
        had enough memory.
        """
        return self.capped and self.total_faults > 100_000

    @property
    def paged_bytes(self) -> int:
        return self.total_faults * 4096

    @property
    def hard_fault_ratio(self) -> float:
        """Share of faulted pages that came off the disk rather than host RAM.

        The honest caveat on this simulation: on a host with spare memory the
        evicted (clean, file-backed) weight pages sit on the standby list and
        are re-faulted from RAM, so the run is faster than the phone it stands
        in for.  The closer this is to 1.0, the closer to real flash latency.
        """
        paged = self.paged_bytes
        if not paged:
            return 0.0
        return min(1.0, self.total_disk_read / paged)

    @property
    def stream_multiplier(self) -> float:
        """Bytes paged in per byte of weights, over the whole session.

        1.0x means the model was read once. 60x means it was re-read sixty
        times, which is what per-token streaming looks like.
        """
        if not self.model_file_bytes:
            return 0.0
        return self.paged_bytes / self.model_file_bytes

    @property
    def bytes_paged_per_token(self) -> float:
        if not self.completion_tokens:
            return 0.0
        return self.paged_bytes / self.completion_tokens

    @property
    def storage_ceiling_tps(self) -> float:
        """Tokens/s the *target device's* storage would allow, if it had to
        serve every page-in from flash.

        This is the answer to the fidelity caveat above rather than an excuse
        for it.  The host's spare RAM makes the measured run optimistic, but
        the page-in *volume* is measured exactly - so dividing it by a real
        device's sequential read bandwidth gives a defensible upper bound on
        what that device could sustain, independent of how fast its CPU is.
        """
        per_token = self.bytes_paged_per_token
        if per_token <= 0 or self.storage_mbps <= 0:
            return 0.0
        return (self.storage_mbps * 1024 * 1024) / per_token

    @property
    def projected_device_tps(self) -> float:
        """Measured throughput, further limited by the device's storage.

        Whichever of compute and I/O is slower sets the pace, so this is the
        smaller of the two.
        """
        ceiling = self.storage_ceiling_tps
        if not self.streamed or ceiling <= 0:
            return self.tps_avg
        return min(self.tps_avg, ceiling)

    @property
    def io_bound(self) -> bool:
        """True when the device's flash, not its CPU, is the binding limit."""
        return self.streamed and 0 < self.storage_ceiling_tps < self.tps_avg

    @property
    def agent_used(self) -> bool:
        return self.agent_mode not in ("", "off")

    @property
    def verdict(self) -> tuple[str, str]:
        """A plain-language judgement: ``(label, css_class)``."""
        if self.turns == 0:
            return "No data", "neutral"
        if self.direct_turns == self.turns:
            return "Every turn served by the router alone - no SLM ran", "great"
        tps = self.tps_avg
        if tps >= 25:
            return "Excellent - feels instant on this device class", "great"
        if tps >= 12:
            return "Good - comfortably usable for chat", "good"
        if tps >= 6:
            return "Usable - noticeable wait, fine for short answers", "ok"
        if tps >= 2:
            return "Sluggish - only viable for one-shot tasks", "warn"
        limit = "RAM" if self.streamed else "CPU"
        return f"Impractical - model is too heavy for this {limit} budget", "bad"

    @property
    def memory_verdict(self) -> tuple[str, str]:
        """How the model coped with the RAM ceiling: ``(label, css_class)``."""
        if not self.capped:
            return "Uncapped - model held fully resident", "neutral"
        if not self.streamed:
            return "Fits - model stayed inside the RAM budget", "great"
        ratio = self.hard_fault_ratio
        multiplier = self.stream_multiplier
        if ratio >= 0.5:
            return (
                f"Streaming from disk - weights re-read {multiplier:.0f}x, "
                "storage-bound like a real device", "warn",
            )
        if ratio >= 0.05:
            return (
                f"Streaming - weights re-read {multiplier:.0f}x, partly "
                "absorbed by the host page cache", "warn",
            )
        return (
            f"Streaming - weights re-read {multiplier:.0f}x, but served from "
            "host RAM; a real device would be slower", "ok",
        )


class SessionRecorder:
    """Accumulates :class:`TurnStats` and folds them into a summary."""

    def __init__(self) -> None:
        self.turns: list[TurnStats] = []
        self.started_at = time.time()
        self.meta: dict = {}

    # ------------------------------------------------------------------
    def start_turn(
        self, prompt: str, thinking_mode: str = "auto", agent_mode: str = "off"
    ) -> TurnStats:
        turn = TurnStats(
            index=len(self.turns) + 1, prompt=prompt, thinking_mode=thinking_mode,
            agent_mode=agent_mode,
        )
        self.turns.append(turn)
        return turn

    def drop_last(self) -> None:
        if self.turns:
            self.turns.pop()

    @property
    def completed(self) -> list[TurnStats]:
        return [
            t for t in self.turns
            if (t.completion_tokens > 0 or t.served_direct) and not t.error
        ]

    # ------------------------------------------------------------------
    def summarize(
        self,
        model_id: str,
        model_name: str,
        quant: str,
        profile: str,
        engine: str,
        threads: int,
        pinned_cores: list[int],
        context_size: int,
        system_prompt: str,
        thinking_mode: str,
        load_seconds: float,
        model_file_bytes: int,
        rss_after_load: int,
        nominal_ram_gb: float,
        sampler_stats,
        cpu_series: list[float],
        ram_series: list[float],
        tps_series: list[float],
        fault_series: list[float] | None = None,
        ram_limit_bytes: int = 0,
        ram_cap_method: str = "",
        storage_mbps: float = 0.0,
        storage_note: str = "",
        agent_mode: str = "off",
        router_model: str = "",
        router_rss: int = 0,
        prompt_cache_ram_mb: int = 0,
    ) -> SessionSummary:
        done = self.completed
        ttfts = [t.ttft_s for t in done if t.ttft_s > 0]
        tpss = [t.decode_tps for t in done if t.decode_tps > 0]
        prefills = [t.prefill_tps for t in done if t.prefill_tps > 0]
        e2es = [t.e2e_tps for t in done if t.e2e_tps > 0]

        summary = SessionSummary(
            model_id=model_id,
            model_name=model_name,
            quant=quant,
            profile=profile,
            engine=engine,
            threads=threads,
            pinned_cores=list(pinned_cores),
            context_size=context_size,
            # The mode can be toggled mid-session with Ctrl+T, so report what
            # the turns actually ran under rather than whatever is set now.
            thinking_mode=_modes_used(done, thinking_mode),
            agent_mode=_modes_used(done, agent_mode, attr="agent_mode"),
            router_model=router_model,
            router_rss=router_rss,
            prompt_cache_ram_mb=prompt_cache_ram_mb,
            system_prompt=system_prompt,
            started_at=self.started_at,
            ended_at=time.time(),
            load_seconds=load_seconds,
            model_file_bytes=model_file_bytes,
            rss_after_load=rss_after_load,
            nominal_ram_gb=nominal_ram_gb,
            ram_limit_bytes=ram_limit_bytes,
            ram_cap_method=ram_cap_method,
            storage_mbps=storage_mbps,
            storage_note=storage_note,
            turns=len(done),
            prompt_tokens=sum(t.prompt_tokens for t in done),
            completion_tokens=sum(t.completion_tokens for t in done),
            reasoning_tokens=sum(t.reasoning_tokens for t in done),
            total_generation_s=sum(t.decode_s for t in done),
            turn_rows=[asdict(t) for t in self.turns],
            cpu_series=list(cpu_series),
            ram_series=list(ram_series),
            tps_series=[v for v in tps_series if v > 0],
            fault_series=list(fault_series or []),
        )
        summary.wall_seconds = summary.ended_at - summary.started_at

        if ttfts:
            summary.ttft_avg = statistics.fmean(ttfts)
            summary.ttft_p50 = statistics.median(ttfts)
            summary.ttft_p95 = _percentile(ttfts, 95)
            summary.ttft_min = min(ttfts)
            summary.ttft_max = max(ttfts)
        if tpss:
            summary.tps_avg = statistics.fmean(tpss)
            summary.tps_best = max(tpss)
            summary.tps_worst = min(tpss)
        if prefills:
            summary.prefill_tps_avg = statistics.fmean(prefills)
        if e2es:
            summary.e2e_tps_avg = statistics.fmean(e2es)

        latencies = [t.total_s for t in done if t.total_s > 0]
        if latencies:
            summary.latency_avg = statistics.fmean(latencies)
            summary.latency_p50 = statistics.median(latencies)
            summary.latency_p95 = _percentile(latencies, 95)
        routed = [t for t in done if t.route_kind]
        if routed:
            summary.router_ms_avg = statistics.fmean(t.router_ms for t in routed)
            summary.encoder_ms_avg = statistics.fmean(t.encoder_ms for t in routed)
        summary.direct_turns = sum(1 for t in done if t.served_direct)
        summary.calls_ok = sum(1 for t in done if t.call_status == "ok")
        summary.text_replies = sum(1 for t in done if t.call_status == "text")
        summary.calls_hallucinated = sum(
            1 for t in done if t.call_status not in ("", "ok", "text")
        )

        if sampler_stats is not None:
            summary.cpu_avg = sampler_stats.avg_cpu_device
            summary.cpu_peak = sampler_stats.peak_cpu_device
            summary.cpu_host_peak = sampler_stats.peak_cpu_host
            summary.ram_peak = sampler_stats.peak_rss
            summary.ram_avg = sampler_stats.avg_rss
            summary.ram_peak_uss = sampler_stats.peak_uss
            summary.total_faults = getattr(sampler_stats, "total_faults", 0)
            summary.total_disk_read = getattr(sampler_stats, "total_disk_read", 0)
            summary.peak_fault_rate = getattr(sampler_stats, "peak_fault_rate", 0.0)
        return summary


def _modes_used(turns: list[TurnStats], fallback: str, attr: str = "thinking_mode") -> str:
    """``"off"`` if every turn ran that way, ``"mixed (auto, off)"`` if not."""
    modes = sorted({getattr(t, attr) for t in turns})
    if not modes:
        return fallback
    if len(modes) == 1:
        return modes[0]
    return "mixed (" + ", ".join(modes) + ")"


def _cap_label(summary: SessionSummary) -> str:
    from ..paths import human_bytes

    if not summary.ram_limit_bytes:
        return "none (model held fully resident)"
    if not summary.ram_cap_method:
        return f"{human_bytes(summary.ram_limit_bytes)} requested, **not applied**"
    return f"{human_bytes(summary.ram_limit_bytes)} enforced via {summary.ram_cap_method}"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def save_json(summary: SessionSummary) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(summary.started_at))
    path = SESSIONS_DIR / f"{stamp}-{summary.model_id}.json"
    path.write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def save_markdown(summary: SessionSummary) -> Path:
    from ..paths import human_bytes

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(summary.started_at))
    path = SESSIONS_DIR / f"{stamp}-{summary.model_id}.md"
    verdict, _ = summary.verdict

    lines = [
        f"# feast AI session report - {summary.model_name}",
        "",
        f"*{time.strftime('%Y-%m-%d %H:%M', time.localtime(summary.started_at))}*",
        "",
        "## Configuration",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Model | `{summary.model_id}` ({summary.quant}) |",
        f"| Engine | {summary.engine or 'llama.cpp'} |",
        f"| Weights on disk | {human_bytes(summary.model_file_bytes)} |",
        f"| Device profile | {summary.profile} |",
        f"| CPU budget | {summary.threads} thread(s), pinned to {summary.pinned_cores or 'not pinned'} |",
        f"| Nominal device RAM | {summary.nominal_ram_gb or '-'} GB |",
        f"| RAM ceiling | {_cap_label(summary)} |",
        f"| Context | {summary.context_size} tokens |",
        f"| Prompt-cache RAM | {summary.prompt_cache_ram_mb or 'off'}"
        f"{' MB' if summary.prompt_cache_ram_mb > 0 else ''} |",
        f"| Thinking | {summary.thinking_mode} |",
        f"| Agent mode | {summary.agent_mode} |",
        f"| Load time | {summary.load_seconds:.2f} s |",
        "",
        "## Throughput",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Turns | {summary.turns} |",
        f"| Decode t/s (avg) | **{summary.tps_avg:.2f}** |",
        f"| Decode t/s (best / worst) | {summary.tps_best:.2f} / {summary.tps_worst:.2f} |",
        f"| Prefill t/s (avg) | {summary.prefill_tps_avg:.2f} |",
        f"| End-to-end t/s (avg) | {summary.e2e_tps_avg:.2f} |",
        f"| TTFT avg / p50 / p95 | {summary.ttft_avg:.2f} / {summary.ttft_p50:.2f} / {summary.ttft_p95:.2f} s |",
        f"| Tokens in / out | {summary.prompt_tokens} / {summary.completion_tokens} |",
        f"| Reasoning tokens | {summary.reasoning_tokens} "
        f"({summary.reasoning_share:.0f}% of output) |",
        f"| Generation time | {summary.total_generation_s:.1f} s of {summary.wall_seconds:.1f} s wall |",
        "",
        "## Resource usage",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| CPU avg (of device budget) | {summary.cpu_avg:.1f}% |",
        f"| CPU peak (of device budget) | {summary.cpu_peak:.1f}% |",
        f"| CPU peak (of whole host) | {summary.cpu_host_peak:.1f}% |",
        f"| RAM peak (RSS) | {human_bytes(summary.ram_peak)} |",
        f"| RAM avg (RSS) | {human_bytes(summary.ram_avg)} |",
        f"| RAM peak (USS) | {human_bytes(summary.ram_peak_uss)} |",
        f"| RSS right after load | {human_bytes(summary.rss_after_load)} |",
    ]
    headroom = summary.ram_headroom_pct
    if headroom is not None:
        lines.append(f"| Share of nominal device RAM | {headroom:.1f}% |")

    if summary.capped:
        memory_verdict, _ = summary.memory_verdict
        lines += [
            "",
            "## Memory pressure",
            "",
            "The RAM ceiling is a *working-set* limit, so exceeding it makes the OS "
            "evict and re-fault pages rather than killing the process. These numbers "
            "are what that cost.",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Ceiling | {human_bytes(summary.ram_limit_bytes)} ({summary.ram_cap_method}) |",
            f"| Page faults | {summary.total_faults:,} |",
            f"| Paged back in | {human_bytes(summary.paged_bytes)} |",
            f"| Weights re-read | {summary.stream_multiplier:.1f}x the model file |",
            f"| Peak fault rate | {summary.peak_fault_rate:,.0f} /s |",
            f"| Disk reads (system-wide) | {human_bytes(summary.total_disk_read)} |",
            f"| Hard-fault ratio | {summary.hard_fault_ratio * 100:.1f}% |",
            "",
            f"**Memory verdict:** {memory_verdict}",
            "",
            "> Hard-fault ratio is the fidelity caveat: pages the host served from "
            "its own page cache instead of the disk made this run faster than the "
            "device it simulates. 100% would mean every re-read hit storage.",
        ]
        if summary.streamed and summary.storage_ceiling_tps > 0:
            lines += [
                "",
                "### Projected on real hardware",
                "",
                "The page-in *volume* is measured exactly even when the host serves "
                "it from RAM, so dividing it by the target device's sequential read "
                "bandwidth bounds what that device could sustain.",
                "",
                "| Metric | Value |",
                "| --- | --- |",
                f"| Paged in per token | {human_bytes(summary.bytes_paged_per_token)} |",
                f"| Device storage | {summary.storage_mbps:,.0f} MB/s "
                f"({summary.storage_note}) |",
                f"| Storage-bound ceiling | {summary.storage_ceiling_tps:.2f} t/s |",
                f"| Measured here | {summary.tps_avg:.2f} t/s |",
                f"| **Projected on device** | **{summary.projected_device_tps:.2f} t/s** |",
                "",
                (
                    "Storage is the binding limit: this model cannot stream fast "
                    "enough on this device's flash, regardless of CPU."
                    if summary.io_bound
                    else "The CPU budget still binds before storage does."
                ),
            ]

    if summary.agent_used:
        lines += [
            "",
            "## Agent pipeline",
            "",
            "`direct` shows the SLM every tool; `hierarchical` routes first with "
            f"{summary.router_model or 'the encoder'} and shows only the intent's "
            "tools, or answers with no SLM at all. Latency is end-to-end, router "
            "included.",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Agent mode | {summary.agent_mode} |",
            f"| Turns served without the SLM | {summary.direct_turns} of {summary.turns} |",
            f"| Latency avg / p50 / p95 | {summary.latency_avg:.2f} / "
            f"{summary.latency_p50:.2f} / {summary.latency_p95:.2f} s |",
            f"| Router avg (wall / encoder) | {summary.router_ms_avg:.1f} / "
            f"{summary.encoder_ms_avg:.1f} ms |",
            f"| Router RSS after load | {human_bytes(summary.router_rss)} |",
            f"| Valid tool calls | {summary.calls_ok} |",
            f"| Hallucinated calls | {summary.calls_hallucinated} |",
            f"| Plain-text replies | {summary.text_replies} |",
        ]

    lines += ["", f"**Verdict:** {verdict}", "", "## Turns", "",
              "| # | Prompt | Think | Route | Call | In | Cached | Out | Reasoning "
              "| TTFT s | Latency s | Decode t/s | CPU % | Peak RAM |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
              "| --- | --- | --- |"]
    for row in summary.turn_rows:
        preview = " ".join(str(row.get("prompt", "")).split())[:40].replace("|", "/")
        think_tokens = row.get("reasoning_tokens", 0)
        share = 100.0 * think_tokens / max(1, row["completion_tokens"])
        reasoning = f"{think_tokens} ({share:.0f}%)" if think_tokens else "-"
        route = (
            f"{row.get('route_intent')} ({row.get('route_kind')})"
            if row.get("route_kind") else row.get("agent_mode", "off")
        )
        call = (
            f"{row.get('tool_call') or '-'} {row.get('call_status')}".strip()
            if row.get("call_status") else "-"
        )
        lines.append(
            f"| {row['index']} | {preview} | {row.get('thinking_mode', 'auto')} "
            f"| {route} | {call} "
            f"| {row['prompt_tokens']} | {row.get('cached_tokens', 0)} "
            f"| {row['completion_tokens']} | {reasoning} "
            f"| {row['ttft_s']:.2f} | {row.get('total_s', 0.0):.2f} | {row['decode_tps']:.2f} "
            f"| {row['avg_cpu_device']:.0f} | {human_bytes(row['peak_rss'])} |"
        )

    if summary.system_prompt:
        lines += ["", "## System prompt", "", "```", summary.system_prompt, "```"]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
