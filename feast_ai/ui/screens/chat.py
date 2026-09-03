"""Chat screen: streaming conversation with live edge telemetry."""

from __future__ import annotations

import asyncio
import time

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Static

from ...engine import ChatClient, make_server
from ...engine.client import Message, ThinkFilter
from ...engine.server import EngineError
from ...metrics import SessionRecorder
from ...paths import human_bytes
from ..widgets import Meter, MiniSparkline, StatTile
from .common import ConfirmScreen, SystemPromptScreen

ROLE_STYLE = {
    "user": ("#4fc1ff", "you"),
    "assistant": ("#ffb454", ""),
    "system": ("#6f7789", "system"),
    "error": ("#ff5f5f", "error"),
}


class Bubble(Static):
    """One message in the transcript."""

    def __init__(self, role: str, speaker: str, text: str = "") -> None:
        super().__init__()
        self.role = role
        self.speaker = speaker
        self._text = text
        self._suffix = ""

    def on_mount(self) -> None:
        self.render_now()

    def set_text(self, text: str, suffix: str = "") -> None:
        self._text = text
        self._suffix = suffix
        self.render_now()

    def append(self, chunk: str) -> None:
        self._text += chunk

    def render_now(self) -> None:
        color, _ = ROLE_STYLE.get(self.role, ("#e6e9f0", ""))
        out = Text()
        out.append(f"{self.speaker}\n", style=f"bold {color}")
        out.append(self._text or ("…" if self.role == "assistant" else ""))
        if self._suffix:
            out.append(f"\n{self._suffix}", style="#545b6b")
        out.append("\n")
        self.update(out)


class ChatScreen(Screen):
    BINDINGS = [
        ("escape", "leave", "Back"),
        ("ctrl+e", "end_session", "End & report"),
        ("ctrl+p", "edit_prompt", "System prompt"),
        ("ctrl+t", "toggle_thinking", "Thinking"),
        ("ctrl+x", "stop_generation", "Stop"),
        ("ctrl+l", "clear_chat", "Clear"),
        ("f2", "open_settings", "Settings"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.recorder = SessionRecorder()
        self.history: list[Message] = []
        self.client: ChatClient | None = None
        self.generating = False
        self._stop = False
        self._timer = None
        self._live_tps = 0.0
        self._live_tokens = 0
        self._first_token_at = 0.0
        self._pending_render: Bubble | None = None
        self._ready = False

    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="chat-body"):
            with Vertical(id="chat-main"):
                yield VerticalScroll(id="transcript")
                yield Static(id="chat-status")
                with Vertical(id="composer"):
                    yield Input(
                        placeholder="Mesajini yaz…  (Enter gonder · Ctrl+E rapor · Ctrl+P system prompt)",
                        id="chat-input",
                    )
            with Vertical(id="telemetry"):
                yield Static("LIVE TELEMETRY", classes="telemetry-head")
                with Vertical(classes="telemetry-section"):
                    yield Meter("CPU", id="m-cpu")
                    yield Meter("RAM", id="m-ram", suffix=" MB", max_value=100)
                    yield MiniSparkline("cpu", width=24, color="#ff8a3d", id="sp-cpu")
                    yield MiniSparkline("ram", width=24, color="#4fc1ff", id="sp-ram")
                    yield MiniSparkline("t/s", width=24, color="#7bd88f", id="sp-tps")
                    # Only meaningful under a RAM ceiling; hidden otherwise so
                    # the sidebar does not carry a permanently flat line.
                    yield MiniSparkline("pg/s", width=24, color="#ff5f5f", id="sp-page")
                with Grid(classes="tiles"):
                    yield StatTile("tokens/s", "—", id="t-tps")
                    yield StatTile("TTFT", "—", "s", id="t-ttft")
                    yield StatTile("RAM peak", "—", id="t-ram")
                    yield StatTile("tokens out", "0", id="t-tokens")
                yield Static(id="paging-facts", classes="paging-facts")
                yield Static("DEVICE", classes="telemetry-head")
                yield Static(id="device-facts")
                yield Static("MODEL", classes="telemetry-head")
                yield Static(id="model-facts")
        yield Footer()

    # ------------------------------------------------------------------
    def on_mount(self) -> None:
        self.query_one("#chat-input", Input).disabled = True
        self.update_side_facts()
        self.boot_engine()

    async def on_screen_suspend(self) -> None:
        if self._timer is not None:
            self._timer.pause()

    def on_screen_resume(self) -> None:
        if self._timer is not None:
            self._timer.resume()
        self.update_side_facts()

    # ------------------------------------------------------------------
    def status(self, markup: str) -> None:
        self.query_one("#chat-status", Static).update(markup)

    def add_bubble(self, role: str, speaker: str, text: str = "") -> Bubble:
        bubble = Bubble(role, speaker, text)
        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.mount(bubble)
        transcript.scroll_end(animate=False)
        return bubble

    # ------------------------------------------------------------------
    @work(exclusive=True)
    async def boot_engine(self) -> None:
        """Start the model's backend under the CPU and RAM budget."""
        app = self.app
        spec = app.active_model
        if spec is None:
            self.status("[#ff5f5f]No model selected.[/]")
            return
        # The model decides its own backend, so a GGUF can never be handed to
        # ONNX Runtime or vice versa.
        if not spec.is_onnx and app.runtime.server is None:
            self.status("[#ff5f5f]llama.cpp runtime is not installed.[/]")
            self.add_bubble(
                "error", "engine",
                "The llama.cpp runtime is missing. Press R on the welcome screen "
                "to download it, or run: python -m feast_ai runtime",
            )
            return

        await app.shutdown_engine()
        try:
            server = make_server(spec.engine, app.settings, app.runtime.server)
        except EngineError as exc:
            self.status("[#ff5f5f]Engine unavailable.[/]")
            self.add_bubble("error", "engine", str(exc))
            return
        app.server = server

        self.add_bubble(
            "system",
            "feast AI",
            f"Loading {spec.name} ({spec.quant}, {human_bytes(spec.local_size)}) on "
            f"{app.settings.cpu_budget_cores} core(s) via {server.engine_name}…",
        )

        def on_status(message: str) -> None:
            self.status(f"[#8b93a7]{message}[/]")

        try:
            report = await server.start(spec.local_path, alias=spec.id, on_status=on_status)
        except EngineError as exc:
            self.status("[#ff5f5f]Engine failed to start.[/]")
            self.add_bubble("error", "engine", str(exc))
            return

        app.sampler.set_budget(app.settings.cpu_budget_cores)
        app.sampler.reset()
        app.sampler.attach(server.snapshot_processes())
        self.client = ChatClient(server.base_url, app.settings, model=spec.id)

        pinned = report.pinned_cores
        ready = (
            f"Ready in {report.load_seconds:.1f}s · RSS after load "
            f"{human_bytes(report.rss_after_load)} · "
            + (f"pinned to cores {pinned}" if pinned else "affinity not pinned")
            + f" · {app.settings.context_size} token context."
        )
        if report.cap.applied:
            oversized = spec.local_size > report.cap.limit_bytes
            ready += (
                f"\nRAM ceiling {report.cap.limit_mb} MB enforced. "
                + (
                    "The weights do not fit, so they will stream through it — "
                    "expect a large throughput drop."
                    if oversized
                    else "The weights fit inside it, so no streaming is needed."
                )
            )
        elif app.settings.ram_limit_enabled:
            ready += f"\nRAM ceiling requested but not applied: {report.cap.detail}"
        self.add_bubble("system", "feast AI", ready)
        if report.cap_warning:
            self.add_bubble("system", "feast AI", report.cap_warning)

        self._ready = True
        self._timer = self.set_interval(
            app.settings.sample_interval_ms / 1000.0, self.tick
        )
        chat_input = self.query_one("#chat-input", Input)
        chat_input.disabled = False
        chat_input.focus()
        self.status(
            "[#7bd88f]Ready.[/] [#6f7789]Type a message. Ctrl+E ends the session "
            "and opens the report.[/]"
        )
        self.update_side_facts()

    # ------------------------------------------------------------------
    def tick(self) -> None:
        """Sample telemetry and repaint the sidebar."""
        app = self.app
        sampler = app.sampler
        sample = sampler.sample(busy=self.generating, tps=self._live_tps)

        cpu = self.query_one("#m-cpu", Meter)
        cpu.update_value(
            sample.cpu_device_pct,
            f"{sample.cpu_device_pct:.0f}%  ({sample.cpu_raw_pct/100:.1f}/{app.settings.cpu_budget_cores} core)",
        )

        nominal_mb = app.settings.nominal_ram_gb * 1024
        ram = self.query_one("#m-ram", Meter)
        rss_mb = sample.rss / (1024 * 1024)
        if nominal_mb > 0:
            ram.max_value = nominal_mb
            ram.update_value(
                rss_mb, f"{human_bytes(sample.rss)} / {app.settings.nominal_ram_gb:g} GB"
            )
        else:
            ram.max_value = max(1.0, sampler.total_ram / (1024 * 1024))
            ram.update_value(rss_mb, human_bytes(sample.rss))

        self.query_one("#sp-cpu", MiniSparkline).set_data(sampler.cpu_series, ceiling=100.0)
        self.query_one("#sp-ram", MiniSparkline).set_data(sampler.ram_series)
        self.query_one("#sp-tps", MiniSparkline).set_data(sampler.tps_series)

        self.query_one("#t-tps", StatTile).set_value(
            f"{self._live_tps:.1f}" if self._live_tps else "—",
            tone="#7bd88f" if self._live_tps >= 8 else "#ffb454",
        )
        self.query_one("#t-ram", StatTile).set_value(human_bytes(sampler.stats.peak_rss))
        self.query_one("#t-tokens", StatTile).set_value(
            str(sum(t.completion_tokens for t in self.recorder.turns) + self._live_tokens)
        )
        self.update_paging(sample)

        if self.generating and self._first_token_at:
            elapsed = time.monotonic() - self._first_token_at
            if elapsed > 0.15:
                self._live_tps = self._live_tokens / elapsed

        if self._pending_render is not None:
            self._pending_render.render_now()
            self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)
            self._pending_render = None

    # ------------------------------------------------------------------
    def update_paging(self, sample) -> None:
        """Show how hard the model is being paged through the RAM window.

        Under a ceiling, RSS pins flat at the limit and stops being informative
        - the fault rate is what actually reveals the pressure, so it only
        appears when a ceiling is in force.
        """
        app = self.app
        spark = self.query_one("#sp-page", MiniSparkline)
        facts = self.query_one("#paging-facts", Static)

        report = app.server.report if app.server else None
        if report is None or not report.cap.applied:
            spark.display = False
            facts.display = False
            return

        spark.display = True
        facts.display = True
        spark.set_data(app.sampler.fault_series)

        paged_mb = sample.fault_rate * 4096 / (1024 * 1024)
        text = Text()
        text.append(f"{'ceiling':<9}", style="#6f7789")
        text.append(f"{report.cap.limit_mb} MB\n", style="#ff5f5f")
        text.append(f"{'faults/s':<9}", style="#6f7789")
        text.append(f"{sample.fault_rate:,.0f}\n", style="#e6e9f0")
        text.append(f"{'paged/s':<9}", style="#6f7789")
        text.append(f"{paged_mb:,.0f} MB\n", style="#e6e9f0")
        text.append(f"{'disk/s':<9}", style="#6f7789")
        text.append(
            f"{sample.disk_read_rate / (1024 * 1024):,.0f} MB\n", style="#8b93a7"
        )
        facts.update(text)

    # ------------------------------------------------------------------
    def update_side_facts(self) -> None:
        app = self.app
        settings = app.settings
        device = settings.device

        facts = Text()

        def row(label: str, value: str, style: str = "#e6e9f0") -> None:
            facts.append(f"{label:<9}", style="#6f7789")
            facts.append(f"{value}\n", style=style)

        row("profile", device.name, "#4fc1ff")
        row("cores", f"{settings.cpu_budget_cores} of {app.sampler.host_cores}", "#7bd88f")
        row("affinity", "pinned" if settings.pin_affinity else "off")
        limit = settings.ram_limit_bytes
        report = app.server.report if app.server else None
        if limit and report is not None and report.cap.applied:
            row("ram cap", f"{report.cap.limit_mb} MB enforced", "#ff5f5f")
        elif limit:
            row("ram cap", f"{limit // (1024 * 1024)} MB — unavailable", "#ffb454")
        else:
            row(
                "ram cap",
                f"{settings.nominal_ram_gb:g} GB nominal"
                if settings.nominal_ram_gb
                else "none",
                "#8b93a7",
            )
        row("priority", settings.priority.replace("_", " "))
        self.query_one("#device-facts", Static).update(facts)

        spec = app.active_model
        model = Text()
        if spec is not None:
            model.append(f"{spec.name}\n", style="bold #ffb454")
            model.append(f"{spec.params}  ·  {spec.quant}\n", style="#8b93a7")
            model.append(f"ctx {settings.context_size}  ·  max {settings.max_tokens}\n", style="#6f7789")
            model.append(
                f"temp {settings.temperature:g}  top_p {settings.top_p:g}\n", style="#6f7789"
            )
            think_tone = {"off": "#7bd88f", "on": "#ffb454", "auto": "#6f7789"}[
                settings.thinking_mode
            ]
            model.append("thinking ", style="#6f7789")
            model.append(
                f"{settings.thinking_mode}",
                style=f"bold {think_tone}",
            )
            model.append("  (Ctrl+T)\n", style="#3a3f4b")
            prompt = " ".join(settings.system_prompt.split())
            model.append(
                "system: " + (prompt[:70] + "…" if len(prompt) > 70 else prompt or "(none)"),
                style="#545b6b",
            )
        self.query_one("#model-facts", Static).update(model)

    # ------------------------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or not self._ready:
            return
        if self.generating:
            self.notify("Still generating - Ctrl+X stops it.", severity="warning")
            return
        event.input.value = ""
        self.send(text)

    @work(exclusive=True)
    async def send(self, text: str) -> None:
        if self.client is None:
            return
        app = self.app
        settings = app.settings

        # The engine belongs to the app, not to this screen. If it is gone the
        # request would fail as an opaque "connection refused", so say what
        # actually happened instead.
        if not app.engine_running or app.server is None:
            self.add_bubble(
                "error",
                "engine",
                "The inference server is no longer running, so this screen is "
                "stale. Go back to the model list and load a model again.",
            )
            self.status("[#ff5f5f]Engine not running.[/]")
            return
        if self.client.base_url != app.server.base_url:
            self.client = ChatClient(
                app.server.base_url,
                settings,
                model=app.active_model.id if app.active_model else "feast",
            )

        self.add_bubble("user", "you", text)
        bubble = self.add_bubble("assistant", app.active_model.name if app.active_model else "model")

        turn = self.recorder.start_turn(text, thinking_mode=settings.thinking_mode)
        messages = self.client.build_messages(self.history, text)

        self.generating = True
        self._stop = False
        self._live_tokens = 0
        self._live_tps = 0.0
        self._first_token_at = 0.0
        started = time.monotonic()
        window_start = time.time()

        think = ThinkFilter()
        visible_parts: list[str] = []
        reasoning_chars = 0
        reasoning_tokens = 0
        usage: dict = {}
        timings: dict = {}
        error = ""

        self.status("[#ffb454]Generating…[/]  [#6f7789]Ctrl+X to stop[/]")

        try:
            async for event in self.client.stream(messages):
                if self._stop:
                    turn.stopped = True
                    break
                if event.kind == "error":
                    error = event.error
                    break
                if event.kind == "done":
                    usage, timings = event.usage, event.timings
                    break

                if event.kind == "reasoning":
                    reasoning_chars += len(event.text)
                    reasoning_tokens += 1
                    self._live_tokens += 1
                    if settings.show_thinking:
                        visible_parts.append(event.text)
                    if not self._first_token_at:
                        self._first_token_at = time.monotonic()
                        turn.ttft_s = self._first_token_at - started
                    continue

                visible, reasoned = think.feed(event.text)
                reasoning_chars += len(reasoned)
                if reasoned and not visible:
                    # Whole delta fell inside a <think> span - count it as a
                    # reasoning token rather than an answer token.
                    reasoning_tokens += 1
                if settings.show_thinking and reasoned:
                    visible_parts.append(reasoned)
                if visible:
                    visible_parts.append(visible)
                self._live_tokens += 1
                if not self._first_token_at:
                    self._first_token_at = time.monotonic()
                    turn.ttft_s = self._first_token_at - started

                bubble.set_text("".join(visible_parts))
                self._pending_render = bubble
        except asyncio.CancelledError:
            turn.stopped = True
        except Exception as exc:  # noqa: BLE001 - surfaced in the transcript
            error = str(exc)

        tail = think.flush()
        if tail:
            visible_parts.append(tail)
        answer = "".join(visible_parts).strip()
        finished = time.monotonic()

        # ---- fold the measurements into the turn ----------------------
        turn.total_s = finished - started
        turn.decode_s = max(1e-6, finished - (self._first_token_at or started))
        turn.response_chars = len(answer)
        turn.reasoning_chars = reasoning_chars
        turn.reasoning_tokens = reasoning_tokens
        turn.error = error

        turn.prompt_tokens = int(
            usage.get("prompt_tokens") or timings.get("prompt_n") or 0
        )
        turn.completion_tokens = int(
            usage.get("completion_tokens") or timings.get("predicted_n") or self._live_tokens
        )

        # llama.cpp's own timings are authoritative when present.
        if timings.get("predicted_per_second"):
            turn.decode_tps = float(timings["predicted_per_second"])
        elif turn.completion_tokens:
            turn.decode_tps = turn.completion_tokens / turn.decode_s
        if timings.get("prompt_per_second"):
            turn.prefill_tps = float(timings["prompt_per_second"])
        elif turn.prompt_tokens and turn.ttft_s > 0:
            turn.prefill_tps = turn.prompt_tokens / turn.ttft_s
        if turn.total_s > 0:
            turn.e2e_tps = turn.completion_tokens / turn.total_s

        # Take one final reading before closing the window. A short reply can
        # finish inside a single sample interval, and without this the turn
        # would be attributed 0% CPU simply because nothing was sampled.
        app.sampler.sample(busy=True, tps=turn.decode_tps)
        avg_cpu, peak_cpu, peak_rss = app.sampler.busy_stats(window_start)
        turn.avg_cpu_device = avg_cpu
        turn.peak_cpu_device = peak_cpu
        turn.peak_rss = peak_rss
        turn.faults, turn.disk_read_bytes, turn.peak_fault_rate = (
            app.sampler.paging_stats(window_start)
        )

        self.generating = False
        self._live_tps = turn.decode_tps
        self._live_tokens = 0

        if error:
            bubble.set_text(answer or "")
            self.add_bubble("error", "engine", error)
            self.status(f"[#ff5f5f]{error[:120]}[/]")
        else:
            if settings.keep_history:
                self.history.append(Message("user", text))
                self.history.append(Message("assistant", answer))
            thinking_note = (
                f" · {turn.reasoning_tokens} thinking ({turn.reasoning_share:.0f}%)"
                if turn.reasoning_tokens
                else ""
            )
            footer = (
                f"{turn.completion_tokens} tok · {turn.decode_tps:.1f} t/s · "
                f"TTFT {turn.ttft_s:.2f}s · prefill {turn.prefill_tps:.0f} t/s · "
                f"CPU {turn.avg_cpu_device:.0f}% · RAM {human_bytes(turn.peak_rss)}"
                + thinking_note
                + ("  [stopped]" if turn.stopped else "")
            )
            bubble.set_text(answer, footer)
            self.status(
                f"[#7bd88f]{turn.decode_tps:.1f} t/s[/] [#6f7789]· TTFT "
                f"{turn.ttft_s:.2f}s · {turn.completion_tokens} tokens · "
                f"CPU {turn.avg_cpu_device:.0f}% of budget[/]"
            )

        self.query_one("#t-ttft", StatTile).set_value(f"{turn.ttft_s:.2f}")
        self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)
        self.query_one("#chat-input", Input).focus()

    # ------------------------------------------------------------------
    def action_stop_generation(self) -> None:
        if self.generating:
            self._stop = True
            self.status("[#ffb454]Stopping…[/]")

    def action_toggle_thinking(self) -> None:
        """Cycle auto -> off -> on. Applies to the next message, no reload."""
        mode = self.app.settings.cycle_thinking()
        self.app.save_settings()
        self.update_side_facts()
        explain = {
            "auto": "each model's own default",
            "off": "reasoning suppressed - fastest replies",
            "on": "reasoning forced where the model supports it",
        }[mode]
        self.notify(f"Thinking: {mode} — {explain}", title="feast AI")
        self.status(
            f"[#4fc1ff]Thinking set to {mode}[/] [#6f7789]· {explain} · "
            "applies from the next message[/]"
        )

    def action_edit_prompt(self) -> None:
        def applied(result: str | None) -> None:
            if result is None:
                return
            self.app.settings.system_prompt = result
            self.app.save_settings()
            self.update_side_facts()
            self.add_bubble(
                "system", "feast AI", "System prompt updated. It applies from the next message."
            )
            self.notify("System prompt updated.")

        self.app.push_screen(SystemPromptScreen(self.app.settings.system_prompt), applied)

    def action_open_settings(self) -> None:
        self.app.push_screen("settings")

    def action_clear_chat(self) -> None:
        self.history.clear()
        for bubble in self.query("#transcript Bubble"):
            bubble.remove()
        self.add_bubble("system", "feast AI", "Context cleared. Metrics keep accumulating.")

    # ------------------------------------------------------------------
    def action_end_session(self) -> None:
        if self.generating:
            self.notify("Wait for the reply to finish, or press Ctrl+X.", severity="warning")
            return
        if not self.recorder.completed:
            self.notify("No completed turns yet - send a message first.", severity="warning")
            return
        self.build_summary()

    def build_summary(self) -> None:
        app = self.app
        spec = app.active_model
        report = app.server.report if app.server else None

        summary = self.recorder.summarize(
            model_id=spec.id if spec else "",
            model_name=spec.name if spec else "",
            quant=spec.quant if spec else "",
            profile=app.settings.device.name,
            engine=app.server.engine_name if app.server else "",
            threads=app.settings.cpu_budget_cores,
            pinned_cores=report.pinned_cores if report else [],
            context_size=app.settings.context_size,
            system_prompt=app.settings.system_prompt,
            thinking_mode=app.settings.thinking_mode,
            load_seconds=report.load_seconds if report else 0.0,
            model_file_bytes=spec.local_size if spec else 0,
            rss_after_load=report.rss_after_load if report else 0,
            nominal_ram_gb=app.settings.nominal_ram_gb,
            sampler_stats=app.sampler.stats,
            cpu_series=list(app.sampler.cpu_series),
            ram_series=list(app.sampler.ram_series),
            tps_series=list(app.sampler.tps_series),
            fault_series=list(app.sampler.fault_series),
            ram_limit_bytes=app.settings.ram_limit_bytes,
            ram_cap_method=report.cap.method if report and report.cap.applied else "",
            storage_mbps=app.settings.device.storage_mbps,
            storage_note=app.settings.device.storage_note,
        )
        app.last_summary = summary
        app.push_screen("summary")

    # ------------------------------------------------------------------
    def action_leave(self) -> None:
        if self.generating:
            self.notify("Generating - Ctrl+X first.", severity="warning")
            return

        def confirmed(result: bool | None) -> None:
            if result:
                self.close_session()

        if self.recorder.completed:
            self.app.push_screen(
                ConfirmScreen(
                    "End this session and unload the model?\n\n"
                    "[#6f7789]Ctrl+E instead if you want the summary report.[/]",
                    confirm_label="End session",
                ),
                confirmed,
            )
        else:
            self.close_session()

    @work
    async def close_session(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        await self.app.shutdown_engine()
        self.app.pop_screen()
