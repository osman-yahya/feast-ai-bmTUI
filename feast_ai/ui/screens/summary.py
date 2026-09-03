"""Post-session report: throughput, resource usage and per-turn breakdown."""

from __future__ import annotations

import time

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Static

from ...metrics.session import save_json, save_markdown
from ...paths import human_bytes
from ..widgets import MiniSparkline, StatTile

VERDICT_COLOR = {
    "great": "#7bd88f",
    "good": "#7bd88f",
    "ok": "#4fc1ff",
    "warn": "#ffb454",
    "bad": "#ff5f5f",
    "neutral": "#6f7789",
}


class SummaryScreen(Screen):
    BINDINGS = [
        ("escape", "back", "Back to chat"),
        ("s", "export", "Save report"),
        ("n", "new_model", "Pick another model"),
        ("q", "quit_app", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="summary-body"):
            yield Static(id="verdict")
            with Grid(classes="summary-grid"):
                yield StatTile("avg tokens/s", "—", id="s-tps")
                yield StatTile("avg TTFT", "—", "s", id="s-ttft")
                yield StatTile("peak RAM", "—", id="s-ram")
                yield StatTile("avg CPU", "—", "%", id="s-cpu")
            with Horizontal(id="summary-cols"):
                yield Static(id="panel-throughput", classes="summary-panel")
                yield Static(id="panel-resources", classes="summary-panel")
                yield Static(id="panel-config", classes="summary-panel")
            yield Static("TIMELINE", classes="panel-title")
            yield MiniSparkline("cpu %", width=70, color="#ff8a3d", id="g-cpu")
            yield MiniSparkline("ram MB", width=70, color="#4fc1ff", id="g-ram")
            yield MiniSparkline("t/s", width=70, color="#7bd88f", id="g-tps")
            yield MiniSparkline("faults/s", width=70, color="#ff5f5f", id="g-page")
            yield Static("PER-TURN BREAKDOWN", classes="panel-title")
            yield DataTable(id="turn-table", zebra_stripes=True, cursor_type="row")
            yield Static(id="export-status", classes="dim")
        with Horizontal(id="summary-actions"):
            yield Button("Save report", id="export", variant="primary")
            yield Button("Back to chat", id="back")
            yield Button("Another model", id="new-model")
            yield Button("Quit", id="quit")
        yield Footer()

    # ------------------------------------------------------------------
    def on_mount(self) -> None:
        table = self.query_one("#turn-table", DataTable)
        for column, width in (
            ("#", 3),
            ("Prompt", 30),
            ("In", 6),
            ("Out", 6),
            ("Think", 12),
            ("TTFT s", 8),
            ("Decode t/s", 11),
            ("Prefill t/s", 12),
            ("CPU %", 7),
            ("Peak RAM", 10),
        ):
            table.add_column(column, width=width)
        self.render_summary()

    def on_screen_resume(self) -> None:
        self.render_summary()

    # ------------------------------------------------------------------
    def render_summary(self) -> None:
        summary = self.app.last_summary
        if summary is None:
            self.query_one("#verdict", Static).update("No session data.")
            return

        label, tone = summary.verdict
        color = VERDICT_COLOR.get(tone, "#e6e9f0")
        head = Text()
        head.append(f"{summary.model_name}", style="bold #ffb454")
        head.append(f"   {summary.quant}  ·  {human_bytes(summary.model_file_bytes)}\n", style="#6f7789")
        head.append(
            f"{summary.profile}  ·  {summary.threads} core(s)"
            + (f" pinned to {summary.pinned_cores}" if summary.pinned_cores else "")
            + f"  ·  {summary.turns} turns  ·  "
            + time.strftime("%H:%M", time.localtime(summary.started_at))
            + "\n\n",
            style="#8b93a7",
        )
        head.append(label, style=f"bold {color}")
        self.query_one("#verdict", Static).update(head)

        self.query_one("#s-tps", StatTile).set_value(f"{summary.tps_avg:.1f}", tone=color)
        self.query_one("#s-ttft", StatTile).set_value(f"{summary.ttft_avg:.2f}")
        self.query_one("#s-ram", StatTile).set_value(human_bytes(summary.ram_peak))
        self.query_one("#s-cpu", StatTile).set_value(f"{summary.cpu_avg:.0f}")

        # ---- throughput ------------------------------------------------
        text = Text()
        text.append("THROUGHPUT\n\n", style="bold #ffb454")

        def line(label: str, value: str, style: str = "#e6e9f0") -> None:
            text.append(f"{label:<22}", style="#6f7789")
            text.append(f"{value}\n", style=style)

        line("Decode avg", f"{summary.tps_avg:.2f} t/s", color)
        line("Decode best", f"{summary.tps_best:.2f} t/s")
        line("Decode worst", f"{summary.tps_worst:.2f} t/s")
        line("Prefill avg", f"{summary.prefill_tps_avg:.1f} t/s")
        line("End-to-end avg", f"{summary.e2e_tps_avg:.2f} t/s")
        text.append("\n")
        line("TTFT avg", f"{summary.ttft_avg:.2f} s")
        line("TTFT p50 / p95", f"{summary.ttft_p50:.2f} / {summary.ttft_p95:.2f} s")
        line("TTFT min / max", f"{summary.ttft_min:.2f} / {summary.ttft_max:.2f} s")
        text.append("\n")
        line("Tokens in / out", f"{summary.prompt_tokens} / {summary.completion_tokens}")
        if summary.reasoning_tokens:
            line(
                "Reasoning tokens",
                f"{summary.reasoning_tokens}  ({summary.reasoning_share:.0f}% of output)",
                "#ffb454",
            )
        line("Generation time", f"{summary.total_generation_s:.1f} s")
        line("Session wall time", f"{summary.wall_seconds:.0f} s")
        self.query_one("#panel-throughput", Static).update(text)

        # ---- resources -------------------------------------------------
        text = Text()
        text.append("RESOURCES\n\n", style="bold #ffb454")
        line("CPU avg (budget)", f"{summary.cpu_avg:.1f} %")
        line("CPU peak (budget)", f"{summary.cpu_peak:.1f} %",
             "#ff5f5f" if summary.cpu_peak > 95 else "#e6e9f0")
        line("CPU peak (host)", f"{summary.cpu_host_peak:.1f} %")
        text.append("\n")
        line("RAM peak (RSS)", human_bytes(summary.ram_peak))
        line("RAM avg (RSS)", human_bytes(summary.ram_avg))
        if summary.ram_peak_uss:
            line("RAM peak (USS)", human_bytes(summary.ram_peak_uss))
        line("RSS after load", human_bytes(summary.rss_after_load))
        headroom = summary.ram_headroom_pct
        if headroom is not None:
            line(
                "Of nominal device RAM",
                f"{headroom:.1f} %  of {summary.nominal_ram_gb:g} GB",
                "#ff5f5f" if headroom > 100 else "#7bd88f",
            )
            if headroom > 100 and not summary.capped:
                text.append(
                    "\nExceeded the profile's nominal RAM. Not enforced -\n"
                    "turn on the RAM ceiling in Settings to make the model\n"
                    "stream through this budget instead.\n",
                    style="#ffb454",
                )

        if summary.capped:
            text.append("\nMEMORY PRESSURE\n", style="bold #ff5f5f")
            line("Ceiling", f"{human_bytes(summary.ram_limit_bytes)}", "#ff5f5f")
            line("Enforced via", summary.ram_cap_method)
            if summary.streamed:
                line("Page faults", f"{summary.total_faults:,}")
                line("Paged back in", human_bytes(summary.paged_bytes))
                line("Weights re-read", f"{summary.stream_multiplier:.1f} x")
                line("Peak fault rate", f"{summary.peak_fault_rate:,.0f} /s")
                line(
                    "Hard-fault ratio",
                    f"{summary.hard_fault_ratio * 100:.0f} %",
                    "#7bd88f" if summary.hard_fault_ratio >= 0.5 else "#ffb454",
                )
            if summary.streamed and summary.storage_ceiling_tps > 0:
                text.append("\nON REAL HARDWARE\n", style="bold #ff5f5f")
                line("Paged per token", human_bytes(summary.bytes_paged_per_token))
                line(
                    "Device storage",
                    f"{summary.storage_mbps:,.0f} MB/s  ({summary.storage_note})",
                )
                line(
                    "Storage ceiling",
                    f"{summary.storage_ceiling_tps:.2f} t/s",
                    "#ff5f5f" if summary.io_bound else "#7bd88f",
                )
                line(
                    "Projected on device",
                    f"{summary.projected_device_tps:.2f} t/s",
                    "#ffb454",
                )
                text.append(
                    "\nMeasured here, but re-read from the host's page cache.\n"
                    "The page-in volume is exact, so dividing it by the\n"
                    "device's flash bandwidth bounds what it could sustain.\n"
                    if not summary.io_bound else
                    "\nFlash bandwidth, not the CPU, is the binding limit\n"
                    "for this model on this device.\n",
                    style="#6f7789",
                )
            mem_label, mem_tone = summary.memory_verdict
            text.append(
                "\n" + mem_label + "\n",
                style={"great": "#7bd88f", "warn": "#ffb454",
                       "ok": "#4fc1ff"}.get(mem_tone, "#8b93a7"),
            )
        text.append("\n")
        line("Model load time", f"{summary.load_seconds:.2f} s")
        self.query_one("#panel-resources", Static).update(text)

        # ---- config ----------------------------------------------------
        text = Text()
        text.append("CONFIGURATION\n\n", style="bold #ffb454")
        line("Device profile", summary.profile, "#4fc1ff")
        line("Engine", summary.engine or "llama.cpp",
             "#c792ea" if "ONNX" in summary.engine else "#8b93a7")
        line("CPU threads", str(summary.threads))
        line("Pinned cores", str(summary.pinned_cores) if summary.pinned_cores else "not pinned")
        line("Context size", f"{summary.context_size} tokens")
        line("Weights", human_bytes(summary.model_file_bytes))
        line("Quantisation", summary.quant)
        line(
            "Thinking mode",
            summary.thinking_mode,
            {"off": "#7bd88f", "on": "#ffb454"}.get(summary.thinking_mode, "#8b93a7"),
        )
        text.append("\nSYSTEM PROMPT\n", style="#6f7789")
        prompt = summary.system_prompt.strip() or "(none)"
        text.append(prompt[:400] + ("…" if len(prompt) > 400 else ""), style="#8b93a7")
        self.query_one("#panel-config", Static).update(text)

        # ---- timeline --------------------------------------------------
        self.query_one("#g-cpu", MiniSparkline).set_data(summary.cpu_series, ceiling=100.0)
        self.query_one("#g-ram", MiniSparkline).set_data(summary.ram_series)
        self.query_one("#g-tps", MiniSparkline).set_data(summary.tps_series)
        page = self.query_one("#g-page", MiniSparkline)
        page.display = summary.streamed
        if summary.streamed:
            page.set_data(summary.fault_series)

        # ---- turns -----------------------------------------------------
        table = self.query_one("#turn-table", DataTable)
        table.clear()
        for row in summary.turn_rows:
            preview = " ".join(str(row.get("prompt", "")).split())
            think_tokens = row.get("reasoning_tokens", 0)
            mode = row.get("thinking_mode", "auto")
            if think_tokens:
                share = 100.0 * think_tokens / max(1, row["completion_tokens"])
                think_cell = Text(f"{think_tokens} ({share:.0f}%)", style="#ffb454")
            else:
                think_cell = Text("off" if mode == "off" else "—", style="#6f7789")
            table.add_row(
                str(row["index"]),
                preview[:28] + ("…" if len(preview) > 28 else ""),
                str(row["prompt_tokens"]),
                str(row["completion_tokens"]),
                think_cell,
                f"{row['ttft_s']:.2f}",
                Text(f"{row['decode_tps']:.2f}", style="#7bd88f"),
                f"{row['prefill_tps']:.1f}",
                f"{row['avg_cpu_device']:.0f}",
                human_bytes(row["peak_rss"]),
            )

        if self.app.settings.auto_save_sessions:
            self.export(quiet=True)

    # ------------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        handlers = {
            "export": self.action_export,
            "back": self.action_back,
            "new-model": self.action_new_model,
            "quit": self.action_quit_app,
        }
        handler = handlers.get(event.button.id or "")
        if handler:
            handler()

    def action_export(self) -> None:
        self.export(quiet=False)

    def export(self, quiet: bool) -> None:
        summary = self.app.last_summary
        if summary is None:
            return
        try:
            json_path = save_json(summary)
            md_path = save_markdown(summary)
        except OSError as exc:
            self.query_one("#export-status", Static).update(f"[#ff5f5f]Export failed: {exc}[/]")
            return
        self.query_one("#export-status", Static).update(
            f"Saved [#7bd88f]{md_path.name}[/] and [#7bd88f]{json_path.name}[/] "
            f"in [#545b6b]{md_path.parent}[/]"
        )
        if not quiet:
            self.notify(f"Report saved to {md_path}", title="feast AI", timeout=8)

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_new_model(self) -> None:
        """Unload the model and return to the catalog."""
        self.app.call_later(self._switch_model)

    async def _switch_model(self) -> None:
        await self.app.shutdown_engine()
        # summary -> chat -> models
        while len(self.app.screen_stack) > 2:
            self.app.pop_screen()

    def action_quit_app(self) -> None:
        self.app.call_later(self.app.action_quit)
