"""Settings: edge device simulation, sampling parameters and system prompt."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    Switch,
    TextArea,
)

from ...config import (
    AGENT_HELP,
    AGENT_LABELS,
    AGENT_MODES,
    DEVICE_PROFILES,
    HOST_CORES,
    PRIORITY_CHOICES,
    PROFILE_ORDER,
    QUANT_PREFERENCES,
    THINKING_HELP,
    THINKING_LABELS,
    THINKING_MODES,
    Settings,
)
from ...prompts import library


def _int(value: str, fallback: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(float(value))))
    except (TypeError, ValueError):
        return fallback


def _float(value: str, fallback: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return fallback


class SettingsScreen(Screen):
    BINDINGS = [
        ("escape", "back", "Back"),
        ("ctrl+s", "save", "Save"),
        ("f5", "restore_defaults", "Defaults"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="settings-body"):
            # ---------------- left: device simulation -----------------
            with VerticalScroll(classes="settings-col"):
                yield Static("EDGE DEVICE SIMULATION", classes="panel-title")

                yield Label("Device profile", classes="field-label")
                yield Select(
                    [(DEVICE_PROFILES[k].label, k) for k in PROFILE_ORDER],
                    id="profile",
                    allow_blank=False,
                )
                yield Static(id="profile-note")

                yield Label(
                    f"CPU threads / cores (1-{HOST_CORES})", classes="field-label"
                )
                yield Static(
                    "llama.cpp worker threads. Also the width of the affinity mask.",
                    classes="field-help",
                )
                yield Input(id="threads", type="integer")

                yield Label("Device RAM (GB)", classes="field-label")
                yield Static(
                    "How much memory the simulated phone has in total.",
                    classes="field-help",
                )
                yield Input(id="ram", type="number")

                with Horizontal(classes="switch-row"):
                    yield Label("Pin CPU affinity")
                    yield Switch(id="affinity")
                yield Static(
                    "Hard OS-level cap: the process may only be scheduled on the "
                    "allotted cores. Strongly recommended.",
                    classes="field-help",
                )

                with Horizontal(classes="switch-row"):
                    yield Label("Enforce RAM ceiling")
                    yield Switch(id="ram-limit")
                yield Static(
                    "Caps *resident* memory, so a model that does not fit streams "
                    "through the limit instead of crashing. Off = the model is held "
                    "fully in RAM and only the CPU is constrained.",
                    classes="field-help",
                )
                yield Static(id="ram-limit-note", classes="field-help")

                yield Label("Share of device RAM the model may use (%)",
                            classes="field-label")
                yield Static(
                    "A phone never gives one app all of RAM - the OS and other "
                    "apps want theirs. 70% is a realistic default.",
                    classes="field-help",
                )
                yield Input(id="ram-pct", type="integer")

                yield Label("Explicit ceiling (MB, 0 = derive from above)",
                            classes="field-label")
                yield Input(id="ram-mb", type="integer")

                yield Label("Process priority", classes="field-label")
                yield Select(
                    [(p.replace("_", " ").title(), p) for p in PRIORITY_CHOICES],
                    id="priority",
                    allow_blank=False,
                )

                yield Static("TELEMETRY", classes="panel-title")
                yield Label("Sample interval (ms)", classes="field-label")
                yield Input(id="interval", type="integer")
                yield Label("Graph history (seconds)", classes="field-label")
                yield Input(id="history", type="integer")

            # ---------------- right: inference + prompt ---------------
            with VerticalScroll(classes="settings-col"):
                yield Static("SYSTEM PROMPT", classes="panel-title")
                yield Select(
                    library.options(),
                    id="prompt-preset",
                    prompt="Load a saved prompt or preset...",
                )
                yield TextArea(id="system-prompt", soft_wrap=True)
                with Horizontal(classes="save-row"):
                    yield Input(placeholder="Save as… (name)", id="prompt-name")
                    yield Button("Save", id="prompt-save")
                    yield Button("Delete", id="prompt-delete")
                yield Static(id="prompt-status", classes="field-help")
                yield Static(
                    "Sent as the system message on every request. ★ = your "
                    "saved prompts, · = built-in. Also editable mid-conversation "
                    "from the chat screen (Ctrl+P).",
                    classes="field-help",
                )

                yield Static("INFERENCE", classes="panel-title")
                yield Label("Context size (tokens)", classes="field-label")
                yield Input(id="ctx", type="integer")
                yield Label("Max new tokens per reply", classes="field-label")
                yield Input(id="maxtok", type="integer")
                yield Label("Batch size", classes="field-label")
                yield Input(id="batch", type="integer")

                yield Label("Temperature", classes="field-label")
                yield Input(id="temp", type="number")
                yield Label("Top-p", classes="field-label")
                yield Input(id="topp", type="number")
                yield Label("Top-k", classes="field-label")
                yield Input(id="topk", type="integer")
                yield Label("Repeat penalty", classes="field-label")
                yield Input(id="repeat", type="number")
                yield Label("Seed (-1 = random)", classes="field-label")
                yield Input(id="seed", type="integer")

                yield Label("Preferred quantisation", classes="field-label")
                yield Select(
                    [(q, q) for q in QUANT_PREFERENCES], id="quant", allow_blank=False
                )

                with Horizontal(classes="switch-row"):
                    yield Label("Keep chat history")
                    yield Switch(id="keep-history")
                yield Label("History window (turns)", classes="field-label")
                yield Input(id="turns", type="integer")

                yield Label("Prompt-cache RAM (MB, 0 = off)", classes="field-label")
                yield Static(
                    "llama-server keeps KV states of earlier prompts in RAM so a "
                    "returning prefix skips prefill. llama.cpp's default is 8192 MB, "
                    "far past a phone budget; 0 keeps RSS honest. Applies on next load.",
                    classes="field-help",
                )
                yield Input(id="cache-ram", type="integer")

                with Horizontal(classes="switch-row"):
                    yield Label("Disable mmap")
                    yield Switch(id="nommap")
                yield Static(
                    "Loads weights into anonymous memory so RSS reflects true "
                    "model cost. Turn off for faster loads, fuzzier RAM numbers.",
                    classes="field-help",
                )

                yield Static("REASONING / THINKING", classes="panel-title")
                yield Label("Thinking mode", classes="field-label")
                yield Select(
                    [(THINKING_LABELS[m], m) for m in THINKING_MODES],
                    id="thinking-mode",
                    allow_blank=False,
                )
                yield Static(THINKING_HELP, classes="field-help")
                yield Static(
                    "Applied per request - takes effect on your next message, "
                    "no model reload. Also togglable with Ctrl+T while chatting.",
                    classes="field-help",
                )

                with Horizontal(classes="switch-row"):
                    yield Label("Show reasoning in transcript")
                    yield Switch(id="show-thinking")
                yield Static(
                    "Display only. Reasoning tokens are counted toward "
                    "throughput either way.",
                    classes="field-help",
                )

                yield Static("AGENT / INTENT ROUTING", classes="panel-title")
                yield Label("Agent mode", classes="field-label")
                yield Select(
                    [(AGENT_LABELS[m], m) for m in AGENT_MODES],
                    id="agent-mode",
                    allow_blank=False,
                )
                yield Static(AGENT_HELP, classes="field-help")
                yield Static(id="router-note", classes="field-help")
                yield Label("Route threshold (cosine, below -> all tools)",
                            classes="field-label")
                yield Input(id="router-threshold", type="number")
                yield Label("Minimum margin between top two intents",
                            classes="field-label")
                yield Input(id="router-margin", type="number")
                with Horizontal(classes="switch-row"):
                    yield Label("Answer trivial requests without the SLM")
                    yield Switch(id="direct-calls")
                yield Label("Direct-answer threshold (stricter)", classes="field-label")
                yield Input(id="direct-threshold", type="number")
                yield Label("Router share of the RAM ceiling (MB)", classes="field-label")
                yield Static(
                    "Only used with the RAM ceiling on: carved out of the model's "
                    "budget so the pair never exceeds the device. Measured router "
                    "RSS: ~275 MB.",
                    classes="field-help",
                )
                yield Input(id="router-ram", type="integer")

                with Horizontal(classes="switch-row"):
                    yield Label("Auto-save session reports")
                    yield Switch(id="autosave")

        with Horizontal(id="settings-actions"):
            yield Button("Save", id="save", variant="primary")
            yield Button("Restore defaults", id="defaults")
            yield Button("Back", id="back")
        yield Footer()

    # ------------------------------------------------------------------
    def on_mount(self) -> None:
        self.load_from(self.app.settings)

    def on_screen_resume(self) -> None:
        self.load_from(self.app.settings)

    def load_from(self, s: Settings) -> None:
        self.query_one("#profile", Select).value = s.profile
        self.query_one("#threads", Input).value = str(s.threads)
        self.query_one("#ram", Input).value = f"{s.nominal_ram_gb:g}"
        self.query_one("#affinity", Switch).value = s.pin_affinity
        self.query_one("#ram-limit", Switch).value = s.ram_limit_enabled
        self.query_one("#ram-pct", Input).value = str(s.ram_budget_pct)
        self.query_one("#ram-mb", Input).value = str(s.ram_limit_mb)
        self._update_ram_note(s)
        self.query_one("#priority", Select).value = s.priority
        self.query_one("#interval", Input).value = str(s.sample_interval_ms)
        self.query_one("#history", Input).value = str(s.history_seconds)

        # The library is shared with the chat modal, so re-read it on entry.
        self._refresh_prompt_options()
        self.query_one("#system-prompt", TextArea).text = s.system_prompt
        saved_as = library.find_name_for(s.system_prompt)
        self.query_one("#prompt-name", Input).value = (
            saved_as if saved_as and not library.is_builtin(saved_as) else ""
        )
        self.query_one("#ctx", Input).value = str(s.context_size)
        self.query_one("#maxtok", Input).value = str(s.max_tokens)
        self.query_one("#batch", Input).value = str(s.batch_size)
        self.query_one("#temp", Input).value = f"{s.temperature:g}"
        self.query_one("#topp", Input).value = f"{s.top_p:g}"
        self.query_one("#topk", Input).value = str(s.top_k)
        self.query_one("#repeat", Input).value = f"{s.repeat_penalty:g}"
        self.query_one("#seed", Input).value = str(s.seed)
        self.query_one("#quant", Select).value = s.preferred_quant
        self.query_one("#keep-history", Switch).value = s.keep_history
        self.query_one("#turns", Input).value = str(s.history_turns)
        self.query_one("#nommap", Switch).value = s.no_mmap
        self.query_one("#cache-ram", Input).value = str(s.prompt_cache_ram_mb)
        self.query_one("#thinking-mode", Select).value = s.thinking_mode
        self.query_one("#show-thinking", Switch).value = s.show_thinking
        self.query_one("#autosave", Switch).value = s.auto_save_sessions
        self.query_one("#agent-mode", Select).value = s.agent_mode
        self.query_one("#router-threshold", Input).value = f"{s.router_threshold:g}"
        self.query_one("#router-margin", Input).value = f"{s.router_margin:g}"
        self.query_one("#direct-calls", Switch).value = s.direct_calls
        self.query_one("#direct-threshold", Input).value = f"{s.direct_threshold:g}"
        self.query_one("#router-ram", Input).value = str(s.router_ram_mb)
        self._update_router_note(s)
        self._update_note()

    # ------------------------------------------------------------------
    def _update_note(self) -> None:
        key = self.query_one("#profile", Select).value
        profile = DEVICE_PROFILES.get(str(key), DEVICE_PROFILES["custom"])
        threads = _int(self.query_one("#threads", Input).value, profile.threads, 1, HOST_CORES)
        share = 100.0 * threads / max(1, HOST_CORES)
        self.query_one("#profile-note", Static).update(
            f"[b #4fc1ff]{profile.name}[/]  [#6f7789]· {profile.example}[/]\n"
            f"{profile.note}\n"
            f"[#7bd88f]{threads}/{HOST_CORES} host cores  ({share:.0f}% of this machine)[/]"
        )

    def _update_ram_note(self, s: Settings | None = None) -> None:
        """Show the derived ceiling and which downloaded models clear it."""
        from ... import catalog as cat
        from ...engine.memcap import MB
        from ...paths import human_bytes

        note = self.query_one("#ram-limit-note", Static)
        enabled = self.query_one("#ram-limit", Switch).value
        if not enabled:
            note.update(
                "[#6f7789]Off — RAM is measured but not constrained. "
                "Only the CPU budget is simulated.[/]"
            )
            return

        ram_gb = _float(self.query_one("#ram", Input).value, 8.0, 0, 512)
        pct = _int(self.query_one("#ram-pct", Input).value, 70, 10, 100)
        explicit = _int(self.query_one("#ram-mb", Input).value, 0, 0, 1048576)
        limit_mb = explicit or int(ram_gb * 1024 * pct / 100)
        if limit_mb <= 0:
            note.update("[#ffb454]Set a device RAM figure or an explicit ceiling.[/]")
            return

        fits = [m for m in cat.CATALOG if m.downloaded and m.local_size <= limit_mb * MB]
        streams = [m for m in cat.CATALOG if m.downloaded and m.local_size > limit_mb * MB]
        lines = [f"[b #ff5f5f]Ceiling: {limit_mb} MB[/] "
                 f"[#6f7789](of {ram_gb:g} GB device RAM)[/]"]
        if fits:
            lines.append(f"[#7bd88f]fits ({len(fits)}):[/] "
                         + ", ".join(m.name for m in fits[:3])
                         + (" …" if len(fits) > 3 else ""))
        if streams:
            worst = max(streams, key=lambda m: m.local_size)
            lines.append(
                f"[#ffb454]streams ({len(streams)}):[/] "
                + ", ".join(m.name for m in streams[:3])
                + (" …" if len(streams) > 3 else "")
                + f"\n[#6f7789]largest is {human_bytes(worst.local_size)}, "
                  f"{worst.local_size / (limit_mb * MB):.1f}x the ceiling[/]"
            )
        if not fits and not streams:
            lines.append("[#6f7789]No models downloaded yet.[/]")
        note.update("\n".join(lines))

    def _update_router_note(self, s: Settings) -> None:
        from ... import encoders
        from ...engine.encoder_server import INSTALL_HINT, missing_packages

        spec = encoders.get(s.router_model)
        missing = missing_packages()
        if missing:
            text = f"[#ffb454]Router needs: {INSTALL_HINT}[/]"
        elif spec is None:
            text = f"[#ff5f5f]Unknown router model '{s.router_model}'.[/]"
        elif not spec.downloaded:
            text = (f"[#ffb454]{spec.name} not downloaded - run: "
                    f"python -m feast_ai pull {spec.id}[/]")
        else:
            text = f"[#7bd88f]Router: {spec.name} ({spec.params}, int8 ONNX) ready.[/]"
        self.query_one("#router-note", Static).update(text)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "ram-limit":
            self._update_ram_note()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "profile" and event.value is not Select.BLANK:
            key = str(event.value)
            if key != "custom":
                profile = DEVICE_PROFILES[key]
                self.query_one("#threads", Input).value = str(profile.threads)
                self.query_one("#ram", Input).value = f"{profile.nominal_ram_gb:g}"
                # A profile switch drops any hand-typed ceiling, matching
                # Settings.apply_profile.
                self.query_one("#ram-mb", Input).value = "0"
            self._update_note()
            self._update_ram_note()
        elif event.select.id == "prompt-preset" and event.value is not Select.BLANK:
            name = str(event.value)
            body = library.get(name)
            if body is not None:
                self.query_one("#system-prompt", TextArea).text = body
                self.query_one("#prompt-name", Input).value = (
                    "" if library.is_builtin(name) else name
                )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id in ("threads", "ram"):
            self._update_note()
        if event.input.id in ("ram", "ram-pct", "ram-mb"):
            self._update_ram_note()

    # ------------------------------------------------------------------
    def collect(self) -> Settings:
        s = self.app.settings
        s.profile = str(self.query_one("#profile", Select).value)
        s.threads = _int(self.query_one("#threads", Input).value, s.threads, 1, HOST_CORES)
        s.nominal_ram_gb = _float(self.query_one("#ram", Input).value, s.nominal_ram_gb, 0, 512)
        s.pin_affinity = self.query_one("#affinity", Switch).value
        s.ram_limit_enabled = self.query_one("#ram-limit", Switch).value
        s.ram_budget_pct = _int(
            self.query_one("#ram-pct", Input).value, s.ram_budget_pct, 10, 100
        )
        s.ram_limit_mb = _int(self.query_one("#ram-mb", Input).value, s.ram_limit_mb, 0, 1048576)
        s.priority = str(self.query_one("#priority", Select).value)
        s.sample_interval_ms = _int(
            self.query_one("#interval", Input).value, s.sample_interval_ms, 50, 5000
        )
        s.history_seconds = _int(
            self.query_one("#history", Input).value, s.history_seconds, 10, 3600
        )

        s.system_prompt = self.query_one("#system-prompt", TextArea).text
        s.context_size = _int(self.query_one("#ctx", Input).value, s.context_size, 256, 131072)
        s.max_tokens = _int(self.query_one("#maxtok", Input).value, s.max_tokens, 16, 32768)
        s.batch_size = _int(self.query_one("#batch", Input).value, s.batch_size, 16, 4096)
        s.temperature = _float(self.query_one("#temp", Input).value, s.temperature, 0.0, 2.0)
        s.top_p = _float(self.query_one("#topp", Input).value, s.top_p, 0.01, 1.0)
        s.top_k = _int(self.query_one("#topk", Input).value, s.top_k, 0, 500)
        s.repeat_penalty = _float(
            self.query_one("#repeat", Input).value, s.repeat_penalty, 0.5, 2.5
        )
        s.seed = _int(self.query_one("#seed", Input).value, s.seed, -1, 2**31 - 1)
        s.preferred_quant = str(self.query_one("#quant", Select).value)
        s.keep_history = self.query_one("#keep-history", Switch).value
        s.history_turns = _int(self.query_one("#turns", Input).value, s.history_turns, 1, 64)
        s.no_mmap = self.query_one("#nommap", Switch).value
        s.prompt_cache_ram_mb = _int(
            self.query_one("#cache-ram", Input).value, s.prompt_cache_ram_mb, -1, 65536
        )
        s.thinking_mode = str(self.query_one("#thinking-mode", Select).value)
        s.show_thinking = self.query_one("#show-thinking", Switch).value
        s.auto_save_sessions = self.query_one("#autosave", Switch).value
        s.agent_mode = str(self.query_one("#agent-mode", Select).value)
        s.router_threshold = _float(
            self.query_one("#router-threshold", Input).value, s.router_threshold, 0.0, 1.0
        )
        s.router_margin = _float(
            self.query_one("#router-margin", Input).value, s.router_margin, 0.0, 1.0
        )
        s.direct_calls = self.query_one("#direct-calls", Switch).value
        s.direct_threshold = _float(
            self.query_one("#direct-threshold", Input).value, s.direct_threshold, 0.0, 1.0
        )
        s.router_ram_mb = _int(self.query_one("#router-ram", Input).value, s.router_ram_mb, 64, 4096)

        s.mark_custom_if_needed()
        s.configured = True
        return s

    # ------------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        handlers = {
            "save": self.action_save,
            "defaults": self.action_restore_defaults,
            "back": self.action_back,
            "prompt-save": self.save_prompt,
            "prompt-delete": self.delete_prompt,
        }
        handler = handlers.get(event.button.id or "")
        if handler:
            handler()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "prompt-name":
            self.save_prompt()

    # ------------------------------------------------------------------
    def _prompt_status(self, message: str, ok: bool) -> None:
        colour = "#7bd88f" if ok else "#ff5f5f"
        self.query_one("#prompt-status", Static).update(f"[{colour}]{message}[/]")

    def _refresh_prompt_options(self, select_name: str | None = None) -> None:
        select = self.query_one("#prompt-preset", Select)
        select.set_options(library.options())
        if select_name and library.get(select_name) is not None:
            select.value = select_name

    def save_prompt(self) -> None:
        name = self.query_one("#prompt-name", Input).value
        text = self.query_one("#system-prompt", TextArea).text
        ok, message = library.save(name, text)
        self._prompt_status(message, ok)
        if ok:
            self._refresh_prompt_options(select_name=name.strip())

    def delete_prompt(self) -> None:
        name = self.query_one("#prompt-name", Input).value.strip()
        if not name:
            selected = self.query_one("#prompt-preset", Select).value
            name = "" if selected is Select.BLANK else str(selected)
        ok, message = library.delete(name)
        self._prompt_status(message, ok)
        if ok:
            self.query_one("#prompt-name", Input).value = ""
            self._refresh_prompt_options()

    def action_save(self) -> None:
        settings = self.collect()
        self.app.settings = settings
        self.app.save_settings()

        note = ""
        if self.app.engine_running and self.app.server is not None:
            pinned = self.app.server.reapply_constraints()
            note = (
                f" Affinity re-applied to {pinned}."
                if pinned
                else " Restart the model for thread changes to take effect."
            )
        self.notify(f"Settings saved.{note}", title="feast AI")
        self.dismiss_or_pop()

    def action_restore_defaults(self) -> None:
        fresh = Settings()
        fresh.configured = True
        self.load_from(fresh)
        self.notify("Defaults restored - press Save to keep them.")

    def action_back(self) -> None:
        self.dismiss_or_pop()

    def dismiss_or_pop(self) -> None:
        if len(self.app.screen_stack) > 1:
            self.app.pop_screen()
