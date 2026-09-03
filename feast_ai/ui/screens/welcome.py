"""Welcome screen: host facts, runtime bootstrap, entry point."""

from __future__ import annotations

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Center, Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, ProgressBar, Static

from ... import catalog as cat
from ...catalog import downloaded_models, recommended_for
from ...engine import ensure_runtime, runtime_info
from ...metrics.sampler import host_snapshot
from ...paths import HOME, human_bytes
from ..widgets import Banner


class WelcomeScreen(Screen):
    BINDINGS = [
        ("enter", "start", "Start"),
        ("m", "models", "Models"),
        ("s", "settings", "Settings"),
        ("r", "bootstrap", "Re-install runtime"),
        ("q", "quit_app", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Center(id="welcome-wrap"):
            with Container(id="welcome-card"):
                yield Banner(id="banner")
                yield Static(id="host-facts")
                yield Static(id="boot-status")
                yield ProgressBar(total=100, show_eta=False, id="boot-progress")
                with Horizontal(id="welcome-actions"):
                    yield Button("Start session", id="start", variant="primary")
                    yield Button("Models", id="models")
                    yield Button("Settings", id="settings")
                    yield Button("Quit", id="quit")
        yield Footer()

    # ------------------------------------------------------------------
    def on_mount(self) -> None:
        self.refresh_facts()
        app = self.app
        if not app.settings.configured:
            self.set_timer(0.35, self._first_run)
        elif not app.runtime.available:
            self.action_bootstrap()

    def on_screen_resume(self) -> None:
        self.refresh_facts()

    def _first_run(self) -> None:
        self.notify(
            "First run - pick a device profile and system prompt.",
            title="feast AI setup",
            timeout=6,
        )
        self.app.push_screen("settings")

    # ------------------------------------------------------------------
    def refresh_facts(self) -> None:
        app = self.app
        settings = app.settings
        host = host_snapshot()
        app.runtime = runtime_info()
        device = settings.device
        have = downloaded_models()

        text = Text()

        def row(label: str, value: str, style: str = "#e6e9f0") -> None:
            text.append(f"  {label:<20}", style="#6f7789")
            text.append(f"{value}\n", style=style)

        text.append("  HOST\n", style="bold #ffb454")
        row("CPU", f"{host.cpu_name[:44] or 'unknown'}")
        row(
            "Cores",
            f"{host.logical_cores} logical"
            + (f" / {host.physical_cores} physical" if host.physical_cores else "")
            + (f" @ {host.freq_mhz/1000:.1f} GHz" if host.freq_mhz else ""),
        )
        row("RAM", human_bytes(host.total_ram))

        text.append("\n  SIMULATED EDGE DEVICE\n", style="bold #ffb454")
        row("Profile", device.name, "#4fc1ff")
        row("Example", device.example, "#6f7789")
        row(
            "CPU budget",
            f"{settings.cpu_budget_cores} core(s)"
            + ("  ·  affinity pinned" if settings.pin_affinity else "  ·  threads only"),
            "#7bd88f",
        )
        row(
            "RAM",
            (
                f"{settings.nominal_ram_gb:g} GB nominal · never enforced"
                if settings.nominal_ram_gb
                else "uncapped (host reference)"
            ),
            "#8b93a7",
        )
        row("Context", f"{settings.context_size} tokens")

        text.append("\n  ENGINE & MODELS\n", style="bold #ffb454")
        if app.runtime.available:
            row(
                "llama.cpp",
                f"{app.runtime.build or 'system'} ({app.runtime.source})",
                "#7bd88f",
            )
        else:
            row("llama.cpp", "not installed - press R to fetch", "#ffb454")
        row("Catalog", f"{len(cat.CATALOG)} edge models  ·  {len(have)} downloaded")
        pick = recommended_for(settings.nominal_ram_gb)
        row("Suggested", f"{pick.name}  ({pick.size_label})", "#4fc1ff")
        row("Data dir", str(HOME), "#545b6b")

        self.query_one("#host-facts", Static).update(text)

        status = self.query_one("#boot-status", Static)
        if not app.runtime.available:
            status.update(
                "[#ffb454]Inference runtime missing.[/] "
                "Press [b]R[/b] to download the llama.cpp CPU build (~30 MB)."
            )
        elif not have:
            status.update(
                "[#8b93a7]No weights yet. Open [b]Models[/b] to download one "
                "(gemma-3-270m is 200 MB and starts in seconds).[/]"
            )
        else:
            status.update(
                f"[#7bd88f]Ready.[/] [#6f7789]Enter starts a session with "
                f"{len(have)} local model(s).[/]"
            )

    # ------------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "start": self.action_start,
            "models": self.action_models,
            "settings": self.action_settings,
            "quit": self.action_quit_app,
        }
        handler = actions.get(event.button.id or "")
        if handler:
            handler()

    def action_start(self) -> None:
        if not self.app.runtime.available:
            self.notify("Install the runtime first (press R).", severity="warning")
            return
        self.app.push_screen("models")

    def action_models(self) -> None:
        self.app.push_screen("models")

    def action_settings(self) -> None:
        self.app.push_screen("settings")

    def action_quit_app(self) -> None:
        self.app.call_later(self.app.action_quit)

    # ------------------------------------------------------------------
    @work(exclusive=True)
    async def action_bootstrap(self) -> None:
        status = self.query_one("#boot-status", Static)
        bar = self.query_one("#boot-progress", ProgressBar)
        bar.add_class("active")
        bar.update(total=100, progress=0)

        def on_status(message: str) -> None:
            status.update(f"[#8b93a7]{message}[/]")

        def on_progress(progress) -> None:
            if progress.total:
                bar.update(total=100, progress=progress.fraction * 100)
            status.update(
                f"[#8b93a7]Downloading runtime  "
                f"{human_bytes(progress.downloaded)}"
                f"{' / ' + human_bytes(progress.total) if progress.total else ''}  ·  "
                f"{human_bytes(progress.speed_bps)}/s[/]"
            )

        try:
            info = await ensure_runtime(on_progress=on_progress, on_status=on_status)
            self.app.runtime = info
            self.notify(f"llama.cpp {info.build} installed.", title="Runtime ready")
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            status.update(f"[#ff5f5f]{exc}[/]")
            self.notify(str(exc), severity="error", title="Runtime install failed", timeout=12)
        finally:
            bar.remove_class("active")
            self.refresh_facts()
