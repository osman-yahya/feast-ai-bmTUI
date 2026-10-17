"""The feast AI Textual application shell.

Holds the process-wide state - settings, the running engine, the active
model - and routes between screens.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import App

from .. import APP_NAME, __version__
from ..catalog import ModelSpec
from ..config import Settings
from ..engine import BaseServer, RuntimeInfo, runtime_info
from ..metrics import Sampler
from ..paths import ensure_dirs
from .screens.models import ModelsScreen
from .screens.settings import SettingsScreen
from .screens.summary import SummaryScreen
from .screens.welcome import WelcomeScreen


class FeastApp(App):
    """feast AI."""

    TITLE = APP_NAME
    SUB_TITLE = f"edge LLM lab v{__version__}"
    CSS_PATH = "feast.tcss"

    #: Named screens are *cached instances* in Textual - pushing a name twice
    #: returns the same object and does not re-run ``on_mount``.  That is fine
    #: for these four, which re-read app state in ``on_screen_resume``.
    #: ChatScreen is deliberately absent: it owns a live server connection and
    #: per-session state, so a fresh instance is pushed for every model.
    #: See :meth:`ModelsScreen.action_launch`.
    SCREENS = {
        "welcome": WelcomeScreen,
        "models": ModelsScreen,
        "settings": SettingsScreen,
        "summary": SummaryScreen,
    }

    def __init__(self) -> None:
        super().__init__()
        ensure_dirs()
        self.settings: Settings = Settings.load()
        self.runtime: RuntimeInfo = runtime_info()
        self.server: BaseServer | None = None
        #: The intent router's worker and policy, alive only while the
        #: hierarchical agent mode has been used this session.
        self.router_server: BaseServer | None = None
        self.router = None
        self.sampler: Sampler = Sampler(
            budget_cores=self.settings.cpu_budget_cores,
            history_points=self.settings.sample_points,
        )
        self.active_model: ModelSpec | None = None
        self.active_model_path: Path | None = None
        self.last_summary = None

    # ------------------------------------------------------------------
    def on_mount(self) -> None:
        self.theme = "textual-dark"
        self.push_screen("welcome")

    # ------------------------------------------------------------------
    def save_settings(self) -> None:
        self.settings.save()
        self.sampler.set_budget(self.settings.cpu_budget_cores)

    @property
    def engine_running(self) -> bool:
        return self.server is not None and self.server.running

    @property
    def router_running(self) -> bool:
        return self.router_server is not None and self.router_server.running

    def attach_telemetry(self) -> None:
        """Sample the model and, when loaded, the router as one device.

        They share the simulated phone's cores and memory, so CPU and RSS
        are reported for the pair - the router is not free.
        """
        procs = self.server.snapshot_processes() if self.server else []
        if self.router_server is not None:
            procs += self.router_server.snapshot_processes()
        self.sampler.attach(procs)

    # ------------------------------------------------------------------
    async def shutdown_router(self) -> None:
        if self.router is not None:
            await self.router.aclose()
        self.router = None
        if self.router_server is not None:
            await self.router_server.stop()
        self.router_server = None

    async def shutdown_engine(self) -> None:
        """Stop the inference server (and router) and detach telemetry."""
        self.sampler.detach()
        await self.shutdown_router()
        if self.server is not None:
            await self.server.stop()
        self.server = None

    async def action_quit(self) -> None:  # type: ignore[override]
        await self.shutdown_engine()
        self.exit()

    async def on_unmount(self) -> None:
        # Belt and braces: never leave an orphaned llama-server behind.
        for server in (self.server, self.router_server):
            if server is None:
                continue
            try:
                await asyncio.wait_for(server.stop(), timeout=5)
            except (asyncio.TimeoutError, OSError):  # pragma: no cover
                pass
