"""Add a model from Hugging Face, or by direct GGUF link from any host."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from ... import catalog as cat
from ...catalog import LookupError_, ModelSpec, RepoFile
from ...paths import human_bytes

EXAMPLES = (
    "unsloth/gemma-3-1b-it-GGUF\n"
    "https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF\n"
    "https://huggingface.co/…/resolve/main/model-Q4_K_M.gguf\n"
    "https://any-host.example/path/model.gguf"
)


class AddModelScreen(ModalScreen[ModelSpec | None]):
    """Look up a repo, choose a quantisation, save it to the catalog."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self.files: list[RepoFile] = []
        self.repo: str = ""
        self.direct_url: str = ""

    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Container(id="add-card"):
            yield Label("Add a model", classes="panel-title")
            yield Static(
                "Paste a Hugging Face repo id or URL, or a direct link to a "
                "GGUF file on any host:",
                classes="field-help",
            )
            yield Static(EXAMPLES, id="add-examples", classes="dim")
            with Horizontal(classes="save-row"):
                yield Input(placeholder="owner/name  or  https://…", id="add-source")
                yield Button("Look up", id="add-lookup", variant="primary")
            yield Static(id="add-status")

            yield Label("Quantisation", classes="field-label")
            yield Select([], id="add-file", prompt="Look up a repo first…")

            with Horizontal(classes="meta-row"):
                yield Input(placeholder="Display name", id="add-name")
                yield Input(placeholder="Params (e.g. 1.2B)", id="add-params")
            with Horizontal(classes="meta-row"):
                yield Input(placeholder="Context (4096)", id="add-ctx", type="integer")
                yield Input(placeholder="Turkish 1-5 (3)", id="add-tr", type="integer")

            yield Static(
                "Turkish score is your own note for the catalog listing - "
                "feast AI cannot measure it for you.",
                classes="field-help",
            )
            with Horizontal(id="confirm-actions"):
                yield Button("Add to catalog", id="add-save", variant="primary")
                yield Button("Cancel", id="add-cancel")

    def on_mount(self) -> None:
        self.query_one("#add-source", Input).focus()

    # ------------------------------------------------------------------
    def _status(self, message: str, tone: str = "#8b93a7") -> None:
        self.query_one("#add-status", Static).update(f"[{tone}]{message}[/]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        handlers = {
            "add-lookup": self.action_lookup,
            "add-save": self.action_save,
            "add-cancel": self.action_cancel,
        }
        handler = handlers.get(event.button.id or "")
        if handler:
            handler()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "add-source":
            self.action_lookup()

    # ------------------------------------------------------------------
    def action_lookup(self) -> None:
        text = self.query_one("#add-source", Input).value.strip()
        if not text:
            self._status("Enter a repo id or URL first.", "#ffb454")
            return
        self._lookup(text)

    @work(exclusive=True)
    async def _lookup(self, text: str) -> None:
        select = self.query_one("#add-file", Select)
        self._status("Looking up…")
        self.files = []
        self.repo = ""
        self.direct_url = ""

        try:
            kind, value = cat.parse_source(text)
        except LookupError_ as exc:
            self._status(str(exc), "#ff5f5f")
            return

        if kind == "url":
            # A direct file link - nothing to enumerate.
            self.direct_url = value
            filename = Path(value.split("?")[0]).name
            self.repo = _repo_from_hf_url(value)
            size = await asyncio.to_thread(cat.remote_size, value)
            self.files = [RepoFile(filename, size or 0)]
            select.set_options([(self.files[0].label, filename)])
            select.value = filename
            self._prefill(Path(filename).stem, size)
            self._status(
                f"Direct link · {filename} · {human_bytes(size) if size else 'size unknown'}",
                "#7bd88f",
            )
            return

        self.repo = value
        try:
            files = await asyncio.to_thread(cat.list_repo_files, value)
        except LookupError_ as exc:
            self._status(str(exc), "#ff5f5f")
            select.set_options([])
            return

        self.files = files
        select.set_options([(f.label, f.filename) for f in files])

        # Default to the user's preferred quantisation when the repo has it.
        preferred = self.app.settings.preferred_quant.upper()
        default = next((f for f in files if f.quant == preferred), None)
        default = default or next((f for f in files if f.quant == "Q4_K_M"), files[0])
        select.value = default.filename
        self._prefill(value.split("/")[-1].replace("-GGUF", ""), default.size)
        self._status(
            f"{value} · {len(files)} GGUF file(s) · defaulting to {default.quant}",
            "#7bd88f",
        )

    def _prefill(self, name: str, size: int | None) -> None:
        name_input = self.query_one("#add-name", Input)
        if not name_input.value.strip():
            name_input.value = name
        ctx = self.query_one("#add-ctx", Input)
        if not ctx.value.strip():
            ctx.value = "4096"
        tr = self.query_one("#add-tr", Input)
        if not tr.value.strip():
            tr.value = "3"

    # ------------------------------------------------------------------
    def action_save(self) -> None:
        select = self.query_one("#add-file", Select)
        if select.value is Select.BLANK or not self.files:
            self._status("Look up a repo and pick a file first.", "#ffb454")
            return

        filename = str(select.value)
        chosen = next((f for f in self.files if f.filename == filename), None)
        if chosen is None:
            self._status("That file is no longer in the list - look up again.", "#ff5f5f")
            return

        spec = cat.build_custom_spec(
            name=self.query_one("#add-name", Input).value,
            repo=self.repo,
            filename=filename,
            size_bytes=chosen.size,
            source_url=self.direct_url,
            params=self.query_one("#add-params", Input).value.strip() or "?",
            context=_int(self.query_one("#add-ctx", Input).value, 4096),
            tr=_int(self.query_one("#add-tr", Input).value, 3),
        )
        cat.add_custom(spec)
        self.dismiss(spec)

    def action_cancel(self) -> None:
        self.dismiss(None)


def _int(text: str, fallback: int) -> int:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return fallback


def _repo_from_hf_url(url: str) -> str:
    """``owner/name`` if this is a Hugging Face resolve URL, else the host."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    segments = [s for s in parsed.path.split("/") if s]
    if parsed.netloc.lower().removeprefix("www.") in ("huggingface.co", "hf.co"):
        if len(segments) >= 2:
            return f"{segments[0]}/{segments[1]}"
    return parsed.netloc
