"""Model catalog: browse, download and launch."""

from __future__ import annotations

import asyncio

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, ProgressBar, Static

from ... import catalog as cat
from ...catalog import ModelSpec
from ...download import clear_partial, download, download_many, partial_bytes
from ...paths import human_bytes
from .addmodel import AddModelScreen
from .chat import ChatScreen
from .common import ConfirmScreen


class ModelsScreen(Screen):
    BINDINGS = [
        ("escape", "back", "Back"),
        ("enter", "launch", "Load & chat"),
        ("d", "download", "Download"),
        ("a", "add_model", "Add model"),
        ("x", "delete", "Delete"),
        ("r", "refresh", "Refresh"),
        ("ctrl+c", "cancel_download", "Cancel download"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._cancel: asyncio.Event | None = None
        self._downloading: str | None = None

    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="models-body"):
            yield DataTable(id="model-table", cursor_type="row", zebra_stripes=True)
            with VerticalScroll(id="model-detail"):
                yield Static(id="detail-name")
                yield Static(id="detail-meta")
                yield Static(id="detail-why", classes="dim")
                yield Static(id="detail-fit")
                yield ProgressBar(total=100, show_eta=True, id="download-bar")
                yield Static(id="download-status")
        with Horizontal(id="model-actions"):
            yield Button("Load & chat", id="launch", variant="primary")
            yield Button("Download", id="download")
            yield Button("Add model", id="add")
            yield Button("Delete", id="delete")
            yield Button("Back", id="back")
        yield Footer()

    # ------------------------------------------------------------------
    def on_mount(self) -> None:
        table = self.query_one("#model-table", DataTable)
        table.add_column("#", width=3)
        table.add_column("Model", width=30)
        table.add_column("Params", width=9)
        table.add_column("Size", width=8)
        table.add_column("TR", width=6)
        table.add_column("Fits", width=5)
        table.add_column("Status", width=12)
        self.refresh_table()

    def on_screen_resume(self) -> None:
        self.refresh_table()

    # ------------------------------------------------------------------
    def refresh_table(self, keep_row: int | None = None) -> None:
        table = self.query_one("#model-table", DataTable)
        row = keep_row if keep_row is not None else table.cursor_row
        table.clear()
        nominal = self.app.settings.nominal_ram_gb

        for spec in cat.CATALOG:
            fits = spec.fits(nominal)
            if spec.downloaded:
                status = Text("● local", style="#7bd88f")
            elif partial_bytes(spec.local_path):
                status = Text("◐ partial", style="#ffb454")
            else:
                status = Text("○ remote", style="#6f7789")

            name = Text(spec.name)
            if spec.tier == "turkish":
                name.stylize("bold #7bd88f")
            elif spec.tier == "custom":
                name.stylize("bold #4fc1ff")
            elif spec.tier == "onnx":
                name.stylize("bold #c792ea")

            table.add_row(
                str(spec.rank),
                name,
                spec.params,
                spec.size_label,
                Text(spec.tr_stars, style="#7bd88f" if spec.tr >= 4 else "#6f7789"),
                Text("yes", style="#7bd88f") if fits else Text("no", style="#ff5f5f"),
                status,
                key=spec.id,
            )

        if cat.CATALOG:
            table.move_cursor(row=min(max(0, row), len(cat.CATALOG) - 1))
            self.show_detail(self.current_spec())

    # ------------------------------------------------------------------
    def current_spec(self) -> ModelSpec | None:
        table = self.query_one("#model-table", DataTable)
        if table.cursor_row < 0 or not table.row_count:
            return None
        try:
            key = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value
        except Exception:
            return None
        return cat.get(str(key))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self.show_detail(cat.get(str(event.row_key.value)))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_launch()

    # ------------------------------------------------------------------
    def show_detail(self, spec: ModelSpec | None) -> None:
        if spec is None:
            return
        settings = self.app.settings

        title = Text()
        title.append(spec.name, style="bold #ffb454")
        if spec.tier == "turkish":
            title.append("   TURKISH-NATIVE", style="bold #7bd88f")
        elif spec.is_onnx:
            title.append("   ONNX RUNTIME", style="bold #c792ea")
        self.query_one("#detail-name", Static).update(title)

        meta = Text()

        def row(label: str, value: str, style: str = "#e6e9f0") -> None:
            meta.append(f"{label:<12}", style="#6f7789")
            meta.append(f"{value}\n", style=style)

        row("Vendor", spec.vendor)
        row("Engine", _engine_label(spec), "#c792ea" if spec.is_onnx else "#8b93a7")
        row("Params", spec.params)
        row("Quant", f"{spec.quant}  ·  {spec.size_label}")
        row("Context", f"{spec.context:,} tokens")
        row("License", spec.license)
        row("Turkish", f"{spec.tr_stars}  ({spec.tr}/5)", "#7bd88f" if spec.tr >= 4 else "#8b93a7")
        row("Repo", spec.repo, "#4fc1ff")
        if spec.is_onnx:
            where = spec.onnx_subdir or "(repo root)"
            row("Files", f"{len(spec.onnx_files)} in {where}", "#545b6b")
        else:
            row("File", spec.effective_filename, "#545b6b")
        if spec.tags:
            meta.append(f"{'Tags':<12}", style="#6f7789")
            meta.append("  ".join(f"[{t}]" for t in spec.tags) + "\n", style="#4fc1ff")
        if spec.downloaded:
            row("On disk", human_bytes(spec.local_size), "#7bd88f")
        else:
            already = partial_bytes(spec.local_path)
            if already:
                row("Partial", f"{human_bytes(already)} downloaded - D resumes", "#ffb454")
        self.query_one("#detail-meta", Static).update(meta)
        self.query_one("#detail-why", Static).update(spec.why)

        fit = Text()
        needed = spec.approx_mb / 1024 + 0.45
        if settings.nominal_ram_gb > 0:
            if spec.fits(settings.nominal_ram_gb):
                fit.append(
                    f"\nFits the {settings.device.name} profile "
                    f"(~{needed:.1f} GB of {settings.nominal_ram_gb:g} GB).",
                    style="#7bd88f",
                )
            else:
                limit = settings.ram_limit_bytes
                consequence = (
                    "The RAM ceiling is on, so it will stream through "
                    f"{limit // (1024 * 1024)} MB — it will run, slowly, and the "
                    "report will show what the paging cost."
                    if limit
                    else "The RAM ceiling is off, so it will simply use more memory "
                    "than the profile claims. Turn the ceiling on in Settings to "
                    "make it stream instead."
                )
                fit.append(
                    f"\nExceeds the {settings.device.name} profile "
                    f"(~{needed:.1f} GB needed vs {settings.nominal_ram_gb:g} GB device RAM).\n"
                    + consequence,
                    style="#ffb454",
                )
        fit.append(
            f"\n\nWill run on {settings.cpu_budget_cores} core(s)"
            + (" with affinity pinning." if settings.pin_affinity else " (threads only)."),
            style="#6f7789",
        )
        self.query_one("#detail-fit", Static).update(fit)

    # ------------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        handlers = {
            "launch": self.action_launch,
            "download": self.action_download,
            "add": self.action_add_model,
            "delete": self.action_delete,
            "back": self.action_back,
        }
        handler = handlers.get(event.button.id or "")
        if handler:
            handler()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_refresh(self) -> None:
        self.refresh_table()

    # ------------------------------------------------------------------
    def action_add_model(self) -> None:
        def added(spec: ModelSpec | None) -> None:
            if spec is None:
                return
            self.refresh_table()
            table = self.query_one("#model-table", DataTable)
            for row, entry in enumerate(cat.CATALOG):
                if entry.id == spec.id:
                    table.move_cursor(row=row)
                    break
            self.notify(
                f"Added {spec.name} ({spec.size_label}). Press D to download it.",
                title="feast AI",
            )

        self.app.push_screen(AddModelScreen(), added)

    # ------------------------------------------------------------------
    def action_launch(self) -> None:
        spec = self.current_spec()
        if spec is None:
            return
        if self._downloading:
            self.notify("A download is in progress.", severity="warning")
            return
        if not spec.downloaded:
            self.notify(
                f"{spec.name} is not downloaded yet - press D ({spec.size_label}).",
                severity="warning",
            )
            return
        if not self.app.runtime.available:
            self.notify("llama.cpp runtime is not installed.", severity="error")
            return
        self.app.active_model = spec
        self.app.active_model_path = spec.local_path
        # A fresh instance every time. Pushing the cached name would reuse the
        # previous session's screen, whose client still points at the old
        # (now dead) server port, and whose transcript and turn stats belong
        # to a different model.
        self.app.push_screen(ChatScreen())

    # ------------------------------------------------------------------
    def action_cancel_download(self) -> None:
        if self._cancel is not None:
            self._cancel.set()
            self.notify("Cancelling download...")

    def action_download(self) -> None:
        spec = self.current_spec()
        if spec is None:
            return
        if self._downloading:
            self.notify("Already downloading - Ctrl+C cancels.", severity="warning")
            return
        if spec.downloaded:
            self.notify(f"{spec.name} is already on disk.")
            return
        self._start_download(spec)

    @work(exclusive=True)
    async def _start_download(self, spec: ModelSpec) -> None:
        bar = self.query_one("#download-bar", ProgressBar)
        status = self.query_one("#download-status", Static)
        self._cancel = asyncio.Event()
        self._downloading = spec.id
        bar.add_class("active")
        bar.update(total=100, progress=0)

        # Honour the preferred quantisation for catalog models. Custom models
        # keep the exact file the user chose when adding them, direct-link
        # models have no repo to enumerate, and ONNX models are a fixed file
        # set with no quant to choose.
        if spec.tier != "custom" and spec.is_hf and not spec.is_onnx:
            status.update("[#8b93a7]Resolving file on Hugging Face...[/]")
            resolved = await asyncio.to_thread(
                cat.resolve_filename, spec, self.app.settings.preferred_quant
            )
            if resolved != spec.effective_filename:
                cat.set_override(spec.id, resolved)
                self.show_detail(spec)

        def on_progress(progress) -> None:
            bar.update(total=100, progress=progress.fraction * 100)
            eta = progress.eta_s
            status.update(
                f"[#8b93a7]{human_bytes(progress.downloaded)}"
                f"{' / ' + human_bytes(progress.total) if progress.total else ''}"
                f"   {human_bytes(progress.speed_bps)}/s"
                f"{'   ETA ' + _fmt_eta(eta) if eta else ''}[/]"
            )

        try:
            if spec.is_onnx:
                # A directory of files, all of which must land before the
                # runtime can load anything.
                status.update(
                    f"[#8b93a7]Downloading {len(spec.onnx_files)} files "
                    f"({spec.size_label})...[/]"
                )
                await download_many(
                    spec.onnx_urls,
                    spec.local_path,
                    total_bytes=spec.approx_mb * 1024 * 1024,
                    on_progress=on_progress,
                    cancel=self._cancel,
                )
            else:
                status.update(f"[#8b93a7]Downloading {spec.effective_filename}...[/]")
                await download(
                    spec.url, spec.local_path,
                    on_progress=on_progress, cancel=self._cancel,
                )
            status.update(
                f"[#7bd88f]{spec.name} ready — {human_bytes(spec.local_size)}[/]"
            )
            self.notify(f"{spec.name} downloaded.", title="feast AI")
        except asyncio.CancelledError:
            status.update("[#ffb454]Cancelled. Press D to resume where it stopped.[/]")
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            status.update(f"[#ff5f5f]{exc}[/]")
            self.notify(str(exc), severity="error", title="Download failed", timeout=10)
        finally:
            bar.remove_class("active")
            self._downloading = None
            self._cancel = None
            self.refresh_table()

    # ------------------------------------------------------------------
    def action_delete(self) -> None:
        spec = self.current_spec()
        if spec is None:
            return
        custom = spec.tier == "custom"
        has_partial = partial_bytes(spec.local_path)
        on_disk = spec.downloaded or has_partial

        if not on_disk and not custom:
            self.notify("Nothing to delete.", severity="warning")
            return

        size = human_bytes(spec.local_size or has_partial)
        if custom:
            question = (
                f"Remove [b]{spec.name}[/] from the catalog?\n\n"
                + (
                    f"[#6f7789]Its weights ({size}) will be deleted too. "
                    "You can add it again from the same link.[/]"
                    if on_disk
                    else "[#6f7789]No weights are downloaded, so this only "
                    "removes the entry.[/]"
                )
            )
            label = "Remove"
        else:
            question = (
                f"Delete the weights for [b]{spec.name}[/]?\n\n"
                f"[#6f7789]{size} will be freed. It stays in the catalog and "
                "can be downloaded again.[/]"
            )
            label = "Delete"

        def confirmed(result: bool | None) -> None:
            if not result:
                return
            if spec.is_onnx:
                # Remove only the files the catalog declared, then the folder
                # if that emptied it. A blind rmtree here would take anything
                # else the user had put in the model directory with it.
                root = spec.local_path
                for name in spec.onnx_files:
                    (root / name).unlink(missing_ok=True)
                    clear_partial(root / name)
                try:
                    root.rmdir()
                except OSError:
                    pass
            else:
                spec.local_path.unlink(missing_ok=True)
                clear_partial(spec.local_path)
            if custom:
                cat.remove_custom(spec.id)
                self.notify(f"Removed {spec.name} from the catalog.")
            else:
                self.notify(f"Deleted {spec.name} ({size} freed).")
            self.refresh_table()

        self.app.push_screen(
            ConfirmScreen(question, confirm_label=label, danger=True), confirmed
        )


def _engine_label(spec) -> str:
    """What runs this model, plus whether that backend is actually installed."""
    if not spec.is_onnx:
        return "llama.cpp (GGUF)"
    from ...engine import onnx_available, onnx_version

    if onnx_available():
        return f"ONNX Runtime GenAI {onnx_version()}"
    return "ONNX Runtime GenAI - NOT INSTALLED"


def _fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
