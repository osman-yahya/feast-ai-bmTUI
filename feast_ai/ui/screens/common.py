"""Modal screens shared across the app."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Input,
    Label,
    ProgressBar,
    Select,
    Static,
    TextArea,
)


class ConfirmScreen(ModalScreen[bool]):
    """A yes/no modal."""

    BINDINGS = [("escape", "dismiss_false", "Cancel")]

    def __init__(self, question: str, confirm_label: str = "Yes", danger: bool = False) -> None:
        super().__init__()
        self.question = question
        self.confirm_label = confirm_label
        self.danger = danger

    def compose(self) -> ComposeResult:
        with Container(id="confirm-card"):
            yield Static(self.question, id="confirm-question")
            with Horizontal(id="confirm-actions"):
                yield Button(
                    self.confirm_label,
                    id="confirm-yes",
                    variant="error" if self.danger else "primary",
                )
                yield Button("Cancel", id="confirm-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-yes")

    def action_dismiss_false(self) -> None:
        self.dismiss(False)


class LoadingScreen(ModalScreen[None]):
    """A blocking progress modal used while the model loads."""

    def __init__(self, title: str, detail: str = "") -> None:
        super().__init__()
        self._title = title
        self._detail = detail

    def compose(self) -> ComposeResult:
        with Container(id="loading-card"):
            yield Static(self._title, id="loading-title")
            yield Static(self._detail, id="loading-detail")
            yield ProgressBar(total=100, show_eta=False, id="loading-progress")

    def set_detail(self, text: str) -> None:
        try:
            self.query_one("#loading-detail", Static).update(text)
        except Exception:
            pass

    def set_progress(self, fraction: float | None) -> None:
        try:
            bar = self.query_one("#loading-progress", ProgressBar)
            if fraction is None:
                bar.update(total=None)
            else:
                bar.update(total=100, progress=max(0.0, min(1.0, fraction)) * 100)
        except Exception:
            pass


class SystemPromptScreen(ModalScreen[str | None]):
    """Edit, load, save and delete system prompts."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("ctrl+s", "apply", "Apply"),
    ]

    def __init__(self, current: str) -> None:
        super().__init__()
        self.current = current

    def compose(self) -> ComposeResult:
        from ...prompts import library

        with Container(id="prompt-card"):
            yield Label("System prompt", classes="panel-title")
            yield Select(
                library.options(),
                id="sp-preset",
                prompt="Load a saved prompt or preset...",
            )
            yield TextArea(self.current, id="sp-text", soft_wrap=True)
            with Horizontal(classes="save-row"):
                yield Input(placeholder="Save as… (name)", id="sp-name")
                yield Button("Save", id="sp-save")
                yield Button("Delete", id="sp-delete")
            yield Static(id="sp-status", classes="field-help")
            yield Static(
                "★ = your saved prompts, · = built-in. Applies from the next "
                "message onward; Ctrl+S applies, Esc cancels.",
                classes="field-help",
            )
            with Horizontal(id="confirm-actions"):
                yield Button("Apply", id="sp-apply", variant="primary")
                yield Button("Cancel", id="sp-cancel")

    def on_mount(self) -> None:
        from ...prompts import library

        existing = library.find_name_for(self.current)
        if existing and not library.is_builtin(existing):
            self.query_one("#sp-name", Input).value = existing
        self.query_one("#sp-text", TextArea).focus()

    # ------------------------------------------------------------------
    def _status(self, message: str, ok: bool = True) -> None:
        colour = "#7bd88f" if ok else "#ff5f5f"
        self.query_one("#sp-status", Static).update(f"[{colour}]{message}[/]")

    def _refresh_options(self, select_name: str | None = None) -> None:
        from ...prompts import library

        select = self.query_one("#sp-preset", Select)
        select.set_options(library.options())
        if select_name and library.get(select_name) is not None:
            select.value = select_name

    # ------------------------------------------------------------------
    def on_select_changed(self, event: Select.Changed) -> None:
        from ...prompts import library

        if event.select.id != "sp-preset" or event.value is Select.BLANK:
            return
        name = str(event.value)
        body = library.get(name)
        if body is None:
            return
        self.query_one("#sp-text", TextArea).text = body
        self.query_one("#sp-name", Input).value = "" if library.is_builtin(name) else name

    def on_button_pressed(self, event: Button.Pressed) -> None:
        handlers = {
            "sp-apply": self.action_apply,
            "sp-cancel": self.action_cancel,
            "sp-save": self.action_save,
            "sp-delete": self.action_delete,
        }
        handler = handlers.get(event.button.id or "")
        if handler:
            handler()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "sp-name":
            self.action_save()

    # ------------------------------------------------------------------
    def action_save(self) -> None:
        from ...prompts import library

        name = self.query_one("#sp-name", Input).value
        text = self.query_one("#sp-text", TextArea).text
        ok, message = library.save(name, text)
        self._status(message, ok)
        if ok:
            self._refresh_options(select_name=name.strip())

    def action_delete(self) -> None:
        from ...prompts import library

        name = self.query_one("#sp-name", Input).value.strip()
        if not name:
            selected = self.query_one("#sp-preset", Select).value
            name = "" if selected is Select.BLANK else str(selected)
        ok, message = library.delete(name)
        self._status(message, ok)
        if ok:
            self.query_one("#sp-name", Input).value = ""
            self._refresh_options()

    def action_apply(self) -> None:
        self.dismiss(self.query_one("#sp-text", TextArea).text)

    def action_cancel(self) -> None:
        self.dismiss(None)
