"""Small reusable display widgets."""

from __future__ import annotations

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

BLOCKS = " ▁▂▃▄▅▆▇█"


def bar(fraction: float, width: int = 14, filled: str = "█", empty: str = "░") -> str:
    """A fixed-width text meter."""
    fraction = max(0.0, min(1.0, fraction))
    n = int(round(fraction * width))
    return filled * n + empty * (width - n)


def level_color(pct: float) -> str:
    if pct >= 90:
        return "#ff5f5f"
    if pct >= 70:
        return "#ffb454"
    if pct >= 35:
        return "#7bd88f"
    return "#4fc1ff"


class Meter(Static):
    """A labelled horizontal meter: ``CPU  ████████░░░░  72%``."""

    value: reactive[float] = reactive(0.0)
    caption: reactive[str] = reactive("")

    def __init__(
        self,
        label: str,
        width: int = 16,
        suffix: str = "%",
        max_value: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__("", **kwargs)
        self.label = label
        self.width = width
        self.suffix = suffix
        self.max_value = max_value

    def update_value(self, value: float, caption: str = "") -> None:
        self.caption = caption
        self.value = value

    def watch_value(self) -> None:
        self._render_now()

    def watch_caption(self) -> None:
        self._render_now()

    def _render_now(self) -> None:
        pct = 0.0 if self.max_value <= 0 else 100.0 * self.value / self.max_value
        color = level_color(pct)
        text = Text()
        text.append(f"{self.label:<7}", style="bold #8b93a7")
        text.append(bar(pct / 100.0, self.width), style=color)
        readout = self.caption or f"{self.value:.0f}{self.suffix}"
        text.append(f" {readout}", style=f"bold {color}")
        self.update(text)


class StatTile(Static):
    """A big-number tile: value on top, label beneath."""

    def __init__(self, label: str, value: str = "—", unit: str = "", **kwargs) -> None:
        super().__init__("", **kwargs)
        self._label = label
        self._value = value
        self._unit = unit
        self._tone = "#e6e9f0"

    def on_mount(self) -> None:
        self._render_now()

    def set_value(self, value: str, unit: str | None = None, tone: str | None = None) -> None:
        self._value = value
        if unit is not None:
            self._unit = unit
        if tone is not None:
            self._tone = tone
        self._render_now()

    def _render_now(self) -> None:
        text = Text()
        text.append(f"{self._value}", style=f"bold {self._tone}")
        if self._unit:
            text.append(f" {self._unit}", style="#6f7789")
        text.append(f"\n{self._label}", style="#8b93a7")
        self.update(text)


class MiniSparkline(Static):
    """A one-line unicode sparkline that never blocks on empty data."""

    def __init__(self, label: str = "", width: int = 24, color: str = "#7bd88f", **kwargs) -> None:
        super().__init__("", **kwargs)
        self.label = label
        self.width = width
        self.color = color
        self._data: list[float] = []

    def set_data(self, data, ceiling: float | None = None) -> None:
        self._data = list(data)[-self.width:]
        self._ceiling = ceiling
        self._render_now()

    def _render_now(self) -> None:
        text = Text()
        if self.label:
            text.append(f"{self.label:<7}", style="#8b93a7")
        if not self._data:
            text.append("·" * self.width, style="#3a3f4b")
        else:
            top = getattr(self, "_ceiling", None) or max(self._data) or 1.0
            for value in self._data:
                index = int(min(1.0, max(0.0, value / top)) * (len(BLOCKS) - 1))
                text.append(BLOCKS[index], style=self.color)
            if len(self._data) < self.width:
                text.append("·" * (self.width - len(self._data)), style="#3a3f4b")
        self.update(text)


class Banner(Static):
    """The feast AI wordmark."""

    def on_mount(self) -> None:
        text = Text()
        text.append("\n")
        text.append("  ███████ ███████  █████  ███████ ████████\n", style="bold #ff8a3d")
        text.append("  ██      ██      ██   ██ ██         ██   \n", style="bold #ff9f5a")
        text.append("  █████   █████   ███████ ███████    ██   \n", style="bold #ffb454")
        text.append("  ██      ██      ██   ██      ██    ██   \n", style="bold #ffc97a")
        text.append("  ██      ███████ ██   ██ ███████    ██   ", style="bold #ffdda1")
        text.append("  AI\n", style="bold #4fc1ff")
        text.append("  edge LLM lab", style="#8b93a7")
        text.append("  ·  ", style="#3a3f4b")
        text.append("CPU inference", style="#8b93a7")
        text.append("  ·  ", style="#3a3f4b")
        text.append("live telemetry", style="#8b93a7")
        self.update(text)
