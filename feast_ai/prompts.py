"""Saved system prompts.

Built-in presets ship with the app and are read-only. Anything the user saves
lands in ``~/.feast-ai/prompts.json`` and can be renamed, overwritten or
deleted. Both sets appear in one picker, with built-ins listed first.
"""

from __future__ import annotations

import json

from .config import SYSTEM_PROMPT_PRESETS
from .paths import HOME

PROMPTS_FILE = HOME / "prompts.json"

BUILTIN_MARK = "·"


class PromptLibrary:
    """Named system prompts, persisted as JSON."""

    def __init__(self) -> None:
        self._user: dict[str, str] = {}
        self.load()

    # ------------------------------------------------------------------
    def load(self) -> None:
        try:
            data = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._user = {}
            return
        prompts = data.get("prompts", data)
        if isinstance(prompts, dict):
            self._user = {
                str(k): str(v) for k, v in prompts.items() if isinstance(v, str)
            }
        else:
            self._user = {}

    def _flush(self) -> None:
        PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            PROMPTS_FILE.write_text(
                json.dumps({"prompts": self._user}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    # ------------------------------------------------------------------
    @property
    def user_prompts(self) -> dict[str, str]:
        return dict(self._user)

    def is_builtin(self, name: str) -> bool:
        return name in SYSTEM_PROMPT_PRESETS

    def get(self, name: str) -> str | None:
        if name in self._user:
            return self._user[name]
        return SYSTEM_PROMPT_PRESETS.get(name)

    def names(self) -> list[str]:
        """Built-ins first, then saved prompts alphabetically."""
        return list(SYSTEM_PROMPT_PRESETS) + sorted(self._user)

    def options(self) -> list[tuple[str, str]]:
        """``(label, name)`` pairs for a Select widget."""
        items = [(f"{BUILTIN_MARK} {n}", n) for n in SYSTEM_PROMPT_PRESETS]
        items += [(f"★ {n}", n) for n in sorted(self._user)]
        return items

    # ------------------------------------------------------------------
    def save(self, name: str, text: str) -> tuple[bool, str]:
        """Store a prompt. Returns ``(ok, message)``."""
        name = " ".join(name.split()).strip()
        if not name:
            return False, "Give the prompt a name first."
        if self.is_builtin(name):
            return False, f"'{name}' is a built-in preset - pick another name."
        if len(name) > 60:
            return False, "Name is too long (60 characters max)."
        existed = name in self._user
        self._user[name] = text
        self._flush()
        return True, f"{'Updated' if existed else 'Saved'} '{name}'."

    def delete(self, name: str) -> tuple[bool, str]:
        if self.is_builtin(name):
            return False, f"'{name}' is a built-in preset and cannot be deleted."
        if name not in self._user:
            return False, f"'{name}' is not a saved prompt."
        del self._user[name]
        self._flush()
        return True, f"Deleted '{name}'."

    def find_name_for(self, text: str) -> str | None:
        """The name a prompt body is stored under, if any."""
        for name, body in self._user.items():
            if body == text:
                return name
        for name, body in SYSTEM_PROMPT_PRESETS.items():
            if body == text:
                return name
        return None


#: Process-wide library. Screens share it so a prompt saved in the chat modal
#: is immediately visible in Settings.
library = PromptLibrary()
