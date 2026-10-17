"""Filesystem layout for feast AI.

Everything lives under ``~/.feast-ai`` so the repo stays clean and model
weights survive a re-clone.  ``FEAST_AI_HOME`` overrides the root.
"""

from __future__ import annotations

import os
from pathlib import Path


def _root() -> Path:
    override = os.environ.get("FEAST_AI_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".feast-ai"


HOME = _root()
CONFIG_FILE = HOME / "config.json"
MODELS_DIR = HOME / "models"
BIN_DIR = HOME / "bin"
RUNTIME_DIR = BIN_DIR / "llama.cpp"
SESSIONS_DIR = HOME / "sessions"
LOG_DIR = HOME / "logs"
#: Encoder models (the intent router), kept apart from the generative catalog.
ENCODERS_DIR = HOME / "encoders"
#: The simulated phone state the agent prompt's CONTEXT block is rendered from.
AGENT_CONTEXT_FILE = HOME / "agent_context.json"
#: Saved llama.cpp slot states - one precomputed KV snapshot per intent
#: prompt, standing in for what a phone would keep on flash.
KV_DIR = HOME / "kv"


def ensure_dirs() -> None:
    """Create every directory feast AI writes to."""
    for path in (HOME, MODELS_DIR, BIN_DIR, RUNTIME_DIR, SESSIONS_DIR, LOG_DIR, ENCODERS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def human_bytes(num: float | int | None) -> str:
    """Format a byte count for display (``1.2 GB``)."""
    if num is None:
        return "—"
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{value:.0f} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"
