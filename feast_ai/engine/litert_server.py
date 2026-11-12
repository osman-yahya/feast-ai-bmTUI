"""Manage the LiteRT-LM worker under the same edge constraints.

The third backend, and the one the phone will run: Google's LiteRT-LM with a
``.litertlm`` bundle (one file, like a GGUF).  As with ONNX, everything that
makes a run comparable - affinity pinning, the RAM ceiling, priority, health
polling, process-tree telemetry - comes from
:class:`feast_ai.engine.server.BaseServer`; this class only says how to
start the worker.

The worker is launched with the current interpreter, because ``litert-lm``
ships as a Python wheel, so availability is a matter of whether the import
succeeds.

One thing the runtime does on its own: the first load of a bundle writes an
XNNPack cache file next to it (752 MB for Gemma 4 E2B, 9-50 s once; later
loads take a fraction of a second).  It is part of what the model costs on
flash, so it is left where the runtime puts it and not hidden.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ..config import Settings
from .server import BaseServer, EngineError, package_on_pythonpath, worker_python

PACKAGE = "litert-lm"
INSTALL_HINT = "pip install litert-lm"
#: The context a bundle was converted for; asking the engine for more fails.
MAX_CONTEXT = 4096


def available() -> bool:
    """Is the LiteRT-LM backend usable in this interpreter?"""
    return importlib.util.find_spec("litert_lm") is not None


def version() -> str:
    try:
        from importlib.metadata import version as dist_version

        return dist_version("litert-lm-api")
    except Exception:  # noqa: BLE001 - a missing version is not an error
        return ""


class LiteRtServer(BaseServer):
    """Owns one ``feast_ai.engine.litert_worker`` process."""

    engine_name = "LiteRT-LM"
    log_prefix = "litert-worker"

    def __init__(self, settings: Settings) -> None:
        super().__init__(Path(worker_python()), settings)

    def _check_available(self) -> None:
        if not available():
            raise EngineError(
                f"{PACKAGE} is not installed in this interpreter "
                f"({sys.executable}).\nInstall it with:  {INSTALL_HINT}"
            )

    def _commands(self, model_path: Path, port: int) -> list[list[str]]:
        s = self.settings
        return [[
            worker_python(),
            "-m", "feast_ai.engine.litert_worker",
            "--model", str(model_path),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--threads", str(s.cpu_budget_cores),
            "--ctx", str(min(s.context_size, MAX_CONTEXT)),
        ]]

    async def _spawn_and_wait(self, cmd, status):
        with package_on_pythonpath():
            return await super()._spawn_and_wait(cmd, status)
