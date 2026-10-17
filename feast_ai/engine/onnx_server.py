"""Manage the ONNX Runtime GenAI worker under the same edge constraints.

This is deliberately thin.  All of the behaviour that makes a run comparable
between engines - affinity pinning, the RAM ceiling, priority, health polling,
process-tree telemetry - lives in :class:`feast_ai.engine.server.BaseServer`,
so a llama.cpp run and an ONNX run of the same model differ only in the
runtime under test.

The worker is launched with the *current interpreter* rather than a bundled
binary, because ``onnxruntime-genai`` ships as a Python wheel.  That makes the
availability check a matter of whether the import succeeds, not whether a file
exists on disk - so :meth:`_check_available` is overridden accordingly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ..config import Settings
from .server import BaseServer, EngineError, package_on_pythonpath

#: Advertised on the models screen and in the session report.
PACKAGE = "onnxruntime-genai"
INSTALL_HINT = "pip install onnxruntime-genai"


def available() -> bool:
    """Is the ONNX backend usable in this interpreter?"""
    return importlib.util.find_spec("onnxruntime_genai") is not None


def version() -> str:
    try:
        import onnxruntime_genai as og

        return str(og.__version__)
    except (ImportError, AttributeError):
        return ""


class OnnxServer(BaseServer):
    """Owns one ``feast_ai.engine.onnx_worker`` process."""

    engine_name = "ONNX Runtime GenAI"
    log_prefix = "onnx-worker"

    def __init__(self, settings: Settings) -> None:
        # BaseServer wants a path to derive the working directory from; the
        # interpreter is the closest true equivalent for a Python worker.
        super().__init__(Path(sys.executable), settings)

    def _check_available(self) -> None:
        if not available():
            raise EngineError(
                f"{PACKAGE} is not installed in this interpreter "
                f"({sys.executable}).\nInstall it with:  {INSTALL_HINT}"
            )

    def _commands(self, model_path: Path, port: int) -> list[list[str]]:
        s = self.settings
        return [[
            sys.executable,
            "-m", "feast_ai.engine.onnx_worker",
            "--model", str(model_path),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--threads", str(s.cpu_budget_cores),
        ]]

    async def _spawn_and_wait(self, cmd, status):
        # The worker is imported as ``feast_ai.engine.onnx_worker``, so it has
        # to run with the project root importable.
        with package_on_pythonpath():
            return await super()._spawn_and_wait(cmd, status)
