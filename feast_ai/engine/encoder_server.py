"""Supervise the ModernBERT-TR intent router under the device budget.

The router is one more process on the simulated phone, so it gets exactly the
treatment the SLM gets from :class:`~feast_ai.engine.server.BaseServer`:
pinned to the same cores (they never run at the same time - route first, then
generate - which is how one SoC would schedule them), its own share of the
RAM ceiling, and a process tree the sampler adds to the session's telemetry.
Running it inside the TUI instead would make it free, which it is not.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from ..config import Settings
from ..encoders import EncoderSpec
from .server import BaseServer, EngineError, package_on_pythonpath

#: What the worker imports. All three are optional for feast AI as a whole.
REQUIRED = ("onnxruntime", "tokenizers", "numpy")
INSTALL_HINT = "pip install onnxruntime tokenizers numpy"


def missing_packages() -> list[str]:
    return [name for name in REQUIRED if importlib.util.find_spec(name) is None]


def available() -> bool:
    return not missing_packages()


class EncoderServer(BaseServer):
    """Owns one ``feast_ai.engine.encoder_worker`` process."""

    engine_name = "ModernBERT-TR router"
    log_prefix = "encoder"
    cap_group = "feast-ai-router"

    def __init__(
        self,
        settings: Settings,
        spec: EncoderSpec,
        bank: list[tuple[str, str]],
    ) -> None:
        super().__init__(Path(sys.executable), settings)
        self.spec = spec
        self.bank = bank

    def _check_available(self) -> None:
        missing = missing_packages()
        if missing:
            raise EngineError(
                f"The intent router needs {', '.join(missing)} in this "
                f"interpreter ({sys.executable}).\nInstall with:  {INSTALL_HINT}"
            )
        if not self.spec.downloaded:
            raise EngineError(
                f"{self.spec.name} is not downloaded.\n"
                f"Fetch it with:  python -m feast_ai pull {self.spec.id}"
            )

    def _write_bank(self, model_path: Path) -> Path:
        path = model_path / "bank.json"
        payload = {
            "prefix": self.spec.query_prefix,
            "items": [{"label": label, "text": text} for label, text in self.bank],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    def _commands(self, model_path: Path, port: int) -> list[list[str]]:
        bank = self._write_bank(model_path)
        return [[
            sys.executable,
            "-m", "feast_ai.engine.encoder_worker",
            "--model", str(model_path),
            "--onnx", self.spec.onnx_file,
            "--bank", str(bank),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--threads", str(self.settings.cpu_budget_cores),
        ]]

    def _cap_warning(self, model_path: Path) -> str:
        # The KV-cache floor heuristic is an LLM concept; an encoder has none.
        return ""

    async def _spawn_and_wait(self, cmd, status):
        with package_on_pythonpath():
            return await super()._spawn_and_wait(cmd, status)
