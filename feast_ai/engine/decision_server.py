"""Supervise a Decision 2.0 model (vLLM Semantic Router) as the intent router.

Decision 2.0 models are small decoders (Kai-0.6B is a Qwen3-0.6B finetune)
with a decision head: given a *state* and a set of questions they return a
probability for every option in one pass, without generating text.  The GGUF
builds need the ``decision2-gguf`` llama.cpp branch, which adds that head and
a ``/v1/systemone`` endpoint; stock llama.cpp cannot serve them.  feast AI
builds that branch's ``llama-server`` once into ``~/.feast-ai/bin`` (see
``python -m feast_ai doctor``) and keeps the SLM on the stock runtime.

It is one more process on the simulated phone, so :class:`BaseServer` pins it
to the model's cores and gives it a RAM share, exactly like the ModernBERT
router.  Its share is larger: the Q4_K_M weights alone are 400 MB.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config import Settings
from ..paths import BIN_DIR, ENCODERS_DIR
from .server import BaseServer, EngineError

DECISION_RUNTIME_DIR = BIN_DIR / "llama.cpp-decision2"
SOURCE_HINT = (
    "git clone --depth 1 --branch decision2-gguf https://github.com/Xunzhuo/llama.cpp, "
    "build llama-server (CPU), and copy it into " + str(DECISION_RUNTIME_DIR)
)

#: id -> (HF repo, GGUF file, sha256, params). Only the phone-sized builds.
DECISION_MODELS = {
    "decision-2.0-kai-0.6b": (
        "vllm-sr/Decision-2.0-Kai-0.6B-GGUF",
        "Decision-2.0-Kai-0.6B-Q4_K_M.gguf",
        "0404cbd066555c96e9a8a9cff818b1645fe6d8868a933cd3938b2b4dea453d58",
        "0.60B",
    ),
}

#: Measured RSS after load with the context below: ~600 MB. The share is
#: carved out of the SLM's budget, like the ModernBERT router's 320 MB.
DECISION_RAM_MB = 640
#: A routing question plus a short message fits easily; the head needs the
#: whole prompt in one micro-batch, so batch = ubatch = context.
DECISION_CTX = 1024


def runtime_path() -> Path:
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    return DECISION_RUNTIME_DIR / name


def model_path(model_id: str) -> Path:
    _, filename, _, _ = DECISION_MODELS[model_id]
    return ENCODERS_DIR / model_id / filename


def is_decision(model_id: str) -> bool:
    return model_id in DECISION_MODELS


class DecisionServer(BaseServer):
    """Owns one decision2-branch ``llama-server`` serving ``/v1/systemone``."""

    engine_name = "Decision 2.0 router"
    log_prefix = "decision"
    cap_group = "feast-ai-router"

    def __init__(self, settings: Settings, model_id: str) -> None:
        super().__init__(runtime_path(), settings)
        self.model_id = model_id

    def _check_available(self) -> None:
        if not self.server_bin.exists():
            raise EngineError(f"Decision 2.0 runtime not built.\n{SOURCE_HINT}")
        if not model_path(self.model_id).exists():
            repo, filename, _, _ = DECISION_MODELS[self.model_id]
            raise EngineError(f"{filename} missing. Download it from huggingface.co/{repo}")

    def _commands(self, model: Path, port: int) -> list[list[str]]:
        n = str(self.settings.cpu_budget_cores)
        ctx = str(DECISION_CTX)
        return [[
            str(self.server_bin), "--model", str(model), "--host", "127.0.0.1",
            "--port", str(port), "--embeddings", "--pooling", "none",
            "-c", ctx, "-b", ctx, "-ub", ctx, "-np", "1",
            "--threads", n, "--threads-batch", n, "--gpu-layers", "0",
        ]]

    def _cap_warning(self, model_path: Path) -> str:
        return ""
