"""Where mustachio keeps its state: ``~/.feast-ai/mustachio/``.

Same rule as feast AI: nothing heavy in the repository.  Every download gets
its own directory (``hf/<owner>--<repo>/``), so a file from the internet
never sits next to code that could import it.
"""

from __future__ import annotations

from pathlib import Path

from feast_ai.paths import HOME as FEAST_HOME

HOME = FEAST_HOME / "mustachio"
#: Hugging Face downloads, one directory per repo.
HF_DIR = HOME / "hf"
#: Text-only checkpoints and bundles this package builds.
BUILD_DIR = HOME / "build"
#: Generated datasets (JSONL) and their stats.
DATA_DIR = HOME / "data"
#: Training runs: adapters, logs, eval reports.
RUNS_DIR = HOME / "runs"

#: The on-device chat template, pinned in the repo (see assets/README).
ASSETS = Path(__file__).resolve().parent / "assets"
CHAT_TEMPLATE = ASSETS / "chat_template.jinja"

#: The checkpoint to fine-tune: QAT weights before quantisation, so a 4-bit
#: export after training lands where the weights were trained to land.
BASE_REPO = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
#: Google's own LiteRT-LM build - the size and speed reference.
LITERT_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
LITERT_FILE = "gemma-4-E2B-it.litertlm"


def repo_dir(repo: str) -> Path:
    return HF_DIR / repo.replace("/", "--")


def ensure_dirs() -> None:
    for path in (HOME, HF_DIR, BUILD_DIR, DATA_DIR, RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)
