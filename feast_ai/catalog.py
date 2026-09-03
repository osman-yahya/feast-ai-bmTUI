"""The feast AI model catalog.

Every entry below was verified against the live Hugging Face API in July 2026:
the repo exists, is public, and ships the quantisation we ask for.

Selection criteria
------------------
1. **Edge-first.** The model was designed or explicitly released for
   phone / SBC / NPU-class deployment, not merely shrunk after the fact.
2. **CPU-friendly.** A GGUF build exists so llama.cpp can run it with no GPU.
3. **Turkish.** Weighted heavily per the project brief: entries are scored
   ``tr`` 1-5 and four Turkish-native fine-tunes are included outright.

``rank`` orders the ten primary edge picks; the Turkish specialists and the
tiny bonus model follow.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .paths import MODELS_DIR

HF_ENDPOINT = "https://huggingface.co"
USER_AGENT = "feast-ai/1.0"

_QUANT_RE = re.compile(r"(IQ\d[\w]*|Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32)", re.IGNORECASE)

#: Persisted map of ``model id -> chosen GGUF filename``.  Written whenever
#: the user downloads a quantisation other than the catalog default, so the
#: app finds the file again on the next run.
OVERRIDE_FILE = MODELS_DIR / "resolved.json"
_OVERRIDES: dict[str, str] | None = None


def _overrides() -> dict[str, str]:
    global _OVERRIDES
    if _OVERRIDES is None:
        try:
            _OVERRIDES = json.loads(OVERRIDE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _OVERRIDES = {}
    return _OVERRIDES


def set_override(model_id: str, filename: str) -> None:
    """Remember which GGUF file a model resolved to."""
    data = _overrides()
    if data.get(model_id) == filename:
        return
    data[model_id] = filename
    OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        OVERRIDE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


@dataclass(frozen=True)
class ModelSpec:
    """One catalog entry."""

    id: str
    name: str
    repo: str
    filename: str
    params: str
    approx_mb: int
    context: int
    vendor: str
    license: str
    tr: int  # Turkish capability, 1 (poor) - 5 (native fine-tune)
    tier: str  # "edge" | "turkish" | "reference"
    rank: int
    why: str
    tags: tuple[str, ...] = ()
    system_prompt_ok: bool = True
    #: Set for models added by direct link (non-Hugging Face hosts). When
    #: present it is downloaded verbatim instead of building a HF resolve URL.
    source_url: str = ""
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    #: Which backend runs this model. ``llama.cpp`` reads a single GGUF file;
    #: ``onnx`` needs a whole directory (weights + tokenizer + genai_config).
    engine: str = "llama.cpp"
    #: ONNX only: the files making up the model, and the sub-directory inside
    #: the repo they live in ("" when the model sits at the repo root).
    onnx_files: tuple[str, ...] = ()
    onnx_subdir: str = ""
    #: ONNX only: what the weights were quantised to, since there is no
    #: filename convention to parse it out of the way GGUF has.
    onnx_precision: str = ""

    # ------------------------------------------------------------------
    @property
    def is_onnx(self) -> bool:
        return self.engine == "onnx"

    @property
    def effective_filename(self) -> str:
        """The GGUF actually in use - the pinned default, or a resolved quant."""
        return _overrides().get(self.id, self.filename)

    @property
    def quant(self) -> str:
        if self.is_onnx:
            return self.onnx_precision or "?"
        found = _QUANT_RE.search(Path(self.effective_filename).stem)
        return found.group(0).upper() if found else "?"

    @property
    def local_path(self) -> Path:
        """The GGUF file, or - for ONNX - the directory the runtime is given."""
        if self.is_onnx:
            return MODELS_DIR / self.id
        return MODELS_DIR / self.id / self.effective_filename

    @property
    def downloaded(self) -> bool:
        if self.is_onnx:
            # Every declared file must be present: a half-fetched ONNX model
            # loads and then fails deep inside the runtime, which is far worse
            # to diagnose than refusing to start.
            root = self.local_path
            return bool(self.onnx_files) and all(
                (root / name).exists() for name in self.onnx_files
            )
        path = self.local_path
        return path.exists() and path.stat().st_size > 1_000_000

    @property
    def local_size(self) -> int:
        if self.is_onnx:
            root = self.local_path
            return sum(
                (root / name).stat().st_size
                for name in self.onnx_files
                if (root / name).exists()
            )
        path = self.local_path
        return path.stat().st_size if path.exists() else 0

    @property
    def onnx_urls(self) -> list[tuple[str, str]]:
        """``(url, relative destination)`` for each file the model needs."""
        prefix = f"{self.onnx_subdir}/" if self.onnx_subdir else ""
        return [
            (
                f"{HF_ENDPOINT}/{self.repo}/resolve/main/{prefix}{name}?download=true",
                name,
            )
            for name in self.onnx_files
        ]

    @property
    def url(self) -> str:
        if self.source_url:
            return self.source_url
        return (
            f"{HF_ENDPOINT}/{self.repo}/resolve/main/"
            f"{self.effective_filename}?download=true"
        )

    @property
    def page_url(self) -> str:
        if self.source_url:
            return self.source_url
        return f"{HF_ENDPOINT}/{self.repo}"

    @property
    def is_hf(self) -> bool:
        return not self.source_url

    @property
    def tr_stars(self) -> str:
        return "*" * self.tr + "." * (5 - self.tr)

    @property
    def size_label(self) -> str:
        if self.approx_mb >= 1024:
            return f"{self.approx_mb / 1024:.1f} GB"
        return f"{self.approx_mb} MB"

    def fits(self, nominal_ram_gb: float) -> bool:
        """Rough 'would this fit the simulated device?' check.

        Weights plus a KV-cache/runtime allowance, against the profile's
        nominal RAM.  Advisory only - feast AI never enforces a RAM ceiling.
        """
        if nominal_ram_gb <= 0:
            return True
        needed_gb = self.approx_mb / 1024 + 0.45
        return needed_gb <= nominal_ram_gb * 0.75


# --------------------------------------------------------------------------
# Ten primary edge picks
# --------------------------------------------------------------------------
CATALOG: list[ModelSpec] = [
    ModelSpec(
        id="gemma-4-e2b",
        name="Gemma 4 E2B Instruct",
        repo="unsloth/gemma-4-E2B-it-GGUF",
        filename="gemma-4-E2B-it-Q4_K_M.gguf",
        params="~2B eff.",
        approx_mb=2900,
        context=131072,
        vendor="Google DeepMind",
        license="Gemma Terms",
        tr=4,
        tier="edge",
        rank=1,
        why=(
            "Google's current edge flagship. The E-series uses per-layer "
            "embeddings so a larger model runs at a ~2B memory footprint. "
            "140+ languages including solid Turkish, 128K context, QAT "
            "weights tuned for 4-bit. The best quality-per-megabyte on phones."
        ),
        tags=("multilingual", "128k ctx", "QAT", "2026"),
    ),
    ModelSpec(
        id="qwen3-1.7b",
        name="Qwen3 1.7B Instruct",
        repo="unsloth/Qwen3-1.7B-GGUF",
        filename="Qwen3-1.7B-Q4_K_M.gguf",
        params="1.7B",
        approx_mb=1100,
        context=32768,
        vendor="Alibaba Qwen",
        license="Apache-2.0",
        tr=4,
        tier="edge",
        rank=2,
        why=(
            "The strongest genuinely multilingual model in the 1-2B band: "
            "100+ languages with real Turkish coverage, Apache-2.0, and a "
            "hybrid thinking/non-thinking mode you can switch per request."
        ),
        tags=("multilingual", "apache-2.0", "thinking mode"),
    ),
    ModelSpec(
        id="lfm2.5-350m",
        name="LFM2.5 350M",
        repo="LiquidAI/LFM2.5-350M-GGUF",
        filename="LFM2.5-350M-Q4_K_M.gguf",
        params="350M",
        approx_mb=240,
        context=32768,
        vendor="Liquid AI",
        license="LFM Open License",
        tr=2,
        tier="edge",
        rank=3,
        why=(
            "Purpose-built for on-device: a hybrid conv + gated architecture "
            "that decodes roughly 2x faster than a transformer of the same "
            "size on CPU. If you want to see high tokens/s on 2 cores, "
            "start here."
        ),
        tags=("fastest cpu", "hybrid arch", "tiny"),
    ),
    ModelSpec(
        id="gemma-3-1b",
        name="Gemma 3 1B Instruct",
        repo="unsloth/gemma-3-1b-it-GGUF",
        filename="gemma-3-1b-it-Q4_K_M.gguf",
        params="1B",
        approx_mb=810,
        context=32768,
        vendor="Google DeepMind",
        license="Gemma Terms",
        tr=3,
        tier="edge",
        rank=4,
        why=(
            "The reference 1B instruct model that most Android on-device demos "
            "are built on. Predictable latency, well-behaved chat template, "
            "and a good baseline to benchmark everything else against."
        ),
        tags=("baseline", "android"),
    ),
    ModelSpec(
        id="lfm2-1.2b",
        name="LFM2 1.2B",
        repo="LiquidAI/LFM2-1.2B-GGUF",
        filename="LFM2-1.2B-Q4_K_M.gguf",
        params="1.2B",
        approx_mb=760,
        context=32768,
        vendor="Liquid AI",
        license="LFM Open License",
        tr=3,
        tier="edge",
        rank=5,
        why=(
            "Matches Qwen3-1.7B on many tasks with 47% fewer parameters, and "
            "keeps the LFM2 CPU decode advantage. The best speed/quality "
            "compromise for a mid-range phone budget."
        ),
        tags=("efficient", "hybrid arch"),
    ),
    ModelSpec(
        id="qwen3-0.6b",
        name="Qwen3 0.6B Instruct",
        repo="unsloth/Qwen3-0.6B-GGUF",
        filename="Qwen3-0.6B-Q4_K_M.gguf",
        params="0.6B",
        approx_mb=400,
        context=32768,
        vendor="Alibaba Qwen",
        license="Apache-2.0",
        tr=3,
        tier="edge",
        rank=6,
        why=(
            "The smallest model here that still answers Turkish coherently. "
            "Under 400 MB at Q4, so it fits a 2 GB device with room for the "
            "KV cache."
        ),
        tags=("multilingual", "apache-2.0", "sub-1b"),
    ),
    ModelSpec(
        id="phi-4-mini",
        name="Phi-4 Mini Instruct",
        repo="unsloth/Phi-4-mini-instruct-GGUF",
        filename="Phi-4-mini-instruct-Q4_K_M.gguf",
        params="3.8B",
        approx_mb=2500,
        context=131072,
        vendor="Microsoft",
        license="MIT",
        tr=3,
        tier="edge",
        rank=7,
        why=(
            "Class leader under 4B on reasoning and maths (~67% MMLU, ~87% "
            "GSM8K). Heavy for a budget phone, but the accuracy ceiling of "
            "the whole catalog on a flagship profile."
        ),
        tags=("reasoning", "mit", "128k ctx"),
    ),
    ModelSpec(
        id="granite-4-1b",
        name="Granite 4.0 1B",
        repo="ibm-granite/granite-4.0-1b-GGUF",
        filename="granite-4.0-1b-Q4_K_M.gguf",
        params="1B",
        approx_mb=700,
        context=32768,
        vendor="IBM",
        license="Apache-2.0",
        tr=2,
        tier="edge",
        rank=8,
        why=(
            "IBM's edge line, trained on 15T tokens with function calling "
            "baked in. The enterprise-friendly pick: Apache-2.0, documented "
            "data provenance, tiny memory profile."
        ),
        tags=("tool calling", "apache-2.0", "enterprise"),
    ),
    ModelSpec(
        id="smollm3-3b",
        name="SmolLM3 3B",
        repo="unsloth/SmolLM3-3B-GGUF",
        filename="SmolLM3-3B-Q4_K_M.gguf",
        params="3B",
        approx_mb=1900,
        context=65536,
        vendor="Hugging Face",
        license="Apache-2.0",
        tr=2,
        tier="edge",
        rank=9,
        why=(
            "Fully open pipeline - data, recipe and weights are all public, "
            "which makes it the honest choice for reproducible thesis "
            "benchmarks. Dual-mode reasoning, 6 official languages."
        ),
        tags=("fully open", "reproducible", "apache-2.0"),
    ),
    ModelSpec(
        id="gemma-3-270m",
        name="Gemma 3 270M Instruct",
        repo="unsloth/gemma-3-270m-it-GGUF",
        filename="gemma-3-270m-it-Q4_K_M.gguf",
        params="270M",
        approx_mb=200,
        context=32768,
        vendor="Google DeepMind",
        license="Gemma Terms",
        tr=1,
        tier="edge",
        rank=10,
        why=(
            "The floor of the field: ~200 MB, runs on a smartwatch-class core "
            "budget. Not a general chatbot - it is the model you fine-tune "
            "for one narrow task. Included to show what 1 core actually buys."
        ),
        tags=("smallest", "270m", "fine-tune base"),
    ),
    # ----------------------------------------------------------------
    # Bonus tiny
    # ----------------------------------------------------------------
    ModelSpec(
        id="granite-4-350m",
        name="Granite 4.0 350M",
        repo="ibm-granite/granite-4.0-350m-GGUF",
        filename="granite-4.0-350m-Q4_K_M.gguf",
        params="350M",
        approx_mb=260,
        context=32768,
        vendor="IBM",
        license="Apache-2.0",
        tr=1,
        tier="edge",
        rank=11,
        why=(
            "IBM's smallest Granite. Pairs with the 270M Gemma as a second "
            "data point at the very bottom of the size curve."
        ),
        tags=("tiny", "apache-2.0"),
    ),
    # ----------------------------------------------------------------
    # Turkish specialists
    # ----------------------------------------------------------------
    ModelSpec(
        id="llama32-1b-tr",
        name="Llama 3.2 1B Instruct · Turkish",
        repo="MageLord/Llama-3.2-1B-Instruct-Turkish-GGUF",
        filename="Llama-3.2-1B-Instruct-Turkish.Q4_K_M.gguf",
        params="1B",
        approx_mb=810,
        context=131072,
        vendor="community fine-tune",
        license="Llama 3.2 Community",
        tr=5,
        tier="turkish",
        rank=12,
        why=(
            "The one model here that is BOTH Turkish-native and genuinely "
            "phone-sized. A Turkish SFT on Llama 3.2 1B at ~800 MB - this is "
            "the entry that makes the Turkish-on-edge story work."
        ),
        tags=("TURKISH", "edge size", "recommended"),
    ),
    ModelSpec(
        id="cosmos-llama-8b-tr",
        name="Cosmos Turkish-Llama 8B Instruct",
        repo="ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1-GGUF",
        filename="Turkish-Llama-8b-Instruct-v0.1.Q4_K_M.gguf",
        params="8B",
        approx_mb=4900,
        context=8192,
        vendor="YTU COSMOS Research Group",
        license="Llama 3 Community",
        tr=5,
        tier="turkish",
        rank=13,
        why=(
            "Yildiz Teknik Universitesi COSMOS group's Llama-3 8B continued-"
            "pretrained on ~30 GB of Turkish. The academic reference for "
            "Turkish LLM quality. Too big for a phone - use it as the "
            "quality ceiling you compare the edge models against."
        ),
        tags=("TURKISH", "academic", "reference quality"),
    ),
    ModelSpec(
        id="cosmos-gemma-9b-t1",
        name="Cosmos Turkish-Gemma 9B T1",
        repo="ytu-ce-cosmos/Turkish-Gemma-9b-T1-GGUF",
        filename="Turkish-Gemma-9b-T1.Q4_K_M.gguf",
        params="9B",
        approx_mb=5800,
        context=16384,
        vendor="YTU COSMOS Research Group",
        license="Gemma Terms",
        tr=5,
        tier="turkish",
        rank=14,
        why=(
            "COSMOS' reasoning-tuned Turkish Gemma-2 9B: 77.4 on Turkish "
            "GSM8K and a 68.7% win rate over its base. The best Turkish "
            "reasoning available as an open GGUF."
        ),
        tags=("TURKISH", "reasoning", "gsm8k-tr"),
    ),
    ModelSpec(
        id="trendyol-8b-tr",
        name="Trendyol LLM 8B Chat v2.0",
        repo="bartowski/Trendyol-LLM-8b-chat-v2.0-GGUF",
        filename="Trendyol-LLM-8b-chat-v2.0-Q4_K_M.gguf",
        params="8B",
        approx_mb=4900,
        context=8192,
        vendor="Trendyol",
        license="Llama 3 Community",
        tr=5,
        tier="turkish",
        rank=15,
        why=(
            "Industry counterpart to the COSMOS models - trained by Trendyol "
            "on Turkish + English with an e-commerce/conversational slant. "
            "Useful when you want a product-facing Turkish tone."
        ),
        tags=("TURKISH", "industry", "chat"),
    ),
    # ----------------------------------------------------------------------
    # ONNX Runtime GenAI builds.
    #
    # The first two deliberately mirror GGUF entries above (Qwen3 0.6B and
    # Llama 3.2 1B) so the two engines can be compared on the same weights,
    # the same core budget and the same RAM ceiling - which is the only way
    # an engine comparison means anything.
    # ----------------------------------------------------------------------
    ModelSpec(
        id="qwen3-0.6b-onnx",
        name="Qwen3 0.6B Instruct (ONNX)",
        repo="justinchuby/qwen3-0.6b-onnx-genai",
        filename="",
        params="0.6B",
        approx_mb=512,
        context=40960,
        vendor="Qwen / ONNX community",
        license="Apache 2.0",
        tr=3,
        tier="onnx",
        rank=16,
        why=(
            "Same weights as the GGUF Qwen3 0.6B above, built for ONNX Runtime "
            "GenAI. Run both to see what the runtime alone is worth on identical "
            "hardware."
        ),
        tags=("ONNX", "A/B", "reasoning"),
        engine="onnx",
        onnx_precision="int8",
        onnx_files=(
            "model.onnx",
            "genai_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            # This repo keeps its chat template in a side-car file rather than
            # inline in tokenizer_config.json. Without it the runtime raises
            # "Empty chat template" and the model produces degenerate output.
            "chat_template.jinja",
        ),
    ),
    ModelSpec(
        id="llama32-1b-onnx",
        name="Llama 3.2 1B Instruct (ONNX)",
        repo="onnx-community/Llama-3.2-1B-Instruct-GENAI-ONNX",
        filename="",
        params="1B",
        approx_mb=1781,
        context=131072,
        vendor="Meta / ONNX community",
        license="Llama 3.2 Community",
        tr=3,
        tier="onnx",
        rank=17,
        why=(
            "Official int4 CPU build (acc-level-4). Pairs with the Turkish "
            "Llama 3.2 1B fine-tune above for a runtime-vs-runtime comparison."
        ),
        tags=("ONNX", "A/B", "int4"),
        engine="onnx",
        onnx_precision="int4",
        onnx_subdir="cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
        onnx_files=(
            "model.onnx",
            "model.onnx.data",
            "genai_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ),
    ),
    ModelSpec(
        id="phi3-mini-onnx",
        name="Phi-3 Mini 4K Instruct (ONNX)",
        repo="microsoft/Phi-3-mini-4k-instruct-onnx",
        filename="",
        params="3.8B",
        approx_mb=2599,
        context=4096,
        vendor="Microsoft",
        license="MIT",
        tr=2,
        tier="onnx",
        rank=18,
        why=(
            "Microsoft's own reference build for ONNX Runtime GenAI on CPU and "
            "mobile - the best-tested model on this runtime, and a useful upper "
            "bound on what a 4 GB phone can hold."
        ),
        tags=("ONNX", "reference", "int4"),
        engine="onnx",
        onnx_precision="int4",
        onnx_subdir="cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
        onnx_files=(
            "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx",
            "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data",
            "genai_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
        ),
    ),
]

BUILTIN: list[ModelSpec] = sorted(CATALOG, key=lambda m: m.rank)

TIER_LABELS = {
    "edge": "Edge pick",
    "turkish": "Turkish",
    "reference": "Reference",
    "custom": "Custom",
}


# --------------------------------------------------------------------------
# User-added models
# --------------------------------------------------------------------------
#: Models the user added by repo id or direct URL, persisted between runs.
CUSTOM_FILE = MODELS_DIR / "custom.json"

#: Fields that survive a round trip through JSON.  ``ModelSpec`` gains fields
#: over time, so unknown keys are dropped rather than raising on load.
_SPEC_FIELDS = {
    "id", "name", "repo", "filename", "params", "approx_mb", "context",
    "vendor", "license", "tr", "tier", "rank", "why", "tags",
    "system_prompt_ok", "source_url",
}


def load_custom() -> list[ModelSpec]:
    try:
        raw = json.loads(CUSTOM_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []

    specs: list[ModelSpec] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        clean = {k: v for k, v in entry.items() if k in _SPEC_FIELDS}
        clean["tags"] = tuple(clean.get("tags") or ())
        clean.setdefault("tier", "custom")
        try:
            specs.append(ModelSpec(**clean))
        except TypeError:
            continue  # entry written by a newer/older version - skip it
    return specs


def save_custom(specs: list[ModelSpec]) -> None:
    CUSTOM_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for spec in specs:
        data = {k: getattr(spec, k) for k in _SPEC_FIELDS if hasattr(spec, k)}
        data["tags"] = list(spec.tags)
        payload.append(data)
    try:
        CUSTOM_FILE.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass


def rebuild() -> None:
    """Recompute ``CATALOG`` and ``BY_ID`` from built-ins plus custom models."""
    global CATALOG, BY_ID
    custom = load_custom()
    for offset, spec in enumerate(custom, start=1):
        object.__setattr__(spec, "rank", len(BUILTIN) + offset)
    CATALOG = BUILTIN + custom
    BY_ID = {m.id: m for m in CATALOG}


def add_custom(spec: ModelSpec) -> None:
    """Add or replace a user model, then refresh the catalog."""
    existing = [s for s in load_custom() if s.id != spec.id]
    save_custom(existing + [spec])
    rebuild()


def remove_custom(model_id: str) -> bool:
    existing = load_custom()
    remaining = [s for s in existing if s.id != model_id]
    if len(remaining) == len(existing):
        return False
    save_custom(remaining)
    rebuild()
    return True


def is_custom(model_id: str) -> bool:
    return any(s.id == model_id for s in load_custom())


rebuild()


def get(model_id: str) -> ModelSpec | None:
    return BY_ID.get(model_id)


def downloaded_models() -> list[ModelSpec]:
    return [m for m in CATALOG if m.downloaded]


def recommended_for(nominal_ram_gb: float, prefer_turkish: bool = True) -> ModelSpec:
    """Best catalog entry for a device budget, Turkish-weighted."""
    candidates = [m for m in CATALOG if m.fits(nominal_ram_gb)] or CATALOG

    def score(m: ModelSpec) -> tuple[float, float]:
        tr_weight = m.tr * (2.0 if prefer_turkish else 0.5)
        return (tr_weight - m.rank * 0.35, -m.approx_mb)

    return max(candidates, key=score)


# --------------------------------------------------------------------------
# Live Hugging Face resolution
# --------------------------------------------------------------------------
def _hf_get(url: str, timeout: float = 20.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def list_repo_ggufs(repo: str, timeout: float = 20.0) -> list[str]:
    """Every ``.gguf`` file in a public HF repo. Empty list on any failure."""
    try:
        data = _hf_get(f"{HF_ENDPOINT}/api/models/{repo}", timeout=timeout)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return []
    return [
        sibling["rfilename"]
        for sibling in data.get("siblings", [])
        if str(sibling.get("rfilename", "")).lower().endswith(".gguf")
    ]


def resolve_filename(
    spec: ModelSpec,
    preferred_quant: str,
    timeout: float = 20.0,
) -> str:
    """Pick the best available GGUF in ``spec.repo`` for ``preferred_quant``.

    Falls back to the catalog's pinned filename whenever the network is
    unavailable or nothing better matches, so the app still works offline.
    """
    if preferred_quant.upper() in spec.filename.upper():
        return spec.filename

    files = list_repo_ggufs(spec.repo, timeout=timeout)
    if not files:
        return spec.filename

    # Skip sharded and multimodal-projector files - we want a single weight file.
    files = [
        f
        for f in files
        if "mmproj" not in f.lower() and not re.search(r"-0000\d-of-\d", f)
    ]
    if not files:
        return spec.filename

    from .config import QUANT_PREFERENCES

    order = [preferred_quant.upper()] + [
        q for q in QUANT_PREFERENCES if q.upper() != preferred_quant.upper()
    ]
    for quant in order:
        for name in files:
            found = _QUANT_RE.search(Path(name).stem)
            if found and found.group(0).upper() == quant:
                return name

    return spec.filename if spec.filename in files else files[0]


# --------------------------------------------------------------------------
# Adding models by repo id or URL
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RepoFile:
    """One downloadable GGUF found on a remote host."""

    filename: str
    size: int

    @property
    def quant(self) -> str:
        found = _QUANT_RE.search(Path(self.filename).stem)
        return found.group(0).upper() if found else "?"

    @property
    def label(self) -> str:
        mb = self.size / (1024 * 1024)
        size = f"{mb/1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"
        return f"{self.quant:<10} {size:>9}   {self.filename}"


class LookupError_(RuntimeError):
    """Raised when user input cannot be turned into a downloadable model."""


_HF_HOSTS = ("huggingface.co", "hf.co")


def parse_source(text: str) -> tuple[str, str]:
    """Classify user input.

    Returns ``("repo", "owner/name")`` for anything resolvable on Hugging
    Face, or ``("url", "https://...gguf")`` for a direct link on any host.
    """
    text = text.strip().strip("<>").rstrip("/")
    if not text:
        raise LookupError_("Enter a Hugging Face repo id or a direct .gguf URL.")

    if "://" not in text:
        parts = [p for p in text.split("/") if p]
        if len(parts) == 2:
            return "repo", "/".join(parts)
        raise LookupError_(
            f"'{text}' is not a repo id. Use owner/name, e.g. "
            "unsloth/gemma-3-1b-it-GGUF."
        )

    from urllib.parse import urlparse

    parsed = urlparse(text)
    host = parsed.netloc.lower().removeprefix("www.")
    segments = [s for s in parsed.path.split("/") if s]

    if host in _HF_HOSTS:
        # .../owner/name/resolve/<rev>/<file>.gguf  -> a specific file
        if "resolve" in segments:
            index = segments.index("resolve")
            if index >= 2 and segments[-1].lower().endswith(".gguf"):
                return "url", text
        if len(segments) >= 2:
            return "repo", f"{segments[0]}/{segments[1]}"
        raise LookupError_("That Hugging Face URL has no owner/name in it.")

    if not text.lower().split("?")[0].endswith(".gguf"):
        raise LookupError_(
            "Only Hugging Face repos and direct .gguf links are supported. "
            f"'{host}' link does not end in .gguf."
        )
    return "url", text


def list_repo_files(repo: str, timeout: float = 25.0) -> list[RepoFile]:
    """Every GGUF in a public repo, with real sizes, largest quant last."""
    url = f"{HF_ENDPOINT}/api/models/{repo}/tree/main?recursive=true"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            entries = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise LookupError_(f"Repo '{repo}' was not found on Hugging Face.") from exc
        if exc.code in (401, 403):
            # Hugging Face answers 401 for both "does not exist" and "you may
            # not see this", so the message has to cover both.
            raise LookupError_(
                f"'{repo}' was not found, or is private/gated. Check the "
                "spelling; feast AI can only download public weights."
            ) from exc
        raise LookupError_(f"Hugging Face returned HTTP {exc.code} for '{repo}'.") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise LookupError_(f"Could not reach Hugging Face: {exc}") from exc

    files: list[RepoFile] = []
    for entry in entries if isinstance(entries, list) else []:
        path = str(entry.get("path", ""))
        if not path.lower().endswith(".gguf"):
            continue
        if "mmproj" in path.lower() or re.search(r"-0000\d-of-\d", path):
            continue  # projector / sharded parts are not standalone weights
        size = (entry.get("lfs") or {}).get("size") or entry.get("size") or 0
        files.append(RepoFile(path, int(size)))

    if not files:
        raise LookupError_(
            f"'{repo}' has no single-file GGUF weights. feast AI needs a GGUF "
            "repo - look for one with '-GGUF' in the name."
        )
    return sorted(files, key=lambda f: f.size)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "custom-model"


def build_custom_spec(
    *,
    name: str,
    repo: str,
    filename: str,
    size_bytes: int,
    source_url: str = "",
    params: str = "?",
    context: int = 4096,
    tr: int = 3,
    vendor: str = "",
    note: str = "",
) -> ModelSpec:
    """Assemble a :class:`ModelSpec` for a user-added model."""
    model_id = slugify(name or repo or Path(filename).stem)
    existing = {s.id for s in BUILTIN}
    if model_id in existing:
        model_id = f"{model_id}-custom"
    return ModelSpec(
        id=model_id,
        name=name.strip() or repo or Path(filename).stem,
        repo=repo,
        filename=filename,
        params=params,
        approx_mb=max(1, int(size_bytes / (1024 * 1024))),
        context=context,
        vendor=vendor or (repo.split("/")[0] if "/" in repo else "custom"),
        license="see model card",
        tr=max(1, min(5, tr)),
        tier="custom",
        rank=999,
        why=note or f"Added by you from {source_url or repo}.",
        tags=("custom",),
        source_url=source_url,
    )


def remote_size(url: str, timeout: float = 20.0) -> int | None:
    """Content-Length for a HF resolve URL, following redirects."""
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            linked = response.headers.get("X-Linked-Size")
            return int(linked or length) if (linked or length) else None
    except (urllib.error.URLError, OSError, ValueError):
        return None
