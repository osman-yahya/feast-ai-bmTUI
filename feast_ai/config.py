"""Settings model, device profiles and JSON persistence.

Design note on resource caps
----------------------------
feast AI simulates a phone-class device along two axes:

* ``threads``           -> how many llama.cpp worker threads exist at all
* ``pin_affinity``      -> a hard OS-level cap; the process may only be
                           scheduled on ``threads`` logical cores
* ``priority``          -> optional scheduling-class penalty
* ``ram_limit_*``       -> a hard ceiling on *resident* memory

The RAM ceiling is deliberately a **working-set** limit, not a commit limit.
A commit limit (Job Object ``ProcessMemoryLimit``, cgroup ``memory.max``) kills
the process the moment it asks for one byte too many, which turns every
oversized model into a crash instead of a measurement - the reason feast AI
originally shipped with no RAM cap at all.  A working-set ceiling instead makes
the OS evict pages and fault them back in, so an oversized model gets *slow*
rather than dead.  That is what a phone actually does, so it is the honest
simulation.  See :mod:`feast_ai.engine.memcap` for the mechanism and its
measured fidelity.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .paths import CONFIG_FILE, ensure_dirs

HOST_CORES = os.cpu_count() or 4


# --------------------------------------------------------------------------
# Device profiles
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DeviceProfile:
    """A synthetic edge target: CPU budget plus a nominal RAM figure."""

    key: str
    name: str
    example: str
    threads: int
    nominal_ram_gb: float
    note: str
    #: Sustained sequential read of the device's storage, MB/s.  Only used to
    #: project what streaming would cost on the real thing - see
    #: :attr:`SessionSummary.storage_ceiling_tps`.
    storage_mbps: float = 1000.0
    storage_note: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} · {self.threads} core / {self.nominal_ram_gb:g} GB"


def _clamp_cores(n: int) -> int:
    return max(1, min(int(n), HOST_CORES))


DEVICE_PROFILES: dict[str, DeviceProfile] = {
    "flagship": DeviceProfile(
        key="flagship",
        name="Flagship phone",
        example="Snapdragon 8 Gen 3 / Apple A17 Pro",
        threads=_clamp_cores(8),
        nominal_ram_gb=12,
        note="Big-core heavy SoC. Comfortable up to ~4B params at Q4.",
        storage_mbps=3500.0,
        storage_note="UFS 4.0",
    ),
    "midrange": DeviceProfile(
        key="midrange",
        name="Mid-range phone",
        example="Snapdragon 7 Gen 3 / Dimensity 7200",
        threads=_clamp_cores(6),
        nominal_ram_gb=8,
        note="The realistic mass-market target. 1B-2B at Q4 is the sweet spot.",
        storage_mbps=1700.0,
        storage_note="UFS 3.1",
    ),
    "budget": DeviceProfile(
        key="budget",
        name="Budget phone",
        example="Snapdragon 680 / Helio G85",
        threads=_clamp_cores(4),
        nominal_ram_gb=4,
        note="Sub-1B models only if you want conversational latency.",
        storage_mbps=800.0,
        storage_note="UFS 2.2",
    ),
    "entry": DeviceProfile(
        key="entry",
        name="Entry / SBC",
        example="Raspberry Pi 5, low-end Android Go",
        threads=_clamp_cores(2),
        nominal_ram_gb=2,
        note="270M-350M class. Expect single-digit tokens/s above 1B.",
        storage_mbps=250.0,
        storage_note="eMMC 5.1 / SD card",
    ),
    "wearable": DeviceProfile(
        key="wearable",
        name="Wearable / MCU-class",
        example="Smartwatch, industrial gateway",
        threads=1,
        nominal_ram_gb=1,
        note="One core. Brutal, and the most honest test of a 270M model.",
        storage_mbps=120.0,
        storage_note="eMMC",
    ),
    "desktop": DeviceProfile(
        key="desktop",
        name="Host (no cap)",
        example=f"This machine · {HOST_CORES} logical cores",
        threads=HOST_CORES,
        nominal_ram_gb=0,
        note="Reference ceiling. Use it to measure how much the caps cost you.",
        storage_mbps=3000.0,
        storage_note="NVMe SSD",
    ),
    "custom": DeviceProfile(
        key="custom",
        name="Custom",
        example="Your own CPU/RAM budget",
        threads=_clamp_cores(4),
        nominal_ram_gb=6,
        note="Set threads and nominal RAM by hand in Settings.",
        storage_mbps=1000.0,
        storage_note="unspecified",
    ),
}

PROFILE_ORDER = [
    "flagship",
    "midrange",
    "budget",
    "entry",
    "wearable",
    "desktop",
    "custom",
]

PRIORITY_CHOICES = ["normal", "below_normal", "idle"]

#: Reasoning ("thinking") control.  Verified against llama.cpp b10144: the
#: ``enable_thinking`` chat-template kwarg is honoured **per request**, so this
#: can be changed mid-conversation without reloading the model.  Templates that
#: do not define the variable simply ignore it.
THINKING_MODES = ["auto", "off", "on"]

THINKING_LABELS = {
    "auto": "Auto - use each model's default",
    "off": "Off - suppress reasoning (fastest)",
    "on": "On - force reasoning where supported",
}

THINKING_HELP = (
    "Reasoning models (Qwen3, SmolLM3, Gemma 4) spend real tokens thinking "
    "before they answer, and on a CPU budget that is the single biggest "
    "factor in perceived latency. Measured here on Qwen3 0.6B at 4 cores: "
    "163 tokens / 2.4 s with thinking, 24 tokens / 0.4 s without. Turn it off "
    "to benchmark answer throughput; leave it on to benchmark reasoning cost."
)

QUANT_PREFERENCES = [
    "Q4_K_M",
    "Q4_0",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
    "Q3_K_M",
]

DEFAULT_SYSTEM_PROMPT = (
    "Sen feast AI icinde calisan, mobil cihazda yerel olarak kosan bir yapay zeka "
    "asistanisin. Turkce sorulara dogal ve akici Turkce, Ingilizce sorulara "
    "Ingilizce cevap ver. Kisa, net ve dogru ol; emin olmadigin seyi uydurma. "
    "Sinirli CPU ve bellek ile calistigin icin gereksiz uzun cevaplardan kacin."
)

SYSTEM_PROMPT_PRESETS: dict[str, str] = {
    "Turkce asistan (varsayilan)": DEFAULT_SYSTEM_PROMPT,
    "English assistant": (
        "You are a helpful assistant running fully offline on a mobile device "
        "inside feast AI. Answer concisely and accurately. Never invent facts. "
        "You are running on a constrained CPU budget, so keep answers tight."
    ),
    "Kisa ve oz (benchmark)": (
        "Cevaplarini en fazla 3 cumlede ver. Giris cumlesi kurma, dogrudan "
        "cevabi yaz. Answer in at most 3 sentences, no preamble."
    ),
    "Ceviri (TR<->EN)": (
        "Sen bir ceviri motorusun. Turkce girdiyi Ingilizceye, Ingilizce girdiyi "
        "Turkceye cevir. Sadece ceviriyi dondur, aciklama ekleme."
    ),
    "Kod asistani": (
        "You are a terse coding assistant on a local edge device. Reply with "
        "working code first and at most two lines of explanation after it."
    ),
    "Bos (system prompt yok)": "",
}


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@dataclass
class Settings:
    """User-editable configuration, persisted as JSON."""

    # --- first run -------------------------------------------------------
    configured: bool = False

    # --- edge device simulation -----------------------------------------
    profile: str = "midrange"
    threads: int = _clamp_cores(6)
    nominal_ram_gb: float = 8.0
    pin_affinity: bool = True
    priority: str = "normal"

    # --- RAM ceiling (working-set, never fatal - see module docstring) ----
    ram_limit_enabled: bool = False
    ram_limit_mb: int = 0      # 0 -> derive from the profile's RAM figure
    ram_budget_pct: int = 70   # share of device RAM the model may hold

    # --- inference -------------------------------------------------------
    context_size: int = 4096
    batch_size: int = 256
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    seed: int = -1
    no_mmap: bool = True  # anonymous memory -> honest RSS accounting

    # --- prompting -------------------------------------------------------
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    keep_history: bool = True
    history_turns: int = 8

    # --- reasoning -------------------------------------------------------
    thinking_mode: str = "auto"   # does the model generate reasoning at all
    show_thinking: bool = False   # is that reasoning shown in the transcript

    # --- models ----------------------------------------------------------
    preferred_quant: str = "Q4_K_M"
    last_model: str = ""

    # --- telemetry / ui --------------------------------------------------
    sample_interval_ms: int = 250
    history_seconds: int = 60
    auto_save_sessions: bool = True

    # --- runtime ---------------------------------------------------------
    server_port: int = 0  # 0 -> pick a free port at launch
    startup_timeout_s: int = 300

    # ------------------------------------------------------------------
    @property
    def sample_points(self) -> int:
        """How many samples the rolling telemetry buffers keep."""
        return max(20, int(self.history_seconds * 1000 / max(50, self.sample_interval_ms)))

    @property
    def device(self) -> DeviceProfile:
        return DEVICE_PROFILES.get(self.profile, DEVICE_PROFILES["custom"])

    @property
    def cpu_budget_cores(self) -> int:
        """Cores the model is actually allowed to use."""
        return _clamp_cores(self.threads)

    @property
    def ram_limit_bytes(self) -> int:
        """Resident-memory ceiling in bytes, or 0 when uncapped.

        An explicit ``ram_limit_mb`` wins.  Otherwise the profile's device RAM
        is scaled by ``ram_budget_pct``, because a phone never gives the whole
        of RAM to one app - the OS, the launcher and the browser want theirs.
        """
        if not self.ram_limit_enabled:
            return 0
        if self.ram_limit_mb > 0:
            return self.ram_limit_mb * 1024 * 1024
        if self.nominal_ram_gb <= 0:
            return 0
        budget = self.nominal_ram_gb * 1024 * max(10, min(100, self.ram_budget_pct)) / 100
        return int(budget) * 1024 * 1024

    @property
    def effective_no_mmap(self) -> bool:
        """``--no-mmap`` is incompatible with a RAM ceiling, so it loses.

        Without mmap the weights live in anonymous memory, so eviction writes
        them to the pagefile and reads them back from there - measured at ~2x
        the disk traffic of the mmap path for identical throughput, plus the
        SSD wear.  With mmap the evicted pages are clean and file-backed, so
        they are simply dropped and re-read from the GGUF, which is exactly
        what a phone streaming from flash does.
        """
        return self.no_mmap and not self.ram_limit_enabled

    def apply_profile(self, key: str) -> None:
        """Adopt a profile's CPU/RAM budget. ``custom`` keeps current values."""
        profile = DEVICE_PROFILES.get(key)
        if profile is None:
            return
        self.profile = key
        if key == "custom":
            return
        self.threads = _clamp_cores(profile.threads)
        self.nominal_ram_gb = float(profile.nominal_ram_gb)
        # Drop any hand-typed ceiling so the new profile's RAM figure applies.
        self.ram_limit_mb = 0

    def mark_custom_if_needed(self) -> None:
        """Switch to ``custom`` once the user deviates from the named profile."""
        profile = DEVICE_PROFILES.get(self.profile)
        if profile is None or profile.key == "custom":
            return
        drifted = (
            self.threads != _clamp_cores(profile.threads)
            or abs(self.nominal_ram_gb - profile.nominal_ram_gb) > 1e-6
            or self.ram_limit_mb > 0
        )
        if drifted:
            self.profile = "custom"

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        settings = cls(**clean)
        settings.threads = _clamp_cores(settings.threads)
        if settings.profile not in DEVICE_PROFILES:
            settings.profile = "custom"
        if settings.priority not in PRIORITY_CHOICES:
            settings.priority = "normal"
        if settings.thinking_mode not in THINKING_MODES:
            settings.thinking_mode = "auto"
        return settings

    # ------------------------------------------------------------------
    def cycle_thinking(self) -> str:
        """Advance the thinking mode: auto -> off -> on -> auto."""
        index = THINKING_MODES.index(self.thinking_mode)
        self.thinking_mode = THINKING_MODES[(index + 1) % len(THINKING_MODES)]
        return self.thinking_mode

    # ------------------------------------------------------------------
    def save(self) -> None:
        ensure_dirs()
        CONFIG_FILE.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls) -> "Settings":
        if not CONFIG_FILE.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            # A corrupt config should never block startup.
            return cls()


@dataclass
class HostInfo:
    """Static facts about the machine, shown on the welcome screen."""

    cpu_name: str = ""
    logical_cores: int = HOST_CORES
    physical_cores: int = 0
    total_ram_gb: float = 0.0
    platform: str = ""
    python: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
