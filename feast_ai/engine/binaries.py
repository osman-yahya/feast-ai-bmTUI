"""Bootstrap the llama.cpp CPU runtime.

feast AI does not compile anything.  It fetches the official prebuilt
``llama-server`` binary for the host platform from the ggml-org/llama.cpp
GitHub releases and caches it under ``~/.feast-ai/bin/llama.cpp``.

Deliberately the **CPU** build: the whole point of the project is to measure
what an edge device does without a GPU.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import stat
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..download import ProgressCallback, download
from ..paths import RUNTIME_DIR

RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
USER_AGENT = "feast-ai/1.0"
VERSION_FILE = RUNTIME_DIR / "VERSION.json"

SERVER_NAMES = ("llama-server.exe", "llama-server", "server.exe", "server")


class RuntimeError_(RuntimeError):
    pass


@dataclass
class RuntimeInfo:
    """Where the runtime is and which build it is."""

    server: Path | None
    build: str | None
    source: str  # "bundled" | "path" | "missing"

    @property
    def available(self) -> bool:
        return self.server is not None and self.server.exists()


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def find_server() -> Path | None:
    """Locate ``llama-server``: cached build first, then ``PATH``."""
    if RUNTIME_DIR.exists():
        for name in SERVER_NAMES:
            direct = RUNTIME_DIR / name
            if direct.exists():
                return direct
        for name in SERVER_NAMES:
            for found in RUNTIME_DIR.rglob(name):
                if found.is_file():
                    return found

    env_override = os.environ.get("FEAST_AI_LLAMA_SERVER")
    if env_override and Path(env_override).exists():
        return Path(env_override)

    for name in ("llama-server", "server"):
        which = shutil.which(name)
        if which:
            return Path(which)
    return None


def _cached_build() -> str | None:
    if VERSION_FILE.exists():
        try:
            return json.loads(VERSION_FILE.read_text(encoding="utf-8")).get("build")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def runtime_info() -> RuntimeInfo:
    server = find_server()
    if server is None:
        return RuntimeInfo(None, None, "missing")
    try:
        bundled = server.is_relative_to(RUNTIME_DIR)
    except ValueError:
        bundled = False
    return RuntimeInfo(
        server=server,
        build=_cached_build() if bundled else "system",
        source="bundled" if bundled else "path",
    )


# --------------------------------------------------------------------------
# Asset selection
# --------------------------------------------------------------------------
def _asset_pattern() -> tuple[str, ...]:
    """Substrings that identify the right release asset for this host."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")

    if system == "windows":
        return ("bin-win-cpu-arm64",) if arm else ("bin-win-cpu-x64",)
    if system == "darwin":
        return ("bin-macos-arm64",) if arm else ("bin-macos-x64",)
    if system == "linux":
        return ("bin-ubuntu-arm64",) if arm else ("bin-ubuntu-x64", "bin-ubuntu-vulkan-x64")
    raise RuntimeError_(f"Unsupported platform: {system}/{machine}")


def _fetch_release() -> dict:
    request = urllib.request.Request(
        RELEASES_API, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError_(
            "Could not reach the llama.cpp releases API. Check your network, "
            "or set FEAST_AI_LLAMA_SERVER to an existing llama-server binary."
        ) from exc


def _pick_asset(release: dict) -> tuple[str, str, int]:
    patterns = _asset_pattern()
    assets = release.get("assets", [])
    for pattern in patterns:
        for asset in assets:
            name = asset.get("name", "")
            if pattern in name and name.endswith((".zip", ".tar.gz")):
                return name, asset["browser_download_url"], asset.get("size", 0)
    available = ", ".join(a.get("name", "?") for a in assets[:12])
    raise RuntimeError_(
        f"No llama.cpp asset matching {patterns} in release "
        f"{release.get('tag_name')}. Saw: {available}"
    )


# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------
def _extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip" or zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(target)


def _make_executable(root: Path) -> None:
    if os.name == "nt":
        return
    for path in root.rglob("*"):
        if path.is_file() and (path.suffix in ("", ".so", ".dylib") or "llama" in path.name):
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


async def ensure_runtime(
    on_progress: ProgressCallback | None = None,
    on_status: "callable | None" = None,
    force: bool = False,
) -> RuntimeInfo:
    """Return a usable runtime, downloading the CPU build if necessary."""

    def status(message: str) -> None:
        if on_status is not None:
            on_status(message)

    if not force:
        info = runtime_info()
        if info.available:
            return info

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    status("Querying llama.cpp releases...")
    release = await asyncio.to_thread(_fetch_release)
    tag = release.get("tag_name", "unknown")
    name, url, size = _pick_asset(release)

    status(f"Downloading llama.cpp {tag} ({name})")
    archive = RUNTIME_DIR / name
    await download(url, archive, on_progress=on_progress)

    status("Extracting runtime...")
    await asyncio.to_thread(_extract, archive, RUNTIME_DIR)
    archive.unlink(missing_ok=True)
    await asyncio.to_thread(_make_executable, RUNTIME_DIR)

    VERSION_FILE.write_text(
        json.dumps({"build": tag, "asset": name, "size": size}, indent=2),
        encoding="utf-8",
    )

    info = runtime_info()
    if not info.available:
        raise RuntimeError_(
            f"Extracted {name} but no llama-server binary was found inside it."
        )
    status(f"llama.cpp {tag} ready.")
    return info
