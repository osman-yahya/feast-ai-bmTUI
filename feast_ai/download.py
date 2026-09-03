"""Resumable HTTP downloads with progress reporting.

Used for both GGUF weights and the llama.cpp runtime archive.  Deliberately
built on ``httpx`` rather than ``huggingface_hub`` so progress lands in the
TUI instead of a tqdm bar fighting with the terminal.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import httpx

USER_AGENT = "feast-ai/1.0"
CHUNK = 1024 * 512


@dataclass
class Progress:
    """A single progress tick."""

    downloaded: int
    total: int | None
    speed_bps: float
    elapsed: float

    @property
    def fraction(self) -> float:
        if not self.total:
            return 0.0
        return min(1.0, self.downloaded / self.total)

    @property
    def eta_s(self) -> float | None:
        if not self.total or self.speed_bps <= 0:
            return None
        return max(0.0, (self.total - self.downloaded) / self.speed_bps)


ProgressCallback = Callable[[Progress], None | Awaitable[None]]


class DownloadError(RuntimeError):
    pass


async def download(
    url: str,
    dest: Path,
    on_progress: ProgressCallback | None = None,
    cancel: asyncio.Event | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download ``url`` to ``dest``, resuming a partial ``.part`` if present.

    Raises :class:`DownloadError` on HTTP failure and
    :class:`asyncio.CancelledError` when ``cancel`` is set.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0

    headers = {"User-Agent": USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    limits = httpx.Limits(max_connections=4)
    started = time.monotonic()
    downloaded = existing
    last_emit = 0.0

    async def emit(force: bool = False) -> None:
        nonlocal last_emit
        now = time.monotonic()
        if not force and now - last_emit < 0.15:
            return
        last_emit = now
        if on_progress is None:
            return
        elapsed = max(1e-6, now - started)
        speed = (downloaded - existing) / elapsed
        result = on_progress(Progress(downloaded, total, speed, elapsed))
        if asyncio.iscoroutine(result):
            await result

    async with httpx.AsyncClient(
        follow_redirects=True, timeout=timeout, limits=limits
    ) as client:
        async with client.stream("GET", url, headers=headers) as response:
            if response.status_code == 416:  # already complete
                part.replace(dest)
                return dest
            if response.status_code == 200 and existing:
                # Server ignored our Range header - start over.
                existing = 0
                downloaded = 0
                part.unlink(missing_ok=True)
            if response.status_code not in (200, 206):
                raise DownloadError(
                    f"HTTP {response.status_code} for {url}"
                )

            content_length = response.headers.get("Content-Length")
            total = int(content_length) + existing if content_length else None
            if response.status_code == 206:
                content_range = response.headers.get("Content-Range", "")
                if "/" in content_range:
                    try:
                        total = int(content_range.rsplit("/", 1)[1])
                    except ValueError:
                        pass

            mode = "ab" if existing else "wb"
            with part.open(mode) as handle:
                async for chunk in response.aiter_bytes(CHUNK):
                    if cancel is not None and cancel.is_set():
                        handle.flush()
                        raise asyncio.CancelledError()
                    handle.write(chunk)
                    downloaded += len(chunk)
                    await emit()

    await emit(force=True)
    dest.unlink(missing_ok=True)
    part.replace(dest)
    return dest


async def download_many(
    items: list[tuple[str, str]],
    root: Path,
    total_bytes: int = 0,
    on_progress: ProgressCallback | None = None,
    cancel: asyncio.Event | None = None,
    timeout: float = 60.0,
) -> Path:
    """Fetch every ``(url, relative_path)`` in ``items`` into ``root``.

    ONNX Runtime GenAI models are directories - weights, tokenizer and
    ``genai_config.json`` all have to arrive before the runtime can load
    anything - so they cannot use the single-file path.

    Progress is reported across the whole set rather than per file, so the UI
    shows one bar that fills once.  ``total_bytes`` seeds it with the catalog's
    figure; without it the bar only knows the file currently in flight.
    """
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    done_before = 0

    for url, relative in items:
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        dest = root / relative
        if dest.exists():
            # Already fetched on an earlier run; count it and move on.
            done_before += dest.stat().st_size
            if on_progress is not None:
                result = on_progress(
                    Progress(done_before, total_bytes or None, 0.0,
                             time.monotonic() - started)
                )
                if inspect.isawaitable(result):
                    await result
            continue

        def relay(progress: Progress, _base: int = done_before) -> None:
            if on_progress is None:
                return
            elapsed = time.monotonic() - started
            combined = _base + progress.downloaded
            speed = combined / elapsed if elapsed > 0 else 0.0
            return on_progress(
                Progress(combined, total_bytes or None, speed, elapsed)
            )

        await download(url, dest, on_progress=relay, cancel=cancel, timeout=timeout)
        done_before += dest.stat().st_size if dest.exists() else 0

    return root


def partial_bytes(dest: Path) -> int:
    """Bytes already fetched for an interrupted download of ``dest``."""
    part = dest.with_suffix(dest.suffix + ".part")
    return part.stat().st_size if part.exists() else 0


def clear_partial(dest: Path) -> None:
    dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
