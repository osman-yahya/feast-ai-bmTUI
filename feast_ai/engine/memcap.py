"""Impose a *soft* RAM ceiling on the inference process.

Why this is not the RAM limit we originally refused to build
------------------------------------------------------------
The obvious way to cap memory - a commit limit (a Windows Job Object's
``ProcessMemoryLimit``, or cgroup ``memory.max``) - kills the process the
instant it asks for one byte too many.  That turns every oversized model into
a crash instead of a measurement, which is why feast AI shipped without a RAM
cap at first.

This module does something different.  It caps the **working set**: how much
of the process's memory the OS keeps resident in physical RAM.  Going over the
cap does not fail an allocation and does not kill anything - the OS simply
evicts pages and faults them back in on the next access.  That is exactly what
a phone does when a model does not fit, so it is the honest way to simulate
one.

Measured on this project (Qwen3 1.7B Q4_K_M, 1056 MB, 4 cores, ctx 2048):

    uncapped        RSS 1995 MB    25.5 tok/s      1.1 K page faults
    cap 512 MB      RSS  512 MB     2.8 tok/s     16.2 M page faults
    cap 256 MB      RSS  256 MB     2.6 tok/s     17.0 M page faults

The process survives, and the generated text is byte-identical across all
three - only the speed changes.  The ceiling holds to within about 1.5% (a
512 MB cap settles at ~520 MB RSS); Windows cannot trim locked and
non-pageable pages, so a small fixed overshoot is expected rather than a bug.

Fidelity caveat
---------------
16.2 M faults x 4 KB is 65 GB of page-ins for 64 tokens, i.e. the full 1 GB of
weights really is re-faulted once per token, exactly like a memory-starved
phone.  But only ~600 MB of that reached the disk: the host's spare RAM holds
the evicted (clean, file-backed) pages on the standby list, so most faults are
soft.  So this models a phone's *paging behaviour* faithfully and its *storage
latency* optimistically.  :func:`fault_report` exposes the hard-fault ratio so
a run can state how close to the real thing it was.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

MB = 1024 * 1024

#: Below this the cap starts evicting llama.cpp's scratch and KV buffers rather
#: than model weights, which measures thrash rather than streaming.
MIN_USEFUL_MB = 192


@dataclass
class CapResult:
    """Outcome of trying to impose the ceiling. Never raises; always reports."""

    applied: bool = False
    limit_bytes: int = 0
    method: str = "none"
    detail: str = ""

    @property
    def limit_mb(self) -> int:
        return self.limit_bytes // MB


def advisory_floor_mb(model_bytes: int, context_size: int) -> int:
    """Smallest cap that still measures weight streaming rather than thrash.

    llama.cpp needs room for the KV cache and compute buffers on top of the
    weights it is streaming.  Squeeze below that and every buffer starts
    faulting too, which is a different (and much less interesting) experiment.
    """
    kv_mb = max(64, context_size // 16)
    return max(MIN_USEFUL_MB, int(model_bytes / MB * 0.15) + kv_mb)


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------
def _apply_windows(pid: int, limit_bytes: int) -> CapResult:
    import ctypes
    import ctypes.wintypes as wt

    QUOTA_LIMITS_HARDWS_MIN_DISABLE = 0x00000002
    QUOTA_LIMITS_HARDWS_MAX_ENABLE = 0x00000004
    PROCESS_SET_QUOTA = 0x0100
    PROCESS_QUERY_INFORMATION = 0x0400

    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wt.HANDLE
        k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        k32.SetProcessWorkingSetSizeEx.argtypes = [
            wt.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wt.DWORD,
        ]
        k32.CloseHandle.argtypes = [wt.HANDLE]
    except (OSError, AttributeError) as exc:
        return CapResult(detail=f"kernel32 unavailable: {exc}")

    handle = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION, False, pid)
    if not handle:
        return CapResult(detail=f"OpenProcess failed (error {ctypes.get_last_error()})")
    try:
        # The minimum stays soft on purpose: only the ceiling is enforced, so
        # the OS is still free to shrink the process further under pressure.
        flags = QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE
        minimum = min(16 * MB, limit_bytes // 2)
        if not k32.SetProcessWorkingSetSizeEx(handle, minimum, limit_bytes, flags):
            code = ctypes.get_last_error()
            return CapResult(detail=f"SetProcessWorkingSetSizeEx failed (error {code})")
        return CapResult(
            applied=True,
            limit_bytes=limit_bytes,
            method="working-set",
            detail=f"hard working-set ceiling {limit_bytes // MB} MB",
        )
    finally:
        k32.CloseHandle(handle)


# --------------------------------------------------------------------------
# Linux
# --------------------------------------------------------------------------
def _apply_linux(pid: int, limit_bytes: int, group_name: str = "feast-ai") -> CapResult:
    """cgroup v2 ``memory.high``: throttle and reclaim, never OOM-kill.

    ``memory.high`` is the right knob precisely because it is *not*
    ``memory.max`` - crossing it triggers reclaim and throttling rather than
    the OOM killer, matching the Windows behaviour above.

    Each server gets its own group.  With one shared group, the intent
    router's small slice would overwrite the model's ``memory.high`` - the
    last writer wins - and squeeze the model far below its budget.
    """
    base = Path("/sys/fs/cgroup")
    if not (base / "cgroup.controllers").exists():
        return CapResult(detail="cgroup v2 not mounted")
    group = base / group_name
    try:
        group.mkdir(exist_ok=True)
        (group / "memory.high").write_text(str(limit_bytes))
        (group / "cgroup.procs").write_text(str(pid))
    except (OSError, PermissionError) as exc:
        return CapResult(detail=f"cgroup write denied ({exc.__class__.__name__}); "
                                "needs delegated cgroup or root")
    return CapResult(
        applied=True,
        limit_bytes=limit_bytes,
        method="cgroup memory.high",
        detail=f"memory.high {limit_bytes // MB} MB",
    )


# --------------------------------------------------------------------------
def apply_cap(pid: int, limit_bytes: int, group: str = "feast-ai") -> CapResult:
    """Cap the resident memory of ``pid``. Best effort; reports what happened.

    ``group`` names the Linux cgroup; Windows caps are per-process already.
    """
    if limit_bytes <= 0:
        return CapResult(detail="no limit requested")
    if sys.platform == "win32":
        return _apply_windows(pid, limit_bytes)
    if sys.platform.startswith("linux"):
        return _apply_linux(pid, limit_bytes, group)
    return CapResult(detail=f"no working-set cap available on {sys.platform}")


def release_cap(pid: int) -> None:
    """Drop the ceiling again (Windows only; Linux dies with the cgroup)."""
    if sys.platform != "win32":
        return
    import ctypes
    import ctypes.wintypes as wt

    QUOTA_LIMITS_HARDWS_MIN_DISABLE = 0x00000002
    QUOTA_LIMITS_HARDWS_MAX_DISABLE = 0x00000008
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wt.HANDLE
        k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        # Each WinDLL() is a fresh object, so the argtypes set in
        # _apply_windows do not carry over. Without them ctypes passes ints as
        # C int and (SIZE_T)-1 overflows.
        k32.SetProcessWorkingSetSizeEx.argtypes = [
            wt.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wt.DWORD,
        ]
        k32.CloseHandle.argtypes = [wt.HANDLE]
        handle = k32.OpenProcess(0x0100 | 0x0400, False, pid)
        if not handle:
            return
        try:
            size = ctypes.c_size_t(-1).value  # (SIZE_T)-1 -> let the OS decide
            k32.SetProcessWorkingSetSizeEx(
                handle, size, size,
                QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE,
            )
        finally:
            k32.CloseHandle(handle)
    except (OSError, AttributeError, ValueError):
        pass


# --------------------------------------------------------------------------
@dataclass
class FaultReport:
    """Paging cost of one measured window."""

    faults: int = 0
    disk_read_bytes: int = 0
    seconds: float = 0.0
    tokens: int = 0

    @property
    def faults_per_second(self) -> float:
        return self.faults / self.seconds if self.seconds else 0.0

    @property
    def faults_per_token(self) -> float:
        return self.faults / self.tokens if self.tokens else 0.0

    @property
    def paged_bytes(self) -> int:
        """Bytes faulted back in, assuming 4 KB pages."""
        return self.faults * 4096

    @property
    def hard_fault_ratio(self) -> float:
        """Share of faulted pages that actually came off the disk.

        1.0 means every re-read hit storage, like a phone with no spare RAM.
        Lower means the host's page cache absorbed them, so the run was
        faster than the device it is standing in for.
        """
        paged = self.paged_bytes
        if not paged:
            return 0.0
        return min(1.0, self.disk_read_bytes / paged)

    @property
    def verdict(self) -> str:
        if not self.faults:
            return "resident - model fits inside the RAM budget"
        ratio = self.hard_fault_ratio
        if ratio >= 0.5:
            return "streaming from disk - storage-bound, like a real device"
        if ratio >= 0.05:
            return "streaming, partly absorbed by the host page cache"
        return "streaming, but served from host RAM - optimistic vs real flash"


def fault_report(faults: int, disk_read_bytes: int, seconds: float, tokens: int) -> FaultReport:
    return FaultReport(faults, disk_read_bytes, seconds, tokens)
