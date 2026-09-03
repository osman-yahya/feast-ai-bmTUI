"""Manage a ``llama-server`` child process under simulated edge constraints.

The CPU budget is applied twice, on purpose:

* ``--threads N`` tells llama.cpp how many worker threads to create;
* ``cpu_affinity([...])`` tells the OS scheduler the process may only run on
  N logical cores.

The second one is what makes the simulation honest.  Threads alone still let
the OS migrate work across every core on the machine, and llama.cpp spawns
extra helper threads for loading and batching.  Pinning gives a hard ceiling,
so a "2-core phone" really cannot consume more than 2 cores' worth of CPU.

RAM is never limited - see the note in :mod:`feast_ai.config`.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import psutil

from ..config import Settings
from ..paths import LOG_DIR
from .memcap import CapResult, advisory_floor_mb, apply_cap


class EngineError(RuntimeError):
    pass


# --------------------------------------------------------------------------
def free_port(preferred: int = 0) -> int:
    """An available localhost port, honouring ``preferred`` when possible."""
    if preferred:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def select_cores(count: int) -> list[int]:
    """Choose ``count`` logical cores to pin to.

    Prefers even indices first: on an SMT machine those are distinct physical
    cores, which is what a phone's big cluster actually looks like.  Falls
    back to the odd (sibling) indices once the physicals run out.
    """
    total = os.cpu_count() or 1
    count = max(1, min(count, total))
    evens = list(range(0, total, 2))
    odds = list(range(1, total, 2))
    return (evens + odds)[:count]


def _priority_value(name: str):
    if os.name == "nt":
        return {
            "normal": psutil.NORMAL_PRIORITY_CLASS,
            "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
            "idle": psutil.IDLE_PRIORITY_CLASS,
        }.get(name, psutil.NORMAL_PRIORITY_CLASS)
    return {"normal": 0, "below_normal": 10, "idle": 19}.get(name, 0)


# --------------------------------------------------------------------------
@dataclass
class LoadReport:
    """What actually happened when the model was brought up."""

    load_seconds: float = 0.0
    rss_after_load: int = 0
    pinned_cores: list[int] = field(default_factory=list)
    threads: int = 0
    port: int = 0
    command: list[str] = field(default_factory=list)
    log_path: Path | None = None
    cap: CapResult = field(default_factory=CapResult)
    #: Set when the requested ceiling is tight enough that llama.cpp's own
    #: buffers start paging, so the run measures thrash rather than streaming.
    cap_warning: str = ""


def _model_bytes(path: Path) -> int:
    """Size of a model on disk, whether it is one GGUF or an ONNX directory."""
    try:
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return path.stat().st_size
    except OSError:
        return 0


class BaseServer:
    """Process supervision shared by every backend.

    Subclasses supply :meth:`_commands` and a display :attr:`engine_name`;
    everything that makes the simulation honest - affinity pinning, the RAM
    ceiling, priority, health polling, teardown - lives here so a new backend
    cannot accidentally opt out of it.
    """

    engine_name = "engine"
    log_prefix = "engine"

    def __init__(self, server_bin: Path, settings: Settings) -> None:
        self.server_bin = Path(server_bin)
        self.settings = settings
        self.process: subprocess.Popen | None = None
        self.proc: psutil.Process | None = None
        self.port: int = 0
        self.model_path: Path | None = None
        self.model_alias: str = "feast"
        self.report = LoadReport()
        self._log_handle = None

    # ------------------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    # ------------------------------------------------------------------
    def _commands(self, model_path: Path, port: int) -> list[list[str]]:
        """Command lines to try, in order. Later entries are fallbacks."""
        raise NotImplementedError

    def _check_available(self) -> None:
        """Raise :class:`EngineError` if the backend cannot run at all."""
        if not self.server_bin.exists():
            raise EngineError(f"{self.engine_name} not found: {self.server_bin}")

    # ------------------------------------------------------------------
    async def start(
        self,
        model_path: Path,
        alias: str = "feast",
        on_status: "callable | None" = None,
    ) -> LoadReport:
        """Launch the server and block until ``/health`` reports ready."""

        def status(message: str) -> None:
            if on_status is not None:
                on_status(message)

        if not Path(model_path).exists():
            raise EngineError(f"Model not found: {model_path}")
        self._check_available()

        await self.stop()

        self.model_path = Path(model_path)
        self.model_alias = alias
        self.port = free_port(self.settings.server_port)

        candidates = self._commands(self.model_path, self.port)
        last_error: Exception | None = None
        for index, cmd in enumerate(candidates):
            try:
                return await self._spawn_and_wait(cmd, status)
            except EngineError as exc:
                last_error = exc
                await self.stop()
                if index + 1 < len(candidates):
                    status("Retrying without optional flags...")
        raise last_error or EngineError(f"{self.engine_name} failed to start")

    # ------------------------------------------------------------------
    async def _spawn_and_wait(self, cmd: list[str], status) -> LoadReport:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{self.log_prefix}-{int(time.time())}.log"
        self._log_handle = log_path.open("w", encoding="utf-8", errors="replace")
        self._log_handle.write(" ".join(cmd) + "\n\n")
        self._log_handle.flush()

        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        started = time.monotonic()
        status(f"Starting {self.engine_name} on {self.settings.cpu_budget_cores} core(s)...")
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                cwd=str(self.server_bin.parent),
            )
        except OSError as exc:
            raise EngineError(f"Could not launch {self.engine_name}: {exc}") from exc

        try:
            self.proc = psutil.Process(self.process.pid)
        except psutil.NoSuchProcess as exc:
            raise EngineError(f"{self.engine_name} exited immediately") from exc

        pinned = self._apply_constraints()
        cap = self._apply_ram_cap()
        warning = self._cap_warning(self.model_path) if self.model_path else ""
        if cap.applied:
            status(f"RAM ceiling {cap.limit_mb} MB ({cap.method}); streaming weights...")
        elif self.settings.ram_limit_enabled:
            status(f"RAM ceiling unavailable: {cap.detail}")
        else:
            status("Loading weights into RAM...")
        await self._await_health(started, log_path)

        rss = 0
        try:
            rss = self.proc.memory_info().rss
        except (psutil.Error, AttributeError):
            pass

        self.report = LoadReport(
            load_seconds=time.monotonic() - started,
            rss_after_load=rss,
            pinned_cores=pinned,
            threads=self.settings.cpu_budget_cores,
            port=self.port,
            command=cmd,
            log_path=log_path,
            cap=cap,
            cap_warning=warning,
        )
        status(f"Ready in {self.report.load_seconds:.1f}s")
        return self.report

    # ------------------------------------------------------------------
    def _apply_constraints(self) -> list[int]:
        """Pin cores and set scheduling priority. Best effort, never fatal."""
        pinned: list[int] = []
        if self.proc is None:
            return pinned

        if self.settings.pin_affinity:
            cores = select_cores(self.settings.cpu_budget_cores)
            try:
                self.proc.cpu_affinity(cores)
                pinned = cores
            except (psutil.Error, AttributeError, OSError, ValueError):
                pinned = []

        if self.settings.priority != "normal":
            try:
                self.proc.nice(_priority_value(self.settings.priority))
            except (psutil.Error, OSError, PermissionError):
                pass
        return pinned

    # ------------------------------------------------------------------
    def _apply_ram_cap(self) -> CapResult:
        """Impose the resident-memory ceiling, before the weights are loaded.

        Capping now rather than after loading is the point: the model must
        never be fully resident, exactly as on a device that cannot hold it.
        """
        limit = self.settings.ram_limit_bytes
        if not limit or self.process is None:
            return CapResult(detail="no limit requested")
        return apply_cap(self.process.pid, limit)

    def _cap_warning(self, model_path: Path) -> str:
        limit = self.settings.ram_limit_bytes
        if not limit:
            return ""
        model_bytes = _model_bytes(model_path)
        if not model_bytes:
            return ""
        floor = advisory_floor_mb(model_bytes, self.settings.context_size)
        if limit // (1024 * 1024) >= floor:
            return ""
        return (
            f"RAM ceiling {limit // (1024 * 1024)} MB is below the ~{floor} MB "
            "this model needs for its KV cache and compute buffers. Expect "
            "thrashing rather than clean weight streaming."
        )

    def reapply_constraints(self) -> list[int]:
        """Re-pin after a settings change without restarting the model."""
        return self._apply_constraints()

    # ------------------------------------------------------------------
    async def _await_health(self, started: float, log_path: Path) -> None:
        deadline = started + self.settings.startup_timeout_s
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if self.process is None or self.process.poll() is not None:
                    raise EngineError(
                        f"{self.engine_name} exited during startup.\n"
                        + _tail(log_path, 18)
                    )
                try:
                    response = await client.get(f"{self.base_url}/health")
                    if response.status_code == 200:
                        return
                except (httpx.HTTPError, OSError):
                    pass
                await asyncio.sleep(0.35)
        raise EngineError(
            f"{self.engine_name} did not become healthy within "
            f"{self.settings.startup_timeout_s}s.\n" + _tail(log_path, 18)
        )

    # ------------------------------------------------------------------
    async def stop(self) -> None:
        """Terminate the server and every thread it owns."""
        process, self.process = self.process, None
        proc, self.proc = self.proc, None

        if process is not None and process.poll() is None:
            try:
                if proc is not None:
                    for child in proc.children(recursive=True):
                        child.terminate()
                process.terminate()
                for _ in range(40):
                    if process.poll() is not None:
                        break
                    await asyncio.sleep(0.05)
                if process.poll() is None:
                    process.kill()
            except (psutil.Error, OSError):
                pass

        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None

    # ------------------------------------------------------------------
    def snapshot_processes(self) -> list[psutil.Process]:
        """The server plus its children, for telemetry sampling."""
        if self.proc is None:
            return []
        procs = [self.proc]
        try:
            procs += self.proc.children(recursive=True)
        except psutil.Error:
            pass
        return procs

    def tail_log(self, lines: int = 20) -> str:
        return _tail(self.report.log_path, lines) if self.report.log_path else ""


class LlamaServer(BaseServer):
    """Owns one ``llama-server`` process and the psutil handle on it."""

    engine_name = "llama.cpp"
    log_prefix = "llama-server"

    #: Flags dropped and retried without if the server refuses to start.
    OPTIONAL_FLAGS = ("--no-webui", "--jinja")

    def _commands(self, model_path: Path, port: int) -> list[list[str]]:
        s = self.settings
        cmd = [
            str(self.server_bin),
            "--model", str(model_path),
            "--alias", self.model_alias,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--ctx-size", str(s.context_size),
            "--threads", str(s.cpu_budget_cores),
            "--threads-batch", str(s.cpu_budget_cores),
            "--batch-size", str(s.batch_size),
            "--ubatch-size", str(min(s.batch_size, 256)),
            "--parallel", "1",
            "--gpu-layers", "0",  # CPU inference engine, explicitly
        ]
        if s.effective_no_mmap:
            cmd.append("--no-mmap")
        if s.seed is not None and s.seed >= 0:
            cmd += ["--seed", str(s.seed)]
        return [cmd + list(self.OPTIONAL_FLAGS), cmd]


def _tail(path: Path | None, lines: int) -> str:
    if path is None or not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def python_hint() -> str:
    return f"{sys.version.split()[0]} on {sys.platform}"
