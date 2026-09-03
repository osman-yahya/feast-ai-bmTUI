"""Live process telemetry for the inference server.

Two CPU numbers are reported and they mean different things:

``cpu_host_pct``
    Share of the *whole machine*. 100% means every logical core is busy.

``cpu_device_pct``
    Share of the *simulated device's* CPU budget. This is the number that
    matters: it answers "is the phone's CPU pegged?", and it saturates at
    100% once the model uses all the cores it was allotted.

RAM is measured as RSS and, where the OS allows it, USS (memory unique to
the process - the closest thing to "what this model actually costs you").
USS is expensive to compute on Windows, so it is sampled at a lower rate.

When a RAM ceiling is in force (see :mod:`feast_ai.engine.memcap`) RSS alone
stops being interesting - it just sits pinned at the limit.  The number that
matters then is the **page-fault rate**: how hard the OS is working to keep
the model flowing through a window too small to hold it.  ``disk_read_rate``
is sampled system-wide on purpose, because memory-mapped page-ins never appear
in a process's own I/O counters on Windows.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import psutil


@dataclass
class Sample:
    """One telemetry tick."""

    t: float
    cpu_raw_pct: float = 0.0      # sum across cores, may exceed 100
    cpu_host_pct: float = 0.0     # normalised to all host cores
    cpu_device_pct: float = 0.0   # normalised to the simulated CPU budget
    rss: int = 0
    uss: int = 0
    threads: int = 0
    sys_cpu_pct: float = 0.0
    sys_ram_pct: float = 0.0
    sys_ram_used: int = 0
    faults: int = 0             # cumulative page faults across the tree
    fault_rate: float = 0.0     # faults per second since the previous tick
    disk_read_rate: float = 0.0  # bytes/s, system-wide (catches mmap page-ins)


@dataclass
class Stats:
    """Rolling aggregates over the sampler's lifetime."""

    peak_cpu_device: float = 0.0
    peak_cpu_host: float = 0.0
    peak_rss: int = 0
    peak_uss: int = 0
    sum_cpu_device: float = 0.0
    sum_rss: float = 0.0
    count: int = 0
    busy_count: int = 0  # samples where the model was actually working
    peak_fault_rate: float = 0.0
    total_faults: int = 0
    total_disk_read: int = 0

    @property
    def avg_cpu_device(self) -> float:
        return self.sum_cpu_device / self.count if self.count else 0.0

    @property
    def avg_rss(self) -> float:
        return self.sum_rss / self.count if self.count else 0.0


def _page_faults(proc: psutil.Process, info) -> int:
    """Cumulative page faults for one process, or 0 where unavailable.

    Windows exposes this on ``memory_info()``; Linux keeps it in
    ``/proc/<pid>/stat`` (field 10 minor, field 12 major) and psutil has no
    portable accessor, so it is read directly there.
    """
    count = getattr(info, "num_page_faults", None)
    if count is not None:
        return count
    try:
        with open(f"/proc/{proc.pid}/stat", "rb") as handle:
            fields = handle.read().rsplit(b")", 1)[-1].split()
        return int(fields[7]) + int(fields[9])  # minflt + majflt
    except (OSError, IndexError, ValueError):
        return 0


def _system_disk_read() -> int:
    """System-wide bytes read. Memory-mapped page-ins land here, not in the
    per-process counters, which is why this is measured globally."""
    try:
        counters = psutil.disk_io_counters()
    except (psutil.Error, RuntimeError, OSError):
        return 0
    return counters.read_bytes if counters else 0


class Sampler:
    """Polls a process tree on a fixed cadence and keeps ring buffers."""

    def __init__(
        self,
        budget_cores: int,
        history_points: int = 240,
        uss_every: int = 8,
    ) -> None:
        self.budget_cores = max(1, budget_cores)
        self.host_cores = psutil.cpu_count(logical=True) or 1
        self.total_ram = psutil.virtual_memory().total
        self.history_points = history_points
        self.uss_every = max(1, uss_every)

        self.samples: deque[Sample] = deque(maxlen=history_points)
        self.cpu_series: deque[float] = deque(maxlen=history_points)
        self.ram_series: deque[float] = deque(maxlen=history_points)
        self.tps_series: deque[float] = deque(maxlen=history_points)
        self.fault_series: deque[float] = deque(maxlen=history_points)

        self.stats = Stats()
        self.latest = Sample(t=time.time())

        self._procs: list[psutil.Process] = []
        self._tick = 0
        self._last_uss = 0
        self._started = time.monotonic()
        self._last_faults = 0
        self._last_disk = 0
        self._last_clock = time.monotonic()

    # ------------------------------------------------------------------
    def attach(self, procs: list[psutil.Process]) -> None:
        """Point the sampler at a process tree and prime the CPU counters."""
        self._procs = list(procs)
        for proc in self._procs:
            try:
                proc.cpu_percent(None)  # prime: first call always returns 0.0
            except psutil.Error:
                continue
        psutil.cpu_percent(None)
        self._last_uss = 0
        # Prime the deltas so the first tick reports a rate, not a lifetime sum.
        self._last_faults = 0
        self._last_disk = _system_disk_read()
        self._last_clock = time.monotonic()

    def detach(self) -> None:
        self._procs = []

    def set_budget(self, cores: int) -> None:
        self.budget_cores = max(1, cores)

    def reset(self) -> None:
        self.samples.clear()
        self.cpu_series.clear()
        self.ram_series.clear()
        self.tps_series.clear()
        self.fault_series.clear()
        self.stats = Stats()
        self._started = time.monotonic()

    # ------------------------------------------------------------------
    def sample(self, busy: bool = False, tps: float = 0.0) -> Sample:
        """Take one reading. Call this on a fixed interval."""
        self._tick += 1
        cpu_raw = 0.0
        rss = 0
        threads = 0
        # USS is expensive on Windows, so it is sampled at a lower rate - but
        # always on the very first tick so the reading is never empty.
        want_uss = self._tick == 1 or self._tick % self.uss_every == 0
        uss_total = 0
        uss_ok = True

        faults_now = 0
        alive: list[psutil.Process] = []
        for proc in self._procs:
            try:
                cpu_raw += proc.cpu_percent(None)
                info = proc.memory_info()
                rss += info.rss
                threads += proc.num_threads()
                faults_now += _page_faults(proc, info)
                if want_uss:
                    try:
                        uss_total += proc.memory_full_info().uss
                    except (psutil.Error, AttributeError, OSError):
                        # A partial failure must not wipe the other processes'
                        # contributions - just skip this round.
                        uss_ok = False
                alive.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
                continue
        self._procs = alive

        if want_uss and uss_ok and uss_total:
            self._last_uss = uss_total
        uss = self._last_uss

        # Rates, not lifetime totals: a ceiling pins RSS flat, so the fault
        # rate is the only signal that says how hard the model is streaming.
        now = time.monotonic()
        elapsed = max(1e-6, now - self._last_clock)
        disk_now = _system_disk_read()
        fault_delta = max(0, faults_now - self._last_faults) if self._last_faults else 0
        disk_delta = max(0, disk_now - self._last_disk) if self._last_disk else 0
        self._last_faults, self._last_disk, self._last_clock = faults_now, disk_now, now

        virtual = psutil.virtual_memory()
        sample = Sample(
            t=time.time(),
            cpu_raw_pct=cpu_raw,
            cpu_host_pct=min(100.0, cpu_raw / self.host_cores),
            cpu_device_pct=min(100.0, cpu_raw / self.budget_cores),
            rss=rss,
            uss=uss,
            threads=threads,
            sys_cpu_pct=psutil.cpu_percent(None),
            sys_ram_pct=virtual.percent,
            sys_ram_used=virtual.used,
            faults=faults_now,
            fault_rate=fault_delta / elapsed,
            disk_read_rate=disk_delta / elapsed,
        )

        self.latest = sample
        self.samples.append(sample)
        self.cpu_series.append(sample.cpu_device_pct)
        self.ram_series.append(sample.rss / (1024 * 1024))
        self.tps_series.append(tps)
        self.fault_series.append(sample.fault_rate)

        st = self.stats
        st.count += 1
        st.sum_cpu_device += sample.cpu_device_pct
        st.sum_rss += sample.rss
        st.peak_cpu_device = max(st.peak_cpu_device, sample.cpu_device_pct)
        st.peak_cpu_host = max(st.peak_cpu_host, sample.cpu_host_pct)
        st.peak_rss = max(st.peak_rss, sample.rss)
        st.peak_uss = max(st.peak_uss, sample.uss)
        st.peak_fault_rate = max(st.peak_fault_rate, sample.fault_rate)
        st.total_faults += fault_delta
        st.total_disk_read += disk_delta
        if busy:
            st.busy_count += 1
        return sample

    # ------------------------------------------------------------------
    def window(self, seconds: float) -> list[Sample]:
        cutoff = time.time() - seconds
        return [s for s in self.samples if s.t >= cutoff]

    def busy_stats(self, since: float) -> tuple[float, float, int]:
        """``(avg_cpu_device, peak_cpu_device, peak_rss)`` since a timestamp."""
        window = [s for s in self.samples if s.t >= since]
        if not window:
            latest = self.latest
            return latest.cpu_device_pct, latest.cpu_device_pct, latest.rss
        cpus = [s.cpu_device_pct for s in window]
        return sum(cpus) / len(cpus), max(cpus), max(s.rss for s in window)

    def paging_stats(self, since: float) -> tuple[int, int, float]:
        """``(faults, disk_read_bytes, peak_fault_rate)`` since a timestamp.

        Counts are reconstructed from the per-tick rates so a turn only ever
        gets charged for the paging that happened inside its own window.
        """
        window = [s for s in self.samples if s.t >= since]
        if not window:
            return 0, 0, 0.0
        interval = max(1e-6, self.history_seconds_per_sample)
        faults = int(sum(s.fault_rate for s in window) * interval)
        disk = int(sum(s.disk_read_rate for s in window) * interval)
        return faults, disk, max(s.fault_rate for s in window)

    @property
    def history_seconds_per_sample(self) -> float:
        """Wall-clock seconds each ring-buffer slot represents."""
        if len(self.samples) < 2:
            return 0.25
        span = self.samples[-1].t - self.samples[0].t
        return span / (len(self.samples) - 1) if span > 0 else 0.25


@dataclass
class HostSnapshot:
    """Static host facts for the welcome / summary screens."""

    cpu_name: str
    logical_cores: int
    physical_cores: int
    total_ram: int
    freq_mhz: float = 0.0
    extras: dict = field(default_factory=dict)


def host_snapshot() -> HostSnapshot:
    import platform

    name = platform.processor() or platform.machine() or "unknown CPU"
    freq = 0.0
    try:
        current = psutil.cpu_freq()
        freq = current.max or current.current or 0.0
    except (psutil.Error, AttributeError, OSError, NotImplementedError):
        pass
    return HostSnapshot(
        cpu_name=name.strip(),
        logical_cores=psutil.cpu_count(logical=True) or 1,
        physical_cores=psutil.cpu_count(logical=False) or 0,
        total_ram=psutil.virtual_memory().total,
        freq_mhz=freq,
        extras={"platform": platform.platform(terse=True)},
    )
