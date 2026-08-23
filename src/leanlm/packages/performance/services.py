"""Resource sampling and metrics assembly.

``psutil`` is used when present. When it is not, the sampler falls back to
``/proc`` and ``resource`` -- because the target machine is a commodity laptop
where a missing wheel must not cost us the memory numbers the challenge scores
on.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from pathlib import Path

from ...contracts.dto import MetricsSnapshot, ResourceObservation
from ...shared.clock import iso_now
from .policies import PerformancePolicy

try:  # pragma: no cover - environment dependent
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def _windows_memory() -> tuple[float, float]:
    """(total_mb, available_mb) on Windows, via the Win32 API through ctypes.

    Written because the /proc fallbacks are Linux-only, and a laptop running
    Windows is not an edge case for this product -- it is the target machine.
    Without this the sampler reported 0 MB, which is not merely a blank field:
    the scoring module read it as an out-of-memory kill and disqualified a
    perfectly healthy run.
    """
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 0.0, 0.0
    return (status.ullTotalPhys / 1048576.0, status.ullAvailPhys / 1048576.0)


def _windows_rss() -> float:
    """Working set of this process, in MB."""
    import ctypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb)
    return (counters.WorkingSetSize / 1048576.0) if ok else 0.0


def _read_meminfo() -> tuple[float, float]:
    """(total_mb, available_mb): /proc on Linux, Win32 on Windows."""
    if sys.platform == "win32":
        try:
            return _windows_memory()
        except Exception:
            return 0.0, 0.0
    total = available = 0.0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total = float(line.split()[1]) / 1024.0
            elif line.startswith("MemAvailable:"):
                available = float(line.split()[1]) / 1024.0
            if total and available:
                break
    except OSError:
        pass
    return total, available


def _read_self_rss() -> float:
    if sys.platform == "win32":
        try:
            return _windows_rss()
        except Exception:
            return 0.0
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kB, macOS reports bytes.
        return maxrss / 1024.0 if maxrss > 1 << 20 else maxrss / 1024.0
    except Exception:
        return 0.0


def _read_temperature() -> tuple[float | None, str]:
    """Highest CPU-ish temperature available, with its source named.

    The source is recorded because 'no thermal sensor' and '41 degrees' are
    different facts, and a thermal score computed from an unavailable sensor
    would be a fabrication.
    """
    if psutil is not None and hasattr(psutil, "sensors_temperatures"):
        try:
            readings = psutil.sensors_temperatures() or {}
            candidates = []
            for chip, entries in readings.items():
                for entry in entries:
                    if entry.current and entry.current > 0:
                        candidates.append((entry.current, f"psutil:{chip}"))
            if candidates:
                value, source = max(candidates)
                return round(float(value), 2), source
        except Exception:
            pass
    zones = sorted(Path("/sys/class/thermal").glob("thermal_zone*")) \
        if Path("/sys/class/thermal").is_dir() else []
    best: tuple[float, str] | None = None
    for zone in zones:
        try:
            zone_type = (zone / "type").read_text().strip()
            raw = float((zone / "temp").read_text().strip())
        except (OSError, ValueError):
            continue
        celsius = raw / 1000.0 if raw > 200 else raw
        if 0 < celsius < 130:
            if best is None or celsius > best[0]:
                best = (celsius, f"sysfs:{zone_type}")
    if best:
        return round(best[0], 2), best[1]
    return None, "unavailable"


class ResourceSampler:
    """Point-in-time observation plus an optional background peak tracker."""

    def __init__(self, policy: PerformancePolicy | None = None) -> None:
        self.policy = policy or PerformancePolicy()
        self._peak_rss = 0.0
        self._peak_cpu = 0.0
        self._peak_temp: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)  # prime the counter
            except Exception:
                pass

    def sample(self) -> ResourceObservation:
        if psutil is not None:
            try:
                virtual = psutil.virtual_memory()
                total_mb = virtual.total / 1048576.0
                available_mb = virtual.available / 1048576.0
                rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1048576.0
                cpu = float(psutil.cpu_percent(interval=None))
                cpu_count = psutil.cpu_count(logical=True) or 1
            except Exception:
                total_mb, available_mb = _read_meminfo()
                rss_mb, cpu, cpu_count = _read_self_rss(), 0.0, os.cpu_count() or 1
        else:
            total_mb, available_mb = _read_meminfo()
            rss_mb, cpu, cpu_count = _read_self_rss(), 0.0, os.cpu_count() or 1
        temperature, source = _read_temperature()
        try:
            disk_free = shutil.disk_usage(Path.cwd()).free / 1048576.0
        except Exception:
            disk_free = 0.0
        observation = ResourceObservation(
            meta=ResourceObservation.build_meta(
                identity=(iso_now(), f"{available_mb:.0f}"), stage="resource_assessment"
            ),
            total_ram_mb=round(total_mb, 2),
            available_ram_mb=round(available_mb, 2),
            process_rss_mb=round(rss_mb, 2),
            cpu_percent=round(cpu, 2),
            cpu_count=int(cpu_count),
            temperature_c=temperature,
            thermal_source=source,
            disk_free_mb=round(disk_free, 2),
            sampled_at=iso_now(),
        )
        self._peak_rss = max(self._peak_rss, observation.process_rss_mb)
        self._peak_cpu = max(self._peak_cpu, observation.cpu_percent)
        if temperature is not None:
            self._peak_temp = max(self._peak_temp or 0.0, temperature)
        return observation.sealed()  # type: ignore[return-value]

    # -- background peak tracking ------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="leanlm-sampler",
                                        daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.policy.sample_interval_s):
            try:
                self.sample()
            except Exception:
                continue

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def peak_rss_mb(self) -> float:
        return round(self._peak_rss, 2)

    @property
    def peak_cpu_percent(self) -> float:
        return round(self._peak_cpu, 2)

    @property
    def peak_temperature_c(self) -> float | None:
        return self._peak_temp

    def reset(self) -> None:
        self._peak_rss = self._peak_cpu = 0.0
        self._peak_temp = None


class MetricsAssembler:
    """Builds the MetricsSnapshot from everything the IEC collected."""

    def assemble(self, iec, *, peak_rss_mb: float = 0.0,
                 peak_temperature_c: float | None = None,
                 peak_cpu_percent: float = 0.0,
                 total_ms: float = 0.0) -> MetricsSnapshot:
        response = iec.response
        prompt = iec.prompt
        optimized = iec.optimized_context
        validation = iec.validation
        resources = iec.resources
        retrieval = iec.trace.get("retrieval", {})
        prompt_trace = iec.trace.get("prompt", {})

        # The ratio that actually demonstrates the layer's value: how much of
        # the corpus never reached the model. The compaction-only ratio is kept
        # separately so the two are never confused in a report.
        corpus_tokens = iec.context_package.total_tokens if iec.context_package else 0
        context_tokens = prompt.context_tokens if prompt else 0
        end_to_end = round(1.0 - (context_tokens / corpus_tokens), 4) if corpus_tokens else 0.0

        # A simulated backend produces no model timings. Emitting a throughput
        # here yields absurd figures (tens of millions of tokens per second)
        # that are one copy-paste away from appearing in a report as if they
        # were real. The layer's own timings stay -- they were measured.
        simulated = bool(response.is_simulated) if response is not None else False
        model_timings_are_real = response is not None and not simulated

        snapshot = MetricsSnapshot(
            meta=MetricsSnapshot.build_meta(
                identity=(iec.request_id, f"{total_ms:.3f}"), stage="metrics_finalization"
            ),
            session_id=iec.session_id,
            request_id=iec.request_id,
            profile_id=iec.profile_id,
            model_id=response.model_id if response else "",
            first_token_latency_ms=(response.first_token_latency_ms
                                    if model_timings_are_real else 0.0),
            inference_ms=response.inference_ms if model_timings_are_real else 0.0,
            total_ms=round(total_ms, 3),
            tokens_per_second=(response.tokens_per_second
                               if model_timings_are_real else 0.0),
            generated_tokens=response.generated_tokens if response else 0,
            prompt_tokens=prompt.prompt_tokens if prompt else 0,
            context_tokens=prompt.context_tokens if prompt else 0,
            peak_rss_mb=round(max(peak_rss_mb,
                                  resources.process_rss_mb if resources else 0.0), 2),
            available_ram_mb=resources.available_ram_mb if resources else 0.0,
            cpu_percent=round(max(peak_cpu_percent,
                                  resources.cpu_percent if resources else 0.0), 2),
            temperature_c=peak_temperature_c if peak_temperature_c is not None
            else (resources.temperature_c if resources else None),
            context_compression_ratio=max(0.0, end_to_end),
            dedup_compression_ratio=optimized.compression_ratio if optimized else 0.0,
            evidence_coverage=validation.evidence_coverage if validation else 0.0,
            retrieval_precision=float(retrieval.get("precision", 0.0)),
            prompt_efficiency=float(prompt_trace.get("prompt_efficiency", 0.0)),
            response_grounding_rate=validation.grounding_rate if validation else 0.0,
            stage_durations=iec.stage_durations(),
            is_simulated=bool(response.is_simulated) if response else False,
        )
        return snapshot.sealed()  # type: ignore[return-value]
