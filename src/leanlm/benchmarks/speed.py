"""Raw model speed, straight from `llama-bench`.

The two numbers that decide 30% of the ADTC score, measured without LeanLM in
the path: prompt processing and generation, in tokens per second.

It exists as a command because running `llama-bench` by hand invites two
mistakes that both produce a plausible-looking wrong answer. Omitting `-m`
benchmarks llama.cpp's default path rather than your model; leaving a
`llama-server` resident holds the weights in RAM and competes with the run that
is trying to measure them.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from ..shared.clock import iso_now
from ..shared.errors import config_error

# | model | size | params | backend | threads | test | t/s |
# The final cell is read as "everything that is not a pipe" rather than as a
# value followed by a specific separator. llama-bench writes "32.96 ± 1.01", and
# insisting on that "±" made the parser hostage to one character surviving a
# decode -- which on Windows it did not.
_ROW = re.compile(
    r"\|\s*(?P<model>[^|]+?)\s*\|[^|]+\|[^|]+\|[^|]+\|\s*(?P<threads>\d+)\s*\|"
    r"\s*(?P<test>[a-z]+\d+)\s*\|(?P<cell>[^|]+)\|")
_VALUE = re.compile(r"(?P<value>\d+(?:\.\d+)?)")
# Two numbers, separated by something that is not a digit or a decimal point.
# A looser pattern matched "32.96" as "32" plus a deviation of "96" -- inventing
# a dispersion the benchmark never reported, which is worse than reporting none.
_STDEV = re.compile(
    r"\d+(?:\.\d+)?\s*[^\d.\s][^\d.]*\s*(?P<stdev>\d+(?:\.\d+)?)")

TPS_REFERENCE = 15.0     # from the official profiler: S_perf = min(TPS/15, 1)


@dataclass(frozen=True, slots=True)
class SpeedRow:
    test: str
    tokens_per_second: float
    stdev: float
    threads: int

    @property
    def kind(self) -> str:
        return "prompt processing" if self.test.startswith("pp") else "generation"

    def to_dict(self) -> dict[str, Any]:
        return {"test": self.test, "kind": self.kind,
                "tokens_per_second": self.tokens_per_second,
                "stdev": self.stdev, "threads": self.threads}


def competing_processes() -> list[str]:
    """Anything holding the model in memory while we try to measure it."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return []
    names = []
    for process in psutil.process_iter(["name"]):
        name = (process.info.get("name") or "").lower()
        if any(marker in name for marker in ("llama-server", "llama-cli")):
            names.append(process.info["name"])
    return sorted(set(names))


def measure(model_path: str, *, threads: int, prompt_tokens: int = 128,
            generated_tokens: int = 32, binary: str = "llama-bench",
            timeout_s: float = 900.0) -> dict[str, Any]:
    if shutil.which(binary) is None:
        raise config_error(
            "SPD-001", f"{binary} is not on PATH", capability="benchmarks",
            recommended_action="Windows: .\\scripts\\provision_windows.ps1  "
                               "Linux: bash scripts/provision_ubuntu.sh",
        )
    if not model_path:
        raise config_error(
            "SPD-002", "no model path is configured", capability="benchmarks",
            recommended_action="set model.path in the runtime profile, or pass --model",
        )

    command = [binary, "-m", model_path, "-p", str(prompt_tokens),
               "-n", str(generated_tokens), "-t", str(threads)]
    # llama.cpp writes UTF-8. Python decodes a subprocess with the platform
    # default, which on Windows is the ANSI code page: "±" arrives as "Â±" and any
    # accented character in a generated answer is corrupted the same way. Decoding
    # is pinned, and undecodable bytes are replaced rather than raising -- a model's
    # answer must never be lost to a byte.
    completed = subprocess.run(command, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=timeout_s)
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise config_error(
            "SPD-003", "llama-bench failed", capability="benchmarks",
            recommended_action="run the command by hand to see the error",
            command=" ".join(command), stderr=output[-400:],
        )

    rows = []
    for match in _ROW.finditer(output):
        cell = match.group("cell")
        value = _VALUE.search(cell)
        if not value:
            continue
        spread = _STDEV.search(cell)
        rows.append(SpeedRow(match.group("test"), float(value.group("value")),
                             float(spread.group("stdev")) if spread else 0.0,
                             int(match.group("threads"))))
    if not rows:
        raise config_error(
            "SPD-004", "llama-bench produced no readable results",
            capability="benchmarks",
            recommended_action=(
                "run `llama-bench -m <model> -p 128 -n 32` by hand and compare. "
                "If the table looks right, the format has changed"),
            output=output[-600:],
        )

    generation = next((r for r in rows if r.test.startswith("tg")), None)
    return {
        "measured_at": iso_now(),
        "model": model_path,
        "threads": threads,
        "command": " ".join(command),
        "rows": [r.to_dict() for r in rows],
        "generation_tps": generation.tokens_per_second if generation else 0.0,
        "throughput_component": (
            round(min(generation.tokens_per_second / TPS_REFERENCE, 1.0), 4)
            if generation else 0.0),
        "competing_processes": competing_processes(),
    }


def render(result: dict[str, Any]) -> str:
    lines = [f"llama-bench on {result['model']} with {result['threads']} threads", ""]
    for row in result["rows"]:
        spread = f" ± {row['stdev']}" if row["stdev"] else ""
        lines.append(f"  {row['kind']:20} {row['tokens_per_second']:>8.2f}{spread} tok/s")

    component = result["throughput_component"]
    lines.extend([
        "",
        f"  throughput component : {component:.0%} of 30 points "
        f"({component * 30:.1f} pts)",
        f"  reference            : {TPS_REFERENCE:.0f} tok/s for full marks",
    ])
    if result["competing_processes"]:
        lines.extend([
            "",
            f"  ! {', '.join(result['competing_processes'])} was running during "
            "this measurement.",
            "    It holds the model in memory and competes for the cores the "
            "benchmark is",
            "    timing. Stop it and measure again.",
        ])
    return "\n".join(lines)
