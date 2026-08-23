"""Benchmark scenarios (BHB s.4).

A scenario is a *hypothesis with a measurement attached*, not a script. Each one
states what it expects to observe before it runs, so the result can contradict
it. That is the difference between evidence and a demo.

The six scenarios are declared in `configs/benchmark.yaml`; the constants below
are the fallback used when no configuration file is reachable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..shared.config import config_dir, load_yaml

DEFAULT_SCENARIOS: tuple[dict[str, Any], ...] = (
    {"id": "S1", "name": "Small document",
     "hypothesis": "A short document sets the latency floor of the layer",
     "questions": ["Quel est le delai de remboursement des notes de frais ?"],
     "adtc_criteria": ["Throughput", "Memory"]},
    {"id": "S2", "name": "Medium corpus",
     "hypothesis": "Nominal behaviour on a realistic multi-document corpus",
     "questions": ["Quel est le plafond journalier pour les repas ?",
                   "Qui valide une demande de conge exceptionnel ?"],
     "adtc_criteria": ["Accuracy", "Throughput"]},
    {"id": "S3", "name": "Large corpus",
     "hypothesis": "Context budgeting keeps prompt size flat as the corpus grows",
     "questions": ["Resume la politique de teletravail."],
     "adtc_criteria": ["Memory", "Thermal"]},
    {"id": "S4", "name": "Repeated queries",
     "hypothesis": "Repeating one request does not drift in speed or memory",
     "questions": ["Quel est le delai de remboursement des notes de frais ?"],
     "repeat_question": 3, "adtc_criteria": ["Throughput", "Thermal"]},
    {"id": "S5", "name": "Cold start",
     "hypothesis": "The first request after load carries the initialisation cost",
     "questions": ["Quelles depenses sont non remboursables ?"],
     "cold_start": True, "adtc_criteria": ["Throughput"]},
    {"id": "S6", "name": "Unanswerable question",
     "hypothesis": "Outside the corpus, the system declares insufficiency",
     "questions": ["Quel est le cours de l'action Tesla aujourd'hui ?"],
     "expect_insufficient": True, "adtc_criteria": ["Accuracy"]},
)


@dataclass(frozen=True, slots=True)
class Scenario:
    id: str
    name: str
    hypothesis: str = ""
    questions: tuple[str, ...] = ()
    documents: int = 0                 # 0 -> the whole corpus
    repeat_question: int = 1
    cold_start: bool = False
    expect_insufficient: bool = False
    adtc_criteria: tuple[str, ...] = ()
    # Free-form controlled variables recorded next to the measurement, so a
    # result can be replayed with the conditions that produced it (PEM).
    variables: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, node: dict[str, Any]) -> "Scenario":
        return cls(
            id=str(node.get("id", "S?")),
            name=str(node.get("name", node.get("id", "scenario"))),
            hypothesis=str(node.get("hypothesis", "")),
            questions=tuple(str(q) for q in (node.get("questions") or ())),
            documents=int(node.get("documents", 0) or 0),
            repeat_question=int(node.get("repeat_question", 1) or 1),
            cold_start=bool(node.get("cold_start", False)),
            expect_insufficient=bool(node.get("expect_insufficient", False)),
            adtc_criteria=tuple(str(c) for c in (node.get("adtc_criteria") or ())),
            variables=dict(node.get("variables") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "hypothesis": self.hypothesis,
            "questions": list(self.questions), "documents": self.documents,
            "repeat_question": self.repeat_question, "cold_start": self.cold_start,
            "expect_insufficient": self.expect_insufficient,
            "adtc_criteria": list(self.adtc_criteria),
            "variables": dict(self.variables),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    repeat: int = 5
    warmup: int = 1
    profile: str = "benchmark"
    results_dir: str = "benchmarks/results"
    baseline_file: str = "benchmarks/results/baseline.json"
    regression: dict[str, float] = field(default_factory=dict)
    scenarios: tuple[Scenario, ...] = ()

    def scenario(self, scenario_id: str) -> Scenario | None:
        for item in self.scenarios:
            if item.id.lower() == scenario_id.lower():
                return item
        return None


def _default_path() -> Path:
    return config_dir() / "benchmark.yaml"


def load_benchmark_config(path: str | Path | None = None) -> BenchmarkConfig:
    target = Path(path) if path else _default_path()
    raw: dict[str, Any] = load_yaml(target) if target.is_file() else {}
    node = raw.get("benchmark", {}) or {}
    declared = raw.get("scenarios") or list(DEFAULT_SCENARIOS)
    return BenchmarkConfig(
        repeat=int(node.get("repeat", 5) or 5),
        warmup=int(node.get("warmup", 1) or 1),
        profile=str(node.get("profile", "benchmark")),
        results_dir=str(node.get("results_dir", "benchmarks/results")),
        baseline_file=str(node.get("baseline_file", "benchmarks/results/baseline.json")),
        regression={k: float(v) for k, v in (node.get("regression") or {}).items()},
        scenarios=tuple(Scenario.from_dict(item) for item in declared),
    )


def load_scenarios(path: str | Path | None = None) -> tuple[Scenario, ...]:
    """Convenience wrapper when only the scenario list is needed."""
    return load_benchmark_config(path).scenarios
