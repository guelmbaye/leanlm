"""Package-local models for CAP-003."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BudgetFactors:
    ram: float
    thermal: float
    cpu: float
    intent: float

    def limiting(self) -> str:
        pairs = (("ram", self.ram), ("thermal", self.thermal), ("cpu", self.cpu))
        name, value = min(pairs, key=lambda kv: kv[1])
        return name if value < 1.0 else "model_context"

    def multiplier(self) -> float:
        return min(self.ram, self.thermal, self.cpu) * self.intent

    def to_dict(self) -> dict[str, float]:
        return {"ram": round(self.ram, 4), "thermal": round(self.thermal, 4),
                "cpu": round(self.cpu, 4), "intent": round(self.intent, 4)}
