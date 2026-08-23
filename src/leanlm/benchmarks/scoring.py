"""The competition's own scoring formula, computed on our own results.

Rationale: a campaign that reports fifteen metrics tells you the system works. It
does not tell you what the system would score. Those are different questions, and
only the second one decides anything.

The rubric this implements -- **50% accuracy, 30% throughput, 20% efficiency**,
with a thermal penalty and disqualification on out-of-memory or crash -- comes
from the strategy synthesis, not from a rulebook we have in hand. Two guards
against that gap doing damage:

  * every constant lives in `configs/scoring.yaml`, so correcting it is an edit
    rather than a rewrite;
  * every score carries `rubric_source`, and while it reads `assumed` the number
    is an internal planning figure. It is labelled that way in the CLI and in the
    report, because a self-assessed score presented as an official one is the
    kind of claim that loses more than it gains.

**Disqualification is not a low score.** An out-of-memory kill or a crash ends
the run at zero regardless of how good the other numbers were, so it is computed
first and short-circuits everything else. On an 8 GB target that is the single
most expensive failure available to us, and it deserves to be modelled as the
cliff it is rather than as a penalty term.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..shared.clock import iso_now
from ..shared.config import config_dir, load_yaml


@dataclass(frozen=True, slots=True)
class ScoringPolicy:
    """Weights and reference points. Every one of them is an assumption to check."""

    accuracy_weight: float = 0.50
    throughput_weight: float = 0.30
    efficiency_weight: float = 0.20

    # Reference points for normalising raw measurements onto 0..1.
    # Constants read from the official profiler, not inferred:
    #   S_perf = min(TPS / 15.0, 1.0) * 100
    #   S_eff  = max(0, (7.0 - peak_rss_gb) / 7.0) * 100
    # The Devpost rules describe throughput as "relative to the maximum observed
    # tokens per second"; the profiler that computes the score uses a fixed
    # reference of 15.0. The implementation wins.
    tps_reference: float = 15.0
    ram_limit_gb: float = 7.0                # not 8: the profile leaves headroom

    # "P_thermal = 10 points deduction, applied if the CPU throttles or core temp
    # exceeds 85 C." Ten points on a hundred-point scale.
    thermal_limit_c: float = 85.0
    thermal_penalty: float = 0.10

    oom_margin_mb: float = 256.0             # available RAM below this = OOM risk

    rubric_source: str = "official-profiler"

    @classmethod
    def load(cls, path: Path | None = None) -> "ScoringPolicy":
        target = path or (config_dir() / "scoring.yaml")
        if not target.is_file():
            return cls()
        raw = load_yaml(target).get("scoring", {}) or {}
        defaults = cls()
        return cls(**{
            field_name: raw.get(field_name, getattr(defaults, field_name))
            for field_name in cls.__dataclass_fields__
        })

    @property
    def weights_sum_to_one(self) -> bool:
        total = self.accuracy_weight + self.throughput_weight + self.efficiency_weight
        return abs(total - 1.0) < 1e-9


@dataclass(frozen=True, slots=True)
class Component:
    name: str
    raw: float | None
    normalized: float
    weight: float
    contribution: float
    basis: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "raw": self.raw, "normalized": self.normalized,
                "weight": self.weight, "contribution": self.contribution,
                "basis": self.basis}


@dataclass(frozen=True, slots=True)
class AdtcScore:
    total: float
    components: tuple[Component, ...]
    thermal_penalty: float
    disqualified: bool
    disqualification_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    rubric_source: str
    is_simulated: bool
    unscorable_weight: float = 0.0
    measured_at: str = field(default_factory=iso_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "components": [c.to_dict() for c in self.components],
            "thermal_penalty": self.thermal_penalty,
            "unscorable_weight": self.unscorable_weight,
            "disqualified": self.disqualified,
            "disqualification_reasons": list(self.disqualification_reasons),
            "warnings": list(self.warnings),
            "rubric_source": self.rubric_source,
            "is_simulated": self.is_simulated,
            "measured_at": self.measured_at,
            "provisional": (not self.rubric_source.startswith("official")
                            or self.is_simulated or bool(self.unscorable_weight)),
        }


def score_campaign(campaign: dict[str, Any], accuracy: dict[str, Any] | None = None,
                   policy: ScoringPolicy | None = None) -> AdtcScore:
    policy = policy or ScoringPolicy.load()
    metrics = campaign.get("metrics", {}) or {}
    warnings: list[str] = []

    # -- disqualification first: it ends the run, it does not reduce it --------
    reasons: list[str] = []
    failed = [s for s in campaign.get("scenarios", []) if s.get("verdict") == "fail"]
    crashed = [s for s in failed
               if any("UNEXPECTED" in (r.get("error") or "")
                      for r in s.get("runs", []))]
    if crashed:
        reasons.append(f"crash during {', '.join(s['scenario']['id'] for s in crashed)}")

    available = _mean(metrics, "available_ram_mb")
    peak_rss = _mean(metrics, "peak_rss_mb")

    # A sensor that reports nothing reports 0.0, and 0 MB available is
    # indistinguishable from an out-of-memory kill unless we insist on the
    # difference. On a machine whose platform we cannot sample -- Windows before
    # the Win32 path existed, a container with no /proc -- treating silence as
    # catastrophe disqualified runs that were perfectly healthy. An unmeasured
    # value is never evidence of failure.
    memory_measured = bool(available) and bool(peak_rss)
    if not memory_measured:
        warnings.append(
            "memory was not measured on this machine, so no out-of-memory check "
            "could be performed. This is a gap in the evidence, not a pass: "
            "install psutil or run `leanlm doctor` to see which sampler is active.")
    if memory_measured and available < policy.oom_margin_mb:
        reasons.append(f"available RAM fell to {available:.0f} MB, below the "
                       f"{policy.oom_margin_mb:.0f} MB out-of-memory margin")
    if peak_rss and peak_rss > policy.ram_limit_gb * 1024.0:
        reasons.append(f"peak RSS {peak_rss / 1024.0:.2f} GB exceeds the "
                       f"{policy.ram_limit_gb:.0f} GB limit")

    if reasons:
        return AdtcScore(
            total=0.0, components=(), thermal_penalty=0.0, disqualified=True,
            disqualification_reasons=tuple(reasons),
            warnings=tuple(["disqualification is not a low score: the run ends at zero",
                            *warnings]),
            rubric_source=policy.rubric_source,
            is_simulated=bool(campaign.get("is_simulated")),
        )

    # -- accuracy (50%) -------------------------------------------------------
    if accuracy and "overall_accuracy" in accuracy:
        accuracy_raw = float(accuracy["overall_accuracy"])
        accuracy_basis = (f"{accuracy.get('probes', 0)} ground-truth probes "
                          f"({accuracy.get('unanswerable', 0)} unanswerable)")
    else:
        accuracy_raw = 0.0
        accuracy_basis = "NO GROUND TRUTH -- scored zero rather than guessed"
        warnings.append(
            "no accuracy evaluation supplied; the heaviest criterion is scored at "
            "zero. Run `leanlm accuracy` and pass its output.")

    # -- throughput (30%) -----------------------------------------------------
    # Scored relative to the maximum observed across all submissions, which no
    # entrant can know. Inventing a target and scoring against it would produce
    # a number that feels precise and means nothing.
    tokens_per_second = _mean(metrics, "tokens_per_second")
    throughput_scorable = True
    if not tokens_per_second:
        throughput_raw, throughput_norm = None, 0.0
        throughput_basis = "not measured (no real model run)"
        warnings.append("throughput is unmeasured; a simulated run reports none.")
    else:
        throughput_raw = tokens_per_second
        throughput_norm = min(1.0, tokens_per_second / policy.tps_reference)
        throughput_basis = (f"{tokens_per_second:.1f} tok/s against the reference "
                            f"{policy.tps_reference:.0f} tok/s "
                            f"(full marks at or above)")
        if tokens_per_second < policy.tps_reference:
            warnings.append(
                f"throughput is {tokens_per_second / policy.tps_reference:.0%} of the "
                f"reference; each tok/s below {policy.tps_reference:.0f} costs 2 points "
                "of the final score.")

    # -- efficiency (20%) -----------------------------------------------------
    if peak_rss:
        # max(0, (RAM_LIMIT_GB - peak_rss_gb) / RAM_LIMIT_GB)
        peak_gb = peak_rss / 1024.0
        efficiency_norm = max(0.0, (policy.ram_limit_gb - peak_gb) / policy.ram_limit_gb)
        efficiency_basis = (f"peak {peak_gb:.2f} GB against a "
                            f"{policy.ram_limit_gb:.0f} GB limit "
                            f"({efficiency_norm:.0%} of the limit left unused)")
    else:
        efficiency_norm, efficiency_basis = 0.0, "peak memory NOT MEASURED -- scored zero"
        warnings.append(
            "peak memory is unmeasured, so efficiency scores zero rather than being "
            "assumed good. Installing psutil, or running on a platform the sampler "
            "supports, recovers 20% of the score.")

    components = (
        Component("accuracy", accuracy_raw, round(accuracy_raw, 4),
                  policy.accuracy_weight,
                  round(accuracy_raw * policy.accuracy_weight, 4), accuracy_basis),
        Component("throughput", throughput_raw, round(throughput_norm, 4),
                  policy.throughput_weight,
                  round(throughput_norm * policy.throughput_weight, 4),
                  throughput_basis),
        Component("efficiency", peak_rss, round(efficiency_norm, 4),
                  policy.efficiency_weight,
                  round(efficiency_norm * policy.efficiency_weight, 4),
                  efficiency_basis),
    )
    scorable = [c for c in components
                if not (c.name == "throughput" and not throughput_scorable)]
    subtotal = sum(c.contribution for c in scorable)
    unscorable_weight = round(sum(c.weight for c in components if c not in scorable), 4)

    # -- thermal penalty ------------------------------------------------------
    temperature = _mean(metrics, "temperature_c")
    throttled = bool(campaign.get("thermal_throttling"))
    penalty = 0.0
    if temperature is None and not throttled:
        warnings.append(
            "no thermal sensor was exposed; the penalty is zero because nothing "
            "was measured, not because the machine stayed cool. The rubric deducts "
            "10 points above 85 °C, so this is an unmeasured risk, not an absence "
            "of one -- install lm-sensors on the evaluation machine.")
    elif throttled or (temperature is not None and temperature > policy.thermal_limit_c):
        penalty = policy.thermal_penalty
        reason = "throttling was flagged" if throttled else \
            f"{temperature:.0f} °C exceeds {policy.thermal_limit_c:.0f} °C"
        warnings.append(f"thermal penalty applied ({reason}): "
                        f"-{policy.thermal_penalty:.0%}")

    if campaign.get("is_simulated"):
        warnings.append(
            "SIMULATED BACKEND: this score describes the optimization layer only. "
            "It is not a competition score and must not be presented as one.")

    return AdtcScore(
        total=round(max(0.0, subtotal - penalty), 4),
        components=components,
        unscorable_weight=unscorable_weight,
        thermal_penalty=penalty,
        disqualified=False,
        disqualification_reasons=(),
        warnings=tuple(warnings),
        rubric_source=policy.rubric_source,
        is_simulated=bool(campaign.get("is_simulated")),
    )


def _mean(metrics: dict[str, Any], key: str) -> float | None:
    node = metrics.get(key)
    if isinstance(node, dict):
        value = node.get("mean")
        return float(value) if isinstance(value, (int, float)) else None
    return float(node) if isinstance(node, (int, float)) else None


def _band(value: float, low: float, high: float) -> float:
    """Linear 0..1 between two reference points, clamped at both ends."""
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def render_score(score: AdtcScore) -> str:
    if score.disqualified:
        lines = ["DISQUALIFIED -- score 0", ""]
        lines.extend(f"  {reason}" for reason in score.disqualification_reasons)
        return "\n".join(lines)

    lines = [f"provisional score: {score.total:.1%}"
             if score.to_dict()["provisional"] else f"score: {score.total:.1%}", ""]
    for component in score.components:
        bar = "#" * int(component.normalized * 24)
        lines.append(f"  {component.name:11} {component.normalized:>6.1%} "
                     f"x{component.weight:.2f} = {component.contribution:>6.1%}  {bar}")
        lines.append(f"              {component.basis}")
    if score.thermal_penalty:
        lines.append(f"  thermal penalty  -{score.thermal_penalty:.1%}")
    if score.unscorable_weight:
        lines.append(f"  {score.unscorable_weight:.0%} of the weight cannot be "
                     "self-scored (graded against other submissions)")
    lines.extend(["", f"  rubric: {score.rubric_source}"])
    if score.warnings:
        lines.append("")
        lines.extend(f"  ! {warning}" for warning in score.warnings)
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A model under consideration, with the three numbers that decide."""

    name: str
    accuracy: float          # 0..1, from lm-eval arc_easy
    tokens_per_second: float
    peak_rss_gb: float
    measured: bool = False   # False means these are estimates

    def score(self, policy: ScoringPolicy | None = None) -> dict[str, Any]:
        policy = policy or ScoringPolicy()
        throughput = min(self.tokens_per_second / policy.tps_reference, 1.0)
        efficiency = max(0.0, (policy.ram_limit_gb - self.peak_rss_gb)
                         / policy.ram_limit_gb)
        parts = {
            "accuracy": round(self.accuracy * policy.accuracy_weight * 100, 1),
            "throughput": round(throughput * policy.throughput_weight * 100, 1),
            "efficiency": round(efficiency * policy.efficiency_weight * 100, 1),
        }
        return {
            "name": self.name, "measured": self.measured, **parts,
            "total": round(sum(parts.values()), 1),
            "tokens_per_second": self.tokens_per_second,
            "peak_rss_gb": self.peak_rss_gb,
            "accuracy_raw": self.accuracy,
            "throughput_saturated": self.tokens_per_second >= policy.tps_reference,
        }


def rank_candidates(candidates: list[Candidate],
                    policy: ScoringPolicy | None = None) -> list[dict[str, Any]]:
    """Rank models by what the rubric actually rewards.

    The non-obvious consequence of `min(TPS / 15, 1.0)`: throughput **saturates**.
    Past 15 tok/s there is no further reward, so speed bought beyond that point
    is spent on nothing. The optimum is therefore the largest model that still
    reaches the reference, not the fastest model available and not the most
    capable one.
    """
    policy = policy or ScoringPolicy()
    return sorted((c.score(policy) for c in candidates),
                  key=lambda row: -row["total"])


def render_candidates(rows: list[dict[str, Any]]) -> str:
    lines = ["| model | acc | tok/s | peak | 50% acc | 30% tp | 20% eff | total |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        saturated = "" if row["throughput_saturated"] else " *"
        flag = "" if row["measured"] else " (est.)"
        lines.append(
            f"| {row['name']}{flag} | {row['accuracy_raw']:.0%} | "
            f"{row['tokens_per_second']:.1f}{saturated} | {row['peak_rss_gb']:.1f} GB | "
            f"{row['accuracy']} | {row['throughput']} | {row['efficiency']} | "
            f"**{row['total']}** |")
    lines.append("")
    lines.append("  * below the 15 tok/s reference: every token per second short "
                 "costs 2 points")
    if any(not row["measured"] for row in rows):
        lines.append("  (est.) estimated, not measured. Replace with real numbers "
                     "before deciding.")
    return "\n".join(lines)
