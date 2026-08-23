"""The competition rubric, applied to our own numbers."""
from __future__ import annotations

from leanlm.benchmarks.scoring import ScoringPolicy, render_score, score_campaign


def _campaign(**overrides):
    payload = {
        "is_simulated": False,
        "scenarios": [{"scenario": {"id": "S1"}, "verdict": "pass", "runs": []}],
        "metrics": {
            "tokens_per_second": {"mean": 12.0},
            "peak_rss_mb": {"mean": 3500.0},
            "available_ram_mb": {"mean": 3000.0},
            "temperature_c": {"mean": 60.0},
        },
    }
    payload.update(overrides)
    return payload


ACCURACY = {"overall_accuracy": 1.0, "probes": 12, "unanswerable": 2}


class TestWeights:
    def test_the_declared_weights_sum_to_one(self):
        assert ScoringPolicy.load().weights_sum_to_one

    def test_the_official_formula_is_applied(self):
        """S = 0.50*acc + 0.30*min(TPS/15,1) + 0.20*(7-peak_gb)/7, minus penalty."""
        score = score_campaign(_campaign(), ACCURACY)
        assert not score.disqualified
        components = {c.name: c for c in score.components}
        assert components["throughput"].normalized == round(12.0 / 15.0, 4)
        expected_efficiency = round((7.0 - 3500 / 1024) / 7.0, 4)
        assert components["efficiency"].normalized == expected_efficiency
        assert score.total == round(0.5 + 0.3 * (12 / 15)
                                    + 0.2 * expected_efficiency, 4)

    def test_reaching_the_reference_throughput_earns_full_marks(self):
        campaign = _campaign(metrics={**_campaign()["metrics"],
                                      "tokens_per_second": {"mean": 18.0}})
        throughput = next(c for c in score_campaign(campaign, ACCURACY).components
                          if c.name == "throughput")
        assert throughput.normalized == 1.0

    def test_efficiency_is_measured_against_seven_gigabytes_not_eight(self):
        """The profile targets 8 GB; the scoring limit is 7. Using 8 would have
        overstated every efficiency figure we reported."""
        campaign = _campaign(metrics={**_campaign()["metrics"],
                                      "peak_rss_mb": {"mean": 2048.0}})
        efficiency = next(c for c in score_campaign(campaign, ACCURACY).components
                          if c.name == "efficiency")
        assert efficiency.normalized == round((7.0 - 2.0) / 7.0, 4)

    def test_accuracy_carries_the_largest_weight(self):
        components = {c.name: c.weight for c in
                      score_campaign(_campaign(), ACCURACY).components}
        assert components["accuracy"] > components["throughput"]
        assert components["throughput"] > components["efficiency"]


class TestMissingMeasurements:
    def test_absent_ground_truth_scores_zero_rather_than_guessing(self):
        """The alternative -- assuming accuracy is fine -- is how a planning
        number becomes a claim."""
        score = score_campaign(_campaign(), None)
        accuracy = next(c for c in score.components if c.name == "accuracy")
        assert accuracy.contribution == 0.0
        assert any("NO GROUND TRUTH" in c.basis for c in score.components)
        assert any("heaviest criterion" in w for w in score.warnings)

    def test_a_simulated_run_cannot_claim_throughput(self):
        campaign = _campaign(is_simulated=True,
                             metrics={**_campaign()["metrics"],
                                      "tokens_per_second": {"mean": 0.0}})
        score = score_campaign(campaign, ACCURACY)
        throughput = next(c for c in score.components if c.name == "throughput")
        assert throughput.normalized == 0.0
        assert score.to_dict()["provisional"] is True

    def test_a_missing_thermal_sensor_is_not_a_cool_machine(self):
        campaign = _campaign(metrics={"tokens_per_second": {"mean": 12.0},
                                      "peak_rss_mb": {"mean": 3500.0}})
        score = score_campaign(campaign, ACCURACY)
        assert score.thermal_penalty == 0.0
        assert any("not because the machine stayed cool" in w for w in score.warnings)


class TestDisqualification:
    def test_running_out_of_memory_ends_at_zero(self):
        """Not a penalty. A cliff."""
        campaign = _campaign(metrics={**_campaign()["metrics"],
                                      "available_ram_mb": {"mean": 100.0}})
        score = score_campaign(campaign, ACCURACY)
        assert score.disqualified
        assert score.total == 0.0
        assert any("out-of-memory" in r for r in score.disqualification_reasons)

    def test_exceeding_the_target_machine_disqualifies(self):
        campaign = _campaign(metrics={**_campaign()["metrics"],
                                      "peak_rss_mb": {"mean": 9000.0}})
        assert score_campaign(campaign, ACCURACY).disqualified

    def test_a_crash_disqualifies(self):
        campaign = _campaign(scenarios=[{
            "scenario": {"id": "S3"}, "verdict": "fail",
            "runs": [{"error": "UNEXPECTED: MemoryError"}]}])
        score = score_campaign(campaign, ACCURACY)
        assert score.disqualified
        assert "S3" in score.disqualification_reasons[0]

    def test_perfect_accuracy_does_not_rescue_a_disqualified_run(self):
        campaign = _campaign(metrics={**_campaign()["metrics"],
                                      "available_ram_mb": {"mean": 10.0}})
        assert score_campaign(campaign, ACCURACY).total == 0.0

    def test_the_rendering_leads_with_the_disqualification(self):
        campaign = _campaign(metrics={**_campaign()["metrics"],
                                      "available_ram_mb": {"mean": 10.0}})
        assert render_score(score_campaign(campaign, ACCURACY)).startswith("DISQUALIFIED")


class TestThermalPenalty:
    def test_the_penalty_is_a_step_at_85_degrees_not_a_ramp(self):
        """The published rule deducts a flat 10 points above 85 C."""
        cool = _campaign(metrics={**_campaign()["metrics"],
                                  "temperature_c": {"mean": 84.0}})
        hot = _campaign(metrics={**_campaign()["metrics"],
                                 "temperature_c": {"mean": 86.0}})
        assert score_campaign(cool, ACCURACY).thermal_penalty == 0.0
        assert score_campaign(hot, ACCURACY).thermal_penalty == 0.10

    def test_flagged_throttling_costs_the_same_as_heat(self):
        campaign = _campaign(thermal_throttling=True)
        assert score_campaign(campaign, ACCURACY).thermal_penalty == 0.10

    def test_a_cool_run_loses_none(self):
        assert score_campaign(_campaign(), ACCURACY).thermal_penalty == 0.0


class TestProvisionalLabelling:
    def test_the_rubric_source_names_the_profiler(self):
        assert (score_campaign(_campaign(), ACCURACY).rubric_source
                == "official-profiler")

    def test_a_simulated_run_stays_provisional(self):
        score = score_campaign(_campaign(is_simulated=True), ACCURACY)
        assert score.to_dict()["provisional"] is True
        assert "provisional" in render_score(score)

    def test_our_accuracy_is_not_the_officially_scored_accuracy(self):
        """The official stage runs lm-eval `arc_easy` against the model, plus
        qualitative judging of four prompts. Ours measures whether the retrieval
        layer answers from a corpus. Reporting one as the other would claim a
        number the judges never compute."""
        from pathlib import Path
        config = Path("configs/scoring.yaml").read_text(encoding="utf-8")
        assert "arc_easy" in config


class TestUnmeasuredIsNotFailure:
    """A silent sensor is a gap in the evidence, never proof of catastrophe.

    Found on a Windows laptop: the /proc fallbacks returned 0 MB, the scoring
    module read 0 MB available as an out-of-memory kill, and a healthy run was
    reported as DISQUALIFIED with a score of zero.
    """

    def _unmeasured(self):
        return _campaign(metrics={"tokens_per_second": {"mean": 12.0},
                                  "peak_rss_mb": {"mean": 0.0},
                                  "available_ram_mb": {"mean": 0.0}})

    def test_zero_memory_readings_do_not_disqualify(self):
        score = score_campaign(self._unmeasured(), ACCURACY)
        assert not score.disqualified
        assert score.total > 0.0

    def test_the_gap_is_stated_rather_than_passed_over(self):
        score = score_campaign(self._unmeasured(), ACCURACY)
        assert any("not measured on this machine" in w for w in score.warnings)

    def test_unmeasured_efficiency_still_scores_zero(self):
        """Not disqualifying is not the same as assuming it is fine."""
        score = score_campaign(self._unmeasured(), ACCURACY)
        efficiency = next(c for c in score.components if c.name == "efficiency")
        assert efficiency.contribution == 0.0
        assert "NOT MEASURED" in efficiency.basis

    def test_a_real_shortage_still_disqualifies(self):
        """The guard must not have been softened into uselessness."""
        campaign = _campaign(metrics={**_campaign()["metrics"],
                                      "available_ram_mb": {"mean": 100.0},
                                      "peak_rss_mb": {"mean": 7900.0}})
        assert score_campaign(campaign, ACCURACY).disqualified


class TestCandidateRanking:
    """Choosing a model is arithmetic, and the arithmetic is not intuitive.

    `min(TPS / 15, 1.0)` means throughput saturates: past the reference there is
    no further reward, so speed bought beyond it is spent on nothing. A more
    capable model that misses the reference pays 2 points per token per second
    short, which a modest accuracy gain cannot repay.
    """

    def _candidates(self):
        from leanlm.benchmarks.scoring import Candidate
        return [
            Candidate("small", 0.60, 35.0, 1.0),
            Candidate("mid", 0.75, 16.0, 1.7),
            Candidate("large", 0.83, 8.0, 3.0),
        ]

    def test_throughput_saturates_at_the_reference(self):
        from leanlm.benchmarks.scoring import Candidate
        at_reference = Candidate("a", 0.5, 15.0, 2.0).score()
        far_above = Candidate("b", 0.5, 60.0, 2.0).score()
        assert at_reference["throughput"] == far_above["throughput"] == 30.0

    def test_the_mid_size_model_wins_on_this_rubric(self):
        """Not the most capable and not the fastest."""
        from leanlm.benchmarks.scoring import rank_candidates
        ranked = rank_candidates(self._candidates())
        assert ranked[0]["name"] == "mid"

    def test_missing_the_reference_costs_two_points_per_token(self):
        from leanlm.benchmarks.scoring import Candidate
        at_reference = Candidate("a", 0.5, 15.0, 2.0).score()
        one_short = Candidate("b", 0.5, 14.0, 2.0).score()
        assert round(at_reference["throughput"] - one_short["throughput"], 1) == 2.0

    def test_a_gigabyte_costs_about_three_points(self):
        from leanlm.benchmarks.scoring import Candidate
        lean = Candidate("a", 0.5, 20.0, 2.0).score()
        heavy = Candidate("b", 0.5, 20.0, 3.0).score()
        assert 2.5 <= lean["efficiency"] - heavy["efficiency"] <= 3.0

    def test_estimates_are_labelled_as_such(self):
        from leanlm.benchmarks.scoring import rank_candidates, render_candidates
        rendered = render_candidates(rank_candidates(self._candidates()))
        assert "(est.)" in rendered
        assert "measure" in rendered.lower()

    def test_a_candidate_below_the_reference_is_flagged(self):
        from leanlm.benchmarks.scoring import rank_candidates, render_candidates
        rendered = render_candidates(rank_candidates(self._candidates()))
        assert "*" in rendered
        assert "2 points" in rendered
