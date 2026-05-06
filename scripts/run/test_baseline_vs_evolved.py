"""Unit tests for the G5 Baseline-vs-Evolved acceptance gate.

These run without invoking any agent — they exercise extract_metrics
and evaluate_gate purely on synthetic inputs so that the gate logic
itself is locked down.

Run with:
    python -m unittest bamboo.scripts.run.test_baseline_vs_evolved
"""

from __future__ import annotations

import unittest

from .baseline_vs_evolved import (
    EFFICIENCY_RATIO,
    RunMetrics,
    evaluate_gate,
    extract_metrics,
)


class ExtractMetricsTest(unittest.TestCase):
    def test_parses_turn_count_and_overall_level(self) -> None:
        result_json = {"pass4": {"overall_level": 2}}
        stderr = "[panda] Session ended after 7 messages, 12 turns\n"
        m = extract_metrics(result_json, stderr, wall_time_ms=42000, exit_code=0)
        self.assertEqual(m.overall_level, 2)
        self.assertEqual(m.turn_count, 12)
        self.assertEqual(m.wall_time_ms, 42000)
        self.assertEqual(m.exit_code, 0)
        self.assertIsNone(m.error)

    def test_missing_result_json_yields_zero_level(self) -> None:
        m = extract_metrics(
            None, "Session ended after 1 messages, 1 turns",
            wall_time_ms=1000, exit_code=1, error="boom",
        )
        self.assertEqual(m.overall_level, 0)
        self.assertEqual(m.turn_count, 1)
        self.assertEqual(m.error, "boom")

    def test_missing_pass4_field_treated_as_level_zero(self) -> None:
        m = extract_metrics(
            {"some_other_field": True}, "",
            wall_time_ms=0, exit_code=0,
        )
        self.assertEqual(m.overall_level, 0)
        self.assertEqual(m.turn_count, 0)

    def test_handles_null_overall_level(self) -> None:
        m = extract_metrics(
            {"pass4": {"overall_level": None}}, "",
            wall_time_ms=0, exit_code=0,
        )
        self.assertEqual(m.overall_level, 0)

    def test_no_session_end_line_means_zero_turns(self) -> None:
        m = extract_metrics(
            {"pass4": {"overall_level": 1}},
            "log line that doesn't mention sessions or turns",
            wall_time_ms=5000, exit_code=0,
        )
        self.assertEqual(m.turn_count, 0)


def _metrics(level: int, turns: int, ms: int) -> RunMetrics:
    return RunMetrics(
        overall_level=level, turn_count=turns,
        wall_time_ms=ms, exit_code=0,
    )


class EvaluateGateTest(unittest.TestCase):
    def test_pass_strict_outcome_improvement(self) -> None:
        cold = _metrics(level=1, turns=20, ms=10_000)
        warm = _metrics(level=2, turns=20, ms=10_000)
        gate = evaluate_gate(cold, warm)
        self.assertTrue(gate["passed"])
        self.assertIn("strict-outcome-improvement", gate["criteria_met"])

    def test_pass_turn_count_cut(self) -> None:
        # Same level, but 60 ≤ 0.7×100 = 70 → cut criterion fires.
        cold = _metrics(level=1, turns=100, ms=20_000)
        warm = _metrics(level=1, turns=60, ms=20_000)
        gate = evaluate_gate(cold, warm)
        self.assertTrue(gate["passed"])
        self.assertIn("turn-count-cut", gate["criteria_met"])
        self.assertNotIn("wall-time-cut", gate["criteria_met"])

    def test_pass_wall_time_cut(self) -> None:
        cold = _metrics(level=2, turns=20, ms=10_000)
        warm = _metrics(level=2, turns=20, ms=6_000)
        gate = evaluate_gate(cold, warm)
        self.assertTrue(gate["passed"])
        self.assertIn("wall-time-cut", gate["criteria_met"])

    def test_fail_when_warm_regressed(self) -> None:
        cold = _metrics(level=2, turns=20, ms=10_000)
        warm = _metrics(level=1, turns=10, ms=5_000)
        gate = evaluate_gate(cold, warm)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["criteria_met"], [])
        self.assertIn("FAIL", gate["rationale"])
        self.assertIn("ablate", gate["rationale"])

    def test_fail_when_warm_equal_but_no_efficiency_gain(self) -> None:
        cold = _metrics(level=1, turns=20, ms=10_000)
        warm = _metrics(level=1, turns=20, ms=10_000)
        gate = evaluate_gate(cold, warm)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["criteria_met"], [])

    def test_fail_when_cold_turn_count_zero_blocks_ratio_compare(self) -> None:
        # Cold log was missing the turn-count line; we cannot compute a ratio,
        # so the turn-count criterion must not fire on its own.
        cold = _metrics(level=1, turns=0, ms=10_000)
        warm = _metrics(level=1, turns=2, ms=10_000)
        gate = evaluate_gate(cold, warm)
        self.assertFalse(gate["passed"])
        self.assertNotIn("turn-count-cut", gate["criteria_met"])

    def test_fail_when_warm_turn_count_zero_blocks_ratio_compare(self) -> None:
        cold = _metrics(level=1, turns=10, ms=10_000)
        warm = _metrics(level=1, turns=0, ms=10_000)
        gate = evaluate_gate(cold, warm)
        self.assertFalse(gate["passed"])
        self.assertNotIn("turn-count-cut", gate["criteria_met"])

    def test_fail_when_both_wall_times_zero_blocks_ratio_compare(self) -> None:
        cold = _metrics(level=1, turns=10, ms=0)
        warm = _metrics(level=1, turns=10, ms=0)
        gate = evaluate_gate(cold, warm)
        self.assertFalse(gate["passed"])
        self.assertNotIn("wall-time-cut", gate["criteria_met"])

    def test_pass_strict_improvement_even_when_warm_slower(self) -> None:
        # Strict outcome improvement is sufficient on its own.
        cold = _metrics(level=1, turns=10, ms=5_000)
        warm = _metrics(level=3, turns=30, ms=20_000)
        gate = evaluate_gate(cold, warm)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["criteria_met"], ["strict-outcome-improvement"])

    def test_threshold_boundary_just_at_ratio(self) -> None:
        # warm exactly at 0.7×cold should pass (≤, not <).
        cold = _metrics(level=1, turns=100, ms=10_000)
        warm = _metrics(level=1, turns=int(EFFICIENCY_RATIO * 100), ms=10_000)
        gate = evaluate_gate(cold, warm)
        self.assertTrue(gate["passed"])
        self.assertIn("turn-count-cut", gate["criteria_met"])

    def test_threshold_boundary_just_above_ratio(self) -> None:
        # warm at 71/100 should fail (above 0.7).
        cold = _metrics(level=1, turns=100, ms=10_000)
        warm = _metrics(level=1, turns=71, ms=10_000)
        gate = evaluate_gate(cold, warm)
        self.assertFalse(gate["passed"])

    def test_multiple_criteria_can_fire_simultaneously(self) -> None:
        cold = _metrics(level=1, turns=100, ms=10_000)
        warm = _metrics(level=2, turns=50, ms=4_000)
        gate = evaluate_gate(cold, warm)
        self.assertTrue(gate["passed"])
        self.assertIn("strict-outcome-improvement", gate["criteria_met"])
        self.assertIn("turn-count-cut", gate["criteria_met"])
        self.assertIn("wall-time-cut", gate["criteria_met"])


if __name__ == "__main__":
    unittest.main()
