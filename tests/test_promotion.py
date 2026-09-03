from __future__ import annotations

import unittest

from fusion_core.promotion import promotion_verdict


class PromotionGateTests(unittest.TestCase):
    def base_aggregate(self) -> dict:
        return {
            "runs": 20,
            "valid_evaluations": 20,
            "variants": {
                "solo": {
                    "quality": 4.20,
                    "win_rate": 0.25,
                    "success_rate": 1.0,
                    "avg_latency_ms": 1000,
                    "avg_cost_usd": 0.10,
                },
                "fusion-current": {
                    "quality": 4.15,
                    "win_rate": 0.25,
                    "success_rate": 1.0,
                    "avg_latency_ms": 2000,
                    "avg_cost_usd": 0.20,
                },
                "fusion-deliberation": {
                    "quality": 4.35,
                    "win_rate": 0.50,
                    "success_rate": 1.0,
                    "avg_latency_ms": 4000,
                    "avg_cost_usd": 0.40,
                },
            },
        }

    def test_promotes_clear_quality_gain(self) -> None:
        verdict = promotion_verdict(self.base_aggregate())
        self.assertEqual(verdict["status"], "PROMOTE")
        self.assertAlmostEqual(verdict["metrics"]["quality_delta"], 0.20)
        self.assertAlmostEqual(verdict["metrics"]["latency_ratio"], 2.0)
        self.assertAlmostEqual(verdict["metrics"]["cost_ratio"], 2.0)

    def test_smoke_run_cannot_promote_by_default(self) -> None:
        aggregate = self.base_aggregate()
        aggregate["runs"] = 3
        aggregate["valid_evaluations"] = 3
        verdict = promotion_verdict(aggregate)
        self.assertEqual(verdict["status"], "HOLD")
        self.assertFalse(verdict["checks"]["valid_evaluations"]["passed"])

    def test_holds_when_quality_gain_is_too_small(self) -> None:
        aggregate = self.base_aggregate()
        aggregate["variants"]["fusion-deliberation"]["quality"] = 4.20
        verdict = promotion_verdict(aggregate)
        self.assertEqual(verdict["status"], "HOLD")
        self.assertFalse(verdict["checks"]["quality_delta"]["passed"])

    def test_holds_when_win_rate_does_not_improve(self) -> None:
        aggregate = self.base_aggregate()
        aggregate["variants"]["fusion-deliberation"]["win_rate"] = 0.25
        verdict = promotion_verdict(aggregate)
        self.assertEqual(verdict["status"], "HOLD")
        self.assertFalse(verdict["checks"]["win_rate_delta"]["passed"])

    def test_holds_on_success_rate_regression(self) -> None:
        aggregate = self.base_aggregate()
        aggregate["variants"]["fusion-deliberation"]["success_rate"] = 0.95
        verdict = promotion_verdict(aggregate)
        self.assertEqual(verdict["status"], "HOLD")
        self.assertFalse(verdict["checks"]["success_rate_regression"]["passed"])

    def test_efficiency_is_report_only_unless_threshold_is_set(self) -> None:
        aggregate = self.base_aggregate()
        aggregate["variants"]["fusion-deliberation"]["avg_latency_ms"] = 20000
        aggregate["variants"]["fusion-deliberation"]["avg_cost_usd"] = 2.0
        verdict = promotion_verdict(aggregate)
        self.assertEqual(verdict["status"], "PROMOTE")
        self.assertNotIn("latency_ratio", verdict["checks"])
        self.assertNotIn("cost_ratio", verdict["checks"])

        constrained = promotion_verdict(aggregate, max_latency_ratio=3.0, max_cost_ratio=3.0)
        self.assertEqual(constrained["status"], "HOLD")
        self.assertFalse(constrained["checks"]["latency_ratio"]["passed"])
        self.assertFalse(constrained["checks"]["cost_ratio"]["passed"])

    def test_missing_variants_holds(self) -> None:
        verdict = promotion_verdict({"runs": 20, "valid_evaluations": 20, "variants": {}})
        self.assertEqual(verdict["status"], "HOLD")


if __name__ == "__main__":
    unittest.main()
