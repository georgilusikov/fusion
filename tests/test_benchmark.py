from __future__ import annotations

import unittest

from scripts import benchmark


class BenchmarkTests(unittest.TestCase):
    def test_score_text(self) -> None:
        case = {"expected_contains": ["429"], "expected_regex": [r"too\s+many"]}
        self.assertEqual(benchmark.score_text("429 Too Many Requests", case), 1.0)
        self.assertEqual(benchmark.score_text("429", case), 0.5)

        fixture = {"checks": {"must_include": ["323"], "max_chars": 20}}
        self.assertEqual(benchmark.score_text("323 = 17 * 19", fixture), 1.0)
        self.assertEqual(benchmark.score_text("wrong", fixture), 0.5)

        json_fixture = {"checks": {"json_keys": ["status", "risks"]}}
        self.assertEqual(benchmark.score_text('{"status": "ok", "risks": []}', json_fixture), 1.0)

    def test_aggregate(self) -> None:
        result = benchmark.aggregate("dq", [
            {"category": "code", "score": 1.0, "passed": True, "latency_ms": 100, "cost_usd": 0.01},
            {"category": "code", "score": 0.5, "passed": False, "latency_ms": 300, "cost_usd": 0.03},
        ])
        self.assertEqual(result["quality"], 0.75)
        self.assertEqual(result["pass_rate"], 0.5)
        self.assertEqual(result["avg_latency_ms"], 200)
        self.assertEqual(result["categories"]["code"], 0.75)


if __name__ == "__main__":
    unittest.main()
