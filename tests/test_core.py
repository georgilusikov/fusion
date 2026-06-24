from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from scripts import fusion


class FusionCoreTests(unittest.TestCase):
    def test_defaults_are_single_source_of_truth(self) -> None:
        parser = fusion.build_parser()
        args = parser.parse_args(["hello", "--dry-run"])
        self.assertEqual(args.strategy, fusion.DEFAULT_STRATEGY)
        self.assertIsNone(args.preset)
        self.assertEqual(args.agent_workspace, "snapshot")
        skill = (fusion.REPO_ROOT / "SKILL.md").read_text()
        readme = (fusion.REPO_ROOT / "README.md").read_text()
        self.assertIn("| `--strategy` | `adaptive` |", skill)
        self.assertNotIn("| `--preset` | `chinese` |", skill)
        self.assertIn("`--strategy adaptive`", readme)

    def test_parse_member_and_confidence(self) -> None:
        member = fusion.parse_member("cursor:composer!agent@skeptic", False, {})
        self.assertEqual(member.backend, "cursor")
        self.assertEqual(member.model, "composer")
        self.assertEqual(member.depth, "agent")
        self.assertEqual(member.role_key, "skeptic")
        self.assertEqual(fusion.extract_confidence("Confidence: 82%"), 0.82)
        self.assertEqual(fusion.extract_confidence("confidence=0.7"), 0.7)
        self.assertEqual(fusion.extract_confidence("Уверенность: 64%"), 0.64)

    def test_model_result_serializes_metrics(self) -> None:
        result = fusion.ModelResult(
            label="x", backend="openai", kind="api", ok=True, answer="ok",
            latency_ms=12, input_tokens=10, output_tokens=5, total_tokens=15,
            cost_usd=0.01, confidence=0.9,
        )
        payload = result.to_dict()
        self.assertEqual(payload["latency_ms"], 12)
        self.assertEqual(payload["total_tokens"], 15)
        self.assertEqual(payload["errors"], [])
        schema = json.loads((fusion.REPO_ROOT / "schemas" / "model-result.schema.json").read_text())
        self.assertEqual(set(payload), set(schema["required"]))

    def test_api_retry_and_usage(self) -> None:
        member = fusion.parse_member("openai:test", True, {})
        calls = {"count": 0}

        def request(_member, _prompt, _timeout):
            calls["count"] += 1
            if calls["count"] < 3:
                raise urllib.error.URLError("temporary")
            return "done Confidence: 75%", {
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
            }

        config = fusion.DispatchConfig(retries=2, backoff=0, pricing={"test": {"input": 1, "output": 2}})
        result = fusion.dispatch_api(member, "prompt", config, request_fn=request, sleep_fn=lambda _seconds: None)
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.total_tokens, 14)
        self.assertEqual(result.confidence, 0.75)
        self.assertAlmostEqual(result.cost_usd or 0, 0.000018)

    def test_failed_results_are_filtered_before_judge(self) -> None:
        good = fusion.ModelResult(label="good", backend="x", kind="api", ok=True, answer="answer")
        bad = fusion.ModelResult(label="bad", backend="y", kind="api", ok=False, answer="partial")
        empty = fusion.ModelResult(label="empty", backend="z", kind="api", ok=True, answer="")
        self.assertEqual([item.label for item in fusion.successful_results([good, bad, empty])], ["good"])
        self.assertEqual({item.label for item in fusion.failed_results([good, bad, empty])}, {"bad", "empty"})

    def test_judge_schema_validation(self) -> None:
        valid = {
            "consensus": ["a"],
            "contradictions": [{"point": "p", "sides": "a vs b"}],
            "coverage_gaps": [],
            "unique_insights": [{"model": "m", "insight": "i"}],
            "blind_spots": [],
            "recommendation": "r",
            "confidence": 0.8,
        }
        self.assertEqual(fusion.validate_judge_payload(valid), [])
        invalid = dict(valid)
        invalid["extra"] = True
        self.assertTrue(fusion.validate_judge_payload(invalid))
        external = json.loads((fusion.REPO_ROOT / "schemas" / "judge.schema.json").read_text())
        self.assertEqual(external, fusion.JUDGE_SCHEMA)

    def test_judge_repair_loop(self) -> None:
        valid = {
            "consensus": [],
            "contradictions": [],
            "coverage_gaps": [],
            "unique_insights": [],
            "blind_spots": [],
            "recommendation": "use the valid result",
        }
        answers = iter(["not json", json.dumps(valid)])

        def fake_dispatch(member, prompt, depth, config, apply_member_prompt):
            return fusion.ModelResult(
                label=member.label,
                backend=member.backend,
                kind=member.kind,
                model=member.model,
                ok=True,
                answer=next(answers),
            )

        judge_member = fusion.parse_member("claude", True, {})
        panel = [fusion.ModelResult(label="panel", backend="x", kind="api", ok=True, answer="answer")]
        result = fusion.run_judge(
            judge_member,
            "question",
            panel,
            fusion.DispatchConfig(),
            repair_attempts=1,
            dispatcher=fake_dispatch,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["parsed"]["recommendation"], "use the valid result")

    def test_adaptive_and_benchmark_selection(self) -> None:
        easy = fusion.select_strategy("What is 2 + 2?", requested="adaptive")
        self.assertEqual(easy.resolved, "lite")
        hard = fusion.select_strategy(
            "Design a production distributed architecture with security, migration, benchmarks, and rollback.",
            requested="adaptive",
        )
        self.assertEqual(hard.resolved, "pro")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps({
                "panels": [
                    {"preset": "dq", "quality": 0.8, "avg_cost_usd": 0.02, "avg_latency_ms": 1000},
                    {"preset": "free", "quality": 0.6, "avg_cost_usd": 0.0, "avg_latency_ms": 500},
                ]
            }))
            selected = fusion.select_strategy("simple", "lite", benchmark_path=path, budget_usd=0.001)
            self.assertEqual(selected.preset, "free")
            self.assertEqual(selected.source, "benchmark")

            path.write_text(json.dumps({
                "panels": [
                    {"preset": "dq", "quality": 0.99, "avg_cost_usd": None, "avg_latency_ms": 100},
                    {"preset": "free", "quality": 0.60, "avg_cost_usd": 0.0, "avg_latency_ms": 500},
                ]
            }))
            constrained = fusion.select_strategy("simple", "lite", benchmark_path=path, budget_usd=0.001)
            self.assertEqual(constrained.preset, "free")

    def test_cli_review_and_escalation_flags(self) -> None:
        parser = fusion.build_parser()
        args = parser.parse_args(["question", "--reviewers", "1", "--no-escalate"])
        self.assertEqual(args.reviewers, 1)
        self.assertTrue(args.no_escalate)

    def test_escalation_reasons_exported(self) -> None:
        panel = [
            fusion.ModelResult(label="a", backend="x", kind="api", ok=True, answer="one"),
            fusion.ModelResult(label="b", backend="y", kind="api", ok=True, answer="two"),
        ]
        judge = {
            "valid": True,
            "parsed": {
                "confidence": 0.5,
                "contradictions": [],
                "coverage_gaps": [],
            },
        }
        self.assertIn("low-judge-confidence", fusion.escalation_reasons(panel, judge))


if __name__ == "__main__":
    unittest.main()
