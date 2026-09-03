from __future__ import annotations

import json
import re
import unittest
from collections import Counter
from pathlib import Path

from fusion_core.config import ModelResult
from fusion_core.deliberation_benchmark import (
    DELIBERATION_EVAL_AXES,
    aggregate_deliberation_runs,
    deliberation_eval_prompt,
    evaluate_case,
    load_deliberation_cases,
    run_fusion_variant,
    run_solo_variant,
    validate_deliberation_case,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def case() -> dict[str, object]:
    return {
        "id": "case-1",
        "category": "decision",
        "difficulty": 3,
        "prompt": "Choose a strategy.",
        "rubric": ["compare alternatives", "trace consequences"],
    }


class DeliberationBenchmarkTests(unittest.TestCase):
    def test_fixture_has_40_cases_and_expected_category_mix(self) -> None:
        cases = load_deliberation_cases(REPO_ROOT / "benchmarks" / "deliberation_cases.jsonl")
        self.assertEqual(len(cases), 40)
        counts = Counter(str(row["category"]) for row in cases)
        self.assertEqual(counts["decision"], 10)
        self.assertEqual(counts["root-cause"], 8)
        self.assertEqual(counts["open-ended"], 8)
        self.assertEqual(counts["architecture"], 6)
        self.assertEqual(counts["adversarial"], 4)
        self.assertEqual(counts["second-order"], 4)

    def test_case_validation_requires_rubric(self) -> None:
        invalid = dict(case())
        invalid["rubric"] = []
        self.assertTrue(validate_deliberation_case(invalid))

    def test_fusion_variant_switches_only_deliberation_mode(self) -> None:
        seen = []

        def fake_runner(args):
            seen.append(args)
            return {
                "draft": "final answer",
                "metrics": {"wall_latency_ms": 12, "calls": 7, "cost_usd": 0.01},
            }, 0

        current = run_fusion_variant(
            "question", label="fusion-current", preset="core", judge="claude", drafter="claude",
            deliberation=False, timeout=10, retries=0, scouts=4, branches=3, critics=2,
            fusion_runner=fake_runner,
        )
        deliberation = run_fusion_variant(
            "question", label="fusion-deliberation", preset="core", judge="claude", drafter="claude",
            deliberation=True, timeout=10, retries=0, scouts=4, branches=3, critics=2,
            fusion_runner=fake_runner,
        )
        self.assertEqual(current["answer"], "final answer")
        self.assertEqual(current["calls"], 7)
        self.assertEqual(seen[0].deliberation, "off")
        self.assertEqual(seen[1].deliberation, "on")
        self.assertEqual(seen[0].preset, seen[1].preset)
        self.assertEqual(deliberation["cost_usd"], 0.01)

    def test_solo_variant_is_exactly_one_call(self) -> None:
        def fake_dispatch(member, prompt, depth, config, apply_member_prompt):
            return ModelResult(
                label=member.label, backend=member.backend, kind=member.kind, model=member.model,
                ok=True, answer="solo answer", input_tokens=10, output_tokens=5, cost_usd=0.001,
            )

        result = run_solo_variant("question", member_spec="openai:test", timeout=10, retries=0, dispatcher=fake_dispatch)
        self.assertTrue(result["ok"])
        self.assertEqual(result["calls"], 1)
        self.assertEqual(result["answer"], "solo answer")

    def test_blind_evaluation_maps_scores_back_to_variant_labels(self) -> None:
        variants = [
            {"label": "solo", "answer": "solo answer", "ok": True},
            {"label": "fusion-current", "answer": "current answer", "ok": True},
            {"label": "fusion-deliberation", "answer": "deliberation answer", "ok": True},
        ]
        captured_prompts: list[str] = []

        def fake_dispatch(member, prompt, depth, config, apply_member_prompt):
            captured_prompts.append(prompt)
            blocks = re.findall(r"### Answer ([A-Z]+)\n(.*?)(?=\n\n### Answer |\Z)", prompt, flags=re.DOTALL)
            scores = []
            winner = None
            best_score = -1
            for label, answer in blocks:
                score = 5 if "deliberation answer" in answer else 4 if "current answer" in answer else 3
                row = {"label": label, **{axis: score for axis in DELIBERATION_EVAL_AXES}, "notes": "ok"}
                scores.append(row)
                if score > best_score:
                    best_score = score
                    winner = label
            payload = {"scores": scores, "ranking": [row["label"] for row in sorted(scores, key=lambda item: item["correctness"], reverse=True)], "winner": winner, "confidence": 0.9}
            return ModelResult(label=member.label, backend=member.backend, kind=member.kind, model=member.model, ok=True, answer=json.dumps(payload))

        evaluation = evaluate_case(case(), variants, evaluator_spec="openai:test", timeout=10, retries=0, dispatcher=fake_dispatch)
        self.assertTrue(evaluation["valid"])
        self.assertEqual(evaluation["winner"], "fusion-deliberation")
        self.assertEqual(evaluation["scores"]["fusion-current"]["robustness"], 4.0)
        self.assertNotIn("fusion-deliberation", captured_prompts[0])
        self.assertNotIn("fusion-current", captured_prompts[0])

    def test_eval_prompt_contains_rubric_but_not_source_labels(self) -> None:
        anonymous = [
            {"blind_label": "A", "source_label": "secret-model", "answer": "answer one"},
            {"blind_label": "B", "source_label": "other-model", "answer": "answer two"},
        ]
        prompt = deliberation_eval_prompt(case(), anonymous)
        self.assertIn("compare alternatives", prompt)
        self.assertNotIn("secret-model", prompt)
        self.assertNotIn("other-model", prompt)

    def test_aggregate_reports_quality_delta_and_operational_cost(self) -> None:
        axes_current = {axis: 3.0 for axis in DELIBERATION_EVAL_AXES}
        axes_delib = {axis: 4.0 for axis in DELIBERATION_EVAL_AXES}
        axes_solo = {axis: 2.0 for axis in DELIBERATION_EVAL_AXES}
        runs = [
            {
                "variants": [
                    {"label": "solo", "ok": True, "latency_ms": 10, "calls": 1, "cost_usd": 0.001},
                    {"label": "fusion-current", "ok": True, "latency_ms": 20, "calls": 8, "cost_usd": 0.01},
                    {"label": "fusion-deliberation", "ok": True, "latency_ms": 35, "calls": 17, "cost_usd": 0.02},
                ],
                "evaluation": {
                    "valid": True,
                    "winner": "fusion-deliberation",
                    "scores": {"solo": axes_solo, "fusion-current": axes_current, "fusion-deliberation": axes_delib},
                },
            }
        ]
        aggregate = aggregate_deliberation_runs(runs)
        self.assertEqual(aggregate["valid_evaluations"], 1)
        self.assertAlmostEqual(aggregate["deliberation_quality_delta"], 1.0)
        self.assertEqual(aggregate["variants"]["fusion-deliberation"]["win_rate"], 1.0)
        self.assertEqual(aggregate["variants"]["fusion-deliberation"]["avg_calls"], 17.0)
        self.assertEqual(aggregate["variants"]["fusion-current"]["avg_cost_usd"], 0.01)


if __name__ == "__main__":
    unittest.main()
