from __future__ import annotations

import json
import re
import unittest

from fusion_core.benchmark_evaluator_panel import EVAL_AXES, evaluate_case_panel
from fusion_core.config import ModelResult
from fusion_core.multi_eval_benchmark import parse_evaluator_specs


def _score_payload(prompt: str, *, invalid: bool = False) -> str:
    if invalid:
        return "not json"
    matches = re.findall(r"### Answer ([A-Z]+)\n(.*?)(?=\n\n### Answer |\Z)", prompt, flags=re.S)
    scores = []
    winner = None
    best_score = -1
    for label, answer in matches:
        base = 5 if "BEST" in answer else (3 if "MID" in answer else 1)
        row = {"label": label, "notes": "fixture"}
        for axis in EVAL_AXES:
            row[axis] = base
        scores.append(row)
        if base > best_score:
            best_score = base
            winner = label
    return json.dumps({
        "scores": scores,
        "ranking": [row["label"] for row in sorted(scores, key=lambda item: item["correctness"], reverse=True)],
        "winner": winner,
        "confidence": 0.8,
    })


class EvaluatorPanelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = {
            "id": "case-1",
            "prompt": "Choose the strongest answer",
            "rubric": ["Prefer the actually stronger answer"],
        }
        self.variants = [
            {"label": "solo", "answer": "LOW", "ok": True},
            {"label": "fusion-current", "answer": "MID", "ok": True},
            {"label": "fusion-deliberation", "answer": "BEST", "ok": True},
        ]

    def test_panel_maps_shuffled_blind_labels_back_to_sources(self) -> None:
        def fake_dispatch(member, prompt, depth, config, apply_member_prompt):
            return ModelResult(
                label=member.label,
                backend=member.backend,
                kind=member.kind,
                model=member.model,
                ok=True,
                answer=_score_payload(prompt),
                cost_usd=0.01,
            )

        result = evaluate_case_panel(
            self.case,
            self.variants,
            evaluator_specs=["or:a/model", "or:b/model", "or:c/model"],
            timeout=10,
            retries=0,
            dispatcher=fake_dispatch,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["valid_evaluators"], 3)
        self.assertEqual(result["winner"], "fusion-deliberation")
        self.assertEqual(result["scores"]["fusion-deliberation"]["correctness"], 5.0)
        self.assertEqual(result["scores"]["fusion-current"]["correctness"], 3.0)
        self.assertEqual(result["scores"]["solo"]["correctness"], 1.0)
        self.assertEqual(result["winner_votes"]["fusion-deliberation"], 3)
        self.assertEqual(result["agreement"], 1.0)
        self.assertAlmostEqual(result["evaluator_cost_usd"], 0.03)

    def test_strict_majority_allows_one_invalid_evaluator(self) -> None:
        def fake_dispatch(member, prompt, depth, config, apply_member_prompt):
            invalid = str(member.model or "").endswith("/broken")
            return ModelResult(
                label=member.label,
                backend=member.backend,
                kind=member.kind,
                model=member.model,
                ok=True,
                answer=_score_payload(prompt, invalid=invalid),
            )

        result = evaluate_case_panel(
            self.case,
            self.variants,
            evaluator_specs=["or:a/good", "or:b/broken", "or:c/good"],
            timeout=10,
            retries=0,
            dispatcher=fake_dispatch,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["valid_evaluators"], 2)
        self.assertEqual(result["required_evaluators"], 2)
        self.assertEqual(result["winner"], "fusion-deliberation")

    def test_panel_fails_without_required_evaluator_majority(self) -> None:
        def fake_dispatch(member, prompt, depth, config, apply_member_prompt):
            valid = str(member.model or "").endswith("/good")
            return ModelResult(
                label=member.label,
                backend=member.backend,
                kind=member.kind,
                model=member.model,
                ok=True,
                answer=_score_payload(prompt, invalid=not valid),
            )

        result = evaluate_case_panel(
            self.case,
            self.variants,
            evaluator_specs=["or:a/good", "or:b/bad", "or:c/bad"],
            timeout=10,
            retries=0,
            dispatcher=fake_dispatch,
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["valid_evaluators"], 1)
        self.assertEqual(result["required_evaluators"], 2)

    def test_parse_evaluator_specs_deduplicates(self) -> None:
        self.assertEqual(
            parse_evaluator_specs("or:a/x, or:b/y,or:a/x"),
            ["or:a/x", "or:b/y"],
        )

    def test_fewer_than_two_valid_answers_is_invalid(self) -> None:
        variants = [
            {"label": "solo", "answer": "only", "ok": True},
            {"label": "fusion-current", "answer": "", "ok": False},
        ]
        result = evaluate_case_panel(
            self.case,
            variants,
            evaluator_specs=["or:a/model"],
            timeout=10,
            retries=0,
            dispatcher=lambda *args, **kwargs: None,
        )
        self.assertFalse(result["valid"])
        self.assertIn("fewer than two", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
