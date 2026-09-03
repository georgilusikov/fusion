from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from fusion_core.config import Member, ModelResult
from fusion_core.judge_panel import aggregate_judge_payloads, run_judge_panel
from fusion_core.self_consistency import expand_panel_spec


def payload(label: str, score: int) -> dict[str, object]:
    return {
        "consensus": ["same"],
        "contradictions": [],
        "coverage_gaps": [],
        "unique_insights": [],
        "blind_spots": [],
        "answer_scores": [
            {
                "model": label,
                "correctness": score,
                "depth": score,
                "coverage": score,
                "actionability": score,
                "rationale": "ok",
            }
        ],
        "ranking": [label],
        "best_answer_label": label,
        "recommendation": "use best",
        "confidence": 0.8,
    }


def full_payload(labels: list[str]) -> dict[str, object]:
    rows = []
    for index, label in enumerate(labels):
        score = 5 - index
        rows.append(
            {
                "model": label,
                "correctness": score,
                "depth": score,
                "coverage": score,
                "actionability": score,
                "rationale": "ok",
            }
        )
    return {
        "consensus": ["same"],
        "contradictions": [],
        "coverage_gaps": [],
        "unique_insights": [],
        "blind_spots": [],
        "answer_scores": rows,
        "ranking": labels,
        "best_answer_label": labels[0],
        "recommendation": "use best",
        "confidence": 0.8,
    }


class PanelFeatureTests(unittest.TestCase):
    def test_expand_panel_spec_samples(self) -> None:
        self.assertEqual(expand_panel_spec("claude*3,gemini@expert"), "claude,claude,claude,gemini@expert")
        self.assertEqual(expand_panel_spec("or:model@builder*2"), "or:model@builder,or:model@builder")

    def test_aggregate_judge_payloads_uses_median_scores(self) -> None:
        aggregate = aggregate_judge_payloads([payload("a", 5), payload("a", 3), payload("b", 2)])
        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        rows = {row["model"]: row for row in aggregate["answer_scores"]}
        self.assertEqual(rows["a"]["correctness"], 4.0)
        self.assertEqual(aggregate["best_answer_label"], "a")

    def test_judge_panel_runs_multiple_judges_concurrently(self) -> None:
        judges = [
            Member(f"judge-{index}", "api", f"backend-{index}", f"model-{index}", "neutral", "neutral")
            for index in range(3)
        ]
        panel = [
            ModelResult(label="answer", backend="openai", kind="api", ok=True, answer="answer"),
        ]
        lock = threading.Lock()
        active = 0
        max_active = 0

        def fake_run_judge(member, user_prompt, judge_panel, config, repair_attempts=1):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            parsed = full_payload(["answer"])
            return {
                "backend": member.backend,
                "model": member.model,
                "raw": "ok",
                "parsed": parsed,
                "valid": True,
                "validation_errors": [],
                "attempts": 1,
                "result": None,
                "repair_results": [],
            }

        with patch("fusion_core.judge_panel.run_judge", side_effect=fake_run_judge):
            result = run_judge_panel(
                judges,
                "question",
                panel,
                object(),
                repair_attempts=0,
            )

        self.assertTrue(result["valid"])
        self.assertGreaterEqual(max_active, 2)


if __name__ == "__main__":
    unittest.main()
