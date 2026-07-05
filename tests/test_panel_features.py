from __future__ import annotations

import unittest

from fusion_core.judge_panel import aggregate_judge_payloads
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


if __name__ == "__main__":
    unittest.main()
