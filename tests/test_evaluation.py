from __future__ import annotations

import unittest

from fusion_core.evaluation import anonymize_candidates, blind_label, median_axis_scores, rematch_verdict, winner_from_scores


class EvaluationTests(unittest.TestCase):
    def test_blind_label_extends_past_alphabet(self) -> None:
        self.assertEqual(blind_label(0), "A")
        self.assertEqual(blind_label(25), "Z")
        self.assertEqual(blind_label(26), "AA")

    def test_anonymize_candidates_is_stable(self) -> None:
        candidates = [
            {"label": "fusion-pro", "answer": "answer one"},
            {"label": "solo-claude", "answer": "answer two"},
        ]
        first = anonymize_candidates(candidates, seed="case")
        second = anonymize_candidates(list(reversed(candidates)), seed="case")
        self.assertEqual(first, second)
        self.assertEqual({item["blind_label"] for item in first}, {"A", "B"})

    def test_median_scores_and_winner(self) -> None:
        payloads = [
            {"scores": [{"label": "A", "correctness": 5, "depth": 4, "coverage": 4, "actionability": 5}]},
            {"scores": [{"label": "A", "correctness": 3, "depth": 2, "coverage": 4, "actionability": 3}]},
            {"scores": [{"label": "B", "correctness": 2, "depth": 2, "coverage": 2, "actionability": 2}]},
        ]
        medians = median_axis_scores(payloads)
        self.assertEqual(medians["A"]["correctness"], 4.0)
        self.assertEqual(winner_from_scores(medians), "A")

    def test_baseline_match_is_actionable(self) -> None:
        verdict = rematch_verdict({"fusion-pro": 4.2, "solo-claude": 4.2})
        self.assertEqual(verdict["status"], "baseline-matches-or-wins")
        self.assertIn("fix synthesis", verdict["action"])
