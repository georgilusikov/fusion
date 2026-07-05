from __future__ import annotations

import unittest

from fusion_core.experiments import debate_prompt, language_axis_prompt, outcome_event, validate_stop_rule


class ExperimentHelperTests(unittest.TestCase):
    def test_debate_prompt_includes_answers(self) -> None:
        prompt = debate_prompt("question", [{"label": "a", "answer": "first"}, {"label": "b", "answer": "second"}])
        self.assertIn("question", prompt)
        self.assertIn("first", prompt)
        self.assertIn("second", prompt)

    def test_language_axis_prompt_rejects_unknown_language(self) -> None:
        self.assertIn("Answer in Chinese", language_axis_prompt("prompt", "zh"))
        with self.assertRaises(ValueError):
            language_axis_prompt("prompt", "xx")

    def test_outcome_event_contract(self) -> None:
        event = outcome_event(case_id="case", answer_id="run", affected_action=True, timestamp="now")
        self.assertEqual(event["kind"], "fusion_outcome_event")
        self.assertTrue(event["affected_action"])

    def test_stop_rule_requires_known_case(self) -> None:
        self.assertEqual(validate_stop_rule({"feature": "x", "case_ids": ["a"]}, {"a"}), [])
        self.assertTrue(validate_stop_rule({"feature": "x", "case_ids": ["missing"]}, {"a"}))


if __name__ == "__main__":
    unittest.main()
