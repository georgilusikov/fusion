from __future__ import annotations

import unittest

from fusion_core.config import Member, ModelResult
from fusion_core.rounds import (
    choose_reviewers,
    combine_unique_results,
    escalation_reasons,
    member_for_result,
    renamed_result,
    review_round,
)


def panel_item(
    label: str,
    backend: str,
    *,
    ok: bool = True,
    answer: str = "answer",
    confidence: float | None = None,
    model: str | None = None,
) -> ModelResult:
    return ModelResult(
        label=label,
        backend=backend,
        kind="api",
        model=model,
        ok=ok,
        answer=answer,
        confidence=confidence,
    )


class RoundsTests(unittest.TestCase):
    def test_escalation_reasons_flags_low_confidence_and_gaps(self) -> None:
        panel = [
            panel_item("a", "x", confidence=0.9),
            panel_item("b", "y", confidence=0.8),
        ]
        judge = {
            "valid": True,
            "parsed": {
                "confidence": 0.4,
                "contradictions": [{"point": "p1", "sides": "a"}, {"point": "p2", "sides": "b"}],
                "coverage_gaps": ["gap1", "gap2"],
            },
        }
        reasons = escalation_reasons(panel, judge)
        self.assertIn("low-judge-confidence", reasons)
        self.assertIn("substantial-contradictions", reasons)
        self.assertIn("substantial-coverage-gaps", reasons)

    def test_escalation_stops_after_invalid_judge(self) -> None:
        panel = [panel_item("only", "x")]
        reasons = escalation_reasons(panel, {"valid": False})
        self.assertEqual(reasons, ["fewer-than-two-successful-members", "invalid-judge-output"])

    def test_choose_reviewers_prefers_backend_diversity(self) -> None:
        panel = [
            panel_item("a", "openai", confidence=0.9),
            panel_item("b", "openai", confidence=0.8),
            panel_item("c", "anthropic", confidence=0.7),
        ]
        chosen = choose_reviewers(panel, 2)
        self.assertEqual([item.label for item in chosen], ["a", "c"])

    def test_combine_unique_results_renames_label_collisions(self) -> None:
        first = [panel_item("codex", "codex", answer="first")]
        second = [panel_item("codex", "codex", answer="second")]
        combined = combine_unique_results(first, second)
        self.assertEqual(combined[0].label, "codex")
        self.assertEqual(combined[1].label, "codex:escalated")

    def test_review_round_revises_selected_members(self) -> None:
        members = [
            Member("a", "api", "openai", "gpt", "neutral", "neutral"),
            Member("b", "api", "anthropic", "sonnet", "neutral", "neutral"),
        ]
        panel = [
            panel_item("a", "openai", answer="first-a", confidence=0.9, model="gpt"),
            panel_item("b", "anthropic", answer="first-b", confidence=0.8, model="sonnet"),
        ]
        prompts: list[str] = []

        def fake_dispatch(member, prompt, depth, config, apply_member_prompt):
            prompts.append(prompt)
            return panel_item(member.label, member.backend, answer=f"revised-{member.label}", model=member.model)

        reviews = review_round(
            "question",
            members,
            panel,
            {"parsed": {"consensus": ["x"], "recommendation": "revise"}},
            "one-shot",
            object(),
            2,
            dispatcher=fake_dispatch,
        )
        self.assertEqual(len(reviews), 2)
        self.assertTrue(all(item.label.endswith(":revision") for item in reviews))
        self.assertIn("Judge analysis", prompts[0])

    def test_member_for_result_matches_label_then_backend(self) -> None:
        members = [Member("a", "api", "openai", "gpt", "neutral", "neutral")]
        result = panel_item("a", "openai", model="gpt")
        self.assertIs(member_for_result(result, members), members[0])
        self.assertIs(member_for_result(renamed_result(result, ":revision"), members), members[0])


if __name__ == "__main__":
    unittest.main()
