from __future__ import annotations

import json
import unittest

from scripts import fusion


class DraftGateTests(unittest.TestCase):
    def test_empty_payload_uses_best(self) -> None:
        answer, replaced, winner = fusion.final_answer_from_gate(None, "best", "draft")
        self.assertEqual(answer, "best")
        self.assertTrue(replaced)
        self.assertEqual(winner, "invalid-gate-output")

    def test_patch_payload_uses_patch(self) -> None:
        payload = {
            "winner": "patched",
            "draft_score": 3,
            "best_score": 4,
            "reason": "use the edited answer",
            "patched_answer": "patched final",
        }
        answer, replaced, winner = fusion.final_answer_from_gate(payload, "best", "draft")
        self.assertEqual(answer, "patched final")
        self.assertTrue(replaced)
        self.assertEqual(winner, "patched")

    def test_run_gate_records_best(self) -> None:
        gate_member = fusion.parse_member("claude", True, {})
        best = fusion.ModelResult(label="best", backend="x", kind="api", ok=True, answer="best answer")
        draft = fusion.ModelResult(label="draft", backend="x", kind="api", ok=True, answer="draft answer")
        payload = {
            "winner": "best",
            "draft_score": 2,
            "best_score": 5,
            "reason": "best is stronger",
            "patched_answer": "",
        }

        def fake_dispatch(member, prompt, depth, config, apply_member_prompt):
            return fusion.ModelResult(
                label=member.label,
                backend=member.backend,
                kind=member.kind,
                model=member.model,
                ok=True,
                answer=json.dumps(payload),
            )

        result = fusion.run_draft_gate(
            gate_member,
            "question",
            best,
            draft,
            fusion.DispatchConfig(),
            dispatcher=fake_dispatch,
        )
        self.assertTrue(result["valid"])
        self.assertTrue(result["replaced_draft"])
        self.assertEqual(result["final_answer"], "best answer")
