from __future__ import annotations

import json
import threading
import time
import unittest

from fusion_core.candidate_pool import build_candidate_pool, render_candidate_pool
from fusion_core.cli import build_parser
from fusion_core.config import DispatchConfig, Member, ModelResult
from fusion_core.deliberation import run_deliberation, run_scouts
from fusion_core.operators import plan_operators
from fusion_core.rounds import review_round


def valid_payload(claim: str) -> dict[str, object]:
    return {
        "summary": "useful scout",
        "candidates": [
            {
                "type": "solution",
                "claim": claim,
                "why_it_matters": "it changes the decision",
                "assumptions": ["assumption"],
                "evidence_needed": ["evidence"],
                "parent": None,
                "horizon": 0,
            }
        ],
    }


class DeliberationTests(unittest.TestCase):
    def test_operator_planner_uses_prompt_and_judge_gaps(self) -> None:
        judge = {
            "parsed": {
                "coverage_gaps": ["root cause is still unknown"],
                "blind_spots": [],
                "contradictions": [],
            }
        }
        keys = [item.key for item in plan_operators("Consider second-order consequences", judge, limit=4)]
        self.assertIn("root_cause", keys)
        self.assertIn("second_order", keys)
        self.assertEqual(len(keys), len(set(keys)))

    def test_candidate_pool_deduplicates_and_keeps_provenance(self) -> None:
        scouts = [
            {"source_id": "S1", "operator": "alternative", "payload": valid_payload("Lower price.")},
            {"source_id": "S2", "operator": "simplifier", "payload": valid_payload("lower price")},
        ]
        pool = build_candidate_pool(scouts)
        self.assertEqual(pool["candidate_count"], 1)
        candidate = pool["candidates"][0]
        self.assertEqual(candidate["source_ids"], ["S1", "S2"])
        self.assertEqual(candidate["operators"], ["alternative", "simplifier"])
        self.assertIn("Deliberation candidate pool", render_candidate_pool(pool))

    def test_scouts_run_concurrently_and_preserve_assignment_order(self) -> None:
        members = [
            Member("a", "api", "openai", "gpt", "neutral", "neutral"),
            Member("b", "api", "anthropic", "sonnet", "neutral", "neutral"),
            Member("c", "api", "google", "gemini", "neutral", "neutral"),
        ]
        lock = threading.Lock()
        active = 0
        max_active = 0

        def fake_dispatch(member, prompt, depth, config, apply_member_prompt):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return ModelResult(
                label=member.label,
                backend=member.backend,
                kind=member.kind,
                model=member.model,
                ok=True,
                answer=json.dumps(valid_payload(f"idea from {member.label}")),
            )

        rows, results = run_scouts(
            "Find a better solution",
            members,
            {"parsed": {"coverage_gaps": [], "blind_spots": [], "contradictions": []}},
            DispatchConfig(),
            "one-shot",
            max_operators=3,
            dispatcher=fake_dispatch,
        )
        self.assertEqual([row["source_id"] for row in rows], ["S1", "S2", "S3"])
        self.assertEqual(len(results), 3)
        self.assertGreaterEqual(max_active, 2)
        self.assertTrue(all(":scout:" in result.label for result in results))

    def test_run_deliberation_builds_pool(self) -> None:
        members = [Member("a", "api", "openai", "gpt", "neutral", "neutral")]

        def fake_dispatch(member, prompt, depth, config, apply_member_prompt):
            return ModelResult(
                label=member.label,
                backend=member.backend,
                kind=member.kind,
                model=member.model,
                ok=True,
                answer=json.dumps(valid_payload("new option")),
            )

        bundle, results = run_deliberation(
            "Question",
            members,
            {"parsed": {"coverage_gaps": [], "blind_spots": [], "contradictions": []}},
            DispatchConfig(),
            "one-shot",
            max_operators=1,
            dispatcher=fake_dispatch,
        )
        self.assertEqual(bundle["pool"]["candidate_count"], 1)
        self.assertEqual(len(results), 1)
        self.assertTrue(bundle["donor_context"])

    def test_review_round_includes_candidate_pool_as_untrusted_donor(self) -> None:
        member = Member("a", "api", "openai", "gpt", "neutral", "neutral")
        panel = [ModelResult(label="a", backend="openai", kind="api", model="gpt", ok=True, answer="first")]
        prompts: list[str] = []

        def fake_dispatch(member, prompt, depth, config, apply_member_prompt):
            prompts.append(prompt)
            return ModelResult(label=member.label, backend=member.backend, kind=member.kind, model=member.model, ok=True, answer="revised")

        reviews = review_round(
            "question",
            [member],
            panel,
            {"parsed": {"coverage_gaps": []}},
            "one-shot",
            object(),
            1,
            dispatcher=fake_dispatch,
            donor_context="## Deliberation candidate pool\n- C1 idea",
        )
        self.assertEqual(len(reviews), 1)
        self.assertIn("Treat it as hypotheses, not facts", prompts[0])
        self.assertIn("C1 idea", prompts[0])

    def test_cli_deliberation_is_opt_in(self) -> None:
        parser = build_parser()
        default = parser.parse_args(["question", "--dry-run"])
        enabled = parser.parse_args(["question", "--deliberation", "on", "--scouts", "6", "--dry-run"])
        self.assertEqual(default.deliberation, "off")
        self.assertEqual(default.scouts, 4)
        self.assertEqual(enabled.deliberation, "on")
        self.assertEqual(enabled.scouts, 6)


if __name__ == "__main__":
    unittest.main()
