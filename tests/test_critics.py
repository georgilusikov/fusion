from __future__ import annotations

import json
import threading
import time
import unittest

from fusion_core.cli import build_parser
from fusion_core.config import DispatchConfig, Member, ModelResult
from fusion_core.critics import plan_critics, run_targeted_critics, validate_critique_payload
from fusion_core.deliberation import run_deliberation


def pool() -> dict[str, object]:
    return {
        "candidate_count": 2,
        "valid_scouts": 1,
        "candidates": [
            {
                "id": "C1",
                "type": "solution",
                "claim": "lower the price",
                "why_it_matters": "demand",
                "assumptions": [],
                "evidence_needed": [],
                "parent": None,
                "horizon": 0,
                "source_ids": ["S1"],
                "operators": ["alternative"],
            },
            {
                "id": "C2",
                "type": "consequence",
                "claim": "competitors may respond",
                "why_it_matters": "feedback",
                "assumptions": [],
                "evidence_needed": [],
                "parent": "C1",
                "horizon": 2,
                "source_ids": ["S1"],
                "operators": ["second_order"],
            },
        ],
    }


def critique_payload(target: str, severity: float) -> dict[str, object]:
    return {
        "summary": "stress test",
        "findings": [
            {
                "target": target,
                "objection": f"problem with {target}",
                "severity": severity,
                "hidden_assumptions": ["assumption"],
                "missing_evidence": ["evidence"],
            }
        ],
    }


def scout_payload() -> dict[str, object]:
    return {
        "summary": "scout",
        "candidates": [
            {
                "type": "solution",
                "claim": "new option",
                "why_it_matters": "useful",
                "assumptions": [],
                "evidence_needed": [],
                "parent": None,
                "horizon": 0,
            }
        ],
    }


class CriticTests(unittest.TestCase):
    def test_critic_planner_selects_causal_and_feasibility_lenses(self) -> None:
        keys = [item.key for item in plan_critics("market decision", pool(), limit=3)]
        self.assertIn("causal", keys)
        self.assertIn("feasibility", keys)
        self.assertEqual(len(keys), len(set(keys)))

    def test_critique_payload_validation(self) -> None:
        self.assertEqual(validate_critique_payload(critique_payload("C1", 0.8)), [])
        invalid = critique_payload("C1", 1.5)
        self.assertTrue(validate_critique_payload(invalid))

    def test_targeted_critics_run_concurrently_and_sort_by_severity(self) -> None:
        members = [
            Member("a", "api", "openai", "gpt", "neutral", "neutral"),
            Member("b", "api", "anthropic", "sonnet", "neutral", "neutral"),
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
            severity = 0.9 if member.label == "a" else 0.4
            target = "C1" if member.label == "a" else "C2"
            return ModelResult(
                label=member.label,
                backend=member.backend,
                kind=member.kind,
                model=member.model,
                ok=True,
                answer=json.dumps(critique_payload(target, severity)),
            )

        bundle, results = run_targeted_critics(
            "market decision",
            pool(),
            members,
            DispatchConfig(),
            "one-shot",
            count=2,
            dispatcher=fake_dispatch,
        )
        self.assertEqual(bundle["valid_critics"], 2)
        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(max_active, 2)
        self.assertGreaterEqual(bundle["findings"][0]["severity"], bundle["findings"][1]["severity"])
        self.assertIn("Targeted critique findings", bundle["context"])

    def test_deliberation_combines_pool_and_critic_context(self) -> None:
        members = [
            Member("a", "api", "openai", "gpt", "neutral", "neutral"),
            Member("b", "api", "anthropic", "sonnet", "neutral", "neutral"),
        ]

        def fake_dispatch(member, prompt, depth, config, apply_member_prompt):
            if "targeted critic" in prompt.lower():
                answer = json.dumps(critique_payload("C1", 0.7))
            else:
                answer = json.dumps(scout_payload())
            return ModelResult(
                label=member.label,
                backend=member.backend,
                kind=member.kind,
                model=member.model,
                ok=True,
                answer=answer,
            )

        bundle, results = run_deliberation(
            "question",
            members,
            {"parsed": {"coverage_gaps": [], "blind_spots": [], "contradictions": []}},
            DispatchConfig(),
            "one-shot",
            max_operators=1,
            critic_count=2,
            dispatcher=fake_dispatch,
        )
        self.assertEqual(bundle["pool"]["candidate_count"], 1)
        self.assertEqual(bundle["critique"]["valid_critics"], 2)
        self.assertIn("Deliberation candidate pool", bundle["donor_context"])
        self.assertIn("Targeted critique findings", bundle["donor_context"])
        self.assertEqual(len(results), 3)

    def test_cli_critic_count_default_and_override(self) -> None:
        parser = build_parser()
        default = parser.parse_args(["question", "--dry-run"])
        custom = parser.parse_args(["question", "--deliberation-critics", "4", "--dry-run"])
        self.assertEqual(default.deliberation_critics, 2)
        self.assertEqual(custom.deliberation_critics, 4)


if __name__ == "__main__":
    unittest.main()
