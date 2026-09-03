from __future__ import annotations

import json
import re
import threading
import time
import unittest

from fusion_core.branching import run_branch_expansions, select_branches, validate_branch_payload
from fusion_core.cli import build_parser
from fusion_core.config import DispatchConfig, Member, ModelResult
from fusion_core.deliberation import run_deliberation


def candidate_pool() -> dict[str, object]:
    return {
        "candidate_count": 4,
        "valid_scouts": 3,
        "candidates": [
            {
                "id": "C1", "type": "solution", "claim": "direct option", "why_it_matters": "impact",
                "assumptions": [], "evidence_needed": [], "parent": None, "horizon": 0,
                "source_ids": ["S1", "S2"], "operators": ["baseline", "alternative"],
            },
            {
                "id": "C2", "type": "solution", "claim": "unusual option", "why_it_matters": "novel",
                "assumptions": [], "evidence_needed": [], "parent": None, "horizon": 0,
                "source_ids": ["S3"], "operators": ["inversion"],
            },
            {
                "id": "C3", "type": "cause", "claim": "underlying cause", "why_it_matters": "leverage",
                "assumptions": [], "evidence_needed": [], "parent": None, "horizon": 0,
                "source_ids": ["S1", "S2"], "operators": ["root_cause"],
            },
            {
                "id": "C4", "type": "risk", "claim": "failure risk", "why_it_matters": "downside",
                "assumptions": [], "evidence_needed": [], "parent": None, "horizon": 0,
                "source_ids": ["S2"], "operators": ["premortem"],
            },
        ],
    }


def branch_payload(target: str) -> dict[str, object]:
    return {
        "target": target,
        "thesis": f"expanded {target}",
        "required_conditions": ["condition"],
        "direct_effects": ["direct"],
        "second_order_effects": ["second"],
        "third_order_effects": ["third"],
        "failure_conditions": ["failure"],
        "disconfirming_evidence": ["counterevidence"],
    }


def scout_payload() -> dict[str, object]:
    return {
        "summary": "scout",
        "candidates": [
            {
                "type": "solution",
                "claim": "candidate",
                "why_it_matters": "useful",
                "assumptions": [],
                "evidence_needed": [],
                "parent": None,
                "horizon": 0,
            }
        ],
    }


def critique_payload() -> dict[str, object]:
    return {
        "summary": "critic",
        "findings": [
            {
                "target": "C1",
                "objection": "weak assumption",
                "severity": 0.7,
                "hidden_assumptions": ["hidden"],
                "missing_evidence": ["test"],
            }
        ],
    }


def target_from_prompt(prompt: str) -> str:
    match = re.search(r'"id"\s*:\s*"(C\d+)"', prompt)
    if not match:
        raise AssertionError("branch target id not found in prompt")
    return match.group(1)


class BranchingTests(unittest.TestCase):
    def test_select_branches_keeps_supported_core_and_diversity_slot(self) -> None:
        selected = select_branches(candidate_pool(), limit=3)
        ids = [row["id"] for row in selected]
        self.assertEqual(len(ids), 3)
        self.assertIn("C1", ids)
        self.assertIn("C3", ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_branch_payload_validation_checks_target_and_effect_arrays(self) -> None:
        self.assertEqual(validate_branch_payload(branch_payload("C1"), expected_target="C1"), [])
        self.assertTrue(validate_branch_payload(branch_payload("C2"), expected_target="C1"))
        invalid = branch_payload("C1")
        invalid["second_order_effects"] = "not-array"
        self.assertTrue(validate_branch_payload(invalid, expected_target="C1"))

    def test_branch_expansions_run_concurrently_and_preserve_selected_order(self) -> None:
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
            target = target_from_prompt(prompt)
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
                answer=json.dumps(branch_payload(target)),
            )

        bundle, results = run_branch_expansions(
            "question",
            candidate_pool(),
            members,
            DispatchConfig(),
            "one-shot",
            count=3,
            dispatcher=fake_dispatch,
        )
        self.assertEqual(bundle["valid_expansions"], 3)
        self.assertEqual([row["target"] for row in bundle["expansions"]], bundle["selected"])
        self.assertEqual(len(results), 3)
        self.assertGreaterEqual(max_active, 2)
        self.assertIn("second-order", bundle["context"])

    def test_deliberation_orders_expansion_before_critique(self) -> None:
        members = [
            Member("a", "api", "openai", "gpt", "neutral", "neutral"),
            Member("b", "api", "anthropic", "sonnet", "neutral", "neutral"),
        ]
        critic_prompts: list[str] = []

        def fake_dispatch(member, prompt, depth, config, apply_member_prompt):
            lowered = prompt.lower()
            if "branch expander" in lowered:
                answer = json.dumps(branch_payload(target_from_prompt(prompt)))
            elif "targeted critic" in lowered:
                critic_prompts.append(prompt)
                answer = json.dumps(critique_payload())
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
            branch_count=1,
            critic_count=1,
            dispatcher=fake_dispatch,
        )
        self.assertEqual(bundle["branches"]["valid_expansions"], 1)
        self.assertEqual(bundle["critique"]["valid_critics"], 1)
        self.assertIn("Bounded branch expansions", bundle["donor_context"])
        self.assertIn("Targeted critique findings", bundle["donor_context"])
        self.assertEqual(len(results), 3)
        self.assertTrue(critic_prompts)
        self.assertIn("Bounded branch expansions", critic_prompts[0])

    def test_cli_branch_expansion_default_and_override(self) -> None:
        parser = build_parser()
        default = parser.parse_args(["question", "--dry-run"])
        custom = parser.parse_args(["question", "--branch-expansions", "4", "--dry-run"])
        self.assertEqual(default.branch_expansions, 3)
        self.assertEqual(custom.branch_expansions, 4)


if __name__ == "__main__":
    unittest.main()
