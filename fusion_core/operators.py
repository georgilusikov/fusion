"""Deliberation operators and deterministic operator planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    key: str
    title: str
    instruction: str


OPERATORS: dict[str, OperatorSpec] = {
    "baseline": OperatorSpec(
        "baseline",
        "Baseline",
        "Find the most reasonable direct solution under the stated constraints.",
    ),
    "root_cause": OperatorSpec(
        "root_cause",
        "Root cause",
        "Look beneath symptoms. Generate plausible underlying causes and leverage points.",
    ),
    "first_principles": OperatorSpec(
        "first_principles",
        "First principles",
        "Rebuild the problem from goals, constraints, invariants, and necessary conditions.",
    ),
    "alternative": OperatorSpec(
        "alternative",
        "Alternative",
        "Find a materially different viable approach, not a cosmetic variation.",
    ),
    "inversion": OperatorSpec(
        "inversion",
        "Inversion",
        "Ask how to avoid, remove, or reverse the problem instead of solving it conventionally.",
    ),
    "stakeholder": OperatorSpec(
        "stakeholder",
        "Stakeholder response",
        "Model how affected actors could react, adapt, resist, or exploit the proposed actions.",
    ),
    "second_order": OperatorSpec(
        "second_order",
        "Second-order effects",
        "Trace direct effects into second- and third-order consequences, including feedback loops.",
    ),
    "premortem": OperatorSpec(
        "premortem",
        "Premortem",
        "Assume the plan failed. Reconstruct the most plausible causes and early warning signs.",
    ),
    "falsifier": OperatorSpec(
        "falsifier",
        "Falsifier",
        "Find hidden assumptions and conditions that would make the leading answer wrong.",
    ),
    "evidence": OperatorSpec(
        "evidence",
        "Evidence",
        "Identify the evidence or tests that would most change the decision between alternatives.",
    ),
    "simplifier": OperatorSpec(
        "simplifier",
        "Simplifier",
        "Find a substantially simpler or cheaper path that preserves most of the desired outcome.",
    ),
}


_DEFAULT_ORDER = ("alternative", "root_cause", "second_order", "simplifier", "falsifier", "evidence")

_HINTS: dict[str, tuple[str, ...]] = {
    "root_cause": ("root cause", "cause", "why", "причин", "почему", "симптом"),
    "first_principles": ("constraint", "invariant", "first principle", "огранич", "инвариант"),
    "alternative": ("alternative", "option", "вариант", "альтернатив"),
    "stakeholder": ("stakeholder", "customer", "competitor", "user", "клиент", "конкурент", "пользоват"),
    "second_order": ("consequence", "second-order", "feedback", "последств", "второго порядка"),
    "premortem": ("failure", "fail", "premortem", "провал", "ошиб"),
    "falsifier": ("assumption", "contradiction", "wrong", "предполож", "противореч"),
    "evidence": ("evidence", "verify", "test", "data", "доказ", "провер", "данн"),
    "simplifier": ("simple", "cheap", "cost", "быстр", "проще", "дешев"),
}


def _judge_context(judge: Mapping[str, Any] | None) -> str:
    if not isinstance(judge, Mapping):
        return ""
    parsed = judge.get("parsed")
    if not isinstance(parsed, Mapping):
        return ""
    compact = {
        "coverage_gaps": parsed.get("coverage_gaps") or [],
        "blind_spots": parsed.get("blind_spots") or [],
        "contradictions": parsed.get("contradictions") or [],
    }
    return json.dumps(compact, ensure_ascii=False)


def plan_operators(
    prompt: str,
    judge: Mapping[str, Any] | None = None,
    *,
    limit: int = 4,
) -> list[OperatorSpec]:
    """Select a small diverse operator set without another model call."""
    if limit <= 0:
        return []
    text = f"{prompt}\n{_judge_context(judge)}".casefold()
    selected: list[str] = []

    def add(key: str) -> None:
        if key in OPERATORS and key not in selected and len(selected) < limit:
            selected.append(key)

    for key, hints in _HINTS.items():
        if any(hint.casefold() in text for hint in hints):
            add(key)

    for key in _DEFAULT_ORDER:
        add(key)

    return [OPERATORS[key] for key in selected[:limit]]
