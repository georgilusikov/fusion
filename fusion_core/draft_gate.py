"""Anti-regression gate comparing a final draft against the best source answer."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable, Mapping

from .config import DispatchConfig, Member, ModelResult, ROLES
from .dispatch import dispatch
from .judge import extract_json

VALID_WINNERS = {"draft", "best", "patched"}


def draft_gate_prompt(user_prompt: str, best: ModelResult, draft: ModelResult) -> str:
    return (
        "You are a regression gate. Compare the final draft against the best source answer. "
        "Prefer the draft only if it is at least as correct and useful as the best source. "
        "If the draft loses but can be fixed safely, return winner=patched and patched_answer. "
        "Return JSON only.\n\n"
        "Required JSON keys: winner, draft_score, best_score, reason, patched_answer. "
        "winner must be draft, best, or patched. Scores are 0..5.\n\n"
        f"## Original prompt\n{user_prompt}\n\n"
        f"## Best source answer ({best.label})\n{best.answer}\n\n"
        f"## Final draft\n{draft.answer}\n"
    )


def validate_draft_gate_payload(payload: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["output is not a JSON object"]
    errors: list[str] = []
    winner = payload.get("winner")
    if winner not in VALID_WINNERS:
        errors.append("winner must be draft, best, or patched")
    for key in ("draft_score", "best_score"):
        value = payload.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 5:
            errors.append(f"{key} must be a number from 0 to 5")
    if not isinstance(payload.get("reason"), str):
        errors.append("reason must be a string")
    if "patched_answer" in payload and not isinstance(payload.get("patched_answer"), str):
        errors.append("patched_answer must be a string")
    return errors


def final_answer_from_gate(payload: Mapping[str, Any] | None, best_answer: str, draft_answer: str) -> tuple[str, bool, str]:
    errors = validate_draft_gate_payload(payload)
    if errors:
        return best_answer, True, "invalid-gate-output"
    assert payload is not None
    winner = str(payload.get("winner"))
    if winner == "draft":
        return draft_answer, False, "draft"
    if winner == "patched":
        patched = str(payload.get("patched_answer") or "").strip()
        if patched:
            return patched, patched != draft_answer, "patched"
    return best_answer, best_answer != draft_answer, "best"


DraftGateDispatcher = Callable[[Member, str, str, DispatchConfig, bool], ModelResult]


def run_draft_gate(
    gate_member: Member,
    user_prompt: str,
    best: ModelResult,
    draft: ModelResult,
    config: DispatchConfig,
    dispatcher: DraftGateDispatcher = dispatch,
) -> dict[str, Any]:
    neutral = dataclasses.replace(gate_member, role_key="neutral", role_text=ROLES["neutral"], depth=None)
    result = dispatcher(neutral, draft_gate_prompt(user_prompt, best, draft), "one-shot", config, False)
    parsed = extract_json(result.answer) if result.ok else None
    validation_errors = validate_draft_gate_payload(parsed)
    final_answer, replaced, winner = final_answer_from_gate(parsed, best.answer, draft.answer)
    return {
        "backend": gate_member.backend,
        "model": gate_member.model,
        "raw": result.answer,
        "parsed": parsed if not validation_errors else None,
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "winner": winner,
        "replaced_draft": replaced,
        "final_answer": final_answer,
        "result": result.to_dict(),
    }
