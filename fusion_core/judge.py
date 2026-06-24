"""Judge JSON extraction, validation, and schema repair loop."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable, Mapping, Sequence

from .config import (
    DEFAULT_REPAIR_ATTEMPTS, JUDGE_INSTRUCTION, JUDGE_SCHEMA, ROLES,
    DispatchConfig, Member, ModelResult,
)
from .dispatch import dispatch
from .routing import successful_results

# --- judge schema and repair loop -----------------------------------------

def extract_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        stripped = parts[1] if len(parts) > 1 else stripped
        if stripped.lstrip().startswith("json"):
            stripped = stripped.lstrip()[4:]
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def validate_judge_payload(payload: Mapping[str, Any] | None) -> list[str]:
    """Validate the judge output against JUDGE_SCHEMA without a runtime dependency."""
    if payload is None:
        return ["output is not a JSON object"]
    required = set(JUDGE_SCHEMA["required"])
    errors: list[str] = []
    missing = required - set(payload)
    if missing:
        errors.append(f"missing required keys: {', '.join(sorted(missing))}")
    allowed = set(JUDGE_SCHEMA["properties"])
    extra = set(payload) - allowed
    if extra:
        errors.append(f"unexpected keys: {', '.join(sorted(extra))}")

    for key in ("consensus", "coverage_gaps", "blind_spots"):
        if key in payload and not _is_string_list(payload[key]):
            errors.append(f"{key} must be an array of strings")
    if "recommendation" in payload and not isinstance(payload["recommendation"], str):
        errors.append("recommendation must be a string")

    contradictions = payload.get("contradictions")
    if contradictions is not None:
        if not isinstance(contradictions, list):
            errors.append("contradictions must be an array")
        else:
            for index, item in enumerate(contradictions):
                if not isinstance(item, dict) or set(item) != {"point", "sides"}:
                    errors.append(f"contradictions[{index}] must contain only point and sides")
                elif not all(isinstance(item[key], str) for key in ("point", "sides")):
                    errors.append(f"contradictions[{index}] fields must be strings")

    insights = payload.get("unique_insights")
    if insights is not None:
        if not isinstance(insights, list):
            errors.append("unique_insights must be an array")
        else:
            for index, item in enumerate(insights):
                if not isinstance(item, dict) or set(item) != {"model", "insight"}:
                    errors.append(f"unique_insights[{index}] must contain only model and insight")
                elif not all(isinstance(item[key], str) for key in ("model", "insight")):
                    errors.append(f"unique_insights[{index}] fields must be strings")

    confidence = payload.get("confidence")
    if confidence is not None:
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            errors.append("confidence must be a number from 0 to 1")
    return errors


def _judge_prompt(user_prompt: str, panel: Sequence[ModelResult]) -> str:
    answers = "\n\n".join(
        f"### Model: {item.label} ({item.backend})\n{item.answer}" for item in panel
    )
    return (
        f"{JUDGE_INSTRUCTION}\n\n"
        f"## JSON Schema\n{json.dumps(JUDGE_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        f"## Original prompt\n{user_prompt}\n\n"
        f"## Successful panel answers\n{answers}"
    )


def _repair_prompt(raw: str, validation_errors: Sequence[str]) -> str:
    return (
        "Repair the invalid judge output below. Preserve its substantive analysis, but return "
        "one JSON object that exactly matches the schema. Return JSON only.\n\n"
        f"## JSON Schema\n{json.dumps(JUDGE_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        f"## Validation errors\n- " + "\n- ".join(validation_errors) + "\n\n"
        f"## Invalid output\n{raw}"
    )


JudgeDispatcher = Callable[[Member, str, str, DispatchConfig, bool], ModelResult]


def run_judge(
    judge_member: Member,
    user_prompt: str,
    panel: Sequence[ModelResult],
    config: DispatchConfig,
    repair_attempts: int = DEFAULT_REPAIR_ATTEMPTS,
    dispatcher: JudgeDispatcher = dispatch,
) -> dict[str, Any]:
    valid_panel = successful_results(panel)
    if not valid_panel:
        return {
            "backend": judge_member.backend,
            "model": judge_member.model,
            "raw": "",
            "parsed": None,
            "valid": False,
            "validation_errors": ["no successful panel results"],
            "attempts": 0,
            "result": None,
            "repair_results": [],
        }

    neutral = dataclasses.replace(judge_member, role_key="neutral", role_text=ROLES["neutral"], depth=None)
    results: list[ModelResult] = []
    current_prompt = _judge_prompt(user_prompt, valid_panel)
    parsed: dict[str, Any] | None = None
    validation_errors: list[str] = []

    for index in range(repair_attempts + 1):
        result = dispatcher(neutral, current_prompt, "one-shot", config, False)
        results.append(result)
        parsed = extract_json(result.answer) if result.ok else None
        validation_errors = validate_judge_payload(parsed)
        if not validation_errors:
            break
        if index < repair_attempts:
            current_prompt = _repair_prompt(result.answer, validation_errors)

    final = results[-1]
    return {
        "backend": judge_member.backend,
        "model": judge_member.model,
        "raw": final.answer,
        "parsed": parsed if not validation_errors else None,
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "attempts": len(results),
        "result": final.to_dict(),
        "repair_results": [item.to_dict() for item in results[:-1]],
    }


