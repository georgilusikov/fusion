"""Experiment helpers for Phase 2 Fusion roadmap work."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SUPPORTED_LANGUAGE_AXES = {
    "ru": "Answer in Russian unless the user asked otherwise.",
    "en": "Answer in English unless the user asked otherwise.",
    "zh": "Answer in Chinese, then preserve key technical terms in English where useful.",
}


def debate_prompt(original_prompt: str, answers: Sequence[Mapping[str, str]]) -> str:
    answer_text = "\n\n".join(
        f"### {item.get('label', 'answer')}\n{item.get('answer', '')}" for item in answers
    )
    return (
        "You are in a debate round. Attack and defend claims from the answers below. "
        "Find concrete errors, unsupported assumptions, missing edge cases, and reusable insights. "
        "Return a replacement answer that survives the critique; do not narrate the process.\n\n"
        f"## Original prompt\n{original_prompt}\n\n"
        f"## Answers to test\n{answer_text}\n"
    )


def language_axis_prompt(original_prompt: str, language: str) -> str:
    try:
        instruction = SUPPORTED_LANGUAGE_AXES[language]
    except KeyError as exc:
        raise ValueError(f"unsupported language axis: {language}") from exc
    return f"{instruction}\n\n{original_prompt}"


def outcome_event(
    *,
    case_id: str,
    answer_id: str,
    affected_action: bool,
    later_corrected: bool = False,
    notes: str = "",
    timestamp: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "fusion_outcome_event",
        "version": 1,
        "timestamp": timestamp or dt.datetime.now(dt.timezone.utc).isoformat(),
        "case_id": case_id,
        "answer_id": answer_id,
        "affected_action": bool(affected_action),
        "later_corrected": bool(later_corrected),
        "notes": notes,
    }


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def validate_stop_rule(packet: Mapping[str, Any], known_case_ids: set[str]) -> list[str]:
    errors: list[str] = []
    feature = str(packet.get("feature") or "").strip()
    if not feature:
        errors.append("feature is required")
    case_ids = packet.get("case_ids", [])
    if not isinstance(case_ids, list) or not case_ids:
        errors.append("case_ids must name at least one failed golden or ledger case")
        return errors
    missing = [str(case_id) for case_id in case_ids if str(case_id) not in known_case_ids]
    if missing:
        errors.append("unknown case_ids: " + ", ".join(sorted(missing)))
    return errors
