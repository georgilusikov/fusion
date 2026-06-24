"""Review rounds and adaptive escalation helpers."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable, Mapping, Sequence

from .config import Member, ModelResult
from .dispatch import dispatch
from .routing import successful_results

DispatchFn = Callable[[Member, str, str, Any, bool], ModelResult]


def judge_confidence(judge: Mapping[str, Any]) -> float | None:
    parsed = judge.get("parsed")
    if isinstance(parsed, Mapping):
        value = parsed.get("confidence")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def escalation_reasons(panel: Sequence[ModelResult], judge: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if len(successful_results(panel)) < 2:
        reasons.append("fewer-than-two-successful-members")
    if not judge.get("valid"):
        reasons.append("invalid-judge-output")
        return reasons
    confidence = judge_confidence(judge)
    if confidence is not None and confidence < 0.65:
        reasons.append("low-judge-confidence")
    parsed = judge.get("parsed") or {}
    if isinstance(parsed, Mapping):
        if len(parsed.get("contradictions") or []) >= 2:
            reasons.append("substantial-contradictions")
        if len(parsed.get("coverage_gaps") or []) >= 2:
            reasons.append("substantial-coverage-gaps")
    return reasons


def renamed_result(result: ModelResult, suffix: str) -> ModelResult:
    return dataclasses.replace(result, label=f"{result.label}{suffix}")


def choose_reviewers(panel: Sequence[ModelResult], count: int) -> list[ModelResult]:
    candidates = list(successful_results(panel))
    candidates.sort(key=lambda item: (item.confidence is None, -(item.confidence or 0.0), item.latency_ms))
    chosen: list[ModelResult] = []
    seen_backends: set[str] = set()
    for item in candidates:
        if item.backend not in seen_backends:
            chosen.append(item)
            seen_backends.add(item.backend)
        if len(chosen) >= count:
            return chosen
    for item in candidates:
        if item not in chosen:
            chosen.append(item)
        if len(chosen) >= count:
            break
    return chosen


def member_for_result(result: ModelResult, members: Sequence[Member]) -> Member | None:
    for member in members:
        if member.label == result.label:
            return member
    for member in members:
        if member.backend == result.backend and member.model == result.model:
            return member
    return None


def review_round(
    prompt: str,
    members: Sequence[Member],
    panel: Sequence[ModelResult],
    judge: Mapping[str, Any],
    depth: str,
    config: Any,
    count: int,
    dispatcher: DispatchFn = dispatch,
    log: Callable[[str], None] | None = None,
) -> list[ModelResult]:
    logger = log or (lambda _: None)
    reviews: list[ModelResult] = []
    judge_json = json.dumps(judge.get("parsed") or judge.get("raw") or {}, ensure_ascii=False, indent=2)
    for prior in choose_reviewers(panel, count):
        member = member_for_result(prior, members)
        if member is None:
            continue
        review_prompt = (
            "Revise your answer after reading the Judge analysis.\n"
            "Resolve concrete contradictions, fill coverage gaps, and keep only claims you can defend.\n"
            "Return a complete replacement answer, not commentary about the revision.\n\n"
            f"Original request:\n{prompt}\n\n"
            f"Your first answer:\n{prior.answer}\n\n"
            f"Judge analysis:\n{judge_json}\n"
        )
        revised = dispatcher(member, review_prompt, depth, config, False)
        revised = renamed_result(revised, ":revision")
        logger(f"review {revised.label}: {'ok' if revised.ok else 'FAIL'}")
        reviews.append(revised)
    return reviews


def combine_unique_results(
    first: Sequence[ModelResult],
    second: Sequence[ModelResult],
) -> list[ModelResult]:
    combined = list(first)
    seen = {(item.backend, item.model, item.answer.strip()) for item in combined if item.answer.strip()}
    for item in second:
        key = (item.backend, item.model, item.answer.strip())
        if key in seen:
            continue
        label_collision = any(existing.label == item.label for existing in combined)
        combined.append(renamed_result(item, ":escalated") if label_collision else item)
        seen.add(key)
    return combined
