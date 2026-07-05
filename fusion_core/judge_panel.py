"""Cross-provider judge panel aggregation."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .config import DispatchConfig, Member, ModelResult, SCORE_AXES
from .judge import run_judge, validate_judge_payload


def _unique_strings(payloads: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for payload in payloads:
        values = payload.get(key, [])
        if not isinstance(values, list):
            continue
        for value in values:
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
    return result


def _unique_dicts(payloads: Sequence[Mapping[str, Any]], key: str, fields: Sequence[str]) -> list[dict[str, str]]:
    seen: set[tuple[str, ...]] = set()
    result: list[dict[str, str]] = []
    for payload in payloads:
        values = payload.get(key, [])
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            row = {field: str(value.get(field, "")).strip() for field in fields}
            identity = tuple(row[field] for field in fields)
            if all(identity) and identity not in seen:
                seen.add(identity)
                result.append(row)
    return result


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _score_total(row: Mapping[str, Any]) -> float:
    values = []
    for axis in SCORE_AXES:
        try:
            values.append(float(row[axis]))
        except (KeyError, TypeError, ValueError):
            continue
    return sum(values) / len(values) if values else 0.0


def aggregate_judge_payloads(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not payloads:
        return None
    score_buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: {axis: [] for axis in SCORE_AXES})
    rationales: dict[str, str] = {}
    for payload in payloads:
        rows = payload.get("answer_scores", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("model"), str):
                continue
            label = str(row["model"])
            rationales.setdefault(label, str(row.get("rationale") or ""))
            for axis in SCORE_AXES:
                value = row.get(axis)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    score_buckets[label][axis].append(float(value))
    answer_scores = []
    for label, axes in sorted(score_buckets.items()):
        row = {"model": label, "rationale": rationales.get(label, "median judge score")}
        for axis in SCORE_AXES:
            row[axis] = _median(axes.get(axis, []))
        answer_scores.append(row)
    ranking = [row["model"] for row in sorted(answer_scores, key=_score_total, reverse=True)]
    confidence_values = [
        float(payload["confidence"])
        for payload in payloads
        if isinstance(payload.get("confidence"), (int, float)) and not isinstance(payload.get("confidence"), bool)
    ]
    aggregate = {
        "consensus": _unique_strings(payloads, "consensus"),
        "contradictions": _unique_dicts(payloads, "contradictions", ("point", "sides")),
        "coverage_gaps": _unique_strings(payloads, "coverage_gaps"),
        "unique_insights": _unique_dicts(payloads, "unique_insights", ("model", "insight")),
        "blind_spots": _unique_strings(payloads, "blind_spots"),
        "answer_scores": answer_scores,
        "ranking": ranking,
        "best_answer_label": ranking[0] if ranking else str(payloads[0].get("best_answer_label") or ""),
        "recommendation": str(payloads[0].get("recommendation") or ""),
        "confidence": _median(confidence_values),
    }
    return aggregate if not validate_judge_payload(aggregate) else None


def run_judge_panel(
    judge_members: Sequence[Member],
    user_prompt: str,
    panel: Sequence[ModelResult],
    config: DispatchConfig,
    repair_attempts: int,
) -> dict[str, Any]:
    members = list(judge_members)
    if not members:
        raise ValueError("judge panel is empty")
    results = [run_judge(member, user_prompt, panel, config, repair_attempts=repair_attempts) for member in members]
    valid_payloads = [item["parsed"] for item in results if item.get("valid") and isinstance(item.get("parsed"), Mapping)]
    if len(members) == 1:
        single = dict(results[0])
        single["judge_results"] = results
        single["aggregation"] = {"mode": "single", "valid_count": len(valid_payloads), "judge_count": 1}
        return single
    aggregate = aggregate_judge_payloads(valid_payloads)
    if aggregate is None:
        fallback = dict(results[0])
        fallback["judge_results"] = results
        fallback["aggregation"] = {"mode": "panel", "valid_count": len(valid_payloads), "judge_count": len(members)}
        return fallback
    return {
        "backend": "judge-panel",
        "model": ",".join(member.label for member in members),
        "raw": json.dumps(aggregate, ensure_ascii=False),
        "parsed": aggregate,
        "valid": True,
        "validation_errors": [],
        "attempts": sum(int(item.get("attempts", 0)) for item in results),
        "result": None,
        "repair_results": [],
        "judge_results": results,
        "aggregation": {"mode": "panel", "valid_count": len(valid_payloads), "judge_count": len(members)},
    }
