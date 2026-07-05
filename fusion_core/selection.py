"""Answer selection and keep-best helpers for judged Fusion panels."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .config import ModelResult, SCORE_AXES
from .routing import successful_results


def _is_score(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 5


def answer_score_rows(parsed: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(parsed, Mapping):
        return []
    rows = parsed.get("answer_scores")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping) and isinstance(row.get("model"), str)]


def answer_score_total(row: Mapping[str, Any]) -> float:
    values = [float(row[axis]) for axis in SCORE_AXES if _is_score(row.get(axis))]
    return sum(values) / len(values) if values else 0.0


def best_label_from_judge(parsed: Mapping[str, Any] | None) -> str | None:
    if not isinstance(parsed, Mapping):
        return None
    best = parsed.get("best_answer_label")
    if isinstance(best, str) and best.strip():
        return best
    ranking = parsed.get("ranking")
    if isinstance(ranking, list):
        for label in ranking:
            if isinstance(label, str) and label.strip():
                return label
    rows = answer_score_rows(parsed)
    if rows:
        return max(rows, key=lambda row: (answer_score_total(row), str(row.get("model"))))["model"]
    return None


def best_panel_result(panel: Sequence[ModelResult], judge: Mapping[str, Any]) -> ModelResult | None:
    successful = successful_results(panel)
    if not successful:
        return None
    parsed = judge.get("parsed") if isinstance(judge, Mapping) else None
    label = best_label_from_judge(parsed if isinstance(parsed, Mapping) else None)
    if label:
        for item in successful:
            if item.label == label:
                return item
    rows = answer_score_rows(parsed if isinstance(parsed, Mapping) else None)
    scored = {str(row["model"]): answer_score_total(row) for row in rows if isinstance(row.get("model"), str)}
    if scored:
        ranked = sorted(successful, key=lambda item: (scored.get(item.label, -1.0), item.confidence or 0.0), reverse=True)
        return ranked[0]
    ranked = sorted(successful, key=lambda item: (item.confidence is not None, item.confidence or 0.0), reverse=True)
    return ranked[0]
