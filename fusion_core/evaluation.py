"""Blind evaluation and rematch helpers for Fusion quality gates."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EVAL_AXES = ("correctness", "depth", "coverage", "actionability")
REMATCH_VARIANTS = ("fusion-pro", "solo-claude", "claude-x3-self-pick")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def blind_label(index: int) -> str:
    if index < 0:
        raise ValueError("index must be non-negative")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label = ""
    value = index
    while True:
        label = alphabet[value % 26] + label
        value = value // 26 - 1
        if value < 0:
            return label


def _stable_order_key(seed: str, source_label: str, answer: str) -> str:
    payload = f"{seed}\0{source_label}\0{answer}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def anonymize_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: str,
) -> list[dict[str, str]]:
    """Return candidates with stable blind labels while retaining private source labels."""
    normalized: list[dict[str, str]] = []
    for item in candidates:
        source = str(item.get("label") or item.get("variant") or "candidate")
        answer = str(item.get("answer") or "")
        if answer.strip():
            normalized.append({"source_label": source, "answer": answer})
    normalized.sort(key=lambda item: _stable_order_key(seed, item["source_label"], item["answer"]))
    return [
        {"blind_label": blind_label(index), **item}
        for index, item in enumerate(normalized)
    ]


def blind_judge_prompt(
    original_prompt: str,
    candidates: Sequence[Mapping[str, str]],
    *,
    axes: Sequence[str] = EVAL_AXES,
) -> str:
    """Build a provider-agnostic prompt for blind answer judging."""
    answers = "\n\n".join(
        f"### Answer {item['blind_label']}\n{item['answer']}" for item in candidates
    )
    axis_contract = "\n".join(f'- "{axis}": integer 0..5' for axis in axes)
    return (
        "You are a blind evaluator. Do not infer which model produced an answer. "
        "Score answer quality, not style markers. Return JSON only.\n\n"
        "## Required JSON\n"
        "{\n"
        "  \"scores\": [\n"
        "    {\"label\": \"A\", \"correctness\": 0, \"depth\": 0, "
        "\"coverage\": 0, \"actionability\": 0, \"notes\": \"...\"}\n"
        "  ],\n"
        "  \"ranking\": [\"A\"],\n"
        "  \"winner\": \"A\",\n"
        "  \"confidence\": 0.0\n"
        "}\n\n"
        f"## Axes\n{axis_contract}\n\n"
        f"## Original prompt\n{original_prompt}\n\n"
        f"## Anonymous answers\n{answers}\n"
    )


def _score_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return 0.0
    if number > 5:
        return 5.0
    return number


def scores_by_label(judge_payload: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    raw_scores = judge_payload.get("scores", [])
    if not isinstance(raw_scores, list):
        return result
    for row in raw_scores:
        if not isinstance(row, Mapping) or not isinstance(row.get("label"), str):
            continue
        axes: dict[str, float] = {}
        for axis in EVAL_AXES:
            number = _score_number(row.get(axis))
            if number is not None:
                axes[axis] = number
        if axes:
            result[row["label"]] = axes
    return result


def median_axis_scores(judge_payloads: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[float]]] = {}
    for payload in judge_payloads:
        for label, axes in scores_by_label(payload).items():
            label_bucket = buckets.setdefault(label, {axis: [] for axis in EVAL_AXES})
            for axis, value in axes.items():
                label_bucket.setdefault(axis, []).append(value)
    medians: dict[str, dict[str, float]] = {}
    for label, axis_values in buckets.items():
        medians[label] = {
            axis: float(statistics.median(values))
            for axis, values in axis_values.items()
            if values
        }
    return medians


def overall_score(axis_scores: Mapping[str, float]) -> float:
    values = [float(axis_scores[axis]) for axis in EVAL_AXES if axis in axis_scores]
    return sum(values) / len(values) if values else 0.0


def winner_from_scores(axis_scores_by_label: Mapping[str, Mapping[str, float]]) -> str | None:
    if not axis_scores_by_label:
        return None
    return max(
        axis_scores_by_label,
        key=lambda label: (overall_score(axis_scores_by_label[label]), label),
    )


def rematch_verdict(
    variant_scores: Mapping[str, float],
    *,
    fusion_key: str = "fusion-pro",
    baseline_keys: Sequence[str] = ("solo-claude", "claude-x3-self-pick"),
    epsilon: float = 0.01,
) -> dict[str, Any]:
    """Apply the kill rule: if a baseline matches or beats Fusion, fix synthesis first."""
    fusion_score = variant_scores.get(fusion_key)
    baselines = {key: variant_scores[key] for key in baseline_keys if key in variant_scores}
    if fusion_score is None or not baselines:
        return {"status": "insufficient-data", "reason": "missing fusion or baseline scores"}
    best_baseline, best_score = max(baselines.items(), key=lambda item: item[1])
    if best_score + epsilon >= fusion_score:
        return {
            "status": "baseline-matches-or-wins",
            "action": "fix synthesis/keep-best before expanding the panel",
            "fusion_score": fusion_score,
            "best_baseline": best_baseline,
            "best_baseline_score": best_score,
        }
    return {
        "status": "fusion-wins",
        "action": "safe to test additional roadmap features",
        "fusion_score": fusion_score,
        "best_baseline": best_baseline,
        "best_baseline_score": best_score,
    }
