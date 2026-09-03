"""Independent blind evaluator panels for deliberation benchmark cases."""

from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Mapping, Sequence

from .config import DispatchConfig, ModelResult
from .dispatch import dispatch
from .evaluation import anonymize_candidates
from .judge import extract_json
from .routing import load_pricing, parse_member

EVAL_AXES = (
    "correctness",
    "depth",
    "coverage",
    "actionability",
    "robustness",
    "insight",
)

EvaluatorDispatcher = Callable[[Any, str, str, DispatchConfig, bool], ModelResult]


def evaluator_prompt(case: Mapping[str, Any], anonymous: Sequence[Mapping[str, str]]) -> str:
    rubric = "\n".join(f"- {item}" for item in case.get("rubric", []))
    answers = "\n\n".join(
        f"### Answer {row['blind_label']}\n{row['answer']}" for row in anonymous
    )
    axes = "\n".join(f'- "{axis}": number 0..5' for axis in EVAL_AXES)
    return (
        "You are one independent blind answer evaluator. Do not infer or reward model identity, "
        "verbosity, stylistic confidence, or familiarity with a provider. Evaluate the complete "
        "answer against the original task and rubric. Return JSON only.\n\n"
        f"## Original task\n{case['prompt']}\n\n"
        f"## Rubric\n{rubric}\n\n"
        f"## Axes\n{axes}\n\n"
        "## Required JSON\n"
        "{\n"
        '  "scores": [{"label": "A", "correctness": 0, "depth": 0, "coverage": 0, '
        '"actionability": 0, "robustness": 0, "insight": 0, "notes": "..."}],\n'
        '  "ranking": ["A"],\n'
        '  "winner": "A",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        f"## Anonymous answers\n{answers}\n"
    )


def validate_payload(payload: Mapping[str, Any] | None, labels: set[str]) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["evaluation is not a JSON object"]
    errors: list[str] = []
    scores = payload.get("scores")
    if not isinstance(scores, list):
        return ["scores must be an array"]
    seen: set[str] = set()
    for index, row in enumerate(scores):
        if not isinstance(row, Mapping):
            errors.append(f"scores[{index}] must be an object")
            continue
        label = row.get("label")
        if not isinstance(label, str) or label not in labels:
            errors.append(f"scores[{index}].label must name an anonymous answer")
            continue
        if label in seen:
            errors.append(f"scores[{index}].label is duplicated")
        seen.add(label)
        for axis in EVAL_AXES:
            value = row.get(axis)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= float(value) <= 5:
                errors.append(f"scores[{index}].{axis} must be 0..5")
    if seen != labels:
        errors.append("scores must cover every anonymous answer exactly once")
    winner = payload.get("winner")
    if not isinstance(winner, str) or winner not in labels:
        errors.append("winner must name an anonymous answer")
    confidence = payload.get("confidence")
    if confidence is not None and (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= float(confidence) <= 1
    ):
        errors.append("confidence must be 0..1 when present")
    return errors


def _candidate_answers(variants: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {"label": str(row["label"]), "answer": str(row.get("answer") or "")}
        for row in variants
        if row.get("ok") and str(row.get("answer") or "").strip()
    ]


def _run_evaluator(
    case: Mapping[str, Any],
    candidates: Sequence[Mapping[str, str]],
    evaluator_spec: str,
    evaluator_index: int,
    *,
    timeout: int,
    retries: int,
    dispatcher: EvaluatorDispatcher,
) -> dict[str, Any]:
    # Different deterministic order per evaluator reduces shared position bias while
    # keeping repeated benchmark runs reproducible.
    anonymous = anonymize_candidates(
        candidates,
        seed=f"{case['id']}::evaluator::{evaluator_index}::{evaluator_spec}",
    )
    labels = {row["blind_label"] for row in anonymous}
    blind_to_source = {row["blind_label"]: row["source_label"] for row in anonymous}
    evaluator = parse_member(evaluator_spec, same_mode=True, seen={})
    config = DispatchConfig(
        timeout=timeout,
        retries=retries,
        backoff=0.5,
        reasoning=False,
        strategy="pro",
        pricing=load_pricing(),
    )
    result = dispatcher(evaluator, evaluator_prompt(case, anonymous), "one-shot", config, False)
    payload = extract_json(result.answer) if result.ok else None
    errors = validate_payload(payload, labels)
    if errors:
        return {
            "evaluator": evaluator_spec,
            "valid": False,
            "errors": errors,
            "raw": result.answer,
            "result": result.to_dict(),
            "presentation": [row["source_label"] for row in anonymous],
        }

    assert isinstance(payload, Mapping)
    scores: dict[str, dict[str, float]] = {}
    for row in payload.get("scores", []):
        if not isinstance(row, Mapping):
            continue
        source = blind_to_source[str(row["label"])]
        scores[source] = {axis: float(row[axis]) for axis in EVAL_AXES}
    winner = blind_to_source[str(payload["winner"])]
    return {
        "evaluator": evaluator_spec,
        "valid": True,
        "errors": [],
        "scores": scores,
        "winner": winner,
        "confidence": payload.get("confidence"),
        "result": result.to_dict(),
        "presentation": [row["source_label"] for row in anonymous],
    }


def _overall_score(axes: Mapping[str, float]) -> float:
    values = [float(axes[axis]) for axis in EVAL_AXES if axis in axes]
    return statistics.mean(values) if values else 0.0


def evaluate_case_panel(
    case: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    *,
    evaluator_specs: Sequence[str],
    timeout: int,
    retries: int,
    min_valid_evaluators: int | None = None,
    dispatcher: EvaluatorDispatcher = dispatch,
) -> dict[str, Any]:
    candidates = _candidate_answers(variants)
    if len(candidates) < 2:
        return {
            "valid": False,
            "errors": ["fewer than two valid variant answers"],
            "scores": {},
            "winner": None,
            "evaluators": [],
        }
    specs = [str(spec).strip() for spec in evaluator_specs if str(spec).strip()]
    if not specs:
        return {
            "valid": False,
            "errors": ["no evaluator specs configured"],
            "scores": {},
            "winner": None,
            "evaluators": [],
        }

    required = min_valid_evaluators
    if required is None:
        required = len(specs) // 2 + 1
    required = max(1, min(int(required), len(specs)))

    def run(item: tuple[int, str]) -> dict[str, Any]:
        index, spec = item
        return _run_evaluator(
            case,
            candidates,
            spec,
            index,
            timeout=timeout,
            retries=retries,
            dispatcher=dispatcher,
        )

    with ThreadPoolExecutor(max_workers=len(specs)) as executor:
        evaluator_runs = list(executor.map(run, enumerate(specs)))

    valid_runs = [row for row in evaluator_runs if row.get("valid")]
    if len(valid_runs) < required:
        return {
            "valid": False,
            "errors": [f"only {len(valid_runs)} of {len(specs)} evaluators valid; require {required}"],
            "scores": {},
            "winner": None,
            "valid_evaluators": len(valid_runs),
            "required_evaluators": required,
            "evaluators": evaluator_runs,
        }

    score_buckets: dict[str, dict[str, list[float]]] = {
        str(candidate["label"]): {axis: [] for axis in EVAL_AXES}
        for candidate in candidates
    }
    winner_votes: dict[str, int] = {str(candidate["label"]): 0 for candidate in candidates}
    confidences: list[float] = []
    for row in valid_runs:
        for source, axes in (row.get("scores") or {}).items():
            if source not in score_buckets or not isinstance(axes, Mapping):
                continue
            for axis in EVAL_AXES:
                value = axes.get(axis)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    score_buckets[source][axis].append(float(value))
        winner = row.get("winner")
        if isinstance(winner, str) and winner in winner_votes:
            winner_votes[winner] += 1
        confidence = row.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            confidences.append(float(confidence))

    median_scores: dict[str, dict[str, float]] = {}
    for source, axes in score_buckets.items():
        median_scores[source] = {
            axis: float(statistics.median(values))
            for axis, values in axes.items()
            if values
        }

    winner = max(
        median_scores,
        key=lambda source: (_overall_score(median_scores[source]), winner_votes.get(source, 0), source),
    )
    max_votes = max(winner_votes.values()) if winner_votes else 0
    agreement = max_votes / len(valid_runs) if valid_runs else 0.0
    evaluator_costs = [
        row.get("result", {}).get("cost_usd")
        for row in valid_runs
        if isinstance(row.get("result"), Mapping) and row.get("result", {}).get("cost_usd") is not None
    ]
    return {
        "valid": True,
        "errors": [],
        "scores": median_scores,
        "winner": winner,
        "winner_votes": winner_votes,
        "agreement": agreement,
        "confidence": float(statistics.median(confidences)) if confidences else None,
        "valid_evaluators": len(valid_runs),
        "required_evaluators": required,
        "evaluator_cost_usd": sum(float(value) for value in evaluator_costs) if evaluator_costs else None,
        "evaluators": evaluator_runs,
    }
