"""Blind quality benchmark for Fusion deliberation modes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .benchmarking import load_cases
from .config import DispatchConfig, ModelResult
from .dispatch import dispatch
from .evaluation import anonymize_candidates
from .judge import extract_json
from .pipeline import run_fusion
from .routing import load_pricing, parse_member


DELIBERATION_EVAL_AXES = (
    "correctness",
    "depth",
    "coverage",
    "actionability",
    "robustness",
    "insight",
)
DEFAULT_VARIANTS = ("solo", "fusion-current", "fusion-deliberation")


def validate_deliberation_case(case: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(case.get("id"), str) or not str(case.get("id")).strip():
        errors.append("id must be a non-empty string")
    if not isinstance(case.get("prompt"), str) or not str(case.get("prompt")).strip():
        errors.append("prompt must be a non-empty string")
    category = case.get("category")
    if not isinstance(category, str) or not category.strip():
        errors.append("category must be a non-empty string")
    rubric = case.get("rubric")
    if not isinstance(rubric, list) or not rubric or not all(isinstance(item, str) and item.strip() for item in rubric):
        errors.append("rubric must be a non-empty array of strings")
    return errors


def load_deliberation_cases(path: Path) -> list[dict[str, Any]]:
    cases = load_cases(path)
    seen: set[str] = set()
    for index, case in enumerate(cases, 1):
        errors = validate_deliberation_case(case)
        if errors:
            raise ValueError(f"{path}: case {index}: " + "; ".join(errors))
        case_id = str(case["id"])
        if case_id in seen:
            raise ValueError(f"{path}: duplicate case id {case_id}")
        seen.add(case_id)
    return cases


def _fusion_args(
    prompt: str,
    *,
    preset: str,
    judge: str,
    drafter: str,
    deliberation: bool,
    timeout: int,
    retries: int,
    scouts: int,
    branches: int,
    critics: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        prompt=prompt,
        strategy="pro",
        preset=preset,
        panel=None,
        judge=judge,
        critics=None,
        reasoning="on",
        mode="role",
        depth="one-shot",
        blind_judge=True,
        deliberation="on" if deliberation else "off",
        scouts=scouts,
        branch_expansions=branches,
        deliberation_critics=critics,
        agent_workspace="snapshot",
        workspace_source=".",
        timeout=timeout,
        retries=retries,
        backoff=0.5,
        repair_attempts=1,
        reviewers=None,
        no_escalate=True,
        auto_draft=drafter,
        no_draft_gate=False,
        benchmark_results="",
        budget_usd=None,
        max_latency_ms=None,
        complexity_threshold=4,
        dry_run=False,
    )


def _bundle_answer(bundle: Mapping[str, Any]) -> str:
    draft = bundle.get("draft")
    if isinstance(draft, str) and draft.strip():
        return draft.strip()
    panel = bundle.get("panel")
    best_label = bundle.get("best_panel_answer_label")
    if isinstance(panel, list) and isinstance(best_label, str):
        for row in panel:
            if isinstance(row, Mapping) and row.get("label") == best_label and isinstance(row.get("answer"), str):
                return str(row["answer"]).strip()
    return ""


def _metrics_from_bundle(bundle: Mapping[str, Any], wall_latency_ms: int) -> dict[str, Any]:
    metrics = bundle.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}
    return {
        "latency_ms": int(metrics.get("wall_latency_ms") or wall_latency_ms),
        "calls": int(metrics.get("calls") or 0),
        "cost_usd": metrics.get("cost_usd"),
        "input_tokens": metrics.get("input_tokens"),
        "output_tokens": metrics.get("output_tokens"),
    }


def run_fusion_variant(
    prompt: str,
    *,
    label: str,
    preset: str,
    judge: str,
    drafter: str,
    deliberation: bool,
    timeout: int,
    retries: int,
    scouts: int,
    branches: int,
    critics: int,
    fusion_runner: Callable[[argparse.Namespace], tuple[dict[str, Any], int]] = run_fusion,
) -> dict[str, Any]:
    args = _fusion_args(
        prompt,
        preset=preset,
        judge=judge,
        drafter=drafter,
        deliberation=deliberation,
        timeout=timeout,
        retries=retries,
        scouts=scouts,
        branches=branches,
        critics=critics,
    )
    started = time.perf_counter()
    bundle, exit_code = fusion_runner(args)
    wall_latency_ms = round((time.perf_counter() - started) * 1000)
    answer = _bundle_answer(bundle)
    return {
        "label": label,
        "answer": answer,
        "ok": bool(answer) and exit_code == 0,
        "exit_code": exit_code,
        **_metrics_from_bundle(bundle, wall_latency_ms),
    }


def run_solo_variant(
    prompt: str,
    *,
    member_spec: str,
    timeout: int,
    retries: int,
    dispatcher=dispatch,
) -> dict[str, Any]:
    member = parse_member(member_spec, same_mode=True, seen={})
    config = DispatchConfig(
        timeout=timeout,
        retries=retries,
        backoff=0.5,
        reasoning=True,
        strategy="pro",
        pricing=load_pricing(),
    )
    started = time.perf_counter()
    result = dispatcher(member, prompt, "one-shot", config, True)
    wall_latency_ms = round((time.perf_counter() - started) * 1000)
    return {
        "label": "solo",
        "answer": result.answer.strip(),
        "ok": bool(result.ok and result.answer.strip()),
        "exit_code": 0 if result.ok else 1,
        "latency_ms": wall_latency_ms,
        "calls": 1,
        "cost_usd": result.cost_usd,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }


def deliberation_eval_prompt(
    case: Mapping[str, Any],
    anonymous: Sequence[Mapping[str, str]],
) -> str:
    rubric = "\n".join(f"- {item}" for item in case.get("rubric", []))
    answers = "\n\n".join(f"### Answer {row['blind_label']}\n{row['answer']}" for row in anonymous)
    axes = "\n".join(f'- "{axis}": number 0..5' for axis in DELIBERATION_EVAL_AXES)
    return (
        "You are a blind answer evaluator. Do not infer or reward model identity, verbosity, or stylistic confidence. "
        "Evaluate the complete answer against the original task and rubric. Return JSON only.\n\n"
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


def validate_eval_payload(payload: Mapping[str, Any] | None, labels: set[str]) -> list[str]:
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
        seen.add(label)
        for axis in DELIBERATION_EVAL_AXES:
            value = row.get(axis)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= float(value) <= 5:
                errors.append(f"scores[{index}].{axis} must be 0..5")
    if seen != labels:
        errors.append("scores must cover every anonymous answer exactly once")
    winner = payload.get("winner")
    if not isinstance(winner, str) or winner not in labels:
        errors.append("winner must name an anonymous answer")
    return errors


def evaluate_case(
    case: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    *,
    evaluator_spec: str,
    timeout: int,
    retries: int,
    dispatcher=dispatch,
) -> dict[str, Any]:
    candidates = [
        {"label": str(row["label"]), "answer": str(row.get("answer") or "")}
        for row in variants
        if row.get("ok") and str(row.get("answer") or "").strip()
    ]
    anonymous = anonymize_candidates(candidates, seed=str(case["id"]))
    if len(anonymous) < 2:
        return {"valid": False, "errors": ["fewer than two valid variant answers"], "scores": {}, "winner": None}

    evaluator = parse_member(evaluator_spec, same_mode=True, seen={})
    config = DispatchConfig(timeout=timeout, retries=retries, backoff=0.5, reasoning=False, strategy="pro", pricing=load_pricing())
    result = dispatcher(evaluator, deliberation_eval_prompt(case, anonymous), "one-shot", config, False)
    payload = extract_json(result.answer) if result.ok else None
    labels = {row["blind_label"] for row in anonymous}
    errors = validate_eval_payload(payload, labels)
    if errors:
        return {"valid": False, "errors": errors, "raw": result.answer, "scores": {}, "winner": None, "result": result.to_dict()}

    assert isinstance(payload, Mapping)
    blind_to_source = {row["blind_label"]: row["source_label"] for row in anonymous}
    scores: dict[str, dict[str, float]] = {}
    for row in payload.get("scores", []):
        if not isinstance(row, Mapping):
            continue
        source = blind_to_source[str(row["label"])]
        scores[source] = {axis: float(row[axis]) for axis in DELIBERATION_EVAL_AXES}
    winner = blind_to_source[str(payload["winner"])]
    return {
        "valid": True,
        "errors": [],
        "scores": scores,
        "winner": winner,
        "confidence": payload.get("confidence"),
        "result": result.to_dict(),
    }


def run_deliberation_case(
    case: Mapping[str, Any],
    *,
    preset: str,
    baseline: str,
    judge: str,
    drafter: str,
    evaluator: str,
    timeout: int,
    retries: int,
    scouts: int,
    branches: int,
    critics: int,
) -> dict[str, Any]:
    prompt = str(case["prompt"])
    variants = [
        run_solo_variant(prompt, member_spec=baseline, timeout=timeout, retries=retries),
        run_fusion_variant(
            prompt, label="fusion-current", preset=preset, judge=judge, drafter=drafter,
            deliberation=False, timeout=timeout, retries=retries, scouts=scouts, branches=branches, critics=critics,
        ),
        run_fusion_variant(
            prompt, label="fusion-deliberation", preset=preset, judge=judge, drafter=drafter,
            deliberation=True, timeout=timeout, retries=retries, scouts=scouts, branches=branches, critics=critics,
        ),
    ]
    evaluation = evaluate_case(case, variants, evaluator_spec=evaluator, timeout=timeout, retries=retries)
    return {
        "case_id": case["id"],
        "category": case["category"],
        "difficulty": case.get("difficulty"),
        "variants": variants,
        "evaluation": evaluation,
    }


def _mean_known(values: Sequence[Any]) -> float | None:
    known = [float(value) for value in values if value is not None]
    return sum(known) / len(known) if known else None


def aggregate_deliberation_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    variant_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    wins: dict[str, int] = defaultdict(int)
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    valid_evaluations = 0

    for run in runs:
        for variant in run.get("variants", []):
            if isinstance(variant, Mapping) and isinstance(variant.get("label"), str):
                variant_rows[str(variant["label"])].append(variant)
        evaluation = run.get("evaluation")
        if not isinstance(evaluation, Mapping) or not evaluation.get("valid"):
            continue
        valid_evaluations += 1
        winner = evaluation.get("winner")
        if isinstance(winner, str):
            wins[winner] += 1
        for variant, axes in (evaluation.get("scores") or {}).items():
            if not isinstance(axes, Mapping):
                continue
            for axis in DELIBERATION_EVAL_AXES:
                value = axes.get(axis)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    scores[str(variant)][axis].append(float(value))

    variants: dict[str, Any] = {}
    for label, rows in sorted(variant_rows.items()):
        variants[label] = {
            "runs": len(rows),
            "success_rate": sum(bool(row.get("ok")) for row in rows) / len(rows) if rows else 0.0,
            "win_rate": wins[label] / valid_evaluations if valid_evaluations else 0.0,
            "avg_latency_ms": _mean_known([row.get("latency_ms") for row in rows]),
            "avg_cost_usd": _mean_known([row.get("cost_usd") for row in rows]),
            "avg_calls": _mean_known([row.get("calls") for row in rows]),
            "axes": {
                axis: statistics.mean(values)
                for axis, values in scores.get(label, {}).items()
                if values
            },
        }
        axis_values = list(variants[label]["axes"].values())
        variants[label]["quality"] = statistics.mean(axis_values) if axis_values else None

    current = variants.get("fusion-current", {})
    deliberation = variants.get("fusion-deliberation", {})
    delta = None
    if current.get("quality") is not None and deliberation.get("quality") is not None:
        delta = float(deliberation["quality"]) - float(current["quality"])
    return {
        "runs": len(runs),
        "valid_evaluations": valid_evaluations,
        "variants": variants,
        "deliberation_quality_delta": delta,
    }
