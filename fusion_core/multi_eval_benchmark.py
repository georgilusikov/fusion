"""Run the deliberation benchmark with an independent evaluator panel."""

from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence

from .benchmark_evaluator_panel import evaluate_case_panel
from .deliberation_benchmark import run_fusion_variant, run_solo_variant


def parse_evaluator_specs(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        raw = value.split(",")
    else:
        raw = list(value)
    result: list[str] = []
    for item in raw:
        spec = str(item).strip()
        if spec and spec not in result:
            result.append(spec)
    return result


def run_multi_eval_case(
    case: Mapping[str, Any],
    *,
    preset: str,
    baseline: str,
    judge: str,
    drafter: str,
    evaluators: Sequence[str],
    timeout: int,
    retries: int,
    scouts: int,
    branches: int,
    critics: int,
    min_valid_evaluators: int | None = None,
) -> dict[str, Any]:
    prompt = str(case["prompt"])
    variants = [
        run_solo_variant(prompt, member_spec=baseline, timeout=timeout, retries=retries),
        run_fusion_variant(
            prompt,
            label="fusion-current",
            preset=preset,
            judge=judge,
            drafter=drafter,
            deliberation=False,
            timeout=timeout,
            retries=retries,
            scouts=scouts,
            branches=branches,
            critics=critics,
        ),
        run_fusion_variant(
            prompt,
            label="fusion-deliberation",
            preset=preset,
            judge=judge,
            drafter=drafter,
            deliberation=True,
            timeout=timeout,
            retries=retries,
            scouts=scouts,
            branches=branches,
            critics=critics,
        ),
    ]
    evaluation = evaluate_case_panel(
        case,
        variants,
        evaluator_specs=evaluators,
        timeout=timeout,
        retries=retries,
        min_valid_evaluators=min_valid_evaluators,
    )
    return {
        "case_id": case["id"],
        "category": case["category"],
        "difficulty": case.get("difficulty"),
        "variants": variants,
        "evaluation": evaluation,
    }


def aggregate_evaluator_diagnostics(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    agreements: list[float] = []
    valid_counts: list[float] = []
    evaluator_costs: list[float] = []
    unanimous = 0
    valid_cases = 0
    for run in runs:
        evaluation = run.get("evaluation")
        if not isinstance(evaluation, Mapping) or not evaluation.get("valid"):
            continue
        valid_cases += 1
        agreement = evaluation.get("agreement")
        if isinstance(agreement, (int, float)) and not isinstance(agreement, bool):
            agreements.append(float(agreement))
            if float(agreement) >= 0.999999:
                unanimous += 1
        valid_evaluators = evaluation.get("valid_evaluators")
        if isinstance(valid_evaluators, (int, float)) and not isinstance(valid_evaluators, bool):
            valid_counts.append(float(valid_evaluators))
        cost = evaluation.get("evaluator_cost_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            evaluator_costs.append(float(cost))
    return {
        "valid_cases": valid_cases,
        "avg_agreement": statistics.mean(agreements) if agreements else None,
        "unanimous_fraction": unanimous / valid_cases if valid_cases else None,
        "avg_valid_evaluators": statistics.mean(valid_counts) if valid_counts else None,
        "avg_evaluator_cost_usd": statistics.mean(evaluator_costs) if evaluator_costs else None,
    }
