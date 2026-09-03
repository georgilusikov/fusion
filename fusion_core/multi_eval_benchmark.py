"""Run the deliberation benchmark with an independent evaluator panel."""

from __future__ import annotations

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
