"""Execute one benchmark case against a selected Fusion panel."""

from __future__ import annotations

import argparse
import time
from typing import Any

from .benchmarking import iter_model_results, result_text, score_text
from .pipeline import run_fusion


def run_case(
    case: dict[str, Any],
    preset: str,
    judge: str,
    drafter: str | None,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    fusion_args = argparse.Namespace(
        prompt=case["prompt"],
        strategy="adaptive",
        preset=preset,
        panel=None,
        judge=judge,
        reasoning="on",
        mode="role",
        depth="one-shot",
        agent_workspace="snapshot",
        workspace_source=".",
        timeout=timeout,
        retries=retries,
        backoff=0.5,
        repair_attempts=1,
        auto_draft=drafter,
        benchmark_results="",
        budget_usd=None,
        max_latency_ms=None,
        complexity_threshold=4,
        dry_run=False,
    )
    started = time.perf_counter()
    bundle, exit_code = run_fusion(fusion_args)
    wall_latency_ms = round((time.perf_counter() - started) * 1000)
    text = result_text(bundle)
    score = score_text(text, case)
    model_results = list(iter_model_results(bundle))
    costs = [item.get("cost_usd") for item in model_results if item.get("cost_usd") is not None]
    return {
        "case_id": case["id"],
        "category": case.get("category", "general"),
        "difficulty": case.get("difficulty", "unknown"),
        "score": score,
        "passed": score >= float(case.get("pass_threshold", 1.0)),
        "latency_ms": wall_latency_ms,
        "cost_usd": sum(float(value) for value in costs) if costs else None,
        "cost_coverage": len(costs) / max(1, len(model_results)),
        "exit_code": exit_code,
        "judge_valid": bool(bundle.get("judge", {}).get("valid")) if isinstance(bundle.get("judge"), dict) else False,
        "text": text,
        "errors": bundle.get("error"),
    }
