"""Promotion rules for deciding whether deliberation should move toward default use."""

from __future__ import annotations

import math
from typing import Any, Mapping


DEFAULT_MIN_VALID_EVALUATIONS = 10
DEFAULT_MIN_VALID_FRACTION = 0.80
DEFAULT_MIN_QUALITY_DELTA = 0.10
DEFAULT_MIN_WIN_RATE_DELTA = 0.05
DEFAULT_MAX_SUCCESS_RATE_REGRESSION = 0.02


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    top = _number(numerator)
    bottom = _number(denominator)
    if top is None or bottom is None or bottom <= 0:
        return None
    return top / bottom


def promotion_verdict(
    aggregate: Mapping[str, Any],
    *,
    min_valid_evaluations: int = DEFAULT_MIN_VALID_EVALUATIONS,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    min_quality_delta: float = DEFAULT_MIN_QUALITY_DELTA,
    min_win_rate_delta: float = DEFAULT_MIN_WIN_RATE_DELTA,
    max_success_rate_regression: float = DEFAULT_MAX_SUCCESS_RATE_REGRESSION,
    max_latency_ratio: float | None = None,
    max_cost_ratio: float | None = None,
) -> dict[str, Any]:
    """Return a machine-readable PROMOTE/HOLD decision from benchmark aggregates.

    Latency and cost are always reported. They become hard gates only when callers
    explicitly provide max ratios, because acceptable efficiency trade-offs are
    product-policy choices rather than universal quality truths.
    """
    runs = int(aggregate.get("runs") or 0)
    valid = int(aggregate.get("valid_evaluations") or 0)
    variants = aggregate.get("variants")
    if not isinstance(variants, Mapping):
        return {"status": "HOLD", "reasons": ["missing variant aggregates"], "checks": {}}

    current = variants.get("fusion-current")
    deliberation = variants.get("fusion-deliberation")
    solo = variants.get("solo")
    if not isinstance(current, Mapping) or not isinstance(deliberation, Mapping):
        return {"status": "HOLD", "reasons": ["missing current or deliberation aggregate"], "checks": {}}

    current_quality = _number(current.get("quality"))
    deliberation_quality = _number(deliberation.get("quality"))
    current_win = _number(current.get("win_rate"))
    deliberation_win = _number(deliberation.get("win_rate"))
    current_success = _number(current.get("success_rate"))
    deliberation_success = _number(deliberation.get("success_rate"))
    solo_quality = _number(solo.get("quality")) if isinstance(solo, Mapping) else None

    quality_delta = (
        deliberation_quality - current_quality
        if deliberation_quality is not None and current_quality is not None
        else None
    )
    win_rate_delta = (
        deliberation_win - current_win
        if deliberation_win is not None and current_win is not None
        else None
    )
    success_rate_delta = (
        deliberation_success - current_success
        if deliberation_success is not None and current_success is not None
        else None
    )
    valid_fraction = valid / runs if runs else 0.0
    required_valid = max(
        max(0, int(min_valid_evaluations)),
        math.ceil(max(0.0, min(1.0, min_valid_fraction)) * runs),
    )

    latency_ratio = _ratio(deliberation.get("avg_latency_ms"), current.get("avg_latency_ms"))
    cost_ratio = _ratio(deliberation.get("avg_cost_usd"), current.get("avg_cost_usd"))

    checks: dict[str, dict[str, Any]] = {}

    def add_check(name: str, passed: bool, actual: Any, required: Any) -> None:
        checks[name] = {"passed": bool(passed), "actual": actual, "required": required}

    add_check("valid_evaluations", valid >= required_valid, valid, f">={required_valid}")
    add_check(
        "quality_delta",
        quality_delta is not None and quality_delta >= min_quality_delta,
        quality_delta,
        f">={min_quality_delta}",
    )
    add_check(
        "win_rate_delta",
        win_rate_delta is not None and win_rate_delta >= min_win_rate_delta,
        win_rate_delta,
        f">={min_win_rate_delta}",
    )
    add_check(
        "success_rate_regression",
        success_rate_delta is not None and success_rate_delta >= -max_success_rate_regression,
        success_rate_delta,
        f">=-{max_success_rate_regression}",
    )

    if solo_quality is not None and deliberation_quality is not None:
        add_check(
            "beats_or_matches_solo",
            deliberation_quality >= solo_quality,
            deliberation_quality - solo_quality,
            ">=0",
        )

    if max_latency_ratio is not None:
        add_check(
            "latency_ratio",
            latency_ratio is not None and latency_ratio <= max_latency_ratio,
            latency_ratio,
            f"<={max_latency_ratio}",
        )
    if max_cost_ratio is not None:
        add_check(
            "cost_ratio",
            cost_ratio is not None and cost_ratio <= max_cost_ratio,
            cost_ratio,
            f"<={max_cost_ratio}",
        )

    failed = [name for name, row in checks.items() if not row["passed"]]
    reasons = [f"failed check: {name}" for name in failed]
    return {
        "status": "PROMOTE" if not failed else "HOLD",
        "reasons": reasons,
        "checks": checks,
        "metrics": {
            "runs": runs,
            "valid_evaluations": valid,
            "valid_fraction": valid_fraction,
            "quality_delta": quality_delta,
            "win_rate_delta": win_rate_delta,
            "success_rate_delta": success_rate_delta,
            "latency_ratio": latency_ratio,
            "cost_ratio": cost_ratio,
            "current_quality": current_quality,
            "deliberation_quality": deliberation_quality,
            "solo_quality": solo_quality,
        },
    }
