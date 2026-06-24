"""Benchmark fixtures, scoring, and aggregation helpers."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not isinstance(case.get("prompt"), str):
            raise ValueError(f"{path}:{line_number}: each case needs string id and prompt")
        cases.append(case)
    if not cases:
        raise ValueError(f"{path}: benchmark set is empty")
    return cases


def _json_object(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def score_text(text: str, case: dict[str, Any]) -> float:
    """Return a deterministic 0..1 score for legacy and fixture-style checks."""
    checks: list[bool] = []
    lowered = text.lower()

    for expected in case.get("expected_contains", []):
        checks.append(str(expected).lower() in lowered)
    for pattern in case.get("expected_regex", []):
        checks.append(re.search(str(pattern), text, flags=re.IGNORECASE | re.MULTILINE) is not None)

    fixture = case.get("checks", {})
    if isinstance(fixture, dict):
        for expected in fixture.get("must_include", []):
            checks.append(str(expected).lower() in lowered)
        for group in fixture.get("must_include_any", []):
            alternatives = group if isinstance(group, list) else [group]
            checks.append(any(str(item).lower() in lowered for item in alternatives))
        for forbidden in fixture.get("must_not_include", []):
            checks.append(str(forbidden).lower() not in lowered)
        if fixture.get("max_chars") is not None:
            try:
                checks.append(len(text) <= int(fixture["max_chars"]))
            except (TypeError, ValueError):
                checks.append(False)
        keys = fixture.get("json_keys", [])
        if keys:
            payload = _json_object(text)
            checks.append(payload is not None and all(str(key) in payload for key in keys))

    if not checks:
        return 1.0 if text.strip() else 0.0
    return sum(checks) / len(checks)


def iter_model_results(bundle: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for item in bundle.get("panel", []):
        if isinstance(item, dict):
            yield item
    judge = bundle.get("judge", {})
    if isinstance(judge, dict):
        result = judge.get("result")
        if isinstance(result, dict):
            yield result
        for repair in judge.get("repair_results", []):
            if isinstance(repair, dict):
                yield repair
    draft = bundle.get("draft_result")
    if isinstance(draft, dict):
        yield draft


def result_text(bundle: dict[str, Any]) -> str:
    draft = bundle.get("draft")
    if isinstance(draft, str) and draft.strip():
        return draft
    judge = bundle.get("judge", {})
    if isinstance(judge, dict):
        parsed = judge.get("parsed")
        if isinstance(parsed, dict) and isinstance(parsed.get("recommendation"), str):
            return parsed["recommendation"]
    return ""


def aggregate(preset: str, runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        by_category[str(run["category"])].append(float(run["score"]))
    known_costs = [float(run["cost_usd"]) for run in runs if run.get("cost_usd") is not None]
    return {
        "preset": preset,
        "quality": sum(float(run["score"]) for run in runs) / len(runs) if runs else 0.0,
        "pass_rate": sum(bool(run["passed"]) for run in runs) / len(runs) if runs else 0.0,
        "avg_latency_ms": round(sum(int(run["latency_ms"]) for run in runs) / len(runs)) if runs else None,
        "avg_cost_usd": sum(known_costs) / len(known_costs) if known_costs else None,
        "avg_cost_coverage": (
            sum(float(run.get("cost_coverage", 0.0)) for run in runs) / len(runs) if runs else 0.0
        ),
        "categories": {
            category: sum(scores) / len(scores) for category, scores in sorted(by_category.items())
        },
        "runs": len(runs),
    }
