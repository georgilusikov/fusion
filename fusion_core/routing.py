"""Environment loading, member parsing, routing, and result helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    API_BACKENDS, API_DEFAULT_MODEL, CATEGORY_TERMS, COMPLEX_TERMS, CONFIDENCE_RE,
    DEFAULT_STRATEGY, KNOWN_CLIS, PRESETS, REPO_ROOT, ROLES,
    STRATEGY_CANDIDATES, STRATEGY_FALLBACK, Member, ModelResult, StrategyDecision,
)

# --- environment and pricing ----------------------------------------------

def load_dotenv() -> None:
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def api_key_for(backend: str) -> str | None:
    if backend in {"or", "openrouter"}:
        return os.environ.get("OPENROUTER_API_KEY")
    if backend == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if backend == "openai":
        return os.environ.get("OPENAI_API_KEY")
    if backend == "google":
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return None


def load_pricing() -> dict[str, dict[str, float]]:
    """Load optional prices in USD per one million tokens.

    Supported sources, in priority order:
      FUSION_PRICING_JSON='{"model": {"input": 1, "output": 2}}'
      FUSION_PRICING_FILE=/path/to/pricing.json
    """
    raw = os.environ.get("FUSION_PRICING_JSON")
    path = os.environ.get("FUSION_PRICING_FILE")
    try:
        data = json.loads(raw) if raw else json.loads(Path(path).read_text()) if path else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, dict[str, float]] = {}
    for name, prices in data.items():
        if not isinstance(prices, dict):
            continue
        try:
            result[str(name)] = {
                "input": float(prices.get("input", 0.0)),
                "output": float(prices.get("output", 0.0)),
            }
        except (TypeError, ValueError):
            continue
    return result


def estimate_cost(
    backend: str,
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    pricing: Mapping[str, Mapping[str, float]],
) -> float | None:
    if not model or input_tokens is None or output_tokens is None:
        return None
    price = pricing.get(f"{backend}:{model}") or pricing.get(model)
    if not price:
        return None
    return round(
        (input_tokens * float(price.get("input", 0.0)) + output_tokens * float(price.get("output", 0.0)))
        / 1_000_000,
        8,
    )


# --- parsing and strategy selection ---------------------------------------

def parse_member(token: str, same_mode: bool, seen: dict[str, int]) -> Member:
    role_key = "neutral"
    head = token.strip()
    if "@" in head:
        head, role_key = head.rsplit("@", 1)
        head, role_key = head.strip(), role_key.strip()
    if same_mode:
        role_key = "neutral"
    if role_key not in ROLES:
        raise ValueError(f"unknown role: {role_key!r} (roles={sorted(ROLES)})")

    depth: str | None = None
    if "!" in head:
        head, depth_key = head.rsplit("!", 1)
        depth_key = depth_key.strip().replace("oneshot", "one-shot")
        if depth_key not in {"agent", "one-shot"}:
            raise ValueError(f"bad depth {depth_key!r} (use agent|one-shot)")
        depth = depth_key

    model: str | None = None
    if ":" in head:
        backend, model = head.split(":", 1)
        backend, model = backend.strip(), model.strip()
    else:
        backend = head.strip()

    if backend in API_BACKENDS:
        kind = "api"
        model = model or API_DEFAULT_MODEL[backend]
    elif backend in KNOWN_CLIS:
        kind = "cli"
    else:
        raise ValueError(
            f"unknown backend: {backend!r} (clis={sorted(KNOWN_CLIS)}, api={sorted(API_BACKENDS)})"
        )

    base = backend if model is None else f"{backend}:{model.split('/')[-1]}"
    seen[base] = seen.get(base, 0) + 1
    label = base if seen[base] == 1 else f"{base}#{seen[base]}"
    return Member(label, kind, backend, model, role_key, ROLES[role_key], depth)


def parse_panel(spec: str, same_mode: bool = False) -> list[Member]:
    seen: dict[str, int] = {}
    return [parse_member(token, same_mode, seen) for token in spec.split(",") if token.strip()]


def classify_prompt(prompt: str) -> tuple[int, str]:
    text = prompt.lower()
    score = 0
    length = len(prompt)
    if length > 500:
        score += 2
    elif length > 200:
        score += 1
    if prompt.count("\n") >= 6 or "```" in prompt:
        score += 2
    hits = sum(1 for term in COMPLEX_TERMS if term in text)
    score += min(4, hits)
    if any(mark in text for mark in ("step by step", "несколько этап", "compare", "сравни")):
        score += 1
    if any(mark in text for mark in ("latest", "current", "today", "актуальн", "сегодня")):
        score += 1

    category = "general"
    best_hits = 0
    for name, terms in CATEGORY_TERMS.items():
        category_hits = sum(1 for term in terms if term in text)
        if category_hits > best_hits:
            category, best_hits = name, category_hits
    return min(score, 10), category


def _load_benchmark_results(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    panels = payload.get("panels", []) if isinstance(payload, dict) else []
    return [item for item in panels if isinstance(item, dict) and isinstance(item.get("preset"), str)]


def _metric_number(item: Mapping[str, Any], key: str) -> float | None:
    value = item.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def select_strategy(
    prompt: str,
    requested: str = DEFAULT_STRATEGY,
    explicit_preset: str | None = None,
    benchmark_path: Path | None = None,
    budget_usd: float | None = None,
    max_latency_ms: int | None = None,
    complexity_threshold: int = 4,
) -> StrategyDecision:
    complexity, category = classify_prompt(prompt)
    resolved = requested
    if requested == "adaptive":
        resolved = "pro" if complexity >= complexity_threshold else "lite"

    if explicit_preset:
        return StrategyDecision(
            requested=requested,
            resolved=resolved,
            preset=explicit_preset,
            category=category,
            complexity_score=complexity,
            source="explicit",
            reason="explicit --preset overrides automatic selection",
        )

    candidates = STRATEGY_CANDIDATES[resolved]
    benchmark_rows = _load_benchmark_results(benchmark_path)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in benchmark_rows:
        if row.get("preset") not in candidates:
            continue
        cost = _metric_number(row, "avg_cost_usd")
        latency = _metric_number(row, "avg_latency_ms")
        if budget_usd is not None and (cost is None or cost > budget_usd):
            continue
        if max_latency_ms is not None and (latency is None or latency > max_latency_ms):
            continue
        quality = _metric_number(row, "quality")
        if quality is None:
            quality = _metric_number(row, "pass_rate") or 0.0
        category_scores = row.get("categories", {})
        category_quality = None
        if isinstance(category_scores, dict):
            try:
                category_quality = float(category_scores.get(category))
            except (TypeError, ValueError):
                category_quality = None
        utility = quality * 100.0
        if category_quality is not None:
            utility += category_quality * 20.0
        if cost is not None:
            utility -= cost * (30.0 if resolved == "lite" else 8.0)
        if latency is not None:
            utility -= latency / (20_000.0 if resolved == "lite" else 60_000.0)
        ranked.append((utility, row))

    if ranked:
        _, chosen = max(ranked, key=lambda pair: pair[0])
        return StrategyDecision(
            requested=requested,
            resolved=resolved,
            preset=str(chosen["preset"]),
            category=category,
            complexity_score=complexity,
            source="benchmark",
            reason=f"highest benchmark utility among {', '.join(candidates)}",
            benchmark_metrics=dict(chosen),
        )

    fallback = "free" if resolved == "lite" and budget_usd == 0 else STRATEGY_FALLBACK[resolved]
    constrained = budget_usd is not None or max_latency_ms is not None
    return StrategyDecision(
        requested=requested,
        resolved=resolved,
        preset=fallback,
        category=category,
        complexity_score=complexity,
        source="fallback",
        reason=(
            "no benchmark row satisfies the requested constraints; fallback cannot guarantee them"
            if constrained else
            "no compatible benchmark results; using deterministic strategy fallback"
        ),
    )


# --- confidence, usage and result helpers ---------------------------------

def extract_confidence(text: str) -> float | None:
    match = CONFIDENCE_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2) or value > 1:
        value /= 100.0
    return max(0.0, min(1.0, value))


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def extract_usage(backend: str, payload: Mapping[str, Any]) -> tuple[int | None, int | None, int | None, float | None]:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None

    if backend in {"or", "openrouter", "openai"}:
        usage = payload.get("usage", {})
        if isinstance(usage, Mapping):
            input_tokens = _int_or_none(usage.get("prompt_tokens"))
            output_tokens = _int_or_none(usage.get("completion_tokens"))
            total_tokens = _int_or_none(usage.get("total_tokens"))
            try:
                raw_cost = usage.get("cost")
                cost = float(raw_cost) if raw_cost is not None else None
            except (TypeError, ValueError):
                cost = None
    elif backend == "anthropic":
        usage = payload.get("usage", {})
        if isinstance(usage, Mapping):
            input_tokens = _int_or_none(usage.get("input_tokens"))
            output_tokens = _int_or_none(usage.get("output_tokens"))
    elif backend == "google":
        usage = payload.get("usageMetadata", {})
        if isinstance(usage, Mapping):
            input_tokens = _int_or_none(usage.get("promptTokenCount"))
            output_tokens = _int_or_none(usage.get("candidatesTokenCount"))
            total_tokens = _int_or_none(usage.get("totalTokenCount"))

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens, cost


def successful_results(results: Sequence[ModelResult]) -> list[ModelResult]:
    return [item for item in results if item.ok and bool(item.answer.strip())]


def failed_results(results: Sequence[ModelResult]) -> list[ModelResult]:
    return [item for item in results if not (item.ok and bool(item.answer.strip()))]


def aggregate_metrics(results: Sequence[ModelResult], wall_latency_ms: int | None = None) -> dict[str, Any]:
    def sum_known(field: str) -> int | float | None:
        values = [getattr(item, field) for item in results if getattr(item, field) is not None]
        return sum(values) if values else None

    return {
        "calls": len(results),
        "successful_calls": sum(1 for item in results if item.ok),
        "failed_calls": sum(1 for item in results if not item.ok),
        "wall_latency_ms": wall_latency_ms,
        "summed_call_latency_ms": sum(item.latency_ms for item in results),
        "input_tokens": sum_known("input_tokens"),
        "output_tokens": sum_known("output_tokens"),
        "total_tokens": sum_known("total_tokens"),
        "cost_usd": sum_known("cost_usd"),
    }


