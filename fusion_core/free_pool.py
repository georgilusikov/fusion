"""Dynamic OpenRouter free-model discovery for optional Fusion panels."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import PRESETS

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_CACHE_TTL = 6 * 60 * 60
DEFAULT_MIN_CONTEXT = 32_000
DEFAULT_POOL_SIZE = 8
ROLE_CYCLE = ("builder", "expert", "falsifier", "contrarian", "generalist", "skeptic")


def default_cache_path() -> Path:
    override = os.environ.get("FUSION_FREE_POOL_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "fusion" / "openrouter-free-models.json"


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_zero_price(model: Mapping[str, Any]) -> bool:
    model_id = str(model.get("id") or "")
    if model_id.endswith(":free"):
        return True
    pricing = model.get("pricing")
    if not isinstance(pricing, Mapping):
        return False
    prompt = _as_float(pricing.get("prompt"))
    completion = _as_float(pricing.get("completion"))
    return prompt == 0.0 and completion == 0.0


def _supports_text_output(model: Mapping[str, Any]) -> bool:
    architecture = model.get("architecture")
    if isinstance(architecture, Mapping):
        outputs = architecture.get("output_modalities")
        if isinstance(outputs, list) and outputs:
            return "text" in {str(item).casefold() for item in outputs}
        modality = architecture.get("modality")
        if isinstance(modality, str) and modality:
            return "text" in modality.casefold()
    model_id = str(model.get("id") or "").casefold()
    blocked = ("embedding", "rerank", "tts", "speech", "transcription")
    return not any(term in model_id for term in blocked)


def _context_length(model: Mapping[str, Any]) -> int:
    value = model.get("context_length")
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _provider_family(model_id: str) -> str:
    return model_id.split("/", 1)[0].casefold() if "/" in model_id else model_id.casefold()


def _capability_score(model: Mapping[str, Any]) -> tuple[int, int, str]:
    """Sort by useful API features, then context. Not a quality benchmark."""
    supported = model.get("supported_parameters")
    supported_set = {str(item).casefold() for item in supported} if isinstance(supported, list) else set()
    feature_score = 0
    if "reasoning" in supported_set:
        feature_score += 3
    if "structured_outputs" in supported_set or "response_format" in supported_set:
        feature_score += 2
    if "tools" in supported_set or "tool_choice" in supported_set:
        feature_score += 1
    return feature_score, _context_length(model), str(model.get("id") or "")


def normalize_free_models(
    payload: Mapping[str, Any],
    *,
    min_context: int = DEFAULT_MIN_CONTEXT,
) -> list[dict[str, Any]]:
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError("OpenRouter models response has no data array")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        model_id = str(raw.get("id") or "").strip()
        if not model_id or model_id == "openrouter/free" or model_id in seen:
            continue
        if not _is_zero_price(raw) or not _supports_text_output(raw):
            continue
        context = _context_length(raw)
        if context < max(0, min_context):
            continue
        seen.add(model_id)
        result.append(
            {
                "id": model_id,
                "name": str(raw.get("name") or model_id),
                "context_length": context,
                "provider_family": _provider_family(model_id),
                "supported_parameters": list(raw.get("supported_parameters") or []),
                "capability_score": _capability_score(raw)[0],
            }
        )
    result.sort(
        key=lambda item: (
            -int(item.get("capability_score") or 0),
            -int(item.get("context_length") or 0),
            str(item.get("id") or ""),
        )
    )
    return result


def _http_models(timeout: int = 20) -> Mapping[str, Any]:
    headers = {"Accept": "application/json", "User-Agent": "fusion-free-pool/1"}
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(OPENROUTER_MODELS_URL, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("OpenRouter models response is not an object")
    return payload


def _load_cache(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_cache(path: Path, models: Sequence[Mapping[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"fetched_at": time.time(), "models": list(models)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        # Discovery should still work if the cache directory is read-only.
        return


def _cached_models(path: Path, *, ttl: int, allow_stale: bool) -> list[dict[str, Any]]:
    payload = _load_cache(path)
    if not payload:
        return []
    fetched_at = _as_float(payload.get("fetched_at")) or 0.0
    if not allow_stale and time.time() - fetched_at > max(0, ttl):
        return []
    rows = payload.get("models")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping) and row.get("id")]


def fallback_free_model_ids() -> list[str]:
    """Extract the repository's static free preset for offline fallback."""
    ids: list[str] = []
    for token in PRESETS.get("free", "").split(","):
        member = token.strip().split("@", 1)[0]
        if not member.startswith("or:"):
            continue
        model_id = member[3:]
        if model_id and model_id not in ids:
            ids.append(model_id)
    return ids


def discover_free_models(
    *,
    min_context: int = DEFAULT_MIN_CONTEXT,
    cache_path: Path | None = None,
    cache_ttl: int = DEFAULT_CACHE_TTL,
    refresh: bool = False,
    timeout: int = 20,
    request_fn: Callable[[int], Mapping[str, Any]] = _http_models,
) -> tuple[list[dict[str, Any]], str]:
    path = cache_path or default_cache_path()
    if not refresh:
        cached = _cached_models(path, ttl=cache_ttl, allow_stale=False)
        if cached:
            return cached, "cache"
    try:
        payload = request_fn(timeout)
        models = normalize_free_models(payload, min_context=min_context)
        if models:
            _write_cache(path, models)
            return models, "openrouter"
    except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        pass
    stale = _cached_models(path, ttl=cache_ttl, allow_stale=True)
    if stale:
        return stale, "stale-cache"
    fallback = [
        {
            "id": model_id,
            "name": model_id,
            "context_length": 0,
            "provider_family": _provider_family(model_id),
            "supported_parameters": [],
            "capability_score": 0,
        }
        for model_id in fallback_free_model_ids()
    ]
    return fallback, "static-fallback"


def select_diverse_models(models: Sequence[Mapping[str, Any]], size: int = DEFAULT_POOL_SIZE) -> list[dict[str, Any]]:
    """Prefer one model per provider family before filling remaining slots."""
    if size <= 0:
        return []
    ordered = [dict(model) for model in models if model.get("id")]
    chosen: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for model in ordered:
        family = str(model.get("provider_family") or _provider_family(str(model["id"]))).casefold()
        if family in seen_families:
            continue
        chosen.append(model)
        seen_families.add(family)
        if len(chosen) >= size:
            return chosen
    chosen_ids = {str(model["id"]) for model in chosen}
    for model in ordered:
        if str(model["id"]) in chosen_ids:
            continue
        chosen.append(model)
        if len(chosen) >= size:
            break
    return chosen


def build_free_panel_spec(models: Sequence[Mapping[str, Any]], *, size: int = DEFAULT_POOL_SIZE) -> str:
    selected = select_diverse_models(models, size=size)
    return ",".join(
        f"or:{model['id']}@{ROLE_CYCLE[index % len(ROLE_CYCLE)]}"
        for index, model in enumerate(selected)
    )


def resolve_free_panel(
    *,
    size: int = DEFAULT_POOL_SIZE,
    min_context: int = DEFAULT_MIN_CONTEXT,
    cache_path: Path | None = None,
    cache_ttl: int = DEFAULT_CACHE_TTL,
    refresh: bool = False,
    timeout: int = 20,
    request_fn: Callable[[int], Mapping[str, Any]] = _http_models,
) -> tuple[str, dict[str, Any]]:
    models, source = discover_free_models(
        min_context=min_context,
        cache_path=cache_path,
        cache_ttl=cache_ttl,
        refresh=refresh,
        timeout=timeout,
        request_fn=request_fn,
    )
    selected = select_diverse_models(models, size=size)
    panel = build_free_panel_spec(selected, size=len(selected))
    return panel, {
        "source": source,
        "available": len(models),
        "selected": selected,
        "panel_spec": panel,
    }
