"""Structured scout parsing and candidate-pool construction."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


CANDIDATE_TYPES = {"solution", "cause", "risk", "assumption", "consequence"}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def validate_scout_payload(payload: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["output is not a JSON object"]
    errors: list[str] = []
    if not isinstance(payload.get("summary"), str):
        errors.append("summary must be a string")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        errors.append("candidates must be an array")
        return errors
    for index, row in enumerate(candidates):
        prefix = f"candidates[{index}]"
        if not isinstance(row, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        candidate_type = row.get("type")
        if candidate_type not in CANDIDATE_TYPES:
            errors.append(f"{prefix}.type must be one of {sorted(CANDIDATE_TYPES)}")
        if not isinstance(row.get("claim"), str) or not str(row.get("claim")).strip():
            errors.append(f"{prefix}.claim must be a non-empty string")
        for key in ("assumptions", "evidence_needed"):
            value = row.get(key, [])
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                errors.append(f"{prefix}.{key} must be an array of strings")
        horizon = row.get("horizon", 0)
        if not isinstance(horizon, int) or isinstance(horizon, bool) or not 0 <= horizon <= 3:
            errors.append(f"{prefix}.horizon must be an integer from 0 to 3")
        parent = row.get("parent")
        if parent is not None and not isinstance(parent, str):
            errors.append(f"{prefix}.parent must be a string or null")
    return errors


def _fingerprint(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.casefold(), flags=re.UNICODE).strip()


def _merge_unique(target: list[str], incoming: Sequence[str]) -> None:
    seen = set(target)
    for item in incoming:
        if item not in seen:
            target.append(item)
            seen.add(item)


def build_candidate_pool(
    scout_rows: Sequence[Mapping[str, Any]],
    *,
    max_candidates: int = 24,
) -> dict[str, Any]:
    """Merge normalized duplicate claims while retaining provenance."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    valid_scouts = 0

    for scout in scout_rows:
        payload = scout.get("payload")
        if validate_scout_payload(payload if isinstance(payload, Mapping) else None):
            continue
        assert isinstance(payload, Mapping)
        valid_scouts += 1
        source_id = str(scout.get("source_id") or "").strip()
        operator = str(scout.get("operator") or "").strip()
        for row in payload.get("candidates", []):
            if not isinstance(row, Mapping):
                continue
            claim = str(row.get("claim") or "").strip()
            candidate_type = str(row.get("type") or "").strip()
            key = (candidate_type, _fingerprint(claim))
            if not key[1]:
                continue
            if key not in merged:
                merged[key] = {
                    "id": f"C{len(order) + 1}",
                    "type": candidate_type,
                    "claim": claim,
                    "why_it_matters": str(row.get("why_it_matters") or "").strip(),
                    "assumptions": _string_list(row.get("assumptions", [])),
                    "evidence_needed": _string_list(row.get("evidence_needed", [])),
                    "parent": row.get("parent") if isinstance(row.get("parent"), str) else None,
                    "horizon": int(row.get("horizon", 0)),
                    "source_ids": [source_id] if source_id else [],
                    "operators": [operator] if operator else [],
                }
                order.append(key)
                if len(order) >= max_candidates:
                    break
            else:
                item = merged[key]
                _merge_unique(item["assumptions"], _string_list(row.get("assumptions", [])))
                _merge_unique(item["evidence_needed"], _string_list(row.get("evidence_needed", [])))
                if source_id:
                    _merge_unique(item["source_ids"], [source_id])
                if operator:
                    _merge_unique(item["operators"], [operator])
        if len(order) >= max_candidates:
            break

    candidates = [merged[key] for key in order]
    return {
        "candidate_count": len(candidates),
        "valid_scouts": valid_scouts,
        "candidates": candidates,
    }


def render_candidate_pool(pool: Mapping[str, Any], *, max_candidates: int = 12) -> str:
    rows = pool.get("candidates") if isinstance(pool, Mapping) else None
    if not isinstance(rows, list) or not rows:
        return ""
    lines = [
        "## Deliberation candidate pool",
        "Use these as donor insights, not as authoritative facts. Preserve provenance and challenge assumptions.",
    ]
    for row in rows[:max_candidates]:
        if not isinstance(row, Mapping):
            continue
        cid = str(row.get("id") or "?")
        kind = str(row.get("type") or "candidate")
        claim = str(row.get("claim") or "").strip()
        horizon = row.get("horizon", 0)
        sources = ", ".join(str(item) for item in row.get("operators", []) if str(item))
        lines.append(f"- {cid} [{kind}, horizon={horizon}] {claim}")
        if sources:
            lines.append(f"  operators: {sources}")
        assumptions = row.get("assumptions") or []
        if assumptions:
            lines.append("  assumptions: " + "; ".join(str(item) for item in assumptions[:3]))
        evidence = row.get("evidence_needed") or []
        if evidence:
            lines.append("  evidence needed: " + "; ".join(str(item) for item in evidence[:3]))
    return "\n".join(lines)
