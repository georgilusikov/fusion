"""Bounded branch selection and expansion for Fusion deliberation."""

from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Mapping, Sequence

from .candidate_pool import render_candidate_pool
from .config import DispatchConfig, Member, ModelResult
from .dispatch import dispatch
from .judge import extract_json


BranchDispatcher = Callable[[Member, str, str, DispatchConfig, bool], ModelResult]


def _candidate_rows(pool: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = pool.get("candidates") if isinstance(pool, Mapping) else None
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping) and row.get("id") and row.get("claim")]


def select_branches(pool: Mapping[str, Any], *, limit: int = 3) -> list[dict[str, Any]]:
    """Select a small supported set plus one diversity wildcard.

    Source count is used only to allocate expansion compute, never as evidence that a claim is true.
    """
    if limit <= 0:
        return []
    rows = _candidate_rows(pool)
    if not rows:
        return []

    eligible = [row for row in rows if row.get("type") in {"solution", "cause", "consequence"}]
    if not eligible:
        eligible = rows

    def support_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
        source_count = len(row.get("source_ids") or [])
        operator_count = len(row.get("operators") or [])
        type_priority = 2 if row.get("type") == "solution" else 1 if row.get("type") == "cause" else 0
        return (source_count, operator_count, type_priority, str(row.get("id")))

    if limit == 1:
        return [max(eligible, key=support_key)]

    primary_count = min(limit - 1, len(eligible))
    primary = sorted(eligible, key=support_key, reverse=True)[:primary_count]
    selected_ids = {str(row["id"]) for row in primary}

    remaining = [row for row in eligible if str(row["id"]) not in selected_ids]
    if remaining and len(primary) < limit:
        used_types = {str(row.get("type")) for row in primary}
        used_ops = {op for row in primary for op in (row.get("operators") or [])}

        def wildcard_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
            type_novelty = int(str(row.get("type")) not in used_types)
            ops = set(str(op) for op in (row.get("operators") or []))
            operator_novelty = len(ops - used_ops)
            low_support = -len(row.get("source_ids") or [])
            return (type_novelty, operator_novelty, low_support, str(row.get("id")))

        primary.append(max(remaining, key=wildcard_key))

    if len(primary) < limit:
        selected_ids = {str(row["id"]) for row in primary}
        for row in rows:
            if str(row["id"]) not in selected_ids:
                primary.append(row)
                selected_ids.add(str(row["id"]))
            if len(primary) >= limit:
                break
    return primary[:limit]


def validate_branch_payload(payload: Mapping[str, Any] | None, expected_target: str | None = None) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["output is not a JSON object"]
    errors: list[str] = []
    target = payload.get("target")
    if not isinstance(target, str) or not target.strip():
        errors.append("target must be a non-empty string")
    elif expected_target is not None and target != expected_target:
        errors.append(f"target must be {expected_target}")
    if not isinstance(payload.get("thesis"), str):
        errors.append("thesis must be a string")
    for key in (
        "required_conditions",
        "direct_effects",
        "second_order_effects",
        "third_order_effects",
        "failure_conditions",
        "disconfirming_evidence",
    ):
        value = payload.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            errors.append(f"{key} must be an array of strings")
    return errors


def _select_members(members: Sequence[Member], count: int) -> list[Member]:
    if count <= 0:
        return []
    chosen: list[Member] = []
    seen_backends: set[str] = set()
    for member in members:
        if member.backend not in seen_backends:
            chosen.append(member)
            seen_backends.add(member.backend)
        if len(chosen) >= count:
            return chosen
    for member in members:
        if member not in chosen:
            chosen.append(member)
        if len(chosen) >= count:
            break
    return chosen


def branch_prompt(user_prompt: str, candidate: Mapping[str, Any], pool: Mapping[str, Any]) -> str:
    target = str(candidate.get("id"))
    return (
        "You are a branch expander. Deepen exactly one candidate without writing the final user answer. "
        "Do not reveal private chain-of-thought. Distinguish predictions from facts and expose assumptions.\n\n"
        f"## Original request\n{user_prompt}\n\n"
        f"## Target candidate\n{json.dumps(dict(candidate), ensure_ascii=False, indent=2)}\n\n"
        f"## Nearby candidate pool\n{render_candidate_pool(pool, max_candidates=12)}\n\n"
        "Trace this branch only. Return JSON only:\n"
        "{\n"
        f'  "target": "{target}",\n'
        '  "thesis": "best concise version of this branch",\n'
        '  "required_conditions": ["..."],\n'
        '  "direct_effects": ["..."],\n'
        '  "second_order_effects": ["..."],\n'
        '  "third_order_effects": ["..."],\n'
        '  "failure_conditions": ["..."],\n'
        '  "disconfirming_evidence": ["..."]\n'
        "}\n"
        "Do not invent certainty. Empty arrays are allowed when an effect cannot be defended."
    )


def render_branch_expansions(expansions: Sequence[Mapping[str, Any]]) -> str:
    valid = [row for row in expansions if isinstance(row, Mapping) and isinstance(row.get("payload"), Mapping)]
    if not valid:
        return ""
    lines = [
        "## Bounded branch expansions",
        "These are forecasts/hypotheses to test, not established facts.",
    ]
    for row in valid:
        payload = row["payload"]
        target = str(payload.get("target") or "?")
        lines.append(f"- {target}: {payload.get('thesis', '')}")
        for key, label in (
            ("required_conditions", "conditions"),
            ("direct_effects", "direct"),
            ("second_order_effects", "second-order"),
            ("third_order_effects", "third-order"),
            ("failure_conditions", "failure"),
            ("disconfirming_evidence", "disconfirming evidence"),
        ):
            values = payload.get(key) or []
            if values:
                lines.append(f"  {label}: " + "; ".join(str(item) for item in values[:4]))
    return "\n".join(lines)


def run_branch_expansions(
    prompt: str,
    pool: Mapping[str, Any],
    members: Sequence[Member],
    config: DispatchConfig,
    depth: str,
    *,
    count: int = 3,
    dispatcher: BranchDispatcher = dispatch,
) -> tuple[dict[str, Any], list[ModelResult]]:
    candidates = select_branches(pool, limit=count)
    branch_members = _select_members(members, len(candidates))
    assignments = list(zip(candidates, branch_members))
    if not assignments:
        return {"selected": [], "expansions": [], "valid_expansions": 0, "context": ""}, []

    rows: list[dict[str, Any] | None] = [None] * len(assignments)
    results: list[ModelResult | None] = [None] * len(assignments)

    def run_one(index: int, candidate: Mapping[str, Any], member: Member) -> None:
        target = str(candidate.get("id"))
        raw_result = dispatcher(member, branch_prompt(prompt, candidate, pool), depth, config, False)
        result = dataclasses.replace(
            raw_result,
            label=f"{member.label}:branch:{target}",
            metadata={**raw_result.metadata, "branch_target": target, "source_member": member.label},
        )
        payload = extract_json(result.answer) if result.ok else None
        errors = validate_branch_payload(payload, expected_target=target)
        rows[index] = {
            "target": target,
            "member_label": member.label,
            "payload": payload if not errors else None,
            "validation_errors": errors,
        }
        results[index] = result

    with ThreadPoolExecutor(max_workers=max(1, len(assignments))) as executor:
        futures = [
            executor.submit(run_one, index, candidate, member)
            for index, (candidate, member) in enumerate(assignments)
        ]
        for future in as_completed(futures):
            future.result()

    clean_rows = [row for row in rows if row is not None]
    valid_count = sum(1 for row in clean_rows if isinstance(row.get("payload"), Mapping))
    return {
        "selected": [str(candidate.get("id")) for candidate in candidates],
        "expansions": clean_rows,
        "valid_expansions": valid_count,
        "context": render_branch_expansions(clean_rows),
    }, [result for result in results if result is not None]
