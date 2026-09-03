"""Experimental deliberative scouts for Fusion v2."""

from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Mapping, Sequence

from .candidate_pool import build_candidate_pool, render_candidate_pool, validate_scout_payload
from .config import DispatchConfig, Member, ModelResult
from .dispatch import dispatch
from .judge import extract_json
from .operators import OperatorSpec, plan_operators


ScoutDispatcher = Callable[[Member, str, str, DispatchConfig, bool], ModelResult]


def _judge_brief(judge: Mapping[str, Any] | None) -> str:
    if not isinstance(judge, Mapping):
        return "{}"
    parsed = judge.get("parsed")
    if not isinstance(parsed, Mapping):
        return "{}"
    compact = {
        "coverage_gaps": parsed.get("coverage_gaps") or [],
        "blind_spots": parsed.get("blind_spots") or [],
        "contradictions": parsed.get("contradictions") or [],
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


def scout_prompt(
    user_prompt: str,
    operator: OperatorSpec,
    judge: Mapping[str, Any] | None,
) -> str:
    return (
        "You are a deliberation scout. Explore only the assigned operator. "
        "Do not write a polished final answer and do not reveal private chain-of-thought. "
        "Return concise decision-relevant claims, assumptions, evidence needs, and consequences.\n\n"
        f"## Operator\n{operator.title}: {operator.instruction}\n\n"
        f"## Original request\n{user_prompt}\n\n"
        f"## Known gaps from the first judge\n{_judge_brief(judge)}\n\n"
        "Return JSON only with this shape:\n"
        "{\n"
        '  "summary": "one concise synthesis",\n'
        '  "candidates": [\n'
        "    {\n"
        '      "type": "solution|cause|risk|assumption|consequence",\n'
        '      "claim": "one concrete claim",\n'
        '      "why_it_matters": "brief reason",\n'
        '      "assumptions": ["..."],\n'
        '      "evidence_needed": ["..."],\n'
        '      "parent": null,\n'
        '      "horizon": 0\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "horizon: 0=current option/cause, 1=direct effect, 2=second-order, 3=third-order. "
        "Keep candidates materially distinct."
    )


def select_scout_members(members: Sequence[Member], count: int) -> list[Member]:
    """Prefer backend diversity, then fill remaining slots in configured order."""
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


def run_scouts(
    prompt: str,
    members: Sequence[Member],
    judge: Mapping[str, Any] | None,
    config: DispatchConfig,
    depth: str,
    *,
    max_operators: int = 4,
    dispatcher: ScoutDispatcher = dispatch,
) -> tuple[list[dict[str, Any]], list[ModelResult]]:
    operators = plan_operators(prompt, judge, limit=max_operators)
    scout_members = select_scout_members(members, len(operators))
    assignments = list(zip(operators, scout_members))
    if not assignments:
        return [], []

    rows: list[dict[str, Any] | None] = [None] * len(assignments)
    results: list[ModelResult | None] = [None] * len(assignments)

    def run_one(index: int, operator: OperatorSpec, member: Member) -> None:
        raw_result = dispatcher(
            member,
            scout_prompt(prompt, operator, judge),
            depth,
            config,
            False,
        )
        result = dataclasses.replace(
            raw_result,
            label=f"{member.label}:scout:{operator.key}",
            metadata={**raw_result.metadata, "operator": operator.key, "source_member": member.label},
        )
        payload = extract_json(result.answer) if result.ok else None
        errors = validate_scout_payload(payload)
        source_id = f"S{index + 1}"
        rows[index] = {
            "source_id": source_id,
            "operator": operator.key,
            "member_label": member.label,
            "payload": payload if not errors else None,
            "validation_errors": errors,
            "summary": str(payload.get("summary") or "") if isinstance(payload, Mapping) and not errors else "",
        }
        results[index] = result

    with ThreadPoolExecutor(max_workers=max(1, len(assignments))) as executor:
        futures = [
            executor.submit(run_one, index, operator, member)
            for index, (operator, member) in enumerate(assignments)
        ]
        for future in as_completed(futures):
            future.result()

    return (
        [row for row in rows if row is not None],
        [result for result in results if result is not None],
    )


def run_deliberation(
    prompt: str,
    members: Sequence[Member],
    judge: Mapping[str, Any] | None,
    config: DispatchConfig,
    depth: str,
    *,
    max_operators: int = 4,
    dispatcher: ScoutDispatcher = dispatch,
) -> tuple[dict[str, Any], list[ModelResult]]:
    scouts, results = run_scouts(
        prompt,
        members,
        judge,
        config,
        depth,
        max_operators=max_operators,
        dispatcher=dispatcher,
    )
    pool = build_candidate_pool(scouts)
    return {
        "operators": [row["operator"] for row in scouts],
        "scouts": scouts,
        "pool": pool,
        "donor_context": render_candidate_pool(pool),
    }, results
