"""Targeted critique operators for deliberation candidate pools."""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .candidate_pool import render_candidate_pool
from .config import DispatchConfig, Member, ModelResult
from .dispatch import dispatch
from .judge import extract_json


@dataclass(frozen=True, slots=True)
class CriticSpec:
    key: str
    title: str
    instruction: str


CRITICS: dict[str, CriticSpec] = {
    "assumption": CriticSpec("assumption", "Assumption critic", "Identify hidden assumptions that could invalidate attractive candidates."),
    "evidence": CriticSpec("evidence", "Evidence critic", "Identify unsupported claims and the evidence most needed to discriminate between candidates."),
    "causal": CriticSpec("causal", "Causal critic", "Attack weak causal links, missing confounders, and unjustified consequence chains."),
    "feasibility": CriticSpec("feasibility", "Feasibility critic", "Stress-test implementation constraints, dependencies, costs, and operational failure modes."),
    "adversarial": CriticSpec("adversarial", "Adversarial critic", "Find how competitors, users, or hostile actors could exploit or break the proposal."),
    "stakeholder": CriticSpec("stakeholder", "Stakeholder critic", "Check whether the candidate ignores incentives or reactions of affected actors."),
}

_DEFAULT_CRITICS = ("assumption", "evidence", "causal", "feasibility")
CriticDispatcher = Callable[[Member, str, str, DispatchConfig, bool], ModelResult]


def plan_critics(prompt: str, pool: Mapping[str, Any], *, limit: int = 2) -> list[CriticSpec]:
    if limit <= 0:
        return []
    selected: list[str] = []
    text = prompt.casefold()
    rows = pool.get("candidates") if isinstance(pool, Mapping) else []
    types = {str(row.get("type")) for row in rows if isinstance(row, Mapping) and row.get("type")}

    def add(key: str) -> None:
        if key in CRITICS and key not in selected and len(selected) < limit:
            selected.append(key)

    if {"cause", "consequence"} & types:
        add("causal")
    if "solution" in types:
        add("feasibility")
    if any(term in text for term in ("user", "customer", "competitor", "market", "клиент", "пользоват", "конкурент", "рынок")):
        add("stakeholder")
    for key in _DEFAULT_CRITICS:
        add(key)
    return [CRITICS[key] for key in selected[:limit]]


def validate_critique_payload(payload: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["output is not a JSON object"]
    errors: list[str] = []
    if not isinstance(payload.get("summary"), str):
        errors.append("summary must be a string")
    findings = payload.get("findings")
    if not isinstance(findings, list):
        errors.append("findings must be an array")
        return errors
    for index, row in enumerate(findings):
        prefix = f"findings[{index}]"
        if not isinstance(row, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        if not isinstance(row.get("target"), str) or not str(row.get("target")).strip():
            errors.append(f"{prefix}.target must be a non-empty string")
        if not isinstance(row.get("objection"), str) or not str(row.get("objection")).strip():
            errors.append(f"{prefix}.objection must be a non-empty string")
        severity = row.get("severity")
        if not isinstance(severity, (int, float)) or isinstance(severity, bool) or not 0 <= float(severity) <= 1:
            errors.append(f"{prefix}.severity must be a number from 0 to 1")
        for key in ("hidden_assumptions", "missing_evidence"):
            value = row.get(key, [])
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                errors.append(f"{prefix}.{key} must be an array of strings")
    return errors


def _select_members(members: Sequence[Member], count: int) -> list[Member]:
    chosen: list[Member] = []
    seen_backends: set[str] = set()
    for member in reversed(list(members)):
        if member.backend not in seen_backends:
            chosen.append(member)
            seen_backends.add(member.backend)
        if len(chosen) >= count:
            return chosen
    for member in reversed(list(members)):
        if member not in chosen:
            chosen.append(member)
        if len(chosen) >= count:
            break
    return chosen


def critic_prompt(
    user_prompt: str,
    critic: CriticSpec,
    pool: Mapping[str, Any],
    expansion_context: str | None = None,
) -> str:
    expansion_text = ""
    if expansion_context and expansion_context.strip():
        expansion_text = f"\n\n{expansion_context.strip()}\n"
    return (
        "You are a targeted critic. Attack the candidate pool using only the assigned critic lens. "
        "Do not produce a replacement final answer and do not reveal private chain-of-thought. "
        "Return concise, decision-relevant objections.\n\n"
        f"## Critic\n{critic.title}: {critic.instruction}\n\n"
        f"## Original request\n{user_prompt}\n\n"
        f"{render_candidate_pool(pool, max_candidates=16)}{expansion_text}\n\n"
        "Return JSON only with this shape:\n"
        "{\n"
        '  "summary": "one concise synthesis",\n'
        '  "findings": [\n'
        "    {\n"
        '      "target": "C1",\n'
        '      "objection": "specific problem",\n'
        '      "severity": 0.0,\n'
        '      "hidden_assumptions": ["..."],\n'
        '      "missing_evidence": ["..."]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Use candidate IDs exactly. Severity 1.0 means potentially fatal; 0.0 means negligible."
    )


def run_targeted_critics(
    prompt: str,
    pool: Mapping[str, Any],
    members: Sequence[Member],
    config: DispatchConfig,
    depth: str,
    *,
    count: int = 2,
    expansion_context: str | None = None,
    dispatcher: CriticDispatcher = dispatch,
) -> tuple[dict[str, Any], list[ModelResult]]:
    if not pool.get("candidates") or count <= 0:
        return {"critics": [], "valid_critics": 0, "findings": [], "context": ""}, []

    critic_specs = plan_critics(prompt, pool, limit=count)
    critic_members = _select_members(members, len(critic_specs))
    assignments = list(zip(critic_specs, critic_members))
    rows: list[dict[str, Any] | None] = [None] * len(assignments)
    results: list[ModelResult | None] = [None] * len(assignments)

    def run_one(index: int, critic: CriticSpec, member: Member) -> None:
        raw_result = dispatcher(member, critic_prompt(prompt, critic, pool, expansion_context), depth, config, False)
        result = dataclasses.replace(
            raw_result,
            label=f"{member.label}:critic:{critic.key}",
            metadata={**raw_result.metadata, "critic": critic.key, "source_member": member.label},
        )
        payload = extract_json(result.answer) if result.ok else None
        errors = validate_critique_payload(payload)
        rows[index] = {"critic": critic.key, "member_label": member.label, "payload": payload if not errors else None, "validation_errors": errors}
        results[index] = result

    with ThreadPoolExecutor(max_workers=max(1, len(assignments))) as executor:
        futures = [executor.submit(run_one, index, critic, member) for index, (critic, member) in enumerate(assignments)]
        for future in as_completed(futures):
            future.result()

    clean_rows = [row for row in rows if row is not None]
    findings: list[dict[str, Any]] = []
    valid_critics = 0
    for row in clean_rows:
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            continue
        valid_critics += 1
        for finding in payload.get("findings", []):
            if not isinstance(finding, Mapping):
                continue
            item = dict(finding)
            item["critic"] = row["critic"]
            item["member_label"] = row["member_label"]
            findings.append(item)
    findings.sort(key=lambda item: float(item.get("severity", 0.0)), reverse=True)

    lines: list[str] = []
    if findings:
        lines.extend(["## Targeted critique findings", "These are objections to test, not authoritative facts."])
        for finding in findings[:12]:
            lines.append(f"- {finding.get('target', '?')} severity={float(finding.get('severity', 0.0)):.2f} [{finding.get('critic', 'critic')}]: {finding.get('objection', '')}")
            assumptions = finding.get("hidden_assumptions") or []
            if assumptions:
                lines.append("  hidden assumptions: " + "; ".join(str(item) for item in assumptions[:3]))
            evidence = finding.get("missing_evidence") or []
            if evidence:
                lines.append("  missing evidence: " + "; ".join(str(item) for item in evidence[:3]))

    return {
        "critics": clean_rows,
        "valid_critics": valid_critics,
        "findings": findings,
        "context": "\n".join(lines),
    }, [result for result in results if result is not None]
