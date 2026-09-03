"""Fusion panel, judge, and drafter pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import PRESETS, DispatchConfig, Member, ModelResult
from .deliberation import run_deliberation
from .dispatch import dispatch
from .draft_gate import run_draft_gate
from .judge_panel import run_judge_panel
from .selection import best_panel_result
from .self_consistency import expand_panel_spec
from .rounds import combine_unique_results, escalation_reasons, review_round
from .routing import (
    aggregate_metrics, failed_results, load_pricing, parse_member,
    parse_panel, select_strategy, successful_results,
)

# --- orchestration ---------------------------------------------------------

def _run_panel(
    members: Sequence[Member],
    prompt: str,
    depth: str,
    config: DispatchConfig,
) -> list[ModelResult]:
    panel_results: list[ModelResult] = []
    with ThreadPoolExecutor(max_workers=max(1, len(members))) as executor:
        futures = {
            executor.submit(dispatch, member, prompt, depth, config): member.label
            for member in members
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                label = futures[future]
                result = ModelResult(
                    label=label,
                    backend="unknown",
                    kind="unknown",
                    ok=False,
                    errors=[f"unhandled dispatch error: {type(exc).__name__}: {exc}"],
                )
            panel_results.append(result)
            print(
                f"[fusion] panel {result.label}: {'ok' if result.ok else 'FAIL'} "
                f"attempts={result.attempts} latency={result.latency_ms}ms",
                file=sys.stderr,
            )

    order = {member.label: index for index, member in enumerate(members)}
    panel_results.sort(key=lambda item: order.get(item.label, len(order)))
    return panel_results


def _append_model_call_result(payload: Mapping[str, Any], sink: list[ModelResult]) -> None:
    result = payload.get("result")
    if isinstance(result, dict):
        sink.append(ModelResult(**result))
    for repair in payload.get("repair_results", []):
        if isinstance(repair, dict):
            sink.append(ModelResult(**repair))


def _append_judge_metrics(judge: Mapping[str, Any], sink: list[ModelResult]) -> None:
    runs = judge.get("judge_results")
    if isinstance(runs, list):
        for run in runs:
            if isinstance(run, Mapping):
                _append_model_call_result(run, sink)
        return
    _append_model_call_result(judge, sink)


def _draft_prompt(
    user_prompt: str,
    judge: Mapping[str, Any],
    panel: Sequence[ModelResult],
    include_panel: bool,
    best_result: ModelResult | None = None,
) -> str:
    judge_text = json.dumps(judge.get("parsed"), ensure_ascii=False, indent=2) if judge.get("valid") else str(judge.get("raw", ""))
    best_text = ""
    if best_result is not None:
        best_text = (
            f"\n\n## Best source answer to preserve as the base\n"
            f"### {best_result.label}\n{best_result.answer}\n"
        )
    panel_text = ""
    if include_panel:
        panel_text = "\n\n## Other source answers as donor material\n" + "\n\n".join(
            f"### {item.label}\n{item.answer}" for item in successful_results(panel)
            if best_result is None or item.label != best_result.label
        )
    return (
        "You are the drafter. Answer the original prompt. Use the best source answer as the base. "
        "Only add donor insights from other answers when they improve correctness, depth, coverage, or actionability. "
        "Resolve contradictions explicitly, fill coverage gaps, and avoid shared blind spots. "
        "Do not mention this orchestration.\n\n"
        f"## Original prompt\n{user_prompt}\n\n"
        f"## Judge analysis\n{judge_text}{best_text}{panel_text}\n\n"
        "Write the final answer now."
    )


def run_fusion(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.perf_counter()
    prompt = args.prompt or sys.stdin.read().strip()
    if not prompt:
        return {"error": "empty prompt"}, 2

    blind_judge = bool(getattr(args, "blind_judge", False))
    deliberation_mode = str(getattr(args, "deliberation", "off"))
    max_scouts = int(getattr(args, "scouts", 4))

    benchmark_path = Path(args.benchmark_results).expanduser() if args.benchmark_results else None
    decision = select_strategy(
        prompt=prompt,
        requested=args.strategy,
        explicit_preset=args.preset,
        benchmark_path=benchmark_path,
        budget_usd=args.budget_usd,
        max_latency_ms=args.max_latency_ms,
        complexity_threshold=args.complexity_threshold,
    )

    if args.panel:
        panel_spec = args.panel
        decision.preset = None
        decision.source = "explicit-panel"
        decision.reason = "explicit --panel overrides preset selection"
    else:
        assert decision.preset is not None
        panel_spec = PRESETS[decision.preset]
    panel_spec = expand_panel_spec(panel_spec)

    config = DispatchConfig(
        timeout=args.timeout,
        retries=args.retries,
        backoff=args.backoff,
        reasoning=args.reasoning == "on",
        strategy=decision.resolved,
        agent_workspace=args.agent_workspace,
        workspace_source=Path(args.workspace_source).expanduser(),
        pricing=load_pricing(),
    )

    try:
        members = parse_panel(panel_spec, same_mode=args.mode == "same")
        if not members:
            raise ValueError("panel is empty")
        judge_member = parse_member(args.judge, same_mode=True, seen={})
        critic_spec = getattr(args, "critics", None)
        if critic_spec is None and decision.resolved == "pro":
            critic_spec = "claude,gemini,codex"
        judge_members = parse_panel(expand_panel_spec(critic_spec), same_mode=True) if critic_spec else [judge_member]
        drafter_member = parse_member(args.auto_draft, same_mode=True, seen={}) if args.auto_draft else None
    except ValueError as exc:
        return {"error": str(exc), "selection": decision.to_dict()}, 2

    if args.dry_run:
        return {
            "prompt": prompt,
            "selection": decision.to_dict(),
            "panel_spec": panel_spec,
            "members": [dataclasses.asdict(member) for member in members],
            "judge": dataclasses.asdict(judge_member),
            "judge_members": [dataclasses.asdict(member) for member in judge_members],
            "blind_judge": blind_judge,
            "deliberation": deliberation_mode,
            "scouts": max_scouts,
            "drafter": dataclasses.asdict(drafter_member) if drafter_member else None,
        }, 0

    print(
        f"[fusion] strategy={decision.resolved} source={decision.source} "
        f"preset={decision.preset or 'custom'} members={[member.label for member in members]} "
        f"judges={[member.label for member in judge_members]} blind_judge={blind_judge} "
        f"deliberation={deliberation_mode}",
        file=sys.stderr,
    )

    panel_results = _run_panel(members, prompt, args.depth, config)
    rounds: list[dict[str, Any]] = [
        {
            "name": "initial",
            "members": [member.label for member in members],
            "judge_valid": None,
        }
    ]

    effective_repairs = max(args.repair_attempts, 2) if decision.resolved == "pro" else args.repair_attempts
    judge = run_judge_panel(
        judge_members,
        prompt,
        successful_results(panel_results),
        config,
        repair_attempts=effective_repairs,
        blind=blind_judge,
    )
    rounds[0]["judge_valid"] = bool(judge.get("valid"))

    deliberation_bundle: dict[str, Any] | None = None
    deliberation_results: list[ModelResult] = []
    donor_context = ""
    if deliberation_mode == "on" and max_scouts > 0 and successful_results(panel_results):
        deliberation_bundle, deliberation_results = run_deliberation(
            prompt,
            members,
            judge,
            config,
            args.depth,
            max_operators=max_scouts,
        )
        donor_context = str(deliberation_bundle.get("donor_context") or "")
        pool = deliberation_bundle.get("pool") or {}
        rounds.append(
            {
                "name": "deliberation-scouts",
                "operators": list(deliberation_bundle.get("operators") or []),
                "candidate_count": int(pool.get("candidate_count") or 0) if isinstance(pool, Mapping) else 0,
                "valid_scouts": int(pool.get("valid_scouts") or 0) if isinstance(pool, Mapping) else 0,
            }
        )

    reviewer_count = (
        args.reviewers
        if args.reviewers is not None
        else (2 if decision.resolved == "pro" else 0)
    )
    if (
        args.reviewers is None
        and deliberation_mode == "on"
        and donor_context
        and successful_results(panel_results)
    ):
        reviewer_count = max(reviewer_count, min(2, len(successful_results(panel_results))))

    if reviewer_count > 0 and successful_results(panel_results):
        reviews = review_round(
            prompt,
            members,
            panel_results,
            judge,
            args.depth,
            config,
            reviewer_count,
            log=lambda message: print(f"[fusion] {message}", file=sys.stderr),
            donor_context=donor_context or None,
        )
        panel_results.extend(reviews)
        if successful_results(reviews):
            judge = run_judge_panel(
                judge_members,
                prompt,
                successful_results(panel_results),
                config,
                repair_attempts=effective_repairs,
                blind=blind_judge,
            )
        rounds.append(
            {
                "name": "review",
                "members": [item.label for item in reviews],
                "judge_valid": bool(judge.get("valid")),
                "deliberation_donor_used": bool(donor_context),
            }
        )

    explicit_panel = args.panel is not None or args.preset is not None
    reasons = escalation_reasons(panel_results, judge)
    if args.strategy == "adaptive" and reasons and not args.no_escalate and not explicit_panel:
        print(f"[fusion] adaptive escalation: {', '.join(reasons)}", file=sys.stderr)
        power_members = parse_panel(expand_panel_spec(PRESETS["power"]), same_mode=args.mode == "same")
        escalated = _run_panel(power_members, prompt, args.depth, config)
        panel_results = combine_unique_results(panel_results, escalated)
        judge = run_judge_panel(
            judge_members,
            prompt,
            successful_results(panel_results),
            config,
            repair_attempts=effective_repairs,
            blind=blind_judge,
        )
        rounds.append(
            {
                "name": "adaptive-escalation",
                "reasons": reasons,
                "members": [item.label for item in escalated],
                "judge_valid": bool(judge.get("valid")),
            }
        )

    successes = successful_results(panel_results)
    failures = failed_results(panel_results)
    best_result = best_panel_result(successes, judge)

    all_results = list(panel_results) + deliberation_results
    _append_judge_metrics(judge, all_results)

    bundle: dict[str, Any] = {
        "prompt": prompt,
        "strategy": decision.resolved,
        "selection": decision.to_dict(),
        "depth": args.depth,
        "reasoning": args.reasoning,
        "blind_judge": blind_judge,
        "deliberation_mode": deliberation_mode,
        "panel": [item.to_dict() for item in panel_results],
        "successful_panel": [item.label for item in successes],
        "failed_panel": [item.label for item in failures],
        "best_panel_answer_label": best_result.label if best_result is not None else None,
        "rounds": rounds,
        "judge": judge,
    }
    if deliberation_bundle is not None:
        bundle["deliberation"] = deliberation_bundle

    draft_result: ModelResult | None = None
    if drafter_member is not None and successes:
        draft_result = dispatch(
            drafter_member,
            _draft_prompt(
                prompt,
                judge,
                panel_results,
                include_panel=decision.resolved == "pro",
                best_result=best_result,
            ),
            "one-shot",
            config,
            apply_member_prompt=False,
        )
        bundle["draft"] = draft_result.answer
        bundle["draft_result"] = draft_result.to_dict()
        all_results.append(draft_result)
        if best_result is not None and draft_result.ok and not getattr(args, "no_draft_gate", False):
            gate = run_draft_gate(judge_members[0], prompt, best_result, draft_result, config)
            bundle["draft_original"] = draft_result.answer
            bundle["draft_gate"] = {key: value for key, value in gate.items() if key != "final_answer"}
            bundle["draft"] = str(gate["final_answer"])
            gate_result = gate.get("result")
            if isinstance(gate_result, dict):
                all_results.append(ModelResult(**gate_result))

    wall_latency_ms = round((time.perf_counter() - started) * 1000)
    bundle["metrics"] = aggregate_metrics(all_results, wall_latency_ms=wall_latency_ms)
    draft_ok = drafter_member is None or (draft_result is not None and draft_result.ok)
    exit_code = 0 if successes and judge.get("valid") and draft_ok else 1
    return bundle, exit_code
