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
from .dispatch import dispatch
from .judge import run_judge
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


def _draft_prompt(
    user_prompt: str,
    judge: Mapping[str, Any],
    panel: Sequence[ModelResult],
    include_panel: bool,
) -> str:
    judge_text = json.dumps(judge.get("parsed"), ensure_ascii=False, indent=2) if judge.get("valid") else str(judge.get("raw", ""))
    panel_text = ""
    if include_panel:
        panel_text = "\n\n## Source answers\n" + "\n\n".join(
            f"### {item.label}\n{item.answer}" for item in successful_results(panel)
        )
    return (
        "You are the drafter. Answer the original prompt. Resolve contradictions explicitly, "
        "fill coverage gaps, and avoid shared blind spots. Do not mention this orchestration.\n\n"
        f"## Original prompt\n{user_prompt}\n\n"
        f"## Judge analysis\n{judge_text}{panel_text}\n\n"
        "Write the final answer now."
    )


def run_fusion(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.perf_counter()
    prompt = args.prompt or sys.stdin.read().strip()
    if not prompt:
        return {"error": "empty prompt"}, 2

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
            "drafter": dataclasses.asdict(drafter_member) if drafter_member else None,
        }, 0

    print(
        f"[fusion] strategy={decision.resolved} source={decision.source} "
        f"preset={decision.preset or 'custom'} members={[member.label for member in members]} "
        f"judge={judge_member.label}",
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
    judge = run_judge(
        judge_member,
        prompt,
        successful_results(panel_results),
        config,
        repair_attempts=effective_repairs,
    )
    rounds[0]["judge_valid"] = bool(judge.get("valid"))

    reviewer_count = (
        args.reviewers
        if args.reviewers is not None
        else (2 if decision.resolved == "pro" else 0)
    )
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
        )
        panel_results.extend(reviews)
        if successful_results(reviews):
            judge = run_judge(
                judge_member,
                prompt,
                successful_results(panel_results),
                config,
                repair_attempts=effective_repairs,
            )
        rounds.append(
            {
                "name": "review",
                "members": [item.label for item in reviews],
                "judge_valid": bool(judge.get("valid")),
            }
        )

    explicit_panel = args.panel is not None or args.preset is not None
    reasons = escalation_reasons(panel_results, judge)
    if args.strategy == "adaptive" and reasons and not args.no_escalate and not explicit_panel:
        print(f"[fusion] adaptive escalation: {', '.join(reasons)}", file=sys.stderr)
        power_members = parse_panel(PRESETS["power"], same_mode=args.mode == "same")
        escalated = _run_panel(power_members, prompt, args.depth, config)
        panel_results = combine_unique_results(panel_results, escalated)
        judge = run_judge(
            judge_member,
            prompt,
            successful_results(panel_results),
            config,
            repair_attempts=effective_repairs,
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

    all_results = list(panel_results)
    judge_result = judge.get("result")
    if isinstance(judge_result, dict):
        all_results.append(ModelResult(**judge_result))
    for repair in judge.get("repair_results", []):
        if isinstance(repair, dict):
            all_results.append(ModelResult(**repair))

    bundle: dict[str, Any] = {
        "prompt": prompt,
        "strategy": decision.resolved,
        "selection": decision.to_dict(),
        "depth": args.depth,
        "reasoning": args.reasoning,
        "panel": [item.to_dict() for item in panel_results],
        "successful_panel": [item.label for item in successes],
        "failed_panel": [item.label for item in failures],
        "rounds": rounds,
        "judge": judge,
    }

    draft_result: ModelResult | None = None
    if drafter_member is not None and successes:
        draft_result = dispatch(
            drafter_member,
            _draft_prompt(prompt, judge, panel_results, include_panel=decision.resolved == "pro"),
            "one-shot",
            config,
            apply_member_prompt=False,
        )
        bundle["draft"] = draft_result.answer
        bundle["draft_result"] = draft_result.to_dict()
        all_results.append(draft_result)

    wall_latency_ms = round((time.perf_counter() - started) * 1000)
    bundle["metrics"] = aggregate_metrics(all_results, wall_latency_ms=wall_latency_ms)
    draft_ok = drafter_member is None or (draft_result is not None and draft_result.ok)
    exit_code = 0 if successes and judge.get("valid") and draft_ok else 1
    return bundle, exit_code
