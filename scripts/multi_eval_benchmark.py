#!/usr/bin/env python3
"""Run blind deliberation comparisons with an independent evaluator panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_core.deliberation_benchmark import (  # noqa: E402
    DEFAULT_VARIANTS,
    aggregate_deliberation_runs,
    load_deliberation_cases,
)
from fusion_core.multi_eval_benchmark import parse_evaluator_specs, run_multi_eval_case  # noqa: E402
from fusion_core.routing import load_dotenv  # noqa: E402

DEFAULT_EVALUATORS = (
    "or:openai/gpt-5.6-luna,"
    "or:google/gemini-3.1-pro-preview,"
    "or:deepseek/deepseek-v4-pro-0813"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blind benchmark: solo vs current Fusion vs deliberation, judged by multiple evaluators"
    )
    parser.add_argument(
        "--cases",
        default=str(REPO_ROOT / "benchmarks" / "deliberation_cases.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "benchmarks" / "deliberation_results.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preset", default="core")
    parser.add_argument("--baseline", default="claude:opus")
    parser.add_argument("--judge", default="claude")
    parser.add_argument("--drafter", default="claude:opus")
    parser.add_argument(
        "--evaluators",
        default=DEFAULT_EVALUATORS,
        help="comma-separated independent evaluator member specs",
    )
    parser.add_argument(
        "--min-valid-evaluators",
        type=int,
        default=None,
        help="minimum valid evaluator outputs per case; default is strict majority",
    )
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--scouts", type=int, default=4)
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--critics", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.timeout <= 0 or args.retries < 0:
        parser.error("timeout must be positive and retries non-negative")
    if not 0 <= args.scouts <= 8:
        parser.error("scouts must be from 0 to 8")
    if not 0 <= args.branches <= 4:
        parser.error("branches must be from 0 to 4")
    if not 0 <= args.critics <= 4:
        parser.error("critics must be from 0 to 4")

    evaluators = parse_evaluator_specs(args.evaluators)
    if not evaluators:
        parser.error("at least one evaluator is required")
    if args.min_valid_evaluators is not None and not 1 <= args.min_valid_evaluators <= len(evaluators):
        parser.error("min-valid-evaluators must be between 1 and evaluator count")

    cases = load_deliberation_cases(Path(args.cases))
    if args.limit is not None:
        cases = cases[: args.limit]

    config = {
        "cases": len(cases),
        "variants": list(DEFAULT_VARIANTS),
        "preset": args.preset,
        "baseline": args.baseline,
        "judge": args.judge,
        "drafter": args.drafter,
        "evaluators": evaluators,
        "min_valid_evaluators": args.min_valid_evaluators,
        "scouts": args.scouts,
        "branches": args.branches,
        "critics": args.critics,
    }
    if args.dry_run:
        print(json.dumps({"config": config, "case_ids": [case["id"] for case in cases]}, ensure_ascii=False, indent=2))
        return 0

    runs = []
    for index, case in enumerate(cases, 1):
        print(f"[multi-eval-bench] {index}/{len(cases)} {case['id']}", file=sys.stderr)
        run = run_multi_eval_case(
            case,
            preset=args.preset,
            baseline=args.baseline,
            judge=args.judge,
            drafter=args.drafter,
            evaluators=evaluators,
            timeout=args.timeout,
            retries=args.retries,
            scouts=args.scouts,
            branches=args.branches,
            critics=args.critics,
            min_valid_evaluators=args.min_valid_evaluators,
        )
        runs.append(run)

    payload = {
        "config": config,
        "aggregate": aggregate_deliberation_runs(runs),
        "runs": runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
