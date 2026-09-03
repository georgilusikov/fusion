#!/usr/bin/env python3
"""Run blind solo-vs-Fusion deliberation quality comparisons."""

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
    run_deliberation_case,
)
from fusion_core.routing import load_dotenv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blind benchmark: solo vs current Fusion vs deliberation Fusion")
    parser.add_argument(
        "--cases",
        default=str(REPO_ROOT / "benchmarks" / "deliberation_cases.jsonl"),
        help="deliberation JSONL fixture",
    )
    parser.add_argument("--output", default=str(REPO_ROOT / "benchmarks" / "deliberation_results.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preset", default="core")
    parser.add_argument("--baseline", default="claude:opus")
    parser.add_argument("--judge", default="claude")
    parser.add_argument("--drafter", default="claude:opus")
    parser.add_argument("--evaluator", default="claude:opus")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--scouts", type=int, default=4)
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--critics", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="validate fixture and print planned configuration only")
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
        "evaluator": args.evaluator,
        "scouts": args.scouts,
        "branches": args.branches,
        "critics": args.critics,
    }
    if args.dry_run:
        print(json.dumps({"config": config, "case_ids": [case["id"] for case in cases]}, ensure_ascii=False, indent=2))
        return 0

    runs = []
    for index, case in enumerate(cases, 1):
        print(f"[delib-bench] {index}/{len(cases)} {case['id']}", file=sys.stderr)
        run = run_deliberation_case(
            case,
            preset=args.preset,
            baseline=args.baseline,
            judge=args.judge,
            drafter=args.drafter,
            evaluator=args.evaluator,
            timeout=args.timeout,
            retries=args.retries,
            scouts=args.scouts,
            branches=args.branches,
            critics=args.critics,
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
