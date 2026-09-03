#!/usr/bin/env python3
"""Evaluate whether a deliberation benchmark is strong enough for promotion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_core.promotion import promotion_verdict  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Return PROMOTE/HOLD for a deliberation benchmark result")
    parser.add_argument("result", help="path to deliberation_results.json")
    parser.add_argument("--min-valid", type=int, default=10)
    parser.add_argument("--min-valid-fraction", type=float, default=0.80)
    parser.add_argument("--min-quality-delta", type=float, default=0.10)
    parser.add_argument("--min-win-rate-delta", type=float, default=0.05)
    parser.add_argument("--max-success-regression", type=float, default=0.02)
    parser.add_argument("--max-latency-ratio", type=float, default=None)
    parser.add_argument("--max-cost-ratio", type=float, default=None)
    parser.add_argument(
        "--fail-on-hold",
        action="store_true",
        help="exit 1 when the verdict is HOLD; default is reporting-only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_valid < 0:
        parser.error("min-valid must be non-negative")
    if not 0 <= args.min_valid_fraction <= 1:
        parser.error("min-valid-fraction must be from 0 to 1")
    if args.max_success_regression < 0:
        parser.error("max-success-regression must be non-negative")
    if args.max_latency_ratio is not None and args.max_latency_ratio <= 0:
        parser.error("max-latency-ratio must be positive")
    if args.max_cost_ratio is not None and args.max_cost_ratio <= 0:
        parser.error("max-cost-ratio must be positive")

    path = Path(args.result)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "HOLD", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    aggregate = payload.get("aggregate") if isinstance(payload, dict) else None
    if not isinstance(aggregate, dict):
        print(json.dumps({"status": "HOLD", "error": "missing aggregate object"}, ensure_ascii=False, indent=2))
        return 2

    verdict = promotion_verdict(
        aggregate,
        min_valid_evaluations=args.min_valid,
        min_valid_fraction=args.min_valid_fraction,
        min_quality_delta=args.min_quality_delta,
        min_win_rate_delta=args.min_win_rate_delta,
        max_success_rate_regression=args.max_success_regression,
        max_latency_ratio=args.max_latency_ratio,
        max_cost_ratio=args.max_cost_ratio,
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    if args.fail_on_hold and verdict.get("status") != "PROMOTE":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
