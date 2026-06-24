"""Fusion command-line parser and entry point."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .config import (
    DEFAULT_AGENT_WORKSPACE, DEFAULT_BACKOFF, DEFAULT_JUDGE, DEFAULT_REPAIR_ATTEMPTS,
    DEFAULT_RETRIES, DEFAULT_STRATEGY, DEFAULT_TIMEOUT, PRESETS, REPO_ROOT,
)
from .pipeline import run_fusion
from .routing import load_dotenv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fusion: adaptive panel -> validated judge -> optional draft")
    parser.add_argument("prompt", nargs="?", help="prompt; if omitted, read stdin")
    parser.add_argument(
        "--strategy", choices=["lite", "pro", "adaptive"], default=DEFAULT_STRATEGY,
        help="pipeline strategy (default: adaptive)",
    )
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), default=None,
        help="explicit named panel; otherwise strategy and benchmark results choose it",
    )
    parser.add_argument("--panel", default=None, help="comma list BACKEND[:MODEL][!DEPTH][@ROLE]")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help=f"judge member (default: {DEFAULT_JUDGE})")
    parser.add_argument("--reasoning", choices=["on", "off"], default="on")
    parser.add_argument("--mode", choices=["role", "same"], default="role")
    parser.add_argument("--depth", choices=["one-shot", "agent"], default="one-shot")
    parser.add_argument(
        "--agent-workspace", choices=["temp", "snapshot", "worktree"],
        default=DEFAULT_AGENT_WORKSPACE,
        help="isolated agent workspace; direct current-directory execution is not supported",
    )
    parser.add_argument(
        "--workspace-source", default=".",
        help="source directory copied into snapshot or used to locate a git worktree",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF)
    parser.add_argument("--repair-attempts", type=int, default=DEFAULT_REPAIR_ATTEMPTS)
    parser.add_argument("--auto-draft", metavar="MEMBER", help="run a final drafter member")
    parser.add_argument(
        "--benchmark-results", default=str(REPO_ROOT / "benchmarks" / "results.json"),
        help="benchmark aggregate used for automatic panel selection",
    )
    parser.add_argument("--budget-usd", type=float, default=None, help="maximum benchmark average cost per run")
    parser.add_argument("--max-latency-ms", type=int, default=None, help="maximum benchmark average latency")
    parser.add_argument("--complexity-threshold", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="show routing and member configuration only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.retries < 0 or args.repair_attempts < 0 or args.timeout <= 0 or args.backoff < 0:
        parser.error("timeout must be positive; retries, repair-attempts, and backoff must be non-negative")
    bundle, exit_code = run_fusion(args)
    print(json.dumps(bundle, ensure_ascii=False, indent=2))
    return exit_code
