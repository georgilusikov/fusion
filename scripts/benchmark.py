#!/usr/bin/env python3
"""Run the Fusion benchmark set and write aggregate panel metrics."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_core.benchmarking import *  # noqa: F401,F403,E402
from fusion_core.benchmark_runner import run_case  # noqa: E402
from fusion_core.routing import load_dotenv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Fusion panel presets")
    parser.add_argument("--cases", default=str(REPO_ROOT / "benchmarks" / "cases.jsonl"))
    parser.add_argument("--presets", nargs="+", default=["dq", "core"])
    parser.add_argument("--judge", default="claude")
    parser.add_argument("--drafter", default="claude")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--output", default=str(REPO_ROOT / "benchmarks" / "results.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    cases = load_cases(Path(args.cases))
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]
    if args.dry_run:
        print(json.dumps({"presets": args.presets, "cases": cases}, ensure_ascii=False, indent=2))
        return 0

    all_runs: dict[str, list[dict[str, Any]]] = {}
    for preset in args.presets:
        runs: list[dict[str, Any]] = []
        for case in cases:
            print(f"[benchmark] preset={preset} case={case['id']}", file=sys.stderr)
            runs.append(run_case(case, preset, args.judge, args.drafter, args.timeout, args.retries))
        all_runs[preset] = runs

    payload = {
        "version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "case_count": len(cases),
        "panels": [aggregate(preset, runs) for preset, runs in all_runs.items()],
        "details": all_runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
