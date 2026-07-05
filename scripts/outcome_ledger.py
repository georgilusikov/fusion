#!/usr/bin/env python3
"""Append Fusion outcome events to a JSONL ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_core.experiments import append_jsonl, outcome_event  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append a Fusion outcome event")
    parser.add_argument("--ledger", default=str(REPO_ROOT / "benchmarks" / "outcomes.jsonl"))
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--answer-id", required=True)
    parser.add_argument("--affected-action", action="store_true")
    parser.add_argument("--later-corrected", action="store_true")
    parser.add_argument("--notes", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    event = outcome_event(
        case_id=args.case_id,
        answer_id=args.answer_id,
        affected_action=args.affected_action,
        later_corrected=args.later_corrected,
        notes=args.notes,
    )
    if not args.dry_run:
        append_jsonl(Path(args.ledger), event)
    print(json.dumps(event, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
