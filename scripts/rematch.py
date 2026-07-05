#!/usr/bin/env python3
"""Prepare blind Fusion-vs-baseline rematches on a golden task set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_core.evaluation import (  # noqa: E402
    REMATCH_VARIANTS, anonymize_candidates, blind_judge_prompt, load_jsonl,
    median_axis_scores, rematch_verdict, winner_from_scores,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fusion golden-set blind rematch helper")
    parser.add_argument("--cases", default=str(REPO_ROOT / "benchmarks" / "golden" / "fusion_golden_set.jsonl"))
    parser.add_argument(
        "--answers",
        default=None,
        help="optional JSONL rows: case_id, variant, answer. Without this, prints the run matrix.",
    )
    parser.add_argument(
        "--judge-payloads",
        default=None,
        help="optional JSONL rows: case_id, judge, payload. Computes median scores and kill-rule verdicts.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run_matrix(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "variants": list(REMATCH_VARIANTS),
        "case_count": len(cases),
        "runs": [
            {"case_id": case["id"], "variant": variant, "prompt": case["prompt"]}
            for case in cases
            for variant in REMATCH_VARIANTS
        ],
    }


def _judge_packets(cases: list[dict[str, Any]], answers_path: Path) -> dict[str, Any]:
    case_by_id = {str(case["id"]): case for case in cases}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in load_jsonl(answers_path):
        case_id = str(row.get("case_id") or "")
        if case_id in case_by_id:
            grouped[case_id].append({"label": str(row.get("variant") or row.get("label") or "candidate"), "answer": str(row.get("answer") or "")})
    packets = []
    for case_id, candidates in sorted(grouped.items()):
        case = case_by_id[case_id]
        anonymous = anonymize_candidates(candidates, seed=case_id)
        packets.append({
            "case_id": case_id,
            "anonymous_candidates": anonymous,
            "judge_prompt": blind_judge_prompt(case["prompt"], anonymous),
        })
    return {"packet_count": len(packets), "packets": packets}


def _verdicts(judge_payloads_path: Path) -> dict[str, Any]:
    grouped_payloads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(judge_payloads_path):
        case_id = str(row.get("case_id") or "")
        payload = row.get("payload")
        if case_id and isinstance(payload, dict):
            grouped_payloads[case_id].append(payload)
    cases = []
    variant_totals: dict[str, list[float]] = defaultdict(list)
    for case_id, payloads in sorted(grouped_payloads.items()):
        medians = median_axis_scores(payloads)
        winner = winner_from_scores(medians)
        cases.append({"case_id": case_id, "median_scores": medians, "winner": winner})
        for label, axes in medians.items():
            if label in REMATCH_VARIANTS:
                values = list(axes.values())
                if values:
                    variant_totals[label].append(sum(values) / len(values))
    variant_scores = {
        label: sum(scores) / len(scores)
        for label, scores in sorted(variant_totals.items())
        if scores
    }
    return {"cases": cases, "variant_scores": variant_scores, "kill_rule": rematch_verdict(variant_scores)}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = load_jsonl(Path(args.cases))
    if args.judge_payloads:
        payload = _verdicts(Path(args.judge_payloads))
    elif args.answers:
        payload = _judge_packets(cases, Path(args.answers))
    else:
        payload = _run_matrix(cases)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output and not args.dry_run:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
