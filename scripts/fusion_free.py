#!/usr/bin/env python3
"""Run Fusion with a dynamically discovered OpenRouter free-model panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_core.cli import main as fusion_main  # noqa: E402
from fusion_core.free_pool import (  # noqa: E402
    DEFAULT_CACHE_TTL,
    DEFAULT_MIN_CONTEXT,
    DEFAULT_POOL_SIZE,
    resolve_free_panel,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve a current OpenRouter free-model panel, then delegate to the normal Fusion CLI.",
        add_help=False,
    )
    parser.add_argument("--free-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--free-min-context", type=int, default=DEFAULT_MIN_CONTEXT)
    parser.add_argument("--free-cache-ttl", type=int, default=DEFAULT_CACHE_TTL)
    parser.add_argument("--refresh-free-pool", action="store_true")
    parser.add_argument("--print-free-panel", action="store_true")
    parser.add_argument("--free-help", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, remaining = parser.parse_known_args(raw)
    if args.free_help:
        parser.print_help()
        print("\nAll unrecognized arguments are forwarded to scripts/fusion.py.")
        return 0
    if args.free_size < 1 or args.free_size > 16:
        parser.error("free-size must be from 1 to 16")
    if args.free_min_context < 0 or args.free_cache_ttl < 0:
        parser.error("free-min-context and free-cache-ttl must be non-negative")
    if "--panel" in remaining or "--preset" in remaining:
        parser.error("do not pass --panel/--preset to fusion_free; the dynamic panel is supplied automatically")

    panel, metadata = resolve_free_panel(
        size=args.free_size,
        min_context=args.free_min_context,
        cache_ttl=args.free_cache_ttl,
        refresh=args.refresh_free_pool,
    )
    if not panel:
        print("[fusion-free] no usable free text models were discovered", file=sys.stderr)
        return 2

    print(
        f"[fusion-free] source={metadata['source']} selected={len(metadata['selected'])} "
        f"models={[item['id'] for item in metadata['selected']]}",
        file=sys.stderr,
    )
    if args.print_free_panel:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0

    return fusion_main([*remaining, "--panel", panel])


if __name__ == "__main__":
    raise SystemExit(main())
