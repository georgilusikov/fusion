#!/usr/bin/env python3
"""Fusion command-line entry point and compatibility exports."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_core import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    raise SystemExit(main())
