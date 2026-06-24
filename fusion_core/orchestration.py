"""Compatibility exports for Fusion orchestration."""

from .pipeline import run_fusion
from .cli import build_parser, main

__all__ = ["run_fusion", "build_parser", "main"]
