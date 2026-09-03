"""Helpers for provider-blind judge evaluation."""

from __future__ import annotations

import dataclasses
import random
from typing import Any, Mapping, Sequence

from .config import ModelResult


def candidate_label(index: int) -> str:
    """Return stable labels Candidate A .. Candidate Z, Candidate AA, ..."""
    value = index + 1
    chars: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "Candidate " + "".join(reversed(chars))


def anonymize_panel(
    panel: Sequence[ModelResult],
    *,
    order_seed: int,
) -> tuple[list[ModelResult], dict[str, str]]:
    """Hide source identity and shuffle presentation order while keeping a reversible map."""
    alias_to_original: dict[str, str] = {}
    anonymized: list[ModelResult] = []
    for index, item in enumerate(panel):
        alias = candidate_label(index)
        alias_to_original[alias] = item.label
        anonymized.append(
            dataclasses.replace(
                item,
                label=alias,
                backend="hidden",
                model=None,
            )
        )
    random.Random(order_seed).shuffle(anonymized)
    return anonymized, alias_to_original


def _replace_aliases(text: str, alias_to_original: Mapping[str, str]) -> str:
    result = text
    for alias in sorted(alias_to_original, key=len, reverse=True):
        result = result.replace(alias, alias_to_original[alias])
    return result


def restore_labels(value: Any, alias_to_original: Mapping[str, str]) -> Any:
    """Recursively restore original panel labels in a parsed judge payload."""
    if isinstance(value, str):
        return _replace_aliases(value, alias_to_original)
    if isinstance(value, list):
        return [restore_labels(item, alias_to_original) for item in value]
    if isinstance(value, Mapping):
        return {key: restore_labels(item, alias_to_original) for key, item in value.items()}
    return value
