"""Self-consistency panel expansion helpers."""

from __future__ import annotations


def _split_sample_suffix(token: str) -> tuple[str, int]:
    head = token.strip()
    if "*" not in head:
        return head, 1
    prefix, suffix = head.rsplit("*", 1)
    try:
        count = int(suffix)
    except ValueError as exc:
        raise ValueError(f"bad sample count in panel member {token!r}") from exc
    if count < 1 or count > 9:
        raise ValueError("sample count must be from 1 to 9")
    return prefix.strip(), count


def expand_panel_spec(spec: str) -> str:
    """Expand BACKEND[:MODEL][!DEPTH][@ROLE]*N into repeated members."""
    expanded: list[str] = []
    for token in spec.split(","):
        if not token.strip():
            continue
        member, count = _split_sample_suffix(token)
        expanded.extend([member] * count)
    return ",".join(expanded)
