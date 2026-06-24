"""Configuration, data contracts, and shared constants for Fusion."""

from __future__ import annotations

import dataclasses
import re
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STRATEGY = "adaptive"
DEFAULT_JUDGE = "claude"
DEFAULT_TIMEOUT = 240
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF = 0.5
DEFAULT_REPAIR_ATTEMPTS = 1
DEFAULT_AGENT_WORKSPACE = "snapshot"
WORKTREE_LOCK = threading.Lock()

KNOWN_CLIS = {"agy", "codex", "gemini", "cursor", "claude", "opencode"}
API_BACKENDS = {"or", "openrouter", "anthropic", "openai", "google"}

API_DEFAULT_MODEL = {
    "or": "deepseek/deepseek-v4-pro",
    "openrouter": "deepseek/deepseek-v4-pro",
    "anthropic": "claude-3-5-sonnet-latest",
    "openai": "gpt-4o",
    "google": "gemini-2.0-flash",
}

ROLES: dict[str, str] = {
    "builder": "Answer concretely and actionably. Give the best direct solution with steps.",
    "skeptic": "Answer, then stress-test it. Flag where it could be wrong or fail.",
    "expert": "Answer with domain depth, edge cases, trade-offs, and caveats.",
    "contrarian": "Challenge the obvious approach and give the strongest viable alternative.",
    "generalist": "Give a balanced, comprehensive answer covering the main angles.",
    "falsifier": (
        "Try to falsify the obvious answer: identify failure modes and hidden assumptions, "
        "then give the answer that survives the attack."
    ),
    "neutral": "Answer the question as well as you can.",
}

PRESETS: dict[str, str] = {
    "core": (
        "claude@generalist,gemini@expert,cursor@builder,agy@contrarian,"
        "or:deepseek/deepseek-v4-pro@falsifier"
    ),
    "power": (
        "codex:gpt-5.5@builder,cursor:composer-2.5!agent@skeptic,"
        "claude:opus@generalist,agy:Gemini 3.5 Flash (High)!agent@expert,"
        "agy:Claude Opus 4.6 (Thinking)@contrarian,"
        "or:deepseek/deepseek-v4-pro@falsifier,or:qwen/qwen3.7-max@skeptic"
    ),
    "dq": (
        "or:deepseek/deepseek-v4-pro@builder,"
        "or:qwen/qwen3.7-max@falsifier"
    ),
    "all": (
        "agy!agent@builder,cursor!agent@falsifier,codex@skeptic,gemini@expert,"
        "claude@generalist,or:deepseek/deepseek-v4-pro@builder,"
        "or:moonshotai/kimi-k2-thinking@falsifier,or:qwen/qwen3.7-max@expert,"
        "or:z-ai/glm-5.1@contrarian,or:minimax/minimax-m3@skeptic"
    ),
    "chinese": (
        "or:deepseek/deepseek-v4-pro@builder,"
        "or:moonshotai/kimi-k2-thinking@falsifier,"
        "or:qwen/qwen3.7-max@expert,or:z-ai/glm-5.1@contrarian,"
        "or:minimax/minimax-m3@skeptic"
    ),
    "deepseek": (
        "or:deepseek/deepseek-v4-pro@builder,"
        "or:deepseek/deepseek-v4-flash@falsifier"
    ),
    "free": (
        "or:meta-llama/llama-3.3-70b-instruct:free@builder,"
        "or:qwen/qwen3-next-80b-a3b-instruct:free@expert,"
        "or:openai/gpt-oss-120b:free@falsifier,"
        "or:google/gemma-4-31b-it:free@contrarian"
    ),
    "cli": "agy@builder,codex@skeptic,gemini@expert,cursor@contrarian,claude@generalist",
    "mixed": (
        "or:openai/gpt-4o@builder,or:anthropic/claude-sonnet-4.6@falsifier,"
        "or:google/gemini-2.5-flash@expert,"
        "or:deepseek/deepseek-v4-pro@contrarian"
    ),
}

STRATEGY_CANDIDATES: dict[str, list[str]] = {
    "lite": ["dq", "deepseek", "free"],
    "pro": ["core", "mixed", "power", "all"],
}
STRATEGY_FALLBACK = {"lite": "dq", "pro": "core"}

REASONING_PREAMBLE = """Before answering:
- Separate observed facts, inferences, source claims, and assumptions.
- State a calibrated confidence for the main conclusion.
- Challenge flawed premises instead of agreeing automatically.
- Check what you may have missed and repair the answer before finalizing.
"""

STRATEGY_INSTRUCTIONS = {
    "lite": (
        "Use a compact decomposition. State the direct answer, assumptions, and confidence. "
        "Prefer verifiable claims over breadth."
    ),
    "pro": (
        "Solve independently before comparing alternatives. Identify hidden assumptions, "
        "failure modes, evidence needed, and the strongest counterargument."
    ),
}

JUDGE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://example.invalid/fusion/judge.schema.json",
    "title": "FusionJudgeResult",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "consensus",
        "contradictions",
        "coverage_gaps",
        "unique_insights",
        "blind_spots",
        "recommendation",
    ],
    "properties": {
        "consensus": {"type": "array", "items": {"type": "string"}},
        "contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["point", "sides"],
                "properties": {
                    "point": {"type": "string"},
                    "sides": {"type": "string"},
                },
            },
        },
        "coverage_gaps": {"type": "array", "items": {"type": "string"}},
        "unique_insights": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["model", "insight"],
                "properties": {
                    "model": {"type": "string"},
                    "insight": {"type": "string"},
                },
            },
        },
        "blind_spots": {"type": "array", "items": {"type": "string"}},
        "recommendation": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

JUDGE_INSTRUCTION = """You are the judge in a multi-model deliberation.
Analyze only the successful answers below. Do not merge them into a final answer.
Return one JSON object matching the supplied JSON Schema. Return JSON only.
"""

CONFIDENCE_RE = re.compile(
    r"(?i)\b(?:confidence|уверенность)\s*(?:score\s*)?[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*(%)?"
)
COMPLEX_TERMS = {
    "architecture",
    "benchmark",
    "migration",
    "security",
    "threat model",
    "distributed",
    "concurrency",
    "debug",
    "research",
    "evidence",
    "trade-off",
    "tradeoff",
    "legal",
    "medical",
    "financial",
    "production",
    "multi-step",
    "optimize",
    "prove",
    "design",
    "сравни",
    "архитектур",
    "безопасн",
    "исследован",
    "доказ",
    "оптимиз",
    "продакш",
}
CATEGORY_TERMS: dict[str, set[str]] = {
    "technical": {"http", "protocol", "status code", "network", "протокол", "статус"},
    "code": {"python", "javascript", "typescript", "code", "bug", "function", "api", "sql", "код"},
    "research": {"research", "sources", "citation", "evidence", "paper", "исследован", "источник"},
    "reliability": {"retry", "idempotency", "distributed", "queue", "payment", "reliability", "надёжн"},
    "reasoning": {"prove", "derive", "logic", "calculate", "why", "доказ", "почему", "рассчитай"},
}


@dataclasses.dataclass(slots=True)
class Member:
    label: str
    kind: str
    backend: str
    model: str | None
    role_key: str
    role_text: str
    depth: str | None = None


@dataclasses.dataclass(slots=True)
class ModelResult:
    label: str
    backend: str
    kind: str
    model: str | None = None
    ok: bool = False
    answer: str = ""
    latency_ms: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    errors: list[str] = dataclasses.field(default_factory=list)
    confidence: float | None = None
    attempts: int = 1
    returncode: int | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(slots=True)
class DispatchConfig:
    timeout: int = DEFAULT_TIMEOUT
    retries: int = DEFAULT_RETRIES
    backoff: float = DEFAULT_BACKOFF
    reasoning: bool = True
    strategy: str = "lite"
    agent_workspace: str = DEFAULT_AGENT_WORKSPACE
    workspace_source: Path = dataclasses.field(default_factory=Path.cwd)
    pricing: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True)
class StrategyDecision:
    requested: str
    resolved: str
    preset: str | None
    category: str
    complexity_score: int
    source: str
    reason: str
    benchmark_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

