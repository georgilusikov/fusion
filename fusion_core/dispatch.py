"""CLI and HTTP model dispatch with metrics, retries, and backoff."""

from __future__ import annotations

import contextlib
import json
import random
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import (
    REASONING_PREAMBLE, STRATEGY_INSTRUCTIONS, DispatchConfig, Member, ModelResult,
)
from .routing import api_key_for, estimate_cost, extract_confidence, extract_usage
from .workspace import prepared_workspace

# --- dispatch --------------------------------------------------------------

def model_args(backend: str, model: str | None) -> list[str]:
    if not model:
        return []
    if backend == "codex":
        return ["-c", f"model={model}"]
    if backend in {"gemini", "opencode"}:
        return ["-m", model]
    return ["--model", model]


def cli_argv(backend: str, model: str | None, prompt: str, depth: str) -> tuple[list[str], str | None]:
    model_flags = model_args(backend, model)
    if depth == "agent":
        commands = {
            "codex": (["codex", "exec", "--sandbox", "workspace-write", *model_flags, "-"], prompt),
            "claude": (["claude", "-p", prompt, "--dangerously-skip-permissions", *model_flags], None),
            "gemini": (["gemini", "-p", prompt, "--yolo", *model_flags], None),
            "cursor": (["cursor-agent", "-p", prompt, "--force", "--trust", *model_flags], None),
            "agy": (["agy", "-p", prompt, "--dangerously-skip-permissions", *model_flags], None),
            "opencode": (["opencode", "run", prompt, *model_flags], None),
        }
    else:
        commands = {
            "codex": (["codex", "exec", "--sandbox", "read-only", *model_flags, "-"], prompt),
            "claude": (["claude", "-p", prompt, *model_flags], None),
            "gemini": (["gemini", "-p", prompt, "--approval-mode", "plan", *model_flags], None),
            "cursor": (["cursor-agent", "-p", prompt, "--mode", "ask", "--trust", *model_flags], None),
            "agy": (["agy", "-p", prompt, "--sandbox", *model_flags], None),
            "opencode": (["opencode", "run", prompt, "--agent", "plan", *model_flags], None),
        }
    try:
        return commands[backend]
    except KeyError as exc:
        raise ValueError(f"unknown CLI backend: {backend!r}") from exc


def _sleep_delay(backoff: float, failed_attempt_index: int) -> float:
    base = max(0.0, backoff) * (2**failed_attempt_index)
    return base + random.uniform(0.0, base * 0.1 if base else 0.0)


def dispatch_cli(
    member: Member,
    prompt: str,
    depth: str,
    config: DispatchConfig,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> ModelResult:
    started = time.perf_counter()
    errors: list[str] = []
    attempts = 0
    returncode: int | None = None
    answer = ""
    final_ok = False
    effective_depth = member.depth or depth

    workspace_context: contextlib.AbstractContextManager[Path | None]
    if effective_depth == "agent":
        workspace_context = prepared_workspace(config.agent_workspace, config.workspace_source)
    else:
        workspace_context = contextlib.nullcontext(None)

    with workspace_context as workspace:
        argv, stdin_payload = cli_argv(member.backend, member.model, prompt, effective_depth)
        for attempt_index in range(config.retries + 1):
            attempts += 1
            try:
                proc = subprocess.run(
                    argv,
                    input=stdin_payload,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout,
                    cwd=str(workspace) if workspace else None,
                )
                returncode = proc.returncode
                answer = (proc.stdout or "").strip()
                stderr = (proc.stderr or "").strip()
                if proc.returncode == 0 and answer:
                    final_ok = True
                    break
                detail = stderr[:500] or "empty stdout"
                errors.append(f"attempt {attempts}: exit {proc.returncode}: {detail}")
            except subprocess.TimeoutExpired:
                errors.append(f"attempt {attempts}: timeout after {config.timeout}s")
                returncode = -1
            except FileNotFoundError:
                errors.append(f"CLI not found: {argv[0]}")
                returncode = -2
                break
            except OSError as exc:
                errors.append(f"attempt {attempts}: {type(exc).__name__}: {exc}")

            if attempt_index < config.retries:
                sleep_fn(_sleep_delay(config.backoff, attempt_index))

    latency_ms = round((time.perf_counter() - started) * 1000)
    return ModelResult(
        label=member.label,
        backend=member.backend,
        kind="cli",
        model=member.model,
        ok=final_ok,
        answer=answer,
        latency_ms=latency_ms,
        errors=errors,
        confidence=extract_confidence(answer),
        attempts=attempts,
        returncode=returncode,
        metadata={"depth": effective_depth, "agent_workspace": config.agent_workspace if effective_depth == "agent" else None},
    )


def _http_json(url: str, headers: Mapping[str, str], body: Mapping[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _provider_request(member: Member, prompt: str, timeout: int) -> tuple[str, dict[str, Any]]:
    key = api_key_for(member.backend)
    if not key:
        raise PermissionError(f"missing API key for {member.backend}")

    if member.backend in {"or", "openrouter"}:
        payload = _http_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "X-Title": "fusion",
            },
            {"model": member.model, "messages": [{"role": "user", "content": prompt}]},
            timeout,
        )
        text = payload["choices"][0]["message"]["content"]
    elif member.backend == "openai":
        payload = _http_json(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            {"model": member.model, "messages": [{"role": "user", "content": prompt}]},
            timeout,
        )
        text = payload["choices"][0]["message"]["content"]
    elif member.backend == "anthropic":
        payload = _http_json(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            {
                "model": member.model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout,
        )
        text = payload["content"][0]["text"]
    elif member.backend == "google":
        payload = _http_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{member.model}:generateContent?key={key}",
            {"Content-Type": "application/json"},
            {"contents": [{"parts": [{"text": prompt}]}]},
            timeout,
        )
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise ValueError(f"unknown API backend: {member.backend}")
    return str(text or "").strip(), payload


def _http_error_text(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read().decode("utf-8", "replace")[:400]
    except OSError:
        body = ""
    return f"HTTP {error.code}: {body}".rstrip()


def _retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError))


def dispatch_api(
    member: Member,
    prompt: str,
    config: DispatchConfig,
    request_fn: Callable[[Member, str, int], tuple[str, dict[str, Any]]] = _provider_request,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> ModelResult:
    started = time.perf_counter()
    errors: list[str] = []
    attempts = 0
    answer = ""
    payload: dict[str, Any] = {}
    ok = False

    for attempt_index in range(config.retries + 1):
        attempts += 1
        try:
            answer, payload = request_fn(member, prompt, config.timeout)
            if not answer:
                raise ValueError("provider returned an empty answer")
            ok = True
            break
        except PermissionError as exc:
            errors.append(str(exc))
            break
        except urllib.error.HTTPError as exc:
            errors.append(f"attempt {attempts}: {_http_error_text(exc)}")
            retryable = _retryable_exception(exc)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            errors.append(f"attempt {attempts}: {type(exc).__name__}: {exc}")
            retryable = True
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"attempt {attempts}: invalid provider response: {type(exc).__name__}: {exc}")
            retryable = False
        except OSError as exc:
            errors.append(f"attempt {attempts}: {type(exc).__name__}: {exc}")
            retryable = True

        if not retryable or attempt_index >= config.retries:
            break
        sleep_fn(_sleep_delay(config.backoff, attempt_index))

    input_tokens, output_tokens, total_tokens, cost = extract_usage(member.backend, payload)
    if cost is None:
        cost = estimate_cost(member.backend, member.model, input_tokens, output_tokens, config.pricing)
    latency_ms = round((time.perf_counter() - started) * 1000)
    return ModelResult(
        label=member.label,
        backend=member.backend,
        kind="api",
        model=member.model,
        ok=ok,
        answer=answer,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost,
        errors=errors,
        confidence=extract_confidence(answer),
        attempts=attempts,
    )


def build_member_prompt(member: Member, prompt: str, config: DispatchConfig) -> str:
    sections: list[str] = []
    if config.reasoning:
        sections.append(REASONING_PREAMBLE.strip())
    sections.append(STRATEGY_INSTRUCTIONS[config.strategy])
    sections.append(member.role_text)
    sections.append("---")
    sections.append(prompt)
    return "\n\n".join(sections)


def dispatch(
    member: Member,
    prompt: str,
    depth: str,
    config: DispatchConfig,
    apply_member_prompt: bool = True,
) -> ModelResult:
    full_prompt = build_member_prompt(member, prompt, config) if apply_member_prompt else prompt
    if member.kind == "cli":
        return dispatch_cli(member, full_prompt, depth, config)
    return dispatch_api(member, full_prompt, config)


