#!/usr/bin/env python3
"""Fusion — local analog of OpenRouter's Fusion plugin.

Pipeline: PANEL (N members answer the same prompt in parallel) -> JUDGE (one
model emits structured analysis: consensus / contradictions / coverage_gaps /
unique_insights / blind_spots) -> DRAFTER (the conversing model, or an
--auto-draft member, writes the final answer).

Two backend families, mixable in one panel:
  - CLI vendors:  agy / codex / gemini / cursor / claude  (local CLIs)
  - API models:   or (OpenRouter) / anthropic / openai / google  (HTTP, key)

Member grammar:  BACKEND[:MODEL][@ROLE]
  codex@skeptic                          CLI, default model, role=skeptic
  claude@builder                         CLI
  or:anthropic/claude-3.5-sonnet@expert  OpenRouter API, explicit model
  openai:gpt-4o@builder                  direct OpenAI API
  anthropic:claude-3-5-sonnet-latest     direct Anthropic API, neutral role

Depth (CLI members only): --depth one-shot (read-only answer) | agent
(full tools/subagents, run in an isolated temp cwd so the repo is never
mutated). API members are always one-shot.

Keys: read from environment, or from a gitignored .env in the repo root
(KEY=VALUE lines). Never printed. Required per backend:
  or/openrouter -> OPENROUTER_API_KEY
  anthropic     -> ANTHROPIC_API_KEY
  openai        -> OPENAI_API_KEY
  google        -> GEMINI_API_KEY or GOOGLE_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
    "skeptic": "Answer, then stress-test your own answer — flag where it could be wrong or fail.",
    "expert": "Answer with domain depth: edge cases, trade-offs, and important caveats.",
    "contrarian": "Give the strongest answer that challenges the obvious/default approach.",
    "generalist": "Give a balanced, comprehensive answer covering the main angles.",
    "falsifier": "Actively try to FALSIFY the obvious answer: name the failure modes, "
                 "the hidden assumptions, and the cases where it breaks. Then give the answer "
                 "that survives that attack.",
    "neutral": "Answer the question as well as you can.",
}

# Named panel presets. Default is `dq` (deepseek + qwen). `--panel` overrides.
# Slugs are the latest available on OpenRouter as of 2026-06-15.
PRESETS: dict[str, str] = {
    # default: 5 diverse advisors — 4 local CLIs (free via subscriptions) + deepseek.
    "core": ("claude@generalist,gemini@expert,cursor@builder,agy@contrarian,"
             "or:deepseek/deepseek-v4-pro@falsifier"),
    # power: 7 advisors — subscription CLIs (cursor composer + agy gemini as
    # agents/subagents) across distinct base models + 2 cheap API families.
    "power": ("codex:gpt-5.5@builder,"
              "cursor:composer-2.5!agent@skeptic,"
              "claude:opus@generalist,"
              "agy:Gemini 3.5 Flash (High)!agent@expert,"
              "agy:Claude Opus 4.6 (Thinking)@contrarian,"
              "or:deepseek/deepseek-v4-pro@falsifier,"
              "or:qwen/qwen3.7-max@skeptic"),
    # deepseek + qwen — cheapest, strong, fast.
    "dq": ("or:deepseek/deepseek-v4-pro@builder,"
           "or:qwen/qwen3.7-max@falsifier"),
    # every available advisor: local CLIs (agy & cursor as full agents = subagents)
    # + the latest Chinese OpenRouter models. ~10 advisors. Heavy/slow/costly.
    "all": ("agy!agent@builder,cursor!agent@falsifier,codex@skeptic,"
            "gemini@expert,claude@generalist,"
            "or:deepseek/deepseek-v4-pro@builder,"
            "or:moonshotai/kimi-k2-thinking@falsifier,"
            "or:qwen/qwen3.7-max@expert,"
            "or:z-ai/glm-5.1@contrarian,"
            "or:minimax/minimax-m3@skeptic"),
    "chinese": ("or:deepseek/deepseek-v4-pro@builder,"
                "or:moonshotai/kimi-k2-thinking@falsifier,"
                "or:qwen/qwen3.7-max@expert,"
                "or:z-ai/glm-5.1@contrarian,"
                "or:minimax/minimax-m3@skeptic"),
    "deepseek": ("or:deepseek/deepseek-v4-pro@builder,"
                 "or:deepseek/deepseek-v4-flash@falsifier"),
    # zero-cost panel via OpenRouter :free slugs, diverse families. Lower
    # quality than paid — use when cost matters more than peak accuracy.
    "free": ("or:meta-llama/llama-3.3-70b-instruct:free@builder,"
             "or:qwen/qwen3-next-80b-a3b-instruct:free@expert,"
             "or:openai/gpt-oss-120b:free@falsifier,"
             "or:google/gemma-4-31b-it:free@contrarian"),
    "cli": "agy@builder,codex@skeptic,gemini@expert,cursor@contrarian,claude@generalist",
    "mixed": ("or:openai/gpt-4o@builder,or:anthropic/claude-sonnet-4.6@falsifier,"
              "or:google/gemini-2.5-flash@expert,or:deepseek/deepseek-v4-pro@contrarian"),
}

# Baked-in reasoning instructions, distilled from the user's CLAUDE.md
# (cognitive honesty + anti-bias + epistemic tags) and the /pizdej self-critique
# skill. Prepended to every PANEL prompt unless --reasoning off.
REASONING_PREAMBLE = """Before answering, reason step by step. Apply these thinking rules:
- Tag key claims by source: [O]bserved (data/logs/experiment) / [I]nferred (deduction) \
/ [S]tated (per a source/opinion) / [A]ssumed (no evidence).
- State a calibrated confidence (%) for your main claim, and why.
- Cognitive honesty: do NOT pander or agree just to please. If the premise is flawed or \
you disagree, say so and argue. If you are unsure, say so — never fake certainty.
- Minimal scope: answer exactly what is asked; do not invent requirements.
Before finalizing, run a self-critique (the "пиздёж check"): What did I NOT consider? \
Where could I be wrong? What alternative explanation exists? Then fix the answer.
Give your final answer after this reasoning.
"""


# --- env / keys ------------------------------------------------------------

def load_dotenv() -> None:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def api_key_for(backend: str) -> str | None:
    if backend in ("or", "openrouter"):
        return os.environ.get("OPENROUTER_API_KEY")
    if backend == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if backend == "openai":
        return os.environ.get("OPENAI_API_KEY")
    if backend == "google":
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return None


# --- member parsing --------------------------------------------------------

class Member:
    def __init__(self, label: str, kind: str, backend: str, model: str | None,
                 role_text: str, depth: str | None = None) -> None:
        self.label = label
        self.kind = kind  # "cli" | "api"
        self.backend = backend
        self.model = model
        self.role_text = role_text
        self.depth = depth  # per-member depth override; None = use global --depth


def parse_member(token: str, same_mode: bool, seen: dict[str, int]) -> Member:
    role_key = "neutral"
    head = token.strip()
    if "@" in head:
        head, role_key = head.split("@", 1)
        head, role_key = head.strip(), role_key.strip()
    if same_mode:
        role_key = "neutral"
    role_text = ROLES.get(role_key, ROLES["neutral"])

    depth: str | None = None
    if "!" in head:
        head, depth_key = head.split("!", 1)
        head, depth_key = head.strip(), depth_key.strip().replace("oneshot", "one-shot")
        if depth_key not in ("agent", "one-shot"):
            raise ValueError(f"bad depth {depth_key!r} (use agent|one-shot)")
        depth = depth_key

    model: str | None = None
    if ":" in head:
        backend, model = head.split(":", 1)
        backend, model = backend.strip(), model.strip()
    else:
        backend = head.strip()

    if backend in API_BACKENDS:
        kind = "api"
        model = model or API_DEFAULT_MODEL.get(backend)
    elif backend in KNOWN_CLIS:
        kind = "cli"
    else:
        raise ValueError(f"unknown backend: {backend!r} (clis={sorted(KNOWN_CLIS)}, "
                         f"api={sorted(API_BACKENDS)})")

    base = backend if model is None else f"{backend}:{model.split('/')[-1]}"
    seen[base] = seen.get(base, 0) + 1
    label = base if seen[base] == 1 else f"{base}#{seen[base]}"
    return Member(label, kind, backend, model, role_text, depth)


def parse_panel(spec: str, same_mode: bool) -> list[Member]:
    seen: dict[str, int] = {}
    members: list[Member] = []
    for tok in spec.split(","):
        if tok.strip():
            members.append(parse_member(tok, same_mode, seen))
    return members


# --- CLI dispatch ----------------------------------------------------------

def model_args(backend: str, model: str | None) -> list[str]:
    if not model:
        return []
    if backend == "codex":
        return ["-c", f"model={model}"]
    if backend in ("gemini", "opencode"):
        return ["-m", model]
    return ["--model", model]  # cursor, agy, claude


def cli_argv(backend: str, model: str | None, prompt: str, depth: str) -> tuple[list[str], str | None]:
    m = model_args(backend, model)
    if depth == "agent":
        if backend == "codex":
            return (["codex", "exec", "--sandbox", "workspace-write", *m, "-"], prompt)
        if backend == "claude":
            return (["claude", "-p", prompt, "--dangerously-skip-permissions", *m], None)
        if backend == "gemini":
            return (["gemini", "-p", prompt, "--yolo", *m], None)
        if backend == "cursor":
            return (["cursor-agent", "-p", prompt, "--force", "--trust", *m], None)
        if backend == "agy":
            return (["agy", "-p", prompt, "--dangerously-skip-permissions", *m], None)
        if backend == "opencode":
            return (["opencode", "run", prompt, *m], None)
    else:  # one-shot, read-only
        if backend == "codex":
            return (["codex", "exec", "--sandbox", "read-only", *m, "-"], prompt)
        if backend == "claude":
            return (["claude", "-p", prompt, *m], None)
        if backend == "gemini":
            return (["gemini", "-p", prompt, "--approval-mode", "plan", *m], None)
        if backend == "cursor":
            return (["cursor-agent", "-p", prompt, "--mode", "ask", "--trust", *m], None)
        if backend == "agy":
            return (["agy", "-p", prompt, "--sandbox", *m], None)
        if backend == "opencode":
            return (["opencode", "run", prompt, "--agent", "plan", *m], None)
    raise ValueError(f"unknown cli backend: {backend!r}")


def dispatch_cli(m: Member, prompt: str, depth: str, timeout: int,
                 agent_cwd: str = "temp") -> dict:
    argv, stdin_payload = cli_argv(m.backend, m.model, prompt, depth)
    # agent mode normally runs in an isolated temp cwd (repo never mutated).
    # agent_cwd="current" keeps the parent cwd so the agent can read/explore the
    # real repo (needed for codebase analysis) — riskier, prompt should say read-only.
    use_temp = (depth == "agent" and agent_cwd == "temp")
    cwd = tempfile.mkdtemp(prefix="fusion_") if use_temp else None
    try:
        proc = subprocess.run(argv, input=stdin_payload, capture_output=True,
                              text=True, timeout=timeout, cwd=cwd)
        answer = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if not answer and err:
            answer = f"[no stdout; stderr]: {err[:500]}"
        return {"label": m.label, "backend": m.backend, "kind": "cli",
                "ok": proc.returncode == 0, "answer": answer, "returncode": proc.returncode}
    except subprocess.TimeoutExpired:
        return {"label": m.label, "backend": m.backend, "kind": "cli", "ok": False,
                "answer": f"[TIMEOUT after {timeout}s]", "returncode": -1}
    except FileNotFoundError:
        return {"label": m.label, "backend": m.backend, "kind": "cli", "ok": False,
                "answer": f"[CLI not found: {argv[0]}]", "returncode": -2}
    finally:
        if cwd:
            shutil.rmtree(cwd, ignore_errors=True)


# --- API dispatch ----------------------------------------------------------

def _http_json(url: str, headers: dict[str, str], body: dict, timeout: int) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def dispatch_api(m: Member, prompt: str, timeout: int) -> dict:
    key = api_key_for(m.backend)
    res = {"label": m.label, "backend": m.backend, "kind": "api", "model": m.model}
    if not key:
        res.update(ok=False, answer=f"[missing API key for {m.backend}]")
        return res
    try:
        if m.backend in ("or", "openrouter"):
            out = _http_json(
                "https://openrouter.ai/api/v1/chat/completions",
                {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "X-Title": "fusion-skill"},
                {"model": m.model, "messages": [{"role": "user", "content": prompt}]},
                timeout)
            text = out["choices"][0]["message"]["content"]
        elif m.backend == "openai":
            out = _http_json(
                "https://api.openai.com/v1/chat/completions",
                {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                {"model": m.model, "messages": [{"role": "user", "content": prompt}]},
                timeout)
            text = out["choices"][0]["message"]["content"]
        elif m.backend == "anthropic":
            out = _http_json(
                "https://api.anthropic.com/v1/messages",
                {"x-api-key": key, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"},
                {"model": m.model, "max_tokens": 4096,
                 "messages": [{"role": "user", "content": prompt}]},
                timeout)
            text = out["content"][0]["text"]
        elif m.backend == "google":
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{m.model}:generateContent?key={key}")
            out = _http_json(url, {"Content-Type": "application/json"},
                             {"contents": [{"parts": [{"text": prompt}]}]}, timeout)
            text = out["candidates"][0]["content"]["parts"][0]["text"]
        else:
            res.update(ok=False, answer=f"[unknown api backend {m.backend}]")
            return res
        res.update(ok=True, answer=(text or "").strip())
        return res
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        res.update(ok=False, answer=f"[HTTP {e.code}]: {body}")
        return res
    except (urllib.error.URLError, KeyError, IndexError, TimeoutError) as e:
        res.update(ok=False, answer=f"[API error: {type(e).__name__}: {e}]")
        return res


def dispatch(m: Member, prompt: str, depth: str, timeout: int, reasoning: bool = True,
             agent_cwd: str = "temp") -> dict:
    head = f"{REASONING_PREAMBLE}\n" if reasoning else ""
    full = f"{head}{m.role_text}\n\n---\n\n{prompt}"
    if m.kind == "cli":
        return dispatch_cli(m, full, m.depth or depth, timeout, agent_cwd)
    return dispatch_api(m, full, timeout)


# --- judge -----------------------------------------------------------------

JUDGE_INSTRUCTION = """You are the JUDGE in a multi-model deliberation. \
Several models answered the same prompt. Do NOT merge them into one answer. \
Instead analyze them and return ONLY a JSON object with these keys:

{
  "consensus": [points all/most agree on],
  "contradictions": [{"point": "...", "sides": "model A says X, model B says Y"}],
  "coverage_gaps": [important aspects no model addressed],
  "unique_insights": [{"model": "label", "insight": "..."}],
  "blind_spots": [shared mistakes or assumptions across answers],
  "recommendation": "one paragraph: how the drafter should synthesize the final answer"
}

Return the JSON and nothing else."""


def run_judge(judge_member: Member, user_prompt: str, panel: list[dict], timeout: int) -> dict:
    answers_block = "\n\n".join(
        f"### Model: {m['label']} ({m['backend']})\n{m['answer']}" for m in panel)
    judge_prompt = (f"{JUDGE_INSTRUCTION}\n\n## Original prompt\n{user_prompt}\n\n"
                    f"## Panel answers\n{answers_block}")
    # judge ignores role text; pass directly via a neutral member dispatch
    jm = Member(judge_member.label, judge_member.kind, judge_member.backend,
                judge_member.model, ROLES["neutral"])
    res = dispatch_cli(jm, judge_prompt, "one-shot", timeout) if jm.kind == "cli" \
        else dispatch_api(jm, judge_prompt, timeout)
    raw = res.get("answer", "")
    return {"backend": jm.backend, "raw": raw, "parsed": _extract_json(raw)}


def _extract_json(text: str) -> dict | None:
    s = text.strip()
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) > 1 else text
        if s.startswith("json"):
            s = s[4:]
    s = s.strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None


# --- main ------------------------------------------------------------------

def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Fusion: panel -> judge (-> optional auto-draft)")
    ap.add_argument("prompt", nargs="?", help="prompt; if omitted, read from stdin")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="core",
                    help="named panel preset (default: core = 5 diverse advisors); ignored if --panel given")
    ap.add_argument("--panel", default=None,
                    help="comma list BACKEND[:MODEL][@ROLE]; overrides --preset")
    ap.add_argument("--judge", default="claude",
                    help="judge member — should be the STRONGEST available (default: claude CLI)")
    ap.add_argument("--reasoning", choices=["on", "off"], default="on",
                    help="prepend baked-in reasoning preamble to panel prompts (default: on)")
    ap.add_argument("--mode", choices=["role", "same"], default="role")
    ap.add_argument("--depth", choices=["one-shot", "agent"], default="one-shot",
                    help="CLI members: one-shot answer vs full agent (tools/subagents)")
    ap.add_argument("--agent-cwd", choices=["temp", "current"], default="temp",
                    help="agent-mode working dir: temp (isolated, safe) or current "
                         "(the real repo — needed to analyze a codebase; use read-only prompts)")
    ap.add_argument("--timeout", type=int, default=240, help="per-call timeout seconds")
    ap.add_argument("--auto-draft", metavar="MEMBER",
                    help="also run this member as drafter; include final answer")
    args = ap.parse_args()

    user_prompt = args.prompt or sys.stdin.read().strip()
    if not user_prompt:
        print("ERROR: empty prompt", file=sys.stderr)
        return 2

    panel_spec = args.panel if args.panel else PRESETS[args.preset]
    reasoning = (args.reasoning == "on")
    members = parse_panel(panel_spec, same_mode=(args.mode == "same"))
    if not members:
        print("ERROR: empty panel", file=sys.stderr)
        return 2
    judge_member = parse_member(args.judge, same_mode=True, seen={})

    src = "panel" if args.panel else f"preset={args.preset}"
    print(f"[fusion] {src} members={[m.label for m in members]} judge={judge_member.label} "
          f"mode={args.mode} depth={args.depth} reasoning={args.reasoning} "
          f"timeout={args.timeout}s", file=sys.stderr)

    panel_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(members)) as ex:
        futs = {ex.submit(dispatch, m, user_prompt, args.depth, args.timeout, reasoning,
                          args.agent_cwd): m.label
                for m in members}
        for fut in as_completed(futs):
            res = fut.result()
            print(f"[fusion] panel {res['label']}: {'ok' if res['ok'] else 'FAIL'} "
                  f"({len(res['answer'])} chars)", file=sys.stderr)
            panel_results.append(res)

    order = {m.label: i for i, m in enumerate(members)}
    panel_results.sort(key=lambda r: order.get(r["label"], 99))

    print(f"[fusion] running judge ({judge_member.label})...", file=sys.stderr)
    judge = run_judge(judge_member, user_prompt, panel_results, args.timeout)

    bundle = {"prompt": user_prompt, "depth": args.depth,
              "reasoning": args.reasoning,
              "preset": None if args.panel else args.preset,
              "panel": panel_results, "judge": judge}

    if args.auto_draft:
        dm = parse_member(args.auto_draft, same_mode=True, seen={})
        print(f"[fusion] auto-draft via {dm.label}...", file=sys.stderr)
        draft_prompt = (
            "You are the DRAFTER. Write the final answer to the original prompt, "
            "using the judge's analysis to resolve contradictions, fill gaps, and "
            "avoid the shared blind spots.\n\n"
            f"## Original prompt\n{user_prompt}\n\n"
            f"## Judge analysis (JSON)\n{judge['raw']}\n\nWrite the final answer now.")
        dr = dispatch_cli(dm, draft_prompt, "one-shot", args.timeout) if dm.kind == "cli" \
            else dispatch_api(dm, draft_prompt, args.timeout)
        bundle["draft"] = dr.get("answer", "")

    print(json.dumps(bundle, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
