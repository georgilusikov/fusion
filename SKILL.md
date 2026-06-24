---
name: fusion
description: >
  Local analog of OpenRouter's Fusion plugin — multi-model deliberation.
  A panel of vendors (local CLIs: agy/codex/gemini/cursor/claude, and/or API
  models via OpenRouter or direct keys) answers a prompt in parallel; a judge
  model emits structured analysis (consensus / contradictions / coverage gaps /
  unique insights / blind spots); then the drafter (you, the conversing model)
  writes the final answer.
  Use when user says "fusion", "fuse", "panel of models", "multi-model answer",
  "ask several models", "deliberation", "consensus answer", "best-of models".
  FOR: GENERATING a high-quality answer by cross-vendor deliberation.
  NOT for: auditing finished work (use /bouncer); self-verification (use /check).
  Difference from bouncer: bouncer = N reviewers score existing work;
  fusion = N generators answer + judge + you synthesize.
---

# Fusion — Multi-Model Deliberation (local OpenRouter Fusion analog)

Three roles: **Panel** (N members answer in parallel) → **Judge** (one model
marks consensus/conflict/gaps/blind-spots as JSON, does NOT merge) → **Drafter**
(YOU read the bundle and write the final answer; or a CLI/API member via
`--auto-draft` for headless runs).

**Strength asymmetry:** the panel can be cheap and diverse (default = Chinese
models via OpenRouter), but the **judge and drafter should be the strongest model
available** — they do the hard reasoning. Default judge = `claude` CLI; default
drafter = you (the conversing model). Use a local CLI for judge/drafter so you
get a top model without extra cost.

## Pre-start protocol (when invoked as a skill)

The user's message is the goal. Before running, ask ONE `AskUserQuestion`
bundling the knobs that aren't obvious from the request:
- **panel/preset** — default `core` (5 diverse: claude/gemini/cursor/agy+deepseek); `dq` cheapest, `all` ~10 advisors, `free` zero-cost, or custom `--panel`
- **depth** — one-shot vs agent (only if task may need tools/research)
- **judge** — default `claude` (strongest local CLI); keep it strong

Skip the question if the user already specified them or said "just run it".
Then run the script, read the bundle, and draft.

## Invocation

```bash
python3 scripts/fusion.py "PROMPT" [options]
echo "PROMPT" | python3 scripts/fusion.py
```

| Flag | Default | Meaning |
|---|---|---|
| `--preset` | `chinese` | named panel preset (see below); ignored if `--panel` given |
| `--panel` | (preset) | comma list of members; overrides `--preset` |
| `--judge` | `claude` | judge member — use the STRONGEST available |
| `--reasoning` | `on` | prepend baked-in reasoning preamble to panel prompts |
| `--mode` | `role` | `role` = role-diverse; `same` = identical neutral role |
| `--depth` | `one-shot` | `one-shot` (read-only answer) / `agent` (full tools/subagents) |
| `--timeout` | `240` | per-call timeout (s) |
| `--auto-draft MEMBER` | off | run MEMBER as drafter, include final answer |

## Panel presets

`--preset NAME` (default `core`). `--panel` overrides. Slugs = latest on
OpenRouter as of 2026-06-15.

| Preset | Members |
|---|---|
| `core` (default) | claude · gemini · cursor · agy (local CLIs) + deepseek-v4-pro — 5 diverse advisors, cheap (CLIs via subscriptions) |
| `power` | 7 advisors across distinct base models: codex(gpt-5.5) · cursor(composer-2.5, agent) · claude(opus) · agy(gemini-3.5-flash, agent) · agy(claude-opus-4.6) + deepseek-v4-pro + qwen3.7-max. Subscriptions + subagents — strongest, slower |
| `dq` | deepseek-v4-pro + qwen3.7-max (cheapest, fast) |
| `all` | **every advisor**: agy + cursor (agents/subagents), codex, gemini, claude (CLIs) + deepseek-v4-pro, kimi-k2-thinking, qwen3.7-max, glm-5.1, minimax-m3. ~10 — heavy/slow/costly |
| `chinese` | deepseek-v4-pro · kimi-k2-thinking · qwen3.7-max · glm-5.1 · minimax-m3 |
| `deepseek` | deepseek-v4-pro + deepseek-v4-flash |
| `free` | llama-3.3-70b · qwen3-next-80b · gpt-oss-120b · gemma-4 (OpenRouter `:free`) — zero cost, but rate-limited (HTTP 429) and lower quality |
| `cli` | agy · codex · gemini · cursor · claude · opencode (local CLIs, one-shot) |
| `mixed` | gpt-4o · claude-sonnet-4.6 · gemini-2.5-flash · deepseek-v4-pro |

Default `dq` for everyday use; `--preset all` for max advisor diversity.

OpenRouter model slugs drift — list valid ones with
`curl -s https://openrouter.ai/api/v1/models -H "Authorization: Bearer $OPENROUTER_API_KEY"`.

## Reasoning preamble (`--reasoning on`, default)

Every panel prompt is prefixed with a baked-in thinking preamble distilled from
the user's CLAUDE.md (cognitive honesty + anti-bias epistemic tags `[O]/[I]/[S]/[A]`
+ confidence calibration + minimal scope) and the `/pizdej` self-critique skill
(what did I NOT consider / where could I be wrong / alternatives, then fix).
Makes panel members reason harder before answering. `--reasoning off` to disable.

## Member grammar — `BACKEND[:MODEL][!DEPTH][@ROLE]`

| Example | Meaning |
|---|---|
| `codex@skeptic` | CLI vendor, default model, role skeptic |
| `claude` | CLI, neutral role |
| `cursor!agent@falsifier` | CLI in **agent mode** (tools/subagents), role falsifier |
| `agy!agent@builder` | antigravity as a full agent (spawns its own subagents) |
| `or:deepseek/deepseek-v3.2@expert` | OpenRouter API, explicit model |
| `or@builder` | OpenRouter, default model (deepseek-v4-pro) |
| `openai:gpt-4o@builder` | direct OpenAI API |
| `google:gemini-2.0-flash@contrarian` | direct Google API |

`!DEPTH` (`agent` or `one-shot`) overrides the global `--depth` for that one
member — e.g. run agy/cursor as agents while the rest stay one-shot.

- **CLI backends**: `agy`, `codex`, `gemini`, `cursor`, `claude`, `opencode`
  (no API key needed; each uses its own configured auth — opencode routes via its
  configured provider, e.g. Z.AI/GLM). Free OpenRouter models: use `:free` slugs
  on the `or` backend, e.g. `or:meta-llama/llama-3.3-70b-instruct:free`.
- **API backends**: `or`/`openrouter` (one key, any model), `anthropic`,
  `openai`, `google`. Bare `or` (no model) defaults to `deepseek/deepseek-v4-pro`.
- **Roles**: `builder`, `skeptic`, `expert`, `contrarian`, `falsifier`, `generalist`, `neutral`.

## Depth modes

- `one-shot` (default) — each CLI answers directly in a read-only mode
  (`codex --sandbox read-only`, `gemini --approval-mode plan`,
  `cursor --mode ask`, `agy --sandbox`, `claude -p`). Fast, cheap, safe.
- `agent` — each CLI runs as a FULL agent (tools, multi-step, can spawn its own
  subagents): `codex --sandbox workspace-write`, `gemini --yolo`,
  `cursor --force --trust`, `agy --dangerously-skip-permissions`,
  `claude --dangerously-skip-permissions`. Each runs in an **isolated temp cwd**
  so the real repo is never mutated. Slower/costlier; use for tasks needing
  research, code, or exploration.
- API members are always one-shot (depth ignored for them).

## API keys

Read from environment, or from a gitignored `.env` in the repo root
(`KEY=VALUE`, see `.env.example`). Never printed. Per backend:

| Backend | Env var |
|---|---|
| `or` / `openrouter` | `OPENROUTER_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `google` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |

A member whose key is missing returns a graceful FAIL row; the panel continues.

## Drafter step (you, after the script)

Read `judge.parsed` and write the final answer: lead with consensus, resolve
each contradiction (pick a side + reason), fill coverage gaps, avoid shared
blind spots, fold in worthwhile unique insights. Show the final answer first,
then a short "panel split on X / Y filled gap Z" note. Offer raw bundle.

## Example panels

```bash
# default: 5 cross-vendor CLIs, role-diverse, judge gemini
python3 .../fusion.py "вопрос"

# mix CLI + OpenRouter + direct, judge via claude CLI
python3 .../fusion.py "вопрос" \
  --panel "codex@skeptic,or:openai/gpt-4o@builder,anthropic:claude-3-5-sonnet-latest@expert" \
  --judge claude

# full-agent panel (research-heavy task), isolated temp cwds
python3 .../fusion.py "вопрос" --depth agent --panel "codex,cursor,claude"

# fully headless: CLI drafter, no human in loop
python3 .../fusion.py "вопрос" --auto-draft "or:anthropic/claude-3.5-sonnet"
```

## Safety / gotchas

- one-shot = read-only everywhere; agent = isolated temp cwd. No vendor mutates
  the real repo.
- Keys never touch code/git (env or gitignored `.env` only).
- Judge JSON is best-effort parsed; `judge.raw` always kept. If `parsed` is
  null, read `raw`.
- Cost scales with panel size × depth. Small one-shot panels for iteration;
  scale up for hard questions.
- Verified: CLI headless flags + depth modes + API no-key graceful path,
  2026-06-15.
