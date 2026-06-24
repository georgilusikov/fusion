---
name: fusion
description: >
  Resilient local multi-model deliberation: adaptive panel selection, structured
  judge validation, retries, metrics, safe agent workspaces, and optional drafting.
---

# Fusion — Multi-Model Deliberation

Pipeline: **Panel → validated Judge → optional Drafter**.

## Canonical defaults

The CLI is the source of truth:

| Flag | Default |
|---|---|
| `--strategy` | `adaptive` |
| `--preset` | automatically selected |
| `--judge` | `claude` |
| `--reasoning` | `on` |
| `--mode` | `role` |
| `--depth` | `one-shot` |
| `--agent-workspace` | `snapshot` |
| `--timeout` | `240` |
| `--retries` | `2` |
| `--backoff` | `0.5` seconds |
| `--repair-attempts` | `1` (`pro` enforces at least `2`) |

`adaptive` classifies prompt complexity. Without benchmark results it maps
`lite → dq` and `pro → core`. An explicit `--preset` or `--panel` wins.

## Invocation

```bash
python3 scripts/fusion.py "PROMPT"
echo "PROMPT" | python3 scripts/fusion.py
python3 scripts/fusion.py "PROMPT" --dry-run
```

## Strategies

- `lite`: compact, inexpensive panel; explicit decomposition and calibrated confidence.
- `pro`: broader panel, stronger schema-repair policy, and source answers supplied to the drafter.
- `adaptive`: routes to `lite` or `pro`, then uses benchmark utility when available.

## Result contract

Every model call produces `ModelResult` (`schemas/model-result.schema.json`):

- identity: label, backend, kind, model
- output: ok, answer, errors, confidence
- operations: latency, attempts, return code
- usage: input/output/total tokens and cost when reported or price-configured

Failed and empty answers remain visible in `panel`, but they are excluded from the judge.

## Judge validation

The judge must match `schemas/judge.schema.json`. Fusion extracts the first JSON
object, validates required fields/types/additional properties, and asks the judge
to repair invalid output. The raw response and every repair call remain in the bundle.

## Retries

Transient API failures (`408`, `409`, `425`, `429`, and `5xx`) and transport
errors use bounded retries with exponential backoff and jitter. CLI failures and
timeouts use the same configured attempt budget; missing executables fail immediately.

## Safe agent mode

There is no `agent-cwd=current` path.

- `snapshot` (default): copies `--workspace-source`, excludes `.git`, `.env`, caches,
  virtual environments and dependencies, then marks copied files read-only.
- `worktree`: detached temporary Git worktree; removed after the call.
- `temp`: empty temporary directory.

Each panel agent gets its own isolated workspace. This prevents routine cwd mutation, but is not an OS-level security sandbox.

## Presets

`core`, `power`, `dq`, `all`, `chinese`, `deepseek`, `free`, `cli`, and `mixed`
remain available. Model slugs may drift; override them with `--panel` when needed.

Member grammar: `BACKEND[:MODEL][!DEPTH][@ROLE]`.

Examples:

```bash
python3 scripts/fusion.py "question" --strategy lite
python3 scripts/fusion.py "question" --strategy pro --auto-draft claude
python3 scripts/fusion.py "Inspect this repo" --depth agent \
  --agent-workspace worktree --workspace-source .
python3 scripts/fusion.py "question" \
  --panel "codex@skeptic,or:openai/gpt-4o@builder" --judge claude
```

## Benchmark-driven routing

`benchmarks/cases.jsonl` contains deterministic checks. Generate aggregate data:

```bash
python3 scripts/benchmark.py --presets dq core --judge claude --drafter claude
```

The generated `benchmarks/results.json` is local and gitignored. Adaptive routing
uses quality, category quality, average latency, average cost, `--budget-usd`, and
`--max-latency-ms`. When data is absent, deterministic fallbacks keep behavior stable.

## API keys and optional pricing

Keys are loaded from environment or a gitignored `.env`:

- OpenRouter: `OPENROUTER_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- OpenAI: `OPENAI_API_KEY`
- Google: `GEMINI_API_KEY` or `GOOGLE_API_KEY`

Provider-reported costs are preferred. For direct APIs, optionally set
`FUSION_PRICING_JSON` or `FUSION_PRICING_FILE`; values are USD per million tokens.

## Verification

```bash
python3 -m compileall -q fusion_core scripts tests
python3 -m unittest discover -s tests -v
python3 scripts/benchmark.py --dry-run --limit 1
```
