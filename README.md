# Fusion

Fusion runs several CLI/API models in parallel, filters failed calls, validates a structured judge response, optionally runs a review round or adaptive escalation, and can draft a final answer.

## Defaults

There is one runtime default path:

- `--strategy adaptive`
- adaptive routes simple prompts to `lite` and complex prompts to `pro`
- without benchmark data, `lite` falls back to `dq` and `pro` falls back to `core`
- `--judge claude`, `--depth one-shot`, `--reasoning on`
- agent mode uses an isolated read-only `snapshot`; direct execution in the current repository is not supported

Inspect routing without invoking a model:

```bash
python3 scripts/fusion.py "Design a payment system" --dry-run
```

## Reliability and observability

Every invocation returns a `ModelResult` for each call with latency, token usage, cost when reported or configured, errors, attempts, and extracted confidence. API and CLI calls use bounded exponential backoff. Failed or empty panel results are excluded from the judge.

The judge must match `schemas/judge.schema.json`. Invalid output is sent through a repair loop. The final JSON bundle includes selection rationale and aggregate metrics.

Optional direct-provider price estimates can be supplied in USD per million tokens:

```bash
export FUSION_PRICING_JSON='{"gpt-4o":{"input":5,"output":15}}'
```

## Strategies

```bash
# cheap, compact panel
python3 scripts/fusion.py "question" --strategy lite

# broader panel, review round, stronger repair policy, source answers retained for drafting
python3 scripts/fusion.py "question" --strategy pro --auto-draft claude

# route by prompt complexity and benchmark results; escalate once to power on risk signals
python3 scripts/fusion.py "question" --strategy adaptive
```

An explicit `--preset` or `--panel` overrides automatic panel selection.

## Review and adaptive escalation

`pro` runs a review round by default: two successful backend-diverse panel members revise their answers after the first judge pass, then the judge evaluates the expanded panel again.

`adaptive` can escalate once to the `power` preset when the first pass looks risky: too few successful members, invalid judge JSON, low judge confidence, or substantial contradictions/gaps. Disable this with `--no-escalate`. Explicit `--panel` or `--preset` also disables escalation.

## Safe agent workspaces

`--agent-workspace snapshot` copies the source tree, excludes `.git`, `.env`, caches and dependencies, and marks copied files read-only. `worktree` creates a detached disposable Git worktree. `temp` gives the agent an empty directory. These modes reduce accidental repository mutation; they are not an OS-level sandbox.

```bash
python3 scripts/fusion.py "Inspect this codebase" --depth agent \
  --agent-workspace worktree --workspace-source .
```

## Benchmarks

The fixture is `benchmarks/cases.jsonl`. Run panels and write `benchmarks/results.json`:

```bash
python3 scripts/benchmark.py --presets dq core --judge claude --drafter claude
```

Adaptive selection uses quality, category score, latency, cost, and optional constraints:

```bash
python3 scripts/fusion.py "question" --strategy adaptive \
  --budget-usd 0.02 --max-latency-ms 15000
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```
