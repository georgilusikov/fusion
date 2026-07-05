# Fusion Golden A/B Protocol

Purpose: prove Fusion is useful before adding orchestration complexity.

## Golden set

Use `benchmarks/golden/fusion_golden_set.jsonl`. Cases should be real prompts or lightly anonymized real prompts, not synthetic keyword fixtures.

## Rematch variants

1. `fusion-pro` — current pro pipeline.
2. `solo-claude` — one strong baseline answer.
3. `claude-x3-self-pick` — three same-model samples with self-pick.

## Blind judging

Judges receive anonymous answers only. They score each answer on:

- correctness
- depth
- coverage
- actionability

Use at least two cross-provider judges when possible. Aggregate by median.

## Kill rule

If `solo-claude` or `claude-x3-self-pick` matches or beats `fusion-pro`, stop expanding the panel and fix synthesis/keep-best first.

## Commands

```bash
python3 scripts/rematch.py --dry-run
python3 scripts/rematch.py --answers tmp/rematch_answers.jsonl --output tmp/blind_packets.json
python3 scripts/rematch.py --judge-payloads tmp/judge_payloads.jsonl
```
