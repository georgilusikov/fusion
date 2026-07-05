# Fusion Phase 2 Experiments

Phase 2 features are not defaults. They must be justified by a failed golden-set or outcome-ledger case.

## Debate round

Use debate when review-round revisions are too polite or miss contradictions between answers. The debate packet asks a model to attack and defend claims, then return a replacement answer that survives the critique.

## Multilingual axis

Use language diversity as an A/B axis, not as a default. Candidate axes are `ru`, `en`, and `zh`. Chinese-model prompting should be measured against the golden set before defaulting it on.

## Outcome ledger

Record whether a Fusion answer changed a real action and whether it was later corrected:

```bash
python3 scripts/outcome_ledger.py --case-id CASE --answer-id RUN --affected-action --notes "used in PR triage"
```

## Stop rule

No new Fusion feature should be added without at least one linked failed case from the golden set or the outcome ledger.
