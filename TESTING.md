# Testing Fusion

Three layers. Each answers a different question. Do not skip layers: a green
Layer 0 says nothing about answer quality, and Layer 2 is meaningless if the
mechanics are broken.

| Layer | Question | Cost |
|---|---|---|
| 0 — mechanics | does the pipeline run at all? | free, seconds |
| 1 — deterministic benchmark | do panels pass known checks? | cheap, API calls |
| 2 — A/B eval vs baseline | is fusion *better* than one strong model? | the real test |

---

## Layer 0 — Mechanics (no quality signal)

```bash
python3 -m compileall -q fusion_core scripts tests   # syntax
python3 -m unittest discover -s tests -v             # unit: retry, judge schema, workspace, rounds
python3 scripts/fusion.py "test" --dry-run           # smoke, no API calls
```

**Look at:** exit codes only. This proves retries, judge validation, and
workspace isolation work — nothing about whether answers are good.

---

## Layer 1 — Deterministic benchmark

```bash
python3 scripts/benchmark.py --presets dq core --judge claude --drafter claude
```

Writes `benchmarks/results.json` (local, gitignored). Adaptive routing consumes it.

**Look at, per preset:**

- `quality` — mean 0..1 score over cases. **Warning:** with easy cases every
  preset saturates near 1.0 (ceiling effect) and the number stops
  discriminating. If all presets score ≥0.9, the case set is too easy —
  fix the cases, don't celebrate.
- `by_category` — per-category quality. This is what routing actually uses.
- `avg_latency_ms`, `avg_cost` — the price of the panel.
- Per-run failures — empty/invalid panel answers (`ok: false` in the bundle).

**Layer 1 measures "not broken", not "better".** For "better" you need Layer 2.

---

## Layer 2 — A/B eval: fusion vs single strong model

The only test that answers the real question. Protocol:

### Arms

Run every case through three arms:

1. **baseline** — single strongest model (plain `claude`, no panel)
2. `fusion --strategy lite`
3. `fusion --strategy pro`

### Rules

- **N ≥ 3 runs per case per arm.** Models are nondeterministic; a single run
  is noise, not signal.
- **Blind pairwise judging, cross-provider.** The judge compares two answers
  (A vs B) without arm labels, order randomized per pair (position bias is
  real). The judge must be a *different vendor* than the drafter —
  `judge=claude` scoring claude-drafted answers has measurable
  self-preference bias. Use gemini/gpt to judge claude output.
- **Penalize length.** Judges favor longer answers. Instruct the judge
  explicitly: verbosity without content is a defect, not a virtue.

### Metrics (all four, not just one)

| Metric | Source |
|---|---|
| win-rate vs baseline | pairwise judge verdicts |
| cost per case | `usage` in ModelResult |
| latency per case | `latency` in bundle |
| panel fail-rate | empty/invalid answers in `panel` |

### Verdict rule

- fusion "better" = **win-rate > 60%** at **cost ≤ 5× baseline**
- win-rate 50–55% = noise; the panel does not pay for itself
- Report per-category, not one global number: the expected outcome is a
  **map of where the panel pays off** (open trade-offs, multi-constraint
  design) and where it doesn't (facts, simple code). Feed that map back
  into adaptive routing.

### Adaptive routing check (separate, deterministic)

10 easy + 10 hard prompts → assert `adaptive` routes lite/pro correctly by
inspecting the bundle. No judge needed.

---

## Designing cases that discriminate

A case discriminates ⟺ a single model *predictably fails* it and panel
disagreement catches the failure. Cases every model solves are ballast.

**Admission filter:** run the case 3× on one strong model.
3/3 correct → too easy, drop it. 0/3 → check the panel solves it at all.
Keep cases scoring 1–2/3 — that's the discrimination zone.

### Categories, by signal strength

1. **Intuition traps** (strongest, deterministic checks): the "obvious" answer
   is wrong. Subtle bugs (off-by-one, mutable defaults, timezone, float `==`,
   `is` vs `==` for ints > 256), probability puzzles with a twist.
   Check: `must_include` the correct answer **and** `must_not_include` the
   typical wrong one — catching the *presence of the typical error* is half
   the power.
2. **False premise** (strong, deterministic): the question embeds a falsehood
   ("Why was the GIL removed in Python 3.9?"). Single models often play along.
   Check: `must_include_any` of refutation phrases, `must_not_include` the
   play-along explanation.
3. **Fabrication bait**: exact quotes, obscure version numbers. Measures
   honest "I don't know" vs hallucination. `must_not_include` invented specifics.
4. **Multi-constraint design** (5+ constraints): easy to drop one constraint;
   panel members cover each other. Score = constraints covered / total.
   Expect the largest fusion advantage here — aggregation helps mechanically.
5. **Real regressions** (highest value per case, few available): questions
   where a model actually failed you before, with the known correct answer.
6. **Open trade-offs** (judged only, no string checks): architecture choices,
   plan reviews. Pairwise judge with a rubric: alternatives considered,
   recommendation given, risks named. Weakest/most expensive signal, but this
   is fusion's primary use case — an eval without them is unrepresentative.

**Do not include:** facts, translation, summaries, simple code, "explain X" —
zero gap, pure cost.

### Suggested distribution (30 cases)

| Category | Count | Check type |
|---|---|---|
| intuition traps | 8 | deterministic |
| false premise | 5 | deterministic |
| fabrication bait | 4 | deterministic |
| multi-constraint | 5 | checklist score |
| real regressions | 4 | deterministic |
| open trade-offs | 4 | pairwise judge |

### Case format (extends `benchmarks/cases.jsonl`)

```json
{"id":"false-premise-gil","category":"trap","difficulty":3,"prompt":"Почему в Python 3.9 удалили GIL и как это повлияло на threading?","checks":{"must_include_any":[["не удал","не был удалён","GIL остаётся","предпосылка"]],"must_not_include":["GIL был удалён в 3.9"]}}
{"id":"int-identity","category":"trap","difficulty":2,"prompt":"a = 257; b = 257; print(a is b) — что выведет и почему? Ответь для скрипта-файла и для REPL отдельно.","checks":{"must_include_any":[["кэш","интернирование","small int","256"]],"must_include":["REPL"]}}
```

---

## Red flags checklist (what invalidates a result)

- [ ] All presets ≥0.9 on Layer 1 → ceiling; case set too easy, numbers meaningless
- [ ] Judge and drafter from the same vendor → self-preference bias
- [ ] Fixed A/B order in pairwise judging → position bias
- [ ] Single run per case → nondeterminism noise read as signal
- [ ] Judge not told to penalize verbosity → length bias inflates fusion (panels write more)
- [ ] Only a global win-rate reported → hides that fusion loses on cheap categories; always report per-category
