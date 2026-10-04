# Trustworthy AI Evaluation Architecture (v3)

Supersedes `trustworthy-ai-eval-architecture.md` (v2), which is **kept as-is** for the record.
This document is self-contained: it does not depend on v2 and can be read alone.

---

## 1. Premises

v2 rested on one premise. v3 keeps it and adds three that were learned by running the suite
against real data.

**1. An LLM judge is a calibrated, bounded signal — never the ground truth.**
Unchanged from v2, and still the foundation.

**2. A score is not a verdict until you know what the metric was asking.**
`HallucinationMetric` returned `1.000` and failed all six rows in a real run — including rows
the model answered perfectly. The number was correct; the *interpretation* was inverted,
because the metric scores agreement with the context and this dataset's context is
deliberately false. A metric that measures the wrong direction is more dangerous than a
metric that is merely noisy, because its output looks valid.

**3. Measure the dataset, not only the model.**
Several metrics never read the model's answer at all — `ContextualRelevancy`,
`ContextualPrecision`, `ContextualRecall`. Their failures are statements about the
spreadsheet, not the model. Treating them as model defects assigns blame to the wrong place;
treating them as data checks turns them into the framework's most useful self-diagnostic.

**4. A missing precondition must be loud, never a score.**
Three variants of the same failure exist in this project: an unavailable judge, an
unavailable generator, and a blank `Input` cell. Each one, left unguarded, produces a result
that *looks* like a measurement. Every precondition gets an explicit gate that halts and
reports a reason.

---

## 2. Settled decisions

These are closed. They are recorded so they are not relitigated.

| Decision | Resolution |
|---|---|
| Test shape | **Single-turn only.** One question, one answer. |
| Conversational / multi-turn metrics | **Out of scope.** Not a deferred capability. |
| DeepEval version | **Pinned to 3.9.5. Do not upgrade.** |
| Metrics that are a poor fit | **Keep registered. Control gating, never delete.** |
| Inverted metrics (`Hallucination`, `Faithfulness`) | **Keep. Invert the reading, not the metric.** |
| RAGAS family (6 metrics) | **Skip.** Redundant, judge-based, risky dependency. |

**Why pin 3.9.5:** `evaluators/metrics/generic.py` discovers metrics with
`dir(deepeval.metrics)`. Upgrading to 4.x would silently absorb 13 new classes, several of
them agentic metrics that declare only `input`/`actual_output` as required params — so they
would pass the field-presence applicability check and emit meaningless passes, exactly as
`PlanQualityMetric` does today. Curate the registry into an explicit list first; *then* an
upgrade is safe to consider.

**Why keep rather than delete:** the convention already exists in
`config/model_config.py:KNOWN_UNSUITABLE_METRICS` — metrics stay "discoverable/registered
with an honest reason, never silently deleted." v3 applies that consistently. A poor fit
becomes `not_applicable` or `advisory`. Nothing is removed.

---

## 3. The metric universe

Read from the installed package by walking every submodule of `deepeval.metrics` — not
`dir()`, which misses six.

| | Count |
|---|---|
| Metric classes installed | 53 |
| Single-turn (the usable shape) | 32 |
| Mapped to an evaluation stage | 22 |
| Builders (`GEval`, `DAGMetric`) | 2 |
| Ignored — wrong shape or wrong task | 29 |

The 29: 7 agentic (need tools/plans/MCP), 15 conversational (need `ConversationalTestCase`),
5 multimodal (need images), 1 summarization (needs a source document), 1 arena (needs two or
more contestants).

**The six invisible metrics.** `RagasMetric`, `RAGASAnswerRelevancyMetric`,
`RAGASFaithfulnessMetric`, `RAGASContextualPrecisionMetric`, `RAGASContextualRecallMetric`,
`RAGASContextualEntitiesRecall` are not exported at the package top level, so the current
`dir()`-based discovery has never seen them. They are skipped by decision, not by accident —
but any future discovery code must know they exist.

**Five metrics that falsely look applicable.** `TaskCompletionMetric`,
`PlanAdherenceMetric`, `PlanQualityMetric`, `StepEfficiencyMetric` and
`SummarizationMetric` declare only `input` + `actual_output` as required params. A
field-presence check waves them through even though they measure agent or summarization
behaviour. `PlanQualityMetric` passed all six rows in a real run. This is the concrete
argument for a curated registry over auto-discovery: `dir()` plus a field check cannot tell
"needs a plan" from "needs an answer."

**No statistical tier exists in DeepEval.** Only `ExactMatchMetric` and
`PatternMatchMetric` avoid an LLM, and both are exact/regex. The `statistical` tier that
`metric_tiering/tier_classifier.py` reserves is empty by its own admission. Filling it
requires `sentence-transformers` (semantic similarity), optionally `rouge-score` /
`sacrebleu`. Semantic similarity is the single highest-value addition in this document: it
is the only correctness signal available with **no LLM in the loop**.

---

## 4. Evaluation pipeline

### 4.1 End-to-end flow

Two things in this picture are not in v2: the profile is **derived from the spreadsheet** rather
than declared in it, and the metrics split into **two groups that are gated differently** — one
reads the answer, one reads only the data. There are also two feedback loops, and they return to
different places.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ EXCEL — one sheet, no type column                                            │
│ ID · Input · Context (optional) · Ground_Truth (optional)                    │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        ▼
                     ┌──────────────────────────────────────┐
                     │ PROFILE DETECTION (§5)               │  deterministic,
                     │ has_context · has_reference          │  no model call,
                     │ expects_abstention · is_numeric      │  cached per row hash
                     │ context_role: adversarial|supporting │
                     └──────────────────┬───────────────────┘
                                        ▼
                     ┌──────────────────────────────────────┐
                     │ PROMPT BUILDER (§8)                  │
                     │ instruction layer  ◄── the only      │
                     │ + Context when present    improvisable│
                     │ + Input (never edited)    surface    │
                     └──────────────────┬───────────────────┘
                                        ▼
                     ┌──────────────────────────────────────┐
                     │ GENERATOR   ollama / phi3            │  the model under test
                     └──────────────────┬───────────────────┘
                                        ▼
                     ┌──────────────────────────────────────┐
                     │ STAGE 0  validation            GATE  │──► STOP
                     └──────────────────┬───────────────────┘    data or infra fault,
                                        │                        
                    ┌───────────────────┴────────────────────┐
                    ▼                                        ▼
    ┌───────────────────────────────┐      ┌──────────────────────────────────┐
    │ ANSWER-SIDE        gated      │      │ DATASET-SIDE    never gated      │
    │                               │      │                                  │
    │ 1 deterministic        GATE   │      │ ContextualRelevancy              │
    │ 2 relevance            GATE   │      │ ContextualPrecision              │
    │ 3 grounding      conditional  │      │ ContextualRecall                 │
    │ 4 reference      conditional  │      │                                  │
    │ 5 safety         never gated  │      │ read Context + Ground truth,     │
    │                               │      │ never the answer — so nothing    │
    │ each depends on the answer    │      │ the model did can suppress them  │
    │ being gradeable at all        │      │                                  │
    └───────────────┬───────────────┘      └─────────────────┬────────────────┘
                    │   JUDGE  anthropic / claude-sonnet-4-6 │
                    │   independent of the model under test  │
                    └────────────────────┬───────────────────┘
                                         ▼
                     ┌──────────────────────────────────────┐
                     │ AGGREGATOR                           │
                     │ score + tier + gating_status,         │
                     │ kept separate, never blended          │
                     └──────────────────┬───────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
┌───────────────────┐      ┌─────────────────────────┐   ┌────────────────────────┐
│ SCORECARDS (§8)   │      │ SUGGESTIONS (§8)        │   │ HUMAN REVIEW GATE      │
│ per prompt        │      │ target: prompt·context· │   │ 5% sampled +           │
│ per metric        │      │ golden·judge·infra      │   │ human_required metrics │
│ split by tier     │      │ quotes the row's cells, │   │                        │
│ + blocking verdict│      │ not metric names        │   │                        │
└───────────────────┘      └───────────┬─────────────┘   └───────────┬────────────┘
                                       ▼                             ▼
                           human edits the Excel          human verdict recorded
                                       │                             │
                                       ▼                             ▼
                        ┌──────────────────────────┐    ┌────────────────────────┐
                        │ LOOP 1  re-run changed   │    │ LOOP 2  golden set     │
                        │ rows only (row hash)     │    │ → calibration agreement│
                        │ → before/after, with     │    │ → confidence label     │
                        │   capability vs assisted │    │ → feeds gating_status  │
                        └──────────────────────────┘    └────────────────────────┘
                                 fixes the data                 earns the trust
```

**The two loops do different jobs and must not be confused.** Loop 1 changes the *test* — a
human edits a prompt, a context or a ground truth and re-runs the affected rows. Loop 2 changes
how much the framework *trusts its own judge* — a human verdict enters the golden set,
calibration recomputes agreement, and that agreement decides whether a judge metric is `trusted`
or `advisory`. Loop 1 without Loop 2 produces a suite that iterates quickly on an unverified
ruler.

### 4.2 The gated stage ladder

Zooming into the answer-side branch. Three stages are **gates** that halt the run; two are
**conditional** on data availability; one is **never gated**.

```
                 Prompt + optional Context + optional Ground truth
                                      │
                      ┌───────────────▼───────────────┐
                      │ STAGE 0   Generation    GATE  │──fail──► STOP: no metric runs
                      │ is there a question?          │          (data or infra fault)
                      │ did the model answer?         │
                      └───────────────┬───────────────┘
                      ┌───────────────▼───────────────┐
                      │ STAGE 1   Deterministic GATE  │──pass──► skip Stage 2 ONLY
                      │ no LLM · free                 │          (a correct refusal;
                      └───────────────┬───────────────┘           see §4.3)
                      ┌───────────────▼───────────────┐
                      │ STAGE 2   Relevance     GATE  │──fail──► STOP quality: "off topic"
                      │ does it address the question? │              │
                      └───────────────┬───────────────┘              │
                      ┌───────────────▼───────────────┐              │
                      │ STAGE 3   Grounding  CONDITIONAL             │
                      │ skipped unless Context is in the prompt      │
                      └───────────────┬───────────────┘              │
                      ┌───────────────▼───────────────┐              │
                      │ STAGE 4   Reference  CONDITIONAL             │
                      │ skipped unless a Ground truth exists         │
                      └───────────────┬───────────────┘              │
                      ┌───────────────▼───────────────┐              │
                      │ STAGE 5   Safety   NEVER GATED│◄─────────────┘
                      │ runs on any response that exists             │
                      └───────────────┬───────────────┘
                                      ▼
                           Scorecard · suggestions
```

**Why this order:** cheapest and most decisive first, so a free check can settle a row before
any model call is spent on it. Each stage's result determines whether the next stage's
question is still meaningful.

**Why Stage 5 sits outside the chain:** a refusal can still be toxic and an off-topic answer
can still leak PII. Safety must run even when every quality stage was skipped. Note the
bypass arrow: a Stage 2 stop still reaches Stage 5.

### 4.3 What a gate may and may not suppress

Gates control *flow*. They do not control *applicability*. Conflating the two is an easy and
costly mistake, so the rule is explicit:

> **A metric may only be suppressed by a condition that affects its own inputs.**

Two consequences that a naive stage-skip gets wrong:

**A refusal does not mean "skip the rest of the row."** A refusal on an `abstention_expected`
row is correct behaviour; the same refusal on a `factual_reference` row is the model **dodging
an answerable question**. The second case is only visible if Stage 4 still runs and shows the
answer failing against a real Ground truth. Suppressing Stage 4 on a refusal would hide
exactly the finding §7 relies on.

**`ContextualRelevancy`, `ContextualPrecision` and `ContextualRecall` never read the answer.**
Nothing on the answer side — a refusal, an off-topic response, a failed relevance gate — has
any bearing on their inputs, so nothing on the answer side may suppress them.

| Condition | Suppresses | Why |
|---|---|---|
| refusal detected **and** profile is `abstention_expected` | Stage 2 relevance **only** | a refusal legitimately scores low relevance; that is a metric artifact, not a finding |
| Ground truth is a behavioural description, not a comparable reference | `AnswerCorrectness`, `semantic_similarity`, `ROUGE`/`BLEU`, `ExactMatchMetric` | there is no string to compare against |
| Context not in the prompt | the answer-vs-context metrics (`Hallucination`, `Faithfulness`) | cannot grade against evidence the model never saw |
| no Ground truth | the reference metrics | nothing to compare against |
| **any answer-side condition** | **never** `ContextualRelevancy` / `ContextualPrecision` / `ContextualRecall` | they do not read the answer |

So the pipeline is better understood as **two orthogonal groups**, not one chain:

- **Answer-side metrics** — gated, because each depends on the answer being gradeable at all.
- **Dataset-side metrics** — outside the chain, for the same structural reason as safety but a
  different purpose: they measure the spreadsheet, not the response. They run whenever their
  own inputs (Context, Ground truth) are present, regardless of what the model did.

Note also that an `abstention_expected` row **still benefits from Stage 3** when a Context is
present: a refusal contradicts a Context that asserts the fabrication, which under the §4
inversion rule is a high resistance score — a genuine, wanted measurement. Skipping Stage 3
would discard it.

### Stage 0 — Generation (gate)

Two checks, one either side of the model call.

- **`input_check`** — *does not exist yet; see §11.* Is the `Input` cell populated? Fail fast
  and spend no model call on an empty row.
- **`response_validator`** — already implemented. Catches `timeout`, `generation_error`,
  `empty_response`. A failure here is an **infrastructure fault, never a quality finding**.

### Stage 1 — Deterministic (gate, no LLM, free)

| Check | Pass means | Fail means |
|---|---|---|
| `refusal_check` | the model declined or hedged | it answered confidently |
| `numeric_match` | extracted figures are equal | they differ |
| `PatternMatchMetric` | required shape present | absent |
| `ExactMatchMetric` | strings identical | they differ |

**`refusal_check` flips meaning with the row.** On an `abstention_expected` row a refusal is
the *correct* answer and ends the row. On a `factual_reference` row the same refusal means the
model dodged a question it should have answered. Same signal, opposite verdict — resolvable
only against the row's profile.

**A `numeric_match` failure is conclusive.** 364 is not 366. Mark correctness failed and skip
the judge-based correctness check: it costs a model call to produce a weaker version of an
answer already in hand.

**`ExactMatchMetric` must be narrowed, not removed.** It is the only `deterministic`-tier
metric, so it is auto-`trusted` and therefore **single-handedly decides pass/fail** — which is
why all six rows failed in a real run despite correct answers. Restrict `is_applicable` to
rows whose Ground truth is a short canonical value (a number, a name, one token). It then does
real work on numeric rows and reports `not_applicable` with a reason on prose.

### Stage 2 — Relevance (gate)

`AnswerRelevancyMetric` — reads question + answer only; it never sees the Ground truth.

A failure stops Stages 3 and 4: if the answer is not about the question, accuracy scores are
measuring nothing, and "off topic" is the whole finding. **Caveat:** a refusal often scores low
here even when refusing is correct, which is why `abstention_expected` rows bypass this stage
after Stage 1 passes.

### Stage 3 — Grounding (conditional on Context being in the prompt)

Only valid if the Context was actually **sent to the model**. Grading an answer against
evidence it never saw is not a measurement.

| Metric | Reads | Note |
|---|---|---|
| `ContextualRelevancyMetric` | question + Context | **never reads the answer** — a data check |
| `HallucinationMetric` | answer + `context` | keep, **invert the reading** |
| `FaithfulnessMetric` | answer + `retrieval_context` | keep, **invert the reading** |

**The inversion rule.** These metrics score how much the answer *contradicts* the Context. On
a deliberately false Context:

| Model answers | Score | Means |
|---|---|---|
| correct ("Canberra") | **1.00** | contradicted the false Context → **resisted the trap** |
| wrong ("Sydney") | **0.00** | agreed with the false Context → **fell for it** |

The score was never broken — it is already a context-resistance meter with the sign flipped.
When `context_role == adversarial`, pass on a *high* score instead of a low one. This removes
the need for a separate custom resistance rubric.

Two requirements: read `metric.score` and apply the project's own rule rather than trusting
`metric.success` (see §9), and know `context_role` — which §5 derives automatically.

### Stage 4 — Reference (conditional on a Ground truth)

Two differently-biased opinions on the same question, deliberately.

| Metric | Tier | Note |
|---|---|---|
| `semantic_similarity` | statistical | **no LLM** — the trust anchor |
| `AnswerCorrectness` (GEval) | judge | richest signal, least trustworthy |
| `ContextualPrecisionMetric` | judge | **never reads the answer** — a data check |
| `ContextualRecallMetric` | judge | **never reads the answer** — a data check |
| `ROUGE` / `BLEU` | statistical | advisory only, never gating |
| `JsonCorrectnessMetric` | judge | activates itself when a schema exists |

`semantic_similarity` solves what `ExactMatchMetric` could not: *"The capital of Australia is
Canberra, chosen as a compromise…"* scores high against *"Canberra is the capital of
Australia."*

Keep `ROUGE`/`BLEU` advisory. A correct answer in different words scores badly — the same
failure mode that makes exact match useless on prose. Its only real job is as a tiebreaker.

`JsonCorrectnessMetric` needs no decision: `GenericMetricModule.is_applicable` already
reports `missing required configuration: expected_schema` and never executes. Supply a schema
and it activates.

### Stage 5 — Compliance and safety (never gated)

`BiasMetric`, `ToxicityMetric`, `PIILeakageMetric`, `PromptAlignmentMetric`, plus
`MisuseMetric`, `NonAdviceMetric`, `RoleViolationMetric` once configured.

**`PromptAlignmentMetric` is the metric the prompt improviser turns on.** It separates two
failures that otherwise look identical:

| Result | Meaning | Action |
|---|---|---|
| alignment **fail** | the model never followed the instruction | **edit the prompt** |
| alignment **pass** + correctness **fail** | it obeyed and still got it wrong | editing the prompt will not help |

**`PIILeakageMetric` is the natural first member of `HUMAN_REQUIRED_METRICS`**, which is empty
today. A wrinkle specific to this dataset: the `fabricated_entities` Contexts invent people
("Emma Richardson is CEO of SkyNova AI"). If the model repeats that name, is it leaking PII or
repeating a fiction? Only a human can judge that.

---

## 5. Profile detection from the spreadsheet -- No need implement for now

**No type column, no per-category sheets.** Today `utils/dataset_loader.py` sets
`category = sheet_name`, which means the category is maintained as five separate Excel sheets.
v3 derives an **evaluation profile** from the row's own content, so the dataset can collapse to
a single sheet.

| Signal | Rule |
|---|---|
| `has_context` / `has_reference` | field non-empty (`_has_value`, already implemented) |
| `expects_abstention` | Ground truth matches a non-existence vocabulary — "does not exist", "fabricated", "fictional", "no such", "misattributed" |
| `is_numeric` | Ground truth contains a numeral |
| `context_role` | `adversarial` when Context and Ground truth disagree, else `supporting` |

| Profile | Derived from | Replaces sheet |
|---|---|---|
| `abstention_expected` | `expects_abstention` | fabricated_entities, citation_hallucination |
| `numeric_reference` | `is_numeric` | numerical_hallucination |
| `factual_reference` | has a declarative reference | factual_errors, context_conflict |
| `no_reference` | no Ground truth | open-ended rows |

`factual_errors` and `context_conflict` are structurally identical and take the same metric
set, so the framework does not need to distinguish them — it needs the *evaluation intent*,
which has fewer distinct values than the five categories did.

**`context_role` requires no new column and no judge call at run time.** A simultaneous
failure of `ContextualPrecisionMetric` and `ContextualRecallMetric` *is* the adversarial
signal — neither metric reads the answer, so a double failure says only that Context and
Ground truth describe different worlds. Compute it once at dataset load, cache it keyed by row
hash.

### Row identity

Use the Excel **`ID` column**, which `dataset_loader` currently does not read. Today identity
is `f"{category}:{category_position}"`, so consolidating sheets or reordering rows would
orphan all prompt-version history. `ID` is stable across both.

---

## 6. The question-only case

A row with no Context and no Ground truth cannot be checked for factual accuracy. **No
DeepEval metric verifies a fact without a reference.** A model can pass all eight
question-plus-answer metrics while fabricating completely.

What is measurable instead is **calibration — does the model know what it does not know?**
That is arguably the property that matters most in production, where there is no ground truth
either. Report it under a heading that cannot be misread as accuracy.

Three reference-free techniques, none of which is a DeepEval metric:

1. **Self-consistency across repeated runs.** The strongest signal: ask the same question
   several times; a model that knows the answer repeats it, a confabulating model varies.
   **Blocked today** — `GENERATOR_TEMPERATURE = 0` makes every run byte-identical, so variance
   is always zero. Needs a separate sampling pass at temperature > 0.
2. **Hedging detection.** Already implemented: `classification/result_classifier.py`
   classifies such rows `UNVERIFIABLE` and returns `appropriately_hedged` /
   `unsupported_claim_likely` / `not_determined`.
3. **Specificity without hedging.** A precise answer — names, dates, figures — to a question
   the model should not know, carrying no qualification, is the classic confabulation
   fingerprint. Detectable deterministically.

---

## 7. Reading metrics against each other

Three combinations carry more information than either metric alone. This is what a per-metric
pass rate cannot express, and why the suggestion engine reads *patterns* rather than scores.

| Combination | Conclusion |
|---|---|
| `semantic_similarity` **pass** + `AnswerCorrectness` **fail** | **Suspect the judge.** The embedding says the meaning matches; only the judge disagrees. Log as calibration evidence, not a model defect. |
| `ContextualPrecision` **fail** + `ContextualRecall` **fail** | **Context and Ground truth disagree.** Says nothing about the model. Expected on an adversarial row; a data bug anywhere else. This is the `context_role` detector. |
| `refusal_check` **pass** + profile `factual_reference` | **The model dodged** an answerable question. The identical refusal on an abstention row would be a pass. |

---

## 8. Scoring, reporting and the improvement loop

### Two scorecards

**Per prompt** — applicable · passed · failed · errored · **pass %** · broken out by tier.

**Per metric** — applicable · `skipped_by_gate` · executed · errored · **pass %** ·
**coverage**.

Report `coverage` beside `pass %`, because the rate alone misleads: a metric reading 80% (4/5)
hides that a sixth row errored out. Segment both by `gating_status`, so an `advisory` 100% is
never read as a verified 100%.

**A bare "% of metrics passed" is not a verdict.** Metrics are neither equally important nor
independent — a prompt can pass 9 of 11 while failing the only one that mattered. Always print
the percentage beside a **blocking verdict**: did the metrics that matter *for this profile*
pass? Never average across tiers.

### Suggestion engine

Deterministic diagnosis first, LLM wording second. Extend the existing
`root_cause_analysis/analyzer.py` — which already emits `root_cause`, `confidence` and
`recommended_action` — with a `target` field: `prompt` | `context` | `golden_truth` | `judge` |
`infra`.

| Failure signature | Target | Suggestion |
|---|---|---|
| Relevancy fail + correctness fail | prompt | tighten the instruction |
| Correctness fail + similarity high | judge | calibration evidence, not a defect |
| `ContextualRelevancy` low | context | the Context does not match the question |
| `ContextualPrecision`/`Recall` low | context + golden | the pair contradict each other |
| `PromptAlignment` low | prompt | the instruction is unclear or too long |
| Safety fail | human review | always, regardless of the rest of the row |
| All judge metrics error | infra | a judge fault, not a prompt problem |

**Write suggestions for a human, not for a metric reader.** Quote the row's actual cells rather
than naming metrics and scores; state what to do; put metric names and numbers behind a
"which checks said this" disclosure.

### Human-in-the-loop loop

```
RUN → report + suggestions → human reviews → human edits the Excel
    → human confirms re-run → before/after comparison
```

Excel remains the single source of truth. Two mechanics make it work:

- **Change detection.** Hash the `(Input, Context, Ground_Truth)` triple per row `ID` so a
  re-run can touch only edited rows. `prompt_versioning/prompt_version_store.py` already
  hashes prompt text; extend it to the triple.
- **Attribution.** `run_comparison/fair_comparison.py` already classifies a comparison as
  `capability` / `assisted` / `mixed` and warns on `mixed`. Surface that as guidance: **change
  one thing per iteration**, or the improvement cannot be attributed.

### Prompt improviser

`utils/prompt_builder.py` returns `test_case["input"]` verbatim — there is no template layer,
so there is nothing to improve except the question itself, and rewriting the question mutates
the test.

**Introduce an instruction layer and make that the improvisable surface. Improve the
instruction; never touch `Input`.** The prompt becomes:

- no Context → question only (`closed_book`)
- Context present → instruction + Context block + question (`open_book`)

For this dataset the instruction should read roughly *"use the context if helpful, but
prioritise factual accuracy"* — a **resistance** framing. A strict *"answer using only the
context"* framing would make the false Context the correct answer and contradict the Ground
truth column.

`record_prompt_usage` already carries a `prompt_assisted` flag for this purpose.

**The guardrail that matters most: the improviser must never see `Ground_Truth` or `Context`.**
Given them, it will encode the answer into the instruction, every metric will turn green, and
nothing will have been measured. Cap iterations at two or three and hold out a validation slice
to detect overfitting to the eval set.

`ArenaGEval` is worth revisiting here specifically. It ranks two or more answers instead of
scoring one, which suits before/after comparison: with a weak judge, a `0.70 → 0.75` delta is
indistinguishable from noise, whereas a forced pairwise preference is far more robust.

### Interactive console

A single-prompt UI (Streamlit — Python-native, imports these modules directly) alongside the
Excel suite, not replacing it. The UI is for authoring and debugging one prompt with instant
feedback; Excel is for the regression suite and the run history that `benchmark/` and
`run_comparison/` depend on. A single-prompt UI structurally cannot do regression detection.

Both call the same `orchestrator.run_test_case`. The UI gets a "save to regression set" action
that appends an authored prompt to the Excel with a fresh `ID`.

`metric_manager.get_applicability_report()` already returns per-metric applicability with
reasons and executes nothing — it is exactly the data source for a free "what will run" preview
before paying for a run. It currently has no callers.

Every report and every console view must state **which models produced and graded the run**.
A score without that provenance is not interpretable.

---

## 9. Models, judge independence, and multi-model comparison

### 9.1 What is configured today

Read from `config/model_config.py`. **Any score is uninterpretable without this table, so it
belongs on every report and every console view.**

| Role | Provider | Model | Temperature | Endpoint |
|---|---|---|---|---|
| **Generator** — writes the answer | `ollama` | **`phi3`** | 0 | `http://localhost:11434` (passed explicitly) |
| **Judge** — scores the answer | `ollama` | **`qwen2.5:7b-instruct`** | 0 | `http://localhost:11434` (passed explicitly) |

**CORRECTED from an earlier draft of this section.** An Anthropic judge
(`claude-sonnet-4-6`) was tried and reverted — it worked, but needs a
separate `console.anthropic.com` account and billing, which was declined.
The judge stays on local Ollama, as a genuinely different model from the
generator. **`ANTHROPIC_API_KEY` is NOT required to run this project.**
Everything below that assumed the Anthropic judge has been struck or
corrected; it's kept only where the reasoning still matters if an Anthropic
judge is reconsidered later.

**The judge is independent of the model under test** — same provider,
different model, different architecture and training (`qwen2.5` vs `phi3`).
`get_judge_independence()` compares the model, not just the provider (§9.2),
so this correctly returns `True` even though both roles say "ollama".

This does *not* on its own make judge-based metrics `trusted` — `gating_status`
still depends on calibration agreement (§11, defect 10), and the golden set
started empty. An independent judge makes calibration *worth* doing; it does
not substitute for it.

~~Why a 4.6-generation model rather than a current-generation one...~~
~~`claude-sonnet-4-6` is the configured judge...~~
~~Requires `ANTHROPIC_API_KEY` in the environment...~~
*(struck: this entire sub-thread only applied while the judge was Anthropic.
See `config/model_config.py` for the current reasoning behind `qwen2.5:7b-instruct`
— sized for CPU-only inference on this machine, chosen for a different
architecture/training recipe from `phi3` rather than raw strength, so it
makes independent mistakes rather than sharing phi3's blind spots.)*

### 9.2 Judge independence

**Independence must be measured on the model, not the provider.** `get_judge_independence()`
(`judge_model_resolver.py:35`) previously compared providers only and had no callers. It now
compares models too — so a legitimate `phi3`-answers / `llama3`-judges setup is no longer
reported as dependent:

```python
def get_judge_independence():
    return (get_judge_provider() != get_generator_provider()
            or model_config.JUDGE_MODEL != model_config.GENERATOR_MODEL)
```

**Fixed:** it now has a caller — `reporting/report_generator.py` prints Generator, Judge, and
this independence flag on every Summary sheet and comparison report, so a reader never sees a
score without knowing who produced and who graded it.

**Temperature 0 on the generator is a deliberate trade.** It makes runs reproducible, which is
right for a regression suite — but it also makes the self-consistency technique in §6 measure
nothing, because every repeat is byte-identical. Self-consistency needs its own sampling pass
at temperature > 0; it cannot share the regression run.

### 9.3 The Generator Layer — comparing several models

v2's diagram showed a Generator Layer spanning Qwen / Claude / Gemini / GPT / Phi3. Only
`ollama` is registered today, but **most of the machinery for multi-model comparison already
exists**, and it is better built than the judge path:

| Piece | State |
|---|---|
| `config/generator_resolver.py` — `GENERATOR_PROVIDER_FUNCTIONS` | the extension point; one entry (`ollama`). Adding a provider is a dict entry plus a `generate_response` function |
| `run_comparison/run_store.py` — `capture_run_metadata()` | **already snapshots generator provider + model on every `save_run`**, automatically, so comparison never depends on a human remembering to record it |
| `run_comparison/run_store.py` — `execute_dataset()` | already loops a whole dataset through the orchestrator, so "run this dataset against another model" is: change config → `execute_dataset` → `save_run` |
| `run_comparison/fair_comparison.py` — `classify_comparison_condition()` | already separates a generator change from a prompt change, and **correctly compares model, not just provider** — unlike `get_judge_independence()` |

So what is missing is a provider function per model and a driver that loops over them. The
accounting that makes a comparison trustworthy is done.

**One coupling to fix first.** `comparator._default_match_key` matches rows by
`(category, position)` and `_find_prompt_version` builds `slot_key` the same way. Moving to the
`ID` column (§5) requires updating both, or every cross-run comparison silently mismatches rows
after a sheet is consolidated or reordered.

### 9.4 Fair comparison: three rules

v2 called this the Fair-Comparison Controller and framed it as prompt parity. Two of its three
rules are enforced in code; the third cannot be, and has to be a discipline.

**Rule 1 — identical prompt across models (enforced).** Comparing models on different prompts
measures prompt fit, not capability. `fair_comparison` labels a run where both the generator and
the prompt changed as `mixed` and attaches an explicit warning that the result "cannot cleanly
attribute score changes to either the model or the prompt."

**Rule 2 — the judge must not be one of the models being compared (not yet enforced; the
hard prerequisite).** An LLM judge systematically favours its own output. Comparing `phi3`
against `qwen` while `phi3` holds the scoring pen is not a weak measurement, it is a biased
one, and the bias points in a known direction. Today both roles are `phi3`, so
**multi-model comparison must wait for §9.2.** This is the single reason the Generator Layer
sits late in the build order rather than early: the plumbing is nearly done, the ruler is not
trustworthy yet.

**Rule 3 — never compare models on a prompt that was improved against one of them (cannot be
enforced in code).** This is the trap where the Prompt Improviser (§8) and the Generator Layer
meet. Tune an instruction against `phi3`, then run `phi3` and `qwen` on that same instruction,
and `fair_comparison` will correctly report the prompt as *unchanged between the two runs* and
label the comparison `capability` — a clean result, and a wrong one. `qwen` loses because the
prompt was fitted to its opponent, not because it is less capable.

The code cannot see this, because the tuning happened in an earlier run. The defences are
procedural: mark any prompt version carrying `prompt_assisted: true` with **which model it was
tuned against**, and treat a capability comparison that uses such a prompt as invalid unless
every contestant got an equal tuning budget.

**Instrument choice.** With a weak or uncalibrated judge, prefer `ArenaGEval` — it ranks two
answers against each other rather than scoring each in isolation, so a forced pairwise
preference replaces a `0.70` vs `0.75` delta that is indistinguishable from noise. It is the one
metric in the excluded 21 worth building a path to.

---

## 10. Layers carried over from v2

v2 specified four further layers. They are not equally necessary, and lumping them together
obscures that two are already built and one should be trimmed rather than dropped.

### Result Aggregator — already exists, under another name

v2 asked for an aggregator that "keeps quality, efficiency and confidence-tier separate."
`orchestrator.run_test_case` already returns `validation`, `evaluation`, `classification`,
`root_cause_analysis` and `human_review` as separate keys, and `metric_tiering` attaches `tier`
and `gating_status` alongside each score without touching it. The two-scorecard design in §8
carries that separation into the report. **No new layer required.**

### Human Review Sampling Gate — already built

`human_review/sampling.py` and `human_review/review_gate.py` are implemented and wired into
the orchestrator. `HUMAN_REVIEW_SAMPLE_RATE = 0.05` routes 5% of rows to a human regardless of
score, and any metric in `HUMAN_REQUIRED_METRICS` routes unconditionally.

Two things remain, and both are configuration rather than code:

- `HUMAN_REQUIRED_METRICS` is **empty**. `PIILeakageMetric` is the natural first member (§4,
  Stage 5), and `NonAdviceMetric` follows once configured.
- Reviewed verdicts should be recorded through
  `judge_calibration.calibration_store.record_calibration_judgment`, which closes the loop into
  the golden set. That is the mechanism that fixes defect 10, and it already exists.

### Efficiency Layer — trim it, do not drop it

v2 specified tokens, latency, dollar cost and calls-per-response, designed for comparing paid
APIs.

**The judge change in §9.1 makes the cost dimension real.** It was dismissible while both roles
ran on local Ollama; with an Anthropic judge, every metric execution is now a billed API call.
A full run is roughly `rows × applicable judge metrics` calls — on the current dataset that is
6 × ~10, and it grows multiplicatively with both. Three consequences:

- **Token and cost capture moves into scope.** Capture `usage` per judge call and total it per
  run, so a report says what it cost to produce.
- **Metric pruning now saves money, not just noise.** The curated registry (phase 3) and the
  gating short-circuits stop being purely cosmetic.
- **Stage ordering earns its keep.** Every row settled by a free Stage 1 check, and every
  Stage 2 gate failure that skips Stages 3–4, is now a cost avoided rather than just time saved.

Latency and call count remain in scope for their own reasons. A real run took **35 minutes 38 seconds for six
rows**, which is the single biggest obstacle to the human-in-the-loop iteration in §8 — a loop
nobody will use if each turn costs half an hour. So instrument the two dimensions that
describe that problem:

| Capture | Why |
|---|---|
| wall-clock latency, per metric and per row | shows *which* metrics are expensive, so gating can be aimed at them |
| judge calls per row | 48 attempted metrics is not 48 calls; knowing the real number tells you what pruning actually saves |

Keep them, as v2 insisted, **a separate vector that is never merged into a quality score** — a
fast wrong answer is still wrong. Skip tokens and dollars until a paid judge is in use, at which
point the cost dimension starts meaning something.

### Black-box / white-box confidence tiering — not needed

v2 §3.5 exists for evaluating systems you cannot inspect. Here the prompt, the context, the
model and the endpoint are all under this project's control, so every run is white-box by
construction. A flag whose value never changes is noise. **Revisit only if a third-party or
hosted system is ever evaluated.**

---

## 11. Defects to fix

Found by reading and running the current code. Each is specific and verifiable.

**Status as of 2026-10-04: all 13 fixed.** 12 are closed outright; defect 10's mechanism is
fixed but the golden set itself is still thin (1 real recorded judgment) — that one keeps
growing over time as real rows get reviewed, it was never going to be a single commit.

| # | Location | Defect | Status |
|---|---|---|---|
| 1 | `evaluators/metrics/generic.py:31` | `RETRIEVAL_CONTEXT` hardcoded `False`, so the four context metrics can never apply even though the Context data exists. **One line blocks four metrics.** | ✅ Fixed |
| 2 | `utils/prompt_builder.py:4` | Context is never sent to the generator, yet it is passed to `LLMTestCase(context=…)` and graded. **The model is judged against evidence it never saw.** | ✅ Fixed |
| 3 | `prompt_versioning/prompt_version_store.py:48` | `(text or "").encode()` — a blank Excel cell is `float('nan')`, and NaN is truthy, so the guard is skipped and `.encode()` is called on a float. **A blank `Input` crashes the run** inside `build_prompt`, uncaught. | ✅ Fixed |
| 4 | `utils/dataset_loader.py` | `_has_value()` guards Context and Ground_Truth — both optional — but never `Input`, the one mandatory field. | ✅ Fixed |
| 5 | `config/judge_model_resolver.py:35` | `get_judge_independence()` has **zero callers**. The one function that reports whether the judge is independent is dead code. | ✅ Fixed — now called from `reporting/report_generator.py` |
| 6 | `config/judge_model_resolver.py:29` | It compares *provider*, not model — so `phi3` answering and `llama3` judging would still report as not independent. Independence is a property of the model. | ✅ Fixed |
| 7 | `config/judge_model_resolver.py:29` | The judge is built without an endpoint while the generator receives `GENERATOR_ENDPOINT`. If Ollama moves off the default host, generation works and judging breaks — with errors the tiering downgrades to `not_applicable`. | ✅ Fixed |
| 8 | `evaluators/metric_manager.py:121` | `"passed" if metric.success else "failed"` hardcodes DeepEval's own threshold rule, leaving no hook for the §4 inversion. | ✅ Fixed |
| 9 | `classification/result_classifier.py:72` | `unsupported_claim_likely` is triggered merely by "any metric executed" — and `BiasMetric` passing says nothing about whether a claim was supported. The corroboration is not evidence. | ✅ Fixed — now purely the hedging heuristic, no false corroboration |
| 10 | `judge_calibration/data/` | `calibration_judgments.json` is `[]` and `calibration_scores.json` is `{}`. The golden set is empty, so **every judge metric resolves `unverified` → `advisory`** and is excluded from gating. All of them. | 🟡 Mechanism fixed, data thin. Store migrated to `calibration_judgments.xlsx` for human editing; 1 real judgment recorded (`HallucinationMetric`/`Prompt`, agreement 1.0 → `trusted` for that category). Also found and fixed: `calibration_runner.py` was comparing against DeepEval's raw `metric.success`, not the inverted reading — would have mislabeled every adversarial-row judgment as a disagreement. |
| 11 | `datasets/…with_types.xlsx`, `context_conflict` sheet | The Context column's text leaked into `Ground_Truth`, prefix included: `"Context: Alexander Fleming discovered penicillin."` Those rows can never match anything. | ✅ Fixed — both rows' `Ground_Truth` cells corrected in the workbook |
| 12 | `datasets/…with_types.xlsx`, `fabricated_entities` sheet | `Ground_Truth` holds behavioural descriptions ("SkyNova AI is a fabricated company"), not reference answers — yet a non-empty value both makes reference metrics applicable and classifies the row `VERIFIED`. | ✅ Fixed — `classify()` now excludes `expects_abstention` rows from `VERIFIED`; reference-metric applicability was already correctly excluded via `metric_registry` |
| 13 | `config/model_config.py` | Generator and judge are both `ollama`/`phi3`. **The model grades itself.** `AnthropicModel` is already registered in `JUDGE_PROVIDER_CLASSES`, so an independent judge is a two-line config change plus an API key. | ✅ Fixed — judge is `ollama`/`qwen2.5:7b-instruct`, a genuinely different model from the `phi3` generator, no API key needed |

---

## 12. Build order

| Phase | Work | Why here | Status |
|---|---|---|---|
| **1** | Read the `ID` column; validate `Input`; fix the NaN guard | Stops a crash; establishes stable row identity that everything else keys off. Defects 3, 4. | ✅ Done |
| **2** | Send Context to the generator; populate `retrieval_context`; add the instruction layer | One line unblocks four metrics; the instruction layer is a prerequisite for the improviser. Defects 1, 2. | ✅ Done |
| **3** | Curated metric registry + profile-driven selection; narrow `ExactMatchMetric`; add the score-interpretation hook | Makes results readable and lets the inversion rule work. Defect 8. | ✅ Done |
| **4** | Separate the judge from the generator; wire `get_judge_independence()`; pass the judge endpoint | Nothing above is trustworthy while the model grades itself. Defects 5, 6, 7, 13. | ✅ Done |
| **5** | Add `sentence-transformers` semantic similarity | Fills the empty statistical tier — the only trustworthy correctness signal that needs no judge. | ✅ Done — `evaluators/metrics/semantic_similarity.py`. Threshold tuned to 0.8 after testing real sentence pairs (an initial 0.6 guess let a wrong-entity swap pass) |
| **6** | Two scorecards with tier breakdown and coverage; per-metric latency and judge-call count | The reporting that makes the rest legible — and latency is what tells you why a run takes 35 minutes. | ✅ Done (scorecards); latency/call-count instrumentation not yet added |
| **7** | Populate `HUMAN_REQUIRED_METRICS`; seed the calibration golden set from reviewed rows | The review gate is already built and already sampling — it just has nothing designated and nowhere to deposit verdicts. Defect 10. | ✅ Done — `PIILeakageMetric` added to `HUMAN_REQUIRED_METRICS`; golden set has 1 seeded row (see defect 10) |
| **8** | Extend `analyzer.py` with `target`; plain-language suggestions | The advisory output. | ⬜ Not started |
| **9** | Row-hash change detection; `--rerun-changed` | Makes the loop fast enough to iterate. | ⬜ Not started — deferred, convenience only |
| **10** | Streamlit console | Needs the phases above to have something worth showing. | ⏭️ **Skipped by decision** — adds a new UI/tool with no testing benefit over the existing Excel reports, for a QA-driven, pytest-based workflow |
| **11** | Prompt improviser on the instruction layer | An improviser tuned against an uncalibrated or inverted metric optimises toward hallucination. | ⏭️ **Skipped by decision** — real risk (could encode the answer into the prompt) for a feature not currently needed |
| **12** | Generator Layer: register further providers; fix `comparator`'s match key to use `ID`; drive multi-model runs | Last, and only because of §9.4 Rule 2 — the metadata and condition-classification are already built, but comparing models while one of them is the judge produces a biased result, not a weak one. | ⬜ Not started — only one model is tested today, no current need |

Dataset repair (defects 11, 12) is independent of all of this and can happen at any time. **Done.**

---

## 13. Deliberately out of scope

Recorded so the absence reads as a decision rather than an omission.

- **Conversational / multi-turn evaluation** — 15 metrics. Needs a different test-case class
  and a different dataset shape.
- **Multimodal** — 5 metrics. Nothing in the pipeline produces or reads an image.
- **Agentic** — 7 metrics. There is no agent, no tool call, no plan.
- **RAGAS** — 6 metrics. Redundant with the native context metrics, judge-based so no gain in
  trust, and an awkward dependency.
- **DeepEval 4.x** — pinned at 3.9.5; see §2.
- **Token and dollar-cost instrumentation** — meaningless against a free local model. Latency
  and call count are *in* scope; see §10.
- **Black-box / white-box confidence tiering** — every run here is white-box by construction;
  see §10.
