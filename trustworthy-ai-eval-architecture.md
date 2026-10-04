# Trustworthy AI Evaluation Architecture (v2)

A revision of the original layered design, built around one premise: **LLM-as-judge is a calibrated, bounded signal — never the ground truth.** Every new component below exists to prevent the framework from becoming a system that trusts AI to grade AI with no external check.

---

## 1. What Changes vs. v1

| v1 assumption | Problem | v2 change |
|---|---|---|
| DeepEval metric score = quality | No validation that the judge is right | New **Judge Calibration Layer** against a human-labeled golden set |
| One "score" per model | Conflates quality with cost/speed | New **Efficiency Layer**, reported as a separate vector, never merged into quality |
| Improver rewrites prompt freely | Confounds capability with prompt engineering | **Fair-Comparison Controller** enforces prompt-parity or explicitly labels the comparison as "prompt-assisted" |
| All systems evaluated identically | Black-box systems can't support process-based metrics | **Confidence-Tiered Evaluation** — reports declare white-box vs black-box mode and suppress metrics that need context you don't have |
| Metrics treated as interchangeable | Some findings need deterministic proof, not LLM opinion | **Metric Tiering** — deterministic > statistical > calibrated-LLM-judge > human-required, enforced in config, not convention |

---

## 2. High-Level Architecture

```
                         Input Dataset (+ Golden/Reference Set)
                                     │
                                     ▼
                              Prompt Builder
                          (versioned, diffable)
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                                ▼
             Generator Layer                  Fair-Comparison Controller
        (Qwen/Claude/Gemini/GPT/Phi3)         (enforces prompt parity OR
                     │                          labels run as "assisted")
                     ▼
             Generated Responses
                     │
        ┌────────────┼─────────────────────────────┐
        ▼            ▼                              ▼
 Deterministic   Statistical/                 LLM-Judge Layer
   Checks        Reference Metrics        (DeepEval + GEval + AI Judge)
 (             (BLEU/ROUGE/               │
                  embedding sim,            ▼
  golden-set       trained classifiers)   Judge Calibration Layer
  exact-match)                            (validated against golden
        │            │                     set; per-domain agreement
        │            │                     score attached to every
        │            │                     LLM-judge result)
        │            │                              │
        └────────────┴──────────────┬───────────────┘
                                     ▼
                          Efficiency Layer (parallel, not merged)
                    tokens / latency / $ cost / calls-per-response
                                     │
                                     ▼
                          Result Aggregator
              (keeps quality, efficiency, confidence-tier separate)
                                     │
                                     ▼
                    Human Review Sampling Gate
          (fixed % routed to humans regardless of automated scores;
           refreshes the golden set over time)
                                     │
                                     ▼
                        Reports / Dashboard
        (quality-vs-cost scorecards, judge-confidence labels,
         black-box vs white-box mode flags)
```

---

## 3. New/Changed Components

### 3.1 Judge Calibration Layer

**Purpose:** stop treating LLM-judge output as truth; measure how much to trust it.

- Maintain a **golden set**: prompts with human-verified correct answers/labels, stratified by domain and task type.
- Periodically (e.g. weekly, or on every DeepEval version bump) run every active LLM-judge metric against the golden set and compute agreement with human labels (Cohen's kappa or correlation, depending on metric type).
- Store a **per-metric, per-domain confidence score**. Every `EvaluationResult` that used an LLM-judge metric carries this confidence score alongside the raw metric score.
- **Enforcement, not just visibility:** if a metric's calibrated agreement falls below a configured threshold for a domain, the framework marks that metric's results in that domain as `advisory` (shown but excluded from pass/fail gating) rather than silently trusting it. A metric that hasn't been calibrated yet is `unverified` and treated the same way.
- This layer is what answers "who validates the judge" with an actual mechanism instead of an assumption.

### 3.2 Fair-Comparison Controller (Improvement Pipeline)

**Purpose:** stop "improvement" scores from secretly measuring prompt engineering.

- Every prompt sent to an Improver is versioned and diffed against the original: token delta, added instructions/examples/constraints are logged as structured metadata, not discarded.
- The framework runs (and reports) **two distinct conditions**, never blended:
  - **Capability condition** — same prompt, different model/version, to isolate model capability.
  - **Assisted condition** — improver-modified prompt, explicitly labeled `prompt_assisted: true` with the token/structural delta attached, so a reader can never mistake it for a same-prompt capability gain.
- Trend/comparison reports refuse to plot capability and assisted results on the same axis without a visual/label distinguishing them.

### 3.3 Efficiency Layer

**Purpose:** make cost/speed a first-class, separately reported dimension.

- Captures: total tokens (prompt + completion), wall-clock latency, number of underlying model/API calls (relevant for agentic or RAG systems that make several hidden calls), and estimated dollar cost using per-model pricing config.
- Never merged into a quality score. Reports present quality and efficiency as a **paired vector** — the recommended visualization is a quality-vs-cost scatter plot per model, so a reader can see Pareto dominance (e.g., Model A matches Model B's faithfulness at a fraction of the tokens) rather than a misleading tie on a blended leaderboard.
- Config lets each use case define its own gating logic (e.g., "latency > 2s = hard fail" for a chat product; "cost is a factor, not a gate" for offline batch summarization) — the framework computes the raw numbers, the product owner decides the trade-off policy.

### 3.4 Metric Tiering (enforced in config, not convention)

Every metric registered in the framework declares a tier:

```
tier: deterministic | statistical | llm_judge_calibrated | human_required
```

- **Deterministic** (schema validity, format compliance, exact-match against golden set, PII/format regex, retrieval hit-rate against known-relevant docs): no model dependency at all — this is your bedrock signal and should carry the most weight in any pass/fail gate.
- **Statistical** (BLEU/ROUGE/BERTScore, embedding similarity, trained non-generative classifiers for toxicity/PII): no generative-LLM dependency, but known to be surface-level — useful as a second, differently-biased signal, not a sole gate.
- **LLM-judge, calibrated** (DeepEval faithfulness/relevancy/hallucination, GEval rubrics): usable in gating **only** once calibration confidence clears the configured threshold for that domain (enforced by the Judge Calibration Layer above); otherwise `advisory`.
- **Human-required**: safety-critical categories (PII leakage in regulated domains, medical/legal/financial claims, policy compliance, brand-voice/ethical nuance, anything outside the golden set's coverage) are configured so no LLM-judge score can auto-pass them — they route to the Human Review Sampling Gate regardless of automated score.

This tiering is what directly answers "which metrics should never rely entirely on another LLM" — it's a config-level guarantee, not a documentation note that can be ignored under deadline pressure.

### 3.5 Confidence-Tiered Evaluation (Black-Box Systems)

**Purpose:** stop reporting black-box systems with the same false confidence as systems you can fully inspect.

- Every evaluated system is tagged `white_box` or `black_box` based on what visibility you actually have (prompt/context/intermediate steps vs. input+output only).
- For `black_box` systems, the framework **suppresses or down-weights** metrics that require intermediate visibility you don't have (e.g., contextual precision/recall need retrieval context you can't see) and instead emphasizes what *is* valid outcome-based testing: input/output consistency across repeated runs (variance as a signal of instability), adversarial/boundary probing, externally observed latency and cost, and scoring against your own golden set.
- Reports for black-box systems carry an explicit **confidence label** ("outcome-based evaluation — internal process not observable") so a reader never mistakes reduced-visibility results for the same rigor as a white-box evaluation.

### 3.6 Human Review Sampling Gate

- A fixed percentage of evaluated items (configurable, e.g. 5–10%, higher for safety-critical domains) is routed to human reviewers **regardless of what automated scores say** — this catches judge drift that a static calibration snapshot would miss, and continuously refreshes the golden set with new labeled examples.
- Human-required tier items (3.4) always route here; other items are sampled probabilistically.

---

## 4. Reporting Changes

- **Scorecards are multi-dimensional by construction**: quality (broken down by metric tier), efficiency, and confidence/calibration status are always three separate sections — never collapsed into one leaderboard number.
- **Comparison reports** show a Pareto-style quality-vs-cost view alongside the metric-by-model matrix from v1, so "same score, very different cost" (your Concern 3 example) is visually obvious rather than hidden by a tie.
- **Every LLM-judge score** in any report is annotated with its calibration confidence (`high` / `advisory` / `unverified`) — a reader should never see a bare number without knowing how much to trust it.
- **Improvement/trend reports** never plot capability-condition and prompt-assisted-condition results on the same line without a visual distinction (per §3.2).

---

## 5. What Stays the Same From v1

The layered structure, plugin/registry pattern for generators and metrics, and the core data contracts (`PromptItem`, `GenerationResult`, `EvaluationResult`) are still the right foundation — the problem in v1 wasn't the plumbing, it was that the plumbing had no mechanism to question the judge, separate cost from quality, or admit reduced confidence on black-box systems. v2 adds exactly those mechanisms as new layers rather than replacing the architecture wholesale.

---

## 6. Practical Starting Point

If building incrementally, prioritize in this order — each addresses one of your five concerns directly:

1. **Golden set + Judge Calibration Layer** — this is the single highest-leverage addition; without it, every other metric's trustworthiness is unknown. (Concern 1)
2. **Metric Tiering config** — cheap to add, immediately prevents safety-critical categories from silently depending on an LLM judge. (Concern 5)
3. **Efficiency Layer + paired reporting** — straightforward instrumentation, immediately surfaces cost-blind "ties." (Concern 3)
4. **Fair-Comparison Controller** — needed before the Improvement Pipeline (already deferred in v1) is turned on. (Concern 2)
5. **Confidence-tiering for black-box mode** — needed once you evaluate any system you didn't build yourself. (Concern 4)
