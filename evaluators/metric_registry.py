# ---------------------------------------------------------------------------
# Curated metric registry.
#
# Replaces "attempt every metric dir(deepeval.metrics) can see, and let field
# presence decide" with an explicit map of which metrics are relevant to which
# kind of test case, and at which stage they run.
#
# Why curation rather than discovery: a field-presence check cannot tell
# "needs a plan" from "needs an answer". Five DeepEval metrics
# (TaskCompletion, PlanAdherence, PlanQuality, StepEfficiency, Summarization)
# declare only input + actual_output as required params, so discovery waves
# them through on plain QA rows -- PlanQualityMetric passed all six rows of a
# real run while measuring nothing. Discovery is still used, but only to find
# metrics; relevance is decided here.
#
# Nothing is deleted. A metric that is a poor fit stays registered and is
# reported not_applicable with a reason, per the convention
# model_config.KNOWN_UNSUITABLE_METRICS already set.
# ---------------------------------------------------------------------------

from utils.dataset_loader import (
    PROFILE_ABSTENTION,
    PROFILE_FACTUAL,
    PROFILE_NO_REFERENCE,
    PROFILE_NUMERIC,
)


# Execution stages, in order. Lower runs first.
STAGE_GENERATION = 0
STAGE_DETERMINISTIC = 1
STAGE_RELEVANCE = 2
STAGE_GROUNDING = 3
STAGE_REFERENCE = 4
STAGE_SAFETY = 5

STAGE_NAMES = {
    STAGE_GENERATION: "generation",
    STAGE_DETERMINISTIC: "deterministic",
    STAGE_RELEVANCE: "relevance",
    STAGE_GROUNDING: "grounding",
    STAGE_REFERENCE: "reference",
    STAGE_SAFETY: "safety",
}

# Stages whose failure halts the stages after them.
GATE_STAGES = frozenset({STAGE_GENERATION, STAGE_DETERMINISTIC, STAGE_RELEVANCE})

# Stage 5 is deliberately outside the gating chain: a refusal can still be
# toxic and an off-topic answer can still leak PII, so safety runs on any
# response that exists.
UNGATED_STAGES = frozenset({STAGE_SAFETY})

ALL_PROFILES = frozenset({
    PROFILE_NO_REFERENCE,
    PROFILE_ABSTENTION,
    PROFILE_NUMERIC,
    PROFILE_FACTUAL,
})

REFERENCE_PROFILES = frozenset({PROFILE_NUMERIC, PROFILE_FACTUAL})


# ---------------------------------------------------------------------------
# requires: extra conditions beyond the profile, checked against the test case
# and the prompt result.
#   "context_in_prompt" -- the Context actually reached the model, so grading
#                          the answer against it is a real measurement
#   "reference"         -- a Ground_Truth exists AND is a comparable answer
#                          (so: not an abstention row, whose Ground_Truth
#                          describes behaviour rather than text to match)
#   "pattern"           -- the row's Excel "Pattern" column holds a regex the
#                          answer is required to contain
#
# inverted: the metric scores agreement with the Context, and this dataset's
# Context is deliberately false. A HIGH score therefore means the response
# contradicted the Context, i.e. resisted the trap. See
# metric_manager._interpret.
# ---------------------------------------------------------------------------

REGISTRY = {

    # --- Stage 1: deterministic, no LLM, free ------------------------------
    "ExactMatchMetric": {
        "stage": STAGE_DETERMINISTIC,
        # Narrowed, not removed. Character-for-character equality against
        # free-form prose fails even on correct answers, and because this is
        # the only deterministic-tier metric it is auto-trusted and so decides
        # the whole suite. Restricted to short canonical answers, where exact
        # equality is the right question.
        "profiles": REFERENCE_PROFILES,
        "requires": ("reference", "short_reference"),
        "note": "exact equality is only meaningful for a short canonical answer",
    },
    "pattern_match": {
        "stage": STAGE_DETERMINISTIC,
        # Relevant to any row, but only actually runs where the tester filled
        # in the Excel "Pattern" column -- so a dataset with no Pattern
        # column behaves exactly as before.
        "profiles": ALL_PROFILES,
        "requires": ("pattern",),
        "note": "no LLM -- regex from the dataset's own Pattern column",
    },
    "numeric_match": {
        "stage": STAGE_DETERMINISTIC,
        # No LLM, free, and conclusive when it fails: 364 is not 366. Only
        # applies to rows whose Ground_Truth actually contains a digit --
        # the metric's own is_applicable() reports an honest reason on the
        # rest rather than scoring them 0.
        "profiles": REFERENCE_PROFILES,
        "requires": ("reference",),
        "note": "no LLM -- a wrong figure is conclusive, unlike a judge's opinion",
    },

    # --- Stage 2: the relevance gate ---------------------------------------
    "AnswerRelevancyMetric": {
        "stage": STAGE_RELEVANCE,
        # Bypassed on abstention rows: a refusal legitimately scores low
        # relevance, which is a metric artifact rather than a finding.
        "profiles": ALL_PROFILES - {PROFILE_ABSTENTION},
        "requires": (),
        "note": "gate -- if the answer is off topic, accuracy scores measure nothing",
    },

    # --- Stage 3: grounding, answer vs Context -----------------------------
    "HallucinationMetric": {
        "stage": STAGE_GROUNDING,
        "profiles": ALL_PROFILES,
        "requires": ("context_in_prompt",),
        "inverted": True,
        "note": "scores contradiction of the Context; inverted on an adversarial Context",
    },
    "FaithfulnessMetric": {
        "stage": STAGE_GROUNDING,
        "profiles": ALL_PROFILES,
        "requires": ("context_in_prompt",),
        "inverted": True,
        "note": "same quantity as Hallucination, against retrieval_context",
    },

    # --- Stage 4: reference, answer vs Ground truth ------------------------
    "AnswerCorrectness": {
        "stage": STAGE_REFERENCE,
        "profiles": REFERENCE_PROFILES,
        "requires": ("reference",),
        "note": "judge rubric; read next to semantic similarity",
    },
    "SemanticSimilarityMetric": {
        "stage": STAGE_REFERENCE,
        # statistical tier, no LLM judge -- the trust anchor this stage was
        # missing. Read next to AnswerCorrectness: if similarity passes but
        # AnswerCorrectness fails, suspect the judge rather than the model
        # (see trustworthy-ai-eval-architecture-v3.md section 7).
        "profiles": REFERENCE_PROFILES,
        "requires": ("reference",),
        "note": "no LLM judge involved -- the trust anchor for Stage 4",
    },

    # --- Stage 5: compliance and safety, never gated -----------------------
    "BiasMetric": {
        "stage": STAGE_SAFETY,
        "profiles": ALL_PROFILES,
        "requires": (),
    },
    "ToxicityMetric": {
        "stage": STAGE_SAFETY,
        "profiles": ALL_PROFILES,
        "requires": (),
    },
    "PIILeakageMetric": {
        "stage": STAGE_SAFETY,
        "profiles": ALL_PROFILES,
        "requires": (),
    },
}


# Metrics that read only the spreadsheet -- Input, Context, Ground truth --
# and never the model's answer. They are not stages: they need no generator
# and no actual_output (LLMTestCase defaults it to None, and none of the three
# declares it required), so they run as a pre-flight pass over the sheet
# before any generation is paid for. See dataset_checks/preflight.py.
DATASET_CHECK_METRICS = (
    "ContextualRelevancyMetric",
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
)

# Everything the registry deliberately leaves out of the answer-side ladder,
# with the reason reported instead of a silent absence.
NOT_IN_LADDER_REASONS = {
    "ContextualRelevancyMetric": "dataset check -- runs pre-flight, not in the answer ladder",
    "ContextualPrecisionMetric": "dataset check -- runs pre-flight, not in the answer ladder",
    "ContextualRecallMetric": "dataset check -- runs pre-flight, not in the answer ladder",
}

# How long a Ground_Truth may be and still be a candidate for exact matching.
SHORT_REFERENCE_MAX_WORDS = 4


def is_short_reference(test_case):
    """True when the Ground_Truth is a short canonical value -- a number, a
    name, a single token -- rather than a sentence. Only then is
    character-for-character equality a meaningful question."""

    reference = test_case.get("expected_answer")

    if not reference:
        return False

    return len(str(reference).split()) <= SHORT_REFERENCE_MAX_WORDS


def get_stage(metric_key):
    entry = REGISTRY.get(metric_key)
    return entry["stage"] if entry else None


def is_registered(metric_key):
    return metric_key in REGISTRY


def is_inverted(metric_key):
    return bool((REGISTRY.get(metric_key) or {}).get("inverted"))


def metrics_for_stage(stage):
    return [key for key, entry in REGISTRY.items() if entry["stage"] == stage]


def stages_in_order():
    return sorted({entry["stage"] for entry in REGISTRY.values()})


def check_relevance(metric_key, test_case, context_in_prompt):
    """Is this metric relevant to THIS row? Deterministic, executes nothing.

    Returns (relevant, reason). The reason is only populated when it is not,
    so it can be reported rather than the metric silently vanishing.
    """

    entry = REGISTRY.get(metric_key)

    if entry is None:
        reason = NOT_IN_LADDER_REASONS.get(
            metric_key,
            "not in the curated registry for single-turn QA evaluation",
        )
        return False, reason

    profile = test_case.get("profile") or PROFILE_NO_REFERENCE

    if profile not in entry["profiles"]:
        return False, f"not relevant to a '{profile}' test case"

    requires = entry.get("requires") or ()

    if "context_in_prompt" in requires and not context_in_prompt:
        return False, (
            "the Context was not sent to the model; grading the answer "
            "against unseen evidence is not a measurement"
        )

    if "reference" in requires:
        if not test_case.get("has_reference"):
            return False, "no Ground_Truth to compare against"
        if test_case.get("expects_abstention"):
            return False, (
                "the Ground_Truth describes expected behaviour, not an "
                "answer to match"
            )

    if "pattern" in requires and not test_case.get("expected_pattern"):
        return False, (
            "no Pattern given for this row (the Excel 'Pattern' column is "
            "empty), so there is no required format to check"
        )

    if "short_reference" in requires and not is_short_reference(test_case):
        return False, (
            "the Ground_Truth is a sentence, not a short canonical value; "
            "character-for-character equality would fail even on a correct answer"
        )

    return True, None


def plan_for_test_case(test_case, context_in_prompt):
    """The full relevance plan for a row, grouped by stage and executing
    nothing. This is what a UI can show before paying for a run, and what the
    report uses to explain why a metric did not appear.

    Returns {stage: [{"metric", "relevant", "reason"}, ...]}
    """

    plan = {}

    for stage in stages_in_order():
        rows = []
        for metric_key in sorted(metrics_for_stage(stage)):
            relevant, reason = check_relevance(
                metric_key, test_case, context_in_prompt
            )
            rows.append({
                "metric": metric_key,
                "relevant": relevant,
                "reason": reason,
            })
        plan[stage] = rows

    return plan
