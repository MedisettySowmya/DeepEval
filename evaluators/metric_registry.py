
from dataset_checks.dataset_loader import (
    PROFILE_ABSTENTION,
    PROFILE_FACTUAL,
    PROFILE_NO_REFERENCE,
    PROFILE_NUMERIC,
)

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

GATE_STAGES = frozenset({STAGE_GENERATION, STAGE_DETERMINISTIC, STAGE_RELEVANCE})

UNGATED_STAGES = frozenset({STAGE_SAFETY})

ALL_PROFILES = frozenset({
    PROFILE_NO_REFERENCE,
    PROFILE_ABSTENTION,
    PROFILE_NUMERIC,
    PROFILE_FACTUAL,
})

REFERENCE_PROFILES = frozenset({PROFILE_NUMERIC, PROFILE_FACTUAL})

REGISTRY = {
    "AnswerCompleteness": {
        "stage": STAGE_REFERENCE, "profiles": REFERENCE_PROFILES,
        "requires": ("reference",),
    },
    "ROUGELMetric": {
        "stage": STAGE_REFERENCE, "profiles": REFERENCE_PROFILES,
        "requires": ("reference",),
    },
    "BLEUMetric": {
        "stage": STAGE_REFERENCE, "profiles": REFERENCE_PROFILES,
        "requires": ("reference",),
    },
    "CitationCorrectness": {
        "stage": STAGE_SAFETY, "profiles": ALL_PROFILES,
        "requires": ("context_in_prompt",),
    },
    "InstructionPriority": {
        "stage": STAGE_SAFETY, "profiles": ALL_PROFILES,
        "requires": ("context_in_prompt",),
    },
    "PromptAlignmentMetric": {
        "stage": STAGE_SAFETY,
        "profiles": ALL_PROFILES,
        "requires": (),
    },
    "MisuseMetric": {
        "stage": STAGE_SAFETY,
        "profiles": ALL_PROFILES,
        "requires": (),
    },
    "NonAdviceMetric": {
        "stage": STAGE_SAFETY,
        "profiles": ALL_PROFILES,
        "requires": (),
    },
    "AbstentionMetric": {
        "stage": STAGE_SAFETY,
        "profiles": frozenset({PROFILE_ABSTENTION}),
        "requires": (),
        "note": "judge-based behaviour check; not a deterministic refusal gate",
    },

    "ExactMatchMetric": {
        "stage": STAGE_DETERMINISTIC,
        "profiles": REFERENCE_PROFILES,
        "requires": ("reference", "short_reference"),
        "note": "exact equality is only meaningful for a short canonical answer",
    },
    "pattern_match": {
        "stage": STAGE_DETERMINISTIC,
        "profiles": ALL_PROFILES,
        "requires": ("pattern",),
        "note": "no LLM -- regex from the dataset's own Pattern column",
    },
    "numeric_match": {
        "stage": STAGE_DETERMINISTIC,
        "profiles": REFERENCE_PROFILES,
        "requires": ("reference",),
        "note": "no LLM -- a wrong figure is conclusive, unlike a judge's opinion",
    },

    "AnswerRelevancyMetric": {
        "stage": STAGE_RELEVANCE,
        "profiles": ALL_PROFILES - {PROFILE_ABSTENTION},
        "requires": (),
        "note": "gate -- if the answer is off topic, accuracy scores measure nothing",
    },

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

    "AnswerCorrectness": {
        "stage": STAGE_REFERENCE,
        "profiles": REFERENCE_PROFILES,
        "requires": ("reference",),
        "note": "judge rubric; read next to semantic similarity",
    },
    "SemanticSimilarityMetric": {
        "stage": STAGE_REFERENCE,
        "profiles": REFERENCE_PROFILES,
        "requires": ("reference",),
        "note": "no LLM judge involved -- the trust anchor for Stage 4",
    },

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

DATASET_CHECK_METRICS = (
    "ContextualRelevancyMetric",
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
)

NOT_IN_LADDER_REASONS = {
    "ContextualRelevancyMetric": "dataset check -- runs pre-flight, not in the answer ladder",
    "ContextualPrecisionMetric": "dataset check -- runs pre-flight, not in the answer ladder",
    "ContextualRecallMetric": "dataset check -- runs pre-flight, not in the answer ladder",
}

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
