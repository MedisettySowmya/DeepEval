# ---------------------------------------------------------------------------
# Root Cause Analysis -- deterministic, rule-based hypotheses about why a
# metric failed or errored. No AI model is called here, and nothing is ever
# automatically applied: every result is a hypothesis for a human tester to
# review, not a verdict.
#
# Important: a metric failure is NEVER automatically treated as
# model_behavior. Each rule below cross-checks the failure against other
# already-computed evidence (other metric results, the classification, the
# dataset's own data-gap flag) before concluding anything about the model,
# and confidence is lowered whenever there's no corroborating signal.
# ---------------------------------------------------------------------------

MODEL_BEHAVIOR = "model_behavior"
DATASET_ISSUE = "dataset_issue"
CONTEXT_ISSUE = "context_issue"
GROUND_TRUTH_ISSUE = "ground_truth_issue"
METRIC_CONFIGURATION_ISSUE = "metric_configuration_issue"
EVALUATION_ERROR = "evaluation_error"
UNKNOWN = "unknown"

# Metrics whose applicability rule only requires input/actual_output but
# which are semantically designed for agent/task/summarization scenarios
# (documented in evaluators/metrics/generic.py's discovery logic) -- a
# failure here is more likely a metric/task mismatch than a real finding
# about a plain QA response.
_NON_QA_FIT_METRICS = {
    "PlanAdherenceMetric", "PlanQualityMetric", "RoleViolationMetric",
    "StepEfficiencyMetric", "SummarizationMetric", "TaskCompletionMetric",
}

_REFERENCE_METRICS = {"AnswerCorrectness", "ExactMatchMetric"}
_CONTEXT_METRICS = {"HallucinationMetric"}
_SAFETY_METRICS = {"BiasMetric", "ToxicityMetric", "PIILeakageMetric"}


def _diagnosis(root_cause, confidence, evidence, recommended_action):
    return {
        "root_cause": root_cause,
        "confidence": confidence,
        "evidence": evidence,
        "recommended_action": recommended_action,
    }


def _diagnose_validation_failure(validation_result):

    failure_type = (validation_result or {}).get("failure_type")

    if failure_type in ("timeout", "generation_error"):
        return _diagnosis(
            EVALUATION_ERROR,
            "high",
            (
                f"Response validation failed with failure_type="
                f"'{failure_type}' before any semantic evaluation could "
                f"run: {(validation_result or {}).get('reason')}"
            ),
            (
                "Investigate generator availability, network/timeout "
                "configuration, or the model service itself, then re-run. "
                "This is an infrastructure signal, not a finding about "
                "model quality."
            ),
        )

    if failure_type == "empty_response":
        return _diagnosis(
            MODEL_BEHAVIOR,
            "medium",
            "The generator produced an empty or null response for this input.",
            (
                "Review the prompt for this input; consider whether the "
                "model is refusing, truncating, or failing silently on "
                "this kind of question."
            ),
        )

    return _diagnosis(
        UNKNOWN,
        "low",
        f"Validation failed with an unrecognized failure_type: {failure_type}",
        "Manually review this test case's generation and validation output.",
    )


def _diagnose_metric_result(metric_key, metric_result, test_case, evaluation_result, classification_result):

    status = metric_result.get("status")

    if status == "error":
        return _diagnosis(
            METRIC_CONFIGURATION_ISSUE,
            "medium",
            (
                f"Metric '{metric_key}' raised an unexpected error during "
                f"measurement rather than producing a score: "
                f"{metric_result.get('reason')}"
            ),
            (
                "Review whether this metric is compatible with this test "
                "case's input/output shape, or requires configuration "
                "this framework does not currently provide. This is a "
                "hypothesis, not a confirmed defect."
            ),
        )

    if status != "failed":
        return None

    if metric_key in _NON_QA_FIT_METRICS:
        return _diagnosis(
            METRIC_CONFIGURATION_ISSUE,
            "medium",
            (
                f"'{metric_key}' is designed for agent/task/summarization "
                "scenarios; it was routed here because its applicability "
                "rule only checks for input/actual_output, so its "
                "criteria may not be meaningful for a plain QA response."
            ),
            (
                "Review whether this metric should apply to this kind of "
                "test case at all, rather than treating this failure as "
                "a finding about the model's response."
            ),
        )

    classification = (classification_result or {}).get("classification")

    if classification == "UNVERIFIABLE":
        behavior_signal = (classification_result or {}).get("behavior_signal") or {}
        confidence = (
            "high" if behavior_signal.get("signal") == "unsupported_claim_likely"
            else "medium"
        )
        return _diagnosis(
            MODEL_BEHAVIOR,
            confidence,
            (
                f"No context or expected answer was available for this "
                f"test case, so '{metric_key}' failing reflects a "
                f"property of the response itself, not a factual "
                f"discrepancy. Behavior signal: "
                f"{behavior_signal.get('signal', 'not_determined')}."
            ),
            (
                "Review the response manually for overconfident or "
                "unsupported claims; consider whether the model should be "
                "prompted to express uncertainty for this kind of question."
            ),
        )

    other_statuses = {
        key: result.get("status")
        for key, result in (evaluation_result or {}).items()
        if key != metric_key
    }

    if metric_key in _REFERENCE_METRICS:
        context_status = next(
            (other_statuses[key] for key in _CONTEXT_METRICS if key in other_statuses),
            None
        )
        if context_status == "passed":
            return _diagnosis(
                GROUND_TRUTH_ISSUE,
                "medium",
                (
                    f"'{metric_key}' failed (response does not match the "
                    "expected answer), but the context-based hallucination "
                    "check passed (response is consistent with the given "
                    "context). This split suggests the context and the "
                    "expected answer may conflict with each other, rather "
                    "than the model clearly being wrong."
                ),
                (
                    "Review this row's Context and Ground_Truth together "
                    "for consistency; either may be the source of the "
                    "conflict, not necessarily the model."
                ),
            )

    if metric_key in _CONTEXT_METRICS:
        reference_status = next(
            (other_statuses[key] for key in _REFERENCE_METRICS if key in other_statuses),
            None
        )
        if reference_status == "passed":
            return _diagnosis(
                CONTEXT_ISSUE,
                "high",
                (
                    f"'{metric_key}' failed (response conflicts with the "
                    "given context), but the response matches the "
                    "expected answer. The model appears to have answered "
                    "correctly despite -- not because of -- the provided "
                    "context, which suggests the context itself may be "
                    "misleading or incorrect."
                ),
                (
                    "Review this row's Context for accuracy; it may be "
                    "the source of the conflict rather than the model."
                ),
            )

    if test_case.get("possible_missing_data"):
        return _diagnosis(
            DATASET_ISSUE,
            "medium",
            (
                "This row was already flagged by the Dataset Manager as a "
                "likely data gap relative to other rows in its category "
                "(possible_missing_data=True)."
            ),
            (
                "Review this row's Context/Ground_Truth completeness "
                "before treating this failure as a model issue."
            ),
        )

    if metric_key in _SAFETY_METRICS:
        return _diagnosis(
            MODEL_BEHAVIOR,
            "high",
            (
                f"'{metric_key}' is a response-safety check independent of "
                "evidence availability; its failure is a direct property "
                "of the generated response."
            ),
            (
                "Review the response content directly; consider prompt "
                "constraints or model-level safety tuning."
            ),
        )

    return _diagnosis(
        MODEL_BEHAVIOR,
        "low",
        (
            f"'{metric_key}' failed and no cross-checking evidence (from "
            "other metrics or dataset flags) points to an alternative "
            "explanation. This is the weakest form of evidence for "
            "model_behavior -- treat as a starting hypothesis only."
        ),
        (
            "Manually review the response, input, and context together "
            "before concluding this is a model defect."
        ),
    )


def analyze_test_case(test_case, validation_result, evaluation_result=None, classification_result=None):
    """Deterministic root-cause hypotheses for a test case's failed/errored
    metrics (or, if generation/validation itself failed, for that).

    Returns:
        {
            "test_case_level": <diagnosis> | None,   # populated only when
                                                       # validation failed
                                                       # and no evaluation
                                                       # ran at all
            "metric_level": {metric_key: <diagnosis>, ...}  # only failed/
                                                              # errored keys
        }
    """

    if not (validation_result or {}).get("valid", False):
        return {
            "test_case_level": _diagnose_validation_failure(validation_result),
            "metric_level": {}
        }

    metric_level = {}

    for metric_key, metric_result in (evaluation_result or {}).items():
        diagnosis = _diagnose_metric_result(
            metric_key, metric_result, test_case, evaluation_result, classification_result
        )
        if diagnosis is not None:
            metric_level[metric_key] = diagnosis

    return {"test_case_level": None, "metric_level": metric_level}
