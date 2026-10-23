"""Check spreadsheet data before generating model answers.

Flow: run_preflight -> check_row -> infer_context_role -> report.
Missing inputs skip a check; judge failures are recorded as errors.
"""
from deepeval.test_case import LLMTestCase

from config import settings
from models.model_client import create_judge
from evaluators.metric_manager import ALL_METRIC_MODULES
from evaluators.metric_registry import DATASET_CHECK_METRICS
from dataset_checks.dataset_loader import (
    CONTEXT_ROLE_ADVERSARIAL,
    CONTEXT_ROLE_SUPPORTING,
    CONTEXT_ROLE_UNKNOWN,
)

def _build_dataset_test_case(test_case):
    """Convert a dataset row to DeepEval's format, without a model answer."""

    context = test_case.get("context") or None

    return LLMTestCase(
        input=test_case.get("input"),
        retrieval_context=context,
        expected_output=test_case.get("expected_answer"),
    )

def check_row(test_case, judge_model=None):
    """Runs the dataset checks applicable to one row.

    Returns {metric_key: {status, score, reason}}. A metric whose inputs are
    absent is reported not_applicable with a reason, never silently dropped.
    """

    results = {}

    # 1. All dataset checks need a valid question and context.
    input_is_valid = test_case.get("input_valid", bool(test_case.get("input")))
    skip_reason = None
    if not input_is_valid:
        skip_reason = "invalid or empty Input"
    elif not test_case.get("has_context"):
        skip_reason = "no Context in this row; nothing to check it against"

    if skip_reason:
        for metric_key in DATASET_CHECK_METRICS:
            results[metric_key] = {
                "status": "not_applicable",
                "reason": skip_reason,
            }
        return results

    llm_test_case = _build_dataset_test_case(test_case)

    for metric_key in DATASET_CHECK_METRICS:

        # 2. Precision and recall need a factual reference answer.
        needs_reference_answer = metric_key != "ContextualRelevancyMetric"
        if needs_reference_answer and test_case.get("expects_abstention"):
            results[metric_key] = {
                "status": "not_applicable",
                "reason": "Ground_Truth describes behaviour, not a reference answer",
            }
            continue

        module = ALL_METRIC_MODULES.get(metric_key)

        if module is None:
            results[metric_key] = {
                "status": "not_applicable",
                "reason": f"metric '{metric_key}' is not registered",
            }
            continue

        applicable, reason = module.is_applicable(test_case)

        if not applicable:
            results[metric_key] = {"status": "not_applicable", "reason": reason}
            continue

        # 3. Run the check and record its result. Keep errors in the report.
        try:
            if judge_model is None:
                judge_model = create_judge()

            metric = module.build(judge_model)
            metric.measure(llm_test_case)

            results[metric_key] = {
                "status": "passed" if metric.success else "failed",
                "score": metric.score,
                "reason": metric.reason,
            }

        except Exception as error:
            results[metric_key] = {"status": "error", "reason": str(error)}

    return results

def infer_context_role(check_results):
    """Estimate context role using existing precision and recall scores.

    Unavailable checks give 'unknown'. All available scores at or below the
    threshold give 'adversarial'; otherwise return 'supporting'. This is a
    heuristic, not proof that the context is deliberately misleading.
    """

    precision = check_results.get("ContextualPrecisionMetric") or {}
    recall = check_results.get("ContextualRecallMetric") or {}

    completed_statuses = ("passed", "failed")
    if precision.get("status") not in completed_statuses:
        return CONTEXT_ROLE_UNKNOWN
    if recall.get("status") not in completed_statuses:
        return CONTEXT_ROLE_UNKNOWN

    threshold = settings.ADVERSARIAL_CONTEXT_THRESHOLD

    scores = []
    for result in (precision, recall):
        score = result.get("score")
        if score is not None:
            scores.append(score)

    if not scores:
        return CONTEXT_ROLE_UNKNOWN

    if all(score <= threshold for score in scores):
        return CONTEXT_ROLE_ADVERSARIAL

    return CONTEXT_ROLE_SUPPORTING

def run_preflight(dataset, judge_model=None, infer_roles=True):
    """Check each row and return detailed findings plus summary counts.

    With infer_roles=True, also update each original row's context_role
    for later scoring. False leaves the original rows unchanged.
    """

    judge_model = judge_model or None
    rows = []

    for test_case in dataset:

        checks = check_row(test_case, judge_model=judge_model)

        role = CONTEXT_ROLE_UNKNOWN
        if infer_roles:
            role = infer_context_role(checks)
            test_case["context_role"] = role

        rows.append({
            "id": test_case.get("id"),
            "category": test_case.get("category"),
            "input": test_case.get("input"),
            "profile": test_case.get("profile"),
            "checks": checks,
            "context_role": role,
            "data_warnings": get_data_warnings(test_case, checks),
        })

    summary = {
        "rows": len(rows),
        "with_warnings": sum(1 for row in rows if row["data_warnings"]),
        "adversarial": sum(
            1 for row in rows if row["context_role"] == CONTEXT_ROLE_ADVERSARIAL
        ),
        "supporting": sum(
            1 for row in rows if row["context_role"] == CONTEXT_ROLE_SUPPORTING
        ),
        "unknown": sum(
            1 for row in rows if row["context_role"] == CONTEXT_ROLE_UNKNOWN
        ),
    }

    return {"rows": rows, "summary": summary}

_LEAKED_PREFIXES = ("context:", "ground truth:", "ground_truth:", "answer:")

def get_data_warnings(test_case, check_results=None):
    """Returns a list of {code, message} warnings about the row's own cells."""

    warnings = []

    if not test_case.get("input_valid"):
        warnings.append({
            "code": "empty_input",
            "message": test_case.get("input_invalid_reason")
                       or "Input cell is empty.",
        })

    reference = str(test_case.get("expected_answer") or "")
    lowered = reference.strip().lower()

    for prefix in _LEAKED_PREFIXES:
        if lowered.startswith(prefix):
            warnings.append({
                "code": "leaked_column_prefix",
                "message": (
                    f"Ground_Truth begins with '{reference.strip()[:len(prefix)]}' -- "
                    "another column's label appears to have been pasted into "
                    "this cell. No reference metric can match it."
                ),
            })
            break

    if test_case.get("expects_abstention"):
        warnings.append({
            "code": "behavioural_reference",
            "message": (
                "Ground_Truth describes expected behaviour rather than an "
                "answer to match, so reference metrics are suppressed for "
                "this row and it routes to the abstention path. Correct for a "
                "fabricated-entity row; a mistake if a real answer was meant."
            ),
        })

    relevancy = (check_results or {}).get("ContextualRelevancyMetric") or {}
    if relevancy.get("status") == "failed":
        warnings.append({
            "code": "context_off_topic",
            "message": (
                "The Context is not relevant to the Input. Every "
                "context-based score on this row is untrustworthy until the "
                "cell is fixed."
            ),
        })

    return warnings
