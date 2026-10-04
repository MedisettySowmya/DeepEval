# ---------------------------------------------------------------------------
# Pre-flight dataset checks.
#
# Three DeepEval metrics read only the spreadsheet -- Input, Context and
# Ground truth -- and never the model's answer:
#
#   ContextualRelevancyMetric   ['input', 'retrieval_context']
#   ContextualPrecisionMetric   ['input', 'retrieval_context', 'expected_output']
#   ContextualRecallMetric      ['input', 'retrieval_context', 'expected_output']
#
# None of them declares actual_output, and LLMTestCase does not require one
# (it defaults to None). So they need no generator, and a metric that cannot
# be affected by the answer does not need to wait for one. They run here, over
# the whole sheet, BEFORE any generation is paid for -- which matters now that
# the judge is a billed API: a row whose Context contradicts its Ground truth
# is worth catching in seconds rather than discovering in a report after a
# long run.
#
# Their failures are statements about the dataset, not about the model. This
# module never reports a model defect.
# ---------------------------------------------------------------------------

from deepeval.test_case import LLMTestCase

from config import model_config
from config.judge_model_resolver import load_model
from evaluators.metric_manager import ALL_METRIC_MODULES
from evaluators.metric_registry import DATASET_CHECK_METRICS
from utils.dataset_loader import (
    CONTEXT_ROLE_ADVERSARIAL,
    CONTEXT_ROLE_SUPPORTING,
    CONTEXT_ROLE_UNKNOWN,
)


def _build_dataset_test_case(test_case):
    """No actual_output: these metrics do not read it, and omitting it makes
    the absence of a model answer explicit rather than faked with a
    placeholder."""

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

    if not test_case.get("has_context"):
        for metric_key in DATASET_CHECK_METRICS:
            results[metric_key] = {
                "status": "not_applicable",
                "reason": "no Context in this row; nothing to check it against",
            }
        return results

    llm_test_case = _build_dataset_test_case(test_case)

    for metric_key in DATASET_CHECK_METRICS:

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

        try:
            if judge_model is None:
                judge_model = load_model()

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
    """Derives whether a row's Context supports or contradicts its Ground
    truth, with no extra model call and no new Excel column.

    ContextualPrecision asks "does the Context support the Ground truth";
    ContextualRecall asks "is the Ground truth recoverable from the Context".
    Neither reads the answer, so a simultaneous failure says only one thing:
    the Context and the Ground truth describe different worlds. That is the
    adversarial signal -- deliberate on a trap row, a data bug anywhere else.

    Returns one of the CONTEXT_ROLE_* constants. Unknown when the evidence is
    absent or errored, so a missing judge never silently produces
    "supporting" and disables the inversion rule.
    """

    precision = check_results.get("ContextualPrecisionMetric") or {}
    recall = check_results.get("ContextualRecallMetric") or {}

    statuses = (precision.get("status"), recall.get("status"))

    if not all(status in ("passed", "failed") for status in statuses):
        return CONTEXT_ROLE_UNKNOWN

    threshold = model_config.ADVERSARIAL_CONTEXT_THRESHOLD

    scores = [
        score for score in (precision.get("score"), recall.get("score"))
        if score is not None
    ]

    if not scores:
        return CONTEXT_ROLE_UNKNOWN

    if all(score <= threshold for score in scores):
        return CONTEXT_ROLE_ADVERSARIAL

    return CONTEXT_ROLE_SUPPORTING


def run_preflight(dataset, judge_model=None, infer_roles=True):
    """Checks every row in the dataset before any generation happens.

    When `infer_roles` is set, each row's `context_role` is filled in from its
    own check results, so the inversion rule in metric_manager has what it
    needs in the same run rather than the next one. The dataset rows are
    mutated in place, which is how the orchestrator receives the value.

    Returns a report:
        {
            "rows": [{"id", "category", "input", "checks", "context_role",
                      "data_warnings"}, ...],
            "summary": {"rows", "with_warnings", "adversarial", "supporting",
                        "unknown"},
        }
    """

    judge_model = judge_model or None
    rows = []

    for test_case in dataset:

        checks = check_row(test_case, judge_model=judge_model)

        role = infer_context_role(checks) if infer_roles else CONTEXT_ROLE_UNKNOWN

        if infer_roles:
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


# ---------------------------------------------------------------------------
# Deterministic data-hygiene checks. No model, no metric -- plain inspection
# of the cells, for the defects that a judge-based check would describe only
# indirectly.
# ---------------------------------------------------------------------------

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
