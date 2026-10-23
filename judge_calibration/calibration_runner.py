import json
import os

from deepeval.test_case import LLMTestCase

from models.model_client import create_judge
from evaluators.metric_manager import ALL_METRIC_MODULES, interpret_metric_result, build_metric
from judge_calibration.calibration_store import load_calibration_judgments

_SCORES_PATH = os.path.join(os.path.dirname(__file__), "data", "calibration_scores.json")

HIGH_CONFIDENCE_THRESHOLD = 0.8

def _ensure_scores_file():
    os.makedirs(os.path.dirname(_SCORES_PATH), exist_ok=True)
    if not os.path.exists(_SCORES_PATH):
        with open(_SCORES_PATH, "w", encoding="utf-8") as scores_file:
            json.dump({}, scores_file, indent=2)

def load_calibration_scores():
    _ensure_scores_file()
    with open(_SCORES_PATH, "r", encoding="utf-8") as scores_file:
        return json.load(scores_file)

def _save_calibration_scores(scores):
    _ensure_scores_file()
    with open(_SCORES_PATH, "w", encoding="utf-8") as scores_file:
        json.dump(scores, scores_file, indent=2)

def _judge_verdict(judgment, judge_model):
    """Re-runs the SAME metric this judgment is about, on the same
    (test_case, actual_output) pair, and returns what the metric itself
    currently says: "pass", "fail", or None if it can't be run (metric no
    longer registered, or no longer applicable to this test case).

    Must apply the same passed/failed reading a live run would --
    previously used raw metric.success directly, which is backwards for
    HallucinationMetric/FaithfulnessMetric on an adversarial Context (see
    metric_manager._interpret). Without this, a correct human judgment on
    an adversarial row would always register as a "disagreement" against
    the judge, which measures the inversion rule, not the judge."""

    metric_module = ALL_METRIC_MODULES.get(judgment["metric_key"])

    if metric_module is None:
        return None

    test_case = judgment["test_case"]

    applicable, _ = metric_module.is_applicable(test_case)
    if not applicable:
        return None

    context = test_case.get("context") or None

    llm_test_case = LLMTestCase(
        input=test_case.get("input"),
        actual_output=judgment["actual_output"],
        context=context,
        retrieval_context=context,
        expected_output=test_case.get("expected_answer"),
    )

    metric = build_metric(metric_module, judge_model, test_case)
    metric.measure(llm_test_case)

    status, _ = interpret_metric_result(judgment["metric_key"], metric, test_case)

    return "pass" if status == "passed" else "fail"

def run_calibration(judgments=None, judge_model=None):
    """Deterministic agreement computation: for every recorded human
    judgment, re-runs the metric it concerns and checks whether the
    metric's current verdict matches the human's. The agreement
    calculation itself is a plain equality count -- no AI decides
    agreement; only the underlying metric re-execution (inherent to
    every metric in this framework already) uses a judge model.

    Returns and persists a mapping:
        { metric_key: { category: {agreement, sample_size, confidence_label} } }
    """

    judgments = judgments if judgments is not None else load_calibration_judgments()
    judge_model = judge_model or create_judge()

    tally = {}

    for judgment in judgments:

        verdict = _judge_verdict(judgment, judge_model)

        if verdict is None:
            continue

        key = (judgment["metric_key"], judgment.get("category") or "uncategorized")
        bucket = tally.setdefault(key, {"agree": 0, "total": 0})
        bucket["total"] += 1
        if verdict == judgment["human_verdict"]:
            bucket["agree"] += 1

    scores = {}

    for (metric_key, category), counts in tally.items():

        agreement = counts["agree"] / counts["total"] if counts["total"] else 0.0

        scores.setdefault(metric_key, {})[category] = {
            "agreement": round(agreement, 4),
            "sample_size": counts["total"],
            "confidence_label": (
                "high" if agreement >= HIGH_CONFIDENCE_THRESHOLD else "advisory"
            ),
        }

    _save_calibration_scores(scores)

    return scores

def get_confidence_label(metric_key, category=None):
    """Deterministic lookup only -- no execution, no judge call. A metric
    with no recorded calibration data is "unverified", never silently
    treated as trustworthy by default."""

    scores = load_calibration_scores()
    metric_scores = scores.get(metric_key)

    if not metric_scores:
        return {"confidence_label": "unverified", "agreement": None, "sample_size": 0}

    if category and category in metric_scores:
        return metric_scores[category]

    if not category:
        total_agree = sum(
            round(entry["agreement"] * entry["sample_size"]) for entry in metric_scores.values()
        )
        total_n = sum(entry["sample_size"] for entry in metric_scores.values())

        if total_n == 0:
            return {"confidence_label": "unverified", "agreement": None, "sample_size": 0}

        agreement = total_agree / total_n

        return {
            "agreement": round(agreement, 4),
            "sample_size": total_n,
            "confidence_label": "high" if agreement >= HIGH_CONFIDENCE_THRESHOLD else "advisory",
        }

    return {"confidence_label": "unverified", "agreement": None, "sample_size": 0}
