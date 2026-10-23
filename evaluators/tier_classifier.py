
from config import settings
from evaluators.metric_manager import ALL_METRIC_MODULES
from judge_calibration.calibration_runner import get_confidence_label

VALID_TIERS = {"deterministic", "statistical", "llm_judge_calibrated", "human_required"}

_UNKNOWN_METRIC_TIER = "llm_judge_calibrated"

def get_metric_tier(metric_key):

    if metric_key in settings.HUMAN_REQUIRED_METRICS:
        return "human_required"

    module = ALL_METRIC_MODULES.get(metric_key)

    if module is None:
        return _UNKNOWN_METRIC_TIER

    tier = getattr(module, "TIER", None) or getattr(module, "tier", None)

    return tier if tier in VALID_TIERS else _UNKNOWN_METRIC_TIER

def get_gating_status(metric_key, metric_result, category=None):
    """trusted | advisory | requires_human_review | not_applicable"""

    if metric_result.get("status") in ("not_applicable", "error"):
        return "not_applicable"

    tier = get_metric_tier(metric_key)

    if tier == "human_required":
        return "requires_human_review"

    if getattr(ALL_METRIC_MODULES.get(metric_key), "ADVISORY_ONLY", False):
        return "advisory"

    if tier in ("deterministic", "statistical"):
        return "trusted"

    confidence = get_confidence_label(metric_key, category)

    return "trusted" if confidence["confidence_label"] == "high" else "advisory"

def annotate_evaluation_result(evaluation_result, category=None):
    """Additive only: returns a new dict with `tier` and `gating_status`
    added to each metric's existing result, without altering status,
    score, or reason. Downstream consumers (reporting, a future pass/fail
    gate) decide what to do with `advisory`/`requires_human_review`
    results -- this module only labels them."""

    return {
        metric_key: {
            **metric_result,
            "tier": get_metric_tier(metric_key),
            "gating_status": get_gating_status(metric_key, metric_result, category),
        }
        for metric_key, metric_result in evaluation_result.items()
    }
