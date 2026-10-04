# ---------------------------------------------------------------------------
# Metric Tiering (from trustworthy-ai-eval-architecture.md section 3.4),
# adapted to this application.
#
# Purpose: not every metric's verdict deserves the same trust. This module
# is what makes that a config-level fact rather than something a reader
# has to already know:
#
#   deterministic         -- no model involved (e.g. ExactMatchMetric).
#                             Trusted outright.
#   statistical            -- reserved for a future non-generative,
#                             non-deterministic metric (BLEU/ROUGE/
#                             embedding similarity); no metric currently
#                             discovered in the installed deepeval version
#                             falls into this tier.
#   llm_judge_calibrated   -- an LLM judge's opinion (most metrics here).
#                             Trusted only once judge_calibration says the
#                             judge actually agrees with humans on it
#                             (confidence_label == "high"); otherwise
#                             marked "advisory", never silently trusted.
#   human_required         -- can never be auto-passed by a judge, no
#                             matter the score. A policy decision a human
#                             opts a metric into via
#                             config.model_config.HUMAN_REQUIRED_METRICS,
#                             not something inferred automatically.
#
# Entirely deterministic: tier lookup is a dict/attribute read, and
# gating_status is a plain conditional on that tier plus (for
# llm_judge_calibrated) a lookup of already-computed calibration data. No
# AI decides trust here -- a metric failing does not automatically mean
# untrustworthy, and passing does not automatically mean trustworthy.
# ---------------------------------------------------------------------------

from config import model_config
from evaluators.metric_manager import ALL_METRIC_MODULES
from judge_calibration.calibration_runner import get_confidence_label


VALID_TIERS = {"deterministic", "statistical", "llm_judge_calibrated", "human_required"}

# Fallback for a metric_key not found in the registry at all (e.g. stale
# calibration data referencing a since-removed metric). The safe default
# is the most cautious tier, not blind trust.
_UNKNOWN_METRIC_TIER = "llm_judge_calibrated"


def get_metric_tier(metric_key):

    if metric_key in model_config.HUMAN_REQUIRED_METRICS:
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

    if tier in ("deterministic", "statistical"):
        return "trusted"

    # llm_judge_calibrated
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
