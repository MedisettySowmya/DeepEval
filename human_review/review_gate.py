from human_review.sampling import should_sample_for_review


# ---------------------------------------------------------------------------
# Combines both halves of the Human Review Sampling Gate into one
# per-test-case decision:
#   - deterministic: any metric already tagged "requires_human_review" by
#     metric_tiering (i.e. in config.model_config.HUMAN_REQUIRED_METRICS)
#   - probabilistic: this test case's slot falls in the sampled fraction
#     (human_review.sampling.should_sample_for_review)
#
# A test case can have both, one, or neither reason. This module makes no
# decision of its own beyond combining the two -- it calls no metric, no
# judge model, and modifies nothing. What a human does with a flagged item
# is entirely up to them; if they review it, recording their verdict uses
# the existing judge_calibration.calibration_store.record_calibration_
# judgment -- no separate feedback-loop mechanism was built, since that
# one already exists and fits exactly.
# ---------------------------------------------------------------------------

def get_review_reasons(test_case, evaluation_result=None):
    """Returns a list of reasons this test case is flagged for human
    review, or None if neither gate applies."""

    reasons = []

    for metric_key, metric_result in (evaluation_result or {}).items():
        if metric_result.get("gating_status") == "requires_human_review":
            reasons.append(f"human_required_metric:{metric_key}")

    if should_sample_for_review(test_case):
        reasons.append("sampled")

    return reasons or None
