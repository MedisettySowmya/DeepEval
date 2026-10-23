from human_review.sampling import should_sample_for_review

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
