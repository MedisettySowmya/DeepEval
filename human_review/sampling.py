import hashlib

from config import model_config


# ---------------------------------------------------------------------------
# Probabilistic half of the Human Review Sampling Gate
# (trustworthy-ai-eval-architecture.md section 3.6): "other items are
# sampled probabilistically" -- the deterministic half ("human-required
# tier items always route here") already exists via
# metric_tiering.tier_classifier's human_required gating status.
#
# Uses hash-based deterministic bucketing rather than true randomness: a
# hash of the test case's stable slot identity (category + position, same
# scheme as run_comparison and prompt_versioning) is mapped to a value in
# [0, 1) and compared against the configured rate. This keeps sampling
# reproducible -- the same test case slot samples the same way every time
# a run is inspected, rather than flapping between checks -- while still
# being effectively uniformly distributed across different test cases,
# consistent with this project's preference for deterministic mechanisms
# over arbitrary randomness everywhere else (Classification, Root Cause
# Analysis, Metric Tiering).
# ---------------------------------------------------------------------------

def _sample_score(slot_key):
    digest = hashlib.sha256(slot_key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def should_sample_for_review(test_case, sample_rate=None):

    sample_rate = (
        sample_rate if sample_rate is not None
        else model_config.HUMAN_REVIEW_SAMPLE_RATE
    )

    slot_key = f"{test_case.get('category')}:{test_case.get('category_position')}"

    return _sample_score(slot_key) < sample_rate
