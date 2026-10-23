import hashlib

from config import settings

def _sample_score(slot_key):
    digest = hashlib.sha256(slot_key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF

def should_sample_for_review(test_case, sample_rate=None):

    sample_rate = (
        sample_rate if sample_rate is not None
        else settings.HUMAN_REVIEW_SAMPLE_RATE
    )

    slot_key = f"{test_case.get('category')}:{test_case.get('category_position')}"

    return _sample_score(slot_key) < sample_rate
