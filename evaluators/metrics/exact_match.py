from deepeval.metrics import ExactMatchMetric
from deepeval.test_case import LLMTestCaseParams

from config import model_config
from evaluators import metric_registry


KEY = "ExactMatchMetric"
METRIC_CLASS = ExactMatchMetric
TIER = "deterministic"
REQUIRED_PARAMS = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.EXPECTED_OUTPUT,
]


def is_applicable(test_case):
    """Narrowed rather than removed.

    This is the only deterministic-tier metric, so metric_tiering grants it
    `trusted` automatically and it therefore decides the suite's pass/fail on
    its own. Character-for-character equality against a generated sentence
    fails even when the answer is correct ("The capital of Australia is
    Canberra, chosen as a compromise..." vs "Canberra is the capital of
    Australia"), which is how a real run failed all six rows while the model
    was right. Restricting it to short canonical answers keeps it registered
    and useful where exact equality is the right question -- numbers, names,
    single tokens -- and reports an honest reason everywhere else.
    """

    if not test_case.get("expected_answer"):
        return False, "missing required test-case data: expected_output"

    if not metric_registry.is_short_reference(test_case):
        return False, (
            "the Ground_Truth is a sentence, not a short canonical value; "
            "exact equality would fail even on a correct answer"
        )

    return True, None


def build(judge_model):
    # Deterministic string comparison -- no judge model involved.
    threshold = model_config.METRIC_THRESHOLDS.get(
        KEY, model_config.DEFAULT_METRIC_THRESHOLD
    )
    return METRIC_CLASS(threshold=threshold)
