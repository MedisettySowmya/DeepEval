from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCaseParams

from config import model_config


KEY = "HallucinationMetric"
METRIC_CLASS = HallucinationMetric
TIER = "llm_judge_calibrated"
REQUIRED_PARAMS = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.CONTEXT,
]


def is_applicable(test_case):
    if not test_case.get("context"):
        return False, "missing required test-case data: context"
    return True, None


def build(judge_model):
    threshold = model_config.METRIC_THRESHOLDS.get(
        KEY, model_config.DEFAULT_METRIC_THRESHOLD
    )
    return METRIC_CLASS(threshold=threshold, model=judge_model)
