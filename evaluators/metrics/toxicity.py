from deepeval.metrics import ToxicityMetric
from deepeval.test_case import LLMTestCaseParams

from config import settings

KEY = "ToxicityMetric"
METRIC_CLASS = ToxicityMetric
TIER = "llm_judge_calibrated"
REQUIRED_PARAMS = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]

def is_applicable(test_case):
    return True, None

def build(judge_model):
    threshold = settings.METRIC_THRESHOLDS.get(
        KEY, settings.DEFAULT_METRIC_THRESHOLD
    )
    return METRIC_CLASS(threshold=threshold, model=judge_model)
