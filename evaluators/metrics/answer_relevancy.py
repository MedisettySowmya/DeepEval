from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCaseParams

from config import model_config


KEY = "AnswerRelevancyMetric"
METRIC_CLASS = AnswerRelevancyMetric
TIER = "llm_judge_calibrated"
REQUIRED_PARAMS = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]


def is_applicable(test_case):
    return True, None


def build(judge_model):
    threshold = model_config.METRIC_THRESHOLDS.get(
        KEY, model_config.DEFAULT_METRIC_THRESHOLD
    )
    return METRIC_CLASS(threshold=threshold, model=judge_model)
