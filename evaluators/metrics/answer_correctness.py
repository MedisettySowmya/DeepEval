from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from config import model_config


KEY = "AnswerCorrectness"
METRIC_CLASS = GEval
TIER = "llm_judge_calibrated"
REQUIRED_PARAMS = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.EXPECTED_OUTPUT,
]

CRITERIA = (
    "Determine whether the actual output is factually consistent "
    "with the expected output."
)


def is_applicable(test_case):
    if not test_case.get("expected_answer"):
        return False, "missing required test-case data: expected_output"
    return True, None


def build(judge_model):
    threshold = model_config.METRIC_THRESHOLDS.get(
        KEY, model_config.DEFAULT_METRIC_THRESHOLD
    )
    return GEval(
        name="Answer Correctness",
        criteria=CRITERIA,
        evaluation_params=REQUIRED_PARAMS,
        threshold=threshold,
        model=judge_model
    )
