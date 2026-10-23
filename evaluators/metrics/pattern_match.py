import re

from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCaseParams

KEY = "pattern_match"
TIER = "deterministic"
REQUIRED_PARAMS = [LLMTestCaseParams.ACTUAL_OUTPUT]

METADATA_KEY = "expected_pattern"

def is_applicable(test_case):
    if not test_case.get("expected_pattern"):
        return False, (
            "no Pattern given for this row (the Excel 'Pattern' column is "
            "empty), so there is no required format to check"
        )

    return True, None

class PatternMatchMetric(BaseMetric):
    """Checks the answer against a regular expression taken from the row.
    No LLM call, no judge model."""

    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def measure(self, test_case, *args, **kwargs):
        metadata = getattr(test_case, "additional_metadata", None) or {}
        pattern = metadata.get(METADATA_KEY)

        if not pattern:
            raise ValueError(
                "pattern_match ran without a pattern. The row's Excel "
                "'Pattern' cell was empty by the time the metric executed."
            )

        try:
            matched = re.search(pattern, str(test_case.actual_output or ""))
        except re.error as error:
            raise ValueError(
                f"The Pattern cell for this row is not a valid regular "
                f"expression: {pattern!r} ({error}). Fix the Excel cell -- "
                f"this is a dataset problem, not a model result."
            )

        self.score = 1.0 if matched else 0.0
        self.success = bool(matched)

        if matched:
            self.reason = (
                f"The answer contains the required pattern {pattern!r} "
                f"(matched text: {matched.group(0)!r}). No LLM judge was used."
            )
        else:
            self.reason = (
                f"The answer does not contain the required pattern "
                f"{pattern!r}. No LLM judge was used -- this is a plain text "
                f"format check."
            )

        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self):
        return bool(self.success)

    @property
    def __name__(self):
        return "Pattern Match"

def build(judge_model):
    return PatternMatchMetric()
