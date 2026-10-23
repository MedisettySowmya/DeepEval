import re

from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCaseParams

KEY = "numeric_match"
TIER = "deterministic"
REQUIRED_PARAMS = [
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.EXPECTED_OUTPUT,
]

_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")

def _numbers_in(text):
    """Every number in the text, as a set of floats.

    Commas are stripped first so "1,000" is read as one number (1000)
    rather than two (1 and 000). Floats rather than strings so that "2"
    and "2.0" count as the same number.
    """

    cleaned = str(text or "").replace(",", "")
    return {float(match) for match in _NUMBER_PATTERN.findall(cleaned)}

def is_applicable(test_case):
    reference = test_case.get("expected_answer")

    if not reference:
        return False, "missing required test-case data: expected_output"

    if not _numbers_in(reference):
        return False, (
            "the Ground_Truth contains no numbers, so there is nothing for a "
            "numeric comparison to check"
        )

    return True, None

class NumericMatchMetric(BaseMetric):
    """Compares the numbers in the answer against the numbers in the
    Ground_Truth. No LLM call, no judge model."""

    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def measure(self, test_case, *args, **kwargs):
        expected = _numbers_in(test_case.expected_output)
        actual = _numbers_in(test_case.actual_output)

        missing = sorted(expected - actual)
        found_count = len(expected) - len(missing)

        self.score = found_count / len(expected) if expected else 0.0
        self.success = not missing

        if missing:
            self.reason = (
                f"Figures {missing} from the Ground_Truth do not appear in the "
                f"answer. The answer's numbers were {sorted(actual) or 'none'}. "
                f"A wrong figure is conclusive -- no judge opinion changes it."
            )
        else:
            self.reason = (
                f"Every figure in the Ground_Truth ({sorted(expected)}) also "
                f"appears in the answer. No LLM judge was used for this check."
            )

        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self):
        return bool(self.success)

    @property
    def __name__(self):
        return "Numeric Match"

def build(judge_model):
    return NumericMatchMetric()
