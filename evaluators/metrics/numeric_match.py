import re

from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCaseParams


# ---------------------------------------------------------------------------
# Numeric match -- are the figures in the answer the same figures as in the
# Ground_Truth?
#
# No AI judge, no cost, instant. This is the most clear-cut check in the
# whole project: 364 is not 366, and no amount of judge opinion changes
# that. trustworthy-ai-eval-architecture-v3.md section 4.2 calls a failure
# here "conclusive".
#
# Why it earns its place next to the AI judge: an LLM judge reading
# "Revenue was $4.8M in 2023" against a Ground_Truth of "$4.2M" often marks
# it correct, because the sentence *sounds* right. Comparing the extracted
# numbers cannot be fooled that way.
#
# The rule: every number in the Ground_Truth must also appear in the
# answer. Extra numbers in the answer are allowed -- "Mars has 2 moons,
# discovered in 1877" still matches a Ground_Truth of "Mars has 2 moons",
# because 1877 is additional detail, not a contradiction.
#
# Known limitation: numbers written as words are not detected. An answer of
# "Mars has two moons" scores 0 against a Ground_Truth of "2 moons" even
# though it is correct. That is why this metric only applies to rows whose
# Ground_Truth actually contains a digit, and why it is one signal among
# several rather than the only one.
# ---------------------------------------------------------------------------

KEY = "numeric_match"
TIER = "deterministic"
REQUIRED_PARAMS = [
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.EXPECTED_OUTPUT,
]

# Matches whole numbers and decimals: 2, 42, 4.2, 1877
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
        # 1.0 = every number in the Ground_Truth must be present. A partial
        # match is still a failure: getting one figure right and another
        # wrong is a wrong answer.
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
        # Nothing to await -- there is no network or judge call here.
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self):
        return bool(self.success)

    @property
    def __name__(self):
        return "Numeric Match"


def build(judge_model):
    # judge_model is accepted and ignored -- metric_manager calls every
    # module's build() the same way, but this metric needs no judge.
    return NumericMatchMetric()
