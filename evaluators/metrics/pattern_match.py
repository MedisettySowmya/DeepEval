import re

from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCaseParams


# ---------------------------------------------------------------------------
# Pattern match -- does the answer contain a required shape of text?
#
# No AI judge, no cost, instant. The pattern comes from the Excel dataset's
# optional "Pattern" column, so a tester can add a format requirement to any
# row without touching code:
#
#   Pattern              Means "the answer must contain..."
#   -------------------  ------------------------------------------------
#   \d{4}                a 4-digit number (e.g. a year)
#   \d{4}-\d{2}-\d{2}    a date like 2026-10-04
#   Canberra             the literal word Canberra
#   (?i)canberra         the word canberra, ignoring upper/lower case
#   ^\{.*\}$             an answer that is entirely a JSON-looking object
#
# The pattern is a regular expression and is matched with "search", i.e. it
# must appear SOMEWHERE in the answer -- it does not have to match the whole
# answer. Use ^ and $ in the pattern if you want the whole answer to match.
#
# Why a hand-written metric instead of DeepEval's own PatternMatchMetric:
# DeepEval's version needs the pattern passed in when the metric object is
# built, but this project builds metrics via build(judge_model), which does
# not see the dataset row -- so a per-row pattern could never reach it.
# DeepEval's PatternMatchMetric therefore stays registered-but-unconfigured
# (it reports "missing required configuration: pattern"), and this module
# does the real work, reading each row's own pattern at measure time.
#
# A bad regex (a typo in the Excel cell) deliberately raises rather than
# scoring 0. metric_manager records it as status "error" with the message,
# which the tiering then excludes from gating -- a broken pattern is a data
# problem to fix, never a verdict about the model. That is premise 4 of
# trustworthy-ai-eval-architecture-v3.md: a missing precondition must be
# loud, never a score.
# ---------------------------------------------------------------------------

KEY = "pattern_match"
TIER = "deterministic"
REQUIRED_PARAMS = [LLMTestCaseParams.ACTUAL_OUTPUT]

# Key under which metric_manager puts the row's pattern on the LLMTestCase.
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
        # 1.0 because this is all-or-nothing: the required shape is either
        # present in the answer or it is not.
        self.threshold = threshold

    def measure(self, test_case, *args, **kwargs):
        metadata = getattr(test_case, "additional_metadata", None) or {}
        pattern = metadata.get(METADATA_KEY)

        if not pattern:
            # Should never happen -- is_applicable() already filtered these
            # out -- but raising beats silently scoring 0 if it ever does.
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
        # Nothing to await -- there is no network or judge call here.
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self):
        return bool(self.success)

    @property
    def __name__(self):
        return "Pattern Match"


def build(judge_model):
    # judge_model is accepted and ignored -- metric_manager calls every
    # module's build() the same way, but this metric needs no judge.
    return PatternMatchMetric()
