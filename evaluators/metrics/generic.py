import deepeval.metrics as deepeval_metrics_module
from deepeval.test_case import LLMTestCaseParams

from config import settings


def _is_field_available(param, test_case):
    if param in (LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT):
        return True
    if param == LLMTestCaseParams.EXPECTED_OUTPUT:
        return bool(test_case.get("expected_answer"))
    if param in (LLMTestCaseParams.CONTEXT, LLMTestCaseParams.RETRIEVAL_CONTEXT):
        return bool(test_case.get("context"))
    return False


class GenericMetricModule:
    TIER = "llm_judge_calibrated"

    def __init__(self, key, metric_class):
        self.key = key
        self.metric_class = metric_class
        self.required_params = metric_class._required_params

    def is_applicable(self, test_case):
        unsuitable_reason = settings.KNOWN_UNSUITABLE_METRICS.get(self.key)
        if unsuitable_reason:
            return False, f"excluded: {unsuitable_reason}"

        missing = []
        for required_field in self.required_params:
            if not _is_field_available(required_field, test_case):
                missing.append(required_field.value)
        if missing:
            return False, f"missing required test-case data: {', '.join(missing)}"
        return True, None

    def build(self, judge_model):
        threshold = settings.METRIC_THRESHOLDS.get(
            self.key, settings.DEFAULT_METRIC_THRESHOLD
        )
        return self.metric_class(model=judge_model, threshold=threshold)


def build_generic_modules(metric_names):
    modules = {}
    for name in metric_names:
        metric_class = getattr(deepeval_metrics_module, name)
        modules[name] = GenericMetricModule(name, metric_class)
    return modules
