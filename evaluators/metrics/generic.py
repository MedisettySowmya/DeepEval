import inspect

import deepeval.metrics as deepeval_metrics_module
from deepeval.test_case import LLMTestCaseParams

from config import model_config


# ---------------------------------------------------------------------------
# Fallback for every DeepEval metric that does NOT have a hand-written
# module of its own (see evaluators/metrics/hallucination.py etc. for those).
#
# Hand-writing one file per metric for all ~47 metrics in deepeval.metrics
# would be almost entirely repeated boilerplate for metrics this framework
# has no specific configuration for. Instead, this module discovers those
# remaining metric classes directly from the installed deepeval package and
# builds one GenericMetricModule instance per metric -- each instance is
# still its own self-contained object, owning its own class, required
# params, and applicability check; only the *code that builds them* is
# shared, not their behavior or state.
# ---------------------------------------------------------------------------

_BASE_METRIC_CLASS_NAMES = {"BaseMetric", "BaseConversationalMetric", "BaseArenaMetric"}
_NON_METRIC_CLASS_NAMES = {"DeepAcyclicGraph"}

_FIELD_AVAILABILITY_CHECKS = {
    LLMTestCaseParams.INPUT: lambda test_case: True,
    LLMTestCaseParams.ACTUAL_OUTPUT: lambda test_case: True,
    LLMTestCaseParams.EXPECTED_OUTPUT: lambda test_case: bool(test_case.get("expected_answer")),
    LLMTestCaseParams.CONTEXT: lambda test_case: bool(test_case.get("context")),
    # Previously hardcoded False, which made the four context metrics
    # (Faithfulness, ContextualRelevancy, ContextualPrecision,
    # ContextualRecall) permanently inapplicable even though the Context data
    # existed -- the single largest cause of the 40-of-48 never-applicable
    # count. metric_manager populates retrieval_context from the same Context
    # column, but only when it was actually sent to the model.
    LLMTestCaseParams.RETRIEVAL_CONTEXT: lambda test_case: bool(test_case.get("context")),
    LLMTestCaseParams.TOOLS_CALLED: lambda test_case: False,
    LLMTestCaseParams.EXPECTED_TOOLS: lambda test_case: False,
    LLMTestCaseParams.MCP_SERVERS: lambda test_case: False,
    LLMTestCaseParams.MCP_TOOLS_CALLED: lambda test_case: False,
    LLMTestCaseParams.MCP_RESOURCES_CALLED: lambda test_case: False,
    LLMTestCaseParams.MCP_PROMPTS_CALLED: lambda test_case: False,
}


def _is_field_available(param, test_case):
    check = _FIELD_AVAILABILITY_CHECKS.get(param)
    return check(test_case) if check else False


def discover_metric_classes():

    discovered = {}

    for name in dir(deepeval_metrics_module):

        if name.startswith("_"):
            continue

        if name in _BASE_METRIC_CLASS_NAMES or name in _NON_METRIC_CLASS_NAMES:
            continue

        obj = getattr(deepeval_metrics_module, name)

        if inspect.isclass(obj):
            discovered[name] = obj

    return discovered


def classify_test_case_type(metric_class):

    mro_names = {base.__name__ for base in metric_class.__mro__}

    if "BaseConversationalMetric" in mro_names:
        return "conversational"

    if "BaseArenaMetric" in mro_names:
        return "arena"

    if "multimodal_metrics" in (metric_class.__module__ or ""):
        return "multimodal"

    if "BaseMetric" in mro_names:
        return "llm"

    return "unknown"


def get_declared_required_params(metric_class):
    required = getattr(metric_class, "_required_params", None)
    return required if isinstance(required, list) else None


def get_required_constructor_args(metric_class):

    sig = inspect.signature(metric_class.__init__)
    params = list(sig.parameters.values())[1:]  # skip self

    return [
        p.name for p in params
        if p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def _infer_tier(metric_class):
    """Deterministic tier inference: every metric in the installed
    deepeval version is either a plain deterministic check (no `model`
    constructor parameter -- e.g. ExactMatchMetric, PatternMatchMetric) or
    an LLM-judge-based check (`model` parameter present). No metric
    currently discovered falls into a "statistical" (BLEU/ROUGE/embedding-
    similarity) tier, so that value is not assigned here, only reserved in
    metric_tiering.tier_classifier.VALID_TIERS for future metrics that
    might need it."""

    sig_params = inspect.signature(metric_class.__init__).parameters
    return "llm_judge_calibrated" if "model" in sig_params else "deterministic"


class GenericMetricModule:

    def __init__(self, key, metric_class, test_case_type, required_params, required_config):
        self.key = key
        self.metric_class = metric_class
        self.test_case_type = test_case_type
        self.required_params = required_params
        self.required_config = required_config
        self.tier = _infer_tier(metric_class)

    def is_applicable(self, test_case):

        unsuitable_reason = model_config.KNOWN_UNSUITABLE_METRICS.get(self.key)
        if unsuitable_reason:
            return False, f"excluded: {unsuitable_reason}"

        if self.test_case_type != "llm":
            return False, (
                f"requires a {self.test_case_type} test case; this "
                "framework only builds single-turn LLMTestCase"
            )

        if self.required_params is None:
            return False, (
                "requires custom per-instance construction (e.g. GEval "
                "criteria/evaluation_params, or a DAG) not defined for "
                "this framework"
            )

        missing = [
            p.value for p in self.required_params
            if not _is_field_available(p, test_case)
        ]
        if missing:
            return False, f"missing required test-case data: {', '.join(missing)}"

        if self.required_config:
            return False, (
                "missing required configuration: "
                f"{', '.join(self.required_config)}"
            )

        return True, None

    def build(self, judge_model):

        sig_params = inspect.signature(self.metric_class.__init__).parameters
        kwargs = {}

        if "threshold" in sig_params:
            kwargs["threshold"] = model_config.METRIC_THRESHOLDS.get(
                self.key, model_config.DEFAULT_METRIC_THRESHOLD
            )

        if "model" in sig_params:
            kwargs["model"] = judge_model

        return self.metric_class(**kwargs)


def build_generic_modules(exclude_keys=frozenset()):

    modules = {}

    for name, metric_class in discover_metric_classes().items():

        if name in exclude_keys:
            continue

        modules[name] = GenericMetricModule(
            key=name,
            metric_class=metric_class,
            test_case_type=classify_test_case_type(metric_class),
            required_params=get_declared_required_params(metric_class),
            required_config=get_required_constructor_args(metric_class)
        )

    return modules
