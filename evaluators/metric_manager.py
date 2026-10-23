from deepeval.test_case import LLMTestCase

from config import settings
from models.model_client import create_judge

from evaluators.metrics import (
    hallucination,
    answer_correctness,
    answer_relevancy,
    exact_match,
    numeric_match,
    pattern_match,
    semantic_similarity,
    bias,
    toxicity,
    pii_leakage,
)
from evaluators.metrics.generic import build_generic_modules
from evaluators.metrics.configured import build_modules
from evaluators.metrics.additional import build_modules as build_additional_modules
from evaluators import metric_registry
from dataset_checks.dataset_loader import CONTEXT_ROLE_ADVERSARIAL

_CORE_MODULES = {
    hallucination.KEY: hallucination,
    answer_correctness.KEY: answer_correctness,
    answer_relevancy.KEY: answer_relevancy,
    exact_match.KEY: exact_match,
    numeric_match.KEY: numeric_match,
    pattern_match.KEY: pattern_match,
    semantic_similarity.KEY: semantic_similarity,
    bias.KEY: bias,
    toxicity.KEY: toxicity,
    pii_leakage.KEY: pii_leakage,
}

_CORE_MODULES.update(build_modules())
_CORE_MODULES.update(build_additional_modules())
_GENERIC_METRIC_NAMES = (
    set(metric_registry.REGISTRY) | set(metric_registry.DATASET_CHECK_METRICS)
) - set(_CORE_MODULES)
_GENERIC_MODULES = build_generic_modules(sorted(_GENERIC_METRIC_NAMES))

ALL_METRIC_MODULES = {**_CORE_MODULES, **_GENERIC_MODULES}

STAGE_RAN = "ran"
STAGE_SKIPPED_BY_GATE = "skipped_by_gate"
STAGE_NOT_RELEVANT = "not_relevant"

def get_available_metrics():
    """1. Discover/register the available metric implementations."""
    return list(ALL_METRIC_MODULES.keys())

def build_metric(module, judge_model, test_case):
    if hasattr(module, "build_for_case"):
        return module.build_for_case(judge_model, test_case)
    return module.build(judge_model)

def is_metric_available(key):
    """2. Check whether a requested metric is available."""
    return key in ALL_METRIC_MODULES

def check_applicability(key, test_case, context_in_prompt=True):
    """3. Is this metric both relevant to this row and supplied with the data
    it needs? Pure check -- never calls metric.measure()."""

    module = ALL_METRIC_MODULES.get(key)

    if module is None:
        return False, f"metric '{key}' is not registered"

    relevant, reason = metric_registry.check_relevance(
        key, test_case, context_in_prompt
    )
    if not relevant:
        return False, reason

    return module.is_applicable(test_case)

def get_applicable_metrics(test_case, context_in_prompt=True):
    """4. Return only the metrics applicable to that test case."""
    applicable_metrics = []
    for metric_key in ALL_METRIC_MODULES:
        applicable, reason = check_applicability(metric_key, test_case, context_in_prompt)
        if applicable:
            applicable_metrics.append(metric_key)
    return applicable_metrics

def get_applicability_report(test_case, context_in_prompt=True):
    """Applicability for every registered metric, without executing any of
    them -- no judge calls made. This is what a console can show before a run
    is paid for."""

    report = {}

    for key in ALL_METRIC_MODULES:
        applicable, reason = check_applicability(key, test_case, context_in_prompt)
        report[key] = {
            "applicable": applicable,
            "reason": reason,
            "stage": metric_registry.get_stage(key),
        }

    return report

def _build_llm_test_case(test_case, prompt_text, actual_output, context_in_prompt):
    """Context is only handed to the metrics when it was actually sent to the
    model. Populating `context`/`retrieval_context` for a prompt that never
    included them is what made the model gradeable against evidence it never
    saw.

    Both fields are set from the same source: `context` is what
    HallucinationMetric reads, `retrieval_context` is what Faithfulness and
    the Contextual* family read. Previously retrieval_context was hardcoded
    unavailable, which is why four metrics could never apply.
    """

    context = test_case.get("context") or None

    return LLMTestCase(
        input=prompt_text,
        actual_output=actual_output,
        context=context if context_in_prompt else None,
        retrieval_context=context if context_in_prompt else None,
        expected_output=test_case.get("expected_answer"),
        additional_metadata={
            pattern_match.METADATA_KEY: test_case.get("expected_pattern"),
        },
    )

def _interpret(metric_key, metric, test_case):
    """Turns a measured score into passed/failed.

    `metric.success` applies DeepEval's own threshold rule, which is right for
    most metrics and backwards for two. HallucinationMetric and
    FaithfulnessMetric score how much the answer *contradicts* the Context --
    so on a deliberately false Context, a high score means the response
    rejected the false claim, which is the wanted behaviour. Trusting
    `metric.success` there fails a model for being correct.

    The inversion only applies when the Context is known to be adversarial.
    On a supporting Context the ordinary reading is correct.
    """

    inverted = (
        metric_registry.is_inverted(metric_key)
        and test_case.get("context_role") == CONTEXT_ROLE_ADVERSARIAL
    )

    if not inverted:
        return ("passed" if metric.success else "failed"), None

    threshold = settings.RESISTANCE_PASS_THRESHOLD
    score = metric.score if metric.score is not None else 0.0
    if metric_key == "FaithfulnessMetric" and metric.score is not None:
        score = 1.0 - metric.score

    return (
        ("passed" if score >= threshold else "failed"),
        (
            f"inverted reading: the Context for this row is adversarial, so a "
            f"high contradiction score means the response resisted it "
            f"(score {score} vs resistance threshold {threshold})"
        ),
    )

def interpret_metric_result(metric_key, metric, test_case):
    """Public entry point to the same passed/failed reading _evaluate uses,
    for callers outside this module that re-run a metric and must apply the
    identical inversion rule -- currently evaluators.calibration_runner,
    which re-runs a metric during calibration and would otherwise compare a
    human's verdict against DeepEval's raw, uninverted metric.success. Returns
    (status, interpretation) -- see _interpret."""

    return _interpret(metric_key, metric, test_case)

def _evaluate(test_case, prompt, actual_output, context_in_prompt=True):
    """5. Execute the relevant metric modules, stage by stage.

    `prompt` accepts either the dict returned by prompts.prompt_builder.
    build_prompt or a plain string (the older shape), so existing callers keep
    working.

    Flow per metric:
      relevant to this row (metric_registry)
        -> does the test case have the required inputs (is_applicable)
             No  -> NOT_APPLICABLE, metric.measure() is never called
             Yes -> build + measure via DeepEval
                      success -> passed / failed (inverted where registered)
                      unexpected runtime exception -> error

    Gating: a failed gate stage suppresses the stages after it, except Stage 5
    (safety), which runs on any response that exists. A gate only suppresses
    metrics whose own inputs the failure bears on -- the dataset checks are
    not here at all, they run pre-flight.
    """

    test_case = dict(test_case)
    options = dict(test_case.get("metric_options") or {})
    instruction = prompt.get("instruction", "") if isinstance(prompt, dict) else ""
    options["PromptAlignmentMetric"] = {
        "prompt_instructions": [instruction] if instruction else []
    }
    test_case["metric_options"] = options
    options["InstructionPriority"] = {
        **settings.METRIC_OPTIONS.get("InstructionPriority", {}),
        **options.get("InstructionPriority", {}),
        "trusted_instructions": [instruction] if instruction else [],
    }

    if isinstance(prompt, dict):
        prompt_text = prompt.get("text") or ""
        context_in_prompt = prompt.get("context_in_prompt", context_in_prompt)
    else:
        prompt_text = prompt or ""

    llm_test_case = _build_llm_test_case(
        test_case, prompt_text, actual_output, context_in_prompt
    )

    judge_model = None
    results = {}
    stage_outcomes = {}
    gate_failed_at = None

    for stage in metric_registry.stages_in_order():

        stage_gated_out = (
            gate_failed_at is not None and stage not in metric_registry.UNGATED_STAGES
        )

        stage_outcomes[stage] = {
            "name": metric_registry.STAGE_NAMES.get(stage, str(stage)),
            "is_gate": stage in metric_registry.GATE_STAGES,
            "outcome": STAGE_SKIPPED_BY_GATE if stage_gated_out else STAGE_RAN,
            "gated_by_stage": gate_failed_at if stage_gated_out else None,
        }

        stage_had_failure = False
        stage_executed_any = False

        for metric_key in sorted(metric_registry.metrics_for_stage(stage)):

            module = ALL_METRIC_MODULES.get(metric_key)

            if module is None:
                results[metric_key] = {
                    "status": "not_applicable",
                    "stage": stage,
                    "reason": f"metric '{metric_key}' is not registered",
                }
                continue

            if stage_gated_out:
                results[metric_key] = {
                    "status": "not_applicable",
                    "stage": stage,
                    "reason": (
                        f"skipped: stage {gate_failed_at} "
                        f"({metric_registry.STAGE_NAMES.get(gate_failed_at)}) "
                        f"failed, so this metric's question is no longer "
                        f"meaningful for this row"
                    ),
                }
                continue

            applicable, reason = check_applicability(
                metric_key, test_case, context_in_prompt
            )
            if not applicable:
                results[metric_key] = {
                    "status": "not_applicable",
                    "stage": stage,
                    "reason": reason,
                }
                continue

            try:
                tier = getattr(module, "TIER", None) or getattr(module, "tier", None)
                if judge_model is None and tier not in ("deterministic", "statistical"):
                    judge_model = create_judge()

                metric = build_metric(module, judge_model, test_case)
                metric.measure(llm_test_case)

                status, interpretation = _interpret(metric_key, metric, test_case)

                stage_executed_any = True
                if status == "failed":
                    stage_had_failure = True

                results[metric_key] = {
                    "status": status,
                    "stage": stage,
                    "score": metric.score,
                    "reason": metric.reason,
                    "interpretation": interpretation,
                    "inverted": metric_registry.is_inverted(metric_key),
                }

            except Exception as error:
                results[metric_key] = {
                    "status": "error",
                    "stage": stage,
                    "reason": str(error),
                }

        if (
            gate_failed_at is None
            and stage in metric_registry.GATE_STAGES
            and stage_executed_any
            and stage_had_failure
        ):
            gate_failed_at = stage

    return results, stage_outcomes

def evaluate_test_case(test_case, prompt, actual_output, context_in_prompt=True):
    """Metric results only -- the original return shape, so every existing
    caller (metric_tiering, root_cause_analysis, reporting, human_review,
    comparator) keeps iterating a dict containing nothing but metric keys."""

    results, _ = _evaluate(test_case, prompt, actual_output, context_in_prompt)
    return results

def evaluate_test_case_detailed(test_case, prompt, actual_output, context_in_prompt=True):
    """Metric results plus the per-stage outcome, for callers that want to show
    which stage gated which metrics:

        {"metrics": {...}, "stages": {stage: {...}}}
    """

    results, stage_outcomes = _evaluate(
        test_case, prompt, actual_output, context_in_prompt
    )

    return {"metrics": results, "stages": stage_outcomes}
