import pytest

from config import model_config
from config.judge_model_resolver import load_model
from utils.dataset_loader import load_all_datasets
from orchestrator.evaluation_orchestrator import run_test_case
from reporting.report_generator import results_data


dataset = load_all_datasets(
    "datasets/hallucination_dataset_with_types.xlsx"
)


def test_judge_model_is_available():
    """Runs before the parametrized pipeline tests below. If the
    configured judge model can't even be constructed, every judge-based
    metric in every test case will individually error out and -- by
    metric_tiering's own design -- get downgraded to gating_status
    "not_applicable" (errors are never "trusted"), which means the
    parametrized tests below would report a false "passed" with zero
    metrics actually having evaluated anything. This makes that failure
    loud and immediate instead of letting it hide across 48 silent rows."""

    try:
        load_model()
    except Exception as error:
        pytest.fail(
            f"Configured judge model (JUDGE_PROVIDER="
            f"{model_config.JUDGE_PROVIDER!r}, JUDGE_MODEL="
            f"{model_config.JUDGE_MODEL!r}) could not be constructed: "
            f"{error}. Every metric below will otherwise silently error "
            "and be excluded from gating -- a 'passed' result from this "
            "suite cannot be trusted until this is fixed."
        )


@pytest.mark.parametrize("test_case", dataset)
def test_pipeline(test_case):

    result = run_test_case(test_case)

    results_data.append({"test_case": test_case, "result": result})

    classification = (result.get("classification") or {}).get("classification")

    print("\n===================================")
    print("Category:", test_case.get("category"))
    print("Input:", test_case.get("input"))
    print("Classification:", classification)
    print("Human Review:", result.get("human_review"))
    print("===================================\n")

    validation = result["validation"]

    assert validation["valid"], (
        f"Generation failed for input {test_case.get('input')!r}: "
        f"{validation.get('failure_type')} -- {validation.get('reason')}"
    )

    evaluation = result.get("evaluation") or {}

    # Only a "trusted" verdict (per metric_tiering) gates this test.
    # advisory/not_applicable/requires_human_review results are surfaced
    # in the report but never fail the test on their own -- that
    # distinction is the entire point of Metric Tiering.
    trusted_failures = [
        f"{metric_key} (status={metric_result.get('status')}, score={metric_result.get('score')})"
        for metric_key, metric_result in evaluation.items()
        if metric_result.get("gating_status") == "trusted"
        and metric_result.get("status") in ("failed", "error")
    ]

    assert not trusted_failures, (
        f"Trusted metric(s) failed for input {test_case.get('input')!r}: "
        + "; ".join(trusted_failures)
    )
