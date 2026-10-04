from orchestrator.evaluation_orchestrator import run_test_case
from benchmark.benchmark_store import load_benchmark_store


def compare_metric(metric_key, baseline_metric, current_metric):

    if baseline_metric is None:
        return {
            "metric": metric_key,
            "outcome": "new_metric",
            "previous_status": None,
            "previous_score": None,
            "current_status": (current_metric or {}).get("status"),
            "current_score": (current_metric or {}).get("score"),
            "reason": (
                f"'{metric_key}' was not present in the approved baseline "
                "(the metric registry may have changed since promotion)."
            ),
        }

    if current_metric is None:
        return {
            "metric": metric_key,
            "outcome": "missing_metric",
            "previous_status": baseline_metric.get("status"),
            "previous_score": baseline_metric.get("score"),
            "current_status": None,
            "current_score": None,
            "reason": (
                f"'{metric_key}' was in the approved baseline but did not "
                "run this time (not_applicable now, or the registry "
                "changed)."
            ),
        }

    previous_status = baseline_metric.get("status")
    current_status = current_metric.get("status")
    previous_score = baseline_metric.get("score")
    current_score = current_metric.get("score")

    if previous_status == "passed" and current_status in ("failed", "error"):
        outcome = "regressed"
        reason = (
            f"PASS -> {current_status.upper()}: score changed from "
            f"{previous_score} to {current_score}."
        )
    elif previous_status in ("failed", "error") and current_status == "passed":
        outcome = "improved"
        reason = (
            f"{previous_status.upper()} -> PASS: score changed from "
            f"{previous_score} to {current_score}."
        )
    elif previous_status == current_status:
        outcome = "unchanged"
        reason = f"Status unchanged ({current_status})."
    else:
        outcome = "changed"
        reason = f"Status changed from {previous_status} to {current_status}."

    return {
        "metric": metric_key,
        "outcome": outcome,
        "previous_status": previous_status,
        "previous_score": previous_score,
        "current_status": current_status,
        "current_score": current_score,
        "reason": reason,
    }


def check_entry_regression(entry, current_result):

    baseline_evaluation = entry["baseline_result"]["evaluation"] or {}
    current_evaluation = (current_result or {}).get("evaluation") or {}

    all_metric_keys = sorted(set(baseline_evaluation.keys()) | set(current_evaluation.keys()))

    return [
        compare_metric(
            metric_key,
            baseline_evaluation.get(metric_key),
            current_evaluation.get(metric_key),
        )
        for metric_key in all_metric_keys
    ]


def run_regression_check(test_case_runner=None):
    """Loads every approved benchmark entry, re-runs its stored test case
    through the existing evaluation pipeline, and compares the result
    against the approved baseline. Results are grouped by test_case_type
    so guardrail regressions stay separately identifiable from normal
    benchmark regressions."""

    test_case_runner = test_case_runner or run_test_case

    entries = load_benchmark_store()

    report = {"benchmark": [], "guardrail": []}

    for entry in entries:

        test_case = entry["test_case"]
        current_result = test_case_runner(test_case)

        metric_comparisons = check_entry_regression(entry, current_result)

        entry_report = {
            "entry_id": entry["entry_id"],
            "input": test_case.get("input"),
            "metric_comparisons": metric_comparisons,
            "has_regression": any(
                comparison["outcome"] == "regressed"
                for comparison in metric_comparisons
            ),
        }

        report[entry["test_case_type"]].append(entry_report)

    return report


def get_regressions_only(regression_report):
    """Flattened list of just the metric comparisons that actually
    regressed, tagged with which store (benchmark/guardrail) they came
    from."""

    flattened = []

    for test_case_type, entries in regression_report.items():
        for entry in entries:
            for comparison in entry["metric_comparisons"]:
                if comparison["outcome"] == "regressed":
                    flattened.append({
                        "test_case_type": test_case_type,
                        "entry_id": entry["entry_id"],
                        "input": entry["input"],
                        **comparison,
                    })

    return flattened
