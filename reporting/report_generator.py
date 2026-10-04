import pandas as pd
from pathlib import Path
from datetime import datetime

from config import model_config
from config.generator_resolver import get_generator_provider
from config.judge_model_resolver import get_judge_provider, get_judge_independence


# Accumulator for a pytest run of the new pipeline (tests/test_evaluation_
# pipeline.py appends {"test_case": ..., "result": ...} here per test
# case; conftest.py reads it at session end). Separate from utils.report_
# generator's own results_data -- the two pytest suites (old and new)
# each keep their own list and produce their own report.
results_data = []


def _build_model_provenance_rows():
    """Which models produced and graded this run, and whether the judge is
    independent of the generator. get_judge_independence() previously had no
    caller anywhere in the codebase -- a score is not interpretable without
    knowing whether the model graded its own homework, so this is the first
    thing on every report, not a buried config detail."""

    return [
        {"Field": "Generator", "Value": f"{get_generator_provider()}/{model_config.GENERATOR_MODEL}"},
        {"Field": "Judge", "Value": f"{get_judge_provider()}/{model_config.JUDGE_MODEL}"},
        {"Field": "Judge Independent of Generator", "Value": get_judge_independence()},
    ]


# ---------------------------------------------------------------------------
# Reporting -- consumes the structured results already produced by the
# pipeline (orchestrator output + the original test case, and optionally a
# benchmark regression report from Step 9). It never executes a metric,
# never constructs an LLMTestCase, and never calls a judge model -- it only
# reads fields that already exist on the dicts handed to it.
#
# Reuses the existing project reporting convention from
# utils/report_generator.py: an execution_reports/<timestamp>/ folder,
# written with pandas.ExcelWriter(engine="openpyxl"), with a "Summary"
# sheet and the same "No test results were available" empty-case handling.
# A new function/module was used instead of extending that file directly,
# because its existing generate_excel_report() is hardcoded to a single
# flat "Hallucination %" column and the old flat result-dict shape -- the
# new pipeline produces a variable number of metrics (up to 48) per test
# case, which that shape cannot represent. tests/test_hallucination.py and
# utils/report_generator.py are left completely untouched.
# ---------------------------------------------------------------------------

def _context_to_text(context):
    if not context:
        return ""
    if isinstance(context, list):
        return " | ".join(str(item) for item in context)
    return str(context)


def _build_test_case_rows(test_case_results):

    rows = []

    for item in test_case_results:

        test_case = item["test_case"]
        result = item["result"]

        validation = result.get("validation") or {}
        classification_result = result.get("classification") or {}
        root_cause_result = result.get("root_cause_analysis") or {}

        root_cause_summaries = []
        recommended_actions = []

        test_case_level_diag = root_cause_result.get("test_case_level")
        if test_case_level_diag:
            root_cause_summaries.append(
                f"(validation) {test_case_level_diag['root_cause']} "
                f"({test_case_level_diag['confidence']})"
            )
            recommended_actions.append(test_case_level_diag["recommended_action"])

        for metric_key, diagnosis in (root_cause_result.get("metric_level") or {}).items():
            root_cause_summaries.append(
                f"{metric_key}: {diagnosis['root_cause']} ({diagnosis['confidence']})"
            )
            recommended_actions.append(diagnosis["recommended_action"])

        rows.append({
            "Category": test_case.get("category"),
            "Input": test_case.get("input"),
            "Context": _context_to_text(test_case.get("context")),
            "Ground_Truth": test_case.get("expected_answer") or "",
            "Actual_Response": validation.get("response") or "",
            "Validation_Valid": validation.get("valid"),
            "Validation_Failure_Type": validation.get("failure_type") or "",
            "Validation_Reason": validation.get("reason") or "",
            "Classification": classification_result.get("classification"),
            "Behavior_Signal": (classification_result.get("behavior_signal") or {}).get("signal", ""),
            "Root_Cause_Summary": "; ".join(root_cause_summaries),
            "Recommended_Actions": "; ".join(dict.fromkeys(recommended_actions)),
            "Human_Review_Reasons": "; ".join(result.get("human_review") or []),
        })

    return rows


def _build_metric_detail_rows(test_case_results):

    rows = []

    for item in test_case_results:

        test_case = item["test_case"]
        result = item["result"]

        evaluation = result.get("evaluation")

        if not evaluation:
            continue

        metric_diagnoses = (result.get("root_cause_analysis") or {}).get("metric_level") or {}

        for metric_key, metric_result in evaluation.items():

            diagnosis = metric_diagnoses.get(metric_key)

            rows.append({
                "Category": test_case.get("category"),
                "Input": test_case.get("input"),
                "Metric": metric_key,
                "Status": metric_result.get("status"),
                "Score": metric_result.get("score"),
                "Reason": metric_result.get("reason"),
                "Tier": metric_result.get("tier", ""),
                "Gating_Status": metric_result.get("gating_status", ""),
                "Root_Cause": diagnosis["root_cause"] if diagnosis else "",
                "Root_Cause_Confidence": diagnosis["confidence"] if diagnosis else "",
                "Recommended_Action": diagnosis["recommended_action"] if diagnosis else "",
            })

    return rows


def _build_summary_sections(test_case_results, benchmark_regression_report):

    classification_counts = {"VERIFIED": 0, "EVALUATED": 0, "UNVERIFIABLE": 0}
    metric_status_counts = {"passed": 0, "failed": 0, "error": 0, "not_applicable": 0}

    for item in test_case_results:

        result = item["result"]
        classification = (result.get("classification") or {}).get("classification")

        if classification in classification_counts:
            classification_counts[classification] += 1

        for metric_result in (result.get("evaluation") or {}).values():
            status = metric_result.get("status")
            if status in metric_status_counts:
                metric_status_counts[status] += 1

    overview_df = pd.DataFrame([
        {"Metric": "Total Test Cases", "Value": len(test_case_results)},
        {"Metric": "VERIFIED", "Value": classification_counts["VERIFIED"]},
        {"Metric": "EVALUATED", "Value": classification_counts["EVALUATED"]},
        {"Metric": "UNVERIFIABLE", "Value": classification_counts["UNVERIFIABLE"]},
    ])

    metric_status_df = pd.DataFrame([
        {"Status": status, "Count": count}
        for status, count in metric_status_counts.items()
    ])

    regression_summary_dfs = {}

    if benchmark_regression_report:
        for store_type in ("benchmark", "guardrail"):

            entries = benchmark_regression_report.get(store_type, [])

            outcome_counts = {}
            for entry in entries:
                for comparison in entry["metric_comparisons"]:
                    outcome = comparison["outcome"]
                    outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

            rows = [
                {"Outcome": "Total Entries", "Count": len(entries)},
                {"Outcome": "Entries With Regression", "Count": sum(
                    1 for entry in entries if entry["has_regression"]
                )},
            ]
            rows.extend({"Outcome": k, "Count": v} for k, v in outcome_counts.items())

            regression_summary_dfs[store_type] = pd.DataFrame(rows)

    return overview_df, metric_status_df, regression_summary_dfs


def _build_regression_detail_rows(entries):

    rows = []

    for entry in entries:
        for comparison in entry["metric_comparisons"]:
            rows.append({
                "Entry_ID": entry["entry_id"],
                "Input": entry["input"],
                **comparison
            })

    return rows


def generate_report(test_case_results, benchmark_regression_report=None, output_dir=None):
    """test_case_results: list of {"test_case": <dataset row dict>,
    "result": <orchestrator.evaluation_orchestrator.run_test_case output>}.

    benchmark_regression_report: optional output of
    benchmark.regression_checker.run_regression_check().

    Purely reads already-computed fields -- no metric execution, no judge
    calls. Returns the path to the generated Excel report.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    report_folder = Path(output_dir) if output_dir else Path("execution_reports") / timestamp
    report_folder.mkdir(parents=True, exist_ok=True)

    excel_path = report_folder / "evaluation_report.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        model_provenance_df = pd.DataFrame(_build_model_provenance_rows())
        model_provenance_df.to_excel(writer, sheet_name="Summary", index=False, startrow=0)
        next_row = len(model_provenance_df) + 3

        if not test_case_results:
            pd.DataFrame([{
                "Status": "No test results were available"
            }]).to_excel(writer, sheet_name="Summary", index=False, startrow=next_row)

        else:
            pd.DataFrame(_build_test_case_rows(test_case_results)).to_excel(
                writer, sheet_name="Test Cases", index=False
            )

            pd.DataFrame(_build_metric_detail_rows(test_case_results)).to_excel(
                writer, sheet_name="Metric Details", index=False
            )

            overview_df, metric_status_df, regression_summary_dfs = _build_summary_sections(
                test_case_results, benchmark_regression_report
            )

            overview_df.to_excel(writer, sheet_name="Summary", index=False, startrow=next_row)
            next_row += len(overview_df) + 3

            metric_status_df.to_excel(writer, sheet_name="Summary", index=False, startrow=next_row)
            next_row += len(metric_status_df) + 3

            for store_type, df in regression_summary_dfs.items():
                pd.DataFrame([{"Section": f"{store_type.capitalize()} Regression Summary"}]).to_excel(
                    writer, sheet_name="Summary", index=False, header=False, startrow=next_row
                )
                next_row += 2
                df.to_excel(writer, sheet_name="Summary", index=False, startrow=next_row)
                next_row += len(df) + 3

            if benchmark_regression_report:
                for store_type in ("benchmark", "guardrail"):
                    detail_rows = _build_regression_detail_rows(
                        benchmark_regression_report.get(store_type, [])
                    )
                    if detail_rows:
                        pd.DataFrame(detail_rows).to_excel(
                            writer,
                            sheet_name=f"{store_type.capitalize()} Regression",
                            index=False
                        )

    print(f"\nEvaluation Report Generated: {excel_path}")

    return str(excel_path)


def _build_comparison_test_case_rows(comparison_report):

    rows = []

    for test_case in comparison_report.get("test_cases", []):

        metric_summary = "; ".join(
            f"{c['metric']}: {c['outcome']}" for c in test_case.get("metric_comparisons", [])
        )

        classification_change = test_case.get("classification_change")
        classification_summary = (
            f"{classification_change['previous_classification']} -> "
            f"{classification_change['current_classification']}"
            if classification_change else ""
        )

        root_cause_summary = "; ".join(
            f"{c['metric']}: {c['previous_root_cause']} -> {c['current_root_cause']}"
            for c in test_case.get("root_cause_changes", [])
        )

        rows.append({
            "Category": test_case.get("category"),
            "Input": test_case.get("input"),
            "Previous_Input": test_case.get("previous_input"),
            "Previous_Prompt_Version": test_case.get("previous_prompt_version"),
            "Current_Prompt_Version": test_case.get("current_prompt_version"),
            "Outcome": test_case.get("outcome"),
            "Metric_Changes": metric_summary,
            "Classification_Change": classification_summary,
            "Root_Cause_Changes": root_cause_summary,
        })

    return rows


def generate_comparison_report(comparison_report, output_dir=None):
    """comparison_report: output of either
    run_comparison.comparator.compare_runs, or
    run_comparison.fair_comparison.compare_runs_with_condition (which adds
    comparison_condition/condition_metadata/condition_warning on top).

    Purely reads already-computed fields -- no metric execution, no judge
    calls, same as generate_report. Returns the path to the generated
    Excel report.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    report_folder = Path(output_dir) if output_dir else Path("execution_reports") / timestamp
    report_folder.mkdir(parents=True, exist_ok=True)

    excel_path = report_folder / "comparison_report.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        info_rows = [
            {"Field": "Judge", "Value": f"{get_judge_provider()}/{model_config.JUDGE_MODEL}"},
            {"Field": "Judge Independent of Generator", "Value": get_judge_independence()},
            {"Field": "Total Compared Test Cases", "Value": len(comparison_report.get("test_cases", []))},
            {"Field": "Newly Evaluated", "Value": len(comparison_report.get("newly_evaluated", []))},
            {"Field": "No Longer Evaluated", "Value": len(comparison_report.get("no_longer_evaluated", []))},
        ]

        if "comparison_condition" in comparison_report:
            info_rows.append({"Field": "Comparison Condition", "Value": comparison_report["comparison_condition"]})
            condition_metadata = comparison_report.get("condition_metadata", {})
            for side in ("previous", "current"):
                side_metadata = condition_metadata.get(side, {})
                info_rows.append({
                    "Field": f"{side.capitalize()} Generator",
                    "Value": f"{side_metadata.get('generator_provider')}/{side_metadata.get('generator_model')}"
                })
            if comparison_report.get("condition_warning"):
                info_rows.append({"Field": "WARNING", "Value": comparison_report["condition_warning"]})

        pd.DataFrame(info_rows).to_excel(writer, sheet_name="Comparison Info", index=False)

        pd.DataFrame(_build_comparison_test_case_rows(comparison_report)).to_excel(
            writer, sheet_name="Test Case Comparisons", index=False
        )

        if comparison_report.get("newly_evaluated"):
            pd.DataFrame(comparison_report["newly_evaluated"]).to_excel(
                writer, sheet_name="Newly Evaluated", index=False
            )

        if comparison_report.get("no_longer_evaluated"):
            pd.DataFrame(comparison_report["no_longer_evaluated"]).to_excel(
                writer, sheet_name="No Longer Evaluated", index=False
            )

    print(f"\nComparison Report Generated: {excel_path}")

    return str(excel_path)
