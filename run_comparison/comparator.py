# ---------------------------------------------------------------------------
# Ordinary run-to-run comparison -- distinct from
# benchmark.regression_checker.run_regression_check (Step 9):
#
#   Step 9  = a fixed, human-approved baseline set (benchmark/data/
#             benchmark_store.json), checked for regressions on demand,
#             permanent until someone re-promotes it.
#
#   Step 11 (this module) = an ad hoc diff between ANY two runs of ANY
#             dataset/subset -- e.g. the same dataset before and after
#             editing a row's prompt/context to fix a failure. Nothing here
#             is promoted or stored permanently; it's a one-off comparison.
#
# Per-metric comparison logic (regressed/improved/unchanged/new_metric/
# missing_metric) is reused directly from benchmark.regression_checker --
# not reimplemented -- since the semantics are identical.
# ---------------------------------------------------------------------------

from benchmark.regression_checker import compare_metric
from prompt_versioning.prompt_version_store import get_prompt_history


def _find_prompt_version(category, position, prompt_text):
    """Looks up which recorded prompt version corresponds to a specific
    piece of prompt text for a given test-case slot -- None if it was
    never recorded (e.g. this test case never actually ran through
    build_prompt)."""

    slot_key = f"{category}:{position}"

    for entry in get_prompt_history(slot_key):
        if entry["prompt_text"] == prompt_text:
            return entry["version"]

    return None


def _default_match_key(test_case_results):
    """Matches test cases by category + position within that category,
    not by input text. This is deliberate: the common case this module
    exists for is a tester editing a row's Input/Context in place to fix
    a failure and re-running -- the edited row should still be recognized
    as "the same test case, changed" so its before/after can be compared,
    even though its Input text is now different. A row added or removed
    shifts the available positions in its category, which is exactly what
    produces "newly evaluated" / "no longer evaluated" below.

    A different matching strategy can be supplied via compare_runs's
    match_key_fn if this default doesn't fit a particular comparison."""

    counters = {}
    keyed = {}

    for item in test_case_results:
        category = item["test_case"].get("category")
        position = counters.get(category, 0)
        counters[category] = position + 1
        keyed[(category, position)] = item

    return keyed


def _diff_classification(previous_result, current_result):

    previous_classification = ((previous_result or {}).get("classification") or {}).get("classification")
    current_classification = ((current_result or {}).get("classification") or {}).get("classification")

    if previous_classification == current_classification:
        return None

    return {
        "previous_classification": previous_classification,
        "current_classification": current_classification,
    }


def _diff_root_causes(previous_result, current_result):

    previous_metric_level = ((previous_result or {}).get("root_cause_analysis") or {}).get("metric_level") or {}
    current_metric_level = ((current_result or {}).get("root_cause_analysis") or {}).get("metric_level") or {}

    all_metric_keys = sorted(set(previous_metric_level.keys()) | set(current_metric_level.keys()))

    changes = []

    for metric_key in all_metric_keys:

        previous_diagnosis = previous_metric_level.get(metric_key)
        current_diagnosis = current_metric_level.get(metric_key)

        previous_cause = previous_diagnosis["root_cause"] if previous_diagnosis else None
        current_cause = current_diagnosis["root_cause"] if current_diagnosis else None

        if previous_cause != current_cause:
            changes.append({
                "metric": metric_key,
                "previous_root_cause": previous_cause,
                "current_root_cause": current_cause,
            })

    return changes


def _overall_outcome(metric_comparisons):

    outcomes = {comparison["outcome"] for comparison in metric_comparisons}

    if not outcomes or outcomes == {"unchanged"}:
        return "unchanged"

    has_regressed = "regressed" in outcomes
    has_improved = "improved" in outcomes

    if has_regressed and has_improved:
        return "mixed"
    if has_regressed:
        return "regressed"
    if has_improved:
        return "improved"

    return "unchanged"


def compare_runs(previous_run, current_run, match_key_fn=None):
    """previous_run / current_run: lists of {"test_case": ..., "result": ...}
    -- the same shape run_comparison.run_store.execute_dataset produces and
    reporting.report_generator.generate_report consumes.

    Deterministic only: every comparison here is a direct field comparison
    on already-computed data. No metric is executed, no judge model is
    called, and nothing is modified -- this only reads and compares.

    Returns:
        {
            "newly_evaluated": [{"category", "input"}, ...],
            "no_longer_evaluated": [{"category", "input"}, ...],
            "test_cases": [
                {
                    "category", "input", "previous_input",
                    "outcome": "improved"|"regressed"|"unchanged"|"mixed",
                    "metric_comparisons": [...],   # reused from Step 9
                    "classification_change": {...} | None,
                    "root_cause_changes": [...],
                },
                ...
            ],
        }
    """

    match_key_fn = match_key_fn or _default_match_key

    previous_keyed = match_key_fn(previous_run)
    current_keyed = match_key_fn(current_run)

    previous_keys = set(previous_keyed.keys())
    current_keys = set(current_keyed.keys())

    newly_evaluated = [
        {
            "category": current_keyed[key]["test_case"].get("category"),
            "input": current_keyed[key]["test_case"].get("input"),
        }
        for key in sorted(current_keys - previous_keys, key=str)
    ]

    no_longer_evaluated = [
        {
            "category": previous_keyed[key]["test_case"].get("category"),
            "input": previous_keyed[key]["test_case"].get("input"),
        }
        for key in sorted(previous_keys - current_keys, key=str)
    ]

    test_case_reports = []

    for key in sorted(previous_keys & current_keys, key=str):

        previous_item = previous_keyed[key]
        current_item = current_keyed[key]

        previous_evaluation = previous_item["result"].get("evaluation") or {}
        current_evaluation = current_item["result"].get("evaluation") or {}

        all_metric_keys = sorted(set(previous_evaluation.keys()) | set(current_evaluation.keys()))

        metric_comparisons = [
            compare_metric(
                metric_key,
                previous_evaluation.get(metric_key),
                current_evaluation.get(metric_key),
            )
            for metric_key in all_metric_keys
        ]

        category, position = key

        test_case_reports.append({
            "category": current_item["test_case"].get("category"),
            "input": current_item["test_case"].get("input"),
            "previous_input": previous_item["test_case"].get("input"),
            "previous_prompt_version": _find_prompt_version(
                category, position, previous_item["test_case"].get("input")
            ),
            "current_prompt_version": _find_prompt_version(
                category, position, current_item["test_case"].get("input")
            ),
            "outcome": _overall_outcome(metric_comparisons),
            "metric_comparisons": metric_comparisons,
            "classification_change": _diff_classification(previous_item["result"], current_item["result"]),
            "root_cause_changes": _diff_root_causes(previous_item["result"], current_item["result"]),
        })

    return {
        "newly_evaluated": newly_evaluated,
        "no_longer_evaluated": no_longer_evaluated,
        "test_cases": test_case_reports,
    }
