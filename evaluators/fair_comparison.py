
from evaluators.run_store import load_run, load_run_metadata
from evaluators.comparator import compare_runs

def classify_comparison_condition(previous_metadata, current_metadata, prompt_changed):

    generator_changed = (
        (previous_metadata or {}).get("generator_provider") != (current_metadata or {}).get("generator_provider")
        or (previous_metadata or {}).get("generator_model") != (current_metadata or {}).get("generator_model")
    )

    if generator_changed and prompt_changed:
        return "mixed"

    if generator_changed:
        return "capability"

    if prompt_changed:
        return "assisted"

    return "unchanged"

def compare_runs_with_condition(previous_run_path, current_run_path, compare_fn=None):
    """Loads two saved runs (including their captured metadata), runs the
    ordinary comparison (Step 11), and classifies the overall comparison
    condition on top. A `mixed` result carries an explicit warning so it's
    never mistaken for a clean capability or assisted signal -- this is
    the enforcement half of the Fair-Comparison Controller."""

    compare_fn = compare_fn or compare_runs

    previous_run = load_run(previous_run_path)
    current_run = load_run(current_run_path)

    previous_metadata = load_run_metadata(previous_run_path)
    current_metadata = load_run_metadata(current_run_path)

    report = compare_fn(previous_run, current_run)

    prompt_changed = any(
        test_case.get("previous_prompt_version") != test_case.get("current_prompt_version")
        for test_case in report["test_cases"]
    )

    condition = classify_comparison_condition(previous_metadata, current_metadata, prompt_changed)

    report["comparison_condition"] = condition
    report["condition_metadata"] = {
        "previous": previous_metadata,
        "current": current_metadata,
    }

    if condition == "mixed":
        report["condition_warning"] = (
            "Both the generator model/provider AND the prompt changed "
            "between these two runs -- this comparison cannot cleanly "
            "attribute score changes to either the model or the prompt. "
            "Treat any improved/regressed outcome here as unreliable for "
            "that purpose."
        )

    return report
