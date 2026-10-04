import json
from pathlib import Path
from datetime import datetime, timezone

from config import model_config
from orchestrator.evaluation_orchestrator import run_test_case as _run_test_case


def execute_dataset(dataset, test_case_runner=None):
    """Runs every test case in `dataset` through the existing evaluation
    pipeline and returns it in the {"test_case": ..., "result": ...} shape
    used by reporting and run comparison. This adds no new evaluation
    logic -- it only loops over the existing orchestrator entry point, so
    a dataset can be "executed again after an improvement" without
    hand-writing the loop each time."""

    test_case_runner = test_case_runner or _run_test_case

    return [
        {"test_case": test_case, "result": test_case_runner(test_case)}
        for test_case in dataset
    ]


def capture_run_metadata():
    """Snapshots which generator/judge provider+model were configured at
    the moment this is called. Used so fair-comparison condition
    classification (capability vs. assisted) doesn't depend on a human
    remembering to record this by hand."""

    return {
        "generator_provider": model_config.GENERATOR_PROVIDER,
        "generator_model": model_config.GENERATOR_MODEL,
        "judge_provider": model_config.JUDGE_PROVIDER,
        "judge_model": model_config.JUDGE_MODEL,
    }


def save_run(test_case_results, output_dir=None, metadata=None):
    """Persists a run's raw structured results + metadata as flat JSON,
    alongside (not replacing) the human-readable Excel report from
    reporting.report_generator, under the same execution_reports/
    <timestamp>/ convention. This is what "run metadata" for comparison
    purposes actually is -- no database, no new format.

    Generator/judge metadata is captured automatically (capture_run_
    metadata) and merged with any explicit `metadata` passed in, which
    takes precedence on key conflicts."""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    folder = Path(output_dir) if output_dir else Path("execution_reports") / timestamp
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / "run_data.json"

    full_metadata = {**capture_run_metadata(), **(metadata or {})}

    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "metadata": full_metadata,
        "test_case_results": test_case_results,
    }

    with open(path, "w", encoding="utf-8") as run_file:
        json.dump(payload, run_file, indent=2, default=str)

    return str(path)


def load_run(path):
    with open(path, "r", encoding="utf-8") as run_file:
        payload = json.load(run_file)
    return payload["test_case_results"]


def load_run_metadata(path):
    with open(path, "r", encoding="utf-8") as run_file:
        payload = json.load(run_file)
    return payload.get("metadata", {})
