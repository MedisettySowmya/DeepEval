import json
import os
import uuid
from datetime import datetime, timezone

import pandas as pd

_JUDGMENTS_PATH = os.path.join(os.path.dirname(__file__), "data", "calibration_judgments.xlsx")

VALID_VERDICTS = {"pass", "fail"}

_COLUMNS = [
    "entry_id", "recorded_at", "recorded_by", "metric_key", "category",
    "row_id", "input", "context", "ground_truth", "profile", "context_role",
    "actual_output", "human_verdict", "notes", "test_case_json",
]

class CalibrationError(Exception):
    pass

def _ensure_file():
    os.makedirs(os.path.dirname(_JUDGMENTS_PATH), exist_ok=True)
    if not os.path.exists(_JUDGMENTS_PATH):
        pd.DataFrame(columns=_COLUMNS).to_excel(
            _JUDGMENTS_PATH, index=False, engine="openpyxl"
        )

def _context_to_text(context):
    if not context:
        return ""
    if isinstance(context, (list, tuple)):
        return "\n".join(str(item) for item in context if str(item).strip())
    return str(context)

def _clean(value):
    """pandas hands back float('nan') for a blank cell -- NaN is truthy in
    Python, so code downstream must never see it as a real value."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return value

def _row_from_entry(entry):

    test_case = entry["test_case"] or {}

    return {
        "entry_id": entry["entry_id"],
        "recorded_at": entry["recorded_at"],
        "recorded_by": entry.get("recorded_by") or "",
        "metric_key": entry["metric_key"],
        "category": entry.get("category") or test_case.get("category") or "",
        "row_id": test_case.get("id") or "",
        "input": test_case.get("input") or "",
        "context": _context_to_text(test_case.get("context")),
        "ground_truth": test_case.get("expected_answer") or "",
        "profile": test_case.get("profile") or "",
        "context_role": test_case.get("context_role") or "",
        "actual_output": entry.get("actual_output") or "",
        "human_verdict": entry["human_verdict"],
        "notes": entry.get("notes") or "",
        "test_case_json": json.dumps(test_case, default=str),
    }

def _entry_from_row(row):

    test_case_json = _clean(row.get("test_case_json"))
    test_case = None

    if isinstance(test_case_json, str) and test_case_json.strip():
        try:
            test_case = json.loads(test_case_json)
        except json.JSONDecodeError:
            test_case = None

    if test_case is None:
        context_text = _clean(row.get("context"))
        test_case = {
            "id": _clean(row.get("row_id")),
            "category": _clean(row.get("category")),
            "input": _clean(row.get("input")),
            "context": [context_text] if context_text else [],
            "expected_answer": _clean(row.get("ground_truth")),
            "profile": _clean(row.get("profile")),
            "context_role": _clean(row.get("context_role")),
        }

    return {
        "entry_id": _clean(row.get("entry_id")),
        "recorded_at": _clean(row.get("recorded_at")),
        "recorded_by": _clean(row.get("recorded_by")),
        "metric_key": _clean(row.get("metric_key")),
        "category": _clean(row.get("category")) or test_case.get("category"),
        "test_case": test_case,
        "actual_output": _clean(row.get("actual_output")),
        "human_verdict": _clean(row.get("human_verdict")),
        "notes": _clean(row.get("notes")),
    }

def load_calibration_judgments():
    _ensure_file()
    df = pd.read_excel(_JUDGMENTS_PATH, engine="openpyxl")
    return [_entry_from_row(row) for row in df.to_dict(orient="records")]

def _save_calibration_judgments(judgments):
    _ensure_file()
    rows = [_row_from_entry(entry) for entry in judgments]
    df = pd.DataFrame(rows, columns=_COLUMNS)
    df.to_excel(_JUDGMENTS_PATH, index=False, engine="openpyxl")

def record_calibration_judgment(
    test_case, actual_output, metric_key, human_verdict,
    category=None, recorded_by=None, notes=None
):
    """Explicit, human-invoked action only -- never called automatically
    by the orchestrator, a metric module, or anything else. Mirrors
    benchmark.benchmark_store.promote_test_case's pattern: a human reviews
    a specific metric's verdict on a specific response and records their
    own independent judgment of what it should have been.

    Appends one row to calibration_judgments.xlsx. Rewrites the whole
    sheet (read all, append, write all) rather than an in-place append --
    the golden set is small by nature (human-reviewed), so this is cheap,
    and it is what keeps pandas' column typing consistent across writes.
    """

    if human_verdict not in VALID_VERDICTS:
        raise CalibrationError(
            f"Unknown human_verdict '{human_verdict}'. "
            f"Must be one of {sorted(VALID_VERDICTS)}."
        )

    entry = {
        "entry_id": str(uuid.uuid4()),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "recorded_by": recorded_by,
        "metric_key": metric_key,
        "category": category or test_case.get("category"),
        "test_case": test_case,
        "actual_output": actual_output,
        "human_verdict": human_verdict,
        "notes": notes,
    }

    judgments = load_calibration_judgments()
    judgments.append(entry)
    _save_calibration_judgments(judgments)

    return entry

def remove_judgment(entry_id):
    """Explicit, human-invoked removal -- e.g. retracting a mistaken
    judgment. Never called automatically."""

    judgments = load_calibration_judgments()
    remaining = [judgment for judgment in judgments if judgment["entry_id"] != entry_id]

    if len(remaining) == len(judgments):
        raise CalibrationError(f"No calibration judgment found with id '{entry_id}'.")

    _save_calibration_judgments(remaining)
