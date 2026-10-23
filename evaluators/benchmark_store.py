import json
import os
import uuid
from datetime import datetime, timezone

from evaluators.result_classifier import VERIFIED

_STORE_PATH = os.path.join(os.path.dirname(__file__), "data", "benchmark_store.json")

VALID_TEST_CASE_TYPES = {"benchmark", "guardrail"}

class PromotionRejected(Exception):
    pass

def _ensure_store_file():
    os.makedirs(os.path.dirname(_STORE_PATH), exist_ok=True)
    if not os.path.exists(_STORE_PATH):
        with open(_STORE_PATH, "w", encoding="utf-8") as store_file:
            json.dump([], store_file, indent=2)

def load_benchmark_store():
    _ensure_store_file()
    with open(_STORE_PATH, "r", encoding="utf-8") as store_file:
        return json.load(store_file)

def _save_benchmark_store(entries):
    _ensure_store_file()
    with open(_STORE_PATH, "w", encoding="utf-8") as store_file:
        json.dump(entries, store_file, indent=2)

def promote_test_case(test_case, run_result, test_case_type, approved_by=None):
    """Add a test case + its current result to the approved benchmark
    store, to be used as a regression baseline.

    This is an explicit, human-invoked action only -- it is never called
    automatically by the orchestrator, the metric manager, or anything
    else in the pipeline. A human reviews `run_result` first and decides
    to call this; nothing here decides that on its own.

    Raises PromotionRejected if the test case isn't eligible:
      - test_case_type must be "benchmark" or "guardrail"
      - the test case's classification must be VERIFIED (never EVALUATED
        or UNVERIFIABLE -- there is no ground truth to anchor a
        regression baseline otherwise)
      - there must be an actual evaluation result to store as the baseline
    """

    if test_case_type not in VALID_TEST_CASE_TYPES:
        raise PromotionRejected(
            f"Unknown test_case_type '{test_case_type}'. "
            f"Must be one of {sorted(VALID_TEST_CASE_TYPES)}."
        )

    classification_result = (run_result or {}).get("classification") or {}
    classification = classification_result.get("classification")

    if classification != VERIFIED:
        raise PromotionRejected(
            "Only VERIFIED test cases can be promoted to the benchmark "
            f"store; this test case is classified '{classification}'."
        )

    evaluation_result = (run_result or {}).get("evaluation")

    if not evaluation_result:
        raise PromotionRejected(
            "This test case has no evaluation result to use as a "
            "baseline (validation may have failed, or evaluation did "
            "not run)."
        )

    entry = {
        "entry_id": str(uuid.uuid4()),
        "test_case_type": test_case_type,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": approved_by,
        "test_case": test_case,
        "baseline_result": {
            "classification": classification_result,
            "evaluation": evaluation_result,
        },
    }

    entries = load_benchmark_store()
    entries.append(entry)
    _save_benchmark_store(entries)

    return entry

def remove_entry(entry_id):
    """Explicit, human-invoked removal -- e.g. retiring a stale benchmark
    case. Never called automatically."""

    entries = load_benchmark_store()
    remaining = [entry for entry in entries if entry["entry_id"] != entry_id]

    if len(remaining) == len(entries):
        raise PromotionRejected(f"No benchmark entry found with id '{entry_id}'.")

    _save_benchmark_store(remaining)
