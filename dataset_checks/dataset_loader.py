import json
import pandas as pd

_ABSTENTION_MARKERS = (
    "does not exist",
    "doesn't exist",
    "do not exist",
    "no such",
    "fabricated",
    "fictional",
    "fictitious",
    "misattributed",
    "not real",
    "is made up",
    "made-up",
    "never existed",
)

PROFILE_NO_REFERENCE = "no_reference"
PROFILE_ABSTENTION = "abstention_expected"
PROFILE_NUMERIC = "numeric_reference"
PROFILE_FACTUAL = "factual_reference"

CONTEXT_ROLE_UNKNOWN = "unknown"
CONTEXT_ROLE_ADVERSARIAL = "adversarial"
CONTEXT_ROLE_SUPPORTING = "supporting"

def _has_value(value):

    if value is None:
        return False

    if isinstance(value, float) and pd.isna(value):
        return False

    if isinstance(value, str) and value.strip() == "":
        return False

    return True

def _text(value):
    """Normalises a cell to a clean string. Guards against pandas handing back
    float('nan') for a blank cell in an otherwise-text column -- NaN is truthy
    in Python, so a bare `value or ""` does not catch it."""

    if not _has_value(value):
        return ""

    return str(value).strip()

def _get_evidence_tier(has_ground_truth, has_context):

    if has_ground_truth:
        return "reference"

    if has_context:
        return "context"

    return "no_evidence"

def _expects_abstention(ground_truth_text):
    lowered = ground_truth_text.lower()
    return any(marker in lowered for marker in _ABSTENTION_MARKERS)

def _is_numeric_reference(ground_truth_text):
    return any(character.isdigit() for character in ground_truth_text)

def _get_profile(has_ground_truth, ground_truth_text):
    """Deterministic: a dict/string inspection, no model involved.

    Order matters. A Ground_Truth of "No such winner exists in 2024" carries a
    numeral but is an abstention row, so the abstention check runs first.
    """

    if not has_ground_truth:
        return PROFILE_NO_REFERENCE

    if _expects_abstention(ground_truth_text):
        return PROFILE_ABSTENTION

    if _is_numeric_reference(ground_truth_text):
        return PROFILE_NUMERIC

    return PROFILE_FACTUAL

def build_test_case(
    row_id,
    category,
    category_position,
    input_text,
    context_text="",
    ground_truth_text="",
    pattern_text="",
    source="dataset",
):
    """Builds one test-case dict from already-cleaned cell text.

    Extracted from load_all_datasets so that anything creating a test case
    outside the spreadsheet -- the single-prompt UI in ui/app.py -- derives
    the profile, evidence tier and validity flags through exactly the same
    code. Duplicating that logic would let the UI and the pytest suite
    disagree about the same row, which would make the UI untrustworthy for
    testing.

    Every value passed in is expected to have been through _text() already
    (or be a plain string), so a blank Excel cell never arrives as NaN.
    """

    has_context = bool(context_text)
    has_ground_truth = bool(ground_truth_text)

    return {

        "id": row_id,

        "category": category,

        "category_position": category_position,

        "input": input_text,

        "context": [context_text] if has_context else [],

        "expected_answer": ground_truth_text if has_ground_truth else None,

        "source": source,

        "evidence_tier": _get_evidence_tier(
            has_ground_truth=has_ground_truth,
            has_context=has_context
        ),

        "possible_missing_data": False,

        "profile": _get_profile(has_ground_truth, ground_truth_text),

        "has_context": has_context,

        "has_reference": has_ground_truth,

        "expected_pattern": pattern_text or None,

        "has_pattern": bool(pattern_text),

        "expects_abstention": (
            has_ground_truth and _expects_abstention(ground_truth_text)
        ),

        "is_numeric_reference": (
            has_ground_truth and _is_numeric_reference(ground_truth_text)
        ),

        "context_role": CONTEXT_ROLE_UNKNOWN,

        "input_valid": bool(input_text),

        "input_invalid_reason": (
            None if input_text
            else "Input cell is empty; nothing to ask the model."
        ),

    }

def load_all_datasets(path):

    excel_file = pd.ExcelFile(path)

    dataset = []

    for sheet_name in excel_file.sheet_names:

        df = pd.read_excel(
            path,
            sheet_name=sheet_name
        )

        category_rows = []

        for position, (_, row) in enumerate(df.iterrows()):

            input_text = _text(row.get("Input"))
            context_text = _text(row.get("Context"))
            ground_truth_text = _text(row.get("Ground_Truth"))

            pattern_text = _text(row.get("Pattern"))

            raw_id = row.get("ID")
            row_id = (
                _text(raw_id) if _has_value(raw_id)
                else f"{sheet_name}:{position}"
            )

            category_rows.append(build_test_case(
                row_id=row_id,
                category=sheet_name,
                category_position=position,
                input_text=input_text,
                context_text=context_text,
                ground_truth_text=ground_truth_text,
                pattern_text=pattern_text,
            ))
            raw_options = _text(row.get("Metric_Options"))
            if raw_options:
                try:
                    options = json.loads(raw_options)
                    if not isinstance(options, dict) or not all(
                        isinstance(value, dict) for value in options.values()
                    ):
                        raise ValueError("expected an object mapping metric names to option objects")
                except (ValueError, TypeError) as error:
                    raise ValueError(
                        f"Invalid Metric_Options in sheet {sheet_name!r}, Excel row {position + 2}: {error}"
                    ) from error
                category_rows[-1]["metric_options"] = options

        no_evidence_count = sum(
            1 for r in category_rows if r["evidence_tier"] == "no_evidence"
        )

        no_evidence_ratio = (
            no_evidence_count / len(category_rows) if category_rows else 0
        )

        for r in category_rows:
            if r["evidence_tier"] == "no_evidence" and no_evidence_ratio < 0.5:
                r["possible_missing_data"] = True

        dataset.extend(category_rows)

    return dataset
