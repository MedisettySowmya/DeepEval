import pandas as pd


# ---------------------------------------------------------------------------
# Dataset Manager.
#
# Loads the Excel sheet and derives an evaluation profile per row from the
# row's own content -- no type column, no per-category sheets required. The
# profile is what the metric registry keys off to decide which metrics are
# relevant to a row (evaluators/metric_registry.py).
#
# Every key the previous version produced is still produced, so the original
# tests/test_hallucination.py suite keeps working unchanged. New keys are
# additive only.
# ---------------------------------------------------------------------------

# Vocabulary that marks a Ground_Truth cell as describing expected *behaviour*
# ("refuse, the thing does not exist") rather than an answer to match. Rows
# like these cannot be scored by reference metrics -- comparing a refusal to
# "SkyNova AI is a fabricated company" is meaningless -- so they route to the
# abstention path instead.
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

            # Optional "Pattern" column: a regular expression the answer is
            # required to contain (e.g. \d{4} for "must mention a 4-digit
            # year"). Entirely optional -- a sheet with no Pattern column, or
            # a blank cell, simply means the pattern check reports
            # not_applicable for that row. Read the same way as every other
            # optional column, so a blank cell never becomes the string "nan".
            pattern_text = _text(row.get("Pattern"))

            has_context = bool(context_text)
            has_ground_truth = bool(ground_truth_text)

            evidence_tier = _get_evidence_tier(
                has_ground_truth=has_ground_truth,
                has_context=has_context
            )

            # Row identity. The ID column is stable across sheet
            # consolidation, row reordering and category changes, unlike
            # category+position -- which prompt version history and run
            # comparison both key off. Falls back to the positional scheme
            # only when the sheet has no ID column.
            raw_id = row.get("ID")
            row_id = (
                _text(raw_id) if _has_value(raw_id)
                else f"{sheet_name}:{position}"
            )

            category_rows.append({

                "id": row_id,

                "category": sheet_name,

                "category_position": position,

                "input": input_text,

                "context": [context_text] if has_context else [],

                "expected_answer": ground_truth_text if has_ground_truth else None,

                "source": "dataset",

                "evidence_tier": evidence_tier,

                "possible_missing_data": False,

                # --- derived profile (no model call) ---

                "profile": _get_profile(has_ground_truth, ground_truth_text),

                "has_context": has_context,

                "has_reference": has_ground_truth,

                # Regex the answer must contain, straight from the Excel
                # "Pattern" column. None when the column or the cell is empty.
                "expected_pattern": pattern_text or None,

                "has_pattern": bool(pattern_text),

                "expects_abstention": (
                    has_ground_truth and _expects_abstention(ground_truth_text)
                ),

                "is_numeric_reference": (
                    has_ground_truth and _is_numeric_reference(ground_truth_text)
                ),

                # Whether the Context supports or contradicts the Ground truth
                # cannot be decided from field presence. It is left unknown
                # here and filled in by the pre-flight dataset checks, which
                # need no generator and no model answer
                # (dataset_checks/preflight.py).
                "context_role": CONTEXT_ROLE_UNKNOWN,

                # --- Input validity (Stage 0's first gate) ---
                # Input is the one mandatory field. It is validated here, the
                # same way Context and Ground_Truth already were, so a blank
                # cell is reported as a skipped row with a reason instead of
                # reaching build_prompt and raising.
                "input_valid": bool(input_text),

                "input_invalid_reason": (
                    None if input_text
                    else "Input cell is empty; nothing to ask the model."
                ),

            })


        # no-evidence category.
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
