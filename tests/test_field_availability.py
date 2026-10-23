"""Verify the inputs supported by the generic metric adapter."""

import pytest
from deepeval.test_case import LLMTestCaseParams
from evaluators.metrics.generic import _is_field_available


@pytest.mark.parametrize("field", list(LLMTestCaseParams))
def test_supported_fields_with_populated_row(field):
    case = {"expected_answer": "Canberra", "context": ["Capital: Canberra"]}
    supported = {
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.CONTEXT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    }
    assert _is_field_available(field, case) == (field in supported)


@pytest.mark.parametrize("case", [{}, {"expected_answer": None, "context": []}])
def test_missing_reference_and_context(case):
    for field in (LLMTestCaseParams.EXPECTED_OUTPUT,
                  LLMTestCaseParams.CONTEXT,
                  LLMTestCaseParams.RETRIEVAL_CONTEXT):
        assert not _is_field_available(field, case)
